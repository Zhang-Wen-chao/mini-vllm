"""PagedAttention: block-wise attention over a sequence's paged KV cache.

The real PagedAttention kernel touches only the physical blocks listed in a
sequence's block table and accumulates results with an online softmax
(FlashAttention-style running max/sum). Here the same structure is expressed
in pure PyTorch: we walk the block table block by block, compute partial
scores, and fold them into a running (max, sum, accumulator) triple.
"""

import torch


def paged_attention(query, table, layer=0, causal=True, scale=None):
    """Attention between `query` and the paged K/V of `table`'s sequence.

    Args:
        query: (num_queries, num_heads, head_dim), already projected.
        table: BlockTable of the target sequence.
        layer: which transformer layer's K/V blocks to read.
        causal: mask out keys that come after each query's position.
        scale: softmax temperature; defaults to 1/sqrt(head_dim).

    Returns:
        (num_queries, num_heads, head_dim) attention output.
    """
    pool = table.pool
    if scale is None:
        scale = query.shape[-1] ** -0.5
    num_queries, num_heads, head_dim = query.shape
    total_tokens = table.num_tokens
    q_start = total_tokens - num_queries  # global position of the first query

    max_score = torch.full((num_queries, num_heads, 1), float("-inf"),
                           device=query.device)
    sum_exp = torch.zeros(num_queries, num_heads, 1, device=query.device)
    acc = torch.zeros_like(query)

    for block_pos in range(0, total_tokens, pool.block_size):
        block_id = table.blocks[block_pos // pool.block_size]
        block_hi = min(block_pos + pool.block_size, total_tokens)
        num_keys = block_hi - block_pos
        k = pool.cache[0, layer, block_id][:num_keys]  # (S, H, D)
        v = pool.cache[1, layer, block_id][:num_keys]

        scores = torch.einsum("qhd,shd->qhs", query, k) * scale  # (nq, H, S)
        if causal:
            q_pos = torch.arange(q_start, q_start + num_queries,
                                 device=query.device)
            k_pos = torch.arange(block_pos, block_hi, device=query.device)
            scores = scores.masked_fill(q_pos[:, None, None] < k_pos[None, None, :],
                                        float("-inf"))

        block_max = scores.max(dim=-1, keepdim=True).values
        # Avoid nan when both running max and block max are -inf (all masked).
        safe_new = torch.where(block_max == float("-inf"),
                               torch.zeros_like(block_max),
                               torch.maximum(max_score, block_max))
        rescale = torch.exp(max_score - safe_new)
        exp_scores = torch.exp(scores - safe_new)
        acc = acc * rescale + torch.einsum("qhs,shd->qhd", exp_scores, v)
        sum_exp = sum_exp * rescale + exp_scores.sum(dim=-1, keepdim=True)
        max_score = safe_new

    return acc / sum_exp


def dense_attention(query, k, v, causal=True, scale=None):
    """Reference implementation over the full contiguous K/V tensors.

    Used only to verify `paged_attention` numerically.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = torch.einsum("qhd,shd->qhs", query, k) * scale  # (nq, H, T)
    if causal:
        total = k.shape[0]
        q_pos = torch.arange(total - query.shape[0], total,
                             device=query.device)
        k_pos = torch.arange(total, device=query.device)
        scores = scores.masked_fill(q_pos[:, None, None] < k_pos[None, None, :],
                                    float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("qht,thd->qhd", probs, v)
