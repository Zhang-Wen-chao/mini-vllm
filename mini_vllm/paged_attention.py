"""PagedAttention: block-wise attention over a sequence's paged KV cache.

The real PagedAttention kernel touches only the physical blocks listed in a
sequence's block table and accumulates results with an online softmax
(FlashAttention-style running max/sum). Here the same structure is expressed
in pure PyTorch: we walk the block table block by block, compute partial
scores, and fold them into a running (max, sum, accumulator) triple.
"""

import torch


def paged_attention(query, table, layer=0, causal=True, scale=None,
                    total_tokens=None):
    """Attention between `query` and the paged K/V of `table`'s sequence.

    Args:
        query: (num_queries, num_heads, head_dim), already projected.
        table: BlockTable of the target sequence.
        layer: which transformer layer's K/V blocks to read.
        causal: mask out keys that come after each query's position.
        scale: softmax temperature; defaults to 1/sqrt(head_dim).
        total_tokens: total stored tokens of the sequence. Defaults to the
            streaming convention: the queries of this step are the tokens
            being written right now, so total = cursor + num_queries.

    Returns:
        (num_queries, num_heads, head_dim) attention output.
    """
    pool = table.pool
    if scale is None:
        scale = query.shape[-1] ** -0.5
    num_queries, num_heads, head_dim = query.shape
    if total_tokens is None:
        # append() already wrote this step's K/V at the cursor, so the
        # sequence length right now is cursor + the tokens being queried.
        total_tokens = table.num_tokens + num_queries
    q_start = total_tokens - num_queries  # global position of the first query

    max_score = torch.full((num_queries, num_heads, 1), float("-inf"),
                           device=query.device, dtype=query.dtype)
    sum_exp = torch.zeros(num_queries, num_heads, 1, device=query.device,
                          dtype=query.dtype)
    acc = torch.zeros_like(query)

    for block_pos in range(0, total_tokens, pool.block_size):
        block_id = table.blocks[block_pos // pool.block_size]
        block_hi = min(block_pos + pool.block_size, total_tokens)
        num_keys = block_hi - block_pos
        k = pool.cache[0, layer, block_id][:num_keys]  # (S, H, D)
        v = pool.cache[1, layer, block_id][:num_keys]
        # GQA: repeat K/V heads to match the query heads
        if k.shape[1] != num_heads:
            repeat = num_heads // k.shape[1]
            idx = torch.arange(num_heads, device=query.device) // repeat
            k = k[:, idx]
            v = v[:, idx]

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


def batched_attention(q, k, v, mask=None, scale=None):
    """Batched attention over a padded batch of sequences.

    Args:
        q: (B, Q, H, D) queries.
        k, v: (B, S, H, D) padded K/V; positions beyond each sequence's real
            length are garbage and must be masked.
        mask: (B, Q, S) bool; True positions are ignored (set to -inf).

    Returns:
        (B, Q, H, D) attention output.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # GQA: repeat K/V heads to match the query heads
    if k.shape[-2] != q.shape[-2]:
        repeat = q.shape[-2] // k.shape[-2]
        idx = torch.arange(q.shape[-2], device=q.device) // repeat
        k = k[..., idx, :]
        v = v[..., idx, :]
    scores = torch.einsum("bqhd,bshd->bqhs", q, k) * scale
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(2), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bqhs,bshd->bqhd", probs, v)


def dense_attention(query, k, v, causal=True, scale=None):
    """Reference implementation over the full contiguous K/V tensors.

    Accepts single-sequence 3-D tensors (T, H, D) or batched 4-D tensors
    (B, T, H, D). Used only to verify `paged_attention` numerically.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    batched = query.dim() == 4
    if not batched:
        query, k, v = query.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    scores = torch.einsum("bqhd,bshd->bqhs", query, k) * scale
    if causal:
        total = k.shape[-3]
        b, q = query.shape[:2]
        q_pos = torch.arange(total - q, total, device=query.device)
        k_pos = torch.arange(total, device=query.device)
        # scores are (B, Q, H, S): mask per (Q, S) broadcast over heads
        mask = (q_pos[:, None, None] < k_pos[None, None, :]).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqht,bthd->bqhd", probs, v)
    return out if batched else out.squeeze(0)
