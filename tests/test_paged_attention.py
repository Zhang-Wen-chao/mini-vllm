import torch

from mini_vllm.kv_cache import BlockTable, BlockPool
from mini_vllm.paged_attention import dense_attention, paged_attention


def make_table(num_tokens, block_size, num_heads, head_dim, num_layers=1,
               dtype=torch.float64):
    pool = BlockPool(num_blocks=8, block_size=block_size, num_heads=num_heads,
                     head_dim=head_dim, num_layers=num_layers, dtype=dtype)
    table = BlockTable(pool)
    k = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    v = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    table.append(0, k, v)
    table.advance(num_tokens)
    return table, k, v


def check_matches_dense(num_tokens, num_queries, block_size, num_heads,
                        head_dim, causal, atol=1e-8):
    table, k, v = make_table(num_tokens, block_size, num_heads, head_dim)
    query = torch.randn(num_queries, num_heads, head_dim, dtype=torch.float64)
    out = paged_attention(query, table, causal=causal,
                              total_tokens=num_tokens)
    ref = dense_attention(query, k, v, causal=causal)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=atol), (out - ref).abs().max()


def test_prefill_single_block_matches_dense():
    check_matches_dense(num_tokens=5, num_queries=5, block_size=16,
                        num_heads=2, head_dim=8, causal=True)


def test_prefill_multi_block_matches_dense():
    check_matches_dense(num_tokens=23, num_queries=23, block_size=4,
                        num_heads=3, head_dim=16, causal=True)


def test_prefill_block_aligned_matches_dense():
    check_matches_dense(num_tokens=8, num_queries=8, block_size=4,
                        num_heads=2, head_dim=8, causal=True)


def test_decode_matches_dense():
    # decode: a single new query attending to the whole stored prefix
    check_matches_dense(num_tokens=30, num_queries=1, block_size=4,
                        num_heads=2, head_dim=8, causal=True)


def test_non_causal_matches_dense():
    check_matches_dense(num_tokens=17, num_queries=17, block_size=4,
                        num_heads=2, head_dim=8, causal=False)


def test_query_attends_only_its_prefix():
    # causal: query at global position 2 must not see keys at positions 3+
    table, k, v = make_table(num_tokens=6, block_size=4, num_heads=1,
                             head_dim=8)
    query = torch.randn(3, 1, 8, dtype=torch.float64)  # queries at pos 3,4,5
    out = paged_attention(query, table, causal=True,
                              total_tokens=6)
    ref = dense_attention(query, k, v, causal=True)
    assert torch.allclose(out, ref)


def test_decode_after_multiple_appends():
    # prefill 7 tokens, then two decode steps, query only the last
    table, _, _ = make_table(num_tokens=7, block_size=4, num_heads=2,
                             head_dim=8)
    table.append(0, torch.randn(1, 2, 8, dtype=torch.float64),
                 torch.randn(1, 2, 8, dtype=torch.float64))
    table.advance(1)
    table.append(0, torch.randn(1, 2, 8, dtype=torch.float64),
                 torch.randn(1, 2, 8, dtype=torch.float64))
    table.advance(1)
    k, v = table.get_kv(0)
    query = torch.randn(1, 2, 8, dtype=torch.float64)
    out = paged_attention(query, table, causal=True,
                          total_tokens=table.num_tokens)
    ref = dense_attention(query, k, v, causal=True)
    assert torch.allclose(out, ref)


def test_fp32_still_matches_within_tolerance():
    torch.manual_seed(0)
    for _ in range(5):
        num_tokens = 20
        table, k, v = make_table(num_tokens, 4, 2, 8, dtype=torch.float32)
        query = torch.randn(5, 2, 8, dtype=torch.float32)
        out = paged_attention(query, table, causal=True,
                                  total_tokens=20)
        ref = dense_attention(query, k, v, causal=True)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_gqa_expansion_matches_dense():
    # kv_heads=2, q_heads=4: paged attention must replicate the expanded
    # dense attention exactly
    torch.manual_seed(0)
    nq, kv_heads, q_heads, hd = 6, 2, 4, 16
    pool = BlockPool(num_blocks=4, block_size=4, num_heads=kv_heads,
                     head_dim=hd, dtype=torch.float64)
    table = BlockTable(pool)
    k = torch.randn(nq, kv_heads, hd, dtype=torch.float64)
    v = torch.randn(nq, kv_heads, hd, dtype=torch.float64)
    table.append(0, k, v)
    table.advance(nq)
    query = torch.randn(nq, q_heads, hd, dtype=torch.float64)
    out = paged_attention(query, table, causal=True, total_tokens=nq)
    # dense reference with expanded heads
    idx = torch.arange(q_heads) // (q_heads // kv_heads)
    k_full = k[:, idx]
    v_full = v[:, idx]
    ref = dense_attention(query, k_full, v_full, causal=True)
    assert out.shape == (nq, q_heads, hd)
    assert torch.allclose(out, ref, atol=1e-8)


def test_gqa_batched_attention():
    torch.manual_seed(0)
    from mini_vllm.paged_attention import batched_attention
    b, s, kv_heads, q_heads, hd = 2, 5, 2, 4, 16
    q = torch.randn(b, 1, q_heads, hd, dtype=torch.float64)
    k = torch.randn(b, s, kv_heads, hd, dtype=torch.float64)
    v = torch.randn(b, s, kv_heads, hd, dtype=torch.float64)
    out = batched_attention(q, k, v, mask=None)
    idx = torch.arange(q_heads) // (q_heads // kv_heads)
    ref = batched_attention(q, k[:, :, idx], v[:, :, idx], mask=None)
    assert torch.allclose(out, ref)
