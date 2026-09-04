"""KV cache quantization tests: int8 + per-token scale, fp8 e4m3 scale-free.

Anchors:
- the int8 pool dequantizes to the original values within the rounding
  bound |x - q*s| <= s/2 (s = max|x|/127 per token);
- the fp8 pool needs NO scale table: e4m3's exponent bits make the error
  RELATIVE (<= 2^-4 ≈ 6.25%) instead of absolute;
- the engine reads dequantized numbers transparently — greedy output
  still matches the fp32 reference (fp8's coarser noise may flip a
  token, which is exactly the accuracy/memory trade quantization sells);
- storage actually shrinks: int8 payload + fp32 scale ≈ half, fp8 a
  quarter of the fp32 pool;
- combinations that don't work yet (quantized × swap) are rejected up
  front instead of silently misbehaving.
"""

import pytest
import torch

from mini_vllm.engine import Engine
from mini_vllm.kv_cache import (BlockPool, BlockTable, Float8KVBlockPool,
                                Int8KVBlockPool)

from test_engine import greedy_reference, make_model


def test_int8_pool_roundtrip_within_bound():
    torch.manual_seed(0)
    pool = Int8KVBlockPool(num_blocks=4, block_size=2, num_heads=2,
                           head_dim=8, num_layers=2)
    assert pool.cache.dtype == torch.int8
    k = torch.randn(2, 2, 8) * 3
    pool.write(0, 0, 1, 0, k)
    got = pool.gather(0, 0, [1], 2)
    assert got.dtype == torch.float32
    # per-token max-abs scale: error per element is at most s/2
    s = k.abs().amax(dim=-1, keepdim=True) / 127.0
    err = (got - k).abs().amax(dim=-1, keepdim=True)
    assert (err <= s / 2 + 1e-6).all(), \
        "dequantization error must respect the rounding bound"
    assert torch.allclose(got, k, atol=0.05)


def test_quantized_blocktable_spans_blocks():
    pool = Int8KVBlockPool(num_blocks=8, block_size=2, num_heads=1,
                           head_dim=8, num_layers=1)
    table = BlockTable(pool)
    k = torch.randn(5, 1, 8)            # 5 tokens across 3 blocks (bs=2)
    table.append(0, k, None)
    table.advance(5)
    got = table.gather_kind(0, 0)
    assert got.shape == (5, 1, 8)
    s = k.abs().amax(dim=-1, keepdim=True) / 127.0
    assert ((got - k).abs().amax(dim=-1, keepdim=True) <= s / 2 + 1e-6).all()


def test_int8_pool_uses_less_memory_than_fp32():
    args = dict(num_blocks=8, block_size=4, num_heads=4, head_dim=8,
                num_layers=2)
    q = Int8KVBlockPool(**args)
    f = BlockPool(**args)
    q_bytes = q.cache.numel() * q.cache.element_size() \
        + q.scale.numel() * q.scale.element_size()
    f_bytes = f.cache.numel() * f.cache.element_size()
    assert q_bytes < f_bytes / 2, \
        "int8 payload + fp32 scales must roughly halve KV storage"


def test_freed_blocks_are_scrubbed():
    pool = Int8KVBlockPool(num_blocks=4, block_size=2, num_heads=1,
                           head_dim=4, num_layers=1)
    pool.write(0, 0, 0, 0, torch.ones(2, 1, 4) * 5)
    pool.free(0)
    assert pool.cache[:, :, 0].abs().sum() == 0
    assert pool.scale[:, :, 0].abs().sum() == 0


def test_engine_int8_kv_matches_reference():
    model = make_model(seed=0)
    engine = Engine(model, block_size=4, num_blocks=32, kv_cache_dtype="int8")
    assert engine.kv.pool.cache.dtype == torch.int8
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    assert engine.output(r)[len(prompt):] == greedy_reference(model, prompt, 6), \
        "quantization noise must not flip greedy tokens"


def test_engine_int8_kv_with_prefix_cache():
    model = make_model(seed=1)
    engine = Engine(model, block_size=4, num_blocks=64, kv_cache_dtype="int8",
                    enable_prefix_cache=True)
    p1 = torch.tensor([3, 15, 27, 9, 42, 7])
    r1 = engine.add_request(p1, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    p2 = torch.tensor([3, 15, 27, 9, 50, 51, 52])
    r2 = engine.add_request(p2, max_new_tokens=4)
    while engine.has_requests():
        engine.step()
    assert engine.prefix_cache.hits_tokens == 4
    assert engine.output(r1)[len(p1):] == greedy_reference(model, p1, 6)
    assert engine.output(r2)[len(p2):] == greedy_reference(model, p2, 4)


def test_int8_rejects_swap_preemption():
    model = make_model()
    with pytest.raises(ValueError):
        Engine(model, block_size=2, num_blocks=8, kv_cache_dtype="int8",
               preemption="swap", swap_num_blocks=8)
    with pytest.raises(ValueError):
        Engine(model, block_size=2, num_blocks=8, kv_cache_dtype="fp8",
               preemption="swap", swap_num_blocks=8)
    with pytest.raises(ValueError):
        Engine(model, kv_cache_dtype="fp4")


# -- fp8 e4m3: relative-error format, no scale table ---------------------------

def test_fp8_pool_roundtrip_within_relative_bound():
    torch.manual_seed(3)
    pool = Float8KVBlockPool(num_blocks=4, block_size=2, num_heads=2,
                             head_dim=8, num_layers=2)
    assert pool.cache.dtype == torch.float8_e4m3fn
    k = torch.randn(2, 2, 8) * 3          # magnitudes in e4m3's normal range
    pool.write(0, 0, 1, 0, k)
    got = pool.gather(0, 0, [1], 2)
    assert got.dtype == torch.float32
    # e4m3 is a FLOATING format: half an ulp of the 3-bit mantissa bounds
    # the error RELATIVELY (<= 2^-4 ≈ 6.25%), with no per-token scale at all
    rel = (got - k).abs() / k.abs().clamp_min(1e-3)
    assert (rel <= 0.0625 + 1e-6).all(), \
        "e4m3 error must be relative to the magnitude (no scale involved)"
    assert not hasattr(pool, "scale"), \
        "fp8 needs no side scale table — that is the point"


def test_fp8_pool_spans_blocks_and_scrubs():
    pool = Float8KVBlockPool(num_blocks=8, block_size=2, num_heads=1,
                             head_dim=8, num_layers=1)
    table = BlockTable(pool)
    k = torch.randn(5, 1, 8)
    table.append(0, k, None)
    table.advance(5)
    got = table.gather_kind(0, 0)
    rel = (got - k).abs() / k.abs().clamp_min(1e-3)
    assert (rel <= 0.0625 + 1e-6).all()
    pool.write(0, 0, 0, 0, torch.ones(2, 1, 8) * 100)
    pool.free(0)
    assert pool.cache[:, :, 0].float().abs().sum() == 0


def test_fp8_pool_uses_quarter_of_fp32_storage():
    args = dict(num_blocks=8, block_size=4, num_heads=4, head_dim=8,
                num_layers=2)
    f = BlockPool(**args)
    i8 = Int8KVBlockPool(**args)
    f8 = Float8KVBlockPool(**args)
    f_bytes = f.cache.numel() * f.cache.element_size()
    i8_bytes = i8.cache.numel() * i8.cache.element_size() \
        + i8.scale.numel() * i8.scale.element_size()
    f8_bytes = f8.cache.numel() * f8.cache.element_size()
    assert f8_bytes == f_bytes // 4
    assert f8_bytes < i8_bytes, \
        "no-scale fp8 beats per-token-scale int8 on bytes"


def test_engine_fp8_kv_stays_close_to_reference():
    model = make_model(seed=0)
    engine = Engine(model, block_size=4, num_blocks=32, kv_cache_dtype="fp8")
    assert engine.kv.pool.cache.dtype == torch.float8_e4m3fn
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    out = engine.output(r)[len(prompt):]
    ref = greedy_reference(model, prompt, 6)
    match = sum(int(a == b) for a, b in zip(out, ref))
    assert match >= len(ref) - 1, \
        f"fp8 noise may flip at most a token or two, got {match}/{len(ref)}"
