"""MLA (Multi-head Latent Attention) tests.

Anchors:
- the absorbed path must equal the explicit up-projected path (the two
  are the same math rearranged — that identity is what lets the kernel
  read the tiny latent cache instead of materialized K/V);
- the engine's paged/absorbed output must equal the model's own
  dense_forward (greedy reference) token for token;
- the pool must actually store ONE latent vector per token — 20 floats vs
  dense MHA's 64 — and those floats must be the real latent vectors.
"""

import torch

from mini_vllm.engine import Engine
from mini_vllm.kv_cache import KVBlockManager
from mini_vllm.mla import MLATransformer
from mini_vllm.model_runner import TinyTransformer


def make_mla(seed=0, **kwargs):
    torch.manual_seed(seed)
    return MLATransformer(**kwargs)


def greedy_reference(model, prompt, n):
    """Greedy decode with the model's own dense (non-absorbed) forward."""
    out = []
    ids = list(prompt)
    for _ in range(n):
        logits = model.dense_forward(torch.tensor(ids))
        tok = int(logits[-1].argmax())
        out.append(tok)
        ids.append(tok)
    return out


def test_absorbed_matches_explicit_attention():
    model = make_mla(seed=1)
    layer = model.layers[0]
    x = torch.randn(5, model.d_model)            # current tokens
    history = torch.randn(11, model.d_model)
    cache = layer.latent(history, torch.arange(11))   # (11, d_latent+d_rope)
    assert cache.shape == (11, model.d_latent + model.d_rope)
    absorbed = layer.attend_absorbed(x, cache, q_start=6)
    explicit = layer.attend_explicit(x, cache, q_start=6)
    assert torch.allclose(absorbed, explicit, atol=1e-5), \
        "W_UK/W_UV absorption must be an exact rearrangement, not an \
approximation"


def test_paged_prefill_matches_dense_forward():
    model = make_mla(seed=2)
    mgr = KVBlockManager(32, 4, model.n_kv_heads, model.head_dim,
                         num_layers=model.n_layers, kinds=model.kv_kinds)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7, 11, 4])
    table = mgr.create_table()
    logits = model.prefill(prompt, table)
    ref = model.dense_forward(prompt)
    assert torch.allclose(logits, ref, atol=1e-4), \
        "paged absorbed prefill must reproduce the dense MLA forward"


def test_batched_paths_match_streaming():
    model = make_mla(seed=3)
    mgr = KVBlockManager(32, 4, model.n_kv_heads, model.head_dim,
                         num_layers=model.n_layers, kinds=model.kv_kinds)
    prompts = [torch.tensor([3, 15, 27, 9]), torch.tensor([11, 4, 5])]
    batched_tables = [mgr.create_table() for _ in prompts]
    stream_tables = [mgr.create_table() for _ in prompts]

    logits_b = model.prefill_batch(prompts, batched_tables)
    logits_s = [model.prefill(p, t) for p, t in zip(prompts, stream_tables)]
    for lb, ls in zip(logits_b, logits_s):
        assert torch.allclose(lb, ls[-1], atol=1e-5)
    next_tokens = [int(lb.argmax()) for lb in logits_b]

    for _ in range(3):
        dec_b = model.decode_batch(
            [torch.tensor(t) for t in next_tokens], batched_tables)
        dec_s = []
        for i, (t, table) in enumerate(zip(next_tokens, stream_tables)):
            dec_s.append(model.decode(torch.tensor(t), table)[0])
        for db, ds in zip(dec_b, dec_s):
            assert torch.allclose(db, ds, atol=1e-5)
        next_tokens = [int(db.argmax()) for db in dec_b]


def test_mla_kv_compression():
    mla = make_mla(seed=4)
    dense = TinyTransformer()
    mla_pool = Engine(mla, block_size=4, num_blocks=32).kv.pool
    dense_pool = Engine(dense, block_size=4, num_blocks=32).kv.pool
    # MLA reports one "head" holding the latent vector, one cache kind
    assert (mla_pool.kinds, mla_pool.num_heads, mla_pool.head_dim) == \
        (1, 1, mla.d_latent + mla.d_rope)
    assert (dense_pool.kinds, dense_pool.num_heads, dense_pool.head_dim) == \
        (2, dense.n_heads, dense.head_dim)
    mla_per_token = mla_pool.kinds * mla_pool.num_heads * mla_pool.head_dim
    dense_per_token = dense_pool.kinds * dense_pool.num_heads \
        * dense_pool.head_dim
    assert mla_per_token == 20 and dense_per_token == 64
    assert mla_per_token / dense_per_token < 1 / 3, \
        "MLA must store strictly less than MHA per token per layer"


def test_cache_holds_real_latent_vectors():
    model = make_mla(seed=5)
    engine = Engine(model, block_size=4, num_blocks=32)
    prompt = torch.tensor([3, 15, 27, 9, 42])
    r = engine.add_request(prompt, max_new_tokens=2)
    while r.num_generated < 1:
        engine.step()

    # per-layer ln1 inputs from a full dense forward, via hooks
    captured = []
    hooks = [layer.ln1.register_forward_hook(
        lambda m, inp, out: captured.append(inp[0]))
        for layer in model.layers]
    model.dense_forward(prompt)
    for h in hooks:
        h.remove()

    table = engine._state[r.request_id]["table"]
    assert table.num_tokens >= len(prompt)
    for l, layer in enumerate(model.layers):
        stored = table.gather_kind(0, l, len(prompt)).squeeze(1)
        expected = layer.latent(captured[l], torch.arange(len(prompt)))
        assert torch.allclose(stored, expected, atol=1e-4), \
            "the cache must hold [c_KV ; k_R], not materialized K/V"


def test_mla_engine_matches_dense_reference():
    model = make_mla(seed=6)
    engine = Engine(model, block_size=4, num_blocks=32)
    prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
               torch.tensor([11, 4, 5])]
    reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
    while engine.has_requests():
        engine.step()
    for p, r in zip(prompts, reqs):
        assert engine.output(r)[len(p):] == greedy_reference(model, p, 6)


def test_mla_prefix_cache_composite():
    model = make_mla(seed=7)
    engine = Engine(model, block_size=4, num_blocks=64,
                    enable_prefix_cache=True)
    p1 = torch.tensor([3, 15, 27, 9, 42, 7])
    r1 = engine.add_request(p1, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    p2 = torch.tensor([3, 15, 27, 9, 50, 51, 52])
    r2 = engine.add_request(p2, max_new_tokens=4)
    while engine.has_requests():
        engine.step()
    assert engine.prefix_cache.hits_tokens == 4, "shared prefix matched"
    assert engine.output(r1)[len(p1):] == greedy_reference(model, p1, 6)
    assert engine.output(r2)[len(p2):] == greedy_reference(model, p2, 4)


def test_mla_ngram_spec_composite():
    model = make_mla(seed=8)
    engine = Engine(model, block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram")
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=8)
    while engine.has_requests():
        engine.step()
    assert engine.output(r)[len(prompt):] == greedy_reference(model, prompt, 8)


def test_mla_swap_composite():
    model = make_mla(seed=9)
    engine = Engine(model, block_size=2, num_blocks=6,
                    preemption="swap", swap_num_blocks=8)
    prompt = torch.tensor([1, 2, 3])
    r = engine.add_request(prompt, max_new_tokens=4)
    while r.num_generated < 1:
        engine.step()

    # drain the pool to force the swap-out branch of _make_room
    drained = [engine.kv.pool.allocate()
               for _ in list(engine.kv.pool.free_blocks)]
    engine._make_room_for_next_tokens()
    assert r.status == "SWAPPED"
    assert engine.swap_space.used_blocks > 0

    for b in drained:
        engine.kv.pool.free(b)
    while engine.has_requests():
        engine.step()
    assert r.status == "FINISHED"
    assert engine.swap_space.swap_ins == 1
    assert engine.output(r)[len(prompt):] == greedy_reference(model, prompt, 4)


# -- MLA × KV quantization -----------------------------------------------------

def test_mla_int8_kv_engine_matches_reference():
    # the latent pool quantizes like any other: per-token int8 noise must
    # not flip greedy tokens
    model = make_mla(seed=10)
    engine = Engine(model, block_size=4, num_blocks=32, kv_cache_dtype="int8")
    assert engine.kv.pool.cache.dtype == torch.int8
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    assert engine.output(r)[len(prompt):] == greedy_reference(model, prompt, 6)


def test_mla_fp8_kv_engine_close_and_compressed():
    model = make_mla(seed=10)
    engine = Engine(model, block_size=4, num_blocks=32, kv_cache_dtype="fp8")
    pool = engine.kv.pool
    assert pool.cache.dtype == torch.float8_e4m3fn
    # the latent vector (20 floats) now costs 20 bytes per token per layer
    # instead of 80 — compression applies to MLA exactly as to dense K/V
    per_token_bytes = pool.kinds * pool.num_heads * pool.head_dim \
        * pool.cache.element_size()
    assert per_token_bytes == model.d_latent + model.d_rope
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    out = engine.output(r)[len(prompt):]
    ref = greedy_reference(model, prompt, 6)
    match = sum(int(a == b) for a, b in zip(out, ref))
    assert match >= len(ref) - 1, \
        f"fp8 latent noise may flip at most a token, got {match}/{len(ref)}"
