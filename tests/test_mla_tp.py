"""MLA×TP tests: the latent cache is REPLICATED, only head maps shard.

Same three verification layers as the dense TP tests:

1. shard math (no processes): per-rank partial attention over local heads
   with the SHARED latent, summed, equals the full attention — pins the
   w_uq/w_qr column slicing and the w_uk/w_uv head slicing;
2. world_size=1 through the engine: shards equal the full weights, so
   greedy output must match the dense MLA model;
3. TP=2 with two real gloo processes: each rank runs the SPMD engine on
   its local head shard — and the teaching assertion is what does NOT
   shard: the pool still stores ONE latent vector per token on every rank
   (dense MHA would shard the pool by heads; MLA has no head axis).
"""

from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from mini_vllm.engine import Engine
from mini_vllm.mla import MLATransformer, MLATransformerTP, apply_rope

from test_mla import greedy_reference, make_mla


# -- 1. shard math (no distributed) ------------------------------------------

def test_mla_tp_shard_math_matches_dense():
    torch.manual_seed(7)
    model = make_mla(seed=7)          # d_model=32, n_heads=4
    layer = model.layers[0]
    ws, lh = 2, 2
    dh, dc = layer.head_dim, layer.d_latent

    t = 5
    x = torch.randn(t, model.d_model)
    q_start = 3
    q_pos = torch.arange(q_start, q_start + t)
    s_pos = torch.arange(9)
    cache = layer.latent(torch.randn(9, model.d_model), torch.arange(9))
    c, k_r = cache[:, :dc], cache[:, dc:]
    ref = layer.attend_absorbed(x, cache, q_start=q_start)

    # per-rank partials: local heads' q_abs/ctx against the SAME latent,
    # then the row-parallel wo partial — the union is the all-reduce sum
    partials = []
    for rank in range(ws):
        lo, hi = rank * lh * dh, (rank + 1) * lh * dh
        hslice = slice(rank * lh, (rank + 1) * lh)
        w_uk = layer.w_uk.weight.view(4, dh, dc)[hslice]
        w_uv = layer.w_uv.weight.view(4, dh, dc)[hslice]
        normed = layer.ln1(x)
        q_nope = layer.w_uq(normed).view(t, 4, dh)[:, hslice]
        q_rope = apply_rope(layer.w_qr(normed), q_pos) \
            .view(t, 4, layer.d_rope)[:, hslice]
        q_abs = torch.einsum("thd,hdc->thc", q_nope, w_uk)
        scores = torch.einsum("thc,sc->ths", q_abs, c) * layer.scale \
            + torch.einsum("thr,sr->ths", q_rope, k_r) * layer.scale
        scores = scores.masked_fill(
            q_pos[:, None, None] < s_pos[None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("ths,sc->thc", probs, c)
        out = torch.einsum("thc,hdc->thd", ctx, w_uv)
        partials.append(torch.nn.functional.linear(
            out.reshape(t, lh * dh), layer.wo.weight[:, lo:hi]))
    assert torch.allclose(sum(partials), ref, atol=1e-5), \
        "head-sharded partials must sum to the full attention output"


# -- 2. world_size=1 through the engine --------------------------------------

def test_mla_tp_world_size_one_matches_dense():
    dense = make_mla(seed=11)
    tp = MLATransformerTP(dense, 1, 0)
    assert tp.n_heads == dense.n_heads

    engine = Engine(tp, block_size=4, num_blocks=32)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    assert engine.output(r)[len(prompt):] == greedy_reference(dense, prompt, 6)


def test_mla_tp_rejects_indivisible_heads():
    dense = make_mla(seed=12)
    with pytest.raises(ValueError):
        MLATransformerTP(dense, 3, 0)


# -- 3. TP=2 with two real gloo processes -------------------------------------

def _mla_tp_worker(rank, world_size, init_file):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(0)
        dense = make_mla(seed=0)
        tp = MLATransformerTP(dense, world_size, rank)
        assert tp.n_heads == dense.n_heads // world_size

        engine = Engine(tp, block_size=4, num_blocks=32)
        # THE MLA×TP point: the latent pool does NOT shard with the heads
        assert (engine.kv.pool.num_heads, engine.kv.pool.head_dim) == \
            (1, tp.d_latent + tp.d_rope), \
            "every rank must hold the FULL latent pool (no head axis)"

        prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
                   torch.tensor([11, 4, 5])]
        reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
        while engine.has_requests():
            engine.step()
        for p, r in zip(prompts, reqs):
            assert engine.output(r)[len(p):] == \
                greedy_reference(dense, p, 6), \
                f"rank {rank}: MLA TP={world_size} must match dense reference"
    finally:
        dist.destroy_process_group()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mla_tp2_engine_matches_dense_reference(tmp_path):
    init_file = str(tmp_path / "mla_tp2_store")
    mp.spawn(_mla_tp_worker, args=(2, init_file), nprocs=2, join=True)
