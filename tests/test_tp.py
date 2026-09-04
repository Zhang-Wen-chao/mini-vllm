"""Tensor-parallel (TP) tests.

Three layers of verification:

1. shard math (no processes): simulate each rank's slice-and-partial-sum
   computation with plain tensors and compare against the dense layer —
   pins down the exact slicing conventions (column = rows of the weight,
   row = columns of the weight, w1 bias shards, w2 bias added once);
2. world_size=1 through the engine: shards equal the full weights and
   all-reduce is a no-op, so output must match the dense reference;
3. TP=2 with two real gloo processes (CPU): each rank runs the full SPMD
   engine on its local head shard and must produce the dense reference
   tokens — with prefix cache and chunked prefill enabled.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from mini_vllm.engine import Engine
from mini_vllm.model_runner import _Layer
from mini_vllm.paged_attention import dense_attention
from mini_vllm.tp import TPTransformer

from test_engine import greedy_reference, make_model


# -- 1. shard math (no distributed) ------------------------------------------

def test_tp_shard_math_matches_dense():
    torch.manual_seed(7)
    d_model, n_heads = 32, 4
    head_dim = d_model // n_heads
    layer = _Layer(d_model, n_heads)
    x = torch.randn(5, d_model)
    ws = 2
    lh = n_heads // ws

    # dense reference: full heads in one matmul
    ln = layer.ln1(x)
    q = layer.split_heads(layer.wq(ln))
    k = layer.split_heads(layer.wk(ln))
    v = layer.split_heads(layer.wv(ln))
    o = dense_attention(q, k, v, causal=True).reshape(5, n_heads * head_dim)
    attn_ref = x + layer.wo(o)

    # per-rank partials: column-parallel q/k/v (weight rows sliced), local
    # heads attended independently, row-parallel wo partial (weight columns
    # sliced) — the union is the all-reduce sum
    partials = []
    for rank in range(ws):
        lo, hi = rank * lh * head_dim, (rank + 1) * lh * head_dim
        q_r = F.linear(ln, layer.wq.weight[lo:hi]).view(5, lh, head_dim)
        k_r = F.linear(ln, layer.wk.weight[lo:hi]).view(5, lh, head_dim)
        v_r = F.linear(ln, layer.wv.weight[lo:hi]).view(5, lh, head_dim)
        o_r = dense_attention(q_r, k_r, v_r, causal=True).reshape(5, lh * head_dim)
        partials.append(F.linear(o_r, layer.wo.weight[:, lo:hi]))
    attn_tp = x + torch.stack(partials).sum(dim=0)
    assert torch.allclose(attn_tp, attn_ref, atol=1e-5)

    # MLP: w1 column-parallel (weight rows AND bias sliced), w2 row-parallel
    # (weight columns sliced, full bias added ONCE after the reduction)
    ln2 = layer.ln2(attn_ref)
    mlp_ref = attn_ref + layer.w2(F.gelu(layer.w1(ln2)))
    m = layer.w1.weight.shape[0] // ws
    mparts = []
    for rank in range(ws):
        lo, hi = rank * m, (rank + 1) * m
        h = F.gelu(F.linear(ln2, layer.w1.weight[lo:hi], layer.w1.bias[lo:hi]))
        mparts.append(F.linear(h, layer.w2.weight[:, lo:hi]))
    mlp_tp = attn_ref + torch.stack(mparts).sum(dim=0) + layer.w2.bias
    assert torch.allclose(mlp_tp, mlp_ref, atol=1e-5)


# -- 2. world_size=1 through the engine --------------------------------------

def _run_requests(engine, prompts, max_new_tokens):
    reqs = [engine.add_request(p, max_new_tokens=max_new_tokens)
            for p in prompts]
    while engine.has_requests():
        engine.step()
    for r in reqs:
        assert r.status == "FINISHED"
    return [engine.output(r) for r in reqs]


def test_tp_world_size_one_matches_dense():
    dense = make_model()
    tp = TPTransformer(dense, 1, 0)
    assert tp.n_heads == dense.n_heads, "world_size=1 keeps every head local"

    # streaming single path
    engine = Engine(tp, block_size=4, num_blocks=16)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    out = _run_requests(engine, [prompt], 8)[0]
    assert out[len(prompt):] == greedy_reference(dense, prompt, 8)

    # batched path (prefill_batch/decode_batch) with concurrent requests
    engine2 = Engine(tp, block_size=4, num_blocks=16)
    prompts = [torch.tensor([3, 15, 27, 9]), torch.tensor([11, 4, 5, 6, 7])]
    outs = _run_requests(engine2, prompts, 4)
    for p, o in zip(prompts, outs):
        assert o[len(p):] == greedy_reference(dense, p, 4)


def test_tp_rejects_indivisible_heads():
    dense = make_model()   # n_heads=4
    with pytest.raises(ValueError):
        TPTransformer(dense, 3, 0)


# -- 3. TP=2 with two real gloo processes -------------------------------------

def _tp_worker(rank, world_size, init_file):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(0)
        dense = make_model()
        tp = TPTransformer(dense, world_size, rank)
        assert tp.n_heads == dense.n_heads // world_size

        # the KV pool shards with the heads: local pool stores local heads
        engine = Engine(tp, block_size=4, num_blocks=16,
                        enable_prefix_cache=True)
        assert engine.kv.pool.num_heads == tp.n_heads

        prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
                   torch.tensor([11, 4, 5])]
        outs = _run_requests(engine, prompts, 6)
        for p, o in zip(prompts, outs):
            assert o[len(p):] == greedy_reference(dense, p, 6), \
                f"rank {rank}: TP={world_size} must match the dense reference"

        # chunked prefill on the same TP shard
        engine2 = Engine(tp, block_size=4, num_blocks=32,
                         max_prefill_tokens=5, chunked_prefill=True)
        long_prompt = torch.tensor([3, 15, 27, 9, 42, 7, 7, 33, 21, 5, 8, 2])
        out2 = _run_requests(engine2, [long_prompt], 6)[0]
        assert out2[len(long_prompt):] == greedy_reference(dense, long_prompt, 6)
    finally:
        dist.destroy_process_group()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tp2_engine_matches_dense_reference(tmp_path):
    init_file = str(tmp_path / "tp2_store")
    mp.spawn(_tp_worker, args=(2, init_file), nprocs=2, join=True)
