"""MoE FFN + Expert Parallelism tests.

Anchors:
- n_experts=1 MoE must equal the dense TinyTransformer bit for bit
  (expert 0 reuses the dense FFN, router weight is exactly 1.0);
- the engine's paged MoE output must equal the model's own dense forward
  (greedy reference) exactly;
- EP=2 (two gloo ranks, all-reduce combine) must equal EP=1 — the sum of
  per-rank expert partials is the same MoE function.
"""

from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from mini_vllm.engine import Engine
from mini_vllm.moe import MoETinyTransformer, Router, _MoELayer

from test_engine import greedy_reference, make_model


def make_moe(seed=0, n_experts=4, top_k=2, **kwargs):
    dense = make_model(seed=seed)
    return MoETinyTransformer(dense, n_experts=n_experts, top_k=top_k,
                              **kwargs)


def test_router_topk_renormalizes_and_is_deterministic():
    torch.manual_seed(3)
    router = Router(8, n_experts=4, top_k=2)
    x = torch.randn(5, 8)
    weights, idx = router(x)
    assert weights.shape == (5, 2) and idx.shape == (5, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5)), \
        "renormalized top-k weights must sum to 1"
    # single-expert router: weight exactly 1.0
    solo = Router(8, n_experts=1, top_k=1)
    w1, i1 = solo(x)
    assert torch.all(w1 == 1.0) and torch.all(i1 == 0)
    with pytest.raises(ValueError):
        Router(8, n_experts=2, top_k=3)


def test_one_expert_moe_equals_dense_bitwise():
    dense = make_model(seed=0)
    moe = MoETinyTransformer(dense, n_experts=1, top_k=1)
    x = torch.randn(7, dense.d_model)
    # same weights, same math: the MoE layer must be a no-op wrapper
    assert torch.equal(moe.layers[0].mlp(x), dense.layers[0].mlp(x))
    ids = torch.tensor([3, 15, 27, 9])
    assert torch.equal(moe.dense_forward(ids), dense.dense_forward(ids))


def test_moe_mlp_matches_hand_rolled_routing():
    model = make_moe(seed=5, n_experts=4, top_k=2)
    layer = model.layers[0]
    x = torch.randn(6, model.d_model)
    out = layer.mlp(x)

    # hand-rolled reference: ln2, route, run experts, combine
    normed = layer.ln2(x)
    probs = F.softmax(layer.router.proj(normed), dim=-1)
    weights, idx = torch.topk(probs, 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ref = torch.zeros_like(x)
    for e, expert in enumerate(layer.experts):
        mask = idx == e
        if not mask.any():
            continue
        pos, slot = mask.nonzero(as_tuple=True)
        w = weights[pos].gather(1, slot.unsqueeze(1))
        ref[pos] += expert(normed[pos]) * w
    assert torch.allclose(out, ref, atol=1e-6), \
        "grouped-by-expert compute must equal the naive per-token combine"


def test_moe_engine_matches_its_dense_reference():
    model = make_moe(seed=1)
    engine = Engine(model, block_size=4, num_blocks=32,
                    enable_prefix_cache=True)
    prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
               torch.tensor([11, 4, 5])]
    reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
    while engine.has_requests():
        engine.step()
    for p, r in zip(prompts, reqs):
        # oracle: the MoE model's own dense (non-paged) forward
        assert engine.output(r)[len(p):] == greedy_reference(model, p, 6)
    # prefix cache is on: finished blocks live in the cache, not the pool
    assert len(engine.kv.pool.free_blocks) + len(engine.prefix_cache) == 32


def test_moe_composes_with_ngram_spec_decode():
    model = make_moe(seed=2)
    engine = Engine(model, block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram")
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    req = engine.add_request(prompt, max_new_tokens=8)
    while engine.has_requests():
        engine.step()
    assert engine.output(req)[len(prompt):] == greedy_reference(model, prompt, 8)


def test_ep_partition_sums_to_full_moe_in_process():
    # EP math without processes: a rank's partial is the router-weighted
    # sum over ONLY its experts; partials over all ranks add up to the
    # full MoE output — the property the all-reduce combine relies on
    model = make_moe(seed=7, n_experts=4, top_k=2)
    layer = model.layers[0]
    x = torch.randn(5, model.d_model)
    full = layer.mlp(x)

    flat = layer.ln2(x).reshape(-1, x.shape[-1])
    weights, idx = layer.router(flat)
    partials = []
    for rank_ids in ([0, 1], [2, 3]):
        out = torch.zeros_like(flat)
        for e in rank_ids:
            mask = idx == e
            if not mask.any():
                continue
            pos, slot = mask.nonzero(as_tuple=True)
            w = weights[pos].gather(1, slot.unsqueeze(1))
            out.index_add_(0, pos, layer.experts[e](flat[pos]) * w)
        partials.append(out)
    assert torch.allclose(sum(partials), full, atol=1e-6), \
        "EP partials must sum to the exact MoE output"


def test_ep_rejects_bad_partition():
    with pytest.raises(ValueError):
        make_moe(n_experts=4, ep_size=3)
    with pytest.raises(ValueError):
        make_moe(n_experts=4, ep_size=2, ep_rank=2)


# -- EP=2 with two real gloo processes ----------------------------------------

def _ep_worker(rank, world_size, init_file, seed):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        dense = make_model(seed=seed)
        model = MoETinyTransformer(dense, n_experts=4, top_k=2,
                                   ep_size=world_size, ep_rank=rank)
        assert len(model.layers[0].experts) == 2, "2 experts per EP rank"
        engine = Engine(model, block_size=4, num_blocks=32,
                        enable_prefix_cache=True)
        prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
                   torch.tensor([11, 4, 5])]
        reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
        while engine.has_requests():
            engine.step()
        for p, r in zip(prompts, reqs):
            assert engine.output(r)[len(p):] == greedy_reference(model, p, 6), \
                f"EP={world_size} rank {rank} must equal the MoE dense reference"
    finally:
        dist.destroy_process_group()


def test_ep2_gloo_matches_ep1_reference(tmp_path):
    init_file = str(tmp_path / "ep2_store")
    mp.spawn(_ep_worker, args=(2, init_file, 9), nprocs=2, join=True)


# -- Phase 18: shared experts / fine-grained experts / p2p combine -------------

def test_shared_expert_is_always_on_and_unweighted():
    # anchor at the dense width: with n_experts=1 (router weight exactly 1.0,
    # expert 0 = the dense FFN) one shared expert = the dense FFN added again
    dense = make_model(seed=0)
    moe = MoETinyTransformer(dense, n_experts=1, top_k=1, n_shared=1)
    x = torch.randn(7, dense.d_model)
    assert torch.equal(moe.layers[0].mlp(x), 2 * dense.layers[0].mlp(x)), \
        "shared expert must add the FFN once, unweighted, unroutered"


def test_shared_expert_computed_once_after_routing():
    torch.manual_seed(42)
    dense = make_model(seed=11)
    layer_a = _MoELayer(dense.layers[0], n_experts=4, top_k=2)
    torch.manual_seed(42)          # pin the router init for both
    layer_b = _MoELayer(dense.layers[0], n_experts=4, top_k=2, n_shared=1)
    x = torch.randn(6, dense.d_model)

    out = layer_b.mlp(x)
    # hand-rolled: routed combine + exactly one shared contribution
    normed = layer_b.ln2(x)
    flat = normed.reshape(-1, x.shape[-1])
    weights, idx = layer_b.router(flat)
    routed = torch.zeros_like(flat)
    for local_e, expert in enumerate(layer_b.experts):
        mask = idx == local_e
        if not mask.any():
            continue
        pos, slot = mask.nonzero(as_tuple=True)
        w = weights[pos].gather(1, slot.unsqueeze(1))
        routed.index_add_(0, pos, expert(flat[pos]) * w)
    ref = routed + layer_b.shared[0](flat)
    assert torch.allclose(out, ref, atol=1e-6)


def test_fine_grained_experts_are_narrow_but_functional():
    dense = make_model(seed=3)
    model = make_moe(seed=3, n_experts=8, top_k=2, inter_dim=16, n_shared=1)
    layer = model.layers[0]
    # fine-grained: narrow intermediate (16 ≪ 4·32), many experts
    for e in layer.experts:
        assert e.w1.weight.shape == (16, dense.d_model)
        assert e.w2.weight.shape == (dense.d_model, 16)
    assert layer.shared[0].w1.weight.shape == (16, dense.d_model)
    # grouped compute still equals the per-token merge
    x = torch.randn(6, dense.d_model)
    out = layer.mlp(x)
    normed = layer.ln2(x)
    flat = normed.reshape(-1, x.shape[-1])
    weights, idx = layer.router(flat)
    ref = torch.zeros_like(flat)
    for local_e, expert in enumerate(layer.experts):
        mask = idx == (layer.expert_offset + local_e)
        if not mask.any():
            continue
        pos, slot = mask.nonzero(as_tuple=True)
        w = weights[pos].gather(1, slot.unsqueeze(1))
        ref.index_add_(0, pos, expert(flat[pos]) * w)
    for expert in layer.shared:
        ref = ref + expert(flat)
    assert torch.allclose(out, ref, atol=1e-6)
    # engine end-to-end: fine-grained MoE matches its own dense forward
    engine = Engine(model, block_size=4, num_blocks=32)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    assert engine.output(r)[len(prompt):] == greedy_reference(model, prompt, 6)


def test_dense_width_anchor_unchanged():
    # the legacy default (inter_dim=None) must keep the bitwise dense anchor
    dense = make_model(seed=0)
    moe = MoETinyTransformer(dense, n_experts=1, top_k=1)
    x = torch.randn(7, dense.d_model)
    assert torch.equal(moe.layers[0].mlp(x), dense.layers[0].mlp(x))


def test_p2p_combine_is_identity_without_distributed():
    x = torch.randn(5, 8)
    from mini_vllm.moe import _p2p_combine
    assert torch.equal(_p2p_combine(x), x), \
        "no process group: combine must be a no-op"
    with pytest.raises(ValueError):
        make_moe(ep_combine="ring")


# -- EP=2 pairwise p2p combine (the all-to-all SHAPE) ---------------------------

def _ep_p2p_worker(rank, world_size, init_file, seed):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        dense = make_model(seed=seed)
        # the TRUE oracle: the same MoE function with every expert local
        # and the shared expert added exactly once, computed with NO
        # collectives ("none") — a rank-local reference would also
        # reproduce a double-counted shared expert, so it cannot catch
        # that bug; identical seeds pin identical router weights
        torch.manual_seed(seed)
        full = MoETinyTransformer(dense, n_experts=4, top_k=2,
                                  n_shared=1, inter_dim=16,
                                  ep_combine="none")
        torch.manual_seed(seed)
        model = MoETinyTransformer(dense, n_experts=4, top_k=2,
                                   ep_size=world_size, ep_rank=rank,
                                   n_shared=1, inter_dim=16,
                                   ep_combine="p2p")
        engine = Engine(model, block_size=4, num_blocks=32)
        prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
                   torch.tensor([11, 4, 5])]
        reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
        while engine.has_requests():
            engine.step()
        for p, r in zip(prompts, reqs):
            assert engine.output(r)[len(p):] == \
                greedy_reference(full, p, 6), \
                f"EP p2p rank {rank} must equal the FULL MoE function"
    finally:
        dist.destroy_process_group()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ep2_p2p_combine_matches_full_moe(tmp_path):
    init_file = str(tmp_path / "ep2_p2p_store")
    mp.spawn(_ep_p2p_worker, args=(2, init_file, 13), nprocs=2, join=True)
