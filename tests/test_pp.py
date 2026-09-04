"""Pipeline parallelism tests.

Anchors:
- PP is a pure reorganization of the same dense weights, so a PP engine's
  output must equal the dense reference EXACTLY (activations cross stages
  as bit-exact copies);
- each stage keeps only its own layers, and its KV pool stores only its
  own layers (pool sharded by stage, like TP shards by head);
- the sampled-token broadcast keeps every rank's schedule identical.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer
from mini_vllm.pp import PPTPTransformer, PPTransformer

from test_engine import greedy_reference


def make_dense(n_layers=4, seed=0):
    torch.manual_seed(seed)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=n_layers,
                           n_heads=4)


def test_pp_partition_shares_weights_and_validates():
    dense = make_dense(4, seed=0)
    m0 = PPTransformer(dense, pp_size=2, pp_rank=0)
    m1 = PPTransformer(dense, pp_size=2, pp_rank=1)
    assert m0.n_layers == 2 and m1.n_layers == 2
    # stage slices are the dense layers themselves (zero-copy share)
    for stage, slab in ((m0, dense.layers[:2]), (m1, dense.layers[2:])):
        for layer, ref in zip(stage.layers, slab):
            for p, q in zip(layer.parameters(), ref.parameters()):
                assert torch.equal(p, q)
    assert m0.embed is dense.embed and m0.lm_head is None
    assert m1.lm_head is dense.lm_head and m1.embed is None
    with pytest.raises(ValueError):
        PPTransformer(make_dense(3), pp_size=2, pp_rank=0)
    with pytest.raises(ValueError):
        PPTransformer(dense, pp_size=2, pp_rank=2)


def test_pp_one_stage_matches_dense_in_process():
    dense = make_dense(2, seed=1)
    model = PPTransformer(dense, pp_size=1, pp_rank=0)
    engine = Engine(model, block_size=4, num_blocks=32)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    r = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    assert engine.output(r)[len(prompt):] == greedy_reference(dense, prompt, 6)


def test_pp_rejects_cuda_graph():
    dense = make_dense(2, seed=2)
    with pytest.raises(ValueError):
        Engine(PPTransformer(dense, pp_size=2, pp_rank=0), block_size=4,
               num_blocks=8, use_cuda_graph=True)


# -- real multi-process pipelines ---------------------------------------------

def _pp_worker(rank, pp_size, init_file, seed, n_layers):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=pp_size,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        dense = make_dense(n_layers, seed=seed)
        model = PPTransformer(dense, pp_size=pp_size, pp_rank=rank)
        assert model.n_layers == n_layers // pp_size
        engine = Engine(model, block_size=4, num_blocks=64,
                        enable_prefix_cache=True)
        # request 1, then a second sharing its first block: exercises the
        # prefix-cache composite on every rank's stage-local pool
        p1 = torch.tensor([3, 15, 27, 9, 42, 7])
        r1 = engine.add_request(p1, max_new_tokens=6)
        while engine.has_requests():
            engine.step()
        p2 = torch.tensor([3, 15, 27, 9, 50, 51, 52])
        r2 = engine.add_request(p2, max_new_tokens=4)
        while engine.has_requests():
            engine.step()
        assert engine.kv.pool.num_layers == model.n_layers, \
            "each stage's KV pool stores only its own layers"
        assert engine.prefix_cache.hits_tokens == 4, "shared prefix matched"
        for p, r in ((p1, r1), (p2, r2)):
            assert engine.output(r)[len(p):] == greedy_reference(dense, p, 6 if r is r1 else 4), \
                f"PP={pp_size} rank {rank} output must equal the dense reference"
    finally:
        dist.destroy_process_group()


def test_pp2_gloo_matches_dense_reference(tmp_path):
    mp.spawn(_pp_worker, args=(2, str(tmp_path / "pp2"), 7, 2),
             nprocs=2, join=True)


def test_pp3_gloo_relays_through_middle_stage(tmp_path):
    mp.spawn(_pp_worker, args=(3, str(tmp_path / "pp3"), 8, 3),
             nprocs=3, join=True)


# -- Phase 19: micro-batch pipeline / PP×TP / PP×speculative -------------------

def test_pp_tp_validates_before_distributed():
    dense = make_dense(4, seed=3)
    with pytest.raises(ValueError):
        PPTPTransformer(dense, pp_size=1, pp_rank=0, tp_size=3, tp_rank=0)
    # world > 1 without an initialized process group is a hard error —
    # silently running a "distributed" model in-process would hide p2p bugs
    with pytest.raises(RuntimeError):
        PPTPTransformer(dense, pp_size=2, pp_rank=0, tp_size=2, tp_rank=0)


def _pp_micro_worker(rank, pp_size, init_file, seed, n_layers):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=pp_size,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        dense = make_dense(n_layers, seed=seed)
        model = PPTransformer(dense, pp_size=pp_size, pp_rank=rank,
                              micro_batch_size=1)
        engine = Engine(model, block_size=4, num_blocks=64)
        # both requests admitted in the same step → one batched prefill of
        # 2 rows, split into 2 micro-batches that pipeline through the stages
        prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
                   torch.tensor([11, 4, 5])]
        reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
        while engine.has_requests():
            engine.step()
        assert model.micro_prefills >= 1, \
            "micro-batch pipeline must have run, not the lockstep fallback"
        for p, r in zip(prompts, reqs):
            assert engine.output(r)[len(p):] == greedy_reference(dense, p, 6), \
                f"micro-batch PP={pp_size} rank {rank} must equal dense"
    finally:
        dist.destroy_process_group()


def test_pp2_microbatch_pipeline_matches_dense(tmp_path):
    mp.spawn(_pp_micro_worker, args=(2, str(tmp_path / "pp_micro"), 11, 4),
             nprocs=2, join=True)


def _pp_tp_worker(rank, init_file, seed):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        dense = make_dense(4, seed=seed)         # 4 layers, 4 heads
        pp_size = tp_size = 2
        model = PPTPTransformer(dense, pp_size=pp_size, pp_rank=rank // 2,
                                tp_size=tp_size, tp_rank=rank % 2)
        # doubly sharded geometry: layers by stage, heads by TP rank
        assert model.n_layers == 2 and model.n_heads == 2
        engine = Engine(model, block_size=4, num_blocks=64)
        assert engine.kv.pool.num_layers == 2, "pool sharded by stage"
        assert engine.kv.pool.num_heads == 2, "pool sharded by head"
        prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
                   torch.tensor([11, 4, 5])]
        reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
        while engine.has_requests():
            engine.step()
        for p, r in zip(prompts, reqs):
            assert engine.output(r)[len(p):] == greedy_reference(dense, p, 6), \
                f"PP×TP rank {rank} output must equal the dense reference"
    finally:
        dist.destroy_process_group()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pp_tp_2x2_gloo_matches_dense_reference(tmp_path):
    mp.spawn(_pp_tp_worker, args=(str(tmp_path / "pp_tp"), 12),
             nprocs=4, join=True)


class _AlwaysPropose:
    """Deterministic injected proposer (Phase 11 pluggability): always
    drafts the same 3 tokens, guaranteeing the verify path runs even
    though a random-weight model's history has no reusable n-grams."""

    def propose(self, history, max_k=None):
        return [5, 6, 7][:3 if max_k is None else max_k]


def _pp_spec_worker(rank, init_file, seed):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        dense = make_dense(4, seed=seed)
        model = PPTransformer(dense, pp_size=2, pp_rank=rank)
        engine = Engine(model, block_size=4, num_blocks=64,
                        speculative_tokens=3, spec_method="ngram",
                        spec_proposer=_AlwaysPropose())
        prompt = torch.tensor([3, 15, 27, 9, 42, 7])
        r = engine.add_request(prompt, max_new_tokens=8)
        while engine.has_requests():
            engine.step()
        assert engine.spec_stats["drafted"] > 0, "spec path must have run"
        assert engine.output(r)[len(prompt):] == \
            greedy_reference(dense, prompt, 8), \
            "PP×spec output must equal the dense reference"
    finally:
        dist.destroy_process_group()


def test_pp2_ngram_spec_composite(tmp_path):
    # PP×spec should be FREE under the SPMD control plane: the verify
    # forward is just another forward, and the sampled-token broadcast
    # keeps the one-hot proxy contract intact — zero engine changes
    mp.spawn(_pp_spec_worker, args=(str(tmp_path / "pp_spec"), 13),
             nprocs=2, join=True)
