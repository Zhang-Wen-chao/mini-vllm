"""Pipeline-parallel demo: PP=2 (or 4) over real gloo processes (CPU only).

The dense model's LAYERS are split into contiguous slabs — one stage per
rank. Rank 0 owns the embedding + first slab, the last rank owns the final
slab + lm_head; activations flow rank by rank and the last stage samples
and broadcasts token ids. Rank 0 asserts its outputs equal the dense greedy
reference computed locally.
"""

import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch

# Match the README command: allow direct execution from a source checkout
# without requiring an editable package installation first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch.distributed as dist
import torch.multiprocessing as mp

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer
from mini_vllm.pp import PPTransformer


PROMPTS = [torch.tensor([3, 15, 27, 9, 42, 7]),
           torch.tensor([11, 4, 5])]
MAX_NEW = 6
N_LAYERS = 4


def make_dense():
    torch.manual_seed(0)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=N_LAYERS,
                           n_heads=4)


def greedy_reference(model, prompt, max_new_tokens):
    ids = prompt.tolist()
    for _ in range(max_new_tokens):
        logits = model.dense_forward(torch.tensor(ids))
        ids.append(int(logits[-1].argmax()))
    return ids[len(prompt):]


def pp_worker(rank, pp_size, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=pp_size,
                            timeout=timedelta(seconds=120))
    try:
        dense = make_dense()
        pp = PPTransformer(dense, pp_size=pp_size, pp_rank=rank)
        assert pp.n_layers == N_LAYERS // pp_size
        engine = Engine(pp, block_size=4, num_blocks=16)
        assert engine.kv.pool.num_layers == pp.n_layers, \
            "each stage's KV pool stores only its own layers"
        reqs = [engine.add_request(p, max_new_tokens=MAX_NEW)
                for p in PROMPTS]
        while engine.has_requests():
            engine.step()
        if rank == 0:
            for p, r in zip(PROMPTS, reqs):
                out = engine.output(r)[len(p):]
                ref = greedy_reference(dense, p, MAX_NEW)
                assert out == ref, "PP output must equal the dense reference"
                print(f"rank0  prompt={p.tolist()} -> {out}  (== dense reference)")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    pp_size = 2
    if len(sys.argv) > 1:
        pp_size = int(sys.argv[1])
    assert N_LAYERS % pp_size == 0, "n_layers must divide evenly across stages"
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(pp_worker, args=(pp_size, f"{tmp}/pp_store"),
                 nprocs=pp_size, join=True)
    print(f"PP={pp_size} demo passed (CPU gloo)")
