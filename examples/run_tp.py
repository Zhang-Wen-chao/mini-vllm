"""Tensor-parallel demo: TP=2 over two real gloo processes (CPU only).

Every rank builds the SAME dense model, keeps only its head shard
(TPTransformer holds shared views of the dense weights), and runs the full
SPMD engine. Rank 0 asserts its outputs equal the dense greedy reference
computed locally — the shard math must be exactly equivalent.
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
from mini_vllm.tp import TPTransformer


PROMPTS = [torch.tensor([3, 15, 27, 9, 42, 7]),
           torch.tensor([11, 4, 5])]
MAX_NEW = 6


def make_dense():
    torch.manual_seed(0)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=2, n_heads=4)


def greedy_reference(model, prompt, max_new_tokens):
    ids = prompt.tolist()
    for _ in range(max_new_tokens):
        logits = model.dense_forward(torch.tensor(ids))
        ids.append(int(logits[-1].argmax()))
    return ids[len(prompt):]


def tp_worker(rank, world_size, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    try:
        dense = make_dense()
        tp = TPTransformer(dense, world_size, rank)
        assert tp.n_heads == dense.n_heads // world_size
        engine = Engine(tp, block_size=4, num_blocks=16)
        assert engine.kv.pool.num_heads == tp.n_heads, \
            "the KV pool shards with the heads"
        reqs = [engine.add_request(p, max_new_tokens=MAX_NEW)
                for p in PROMPTS]
        while engine.has_requests():
            engine.step()
        if rank == 0:
            for p, r in zip(PROMPTS, reqs):
                out = engine.output(r)[len(p):]
                ref = greedy_reference(dense, p, MAX_NEW)
                assert out == ref, "TP output must equal the dense reference"
                print(f"rank0  prompt={p.tolist()} -> {out}  (== dense reference)")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(tp_worker, args=(world_size, f"{tmp}/tp_store"),
                 nprocs=world_size, join=True)
    print(f"TP={world_size} demo passed (CPU gloo)")
