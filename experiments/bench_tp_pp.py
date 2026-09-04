"""TP / PP / micro-batch pipeline benchmark over real multi-process gloo or
NCCL (mechanism level).

Workload: B prompts x L prompt tokens, generated to max_new tokens, greedy.
Two timings per run:
  prefill-only  — max_new_tokens=1, so the wall time is one step: the batched
                  prefill (the micro-batch pipeline path under PP)
  generation    — max_new_tokens=N, prefill + steady decode

Rank 0 asserts its outputs equal the dense greedy reference computed locally,
so every timed configuration is also a correctness check.

Backends: gloo (CPU examples' backend, works over CUDA tensors) and nccl.
L20 has no NVLink — multi-GPU traffic goes over PCIe x16 either way; with
NCCL set NCCL_SHM_DISABLE=1 in the 1g-shm container.

Usage:
  python experiments/bench_tp_pp.py --mode dense            # single GPU
  python experiments/bench_tp_pp.py --mode tp --world 2 --backend nccl
  python experiments/bench_tp_pp.py --mode pp --world 2 --backend nccl
  python experiments/bench_tp_pp.py --mode pp --world 2 --micro 2 ...
"""

import argparse
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch.distributed as dist
import torch.multiprocessing as mp

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer
from mini_vllm.tp import TPTransformer
from mini_vllm.pp import PPTransformer, PPTPTransformer

B = 8
PROMPT_LEN = 256
MAX_NEW = 32
VOCAB = 16384
D_MODEL = 1536
N_LAYERS = 16
N_HEADS = 16
REPEATS = 3


def make_dense(device):
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=VOCAB, d_model=D_MODEL,
                            n_layers=N_LAYERS, n_heads=N_HEADS,
                            max_positions=1024).eval()
    return model.to(device)


def make_prompts():
    g = torch.Generator().manual_seed(7)
    return [torch.randint(0, VOCAB, (PROMPT_LEN,), generator=g)
            for _ in range(B)]


def greedy_reference(model, prompt, max_new_tokens):
    ids = prompt.tolist()
    for _ in range(max_new_tokens):
        logits = model.dense_forward(torch.tensor(ids))
        ids.append(int(logits[-1].argmax()))
    return ids[len(prompt):]


def timed_run(engine, prompts, max_new, device, repeats=REPEATS):
    """Returns (best seconds, outputs of the last repeat)."""
    best, outputs = None, None
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        reqs = [engine.add_request(p, max_new_tokens=max_new)
                for p in prompts]
        while engine.has_requests():
            engine.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        best = elapsed if best is None else min(best, elapsed)
        outputs = [engine.output(r)[len(p):] for p, r in zip(prompts, reqs)]
    return best, outputs


def worker(rank, world_size, mode, micro, backend, init_file, device_ids):
    torch.cuda.set_device(device_ids[rank])
    device = f"cuda:{device_ids[rank]}"
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group(backend, init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=300))
    try:
        dense = make_dense(device)
        if mode == "tp":
            model = TPTransformer(dense, world_size, rank)
        elif mode == "pp":
            model = PPTransformer(dense, pp_size=world_size, pp_rank=rank,
                                  micro_batch_size=micro)
        elif mode == "pptp":
            tp_size = 2
            assert world_size % tp_size == 0, "world must be pp_size * tp_size"
            pp_size = world_size // tp_size
            model = PPTPTransformer(dense, pp_size=pp_size,
                                    pp_rank=rank // tp_size,
                                    tp_size=tp_size, tp_rank=rank % tp_size)
        else:
            raise ValueError(mode)
        engine = Engine(model, block_size=16, num_blocks=512,
                        max_prefill_tokens=8192, max_running_tokens=8192)
        prompts = make_prompts()

        prefill, _ = timed_run(engine, prompts, 1, device, repeats=2)
        gen, outputs = timed_run(engine, prompts, MAX_NEW, device)
        if rank == 0:
            for prompt, out in zip(prompts, outputs):
                assert out == greedy_reference(dense, prompt, MAX_NEW), \
                    f"{mode} output must equal the dense reference"
            gen_tokens = B * MAX_NEW
            print(f"mode={mode} world={world_size} micro={micro} "
                  f"backend={backend} devices={device_ids}  "
                  f"prefill_only={prefill*1e3:8.1f} ms  "
                  f"generation={gen*1e3:8.1f} ms  "
                  f"throughput={gen_tokens/gen:7.1f} tok/s  "
                  f"({B}x{PROMPT_LEN} prompt + {MAX_NEW} new, == dense ref)")
    finally:
        dist.destroy_process_group()


def dense_main(device):
    torch.backends.cuda.matmul.allow_tf32 = True
    model = make_dense(device)
    engine = Engine(model, block_size=16, num_blocks=512,
                    max_prefill_tokens=8192, max_running_tokens=8192)
    prompts = make_prompts()
    prefill, _ = timed_run(engine, prompts, 1, device, repeats=2)
    gen, outputs = timed_run(engine, prompts, MAX_NEW, device)
    for prompt, out in zip(prompts, outputs):
        assert out == greedy_reference(model, prompt, MAX_NEW)
    print(f"mode=dense world=1 micro=None backend=none devices=[{device}]  "
          f"prefill_only={prefill*1e3:8.1f} ms  "
          f"generation={gen*1e3:8.1f} ms  "
          f"throughput={B*MAX_NEW/gen:7.1f} tok/s  "
          f"({B}x{PROMPT_LEN} prompt + {MAX_NEW} new, == dense ref)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["dense", "tp", "pp", "pptp"])
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--micro", type=int, default=None)
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--gpus", default=None,
                        help="comma-separated cuda device ids, default 0..N-1")
    args = parser.parse_args()

    if args.mode == "dense":
        dense_main("cuda:0")
        return
    assert N_LAYERS % args.world == 0 or args.mode == "tp", \
        "n_layers must divide evenly across stages"
    device_ids = ([int(x) for x in args.gpus.split(",")] if args.gpus
                  else list(range(args.world)))
    assert len(device_ids) == args.world
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker,
                 args=(args.world, args.mode, args.micro, args.backend,
                       f"{tmp}/bench_store", device_ids),
                 nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
