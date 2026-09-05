"""torch.profiler over the engine inner loop (single GPU, mechanism level).

Two profiled phases with record_function labels, so each aggregate table
literally shows the "## ..." label at the top and the kernels underneath it:

  prefill — one 1024-token prompt through add_request + one engine.step()
            (max_new_tokens=1), the TTFT-shaped workload
  decode  — 8 requests batched, decode_tokens engine.step() calls, uniform
            lengths so every step runs the full batch (no tail)

Money metric: wall time per decode step (measured WITHOUT the profiler)
vs summed CUDA kernel time per step (measured WITH it).  The gap is host
scheduling + kernel-launch overhead — the motivation for CUDA-graph capture,
which is the stated next optimization for this engine.  Honest caveat printed
with the numbers: profiler overhead means the kernel share is indicative,
and the wall number comes from an unprofiled run of identical shape.

Usage: python experiments/bench_profiler.py [--device cuda:0]
       (CPU works too; CUDA columns then absent)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

# allow direct execution from a source checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer


def make_model(device, d_model, n_layers, n_heads):
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=16384, d_model=d_model, n_layers=n_layers,
                            n_heads=n_heads, max_positions=2048).eval()
    return model.to(device)


def run_to_completion(engine):
    while engine.has_requests():
        engine.step()


def fresh_engine(model):
    return Engine(model, block_size=16, num_blocks=512,
                  max_prefill_tokens=8192, max_running_tokens=8192)


def activities_for(device):
    acts = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    return acts


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def wall_decode_steps(model, device, decode_tokens, n_requests):
    """Unprofiled: wall time per decode step at batch=n_requests."""
    engine = fresh_engine(model)
    for _ in range(n_requests):
        engine.add_request(
            torch.randint(0, model.vocab_size, (64,)), max_new_tokens=decode_tokens)
    sync(device)
    t0 = time.perf_counter()
    run_to_completion(engine)
    sync(device)
    return (time.perf_counter() - t0) / decode_tokens


def gpu_time_ms_from_trace(path):
    """Total GPU time (ms) parsed from an exported chrome trace.

    key_averages() rows are not trustworthy for this sum: the same kernel is
    attributed to both its launching aten op and its own kernel row (and the
    record_function label row carries an inflated aggregate), so row sums
    shift between torch versions.  The device-track events in the trace are
    the ground truth -- and they match the table's printed
    "Self CUDA time total" line exactly.
    """
    with open(path) as f:
        tr = json.load(f)
    return sum(e.get("dur", 0) for e in tr["traceEvents"]
               if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")) / 1e3


def wall_prefill(model, device, prefix_len):
    """Unprofiled: wall time of one prefill to first token."""
    engine = fresh_engine(model)
    prompt = torch.randint(0, model.vocab_size, (prefix_len,))
    sync(device)
    t0 = time.perf_counter()
    engine.add_request(prompt, max_new_tokens=1)
    run_to_completion(engine)
    sync(device)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--trace", default="experiments/trace_profiler.json")
    parser.add_argument("--trace-prefill", default="experiments/trace_profiler_prefill.json")
    args = parser.parse_args()
    device = args.device
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    model = make_model(device, args.d_model, args.layers, args.heads)
    print(f"model: TinyTransformer d_model={model.d_model} layers={model.n_layers} "
          f"heads={model.n_heads} (~{sum(p.numel() for p in model.parameters())/1e6:.0f}M, "
          f"fp32)  device={device}")

    # warm-up: lazy CUDA kernel setup; throwaway request, nothing measured
    warm = fresh_engine(model)
    warm.add_request(torch.randint(0, model.vocab_size, (64,)), max_new_tokens=1)
    run_to_completion(warm)

    # --- phase 1: prefill (TTFT shape) ---
    with profile(activities=activities_for(device), record_shapes=True) as prof_p:
        with record_function("## prefill"):
            engine = fresh_engine(model)
            engine.add_request(
                torch.randint(0, model.vocab_size, (args.prefix_len,)),
                max_new_tokens=1)
            run_to_completion(engine)
    prof_p.export_chrome_trace(args.trace_prefill)
    prefill_cuda_ms = gpu_time_ms_from_trace(args.trace_prefill)

    # --- phase 2: decode (batch, uniform lengths, no tail) ---
    engine = fresh_engine(model)
    for _ in range(args.num_requests):
        engine.add_request(
            torch.randint(0, model.vocab_size, (64,)),
            max_new_tokens=args.decode_tokens)
    with profile(activities=activities_for(device), record_shapes=True) as prof_d:
        for _ in range(args.decode_tokens):
            with record_function("## decode_step"):
                run_to_completion(engine)
    prof_d.export_chrome_trace(args.trace)
    decode_cuda_ms = gpu_time_ms_from_trace(args.trace)

    # --- money metric ---
    print()
    if on_cuda:
        wall_prefill_s = wall_prefill(model, device, args.prefix_len)
        wall_decode = wall_decode_steps(model, device, args.decode_tokens,
                                        args.num_requests)
        kernel_per_step_ms = decode_cuda_ms / args.decode_tokens
        gap = max(0.0, 1.0 - (kernel_per_step_ms / (wall_decode * 1e3)))
        print(f"prefill (1 x {args.prefix_len} tok -> 1 token):   "
              f"wall={wall_prefill_s*1e3:7.2f} ms  cuda kernels={prefill_cuda_ms:7.2f} ms")
        print(f"decode  ({args.num_requests} req batch, "
              f"{args.decode_tokens} steps):")
        print(f"  wall per step (no profiler) = {wall_decode*1e3:7.3f} ms")
        print(f"  cuda kernels per step (profiled) = {kernel_per_step_ms:7.3f} ms")
        print(f"  non-kernel share (host scheduling + launch) ≈ {gap*100:4.1f}%")
        print("  caveat: kernel time measured under profiler overhead; wall from "
              "an unprofiled run of identical shape — ratio is indicative.")
    else:
        print("CPU device: no CUDA kernel columns; run on GPU for the "
              "kernel-vs-wall split.")

    # --- aggregate tables: label row on top, kernels underneath ---
    sort_key = "cuda_time_total" if on_cuda else "cpu_time_total"
    print("\n=== decode phase table ===")
    print(prof_d.key_averages().table(sort_by=sort_key, row_limit=15))
    print("=== prefill phase table ===")
    print(prof_p.key_averages().table(sort_by=sort_key, row_limit=10))

    print(f"chrome traces -> {args.trace_prefill}, {args.trace}  "
          f"(chrome://tracing or https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
