"""Decode-path profiler driver: mini-vllm vs vLLM on the identical workload.

Kernel-level attribution for the bench_fair numbers: where does the wall
time go — CUDA kernels, or host scheduling/launch overhead around them?

Usage (nsys wraps the whole process; NVTX ranges mark the steady decode;
run with the venv's python that has vllm installed):

    nsys profile -t cuda,nvtx --cuda-graph-trace=node \
        -o experiments/nsys_mini_gpt2 \
        python experiments/profile_decode.py --engine mini
    VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile -t cuda,nvtx \
        --cuda-graph-trace=node -o experiments/nsys_vllm_gpt2 \
        python experiments/profile_decode.py --engine vllm

Two pitfalls baked into those flags:
- ``--cuda-graph-trace=node``: nsys does not trace kernels inside CUDA graph
  replays by default, so kernel share reads falsely low (5% instead of ~50%).
- ``VLLM_ENABLE_V1_MULTIPROCESSING=0``: vLLM's EngineCore runs in a
  subprocess, so profilers in the parent process see zero CUDA activity.
  Only for profiling — performance numbers should keep the default
  multiprocess mode (vLLM's own benchmarks/startup.py does the same).

With --torch-profiler, an in-process kineto profile of the same region is
also exported as JSON, and the money metric is printed directly:

    wall time per decode step (unprofiled run)
  vs summed CUDA kernel time per step (profiled run)
    -> the gap is host scheduling + kernel-launch overhead.

Prompts/metrics mirror examples/bench_fair.py exactly (same 8 base prompts,
same repeat-to-length rule, greedy sampling, warmup before timing).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPTS = [
    "The meaning of life is", "Once upon a time",
    "In the beginning, the universe", "The secret to happiness is",
    "Artificial intelligence will", "A wise old owl once said",
    "The future of technology", "When I was young, I believed",
]


def make_prompts(n, prompt_len, tok):
    out = []
    for p in PROMPTS[:n]:
        ids = tok.encode(p)
        while len(ids) < prompt_len:
            ids = ids + ids[: prompt_len - len(ids)]
        out.append(tok.decode(ids[:prompt_len]))
    return out


def make_adapter(hf_model, model_name):
    if "gpt" in model_name.lower():
        from examples.hf_gpt2 import HFGPT2Paged
        return HFGPT2Paged(hf_model)
    from examples.hf_llama import HFQwenPaged
    return HFQwenPaged(hf_model)


def mini_side(args, tok, prompts):
    from transformers import AutoModelForCausalLM
    from mini_vllm.engine import Engine

    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).cuda().eval()
    model = make_adapter(hf, args.model)
    engine = Engine(model, block_size=16, num_blocks=512,
                    device=model.device, dtype=model.dtype,
                    use_cuda_graph=True)
    engine.warmup(len(prompts), max(len(tok.encode(p)) for p in prompts),
                  args.max_new)
    for p in prompts:
        engine.add_request(torch.tensor(tok.encode(p)),
                           max_new_tokens=args.max_new)

    # unprofiled wall time: step 0 = prefill, rest = steady decode
    step_times = []
    while engine.has_requests():
        t0 = time.time()
        engine.step()
        step_times.append(time.time() - t0)
    wall_ms = 1000.0 * sum(step_times[1:]) / max(len(step_times) - 1, 1)
    print(f"[mini] unprofiled: {len(step_times)-1} decode steps, "
          f"wall {wall_ms:.2f} ms/step")

    # profiled pass: same shape, fresh engine
    del engine
    torch.cuda.empty_cache()
    engine = Engine(model, block_size=16, num_blocks=512,
                    device=model.device, dtype=model.dtype,
                    use_cuda_graph=True)
    engine.warmup(len(prompts), max(len(tok.encode(p)) for p in prompts),
                  args.max_new)
    for p in prompts:
        engine.add_request(torch.tensor(tok.encode(p)),
                           max_new_tokens=args.max_new)
    engine.step()  # prefill only, outside the window; rest is decode-only

    if args.torch_profiler:
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.nvtx.range_push("steady_decode")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            steps = 0
            while engine.has_requests():
                engine.step()
                steps += 1
        torch.cuda.nvtx.range_pop()
        out = str(Path(__file__).parent / "profile_mini_decode.json")
        prof.export_chrome_trace(out)
        ka = prof.key_averages()
        kernel_us = sum(e.self_device_time_total for e in ka
                        if e.device_type == torch.autograd.DeviceType.CUDA)
        print(f"[mini] profiled: {steps} steps, kernel-sum "
              f"{kernel_us/1000.0/max(steps,1):.2f} ms/step "
              f"(gap wall-kernel = host overhead)")
    else:
        torch.cuda.nvtx.range_push("steady_decode")
        while engine.has_requests():
            engine.step()
        torch.cuda.nvtx.range_pop()
    return {"engine": "mini", "wall_ms_per_step": wall_ms}


def vllm_side(args, tok, prompts):
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=0.9)
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=4))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    # unprofiled wall
    t0 = time.time()
    llm.generate(prompts, sp)
    wall_s = time.time() - t0
    total_tokens = len(prompts) * args.max_new
    print(f"[vllm] unprofiled: generate wall {wall_s:.2f}s, "
          f"{total_tokens/wall_s:.0f} tok/s, "
          f"{1000.0*wall_s/max(total_tokens-len(prompts),1):.2f} ms/token-step")

    if args.torch_profiler:
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.nvtx.range_push("steady_decode")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            t0 = time.time()
            llm.generate(prompts, sp)
            wall_s = time.time() - t0
        torch.cuda.nvtx.range_pop()
        out = str(Path(__file__).parent / "profile_vllm_generate.json")
        prof.export_chrome_trace(out)
        ka = prof.key_averages()
        kernel_us = sum(e.self_device_time_total for e in ka
                        if e.device_type == torch.autograd.DeviceType.CUDA)
        print(f"[vllm] profiled: generate wall {wall_s:.2f}s, kernel-sum "
              f"{kernel_us/1e6:.2f}s, share {kernel_us/1e6/wall_s*100:.1f}%")
    else:
        torch.cuda.nvtx.range_push("steady_decode")
        llm.generate(prompts, sp)
        torch.cuda.nvtx.range_pop()
    return {"engine": "vllm", "wall_s": wall_s}


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mini", "vllm"], required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--torch-profiler", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = make_prompts(args.batch, args.prompt_len, tok)
    if args.engine == "mini":
        summary = mini_side(args, tok, prompts)
    else:
        summary = vllm_side(args, tok, prompts)
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
