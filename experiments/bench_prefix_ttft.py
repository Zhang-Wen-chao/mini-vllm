"""Prefix-cache TTFT benchmark (single GPU, mechanism level).

Protocol: N requests arrive one at a time; each runs to exactly one
generated token (max_new_tokens=1), so wall time per request IS its time
to first token (TTFT) — the prefill forward plus scheduling, with decode
cost removed.

Groups:
  shared  — all requests share one long common prefix, suffixes differ;
            with the cache on, requests 2..N must hit whole prefix blocks
  unique  — every request is a different prompt (no hit possible)

Compared configurations: prefix cache on vs off, identical prompt sets,
identical freshly built engine per configuration.  The honest claim this
measures is "prefill compute skipped by whole-block prefix hits", not
end-to-end serving latency.

Usage: python experiments/bench_prefix_ttft.py [--device cuda:0]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# allow direct execution from a source checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer


def make_model(device):
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=16384, d_model=1024, n_layers=12,
                            n_heads=16, max_positions=2048).eval()
    return model.to(device)


def measure(engine, prompts, device):
    """Add requests one at a time (each to 1 token); return per-request TTFT."""
    ttfts = []
    for prompt in prompts:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        engine.add_request(prompt, max_new_tokens=1)
        while engine.has_requests():
            engine.step()
        torch.cuda.synchronize(device)
        ttfts.append(time.perf_counter() - t0)
    return ttfts


def run_group(model, prompts, enable_prefix_cache, device, repeats=3):
    stats = []
    for _ in range(repeats):
        engine = Engine(model, block_size=16, num_blocks=512,
                        enable_prefix_cache=enable_prefix_cache,
                        max_prefill_tokens=8192, max_running_tokens=8192)
        # one throwaway request triggers lazy CUDA kernel setup before
        # timing; deliberately NOT a prefix of any measured prompt so it
        # cannot pollute hits_tokens
        warm = torch.full((64,), model.vocab_size - 1, dtype=torch.long)
        engine.add_request(warm, max_new_tokens=1)
        while engine.has_requests():
            engine.step()
        stats.append(measure(engine, prompts, device))
        hits = engine.prefix_cache.hits_tokens if engine.prefix_cache else 0
    return stats, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=8)
    args = parser.parse_args()
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    model = make_model(device)
    g = torch.Generator().manual_seed(7)
    shared = torch.randint(0, model.vocab_size, (args.prefix_len,), generator=g)
    shared_prompts = [
        torch.cat([shared,
                   torch.randint(0, model.vocab_size, (args.suffix_len,),
                                 generator=g)])
        for _ in range(args.num_requests)]
    unique_prompts = [
        torch.randint(0, model.vocab_size, (args.prefix_len + args.suffix_len,),
                      generator=g)
        for _ in range(args.num_requests)]

    print(f"model: TinyTransformer d_model={model.d_model} "
          f"layers={model.n_layers} heads={model.n_heads} "
          f"(~{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, fp32)")
    print(f"prompts: {args.num_requests} x "
          f"({args.prefix_len} prefix + {args.suffix_len} suffix) tokens, "
          f"device={device}, tf32={torch.backends.cuda.matmul.allow_tf32}")

    for name, prompts in (("shared", shared_prompts),
                          ("unique", unique_prompts)):
        for cache_on in (True, False):
            stats, hits = run_group(model, prompts, cache_on, device)
            means = [sum(r) / len(r) for r in stats]
            best = min(means)
            label = f"{name:6s} prefix_cache={'on ' if cache_on else 'off'}"
            print(f"{label}  TTFT mean={means[-1]*1e3:8.2f} ms  "
                  f"best={best*1e3:8.2f} ms  hits_tokens={hits}")
            if name == "shared" and cache_on:
                assert hits > 0, "shared group must hit with the cache on"
            if name == "shared" and not cache_on:
                assert hits == 0


if __name__ == "__main__":
    main()
