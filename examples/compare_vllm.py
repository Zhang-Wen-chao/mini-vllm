"""Compare mini-vllm against real vLLM on the same model and prompts.

Runs on the L20 host (vLLM requires GPU). Both engines get the same greedy
sampling setup; we compare:

1. token-exact correctness of generated outputs
2. end-to-end throughput (tokens/s) for a batch of concurrent requests
3. KV block bookkeeping (mini-vllm side; vLLM also uses block_size=16)

Usage (inside a container with vLLM installed):
    python examples/compare_vllm.py --model gpt2 --max-new 20 --n-requests 8
"""

import argparse
import time

import torch

from mini_vllm.engine import Engine
from examples.hf_gpt2 import HFGPT2Paged

PROMPTS = [
    "The meaning of life is",
    "Once upon a time",
    "In the beginning, the universe",
    "The secret to happiness is",
    "Artificial intelligence will",
    "A wise old owl once said",
    "The future of technology",
    "When I was young, I believed",
]


def run_mini(hf_model, tok, prompts, max_new, block_size=16, num_blocks=4096):
    model = HFGPT2Paged(hf_model)
    engine = Engine(model, block_size=block_size, num_blocks=num_blocks,
                    device=model.device)
    reqs = []
    for p in prompts:
        ids = torch.tensor(tok.encode(p))
        reqs.append((p, ids, engine.add_request(ids, max_new_tokens=max_new)))
    t0 = time.time()
    steps = 0
    peak_blocks = 0
    while engine.has_requests():
        engine.step()
        steps += 1
        peak_blocks = max(peak_blocks,
                          engine.kv.pool.num_blocks - len(engine.kv.pool.free_blocks))
    dt = time.time() - t0
    outs = []
    for p, ids, r in reqs:
        outs.append((p, engine.output(r)[len(ids):]))
    total_tokens = sum(len(o) for _, o in outs)
    return outs, dt, total_tokens, steps, peak_blocks


def run_vllm(tok, prompts, max_new, dtype="float32"):
    from vllm import LLM, SamplingParams
    llm = LLM(model="gpt2", dtype=dtype, max_model_len=1024,
              gpu_memory_utilization=0.4)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    t0 = time.time()
    results = llm.generate(prompts, sp)
    dt = time.time() - t0
    outs = [(p, r.outputs[0].token_ids)
            for p, r in zip(prompts, results)]
    total_tokens = sum(len(o) for _, o in outs)
    return outs, dt, total_tokens


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--n-requests", type=int, default=8)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    prompts = PROMPTS[:args.n_requests]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype).eval()

    print(f"== {args.model} | {args.n_requests} requests | max_new={args.max_new} | {args.dtype} ==")

    # mini-vllm
    t0 = time.time()
    m_outs, m_dt, m_tokens, steps, peak_blocks = run_mini(hf, tok, prompts,
                                                          args.max_new)
    m_load = time.time() - t0
    print(f"mini-vllm : {m_tokens} tokens in {m_dt:.2f}s = "
          f"{m_tokens / m_dt:.1f} tok/s | {steps} steps | peak {peak_blocks} KV blocks")

    # vLLM
    t0 = time.time()
    v_outs, v_dt, v_tokens = run_vllm(tok, prompts, args.max_new, args.dtype)
    v_load = time.time() - t0
    print(f"vllm      : {v_tokens} tokens in {v_dt:.2f}s = "
          f"{v_tokens / v_dt:.1f} tok/s")

    # correctness: compare text (canonical) and token ids (excluding EOS)
    mismatches = 0
    for (pm, om), (pv, ov) in zip(m_outs, v_outs):
        eos = 50256
        ov_clean = [t for t in ov if t != eos]
        text_match = tok.decode(om) == tok.decode(ov_clean)
        ids_match = om == ov_clean
        if not (text_match and ids_match):
            mismatches += 1
            print(f"  {pm[:30]!r}: text={'OK' if text_match else 'DIFF'} "
                  f"ids={'OK' if ids_match else 'DIFF'} "
                  f"(mini {len(om)} vs vllm {len(ov)} ids)")
            if not text_match:
                print(f"    mini: {tok.decode(om)!r}")
                print(f"    vllm: {tok.decode(ov_clean)!r}")
        else:
            print(f"  {pm[:30]!r}: OK ({len(om)} tokens)")
    print(f"exact match: {args.n_requests - mismatches}/{args.n_requests}")
    print(f"throughput ratio mini/vllm: {m_tokens / m_dt / (v_tokens / v_dt):.3f}")


if __name__ == "__main__":
    main()
