"""Fair benchmark: mini-vllm vs vLLM on equal footing.

Rules of fairness:
- fp16 only (vLLM's native domain; V1 engine at gpu_memory_utilization=0.9)
- same model (gpt2), same prompts, same greedy sampling
- TTFT measured identically: single request, max_tokens=1
- TPOT measured identically: (e2e - TTFT) / (total tokens - num requests)
- batch throughput: all requests together

Usage (container with vLLM + transformers 4.49):
    python examples/bench_fair.py [--batch 8] [--max-new 100] [--prompt-len 16]
"""

import argparse
import os
import time

import torch

from mini_vllm.engine import Engine
from examples.hf_gpt2 import HFGPT2Paged

PROMPTS = [
    "The meaning of life is", "Once upon a time",
    "In the beginning, the universe", "The secret to happiness is",
    "Artificial intelligence will", "A wise old owl once said",
    "The future of technology", "When I was young, I believed",
]


def make_adapter(hf_model, model_name):
    if "gpt" in model_name.lower():
        from examples.hf_gpt2 import HFGPT2Paged
        return HFGPT2Paged(hf_model)
    from examples.hf_llama import HFQwenPaged
    return HFQwenPaged(hf_model)


def make_prompts(n, prompt_len, tok):
    base = PROMPTS[:n]
    # extend prompts to approximately prompt_len tokens by repeating
    out = []
    for p in base:
        ids = tok.encode(p)
        while len(ids) < prompt_len:
            ids = ids + ids[: prompt_len - len(ids)]
        out.append(tok.decode(ids[:prompt_len]))
    return out


def mini_metrics(model, tok, prompts, max_new):
    # warmup: one round so CUDA graph capture happens outside the timing
    warm = Engine(model, block_size=16, num_blocks=512, device=model.device,
                  dtype=model.dtype, use_cuda_graph=True)
    warm_ids = torch.tensor(tok.encode(prompts[0]))
    warm.add_request(warm_ids, max_new_tokens=4)
    while warm.has_requests():
        warm.step()
    del warm
    torch.cuda.empty_cache()
    # TTFT: single-request, one token (engine warmed up like vLLM init)
    ttfts = []
    for p in prompts:
        ids = torch.tensor(tok.encode(p))
        eng = Engine(model, block_size=16, num_blocks=512,
                     device=model.device, dtype=model.dtype,
                     use_cuda_graph=True)
        eng.warmup(1, ids.shape[0], 1)
        eng.add_request(ids, max_new_tokens=1)
        t0 = time.time()
        while eng.has_requests():
            eng.step()
        ttfts.append(time.time() - t0)
        del eng
        torch.cuda.empty_cache()
    ttft = sum(ttfts) / len(ttfts)
    # batch e2e (steady-state: warmup pre-captures all graphs like vLLM's
    # init; timed run then reuses them with zero capture cost)
    engine = Engine(model, block_size=16, num_blocks=512,
                    device=model.device, dtype=model.dtype,
                    use_cuda_graph=True)
    engine.warmup(len(prompts), max(len(tok.encode(p)) for p in prompts),
                  max_new)
    for p in prompts:
        ids = torch.tensor(tok.encode(p))
        engine.add_request(ids, max_new_tokens=max_new)
    step_times = []
    while engine.has_requests():
        t0 = time.time()
        engine.step()
        step_times.append(time.time() - t0)
    # step 1 = prefill (graphs already captured); rest are steady decode
    steady = sum(step_times[1:])
    total_tokens = len(prompts) * max_new
    e2e = max(steady, 1e-6)
    tpot = e2e / max(total_tokens - len(prompts), 1)
    return {
        "ttft_ms": ttft * 1000, "tpot_ms": tpot * 1000,
        "e2e_s": e2e, "tok_s": total_tokens / e2e,
    }


def vllm_metrics(tok, prompts, max_new, model_name="gpt2"):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name, dtype="float16", max_model_len=2048,
              gpu_memory_utilization=0.9)
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=4))
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    ttfts = []
    for p in prompts:
        t0 = time.time()
        llm.generate([p], sp1)
        ttfts.append(time.time() - t0)
    ttft = sum(ttfts) / len(ttfts)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    t0 = time.time()
    llm.generate(prompts, sp)
    e2e = time.time() - t0
    total_tokens = len(prompts) * max_new
    tpot = (e2e - ttft) / max(total_tokens - len(prompts), 1)
    return {
        "ttft_ms": ttft * 1000, "tpot_ms": tpot * 1000,
        "e2e_s": e2e, "tok_s": total_tokens / e2e,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = make_prompts(args.batch, args.prompt_len, tok)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).cuda().eval()
    model = make_adapter(hf, args.model)

    print(f"== {args.model} fp16 | batch={args.batch} | prompt_len={args.prompt_len} "
          f"| max_new={args.max_new} ==")
    m = mini_metrics(model, tok, prompts, args.max_new)
    print(f"mini-vllm : TTFT {m['ttft_ms']:.0f} ms | TPOT {m['tpot_ms']:.1f} ms "
          f"| e2e {m['e2e_s']:.2f}s | {m['tok_s']:.0f} tok/s")
    del model, hf
    torch.cuda.empty_cache()
    v = vllm_metrics(tok, prompts, args.max_new, args.model)
    print(f"vllm (V1): TTFT {v['ttft_ms']:.0f} ms | TPOT {v['tpot_ms']:.1f} ms "
          f"| e2e {v['e2e_s']:.2f}s | {v['tok_s']:.0f} tok/s")
    print(f"ratio: TTFT {m['ttft_ms']/v['ttft_ms']:.2f}x | "
          f"TPOT {m['tpot_ms']/v['tpot_ms']:.2f}x | "
          f"throughput {m['tok_s']/v['tok_s']:.2f}x")


if __name__ == "__main__":
    main()
