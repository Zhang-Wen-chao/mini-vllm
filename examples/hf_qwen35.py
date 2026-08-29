"""Run a text-only Qwen3.5 request through mini-vllm.

This path uses Transformers' native hybrid cache, which carries both the
full-attention KV state and the Gated DeltaNet recurrent state.  Pass
``--speculative-tokens N`` to load the checkpoint's separate ``mtp.*`` weights
and enable draft/verify decoding.
"""

import argparse

import torch

from mini_vllm.engine import Engine
from mini_vllm.transformers_adapter import TransformersAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--prompt", default="Explain why Gated DeltaNet helps long context.")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto"
    ).to(device).eval()
    adapter = TransformersAdapter(model)
    if not adapter.is_qwen35:
        raise RuntimeError(f"expected Qwen3.5, got {model.config.model_type!r}")
    if args.speculative_tokens:
        adapter.enable_mtp(model_path=args.model)

    ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long,
                       device=adapter.device)
    engine = Engine(adapter, block_size=16, num_blocks=256,
                    device=adapter.device, dtype=adapter.dtype,
                    speculative_tokens=args.speculative_tokens)
    request = engine.add_request(ids, max_new_tokens=args.max_new)
    while engine.has_requests():
        engine.step()
    generated = engine.output(request)[len(ids):]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
