"""Async engine demo: EngineCore process + AsyncLLM streaming frontend.

Three requests stream concurrently through one EngineCore subprocess; each
token arrives on the request's own asyncio.Queue the moment it is sampled.
The streamed sequences are asserted equal to the dense greedy reference.
"""

import asyncio
from pathlib import Path
import sys

import torch

# Match the README command: allow direct execution from a source checkout
# without requiring an editable package installation first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.async_engine import AsyncLLM
from mini_vllm.model_runner import TinyTransformer


PROMPTS = [[3, 15, 27, 9, 42, 7], [11, 4, 5], [23, 8, 14, 2, 31, 19, 44]]
MAX_NEW = 6


def make_model():
    torch.manual_seed(0)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=2, n_heads=4)


def greedy_reference(model, prompt, max_new_tokens):
    ids = list(prompt)
    for _ in range(max_new_tokens):
        logits = model.dense_forward(torch.tensor(ids))
        ids.append(int(logits[-1].argmax()))
    return ids[len(prompt):]


async def main():
    llm = AsyncLLM(make_model, block_size=4, num_blocks=32)
    try:
        async def one(prompt):
            got = []
            async for tok in llm.generate(prompt, MAX_NEW):
                got.append(tok)
            return got

        outs = await asyncio.gather(*[one(p) for p in PROMPTS])
        assert llm.num_active_requests() == 0, "all requests finished"
    finally:
        llm.shutdown()

    model = make_model()
    for prompt, out in zip(PROMPTS, outs):
        expected = greedy_reference(model, prompt, MAX_NEW)
        assert out == expected, "stream must equal the dense reference"
        print(f"prompt={prompt} -> streamed {out}")


if __name__ == "__main__":
    asyncio.run(main())
    print("async demo passed (EngineCore process + streaming)")
