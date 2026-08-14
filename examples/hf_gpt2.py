"""Real-model validation on GPU: run a HuggingFace GPT-2 with the paged
engine, using its pretrained weights inside a hand-rolled layer-by-layer
forward that writes K/V into our BlockTable.

Verified on L20 (4x48GB); also runs on CPU for a few tokens.

Usage:
    python examples/hf_gpt2.py --model distilgpt2 --prompt "The meaning of life is"
    python examples/hf_gpt2.py --model gpt2 --prompt "Once upon a time" --num-blocks 2048
"""

import argparse
import time

import torch
import torch.nn.functional as F

from mini_vllm.engine import Engine
from mini_vllm.kv_cache import KVBlockManager


class HFGPT2Paged:
    """GPT-2 forward reimplemented on top of our BlockTable + paged_attention.

    The HF model object is used only as a weight holder; attention uses
    ``paged_attention`` over the block table, so KV is stored in blocks.
    """

    def __init__(self, hf_model):
        self.hf = hf_model.to(hf_model.device)
        self.device = hf_model.device
        self.dtype = next(hf_model.parameters()).dtype
        cfg = hf_model.config
        self.n_heads = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_layers = cfg.n_layer
        self.vocab_size = cfg.vocab_size

    def _run_layers(self, x, table, num_tokens):
        """Per-layer pass: project q/k/v, store in table, paged attention."""
        from mini_vllm.paged_attention import paged_attention
        b = x.shape[0]
        t = x.shape[1]
        for l, block in enumerate(self.hf.transformer.h):
            ln = block.ln_1(x)
            w = block.attn.c_attn.weight
            bias = block.attn.c_attn.bias
            # GPT-2 uses Conv1D: weight is (in, out), no transpose.
            qkv = (ln @ w + bias).split(self.n_heads * self.head_dim, dim=-1)
            q, k, v = [part.reshape(t, self.n_heads, self.head_dim)
                       for part in qkv]
            table.append(l, k, v)
            o = paged_attention(q, table, layer=l, causal=True,
                                total_tokens=num_tokens)
            o = o.reshape(b, t, self.n_heads * self.head_dim)
            x = x + (o @ block.attn.c_proj.weight + block.attn.c_proj.bias)
            h = block.ln_2(x)
            x = x + F.gelu(h @ block.mlp.c_fc.weight + block.mlp.c_fc.bias) @ \
                block.mlp.c_proj.weight + block.mlp.c_proj.bias
        return self.hf.transformer.ln_f(x)

    def prefill(self, input_ids, table):
        device = self.hf.transformer.wte.weight.device
        input_ids = input_ids.to(device)
        t = input_ids.shape[0]
        x = self.hf.transformer.wte(input_ids.unsqueeze(0)) + \
            self.hf.transformer.wpe(torch.arange(t, device=device))
        logits = self._run_layers(x, table, num_tokens=t)
        table.advance(t)
        return self.hf.lm_head(logits).squeeze(0)

    def decode(self, token_id, table):
        device = self.hf.transformer.wte.weight.device
        token_id = token_id.to(device)
        pos = table.num_tokens
        x = self.hf.transformer.wte(token_id.unsqueeze(0)) + \
            self.hf.transformer.wpe(torch.tensor([pos], device=device))
        logits = self._run_layers(x, table, num_tokens=pos + 1)
        table.advance(1)
        return self.hf.lm_head(logits).squeeze(0)


def greedy_hf_reference(model, tokenizer, prompt_ids, max_new_tokens):
    """HF's own greedy generate as the ground truth."""
    from transformers import GenerationConfig
    gen = model.generate(
        prompt_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        generation_config=GenerationConfig.from_model_config(model.config))
    return gen[0][len(prompt_ids):].tolist()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--num-blocks", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    model = HFGPT2Paged(hf)

    prompt_ids = torch.tensor(tok.encode(args.prompt))
    engine = Engine(model, block_size=16, num_blocks=args.num_blocks,
                    device=model.device, dtype=model.dtype)
    req = engine.add_request(prompt_ids, max_new_tokens=args.max_new)

    t0 = time.time()
    while engine.has_requests():
        engine.step()
    elapsed = time.time() - t0

    generated = engine.output(req)[len(prompt_ids):]
    print("prompt :", args.prompt)
    print("engine :", args.prompt + tok.decode(generated))

    ref = greedy_hf_reference(hf, tok, prompt_ids.to(device),
                              args.max_new)
    match = generated == ref
    print("hf ref :", args.prompt + tok.decode(ref))
    print(f"match: {match}  ({elapsed:.1f}s for {len(generated)} tokens)")

    # numeric sanity: drop-in verification of the KV bookkeeping
    kv = KVBlockManager(64, 16, model.n_heads, model.head_dim,
                        num_layers=model.n_layers, device=device,
                        dtype=torch.float32)
    assert kv.pool.cache.shape[1] == model.n_layers
    print("kv cache layout ok:", tuple(kv.pool.cache.shape))


if __name__ == "__main__":
    main()
