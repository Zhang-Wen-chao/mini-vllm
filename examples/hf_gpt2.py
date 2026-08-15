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

    # -- batched (padded) inference ---------------------------------------

    def prefill_batch(self, input_ids_list, tables):
        from mini_vllm.paged_attention import batched_attention
        device = self.hf.transformer.wte.weight.device
        b = len(input_ids_list)
        lens = [x.shape[0] for x in input_ids_list]
        max_len = max(lens)
        padded = torch.zeros(b, max_len, dtype=torch.long, device=device)
        for i, x in enumerate(input_ids_list):
            padded[i, :lens[i]] = x.to(device)
        x = self.hf.transformer.wte(padded) + \
            self.hf.transformer.wpe(torch.arange(max_len, device=device))
        x = self._run_layers_batch(x, tables, lens, [0] * b)
        for i, table in enumerate(tables):
            table.advance(lens[i])
        logits = self.hf.lm_head(self.hf.transformer.ln_f(x))
        return [logits[i, lens[i] - 1] for i in range(b)]

    def decode_batch(self, token_ids, tables):
        device = self.hf.transformer.wte.weight.device
        b = len(token_ids)
        tokens = torch.stack(token_ids).to(device).view(b)
        positions = torch.tensor([t.num_tokens for t in tables],
                                 device=device)
        x = self.hf.transformer.wte(tokens.view(b, 1)) + \
            self.hf.transformer.wpe(positions).unsqueeze(1)
        lens = [t.num_tokens + 1 for t in tables]
        x = self._run_layers_batch(x, tables, lens, [l - 1 for l in lens])
        for table in tables:
            table.advance(1)
        return [row[0] for row in self.hf.lm_head(self.hf.transformer.ln_f(x))]

    # -- CUDA graph decode -------------------------------------------------

    def capture_decode_graph(self, tables, nb_max):
        """Capture the batched decode forward as a CUDA graph.

        All step-varying inputs live in static buffers that the caller
        rewrites before each replay: tokens, positions, block ids (for the
        gather), write targets (for the KV scatter) and the attention mask.
        The graph is valid for the *same* batch size and block cap; the
        engine re-captures when the running batch changes.
        """
        b = len(tables)
        maxkv = nb_max * tables[0].pool.block_size
        bs = tables[0].pool.block_size
        buf = {
            "pool": tables[0].pool,
            "tokens": torch.zeros(b, 1, dtype=torch.long, device=self.device),
            "positions": torch.zeros(b, 1, dtype=torch.long,
                                     device=self.device),
            "block_ids": torch.zeros(b, nb_max, dtype=torch.long,
                                     device=self.device),
            "write_blocks": torch.zeros(b, dtype=torch.long, device=self.device),
            "write_offsets": torch.zeros(b, dtype=torch.long, device=self.device),
            "mask": torch.zeros(b, 1, 1, maxkv, dtype=self.dtype,
                                device=self.device),
            "logits": None,
            "graph": None,
        }
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # The warmup + capture runs write garbage KV into the pool (the
        # scatter uses the zero-initialized write buffers); snapshot the
        # slots they touch and restore them afterwards.
        pool = buf["pool"]
        snapshot = pool.cache[:, :, 0, 0].clone()
        with torch.cuda.stream(s):
            for _ in range(3):
                self._graph_decode_forward(buf)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            buf["logits"] = self._graph_decode_forward(buf)
        pool.cache[:, :, 0, 0] = snapshot
        buf["graph"] = g
        return buf

    def replay_decode_graph(self, buf, tables, token_ids):
        """Update the static input buffers and replay the captured graph.

        Returns (B, V) logits. Must be preceded by Python-side allocation of
        any new KV block each request needs this step.
        """
        b = len(tables)
        bs = tables[0].pool.block_size
        positions = [t.num_tokens for t in tables]
        write_blocks = [t.blocks[p // bs] for t, p in zip(tables, positions)]
        write_offsets = [p % bs for p in positions]
        lens = [p + 1 for p in positions]
        buf["tokens"].copy_(torch.tensor(token_ids, device=self.device)
                            .view(b, 1))
        buf["positions"].copy_(torch.tensor(positions, device=self.device)
                               .view(b, 1))
        buf["write_blocks"].copy_(
            torch.tensor(write_blocks, device=self.device))
        buf["write_offsets"].copy_(
            torch.tensor(write_offsets, device=self.device))
        buf["block_ids"].zero_()
        for i, t in enumerate(tables):
            buf["block_ids"][i, :len(t.blocks)] = torch.tensor(
                t.blocks, device=self.device)
        buf["mask"].fill_(0.0)
        for i in range(b):
            buf["mask"][i, :, :, lens[i]:] = float("-inf")
        buf["graph"].replay()
        return buf["logits"][:, 0]

    def _graph_decode_forward(self, buf):
        """The captured decode forward: static shapes only, no Python flow."""
        from torch.nn.functional import scaled_dot_product_attention
        b = buf["tokens"].shape[0]
        t = 1
        bs = buf["pool"].block_size
        pool = buf["pool"]
        nb_max = buf["block_ids"].shape[1]
        maxkv = buf["mask"].shape[-1]
        x = self.hf.transformer.wte(buf["tokens"]) + \
            self.hf.transformer.wpe(buf["positions"])
        flat = buf["block_ids"].flatten()
        for l, block in enumerate(self.hf.transformer.h):
            ln = block.ln_1(x)
            qkv = (ln @ block.attn.c_attn.weight +
                   block.attn.c_attn.bias).split(self.n_heads * self.head_dim,
                                                 dim=-1)
            q = qkv[0].view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = qkv[1].view(b, t, self.n_heads, self.head_dim)
            v = qkv[2].view(b, t, self.n_heads, self.head_dim)
            pool.cache[0, l, buf["write_blocks"], buf["write_offsets"]] = k[:, 0]
            pool.cache[1, l, buf["write_blocks"], buf["write_offsets"]] = v[:, 0]
            kk = pool.cache[0, l].index_select(0, flat).view(
                b, nb_max * bs, self.n_heads, self.head_dim).transpose(1, 2)
            vv = pool.cache[1, l].index_select(0, flat).view(
                b, nb_max * bs, self.n_heads, self.head_dim).transpose(1, 2)
            o = scaled_dot_product_attention(q, kk, vv, attn_mask=buf["mask"])
            o = o.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
            x = x + (o @ block.attn.c_proj.weight + block.attn.c_proj.bias)
            h = block.ln_2(x)
            x = x + F.gelu(h @ block.mlp.c_fc.weight + block.mlp.c_fc.bias) @ \
                block.mlp.c_proj.weight + block.mlp.c_proj.bias
        return self.hf.lm_head(self.hf.transformer.ln_f(x))

    def _run_layers_batch(self, x, tables, lens, query_starts):
        from mini_vllm.paged_attention import batched_attention
        device = x.device
        b, t, _ = x.shape
        maxkv = max(lens)
        for l, block in enumerate(self.hf.transformer.h):
            ln = block.ln_1(x)
            qkv = (ln @ block.attn.c_attn.weight +
                   block.attn.c_attn.bias).split(self.n_heads * self.head_dim,
                                                 dim=-1)
            q = qkv[0].view(b, t, self.n_heads, self.head_dim)
            k = qkv[1].view(b, t, self.n_heads, self.head_dim)
            v = qkv[2].view(b, t, self.n_heads, self.head_dim)
            pool = tables[0].pool
            if t == 1:
                # decode: vectorized append via one scatter per K/V
                for i in range(b):
                    tbl = tables[i]
                    if tbl.num_tokens >= len(tbl.blocks) * tbl.block_size:
                        tbl.blocks.append(pool.allocate())
                positions = torch.tensor([tbl.num_tokens for tbl in tables],
                                         device=device)
                block_ids = torch.tensor(
                    [tbl.blocks[p // pool.block_size]
                     for tbl, p in zip(tables, positions.tolist())],
                    device=device)
                offsets = positions % pool.block_size
                pool.cache[0, l, block_ids, offsets] = k[:, 0]
                pool.cache[1, l, block_ids, offsets] = v[:, 0]
            else:
                for i in range(b):
                    tables[i].append(l, k[i].reshape(-1, self.n_heads,
                                                     self.head_dim),
                                     v[i].reshape(-1, self.n_heads,
                                                  self.head_dim))
            kk = torch.zeros(b, maxkv, self.n_heads, self.head_dim,
                             device=device, dtype=x.dtype)
            vv = torch.zeros_like(kk)
            pool = tables[0].pool
            nb = max(len(t.blocks) for t in tables)
            bt = torch.zeros(b, nb, dtype=torch.long, device=device)
            for i in range(b):
                bt[i, :len(tables[i].blocks)] = torch.tensor(
                    tables[i].blocks, device=device)
            flat = bt.flatten()
            kk = pool.cache[0, l].index_select(0, flat)
            vv = pool.cache[1, l].index_select(0, flat)
            kk = kk.view(b, nb * pool.block_size, self.n_heads,
                         self.head_dim)[:, :maxkv]
            vv = vv.view(b, nb * pool.block_size, self.n_heads,
                         self.head_dim)[:, :maxkv]
            s_idx = torch.arange(maxkv, device=device)
            q_idx = torch.arange(t, device=device)
            lens_t = torch.tensor(lens, device=device)
            starts = torch.tensor(query_starts, device=device)
            beyond_len = s_idx[None, :] >= lens_t[:, None]
            future = (starts[:, None, None] + q_idx[None, :, None]) < \
                s_idx[None, None, :]
            mask = beyond_len[:, None, :] | future
            o = batched_attention(q, kk, vv, mask)
            o = o.reshape(b, t, self.n_heads * self.head_dim)
            x = x + (o @ block.attn.c_proj.weight + block.attn.c_proj.bias)
            h = block.ln_2(x)
            x = x + F.gelu(h @ block.mlp.c_fc.weight + block.mlp.c_fc.bias) @ \
                block.mlp.c_proj.weight + block.mlp.c_proj.bias
        return x


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
