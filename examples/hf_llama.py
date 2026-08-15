"""Llama-family (Qwen2) adapter: RMSNorm + RoPE + SwiGLU + GQA on paged KV.

The HF model is used as a weight holder; attention runs through our
``paged_attention``/``batched_attention`` over the block table, with GQA
head expansion handled by those functions. RoPE reuses HF's rotary
embeddings and ``apply_rotary_pos_emb`` so numerics match HF exactly.
"""

import torch
import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from mini_vllm.paged_attention import batched_attention, paged_attention


class HFQwenPaged:
    def __init__(self, hf_model):
        self.hf = hf_model.to(hf_model.device)
        self.device = hf_model.device
        self.dtype = next(hf_model.parameters()).dtype
        cfg = hf_model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.n_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        # merged projections per layer: qkv as one matmul, gate/up as one
        with torch.no_grad():
            self.w_qkv = [
                torch.cat([layer.self_attn.q_proj.weight,
                           layer.self_attn.k_proj.weight,
                           layer.self_attn.v_proj.weight], dim=0).to(
                    self.device)
                for layer in hf_model.model.layers]
            self.b_qkv = [
                torch.cat([layer.self_attn.q_proj.bias,
                           layer.self_attn.k_proj.bias,
                           layer.self_attn.v_proj.bias], dim=0).to(
                    self.device)
                for layer in hf_model.model.layers]
            self.w_gateup = [
                torch.cat([layer.mlp.gate_proj.weight,
                           layer.mlp.up_proj.weight], dim=0).to(self.device)
                for layer in hf_model.model.layers]

    # -- layer primitives --------------------------------------------------

    def _qkv(self, x, l):
        """Merged q/k/v projection -> (q, k, v) in (T, H, D) layout."""
        qkv = x @ self.w_qkv[l].T + self.b_qkv[l]
        t = x.shape[0]
        q = qkv[:, :self.n_heads * self.head_dim].view(
            t, self.n_heads, self.head_dim)
        k = qkv[:, self.n_heads * self.head_dim:
                self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim
                ].view(t, self.n_kv_heads, self.head_dim)
        v = qkv[:, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim:
                ].view(t, self.n_kv_heads, self.head_dim)
        return q, k, v


    def _rotate(self, q, k, pos_start, t):
        """Apply HF's rotary embeddings to q/k of shape (T, H, D)."""
        positions = torch.arange(pos_start, pos_start + t,
                                 device=self.device).view(1, -1)
        cos, sin = self.hf.model.rotary_emb(q, positions)
        # HF's helper expects (B, H, S, D); add the batch dim explicitly
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin, unsqueeze_dim=1)
        return q4.squeeze(0).transpose(0, 1), k4.squeeze(0).transpose(0, 1)

    def _attn_block(self, x, l, table, pos_start, t, num_tokens):
        """One self-attention block with paged attention (single sequence)."""
        layer = self.hf.model.layers[l]
        ln = layer.input_layernorm(x)
        q, k, v = self._qkv(ln, l)
        q, k = self._rotate(q, k, pos_start, t)
        table.append(l, k, v)
        o = paged_attention(q, table, layer=l, causal=True,
                            total_tokens=num_tokens)
        o = o.reshape(t, self.n_heads * self.head_dim)
        return x + layer.self_attn.o_proj(o)

    def _mlp_block(self, x, l):
        layer = self.hf.model.layers[l]
        h = layer.post_attention_layernorm(x)
        gateup = h @ self.w_gateup[l].T
        half = gateup.shape[-1] // 2
        gate = F.silu(gateup[..., :half])
        up = gateup[..., half:]
        return x + layer.mlp.down_proj(gate * up)

    # -- streaming (single sequence) ---------------------------------------

    def prefill(self, input_ids, table):
        input_ids = input_ids.to(self.device)
        t = input_ids.shape[0]
        x = self.hf.model.embed_tokens(input_ids)
        for l in range(self.n_layers):
            x = self._attn_block(x, l, table, 0, t, num_tokens=t)
            x = self._mlp_block(x, l)
        table.advance(t)
        return self.hf.lm_head(self.hf.model.norm(x))

    def decode(self, token_id, table):
        token_id = token_id.to(self.device)
        pos = table.num_tokens
        x = self.hf.model.embed_tokens(token_id)
        for l in range(self.n_layers):
            x = self._attn_block(x, l, table, pos, 1, num_tokens=pos + 1)
            x = self._mlp_block(x, l)
        table.advance(1)
        return self.hf.lm_head(self.hf.model.norm(x))

    # -- batched (padded) eager inference ----------------------------------

    def prefill_batch(self, input_ids_list, tables):
        b = len(input_ids_list)
        lens = [x.shape[0] for x in input_ids_list]
        max_len = max(lens)
        padded = torch.zeros(b, max_len, dtype=torch.long, device=self.device)
        for i, x in enumerate(input_ids_list):
            padded[i, :lens[i]] = x.to(self.device)
        x = self.hf.model.embed_tokens(padded)
        x = self._run_layers_batch(x, tables, max_len, lens, [0] * b)
        for i, table in enumerate(tables):
            table.advance(lens[i])
        logits = self.hf.lm_head(self.hf.model.norm(x))
        return [logits[i, lens[i] - 1] for i in range(b)]

    def decode_batch(self, token_ids, tables):
        b = len(token_ids)
        tokens = torch.stack(token_ids).to(self.device).view(b)
        positions = torch.tensor([t.num_tokens for t in tables],
                                 device=self.device)
        x = self.hf.model.embed_tokens(tokens.view(b, 1))
        lens = [t.num_tokens + 1 for t in tables]
        q_starts = [l - 1 for l in lens]
        # rotary needs positions of the query tokens
        self._batch_rot_pos = positions
        x = self._run_layers_batch(x, tables, 1, lens, q_starts)
        for table in tables:
            table.advance(1)
        return [row[0] for row in self.hf.lm_head(self.hf.model.norm(x))]

    def _run_layers_batch(self, x, tables, t, lens, q_starts):
        b, _, _ = x.shape
        device = self.device
        for l, layer in enumerate(self.hf.model.layers):
            ln = layer.input_layernorm(x)
            qkv = ln @ self.w_qkv[l].T + self.b_qkv[l]
            q = qkv[:, :, :self.n_heads * self.head_dim].view(
                b, t, self.n_heads, self.head_dim)
            k = qkv[:, :, self.n_heads * self.head_dim:
                    self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim
                    ].view(b, t, self.n_kv_heads, self.head_dim)
            v = qkv[:, :, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim:
                    ].view(b, t, self.n_kv_heads, self.head_dim)
            # rotary: per-row position = q_start + j
            positions = torch.arange(t, device=device)
            positions = positions.unsqueeze(0) + \
                torch.tensor(q_starts, device=device).unsqueeze(1)
            cos, sin = self.hf.model.rotary_emb(q, positions)
            q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2),
                                        cos, sin, unsqueeze_dim=1)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            pool = tables[0].pool
            if t == 1:
                for i in range(b):
                    tbl = tables[i]
                    if tbl.num_tokens >= len(tbl.blocks) * tbl.block_size:
                        tbl.blocks.append(pool.allocate())
                positions_t = torch.tensor(
                    [tbl.num_tokens for tbl in tables], device=device)
                block_ids = torch.tensor(
                    [tbl.blocks[p // pool.block_size]
                     for tbl, p in zip(tables, positions_t.tolist())],
                    device=device)
                offsets = positions_t % pool.block_size
                pool.cache[0, l, block_ids, offsets] = k[:, 0]
                pool.cache[1, l, block_ids, offsets] = v[:, 0]
            else:
                for i in range(b):
                    tables[i].append(l, k[i].reshape(-1, self.n_kv_heads,
                                                     self.head_dim),
                                     v[i].reshape(-1, self.n_kv_heads,
                                                  self.head_dim))
            maxkv = max(lens)
            nb = max(len(t.blocks) for t in tables)
            bt = torch.zeros(b, nb, dtype=torch.long, device=device)
            for i in range(b):
                bt[i, :len(tables[i].blocks)] = torch.tensor(
                    tables[i].blocks, device=device)
            flat = bt.flatten()
            kk = pool.cache[0, l].index_select(0, flat)
            vv = pool.cache[1, l].index_select(0, flat)
            kk = kk.view(b, nb * pool.block_size, self.n_kv_heads,
                         self.head_dim)[:, :maxkv]
            vv = vv.view(b, nb * pool.block_size, self.n_kv_heads,
                         self.head_dim)[:, :maxkv]
            s_idx = torch.arange(maxkv, device=device)
            q_idx = torch.arange(t, device=device)
            lens_t = torch.tensor(lens, device=device)
            starts = torch.tensor(q_starts, device=device)
            beyond_len = s_idx[None, :] >= lens_t[:, None]
            future = (starts[:, None, None] + q_idx[None, :, None]) < \
                s_idx[None, None, :]
            mask = beyond_len[:, None, :] | future
            o = batched_attention(q, kk, vv, mask)
            o = o.reshape(b, t, self.n_heads * self.head_dim)
            x = x + layer.self_attn.o_proj(o)
            h = layer.post_attention_layernorm(x)
            gateup = h @ self.w_gateup[l].T
            half = gateup.shape[-1] // 2
            x = x + layer.mlp.down_proj(F.silu(gateup[..., :half]) *
                                        gateup[..., half:])
        return x

    # -- CUDA graph decode ---------------------------------------------------

    def capture_decode_graph(self, tables, nb_max):
        b = len(tables)
        bs = tables[0].pool.block_size
        maxkv = nb_max * bs
        buf = {
            "pool": tables[0].pool,
            "tokens": torch.zeros(b, 1, dtype=torch.long, device=self.device),
            "positions": torch.zeros(b, 1, dtype=torch.long,
                                     device=self.device),
            "block_ids": torch.zeros(b, nb_max, dtype=torch.long,
                                     device=self.device),
            "write_blocks": torch.zeros(b, dtype=torch.long,
                                        device=self.device),
            "write_offsets": torch.zeros(b, dtype=torch.long,
                                         device=self.device),
            "mask": torch.zeros(b, 1, 1, maxkv, dtype=self.dtype,
                                device=self.device),
            "logits": None,
            "graph": None,
        }
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
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
        from torch.nn.functional import scaled_dot_product_attention
        b = buf["tokens"].shape[0]
        t = 1
        pool = buf["pool"]
        bs = pool.block_size
        nb_max = buf["block_ids"].shape[1]
        x = self.hf.model.embed_tokens(buf["tokens"])
        flat = buf["block_ids"].flatten()
        for l, layer in enumerate(self.hf.model.layers):
            ln = layer.input_layernorm(x)
            qkv = ln @ self.w_qkv[l].T + self.b_qkv[l]
            q = qkv[:, :, :self.n_heads * self.head_dim].view(
                b, 1, self.n_heads, self.head_dim)
            k = qkv[:, :, self.n_heads * self.head_dim:
                    self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim
                    ].view(b, 1, self.n_kv_heads, self.head_dim)
            v = qkv[:, :, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim:
                    ].view(b, 1, self.n_kv_heads, self.head_dim)
            cos, sin = self.hf.model.rotary_emb(q, buf["positions"])
            q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2),
                                        cos, sin, unsqueeze_dim=1)
            k = k.transpose(1, 2)          # (b, 1, kv, D) for the scatter
            pool.cache[0, l, buf["write_blocks"], buf["write_offsets"]] = k[:, 0]
            pool.cache[1, l, buf["write_blocks"], buf["write_offsets"]] = v[:, 0]
            kk = pool.cache[0, l].index_select(0, flat).view(
                b, nb_max * bs, self.n_kv_heads, self.head_dim).transpose(1, 2)
            vv = pool.cache[1, l].index_select(0, flat).view(
                b, nb_max * bs, self.n_kv_heads, self.head_dim).transpose(1, 2)
            # GQA: expand KV heads to match query heads for SDPA
            if self.n_kv_heads != self.n_heads:
                repeat = self.n_heads // self.n_kv_heads
                h_idx = torch.arange(self.n_heads, device=self.device) // repeat
                kk = kk[:, h_idx]
                vv = vv[:, h_idx]
            o = scaled_dot_product_attention(q, kk, vv, attn_mask=buf["mask"])
            o = o.transpose(1, 2).reshape(b, 1, self.n_heads * self.head_dim)
            x = x + layer.self_attn.o_proj(o)
            h = layer.post_attention_layernorm(x)
            gateup = h @ self.w_gateup[l].T
            half = gateup.shape[-1] // 2
            x = x + layer.mlp.down_proj(F.silu(gateup[..., :half]) *
                                        gateup[..., half:])
        return self.hf.lm_head(self.hf.model.norm(x))

    # -- CUDA graph prefill -------------------------------------------------

    def capture_prefill_graph(self, tables, bucket_len):
        b = len(tables)
        bs = tables[0].pool.block_size
        nb_max = bucket_len // bs + 1
        maxkv = nb_max * bs
        q_idx = torch.arange(bucket_len, device=self.device)
        s_idx = torch.arange(maxkv, device=self.device)
        causal = (q_idx[:, None] < s_idx[None, :])
        buf = {
            "pool": tables[0].pool,
            "input_ids": torch.zeros(b, bucket_len, dtype=torch.long,
                                     device=self.device),
            "beyond": torch.zeros(b, 1, maxkv, dtype=self.dtype,
                                  device=self.device),
            "causal": torch.where(causal, torch.tensor(float("-inf"), dtype=self.dtype, device=self.device), torch.zeros((), dtype=self.dtype, device=self.device)),
            "positions": torch.zeros(b, bucket_len, dtype=torch.long,
                                     device=self.device),
            "block_ids": torch.zeros(b, nb_max, dtype=torch.long,
                                     device=self.device),
            "write_blocks": torch.zeros(b, bucket_len, dtype=torch.long,
                                        device=self.device),
            "write_offsets": torch.zeros(b, bucket_len, dtype=torch.long,
                                         device=self.device),
            "logits": None,
            "graph": None,
        }
        # scratch block: padded (past-the-prompt) positions write here so the
        # capture/replay scatter never touches real request KV
        buf["scratch"] = tables[0].pool.allocate()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        pool = buf["pool"]
        snapshot = pool.cache[:, :, 0, 0].clone()
        with torch.cuda.stream(s):
            for _ in range(3):
                self._graph_prefill_forward(buf)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            buf["logits"] = self._graph_prefill_forward(buf)
        pool.cache[:, :, 0, 0] = snapshot
        buf["graph"] = g
        return buf

    def replay_prefill_graph(self, buf, tables, input_ids_list):
        b = len(tables)
        bucket_len = buf["input_ids"].shape[1]
        bs = tables[0].pool.block_size
        lens = [x.shape[0] for x in input_ids_list]
        buf["input_ids"].zero_()
        buf["beyond"].fill_(0.0)
        buf["write_blocks"].zero_()
        buf["write_offsets"].zero_()
        buf["block_ids"].zero_()
        for i, (t, ids) in enumerate(zip(tables, input_ids_list)):
            buf["input_ids"][i, :lens[i]] = ids.to(self.device)
            buf["beyond"][i, :, lens[i]:] = float("-inf")
            nb = (lens[i] + bs - 1) // bs
            while len(t.blocks) < nb:
                t.blocks.append(t.pool.allocate())
            pos = torch.arange(lens[i], device=self.device)
            blocks = torch.tensor([t.blocks[p // bs] for p in range(lens[i])],
                                  device=self.device)
            buf["write_blocks"][i, :lens[i]] = blocks
            buf["write_offsets"][i, :lens[i]] = pos % bs
            if lens[i] < bucket_len:
                buf["write_blocks"][i, lens[i]:] = buf["scratch"]
            buf["block_ids"][i, :len(t.blocks)] = torch.tensor(
                t.blocks, device=self.device)
        buf["graph"].replay()
        logits = buf["logits"]
        return [logits[i, lens[i] - 1] for i in range(b)]

    def _graph_prefill_forward(self, buf):
        from torch.nn.functional import scaled_dot_product_attention
        b, L = buf["input_ids"].shape
        pool = buf["pool"]
        bs = pool.block_size
        nb_max = buf["block_ids"].shape[1]
        x = self.hf.model.embed_tokens(buf["input_ids"])
        flat = buf["block_ids"].flatten()
        for l, layer in enumerate(self.hf.model.layers):
            ln = layer.input_layernorm(x)
            qkv = ln @ self.w_qkv[l].T + self.b_qkv[l]
            q = qkv[:, :, :self.n_heads * self.head_dim].view(
                b, L, self.n_heads, self.head_dim)
            k = qkv[:, :, self.n_heads * self.head_dim:
                    self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim
                    ].view(b, L, self.n_kv_heads, self.head_dim)
            v = qkv[:, :, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim:
                    ].view(b, L, self.n_kv_heads, self.head_dim)
            positions = torch.arange(L, device=self.device)
            positions = positions.unsqueeze(0).expand(b, L)
            cos, sin = self.hf.model.rotary_emb(q, positions)
            q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2),
                                        cos, sin, unsqueeze_dim=1)
            k = k.transpose(1, 2)          # (b, L, kv, D) for the scatter
            pool.cache[0, l, buf["write_blocks"].flatten(),
                       buf["write_offsets"].flatten()] = k.reshape(
                           -1, self.n_kv_heads, self.head_dim)
            pool.cache[1, l, buf["write_blocks"].flatten(),
                       buf["write_offsets"].flatten()] = v.reshape(
                           -1, self.n_kv_heads, self.head_dim)
            kk = pool.cache[0, l].index_select(0, flat).view(
                b, nb_max * bs, self.n_kv_heads, self.head_dim).transpose(1, 2)
            vv = pool.cache[1, l].index_select(0, flat).view(
                b, nb_max * bs, self.n_kv_heads, self.head_dim).transpose(1, 2)
            # GQA: expand KV heads to match query heads for SDPA
            if self.n_kv_heads != self.n_heads:
                repeat = self.n_heads // self.n_kv_heads
                h_idx = torch.arange(self.n_heads, device=self.device) // repeat
                kk = kk[:, h_idx]
                vv = vv[:, h_idx]
            mask = buf["beyond"] + buf["causal"].unsqueeze(0)
            mask = mask.unsqueeze(1)
            o = scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            o = o.transpose(1, 2).reshape(b, L, self.n_heads * self.head_dim)
            x = x + layer.self_attn.o_proj(o)
            h = layer.post_attention_layernorm(x)
            gateup = h @ self.w_gateup[l].T
            half = gateup.shape[-1] // 2
            x = x + layer.mlp.down_proj(F.silu(gateup[..., :half]) *
                                        gateup[..., half:])
        return self.hf.lm_head(self.hf.model.norm(x))
