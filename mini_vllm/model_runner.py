"""A tiny transformer used to exercise the paged inference path.

The model exposes two complementary forward modes:

- ``prefill`` / ``decode``: streaming modes that write K/V into a
  ``BlockTable`` layer by layer and read them back with ``paged_attention``.
- ``dense_forward``: a full forward over all tokens at once (recomputing
  attention from scratch), used as the ground truth in tests.

No ``transformers`` dependency; the engine's model interface is exactly
``prefill(input_ids, table) -> logits`` and
``decode(token_id, table) -> logits``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paged_attention import batched_attention, dense_attention, paged_attention


class _Layer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)

    def split_heads(self, x):
        if x.dim() == 3:  # (B, T, D)
            b, t, _ = x.shape
            return x.view(b, t, self.n_heads, self.head_dim)
        t = x.shape[0]
        return x.view(t, self.n_heads, self.head_dim)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                 max_positions=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_positions, d_model)
        self.layers = nn.ModuleList(
            [_Layer(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # -- streaming (paged) inference --------------------------------------

    def prefill(self, input_ids, table):
        """Process a full prompt, store its K/V, return logits per position."""
        input_ids = input_ids.to(self.embed.weight.device)
        t = input_ids.shape[0]
        x = self.embed(input_ids) + self.pos(
            torch.arange(t, device=input_ids.device))
        for l, layer in enumerate(self.layers):
            x = self._attn_layer(x, layer, table, l)
            x = x + layer.w2(F.gelu(layer.w1(layer.ln2(x))))
        table.advance(t)
        return self.lm_head(self.ln_f(x))

    def decode(self, token_id, table):
        """Process one new token against stored K/V, return logits (1, V)."""
        token_id = token_id.to(self.embed.weight.device)
        pos = table.num_tokens
        x = self.embed(token_id) + self.pos(
            torch.tensor([pos], device=token_id.device))
        for l, layer in enumerate(self.layers):
            x = self._attn_layer(x, layer, table, l)
            x = x + layer.w2(F.gelu(layer.w1(layer.ln2(x))))
        table.advance(1)
        return self.lm_head(self.ln_f(x))

    def _attn_layer(self, x, layer, table, l):
        t = layer.ln1(x)
        k = layer.split_heads(layer.wk(t))
        v = layer.split_heads(layer.wv(t))
        q = layer.split_heads(layer.wq(t))
        table.append(l, k, v)
        o = paged_attention(q, table, layer=l, causal=True)
        return x + layer.wo(o.reshape(x.shape))

    # -- batched (padded) inference ---------------------------------------

    def prefill_batch(self, input_ids_list, tables):
        """Prefill a batch of prompts in one forward (padded to max length).

        Args:
            input_ids_list: list of (T_i,) tensors.
            tables: matching list of BlockTable.

        Returns:
            list of (V,) logits for the last token of each prompt.
        """
        b = len(input_ids_list)
        lens = [x.shape[0] for x in input_ids_list]
        max_len = max(lens)
        device = self.embed.weight.device
        padded = torch.zeros(b, max_len, dtype=torch.long, device=device)
        for i, x in enumerate(input_ids_list):
            padded[i, :lens[i]] = x.to(device)
        x = self.embed(padded) + self.pos(
            torch.arange(max_len, device=device))
        x = self._run_layers_batch(x, tables, max_len, lens,
                                   query_starts=[0] * b)
        for i, table in enumerate(tables):
            table.advance(lens[i])
        logits = self.lm_head(self.ln_f(x))  # (B, T, V)
        return [logits[i, lens[i] - 1] for i in range(b)]

    def decode_batch(self, token_ids, tables):
        """Decode one new token for a batch of sequences in one forward."""
        b = len(token_ids)
        device = self.embed.weight.device
        tokens = torch.stack(token_ids).to(device).view(b)
        positions = torch.tensor([t.num_tokens for t in tables],
                                 device=device)
        x = self.embed(tokens.view(b, 1)) + self.pos(positions).unsqueeze(1)
        lens = [t.num_tokens + 1 for t in tables]
        x = self._run_layers_batch(x, tables, 1, lens,
                                   query_starts=[l - 1 for l in lens])
        for table in tables:
            table.advance(1)
        return [row[0] for row in self.lm_head(self.ln_f(x))]

    def _run_layers_batch(self, x, tables, t, lens, query_starts):
        """Shared batched layer stack; writes K/V per request, gathers the
        padded K/V back, and runs one masked attention per layer."""
        b, _, _ = x.shape
        device = x.device
        for l, layer in enumerate(self.layers):
            ln = layer.ln1(x)
            k = layer.split_heads(layer.wk(ln))   # (B, T, H, D)
            v = layer.split_heads(layer.wv(ln))
            q = layer.split_heads(layer.wq(ln))
            for i in range(b):
                tables[i].append(l, k[i].reshape(-1, k.shape[-2], k.shape[-1]),
                                    v[i].reshape(-1, v.shape[-2], v.shape[-1]))
            maxkv = max(lens)
            pool = tables[0].pool
            nb = max(len(t.blocks) for t in tables)
            bt = torch.zeros(b, nb, dtype=torch.long, device=device)
            for i in range(b):
                bt[i, :len(tables[i].blocks)] = torch.tensor(
                    tables[i].blocks, device=device)
            flat = bt.flatten()
            kk = pool.cache[0, l].index_select(0, flat)
            vv = pool.cache[1, l].index_select(0, flat)
            kk = kk.view(b, nb * pool.block_size, k.shape[-2], k.shape[-1])[:, :maxkv]
            vv = vv.view(b, nb * pool.block_size, v.shape[-2], v.shape[-1])[:, :maxkv]
            # mask: hide (a) keys beyond a row's real length, (b) future keys
            s_idx = torch.arange(maxkv, device=device)
            q_idx = torch.arange(t, device=device)
            lens_t = torch.tensor(lens, device=device)
            starts = torch.tensor(query_starts, device=device)
            beyond_len = s_idx[None, :] >= lens_t[:, None]        # (B, S)
            future = (starts[:, None, None] + q_idx[None, :, None]) < \
                s_idx[None, None, :]                              # (B, Q, S)
            mask = beyond_len[:, None, :] | future                # (B, Q, S)
            o = batched_attention(q, kk, vv, mask)
            o = o.reshape(b, t, self.d_model)
            x = x + layer.wo(o)
            x = x + layer.w2(F.gelu(layer.w1(layer.ln2(x))))
        return x

    # -- dense ground truth ------------------------------------------------

    def dense_forward(self, input_ids):
        """Full forward over all tokens; recomputes attention each call.

        Accepts a single sequence (T,) or a batch (B, T); returns logits of
        the same leading shape.
        """
        input_ids = input_ids.to(self.embed.weight.device)
        batched = input_ids.dim() > 1
        if not batched:
            input_ids = input_ids.unsqueeze(0)
        b, t = input_ids.shape
        x = self.embed(input_ids) + self.pos(
            torch.arange(t, device=input_ids.device).expand(b, t))
        for l, layer in enumerate(self.layers):
            ln = layer.ln1(x)
            k = layer.split_heads(layer.wk(ln))
            v = layer.split_heads(layer.wv(ln))
            q = layer.split_heads(layer.wq(ln))
            o = dense_attention(q, k, v, causal=True)
            x = x + layer.wo(o.reshape(b, t, self.d_model))
            x = x + layer.w2(F.gelu(layer.w1(layer.ln2(x))))
        logits = self.lm_head(self.ln_f(x))
        return logits if batched else logits.squeeze(0)
