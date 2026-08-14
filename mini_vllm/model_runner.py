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

from .paged_attention import dense_attention, paged_attention


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
