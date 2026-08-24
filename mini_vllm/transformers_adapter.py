"""Transformers-model adapter for the mini-vllm paged-attention engine.

Lets the mini-vllm Engine drive a real Hugging Face causal-LM (Qwen3) instead
of the toy TinyTransformer. The engine keeps its scheduler / KV pool / sampling;
this adapter provides prefill and decode by running the transformers model.

Correctness-first design:
- prefill: full forward, capture per-layer K/V via hooks into the BlockTable
  (the paged cache is populated; future work can make decode read it).
- decode_with_history: re-run the model over the full sequence and return the
  logits of the last position. This recomputes attention (not yet leveraging
  the paged K/V for decode), but the wiring is correct and runs on GPU fast
  enough for short responses.

The adapter exposes the attribute surface the engine reads (n_layers,
n_heads, n_kv_heads, head_dim, vocab_size) so KVBlockManager sizing works.
"""

from __future__ import annotations

from typing import Any

import torch


class TransformersAdapter:
    def __init__(self, model: Any):
        self.model = model
        cfg = getattr(model, "config", None)
        if cfg is not None:
            self.n_layers = getattr(cfg, "num_hidden_layers", None)
            self.n_heads = getattr(cfg, "num_attention_heads", None)
            self.n_kv_heads = getattr(cfg, "num_key_value_heads", None) or self.n_heads
            self.head_dim = getattr(cfg, "head_dim", None) or (
                getattr(cfg, "hidden_size", None) // self.n_heads
            )
            self.vocab_size = getattr(cfg, "vocab_size", None)
        else:
            # mini-megatron GPT has no config; infer from structure.
            decoder = getattr(model, "decoder", None)
            layers = getattr(decoder, "layers", None) if decoder is not None else None
            self.n_layers = len(layers) if layers else 2
            first = layers[0] if layers else None
            self.n_heads = getattr(first, "num_heads", None) or getattr(first, "n_heads", 4)
            self.n_kv_heads = self.n_heads
            self.head_dim = getattr(first, "head_dim", None)
            if self.head_dim is None:
                emb = getattr(model, "embedding", None)
                hidden = getattr(emb, "d_model", None)
                self.head_dim = (hidden or 32) // self.n_heads
            emb = getattr(model, "embedding", None)
            self.vocab_size = getattr(emb, "vocab_size", None) if emb is not None else None
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self._layer_kvs: list[Any] = [None] * self.n_layers
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        # transformers models expose self_attn per layer; mini-megatron's
        # DecoderLayer uses its own attention module.
        for i, layer in enumerate(self.model.model.layers if hasattr(self.model, "model") else self.model.decoder.layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            hook = attn.register_forward_hook(
                lambda module, args, output, i=i: self._capture(i, output)
            )
            self._hooks.append(hook)

    # -- torch module surface (engine reads device/dtype from model) ----

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    @property
    def training(self):
        return self.model.training

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def _capture(self, layer_idx: int, output: Any) -> None:
        # Qwen3 eager attention returns (attn_output, present_key_value) tuple.
        if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], tuple):
            k, v = output[1]
            self._layer_kvs[layer_idx] = (k, v)
        else:
            self._layer_kvs[layer_idx] = None

    def _forward_logits(self, ids: torch.Tensor) -> torch.Tensor:
        self._layer_kvs = [None] * self.n_layers
        with torch.no_grad():
            try:
                out = self.model(
                    input_ids=ids.unsqueeze(0),
                    attention_mask=torch.ones_like(ids).unsqueeze(0),
                    use_cache=False,
                )
            except TypeError:
                # mini-megatron GPT.forward(input_ids, labels, loss_mask) has
                # no attention_mask / use_cache kwargs.
                out = self.model(input_ids=ids.unsqueeze(0))
        if isinstance(out, tuple):
            out = out[0]
        logits = getattr(out, "logits", out)
        return logits[0]  # [T, V]

    # -- mini-vllm engine interface --------------------------------------

    def prefill(self, input_ids: torch.Tensor, table: Any) -> torch.Tensor:
        """Full-prompt forward; store K/V into the table; return the LAST
        position's logits (the engine samples the first generated token from
        it)."""
        toks = input_ids.to(self.device)
        logits = self._forward_logits(toks)
        for layer_idx, kv in enumerate(self._layer_kvs):
            if kv is not None:
                k, v = kv[0]  # [H, T, D]
                table.append(layer_idx, k.transpose(0, 1).to(self.dtype),
                             v.transpose(0, 1).to(self.dtype))
        table.advance(toks.shape[0])
        return logits[-1]  # [V] 1D, so engine argmax/multinomial pick one token

    def decode(self, token_id: torch.Tensor, table: Any) -> torch.Tensor:
        # fallback (should not be used when decode_with_history exists)
        return self.decode_with_history(token_id, table)

    def decode_with_history(self, history: torch.Tensor, table: Any) -> torch.Tensor:
        """Run the full sequence and return the last-position logits (1D [V])."""
        ids = history.to(self.device)
        logits = self._forward_logits(ids)
        return logits[-1]  # [V]
