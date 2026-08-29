"""Transformers-model adapter for the mini-vllm engine.

The adapter supports ordinary decoder-only models (the existing Qwen3 path)
and Qwen3.5's hybrid language model.  Qwen3.5 is special: its cache contains
both full-attention KV tensors and Gated DeltaNet recurrent state, so it must
use Transformers' native hybrid cache instead of mini-vllm's uniform paged-KV
layout.  The engine still owns request scheduling; this adapter owns the
model-specific incremental state.

Correctness-first design:
- ordinary models: preserve the original hook-based compatibility path.
- Qwen3.5: prefill once, then pass the native HF cache back for one-token
  decode.  This preserves GDN state and avoids quadratic full-history reruns.

The adapter exposes the attribute surface the engine reads (n_layers,
n_heads, n_kv_heads, head_dim, vocab_size) so KVBlockManager sizing works.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class _Qwen35MTP(nn.Module):
    """The one-layer Qwen3.5 next-token predictor.

    Transformers intentionally omits ``mtp.*`` from the model class.  This
    small module mirrors the checkpoint layout and reuses the target model's
    embedding, rotary embedding, and LM head.
    """

    def __init__(self, text_model, lm_head, config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
        )

        self.embed_tokens = text_model.embed_tokens
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        mtp_config = copy.deepcopy(config)
        mtp_config.layer_types = ["full_attention"]
        mtp_config.num_hidden_layers = 1
        self.layer = Qwen3_5DecoderLayer(mtp_config, layer_idx=0)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(self, token_id, hidden, position):
        # MTP predicts the token after ``token_id`` using the target hidden
        # state at that same position.
        embedding = self.pre_fc_norm_embedding(self.embed_tokens(token_id))
        hidden = self.pre_fc_norm_hidden(hidden)
        mixed = self.fc(torch.cat((embedding, hidden), dim=-1))
        position_embeddings = self.rotary_emb(mixed, position)
        mixed = self.layer(
            mixed,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position,
        )
        mixed = self.norm(mixed)
        return self.lm_head(mixed), mixed

    def load_mtp_state_dict(self, state):
        translated = {}
        for name, value in state.items():
            if name.startswith("mtp."):
                name = name[4:]
            name = name.replace("layers.0.", "layer.")
            if name in self.state_dict():
                translated[name] = value
        missing, unexpected = self.load_state_dict(translated, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected Qwen3.5 MTP weights: {unexpected}")
        required = {
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
            "fc.weight",
            "layer.input_layernorm.weight",
            "layer.post_attention_layernorm.weight",
            "layer.self_attn.q_proj.weight",
            "layer.self_attn.k_proj.weight",
            "layer.self_attn.v_proj.weight",
            "layer.self_attn.o_proj.weight",
            "layer.self_attn.q_norm.weight",
            "layer.self_attn.k_norm.weight",
            "layer.mlp.gate_proj.weight",
            "layer.mlp.up_proj.weight",
            "layer.mlp.down_proj.weight",
            "norm.weight",
        }
        missing = sorted(required.intersection(missing))
        if missing:
            raise RuntimeError(f"missing Qwen3.5 MTP weights: {missing}")

    @staticmethod
    def load_checkpoint_weights(model_path):
        from safetensors import safe_open

        root = Path(model_path)
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(str(model_path)))
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            import json

            names = sorted(
                set(json.loads(index_path.read_text())["weight_map"].values())
            )
        else:
            names = sorted(p.name for p in root.glob("*.safetensors"))
        state = {}
        for name in names:
            with safe_open(str(root / name), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key.startswith("mtp."):
                        state[key] = shard.get_tensor(key)
        if not state:
            raise RuntimeError(f"no mtp.* weights found in {root}")
        return state


class TransformersAdapter:
    def __init__(self, model: Any):
        self.model = model
        cfg = getattr(model, "config", None)
        # Qwen3.5 wraps the text config inside a vision-language config.
        self.text_config = getattr(cfg, "text_config", cfg)
        model_type = str(getattr(self.text_config, "model_type", ""))
        self.is_qwen35 = model_type in {"qwen3_5", "qwen3_5_text"}
        if cfg is not None:
            self.n_layers = getattr(self.text_config, "num_hidden_layers", None)
            self.n_heads = getattr(self.text_config, "num_attention_heads", None)
            self.n_kv_heads = getattr(self.text_config, "num_key_value_heads", None) or self.n_heads
            self.head_dim = getattr(self.text_config, "head_dim", None) or (
                getattr(self.text_config, "hidden_size", None) // self.n_heads
            )
            self.vocab_size = getattr(self.text_config, "vocab_size", None)
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
        root = getattr(model, "model", None)
        self._text_model = getattr(root, "language_model", None) or root
        self._lm_head = getattr(model, "lm_head", None)
        self.layer_types = list(getattr(self.text_config, "layer_types", []))
        self.mtp_num_hidden_layers = int(
            getattr(self.text_config, "mtp_num_hidden_layers", 0) or 0
        )
        # MTP is opt-in because loading its separate checkpoint weights is
        # expensive.  ``enable_mtp`` turns on the speculative path.
        self.supports_mtp = False
        self.mtp = None
        self._layer_kvs: list[Any] = [None] * self.n_layers
        self._hooks: list[Any] = []
        self._native_caches: dict[int, dict[str, Any]] = {}
        if not self.is_qwen35:
            self._register_hooks()

    def _register_hooks(self) -> None:
        # transformers models expose self_attn per layer; mini-megatron's
        # DecoderLayer uses its own attention module.
        layers = self._layers()
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            hook = attn.register_forward_hook(
                lambda module, args, output, i=i: self._capture(i, output)
            )
            self._hooks.append(hook)

    def _layers(self):
        """Find decoder layers across HF and mini-megatron wrappers."""
        root = getattr(self.model, "model", None)
        language_model = getattr(root, "language_model", None)
        for candidate in (language_model, root, getattr(self.model, "decoder", None)):
            layers = getattr(candidate, "layers", None)
            if layers is not None:
                return layers
        return []

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
        if self.mtp is not None:
            self.mtp.eval()

    def enable_mtp(self, model_path=None, state_dict=None):
        """Load Qwen3.5 ``mtp.*`` weights and enable speculative decoding."""
        if not self.is_qwen35:
            raise ValueError("MTP is only implemented for Qwen3.5")
        if self.mtp_num_hidden_layers < 1:
            raise ValueError("checkpoint has no MTP layers")
        if self.mtp_num_hidden_layers != 1:
            raise NotImplementedError(
                "mini-vllm currently supports one Qwen3.5 MTP layer"
            )
        if model_path is not None and state_dict is not None:
            raise ValueError("pass either model_path or state_dict, not both")
        self.mtp = _Qwen35MTP(self._text_model, self._lm_head, self.text_config)
        if model_path is not None:
            state_dict = _Qwen35MTP.load_checkpoint_weights(model_path)
        if state_dict is None:
            raise ValueError("MTP weights are required")
        self.mtp.load_mtp_state_dict(state_dict)
        self.mtp.to(device=self.device, dtype=self.dtype)
        self.mtp.eval()
        self.supports_mtp = True

    def reset_cache(self, table: Any) -> None:
        """Drop model-native state when the engine preempts or finishes."""
        self._native_caches.pop(id(table), None)

    def _new_native_cache(self):
        """Construct the cache supported by the installed Transformers build.

        Qwen3.5's model code accepts ``DynamicCache(config=...)`` in current
        releases.  The no-argument fallback keeps this adapter compatible with
        older development snapshots whose cache constructor inferred config.
        """
        from transformers.cache_utils import DynamicCache

        try:
            return DynamicCache(config=self.text_config)
        except TypeError:
            return DynamicCache()

    @staticmethod
    def _output_logits_and_cache(out):
        if isinstance(out, tuple):
            logits = out[0]
            cache = out[1] if len(out) > 1 else None
        else:
            logits = getattr(out, "logits", out)
            cache = getattr(out, "past_key_values", None)
        return logits, cache

    def _native_forward(self, ids: torch.Tensor, cache=None, start=0,
                        return_hidden=False):
        """Run Qwen3.5 through HF's hybrid KV/GDN cache API."""
        kwargs = {
            "input_ids": ids.unsqueeze(0),
            "attention_mask": torch.ones(
                (1, start + ids.numel()), dtype=torch.long, device=ids.device
            ),
            "use_cache": True,
            "return_dict": True,
            "cache_position": torch.arange(
                start, start + ids.numel(), device=ids.device
            ),
        }
        if cache is not None:
            kwargs["past_key_values"] = cache
        # Calling the text submodel directly exposes the final hidden state,
        # which is required by Qwen3.5 MTP.  The public wrapper is retained as
        # a fallback for older Transformers snapshots.
        target = self._text_model if return_hidden else self.model
        with torch.no_grad():
            try:
                out = target(**kwargs)
            except TypeError:
                kwargs.pop("return_dict", None)
                kwargs.pop("cache_position", None)
                out = target(**kwargs)
        if return_hidden:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None and isinstance(out, tuple):
                hidden = out[0]
            logits = self._lm_head(hidden)
            cache = getattr(out, "past_key_values", None)
            return logits, cache, hidden
        return self._output_logits_and_cache(out)

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
        if self.is_qwen35:
            result = self._native_forward(
                toks, cache=None, start=0, return_hidden=self.supports_mtp
            )
            logits, cache = result[:2]
            state = {
                "cache": cache,
                "ids": toks.detach().clone(),
            }
            if self.supports_mtp:
                state["hidden"] = result[2][:, -1:].detach()
                state["logits"] = logits[:, -1].detach()
            self._native_caches[id(table)] = state
            # Keep the table cursor coherent for scheduler/debugging.  The
            # actual Qwen3.5 state lives in the native hybrid cache above.
            table.advance(toks.shape[0])
            return logits[0, -1]
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
        if self.is_qwen35:
            state = self._native_caches.get(id(table))
            # Engine passes full history for compatibility.  Feed only the
            # suffix when it matches the cache prefix; rebuild after a
            # preemption or any caller-side history edit.
            if state is not None:
                prefix = state["ids"]
                if ids.numel() == prefix.numel() + 1 and torch.equal(ids[:-1], prefix):
                    result = self._native_forward(
                        ids[-1:], state["cache"], start=prefix.numel(),
                        return_hidden=self.supports_mtp,
                    )
                    logits, cache = result[:2]
                    state["cache"] = cache
                    state["ids"] = ids.detach().clone()
                    if self.supports_mtp:
                        state["hidden"] = result[2][:, -1:].detach()
                        state["logits"] = logits[:, -1].detach()
                    table.advance(1)
                    return logits[0, -1]
            cache = self._new_native_cache()
            result = self._native_forward(
                ids, cache=cache, start=0, return_hidden=self.supports_mtp
            )
            logits, cache = result[:2]
            state = {
                "cache": cache,
                "ids": ids.detach().clone(),
            }
            if self.supports_mtp:
                state["hidden"] = result[2][:, -1:].detach()
                state["logits"] = logits[:, -1].detach()
            self._native_caches[id(table)] = state
            table.num_tokens = ids.numel()
            return logits[0, -1]
        logits = self._forward_logits(ids)
        return logits[-1]  # [V]

    def speculative_decode(self, history: torch.Tensor, table: Any,
                           max_tokens: int = 2) -> list[int]:
        """Draft with Qwen3.5 MTP and verify in one target-model pass.

        This deliberately omits the optional bonus token: accepting ``N``
        drafts returns those ``N`` tokens and leaves the target logits for the
        next step in the cache state.  On rejection, a full hybrid-cache
        snapshot is restored before the replacement token is committed.
        """
        if not self.supports_mtp or self.mtp is None:
            raise RuntimeError("Qwen3.5 MTP is not enabled")
        if max_tokens <= 0:
            return []
        ids = history.to(self.device)
        state = self._native_caches.get(id(table))
        if state is None or not torch.equal(ids, state["ids"]):
            self.decode_with_history(ids, table)
            state = self._native_caches[id(table)]
        base = ids.numel()

        # Run the small MTP chain.  Each layer consumes the previous MTP
        # hidden state and the embedding of the token it just predicted.
        current = int(ids[-1])
        hidden = state["hidden"]
        drafts = []
        for offset in range(max_tokens):
            token = torch.tensor([[current]], dtype=torch.long, device=self.device)
            position = torch.tensor([[base + offset - 1]], dtype=torch.long,
                                    device=self.device)
            draft_logits, hidden = self.mtp(token, hidden, position)
            current = int(torch.argmax(draft_logits[0, -1]))
            drafts.append(current)

        verify = torch.tensor(drafts, dtype=torch.long, device=self.device)
        # DynamicCache.crop cannot rewind GDN recurrent states.  Keep a
        # complete tensor snapshot so a rejected draft restores every hybrid
        # component (conv, recurrent, and full-attention KV).
        rollback_cache = copy.deepcopy(state["cache"])
        logits, cache, target_hidden = self._native_forward(
            verify, state["cache"], start=base, return_hidden=True
        )
        target_logits = state["logits"][0]
        accepted = 0
        replacement = None
        for i, token in enumerate(drafts):
            expected = int(torch.argmax(target_logits))
            if expected != token:
                replacement = expected
                break
            accepted += 1
            target_logits = logits[0, i]

        if accepted == len(drafts):
            state["cache"] = cache
            state["ids"] = torch.cat((ids, verify))
            state["hidden"] = target_hidden[:, -1:]
            state["logits"] = logits[:, -1]
            table.advance(accepted)
            return drafts

        # Discard rejected candidates, then commit the target replacement.
        cache = rollback_cache
        keep_ids = verify[:accepted]
        if replacement is not None:
            replacement_ids = torch.tensor([replacement], dtype=torch.long,
                                           device=self.device)
            result = self._native_forward(
                replacement_ids, cache, start=base + accepted,
                return_hidden=True,
            )
            replacement_logits, cache, replacement_hidden = result
            new_ids = torch.cat((ids, keep_ids, replacement_ids))
            state["hidden"] = replacement_hidden[:, -1:]
            state["logits"] = replacement_logits[:, -1]
            output = drafts[:accepted] + [replacement]
        else:
            # The first draft can only be rejected with a replacement token;
            # this branch is defensive for unusual empty target logits.
            new_ids = torch.cat((ids, keep_ids))
            state["hidden"] = state["hidden"]
            state["logits"] = target_logits.unsqueeze(0)
            output = drafts[:accepted]
        state["cache"] = cache
        state["ids"] = new_ids
        table.advance(len(output))
        return output
