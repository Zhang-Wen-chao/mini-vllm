"""Inference-side tensor parallelism (TP) for TinyTransformer.

Megatron-style sharding across ``world_size`` ranks:

- attention q/k/v projections and the MLP up-projection are COLUMN
  parallel: their output dim (attention heads / intermediate width) is
  split per rank, so each rank computes a disjoint slice of heads;
- attention out-projection and the MLP down-projection are ROW parallel:
  their input dim is split per rank and each ends in ONE all-reduce that
  sums the partial outputs back into the full hidden dim;
- embeddings, layer norms and lm_head are replicated on every rank.

The KV cache shards with the heads: every rank's block pool stores only its
local heads, exactly like vLLM's per-worker paged KV. The engine runs SPMD:
each rank executes the identical schedule and greedy-samples identical
logits, so the control plane needs no communication — only the two
all-reduces per layer per forward pass.

Boundary: MHA only (TinyTransformer has no GQA grouping to split); sampling
beyond greedy would need a synced RNG or a broadcast of the sampled token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .paged_attention import batched_attention, paged_attention
from .model_runner import TinyTransformer


def _all_reduce(x, group=None):
    """Sum ``x`` across ranks; ``group`` scopes the reduction to a
    subgroup (PP×TP reduces within the stage's TP group only)."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, group=group)
    return x


class ColumnParallelLinear(nn.Module):
    """Y = X @ W^T with W's output dim sharded; no communication.

    The bias shards with the same output range as the weight.
    """

    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else \
            nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Y = sum_r(X_r @ W_r^T) + b: input dim sharded, one all-reduce.

    The bias is added AFTER the reduction (replicated, not sharded), the
    Megatron convention — adding a sharded bias before would double-count.
    ``group`` scopes the reduction (PP×TP: within the stage's TP group).
    """

    def __init__(self, weight, bias=None, group=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else \
            nn.Parameter(bias, requires_grad=False)
        self.group = group

    def forward(self, x):
        out = F.linear(x, self.weight)
        out = _all_reduce(out, self.group)
        if self.bias is not None:
            out = out + self.bias
        return out


class _TPLayer(nn.Module):
    """One transformer layer with its linears sharded across ranks.

    ``group`` scopes the row-parallel all-reduces to a subgroup (PP×TP:
    ranks of one pipeline stage reduce among themselves only).
    """

    def __init__(self, dense_layer, world_size, rank, group=None):
        super().__init__()
        self.n_heads = dense_layer.n_heads // world_size
        self.head_dim = dense_layer.head_dim
        hd, lh = self.head_dim, self.n_heads
        d = dense_layer.ln1.weight.shape[0]
        lo, hi = rank * lh * hd, (rank + 1) * lh * hd
        m = dense_layer.w1.weight.shape[0]      # 4 * d_model (full out dim)
        mlo, mhi = rank * (m // world_size), (rank + 1) * (m // world_size)
        self.ln1 = dense_layer.ln1              # replicated
        self.wq = ColumnParallelLinear(dense_layer.wq.weight[lo:hi].clone())
        self.wk = ColumnParallelLinear(dense_layer.wk.weight[lo:hi].clone())
        self.wv = ColumnParallelLinear(dense_layer.wv.weight[lo:hi].clone())
        # wo: (n_heads*hd, d_model), row parallel → shard the input dim
        self.wo = RowParallelLinear(
            dense_layer.wo.weight[:, lo:hi].contiguous().clone(),
            group=group)
        self.ln2 = dense_layer.ln2              # replicated
        self.w1 = ColumnParallelLinear(
            dense_layer.w1.weight[mlo:mhi].clone(),
            dense_layer.w1.bias[mlo:mhi].clone())
        self.w2 = RowParallelLinear(
            dense_layer.w2.weight[:, mlo:mhi].contiguous().clone(),
            dense_layer.w2.bias.clone(), group=group)

    def split_heads(self, x):
        t = x.shape[0] if x.dim() == 2 else x.shape[1]
        if x.dim() == 3:
            return x.view(x.shape[0], t, self.n_heads, self.head_dim)
        return x.view(t, self.n_heads, self.head_dim)

    def mlp(self, x):
        # w1 shard (column) → gelu → w2 shard (row) + all-reduce inside w2
        return self.w2(F.gelu(self.w1(self.ln2(x))))


class TPTransformer(TinyTransformer):
    """Rank-local shard of a TinyTransformer for tensor-parallel inference.

    Built from a full ("dense") model: every rank slices the same dense
    weights, so the union of the shards reproduces the original exactly.
    ``n_heads`` is the LOCAL head count — the engine and the block pool see
    only this rank's slice of the KV cache.
    """

    supports_prefix_cache = True

    def __init__(self, dense: TinyTransformer, world_size: int, rank: int):
        nn.Module.__init__(self)
        if dense.n_heads % world_size != 0:
            raise ValueError("n_heads must be divisible by world_size")
        self.world_size = world_size
        self.rank = rank
        self.vocab_size = dense.vocab_size
        self.d_model = dense.d_model
        self.n_layers = dense.n_layers
        self.n_heads = dense.n_heads // world_size   # local heads
        self.head_dim = dense.head_dim
        self.embed = dense.embed                     # replicated
        self.pos = dense.pos                         # replicated
        self.layers = nn.ModuleList(
            _TPLayer(layer, world_size, rank) for layer in dense.layers)
        self.ln_f = dense.ln_f                       # replicated
        self.lm_head = dense.lm_head                 # replicated

    # -- streaming (paged) inference --------------------------------------

    def prefill(self, input_ids, table):
        input_ids = input_ids.to(self.embed.weight.device)
        t = input_ids.shape[0]
        start = table.num_tokens
        x = self.embed(input_ids) + self.pos(
            torch.arange(start, start + t, device=input_ids.device))
        for l, layer in enumerate(self.layers):
            x = self._attn_layer(x, layer, table, l)
            x = x + layer.mlp(x)
        table.advance(t)
        return self.lm_head(self.ln_f(x))

    def decode(self, token_id, table):
        token_id = token_id.to(self.embed.weight.device)
        pos = table.num_tokens
        x = self.embed(token_id) + self.pos(
            torch.tensor([pos], device=token_id.device))
        for l, layer in enumerate(self.layers):
            x = self._attn_layer(x, layer, table, l)
            x = x + layer.mlp(x)
        table.advance(1)
        return self.lm_head(self.ln_f(x))

    def _attn_layer(self, x, layer, table, l):
        t = layer.ln1(x)
        k = layer.split_heads(layer.wk(t))
        v = layer.split_heads(layer.wv(t))
        q = layer.split_heads(layer.wq(t))
        table.append(l, k, v)
        o = paged_attention(q, table, layer=l, causal=True)
        # local width = local heads * head_dim, NOT d_model
        o = o.reshape(*x.shape[:-1], self.n_heads * self.head_dim)
        return x + layer.wo(o)

    # -- batched (padded) inference ---------------------------------------

    def prefill_batch(self, input_ids_list, tables):
        b = len(input_ids_list)
        lens = [x.shape[0] for x in input_ids_list]
        starts = [t.num_tokens for t in tables]
        kv_lens = [starts[i] + lens[i] for i in range(b)]
        max_len = max(lens)
        device = self.embed.weight.device
        padded = torch.zeros(b, max_len, dtype=torch.long, device=device)
        for i, x in enumerate(input_ids_list):
            padded[i, :lens[i]] = x.to(device)
        pos = torch.arange(max_len, device=device)[None, :] + \
            torch.tensor(starts, device=device)[:, None]
        x = self.embed(padded) + self.pos(pos)
        x = self._run_layers_batch(x, tables, max_len, kv_lens,
                                   query_starts=starts)
        for i, table in enumerate(tables):
            table.advance(lens[i])
        logits = self.lm_head(self.ln_f(x))
        return [logits[i, lens[i] - 1] for i in range(b)]

    def decode_batch(self, token_ids, tables):
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
        b, _, _ = x.shape
        device = x.device
        for l, layer in enumerate(self.layers):
            ln = layer.ln1(x)
            k = layer.split_heads(layer.wk(ln))
            v = layer.split_heads(layer.wv(ln))
            q = layer.split_heads(layer.wq(ln))
            for i in range(b):
                tables[i].append(l, k[i].reshape(-1, k.shape[-2], k.shape[-1]),
                                 v[i].reshape(-1, v.shape[-2], v.shape[-1]))
            maxkv = max(lens)
            pool = tables[0].pool
            nb = max(len(tb.blocks) for tb in tables)
            bt = torch.zeros(b, nb, dtype=torch.long, device=device)
            for i in range(b):
                bt[i, :len(tables[i].blocks)] = torch.tensor(
                    tables[i].blocks, device=device)
            kk = pool.gather_batch(0, l, bt, maxkv)
            vv = pool.gather_batch(1, l, bt, maxkv)
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
            x = x + layer.wo(o)          # row-parallel: all-reduce inside
            x = x + layer.mlp(x)
        return x
