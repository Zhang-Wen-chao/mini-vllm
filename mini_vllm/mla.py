"""Multi-head Latent Attention (DeepSeek MLA, miniaturized).

The MLA idea: factorize K/V through a small shared LATENT vector so the
cache stores far fewer floats per token. Per layer, a token's cache entry
is one vector ``[c_KV ; k_R]`` of size ``d_latent + d_rope``:

- ``c_KV = W_DKV h``            (d_latent) — the shared latent, cross-head;
- ``k_R  = R_pos·W_KR h``       (d_rope)   — the decoupled RoPE'd key.

``k_R`` is why the decoupling exists: RoPE's rotation depends on the
token's ABSOLUTE position, so it cannot be absorbed into a position-free
linear map the way W_UK can. The cache holds k_R ALREADY ROTATED (vLLM's
MLA backend does the same: rotate at write time, the kernel then reads
position-free vectors) — positions come from the block-table cursor.

Attention per head uses the up-projected key ``k_c = W_UK c_KV`` and
value ``v = W_UV c_KV``. The **absorbed path** never materializes those:
it pushes the up-projections into the query and the output,

    q_nope · k_c = (W_UKᵀ q_nope) · c_KV        (q-side absorption)
    Σ_s p_s v_s  = W_UV (Σ_s p_s c_KV)          (v-side absorption, linear)

so the kernel reads exactly what the cache holds. Scores add the
non-absorbable rope term ``q_R · k_R`` with both sides rotated by their
own positions. The cache therefore holds ``d_latent + d_rope``
floats/token instead of MHA's ``2·n_heads·d_head``.

Weight-shape conventions (``nn.Linear`` weight is ``(out, in)``):
``w_uk``/``w_uv`` map d_latent -> n_heads*head_dim, viewed
``(H, d_h, d_c)`` — ``W[h, d, c]``, output feature d, input feature c.
Every einsum below indexes them as ``hdc`` so both attention paths and
``dense_forward`` share one convention.

Cache layout in the pool: ``kv_kinds=1`` and one "head" of
``head_dim = d_latent + d_rope`` per token (see BlockPool.kinds).

MLA×TP (``MLATransformerTP``): the per-head maps (w_uq/w_qr column
parallel, w_uk/w_uv head-sliced, wo row parallel) shard across ranks, but
the latent projections (w_dkv/w_kr) and the CACHE stay replicated — a
latent vector has no head axis, so unlike dense MHA there is nothing to
shard in the pool; that is precisely MLA's TP story.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_runner import TinyTransformer
from .tp import ColumnParallelLinear, RowParallelLinear


def apply_rope(x, positions, base=10000.0):
    """Rotary-embed ``x``: rotate feature PAIRS by each token's position.

    x: (..., d) with even d; positions: (...,) broadcastable to
    ``x.shape[:-1]``. Pair ``(x[2i], x[2i+1])`` rotates by
    ``pos · base^(-2i/d)``, so an inner product between two rotated
    vectors depends only on their RELATIVE position — the property that
    makes RoPE a positional encoding. Orthogonal rotation: norms are
    preserved, and a rotation at write time commutes with nothing else —
    which is exactly why it cannot be absorbed into W_UK.
    """
    d = x.shape[-1]
    if d % 2:
        raise ValueError("rope needs an even feature dim")
    half = d // 2
    inv_freq = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    ang = positions.float().unsqueeze(-1) * inv_freq      # (..., half)
    cos, sin = ang.cos(), ang.sin()
    pairs = x.float().reshape(*x.shape[:-1], half, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    out = torch.stack([even * cos - odd * sin,
                       even * sin + odd * cos], dim=-1)
    return out.reshape(*x.shape[:-1], d).to(x.dtype)


class _MLALayer(nn.Module):
    def __init__(self, d_model, n_heads, d_latent, d_rope):
        super().__init__()
        if d_rope % 2:
            raise ValueError("d_rope must be even (rope rotates pairs)")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_latent = d_latent
        self.d_rope = d_rope
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.w_dkv = nn.Linear(d_model, d_latent, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)
        self.w_uq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_qr = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.w_uk = nn.Linear(d_latent, n_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(d_latent, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        # FFN identical to the dense layer
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)
        # one temperature for both score terms, as in vLLM's MLA backend
        self.scale = (self.head_dim + d_rope) ** -0.5

    def latent(self, x, positions):
        """The per-token cache vector [c_KV ; k_R], (..., d_latent + d_rope).

        ``k_R`` is stored ROTATED by its absolute position (rotate at
        write time, like vLLM's MLA backend), so the cache holds
        position-final vectors and attention never re-rotates keys.
        """
        t = self.ln1(x)
        return torch.cat([self.w_dkv(t),
                          apply_rope(self.w_kr(t), positions)], dim=-1)

    def _queries(self, x, positions):
        """(q_nope, q_rope) from pre-norm hidden states, each (..., H, d).

        ``q_rope`` is rotated by the QUERY positions; the key side was
        rotated at write time.
        """
        t = x.shape[:-1]
        h = self.n_heads
        normed = self.ln1(x)
        q_nope = self.w_uq(normed).view(*t, h, self.head_dim)
        q_rope = apply_rope(self.w_qr(normed), positions)
        q_rope = q_rope.view(*t, h, self.d_rope)
        return q_nope, q_rope

    def attend_absorbed(self, x, cache, q_start):
        """Absorbed attention over gathered cache rows.

        x: (t, d_model) current-step tokens (pre-norm hidden); cache:
        (S, d_latent + d_rope) stored latent vectors INCLUDING this step's
        rows (append happened before); q_start: absolute position of the
        first query.
        """
        t = x.shape[0]
        h, dh, dc = self.n_heads, self.head_dim, self.d_latent
        c = cache[:, :dc]                       # (S, d_c)
        k_r = cache[:, dc:]                     # (S, d_rope), pre-rotated
        q_pos = torch.arange(q_start, q_start + t, device=x.device)
        q_nope, q_rope = self._queries(x, q_pos)
        # q-side absorption: fold W_UK into the query
        w_uk = self.w_uk.weight.view(h, dh, dc)
        q_abs = torch.einsum("thd,hdc->thc", q_nope, w_uk)
        scores = torch.einsum("thc,sc->ths", q_abs, c) * self.scale \
            + torch.einsum("thr,sr->ths", q_rope, k_r) * self.scale
        s_pos = torch.arange(cache.shape[0], device=x.device)
        scores = scores.masked_fill(
            q_pos[:, None, None] < s_pos[None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        # latent context, then v-side absorption (W_UV after the softmax)
        ctx = torch.einsum("ths,sc->thc", probs, c)
        w_uv = self.w_uv.weight.view(h, dh, dc)
        out = torch.einsum("thc,hdc->thd", ctx, w_uv)
        return self.wo(out.reshape(t, h * dh))

    def attend_explicit(self, x, cache, q_start):
        """Non-absorbed reference: up-project k/v for every cached token.

        Mathematically identical to attend_absorbed; used to verify the
        absorption identities numerically (vLLM's FlashMLA kernel computes
        the absorbed form; the explicit form is what the math describes).
        """
        t = x.shape[0]
        h, dh, dc = self.n_heads, self.head_dim, self.d_latent
        c = cache[:, :dc]
        k_r = cache[:, dc:]
        q_pos = torch.arange(q_start, q_start + t, device=x.device)
        q_nope, q_rope = self._queries(x, q_pos)
        w_uk = self.w_uk.weight.view(h, dh, dc)
        w_uv = self.w_uv.weight.view(h, dh, dc)
        k_c = torch.einsum("sc,hdc->shd", c, w_uk)      # (S, H, d_h)
        v = torch.einsum("sc,hdc->shd", c, w_uv)        # (S, H, d_h)
        scores = torch.einsum("thd,shd->ths", q_nope, k_c) * self.scale \
            + torch.einsum("thr,sr->ths", q_rope, k_r) * self.scale
        q_pos = torch.arange(q_start, q_start + t, device=x.device)
        s_pos = torch.arange(cache.shape[0], device=x.device)
        scores = scores.masked_fill(
            q_pos[:, None, None] < s_pos[None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("ths,shd->thd", probs, v)
        return self.wo(out.reshape(t, h * dh))

    def attend_batch(self, x, cache, kv_lens, query_starts, q_len):
        """Absorbed attention for a padded batch (B, T, d) over (B, S, D).

        Mirrors TinyTransformer._run_layers_batch's mask: keys beyond a
        row's real length and future keys are hidden.
        """
        b = x.shape[0]
        h, dh, dc = self.n_heads, self.head_dim, self.d_latent
        device = x.device
        c = cache[..., :dc]                     # (B, S, d_c)
        k_r = cache[..., dc:]                   # (B, S, d_rope)
        s_idx = torch.arange(cache.shape[1], device=device)
        q_idx = torch.arange(q_len, device=device)
        lens_t = torch.tensor(kv_lens, device=device)
        starts = torch.tensor(query_starts, device=device)
        q_pos = starts[:, None] + q_idx[None, :]          # (B, T)
        q_nope, q_rope = self._queries(x, q_pos)          # (B, T, H, d)
        w_uk = self.w_uk.weight.view(h, dh, dc)
        q_abs = torch.einsum("bthd,hdc->bthc", q_nope, w_uk)
        scores = torch.einsum("bthc,bsc->bths", q_abs, c) * self.scale \
            + torch.einsum("bthr,bsr->bths", q_rope, k_r) * self.scale
        beyond_len = s_idx[None, :] >= lens_t[:, None]          # (B, S)
        future = (starts[:, None, None] + q_idx[None, :, None]) < \
            s_idx[None, None, :]                                # (B, T, S)
        # scores are (B, T, H, S): the mask must broadcast as (B, T, 1, S)
        mask = beyond_len[:, None, None, :] | future[:, :, None, :]
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bths,bsc->bthc", probs, c)
        w_uv = self.w_uv.weight.view(h, dh, dc)
        out = torch.einsum("bthc,hdc->bthd", ctx, w_uv)
        return self.wo(out.reshape(b, q_len, h * dh))

    def mlp(self, x):
        return self.w2(F.gelu(self.w1(self.ln2(x))))


class MLATransformer(TinyTransformer):
    """TinyTransformer whose attention is MLA with a latent paged cache.

    Engine-facing cache geometry: ONE cached vector per token per layer of
    size ``d_latent + d_rope`` — expressed to the block pool as
    ``n_kv_heads=1, head_dim=d_latent+d_rope, kv_kinds=1``. The REAL head
    count/dims live on the layers. ``supports_prefix_cache`` holds: like
    TinyTransformer, positions derive from the block-table cursor.
    """

    supports_prefix_cache = True

    def __init__(self, vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                 d_latent=16, d_rope=4, max_positions=512):
        nn.Module.__init__(self)
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads                  # real attention heads
        # engine-facing cache geometry: one latent vector per token
        self.n_kv_heads = 1
        self.head_dim = d_latent + d_rope
        self.kv_kinds = 1
        self.d_latent = d_latent
        self.d_rope = d_rope
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_positions, d_model)
        self.layers = nn.ModuleList(
            _MLALayer(d_model, n_heads, d_latent, d_rope)
            for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # -- streaming (paged) inference --------------------------------------

    def prefill(self, input_ids, table):
        input_ids = input_ids.to(self.embed.weight.device)
        t = input_ids.shape[0]
        start = table.num_tokens
        x = self.embed(input_ids) + self.pos(
            torch.arange(start, start + t, device=input_ids.device))
        for l, layer in enumerate(self.layers):
            table.append(l, layer.latent(
                x, torch.arange(start, start + t,
                                device=input_ids.device)).unsqueeze(1), None)
            cache = table.gather_kind(0, l, table.num_tokens + t).squeeze(1)
            x = x + layer.attend_absorbed(x, cache, q_start=start)
            x = x + layer.mlp(x)
        table.advance(t)
        return self.lm_head(self.ln_f(x))

    def decode(self, token_id, table):
        token_id = token_id.to(self.embed.weight.device)
        pos = table.num_tokens
        x = self.embed(token_id) + self.pos(
            torch.tensor([pos], device=token_id.device))
        for l, layer in enumerate(self.layers):
            table.append(l, layer.latent(
                x, torch.tensor([pos], device=token_id.device)).unsqueeze(1),
                None)
            cache = table.gather_kind(0, l, table.num_tokens + 1).squeeze(1)
            x = x + layer.attend_absorbed(x, cache, q_start=pos)
            x = x + layer.mlp(x)
        table.advance(1)
        return self.lm_head(self.ln_f(x))

    # -- batched (padded) inference ---------------------------------------

    def prefill_batch(self, input_ids_list, tables):
        """Prefill a batch of prompt suffixes in one forward (padded)."""
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
        """Shared batched layer stack; MLA edition of the dense version.

        Padded positions write garbage latent vectors that are never read:
        the gather is cut to each row's real length and the mask hides the
        rest — exactly the dense model's convention.
        """
        b = x.shape[0]
        device = x.device
        # absolute positions per row: cursor + offset within the chunk
        # (cursors don't move until the caller advances after the stack)
        pos = torch.arange(x.shape[1], device=device)[None, :] + \
            torch.tensor([tb.num_tokens for tb in tables],
                         device=device)[:, None]
        for l, layer in enumerate(self.layers):
            vec = layer.latent(x, pos)                  # (B, T, D)
            for i in range(b):
                tables[i].append(l, vec[i].unsqueeze(1), None)
            maxkv = max(lens)
            pool = tables[0].pool
            nb = max(len(tb.blocks) for tb in tables)
            bt = torch.zeros(b, nb, dtype=torch.long, device=device)
            for i in range(b):
                bt[i, :len(tables[i].blocks)] = torch.tensor(
                    tables[i].blocks, device=device)
            lat = pool.gather_batch(0, l, bt, maxkv).squeeze(2)
            o = layer.attend_batch(x, lat, lens, query_starts, q_len=t)
            x = x + o.reshape(x.shape)
            x = x + layer.mlp(x)
        return x

    # -- dense ground truth -------------------------------------------------

    def dense_forward(self, input_ids):
        """Full forward with explicit (non-absorbed) MLA math, no cache.

        Accepts a single sequence (T,) or a batch (B, T); returns logits of
        the same leading shape. The absorbed-vs-explicit identity makes
        this the greedy oracle for the paged path.
        """
        input_ids = input_ids.to(self.embed.weight.device)
        batched = input_ids.dim() > 1
        if not batched:
            input_ids = input_ids.unsqueeze(0)
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).expand(b, t)
        x = self.embed(input_ids) + self.pos(pos)
        for layer in self.layers:
            normed = layer.ln1(x)
            h, dh, dc = layer.n_heads, layer.head_dim, layer.d_latent
            c = layer.w_dkv(normed)                     # (B, T, d_c)
            k_r = apply_rope(layer.w_kr(normed), pos)   # rotated, as cached
            q_nope = layer.w_uq(normed).view(b, t, h, dh)
            q_rope = apply_rope(layer.w_qr(normed), pos).view(b, t, h,
                                                              layer.d_rope)
            w_uk = layer.w_uk.weight.view(h, dh, dc)
            w_uv = layer.w_uv.weight.view(h, dh, dc)
            k_c = torch.einsum("btc,hdc->bthd", c, w_uk)   # (B, T, H, d_h)
            v = torch.einsum("btc,hdc->bthd", c, w_uv)
            scores = torch.einsum("bthd,bshd->bhts", q_nope, k_c) \
                * layer.scale \
                + torch.einsum("bthr,bshr->bhts", q_rope,
                               k_r[:, :, None, :].expand(-1, -1, h, -1)) \
                * layer.scale
            q_pos = torch.arange(t, device=x.device)[:, None]
            k_pos = torch.arange(t, device=x.device)[None, :]
            scores = scores.masked_fill(
                q_pos[None, None] < k_pos[None, None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)       # (B, H, T, S)
            out = torch.einsum("bhts,bshd->bthd", probs, v)
            x = x + layer.wo(out.reshape(b, t, h * dh))
            x = x + layer.mlp(x)
        logits = self.lm_head(self.ln_f(x))
        return logits if batched else logits.squeeze(0)


class _MLATPLayer(_MLALayer):
    """MLA layer with the per-head maps sharded, latent maps replicated.

    Same slicing conventions as the dense TP layer (``tp._TPLayer``):
    w_uq/w_qr are COLUMN parallel (weight rows sliced), wo is ROW parallel
    (weight columns sliced, one all-reduce inside), w_uk/w_uv slice by
    HEAD — their out dim is per-head so rows ``[lo:hi]`` are exactly the
    local heads' maps. w_dkv/w_kr and the FFN follow the dense pattern.

    The attend/latent methods are inherited unchanged: they view
    ``w_uk.weight`` as ``(self.n_heads, d_h, d_c)`` and self.n_heads is
    the LOCAL count on a shard.
    """

    def __init__(self, dense_layer, world_size, rank):
        nn.Module.__init__(self)     # not _MLALayer.__init__: weights sliced
        if dense_layer.n_heads % world_size:
            raise ValueError("n_heads must be divisible by world_size")
        self.n_heads = dense_layer.n_heads // world_size
        self.head_dim = dense_layer.head_dim
        self.d_latent = dense_layer.d_latent
        self.d_rope = dense_layer.d_rope
        self.scale = dense_layer.scale
        hd, lh = self.head_dim, self.n_heads
        lo, hi = rank * lh * hd, (rank + 1) * lh * hd
        rlo, rhi = rank * lh * self.d_rope, (rank + 1) * lh * self.d_rope
        self.ln1 = dense_layer.ln1              # replicated
        self.ln2 = dense_layer.ln2
        # the latent projections have NO head axis: replicated on all ranks
        self.w_dkv = dense_layer.w_dkv
        self.w_kr = dense_layer.w_kr
        self.w_uq = ColumnParallelLinear(
            dense_layer.w_uq.weight[lo:hi].clone())
        self.w_qr = ColumnParallelLinear(
            dense_layer.w_qr.weight[rlo:rhi].clone())
        self.w_uk = ColumnParallelLinear(
            dense_layer.w_uk.weight[lo:hi].clone())
        self.w_uv = ColumnParallelLinear(
            dense_layer.w_uv.weight[lo:hi].clone())
        self.wo = RowParallelLinear(
            dense_layer.wo.weight[:, lo:hi].contiguous().clone())
        m = dense_layer.w1.weight.shape[0]      # 4 * d_model (full out dim)
        mlo, mhi = rank * (m // world_size), (rank + 1) * (m // world_size)
        self.w1 = ColumnParallelLinear(
            dense_layer.w1.weight[mlo:mhi].clone(),
            dense_layer.w1.bias[mlo:mhi].clone())
        self.w2 = RowParallelLinear(
            dense_layer.w2.weight[:, mlo:mhi].contiguous().clone(),
            dense_layer.w2.bias.clone())


class MLATransformerTP(MLATransformer):
    """Rank-local MLA shard for tensor-parallel inference.

    The MLA×TP contrast with dense MHA: a latent vector has NO head axis,
    so the KV pool cannot shard — every rank stores the FULL latent pool
    (``n_kv_heads=1, head_dim=d_latent+d_rope``), and only the per-head
    up/down projections split. vLLM's MLA workers do the same: attention
    heads split, the compressed cache is replicated.
    """

    def __init__(self, dense: MLATransformer, world_size: int, rank: int):
        nn.Module.__init__(self)
        if dense.n_heads % world_size:
            raise ValueError("n_heads must be divisible by world_size")
        self.world_size = world_size
        self.rank = rank
        self.vocab_size = dense.vocab_size
        self.d_model = dense.d_model
        self.n_layers = dense.n_layers
        self.n_heads = dense.n_heads // world_size   # local real heads
        # engine-facing cache geometry: REPLICATED latent pool (not sharded)
        self.n_kv_heads = 1
        self.head_dim = dense.head_dim
        self.kv_kinds = 1
        self.d_latent = dense.d_latent
        self.d_rope = dense.d_rope
        self.embed = dense.embed                     # replicated
        self.pos = dense.pos
        self.layers = nn.ModuleList(
            _MLATPLayer(layer, world_size, rank) for layer in dense.layers)
        self.ln_f = dense.ln_f
        self.lm_head = dense.lm_head
