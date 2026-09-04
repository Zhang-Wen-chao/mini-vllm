"""Pipeline parallelism: split the transformer's LAYERS across ranks.

Where tensor parallelism slices every matrix inside one layer, pipeline
parallelism gives each rank a contiguous SLAB of layers — a *stage*: rank 0
owns the embedding plus the first ``L/P`` layers, the last rank owns the
final layers plus ``ln_f``/``lm_head``. Activations flow forward rank by
rank (p2p send/recv); only the last stage ever sees logits.

The control plane stays SPMD, exactly like our TP engine: every rank runs
the identical scheduler over identical requests, which buys two things:

- every stage-to-stage activation's SHAPE is locally computable from the
  request list — no metadata channel. vLLM instead broadcasts the scheduled
  batch (metadata + token ids) to every worker each step; the p2p shape
  problem is the same, only the answer differs.
- greedy sampling happens on the last stage (the only one with lm_head)
  and the sampled token ids are broadcast to everyone. Ranks without
  logits return a one-hot proxy so the engine's ``argmax(logits)``
  contract is untouched. This is the same data vLLM ships across the
  process boundary: token ids, never logits.

The KV pool shards along too: each rank's BlockPool stores only ITS layers
(``num_layers`` = local count) — the layer analogue of TP's head sharding.

Phase 19 adds two compositions:

- **micro-batch pipeline** (``micro_batch_size``): the batch is split into
  micro-batches that flow through the stages overlapped (GPipe forward-only
  schedule — see ``_prefill_microbatch``);
- **PP×TP** (``PPTPTransformer``): each stage becomes a TP group — world =
  pp_size × tp_size, rank = pp_rank·tp_size + tp_rank, the standard
  2-D parallel layout; the KV pool is then sharded by stage AND by head.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .model_runner import TinyTransformer
from .tp import TPTransformer, _TPLayer


class PPTransformer(TinyTransformer):
    supports_prefix_cache = True

    def __init__(self, dense, pp_size=1, pp_rank=0, micro_batch_size=None):
        nn.Module.__init__(self)
        if dense.n_layers % pp_size != 0:
            raise ValueError(
                f"n_layers={dense.n_layers} is not divisible by "
                f"pp_size={pp_size}")
        if not 0 <= pp_rank < pp_size:
            raise ValueError(
                f"pp_rank={pp_rank} out of range [0, {pp_size})")
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.micro_batch_size = micro_batch_size
        self.micro_prefills = 0          # observability: micro path ran
        # cross-stage endpoints in GLOBAL rank space. Plain PP: pp_rank±1.
        # PP×TP overrides both with (pp_rank±1)·tp_size + tp_rank.
        self._prev_rank = pp_rank - 1 if pp_rank > 0 else None
        self._next_rank = pp_rank + 1 if pp_rank < pp_size - 1 else None
        # sampled-token broadcast source: the last stage. PP×TP re-points
        # this at the last stage's lead (tp_rank 0) global rank.
        self._sample_src = pp_size - 1
        per_stage = dense.n_layers // pp_size
        lo, hi = pp_rank * per_stage, (pp_rank + 1) * per_stage
        self.vocab_size = dense.vocab_size
        self.d_model = dense.d_model
        self.n_heads = dense.n_heads
        self.head_dim = dense.head_dim
        self.device = next(dense.parameters()).device
        # engine-facing: the local KV pool stores local layers only
        self.n_layers = per_stage
        # stage slices are shared views of the dense weights (zero-copy —
        # every rank constructs the same dense model, keeps only its slab)
        self.layers = nn.ModuleList(list(dense.layers)[lo:hi])
        # boundary modules belong to exactly one stage each
        self.embed = dense.embed if pp_rank == 0 else None
        self.pos = dense.pos if pp_rank == 0 else None
        last = pp_rank == pp_size - 1
        self.ln_f = dense.ln_f if last else None
        self.lm_head = dense.lm_head if last else None

    # -- stage plumbing ----------------------------------------------------

    # gloo's send/recv bind the raw tensor pointer and cannot read GPU
    # memory (its collectives stage via host internally; its p2p does
    # not) — CUDA p2p must hop through host memory when on gloo.

    def _p2p_send(self, x, dst):
        assert dist.is_initialized(), \
            "PP>1 forward requires an initialized process group"
        x = x.contiguous()
        if x.is_cuda and dist.get_backend() == "gloo":
            dist.send(x.cpu(), dst)
        else:
            dist.send(x, dst)

    def _p2p_isend(self, x, dst):
        """Async send; returns the (possibly host-staged) buffer and the
        work handle — the caller must keep both alive until wait()."""
        assert dist.is_initialized(), \
            "PP>1 forward requires an initialized process group"
        x = x.contiguous()
        if x.is_cuda and dist.get_backend() == "gloo":
            x = x.cpu()
        return x, dist.isend(x, dst)

    def _broadcast_sampled(self, tokens=None, lead=None):
        """Broadcast the last stage's sampled token ids, returned on host
        memory. Staging device follows the backend: NCCL only moves CUDA
        tensors, gloo (here) only host memory."""
        assert dist.is_initialized(), \
            "PP>1 forward requires an initialized process group"
        nccl = dist.get_backend() == "nccl"
        if tokens is None:              # consumer side allocates the buffer
            tokens = torch.empty(lead, dtype=torch.long,
                                 device=self.device if nccl else "cpu")
        elif nccl:
            tokens = tokens.to(self.device)
        dist.broadcast(tokens, src=self._sample_src)
        return tokens.cpu()

    def _p2p_recv(self, shape, src):
        assert dist.is_initialized(), \
            "PP>1 forward requires an initialized process group"
        if self.device.type == "cuda" and dist.get_backend() == "gloo":
            cpu = torch.empty(shape)
            dist.recv(cpu, src)
            return cpu.to(self.device)
        buf = torch.empty(shape, device=self.device)
        dist.recv(buf, src)
        return buf

    def _stage_forward(self, x, table):
        """The rank's own layers; layer indices are LOCAL to this stage and
        address this rank's KV pool."""
        for l, layer in enumerate(self.layers):
            x = self._attn_layer(x, layer, table, l)
            x = x + layer.mlp(x)
        return x

    def _finish(self, x):
        """Last stage samples and broadcasts token ids; every rank returns
        a (..., V) tensor whose argmax is those ids.

        The one-hot proxy keeps the engine's ``argmax(logits)`` sampling
        contract without shipping logits between processes.
        """
        lead = tuple(x.shape[:-1])
        if self.pp_rank == self.pp_size - 1:
            logits = self.lm_head(self.ln_f(x))
            if self.pp_size > 1:
                self._broadcast_sampled(
                    tokens=logits.argmax(dim=-1).long().contiguous())
            return logits
        if self.pp_size > 1:
            tokens = self._broadcast_sampled(lead=lead)
        else:                       # unreachable: a 1-stage model IS last
            raise RuntimeError("non-last rank in a 1-stage pipeline")
        proxy = torch.zeros(lead + (self.vocab_size,))
        proxy.scatter_(-1, tokens.unsqueeze(-1), 1.0)
        return proxy

    # -- streaming (paged) inference --------------------------------------

    def prefill(self, input_ids, table):
        t = input_ids.shape[0]
        if self.pp_rank == 0:
            start = table.num_tokens
            x = self.embed(input_ids) + self.pos(
                torch.arange(start, start + t, device=input_ids.device))
        else:
            # SPMD: every rank holds the same request list, so the incoming
            # activation's shape is computable locally
            x = self._p2p_recv((t, self.d_model), self._prev_rank)
        x = self._stage_forward(x, table)
        # advance the table cursor (local bookkeeping, every rank): the
        # streaming paths were never exercised by the engine's batched
        # schedule until the spec-verify forward used them — the missing
        # advance surfaced there as a truncate "extension"
        table.advance(t)
        if self._next_rank is not None:
            self._p2p_send(x, self._next_rank)
        return self._finish(x)

    def decode(self, token_id, table):
        if self.pp_rank == 0:
            pos = table.num_tokens
            token_id = token_id.to(self.embed.weight.device)
            x = self.embed(token_id) + self.pos(
                torch.tensor([pos], device=token_id.device))
        else:
            x = self._p2p_recv((1, self.d_model), self._prev_rank)
        x = self._stage_forward(x, table)
        table.advance(1)
        if self._next_rank is not None:
            self._p2p_send(x, self._next_rank)
        return self._finish(x)

    # -- batched (padded) inference ---------------------------------------

    def prefill_batch(self, input_ids_list, tables):
        b = len(input_ids_list)
        if (self.pp_size > 1 and self.micro_batch_size
                and b > self.micro_batch_size):
            # enough rows to fill the pipeline: overlap micro-batches
            # through the stages instead of running the whole batch lockstep
            return self._prefill_microbatch(input_ids_list, tables)
        lens = [x.shape[0] for x in input_ids_list]
        starts = [tb.num_tokens for tb in tables]
        kv_lens = [starts[i] + lens[i] for i in range(b)]
        max_len = max(lens)
        if self.pp_rank == 0:
            device = self.embed.weight.device
            padded = torch.zeros(b, max_len, dtype=torch.long, device=device)
            for i, xid in enumerate(input_ids_list):
                padded[i, :lens[i]] = xid.to(device)
            pos = torch.arange(max_len, device=device)[None, :] + \
                torch.tensor(starts, device=device)[:, None]
            x = self.embed(padded) + self.pos(pos)
        else:
            x = self._p2p_recv((b, max_len, self.d_model), self._prev_rank)
        x = self._run_layers_batch(x, tables, max_len, kv_lens,
                                   query_starts=starts)
        for i, tb in enumerate(tables):
            tb.advance(lens[i])
        if self._next_rank is not None:
            self._p2p_send(x, self._next_rank)
        out = self._finish(x)
        return [out[i, lens[i] - 1] for i in range(b)]

    def decode_batch(self, token_ids, tables):
        b = len(token_ids)
        if self.pp_rank == 0:
            device = self.embed.weight.device
            tokens = torch.stack(token_ids).to(device).view(b)
            positions = torch.tensor([tb.num_tokens for tb in tables],
                                     device=device)
            x = self.embed(tokens.view(b, 1)) + \
                self.pos(positions).unsqueeze(1)
        else:
            x = self._p2p_recv((b, 1, self.d_model), self._prev_rank)
        lens = [tb.num_tokens + 1 for tb in tables]
        x = self._run_layers_batch(x, tables, 1, lens,
                                   query_starts=[l - 1 for l in lens])
        for tb in tables:
            tb.advance(1)
        if self._next_rank is not None:
            self._p2p_send(x, self._next_rank)
        out = self._finish(x)                       # (b, 1, V)
        return [row[0] for row in out]

    # -- micro-batch pipeline (GPipe forward-only) --------------------------

    def _prefill_microbatch(self, input_ids_list, tables):
        """Prefill a batch as overlapping micro-batches through the stages.

        The batch is split into micro-batches of ``micro_batch_size`` rows;
        stage ``s`` computes micro-batch ``m`` at slot ``m + s``, so stages
        work on DIFFERENT micro-batches simultaneously instead of the whole
        batch moving stage by stage in lockstep. Bubble fraction drops from
        lockstep's (P-1)/P to GPipe's (P-1)/(M+P-1) — with M micro-batches
        the bubble shrinks as the batch grows.

        Honest scope note: this is GPipe, not 1F1B. 1F1B's warmup depth
        (``pp_size - pp_rank - 1`` in-flight micro-batches per rank) exists
        to bound activation memory when BACKWARD passes interleave with
        forward; forward-only inference has no backward, so the GPipe
        schedule IS the inference schedule (vLLM likewise splits large
        prefills into micro-batches that pipeline through the stages).

        Sends are async (``isend``, buffers kept alive until drained) so
        stage 0 can run ahead of stage 1; every rank advances ITS OWN tables
        per micro-batch (SPMD: identical request lists, so the incoming
        activation shape per micro-batch is locally computable) and all
        ranks join ONE final token broadcast after the pipeline drains.
        Decode stays lockstep — decode batches are already micro-batch
        sized (one token per sequence).
        """
        b = len(input_ids_list)
        lens = [x.shape[0] for x in input_ids_list]
        bounds = list(range(0, b, self.micro_batch_size))
        sends = []                       # keep send buffers alive
        outs = [None] * b                # last-position logits per row
        for lo in bounds:
            rows = range(lo, min(lo + self.micro_batch_size, b))
            n_mb = len(tuple(rows))
            starts = [tables[i].num_tokens for i in rows]
            kv_lens = [starts[j] + lens[i] for j, i in enumerate(rows)]
            max_len = max(lens[i] for i in rows)
            if self.pp_rank == 0:
                device = self.embed.weight.device
                padded = torch.zeros(n_mb, max_len, dtype=torch.long,
                                     device=device)
                for j, i in enumerate(rows):
                    padded[j, :lens[i]] = input_ids_list[i].to(device)
                pos = torch.arange(max_len, device=device)[None, :] + \
                    torch.tensor(starts, device=device)[:, None]
                x = self.embed(padded) + self.pos(pos)
            else:
                x = self._p2p_recv((n_mb, max_len, self.d_model),
                                   self._prev_rank)
            mb_tables = [tables[i] for i in rows]
            x = self._run_layers_batch(x, mb_tables, max_len, kv_lens,
                                       query_starts=starts)
            for i in rows:               # advance per micro-batch, not per batch
                tables[i].advance(lens[i])
            if self.pp_rank == self.pp_size - 1:
                logits = self.lm_head(self.ln_f(x))
                for j, i in enumerate(rows):
                    outs[i] = logits[j, lens[i] - 1]
            elif self._next_rank is not None:
                sends.append(self._p2p_isend(x, self._next_rank))
        for _, req in sends:             # drain: sends matched by now
            req.wait()
        # single broadcast AFTER the drain — collective order identical on
        # every rank, and the LAST stage joins too (it is the src): a
        # broadcast only the consumers call is a distributed deadlock
        if self.pp_rank == self.pp_size - 1:
            self._broadcast_sampled(
                tokens=torch.stack([o.argmax() for o in outs]).long())
        else:
            tokens = self._broadcast_sampled(lead=(b,))
            proxy = torch.zeros(b, self.vocab_size)
            proxy.scatter_(1, tokens.unsqueeze(1), 1.0)
            outs = [proxy[i] for i in range(b)]
        self.micro_prefills += 1
        return outs


class PPTPTransformer(PPTransformer):
    """PP×TP: a pipeline stage IS a tensor-parallel group.

    World = pp_size × tp_size with rank = pp_rank·tp_size + tp_rank — the
    standard 2-D parallel layout. Each stage's tp_size ranks hold the SAME
    layer slab, TP-sharded: column-parallel weights sliced per rank,
    row-parallel linears all-reducing WITHIN the stage's group (a
    group-scoped all-reduce — reducing across the whole world would mix
    stages and silently corrupt activations; with tp_size=1 the singleton
    stage group still must be created, since group=None means the WORLD).

    The KV pool is doubly sharded: layers by stage (PP) AND heads by rank
    (TP) — pool.num_layers == n_layers/pp_size and pool.num_heads ==
    n_heads/tp_size. Cross-stage p2p ships the (replicated per TP rank)
    hidden state from every rank of stage s to its same-tp_rank peer in
    stage s+1; only the last stage's ranks run lm_head, and the sampled
    token broadcast comes from that stage's lead rank.

    Everything else — SPMD control plane, p2p plumbing, one-hot proxy,
    prefix-cache convention — is inherited from PPTransformer unchanged.
    Micro-batch pipelining is a plain-PP feature (``micro_batch_size``
    forced to None here): composing the micro schedule with the TP batched
    path is out of teaching scope, batches run lockstep as in Phase 14.
    """

    def __init__(self, dense, pp_size=1, pp_rank=0, tp_size=1, tp_rank=0):
        if dense.n_heads % tp_size != 0:
            raise ValueError(
                f"n_heads={dense.n_heads} is not divisible by "
                f"tp_size={tp_size}")
        world = pp_size * tp_size
        if world > 1 and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "PP×TP with world>1 needs an initialized process group")
        super().__init__(dense, pp_size=pp_size, pp_rank=pp_rank,
                         micro_batch_size=None)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # engine-facing geometry: local layers (PP) × local heads (TP)
        self.n_heads = dense.n_heads // tp_size
        # new_group is a WORLD-wide operation: every rank must create ALL
        # stage groups (creating only your own is a classic distributed
        # hang); non-member ranks receive a placeholder they never use
        self.tp_group = None
        if pp_size > 1:
            groups = [dist.new_group(list(range(s * tp_size,
                                                (s + 1) * tp_size)))
                      for s in range(pp_size)]
            self.tp_group = groups[pp_rank]
        # cross-stage endpoints in global rank space
        if pp_rank > 0:
            self._prev_rank = (pp_rank - 1) * tp_size + tp_rank
        if pp_rank < pp_size - 1:
            self._next_rank = (pp_rank + 1) * tp_size + tp_rank
        self._sample_src = (pp_size - 1) * tp_size   # last stage, tp_rank 0
        # rebuild the stage slab with TP-sharded linears (scoped all-reduce)
        per_stage = dense.n_layers // pp_size
        lo, hi = pp_rank * per_stage, (pp_rank + 1) * per_stage
        self.layers = nn.ModuleList(
            _TPLayer(layer, tp_size, tp_rank, group=self.tp_group)
            for layer in list(dense.layers)[lo:hi])

    # the TP reshape rules: attention output is LOCAL width
    # (n_heads·head_dim), NOT d_model — the PP base reshapes would silently
    # mis-shard whenever tp_size > 1. Everything else (stage plumbing, p2p,
    # broadcast, sampling proxy) is inherited unchanged.
    _attn_layer = TPTransformer._attn_layer
    _run_layers_batch = TPTransformer._run_layers_batch
