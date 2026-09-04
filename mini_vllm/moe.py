"""Mixture-of-Experts FFN and Expert Parallelism (vLLM MoE + EP, miniaturized).

A MoE layer replaces the dense FFN with N expert FFNs and a router: every
token is scored against all experts, the top-k are selected and their
outputs combined weighted by (renormalized) router probabilities. Compute
is grouped per expert — each expert runs once on the tokens routed to it
(the "grouped GEMM" vLLM fuses into one kernel).

DeepSeek-style refinements (``inter_dim`` / ``n_shared``):

- FINE-GRAINED experts: many narrow experts (``inter_dim`` ≪ 4·d_model)
  instead of few wide ones — combinatorially more route combinations at
  the same activated-parameter budget;
- SHARED experts: ``n_shared`` experts added to EVERY token, unweighted
  and unroutered (always-on, weight 1.0) — they capture common knowledge
  so the routed experts specialize. They are computed AFTER the EP
  combine so replication never double-counts them.

Expert Parallelism shards the EXPERTS across ranks (EP=2 with 4 experts →
2 experts per rank). The router is replicated, so every rank sees every
token's routing decision, computes ONLY its own experts' contributions and
combines ranks into the full sum. Two combine shapes are implemented:

- ``"all_reduce"`` — one collective sums the per-rank partials;
- ``"p2p"`` — pairwise point-to-point send/recv with every peer, the
  all-to-all SHAPE vLLM's dispatch/combine uses (gloo has no all_to_all
  collective, which is the usual reason to write this form).

Both exchange full hidden states rather than routing tokens, so they are
the same sum:

    MoE(x) = Σ_e p_e(x) · E_e(x) = Σ_ranks ( Σ_{e ∈ rank} p_e · E_e )

Deterministic anchor: with n_experts=1 the router weight is exactly 1.0 and
expert 0 reuses the dense FFN weights, so a 1-expert MoE model must match
the dense TinyTransformer bit for bit (and n_shared=1 on top doubles it —
the shared expert IS the dense FFN added again).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tp import _all_reduce
from .model_runner import TinyTransformer


def _p2p_combine(x):
    """Sum per-rank partials with pairwise point-to-point send/recv.

    Every rank exchanges its contribution directly with every peer
    (2·(world-1) ops per rank) and accumulates — numerically the all-reduce
    sum, but shaped like the all-to-all exchange vLLM's EP dispatch/
    combine performs. Requires contiguous tensors (gloo contract).
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world, rank = dist.get_world_size(), dist.get_rank()
    out = x.contiguous().clone()
    for peer in range(world):
        if peer == rank:
            continue
        send_buf = x.contiguous().clone()
        recv_buf = torch.empty_like(out)
        reqs = [dist.isend(send_buf, peer), dist.irecv(recv_buf, peer)]
        for r in reqs:
            r.wait()
        out = out + recv_buf
    return out


class Router(nn.Module):
    """Top-k token router: softmax over all experts, renormalized top-k."""

    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= n_experts")
        self.top_k = top_k
        self.proj = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        """(T, d) → ((T, k) weights summing to 1, (T, k) expert indices)."""
        probs = F.softmax(self.proj(x), dim=-1)
        weights, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, idx


class Expert(nn.Module):
    """One FFN expert, w1 → gelu → w2 (narrow when fine-grained)."""

    def __init__(self, w1, w2):
        super().__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


def _make_expert(d_model, expert_id, inter_dim=None):
    """A freshly initialized expert; seeded per expert id so every rank
    builds identical expert weights regardless of construction order.
    ``inter_dim`` narrows the intermediate width (fine-grained experts)."""
    inter = 4 * d_model if inter_dim is None else inter_dim
    torch.manual_seed(1000 + expert_id)
    return Expert(nn.Linear(d_model, inter), nn.Linear(inter, d_model))


class _MoELayer(nn.Module):
    """Dense attention + top-k MoE FFN (optional shared experts).

    Attention modules are SHARED with the source dense layer (inference
    only). ``expert_ids`` lists the expert indices this instance computes —
    the full range for EP=1, this rank's slice for EP. Shared experts are
    replicated on every rank and never routed.
    """

    def __init__(self, dense_layer, n_experts, top_k, expert_ids=None,
                 inter_dim=None, n_shared=0, ep_combine="all_reduce"):
        super().__init__()
        if ep_combine not in ("all_reduce", "p2p", "none"):
            raise ValueError("ep_combine must be 'all_reduce', 'p2p' or 'none'")
        d = dense_layer.ln1.weight.shape[0]
        self.n_heads = dense_layer.n_heads
        self.head_dim = dense_layer.head_dim
        self.ln1 = dense_layer.ln1
        self.wq = dense_layer.wq
        self.wk = dense_layer.wk
        self.wv = dense_layer.wv
        self.wo = dense_layer.wo
        self.ln2 = dense_layer.ln2
        self.n_experts = n_experts
        self.ep_combine = ep_combine
        self.router = Router(d, n_experts, top_k)
        ids = list(range(n_experts)) if expert_ids is None \
            else list(expert_ids)
        if not ids:
            raise ValueError("an EP rank must own at least one expert")
        self.expert_offset = min(ids)
        # expert 0 keeps the dense FFN weights ONLY at the dense width —
        # a fine-grained expert has nowhere to copy 4·d weights from
        self.experts = nn.ModuleList(
            Expert(dense_layer.w1, dense_layer.w2)
            if (e == 0 and inter_dim is None)
            else _make_expert(d, e, inter_dim)
            for e in ids)
        # shared experts: always active, unweighted, replicated across EP
        self.shared = nn.ModuleList(
            Expert(dense_layer.w1, dense_layer.w2)
            if (i == 0 and inter_dim is None)
            else _make_expert(d, 5000 + i, inter_dim)
            for i in range(n_shared))

    def split_heads(self, x):
        if x.dim() == 3:
            b, t, _ = x.shape
            return x.view(b, t, self.n_heads, self.head_dim)
        return x.view(x.shape[0], self.n_heads, self.head_dim)

    def mlp(self, x):
        """Route every token to its top-k experts and combine.

        With EP, only ``self.experts`` (this rank's slice) contribute; the
        ranks' partials are summed by the configured combine. Shared
        experts run AFTER the combine so each rank adds them exactly once.
        """
        shape = x.shape
        # ln2 first (the dense MLP's pre-norm): routing and experts both
        # consume the normalized hidden state
        flat = self.ln2(x).reshape(-1, shape[-1])
        weights, idx = self.router(flat)          # (T, k), (T, k)
        out = torch.zeros_like(flat)
        for local_e, expert in enumerate(self.experts):
            e = self.expert_offset + local_e
            mask = idx == e                       # (T, k)
            if not mask.any():
                continue
            token_pos, slot = mask.nonzero(as_tuple=True)
            h = expert(flat[token_pos])           # (n_e, d)
            w = weights[token_pos].gather(1, slot.unsqueeze(1))   # (n_e, 1)
            out.index_add_(0, token_pos, h * w)
        if self.ep_combine == "p2p":
            out = _p2p_combine(out)
        elif self.ep_combine == "all_reduce":
            out = _all_reduce(out)
        # "none": this instance owns every expert (or is a reference
        # oracle inside a distributed job) — the local sum IS the full sum
        for expert in self.shared:                # always-on, weight 1.0
            out = out + expert(flat)
        return out.view_as(x)


class MoETinyTransformer(TinyTransformer):
    """TinyTransformer whose FFNs are top-k MoE layers.

    Built from a dense model: expert 0 of every layer keeps the dense FFN
    weights (1-expert MoE == dense, bit for bit), the router and extra
    experts are freshly initialized (per-expert seeded, see _make_expert).

    For Expert Parallelism, ``ep_size/ep_rank`` restrict each instance to
    its slice of experts; run one model per rank under torch.distributed
    and the combine (``ep_combine``: all-reduce or pairwise p2p) sums the
    partials (module docstring).
    """

    supports_prefix_cache = True

    def __init__(self, dense: TinyTransformer, n_experts=4, top_k=2,
                 ep_size=1, ep_rank=0, inter_dim=None, n_shared=0,
                 ep_combine="all_reduce"):
        nn.Module.__init__(self)
        if n_experts % ep_size != 0:
            raise ValueError("n_experts must be divisible by ep_size")
        if not 0 <= ep_rank < ep_size:
            raise ValueError("ep_rank out of range")
        self.vocab_size = dense.vocab_size
        self.d_model = dense.d_model
        self.n_layers = dense.n_layers
        self.n_heads = dense.n_heads
        self.head_dim = dense.head_dim
        self.embed = dense.embed
        self.pos = dense.pos
        self.ln_f = dense.ln_f
        self.lm_head = dense.lm_head
        per_rank = n_experts // ep_size
        expert_ids = (list(range(ep_rank * per_rank, (ep_rank + 1) * per_rank))
                      if ep_size > 1 else None)
        self.layers = nn.ModuleList(
            _MoELayer(dense_layer, n_experts, top_k, expert_ids,
                      inter_dim=inter_dim, n_shared=n_shared,
                      ep_combine=ep_combine)
            for dense_layer in dense.layers)
