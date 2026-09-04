"""Speculative decoding components (vLLM v1 spec_decode, miniaturized).

``NgramProposer`` is the engine-side draft model that needs no weights at
all: it looks up the trailing n-gram of the request's own token history and
proposes what followed it last time (prompt-lookup decoding). The engine
then verifies the whole draft in ONE target forward — greedy acceptance
takes the longest argmax-consistent prefix; sampled acceptance uses the
Leviathan et al. rejection test so the OUTPUT distribution stays exactly
the target's (see ``verify_drafts_sampled``).

The proposer is deliberately a plain object (not an nn.Module) so the
engine can accept an injected replacement — the same pluggability that
lets vLLM v1 swap n-gram, EAGLE or Medusa proposers behind one interface.
"""

import torch

from .sampler import probs_from_logits


class NgramProposer:
    """Draft continuations from the request's own history.

    The match window is the last ``match_len`` tokens; it is searched in the
    history STRICTLY BEFORE itself (a window must end no later than
    ``len(history) - match_len``, so the key can never match itself). The
    RIGHTMOST occurrence wins and the draft is up to ``k`` tokens that
    followed it — the proposal may re-read the key region, which is exactly
    what keeps repetitive text drafting successfully.
    """

    def __init__(self, match_len=3, k=3):
        if match_len < 1:
            raise ValueError("match_len must be >= 1")
        self.match_len = match_len
        self.k = k

    def propose(self, history, max_k=None):
        """Return up to `max_k` draft tokens, or [] when nothing matches."""
        k = self.k if max_k is None else min(self.k, max_k)
        n = self.match_len
        if k <= 0 or len(history) < 2 * n:
            # the key (n tokens) plus a non-overlapping occurrence (n more)
            # cannot both fit
            return []
        key = history[-n:]
        limit = len(history) - 2 * n      # last possible window start
        for start in range(limit, -1, -1):
            if history[start:start + n] == key:
                return history[start + n:start + n + k]
        return []


def verify_drafts_sampled(logits, drafts, temperature=1.0, top_k=None,
                          top_p=1.0, generator=None):
    """Speculative sampling (Leviathan et al. 2023) for a POINT-MASS proposer.

    ``logits`` is the verify forward's ``(k+1, V)`` output: position i
    judges ``drafts[i]``. An n-gram proposal is deterministic — its draft
    distribution q is a one-hot on each draft token — which reduces the
    general test to:

    - accept draft i with probability ``min(1, p_i(d_i))`` (q(d)=1);
    - on the first rejection, resample ONE token from
      ``norm(max(p_i - q_i, 0))`` — the target distribution with the
      rejected draft's mass zeroed out;
    - if every draft is accepted, the last position's distribution yields
      the bonus token.

    The theorem's punchline survives the reduction: the per-position output
    distribution equals the target's EXACTLY, for any proposer q. With
    q = δ_d the arithmetic is visible — accept branch contributes
    ``p(d)·δ_d``; the resample branch contributes ``(1-p(d))·p(x)/(1-p(d))
    = p(x)`` off the draft token and zero on it. Greedy (``temperature<=0``)
    degenerates to the argmax acceptance rule: one-hot p accepts iff the
    draft matches the argmax and resamples the argmax as the correction.

    Returns ``(new_tokens, num_accepted_drafts)``; ``new_tokens`` is
    ``accepted_drafts + [resampled_or_bonus]``.
    """
    new_tokens = []
    for i, draft in enumerate(drafts):
        p = probs_from_logits(logits[i], temperature, top_k, top_p)
        if torch.rand((), generator=generator).item() < float(p[draft]):
            new_tokens.append(int(draft))
            continue
        residual = p.clone()
        residual[draft] = 0.0            # the rejected draft loses ALL mass
        residual = residual / residual.sum()
        new_tokens.append(
            int(torch.multinomial(residual, 1, generator=generator).item()))
        return new_tokens, i             # i drafts accepted, then corrected
    p_last = probs_from_logits(logits[len(drafts)], temperature, top_k,
                               top_p)
    new_tokens.append(
        int(torch.multinomial(p_last, 1, generator=generator).item()))
    return new_tokens, len(drafts)       # full accept: last token is bonus

