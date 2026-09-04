"""Sampling: temperature / top-k / top-p over one logits vector.

Extracted from the engine so the sampler chain and the speculative-verify
math share ONE definition of the target distribution (vLLM does the same:
the logits processors feed both the direct sampler and spec-decode
acceptance).

Chain order matches vLLM's sampler: temperature scaling -> top-k -> top-p,
each step renormalizing what is left. Filtering happens in probability
space so a filtered token's mass is EXACTLY zero (a log-domain roundtrip
would leave ~1e-20 behind). ``temperature <= 0`` short-circuits to greedy
argmax — the default that keeps the engine deterministic and comparable to
the dense reference.
"""

import torch


def probs_from_logits(logits, temperature=0.0, top_k=None, top_p=1.0):
    """The sampler's distribution: temperature, then top-k, then top-p.

    Returns a probability vector that sums to 1; filtered entries have
    exactly zero mass. Greedy (``temperature <= 0``) returns a one-hot on
    the argmax so downstream math never needs a special case.
    """
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[int(torch.argmax(logits))] = 1.0
        return probs
    scaled = logits / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        k = min(int(top_k), scaled.shape[-1])
        thresh = torch.topk(scaled, k, dim=-1).values[-1]
        scaled = torch.where(scaled < thresh,
                             torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, sorted_idx, sorted_probs)
        probs = filtered / filtered.sum()
    return probs


def sample(logits, temperature=0.0, top_k=None, top_p=1.0, generator=None):
    """Draw one token id; greedy argmax when ``temperature <= 0``."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    probs = probs_from_logits(logits, temperature, top_k, top_p)
    return int(torch.multinomial(probs, 1, generator=generator).item())
