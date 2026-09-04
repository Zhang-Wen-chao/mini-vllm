"""Speculative SAMPLING tests: top-k in the sampler chain and the
Leviathan rejection test for n-gram (point-mass) drafts.

Three layers:
- sampler units: top-k filters to exactly k tokens with EXACTLY zero mass
  on the rest, and composes with top-p;
- verify units: the mathematical core — with a point-mass proposer the
  rejection test's per-position output distribution equals the target's
  EXACTLY (p(d)·δ_d + (1-p(d))·norm(p - δ_d) = p), greedy degenerates to
  the argmax rule, and the bonus token comes from the last position's
  distribution;
- engine level: the sampled verify path keeps every spec invariant
  (≥1 token per step, stats consistent, KV rolled back) while drawing
  from the same sampler chain as plain decode.
"""

import torch

import pytest

from mini_vllm.engine import Engine
from mini_vllm.sampler import probs_from_logits, sample
from mini_vllm.spec_decode import verify_drafts_sampled

from test_engine import make_model


# -- sampler chain: top-k ------------------------------------------------------

def test_top_k_one_is_greedy_one_hot():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
    probs = probs_from_logits(logits, temperature=1.0, top_k=1)
    assert probs.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert sample(logits, temperature=1.0, top_k=1) == 1


def test_top_k_keeps_exactly_k_tokens_with_zero_mass_elsewhere():
    torch.manual_seed(0)
    logits = torch.randn(50)
    probs = probs_from_logits(logits, temperature=1.0, top_k=5)
    nonzero = (probs > 0).sum().item()
    assert nonzero == 5
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    # filtered tokens must carry EXACTLY zero mass (probability-space
    # filtering, not a log-domain roundtrip that leaves ~1e-20 behind)
    assert (probs[probs == 0] == 0).all()
    top5 = torch.topk(logits, 5).indices.tolist()
    support = (probs > 0).nonzero().flatten().tolist()
    assert sorted(support) == sorted(top5)


def test_top_k_larger_than_vocab_is_a_noop():
    torch.manual_seed(1)
    logits = torch.randn(8)
    full = probs_from_logits(logits, temperature=1.0)
    capped = probs_from_logits(logits, temperature=1.0, top_k=100)
    assert torch.allclose(full, capped)


def test_top_k_composes_with_top_p():
    torch.manual_seed(2)
    logits = torch.randn(32)
    probs = probs_from_logits(logits, temperature=1.5, top_k=10, top_p=0.8)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    assert (probs >= 0).all()
    assert (probs > 0).sum().item() <= 10, "top-p can only narrow top-k"


def test_top_k_validation():
    with pytest.raises(ValueError):
        probs_from_logits(torch.randn(4), temperature=1.0, top_k=0)
    with pytest.raises(ValueError):
        Engine(make_model(), top_k=0)


# -- verify_drafts_sampled: the rejection-test math ------------------------------

def _probs_from_p(p):
    """Logits whose temperature=1.0 softmax is exactly `p` (log-probs)."""
    p = torch.tensor(p, dtype=torch.float64)
    return p.log().float()


def test_greedy_temperature_zero_degenerates_to_argmax_rule():
    # argmax at both positions is token 2
    logits = torch.full((2, 5), -1.0)
    logits[0, 2] = 4.0
    logits[1, 2] = 4.0
    # matching draft: accepted, and the last position yields the bonus
    toks, m = verify_drafts_sampled(logits, [2], temperature=0.0)
    assert toks == [2, 2] and m == 1
    # mismatching draft: rejected, correction = the argmax
    toks, m = verify_drafts_sampled(logits, [3], temperature=0.0)
    assert toks == [2] and m == 0
    # multi-draft: longest argmax-consistent prefix
    toks, m = verify_drafts_sampled(logits, [2, 3, 2], temperature=0.0)
    assert toks == [2, 2] and m == 1


def test_point_mass_output_distribution_equals_target_exactly():
    # THE theorem, numerically: for a point-mass draft the per-position
    # output marginal is the target distribution p — not approximately,
    # exactly (p(d)·δ_d + (1-p(d))·norm(p-δ_d) telescopes back to p).
    torch.manual_seed(3)
    p = [0.5, 0.3, 0.15, 0.05]
    logits = torch.stack([_probs_from_p(p), _probs_from_p(p)])
    draft = 1                      # p(draft) = 0.3, so 70% of trials resample
    trials = 20000
    counts = torch.zeros(4)
    for _ in range(trials):
        toks, m = verify_drafts_sampled(logits, [draft], temperature=1.0)
        assert m == 1 or toks[0] != draft, \
            "a rejected draft must never be resampled (its mass is zeroed)"
        counts[toks[0]] += 1
    freq = counts / trials
    assert torch.allclose(freq, torch.tensor(p), atol=0.02), \
        f"output distribution must equal the target: {freq.tolist()} vs {p}"


def test_rejected_draft_is_never_resampled():
    torch.manual_seed(4)
    p = [0.05, 0.35, 0.35, 0.25]
    logits = torch.stack([_probs_from_p(p), _probs_from_p(p)])
    resampled = []
    for _ in range(4000):
        toks, m = verify_drafts_sampled(logits, [0], temperature=1.0)
        if m == 0:                 # the draft was rejected
            resampled.append(toks[0])
    assert len(resampled) > 3000, "p(draft)=0.05 must reject most of the time"
    assert 0 not in resampled, "resampling draws from p minus the draft"


def test_full_accept_bonus_comes_from_last_position():
    torch.manual_seed(5)
    p_last = [0.1, 0.2, 0.3, 0.4]
    # position 0 nearly always accepts the draft: one-hot with margin 24
    # (p(d) ≈ 1 - 2·e⁻¹², finite logits — no ±inf softmax edge cases)
    p_first = torch.full((4,), -12.0)
    p_first[3] = 12.0
    logits = torch.stack([p_first,
                          _probs_from_p(p_last)])
    bonuses = torch.zeros(4)
    trials = 8000
    for _ in range(trials):
        toks, m = verify_drafts_sampled(logits, [3], temperature=1.0)
        assert m == 1 and len(toks) == 2
        bonuses[toks[1]] += 1
    freq = bonuses / trials
    assert torch.allclose(freq, torch.tensor(p_last), atol=0.02), \
        f"bonus token must follow p_last: {freq.tolist()} vs {p_last}"


def test_chain_output_matches_target_per_position():
    # sequential correctness: position i's output marginal is p_i whether it
    # came from an accepted draft or a resample, for ANY proposer q — here
    # fixed point-mass drafts (the worst case, accept prob = p_i(d_i))
    torch.manual_seed(6)
    ps = [[0.6, 0.25, 0.1, 0.05],
          [0.1, 0.4, 0.4, 0.1],
          [0.2, 0.2, 0.2, 0.4]]
    logits = torch.stack([_probs_from_p(p) for p in ps] +
                         [_probs_from_p(ps[0])])   # bonus position, unjudged
    drafts = [0, 2, 1]
    trials = 20000
    counts = torch.zeros(3, 4)
    reached = torch.zeros(3)
    for _ in range(trials):
        toks, m = verify_drafts_sampled(logits, drafts, temperature=1.0)
        for i, tok in enumerate(toks[:3]):
            counts[i, tok] += 1
            reached[i] += 1
    for i, p in enumerate(ps):
        freq = counts[i] / reached[i]
        assert torch.allclose(freq, torch.tensor(p), atol=0.03), \
            f"position {i}: {freq.tolist()} vs {p} ({int(reached[i])} samples)"


def test_draft_outside_top_k_support_is_always_rejected():
    torch.manual_seed(7)
    logits = torch.randn(8).unsqueeze(0).repeat(2, 1)
    top2 = torch.topk(logits[0], 2).indices.tolist()
    outside = next(t for t in range(8) if t not in top2)
    toks, m = verify_drafts_sampled(logits, [outside], temperature=1.0,
                                    top_k=2)
    assert m == 0 and toks[0] in top2, \
        "zero support means zero acceptance probability"


# -- engine level: sampled spec path ----------------------------------------------

def test_sampled_spec_engine_matches_cycle_reference():
    # CycleModel's logits are one-hot with margin 20 → p(argmax) ≈ 1, so the
    # sampled path must reproduce the greedy cycle AND keep accepting drafts
    from test_spec_decode import CycleModel, cycle_reference
    model = CycleModel()
    engine = Engine(model, block_size=4, num_blocks=16,
                    speculative_tokens=3, spec_method="ngram",
                    temperature=1.0)
    prompt = torch.tensor([7])
    req = engine.add_request(prompt, max_new_tokens=12)
    while engine.has_requests():
        engine.step()
    assert engine.output(req)[len(prompt):] == cycle_reference(prompt, 12)
    s = engine.spec_stats
    assert s["drafted"] > 0
    assert engine.acceptance_rate > 0.9
    assert s["fallbacks"] > 0
    assert len(engine.kv.pool.free_blocks) == 16, "no KV leak"


def test_sampled_spec_engine_is_seeded_deterministic():
    model = make_model()          # weight init consumes RNG: seed AFTER it
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    torch.manual_seed(11)
    engine = Engine(model, block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram",
                    temperature=0.9, top_k=20, top_p=0.95)
    req = engine.add_request(prompt, max_new_tokens=10)
    while engine.has_requests():
        engine.step()
    first = engine.output(req)[len(prompt):]
    s1 = dict(engine.spec_stats)
    torch.manual_seed(11)
    engine2 = Engine(model, block_size=4, num_blocks=32,
                     speculative_tokens=3, spec_method="ngram",
                     temperature=0.9, top_k=20, top_p=0.95)
    req2 = engine2.add_request(prompt, max_new_tokens=10)
    while engine2.has_requests():
        engine2.step()
    assert engine2.output(req2)[len(prompt):] == first, \
        "same seed must reproduce the same sampled speculation"
    assert engine2.spec_stats == s1


def test_sampled_spec_engine_invariants_on_real_model():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram",
                    temperature=0.8, top_k=16)
    prompt = torch.tensor([11, 4, 5])
    req = engine.add_request(prompt, max_new_tokens=10)
    multi = []
    while engine.has_requests():
        for rid, tokens, _finished in engine.step_deltas():
            assert len(tokens) >= 1, "a spec step never regresses below 1"
            if len(tokens) > 1:
                multi.append(len(tokens))
    out = engine.output(req)[len(prompt):]
    assert req.status == "FINISHED" and len(out) == 10
    assert all(0 <= t < 64 for t in out)
    s = engine.spec_stats
    assert s["accepted"] <= s["drafted"]
    assert s["steps"] + s["fallbacks"] <= 10
    assert s["full_accepts"] <= s["steps"]
    assert len(engine.kv.pool.free_blocks) == 32, \
        "rejected draft KV must be rolled back to the pool"


def test_sampled_fallback_uses_the_same_sampler_chain():
    # the no-draft fallback path must sample through the same chain
    # (a temperature of exactly 0 there must stay greedy-deterministic)
    model = make_model()
    prompt = torch.tensor([3, 15, 27, 9])
    outs = set()
    for _ in range(6):
        engine = Engine(model, block_size=4, num_blocks=32,
                        speculative_tokens=3, spec_method="ngram",
                        temperature=1.0)
        req = engine.add_request(prompt, max_new_tokens=6)
        while engine.has_requests():
            engine.step()
        outs.add(tuple(engine.output(req)[len(prompt):]))
    assert len(outs) >= 2, "sampled fallbacks must vary across seeds"
