"""Speculative decoding v2 tests: n-gram proposer, engine-level verify,
bonus token, acceptance statistics, KV rollback.

Two models:
- TinyTransformer: the real paged model — acceptance depends on where its
  greedy decoding happens to loop, so only output equality and stats
  consistency can be asserted;
- CycleModel: a scripted model whose greedy output is a fixed token cycle,
  making draft acceptance (and the bonus token) fully deterministic.
"""

import pytest
import torch

from mini_vllm.engine import Engine
from mini_vllm.spec_decode import NgramProposer

from test_engine import greedy_reference, make_model


# -- NgramProposer units -------------------------------------------------------

def test_proposer_needs_a_non_overlapping_occurrence():
    p = NgramProposer(match_len=3, k=3)
    # the key [2,3,9] never occurs earlier: nothing to propose
    assert p.propose([1, 2, 3, 2, 3, 9]) == []
    # history too short to hold key + a full occurrence
    assert p.propose([1, 2, 3, 1, 2]) == []
    assert p.propose([1, 2], max_k=3) == []
    assert p.propose([1, 2, 3, 4], max_k=0) == []


def test_proposer_takes_the_rightmost_occurrence_and_caps_at_k():
    p = NgramProposer(match_len=2, k=2)
    # key = last 2 tokens = [2,3]; it occurs at starts 0 and 3 → the
    # rightmost occurrence wins and the draft is what followed it
    history = [2, 3, 9, 2, 3, 4, 2, 3]
    assert p.propose(history) == [4, 2], "rightmost occurrence, capped at k"
    p1 = NgramProposer(match_len=2, k=1)
    assert p1.propose(history) == [4]
    # no earlier occurrence of the key at all
    assert p.propose([1, 2, 3, 4, 5, 2, 3, 9]) == []
    # proposal may re-read the key region (repetitive text keeps drafting),
    # but only as far as tokens that actually exist in the history
    p2 = NgramProposer(match_len=2, k=4)
    assert p2.propose([1, 2, 1, 2, 1, 2]) == [1, 2]
    # an occurrence followed by enough history yields a full-width draft
    assert p2.propose([1, 2, 9, 9, 9, 9, 9, 1, 2]) == [9, 9, 9, 9]


def test_proposer_rejects_bad_config():
    with pytest.raises(ValueError):
        NgramProposer(match_len=0)


# -- deterministic scripted model ---------------------------------------------

class CycleModel(torch.nn.Module):
    """Greedy decoding follows a fixed cycle regardless of history.

    next(x) = cycle[(index(x) + 1) % len(cycle)], x outside the cycle
    enters it at cycle[0]. Logits are one-hot with a wide margin, so argmax
    is unambiguous. Attention is scripted (zero KV, no attention math).
    """

    supports_prefix_cache = True

    def __init__(self, vocab_size=32, cycle=(7, 11, 23)):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = 1
        self.n_heads = 1
        self.head_dim = 4
        self.cycle = list(cycle)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def next_token(self, x):
        x = int(x)
        if x in self.cycle:
            return self.cycle[(self.cycle.index(x) + 1) % len(self.cycle)]
        return self.cycle[0]

    def _logits(self, tokens):
        nxt = [self.next_token(t) for t in tokens.tolist()]
        logits = torch.full((len(nxt), self.vocab_size), -10.0)
        logits[torch.arange(len(nxt)), torch.tensor(nxt)] = 10.0
        return logits

    def prefill(self, input_ids, table):
        input_ids = input_ids.view(-1)
        kv = torch.zeros(input_ids.shape[0], self.n_heads, self.head_dim)
        table.append(0, kv, kv.clone())
        table.advance(input_ids.shape[0])
        return self._logits(input_ids)

    def decode(self, token_id, table):
        kv = torch.zeros(1, self.n_heads, self.head_dim)
        table.append(0, kv, kv.clone())
        table.advance(1)
        return self._logits(token_id.view(-1))


def cycle_reference(prompt, max_new_tokens):
    out = []
    x = int(prompt[-1])
    model = CycleModel()
    for _ in range(max_new_tokens):
        x = model.next_token(x)
        out.append(x)
    return out


# -- engine-level speculative tests ---------------------------------------------

def test_ngram_full_accept_emits_bonus_and_saves_steps():
    model = CycleModel()
    engine = Engine(model, block_size=4, num_blocks=16,
                    speculative_tokens=3, spec_method="ngram")
    prompt = torch.tensor([7])          # greedy: 11,23,7,11,23,...
    req = engine.add_request(prompt, max_new_tokens=12)
    steps = 0
    while engine.has_requests():
        engine.step()
        steps += 1
    assert engine.output(req)[len(prompt):] == cycle_reference(prompt, 12)
    s = engine.spec_stats
    assert s["drafted"] > 0 and s["accepted"] == s["drafted"], \
        "the cycle always agrees with its own n-gram draft"
    assert s["full_accepts"] >= 1 and s["bonus"] >= 1, \
        "a full accept must contribute the bonus token"
    assert s["fallbacks"] > 0, "early steps have no repeatable n-gram yet"
    assert s["steps"] + s["fallbacks"] < 12, \
        "speculation must finish in fewer forwards than tokens"
    assert steps < 12
    assert engine.acceptance_rate == 1.0
    assert len(engine.kv.pool.free_blocks) == 16, "no KV leak"


def test_ngram_respects_max_new_tokens_cap():
    # 12 tokens with a 4-token full-accept step: the last step must be
    # capped at max_new_tokens and the truncated bonus KV rolled back
    model = CycleModel()
    engine = Engine(model, block_size=4, num_blocks=16,
                    speculative_tokens=3, spec_method="ngram")
    prompt = torch.tensor([7])
    req = engine.add_request(prompt, max_new_tokens=8)
    while engine.has_requests():
        engine.step()
    out = engine.output(req)[len(prompt):]
    assert out == cycle_reference(prompt, 8)
    assert req.num_generated == 8
    assert len(engine.kv.pool.free_blocks) == 16, \
        "capped bonus KV must be truncated back into the pool"


def test_rejected_drafts_fall_back_to_correction_tokens():
    class _WrongProposer:
        # drafts are never in the cycle's next set: the target must reject
        # every draft and emit its own correction token instead
        def propose(self, history, max_k=None):
            k = max_k if max_k is not None else 3
            if k <= 0 or len(history) < 3:
                return []
            return [(t + 13) % 32 for t in history[-k:]]

    model = CycleModel()
    engine = Engine(model, block_size=4, num_blocks=16,
                    speculative_tokens=3, spec_method="ngram",
                    spec_proposer=_WrongProposer())
    prompt = torch.tensor([7])
    req = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    s = engine.spec_stats
    assert engine.output(req)[len(prompt):] == cycle_reference(prompt, 6), \
        "rejected speculation must not change the output"
    assert s["drafted"] > 0 and s["accepted"] == 0
    assert engine.acceptance_rate == 0.0
    assert s["bonus"] == 0 and s["full_accepts"] == 0


def test_ngram_with_real_model_matches_reference_and_no_leak():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram")
    prompts = [torch.tensor([3, 15, 27, 9, 42, 7]),
               torch.tensor([11, 4, 5])]
    reqs = [engine.add_request(p, max_new_tokens=8) for p in prompts]
    while engine.has_requests():
        engine.step()
    for p, r in zip(prompts, reqs):
        assert r.status == "FINISHED"
        assert engine.output(r)[len(p):] == greedy_reference(model, p, 8), \
            "speculation is an optimization, never a semantic change"
    s = engine.spec_stats
    assert s["accepted"] <= s["drafted"]
    assert s["steps"] + s["fallbacks"] <= 16
    assert len(engine.kv.pool.free_blocks) == 32, \
        "rejected draft KV must be rolled back to the pool"


def test_step_deltas_carry_multi_token_steps():
    model = CycleModel()
    engine = Engine(model, block_size=4, num_blocks=16,
                    speculative_tokens=3, spec_method="ngram")
    prompt = torch.tensor([7])
    req = engine.add_request(prompt, max_new_tokens=12)
    multi = []
    while engine.has_requests():
        for rid, tokens, finished in engine.step_deltas():
            assert rid == req.request_id
            if len(tokens) > 1:
                multi.append(len(tokens))
    assert multi, "a full accept must surface as a multi-token delta"


def test_spec_decode_validations():
    with pytest.raises(ValueError):
        Engine(make_model(), use_cuda_graph=True, speculative_tokens=3,
               spec_method="ngram")
    with pytest.raises(ValueError):
        Engine(make_model(), speculative_tokens=2, spec_method="eagle")


def test_spec_composes_with_prefix_cache_and_chunked_prefill():
    # Phase 20 unlocked both mutexes: the verify forward derives positions
    # from the table cursor (the supports_prefix_cache contract), and the
    # scheduler reserves k+1 tokens per verify step in the chunked budget
    engine = Engine(make_model(), block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram",
                    enable_prefix_cache=True)
    assert engine._ngram_proposer is not None
    engine = Engine(make_model(), block_size=4, num_blocks=32,
                    speculative_tokens=3, spec_method="ngram",
                    chunked_prefill=True, max_prefill_tokens=4)
    assert engine.scheduler.tokens_per_decode == 4
