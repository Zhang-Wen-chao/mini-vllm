"""Phase 20: feature-composition matrix cleanup.

The Phase 7/8 mutexes "spec × prefix cache" and "spec × chunked prefill"
were conservative gates, not fundamental conflicts. This file pins the
conditions that make each composition legal:

- spec × prefix cache: the verify forward derives positions from the
  table cursor (the ``supports_prefix_cache`` contract), post-verify
  truncate keeps whole-block granularity, and registration funnels
  through ``_append_tokens`` — the same "last confirmed token's KV is
  not yet written" invariant as plain decode;
- spec × chunked prefill: a verify forward costs k+1 tokens, so the
  scheduler must reserve k+1 per decode request in the chunked budget
  (vLLM v1 counts draft tokens in ``max_num_batched_tokens`` too).
"""

import torch

from mini_vllm.engine import Engine
from mini_vllm.scheduler import Scheduler

from test_engine import greedy_reference, make_model


class _AlwaysPropose:
    """Deterministic injected proposer (the Phase 11 pluggability): every
    decode step drafts 3 tokens so the verify path is guaranteed to run
    even though a random-weight model never repeats an n-gram."""

    def propose(self, history, max_k=None):
        return [5, 6, 7][:3 if max_k is None else max_k]


def _run(engine, prompt, max_new_tokens):
    req = engine.add_request(prompt, max_new_tokens=max_new_tokens)
    while engine.has_requests():
        engine.step()
    return req


def test_spec_with_prefix_cache_shares_blocks_and_matches_reference():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=64,
                    enable_prefix_cache=True,
                    speculative_tokens=3, spec_method="ngram",
                    spec_proposer=_AlwaysPropose())
    p1 = torch.tensor([3, 15, 27, 9, 42, 7])
    p2 = torch.tensor([3, 15, 27, 9, 50, 51, 52])
    r1 = _run(engine, p1, 6)
    r2 = _run(engine, p2, 4)
    assert engine.prefix_cache.hits_tokens == 4, "shared prefix matched"
    assert engine.spec_stats["drafted"] > 0, "verify path ran"
    assert engine.output(r1)[len(p1):] == greedy_reference(model, p1, 6)
    assert engine.output(r2)[len(p2):] == greedy_reference(model, p2, 4)


def test_spec_with_chunked_prefill_and_concurrent_decode():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=64,
                    chunked_prefill=True, max_prefill_tokens=4,
                    max_running_tokens=64,
                    speculative_tokens=3, spec_method="ngram",
                    spec_proposer=_AlwaysPropose())
    long_prompt = torch.tensor([3, 15, 27, 9, 42, 7, 11, 4, 5, 6])
    short_prompt = torch.tensor([21, 5, 8])
    long_req = engine.add_request(long_prompt, max_new_tokens=6)
    short_req = engine.add_request(short_prompt, max_new_tokens=6)
    steps = 0
    while engine.has_requests():
        engine.step()
        steps += 1
    assert steps >= 4, "the long prompt must have been chunked across steps"
    assert engine.spec_stats["drafted"] > 0, "verify path ran"
    assert engine.output(long_req)[len(long_prompt):] == \
        greedy_reference(model, long_prompt, 6)
    assert engine.output(short_req)[len(short_prompt):] == \
        greedy_reference(model, short_prompt, 6)


def test_chunked_budget_reserves_spec_verify_tokens():
    s = Scheduler(block_size=4, max_prefill_tokens=8, chunked_prefill=True)
    s.tokens_per_decode = 4           # k=3: a verify forward costs k+1
    decode_req = s.add_request(prompt_len=2, max_new_tokens=8)
    long_req = s.add_request(prompt_len=10, max_new_tokens=4)
    s.schedule(free_blocks=64)        # admits both: chunks 2 and 6
    decode_req.num_prefilled = 2      # decode_req is now decoding
    decode_req.num_generated = 1
    new, _ = s.schedule(free_blocks=64)
    assert long_req.chunk_tokens == 4, \
        "the verify step must reserve k+1 tokens before prefill chunks"
