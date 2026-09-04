import pytest
import torch

from mini_vllm.engine import Engine
from mini_vllm.scheduler import Scheduler

from test_engine import greedy_reference, make_model


def run_on(engine, prompt, max_new_tokens):
    req = engine.add_request(prompt, max_new_tokens=max_new_tokens)
    while engine.has_requests():
        engine.step()
    assert req.status == "FINISHED"
    return req


# -- scheduler unit tests ----------------------------------------------------

def test_chunked_admission_splits_long_prompt():
    s = Scheduler(block_size=4, max_prefill_tokens=8, chunked_prefill=True)
    r1 = s.add_request(prompt_len=5)
    r2 = s.add_request(prompt_len=4)
    new, running = s.schedule(free_blocks=100)
    assert new == [r1, r2], "chunked mode admits both with partial chunks"
    assert r1.chunk_tokens == 5
    assert r2.chunk_tokens == 3, "only the budget left after r1"


def test_chunked_full_prompt_gate_unchanged_when_disabled():
    s = Scheduler(block_size=4, max_prefill_tokens=8)
    s.add_request(prompt_len=5)
    r2 = s.add_request(prompt_len=4)
    new, _ = s.schedule(free_blocks=100)
    assert new != [] and r2 not in new, "default mode rejects the 4-token prompt"


def test_chunked_running_prefill_owns_the_budget():
    s = Scheduler(block_size=4, max_prefill_tokens=8, chunked_prefill=True)
    r1 = s.add_request(prompt_len=10)
    s.schedule(free_blocks=100)
    r1.num_prefilled = 8                     # engine prefilled 8 of 10
    r2 = s.add_request(prompt_len=4)
    s.schedule(free_blocks=100)
    # r1 continues first (2 tokens), leaving 6 for r2's chunk of 4
    assert r1.chunk_tokens == 2
    assert r2.chunk_tokens == 4
    assert r2.status == "RUNNING"


# -- engine integration -------------------------------------------------------

def test_chunked_prefill_matches_dense_reference():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=32,
                    max_prefill_tokens=6, chunked_prefill=True)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7, 7, 33, 21, 5])
    r = engine.add_request(prompt, max_new_tokens=6)
    progress = []
    while engine.has_requests():
        engine.step()
        progress.append(r.num_prefilled)
    assert r.status == "FINISHED"
    assert min(progress) < 10 and progress[-1] == 10, \
        "a 10-token prompt under a 6-token budget must span steps"
    expected = greedy_reference(model, prompt, 6)
    assert engine.output(r)[len(prompt):] == expected


def test_chunked_prefill_interleaves_decode_with_long_prefill():
    # the property chunked prefill buys (vLLM v1): a request that is already
    # decoding keeps decoding every step while a long prompt chunks through
    # the prefill budget — decodes are budgeted before prefill chunks
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=32,
                    max_prefill_tokens=6, chunked_prefill=True)
    short_prompt = torch.tensor([11, 4])
    short_r = engine.add_request(short_prompt, max_new_tokens=3)
    run_one = False
    while not run_one:
        engine.step()
        run_one = short_r.num_generated > 0
    long_prompt = torch.tensor([3, 15, 27, 9, 42, 7, 7, 33, 21, 5, 8, 2])
    long_r = engine.add_request(long_prompt, max_new_tokens=8)
    interleaved = False
    short_gens = 0
    while engine.has_requests():
        before = short_r.num_generated
        engine.step()
        progressed_prefill = 0 < long_r.num_prefilled < len(long_prompt)
        if progressed_prefill and short_r.num_generated > before:
            interleaved = True
        short_gens = max(short_gens, short_r.num_generated)
    assert short_r.status == "FINISHED"
    assert interleaved, \
        "short decodes must interleave with the long prompt's chunks"
    assert short_gens == 3
    for p, r, n in ((long_prompt, long_r, 8), (short_prompt, short_r, 3)):
        expected = greedy_reference(model, p, n)
        assert engine.output(r)[len(p):] == expected


def test_chunked_prefill_with_prefix_cache():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=32,
                    max_prefill_tokens=6, chunked_prefill=True,
                    enable_prefix_cache=True)
    common = [5, 11, 23, 8, 14, 2, 31, 19]
    p1 = torch.tensor(common + [1, 2, 3, 4, 5, 6])
    run_on(engine, p1, max_new_tokens=4)
    assert engine.prefix_cache.hits_tokens == 0
    p2 = torch.tensor(common + [7, 8, 9, 10, 11, 12])
    r2 = engine.add_request(p2, max_new_tokens=4)
    progress = []
    while engine.has_requests():
        engine.step()
        progress.append(r2.num_prefilled)
    assert engine.prefix_cache.hits_tokens == 8, "shared prefix matched"
    assert r2.num_prefilled == len(p2)
    expected = greedy_reference(model, p2, 4)
    assert engine.output(r2)[len(p2):] == expected


def test_chunked_prefill_rejects_cuda_graph():
    with pytest.raises(ValueError):
        Engine(make_model(), use_cuda_graph=True, chunked_prefill=True)
