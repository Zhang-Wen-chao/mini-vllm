import pytest
import torch

from mini_vllm.engine import Engine
from mini_vllm.kv_cache import CpuSwapSpace, BlockPool, BlockTable

from test_engine import greedy_reference, make_model


def test_cpu_swap_space_accounting():
    pool = BlockPool(num_blocks=4, block_size=2, num_heads=1, head_dim=4)
    table = BlockTable(pool)
    table.blocks = [pool.allocate() for _ in range(2)]
    table.num_tokens = 4
    space = CpuSwapSpace(max_blocks=3)
    assert space.can_fit(2) and space.can_fit(3)
    handle = space.swap_out(table)
    assert handle.num_blocks == 2 and space.used_blocks == 2
    assert space.can_fit(1) and not space.can_fit(2)
    assert space.swap_in(handle).num_blocks == 2
    assert space.used_blocks == 0
    full = CpuSwapSpace(max_blocks=1)
    assert full.swap_out(table) is None, "budget refusal returns None"


def test_swap_preserves_progress_and_matches_reference():
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=6, enable_prefix_cache=True,
                    preemption="swap", swap_num_blocks=8)
    prompt = torch.tensor([1, 2, 3])
    r = engine.add_request(prompt, max_new_tokens=4)
    while r.num_generated < 1:
        engine.step()
    assert r.num_generated == 1 and r.num_prefilled == 3

    # drain the pool to force the swap-out branch of _make_room
    drained = [engine.kv.pool.allocate()
               for _ in list(engine.kv.pool.free_blocks)]
    engine._make_room_for_next_tokens()
    assert r.status == "SWAPPED"
    assert r.num_generated == 1, "swap keeps generation progress"
    assert r.num_prefilled == 3, "swap keeps prefill progress"
    assert engine.swap_space.used_blocks > 0
    assert len(engine.kv.pool.free_blocks) > 0, "GPU blocks were freed"

    for b in drained:
        engine.kv.pool.free(b)
    while engine.has_requests():
        engine.step()
    assert r.status == "FINISHED"
    assert engine.swap_space.swap_ins == 1
    expected = greedy_reference(model, prompt, 4)
    assert engine.output(r)[len(prompt):] == expected, \
        "restored sequence must generate exactly the reference tokens"


def test_swap_under_contention_keeps_both_outputs_correct():
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=8,
                    preemption="swap", swap_num_blocks=16)
    prompts = [torch.tensor([5, 11, 2]), torch.tensor([9, 4, 4, 21, 13])]
    reqs = [engine.add_request(p, max_new_tokens=5) for p in prompts]
    swapped = False
    steps = 0
    while engine.has_requests():
        assert steps < 300, "engine failed to make progress (livelock)"
        engine.step()
        steps += 1
        swapped = swapped or any(q.status == "SWAPPED" for q in reqs)
    assert swapped or engine.swap_space.swap_outs == 0
    for p, r in zip(prompts, reqs):
        expected = greedy_reference(model, p, 5)
        assert engine.output(r)[len(p):] == expected
    assert engine.swap_space.used_blocks == 0, "everything swapped back in"
    assert len(engine.kv.pool.free_blocks) == 8, "no block leak"


def test_swap_space_full_falls_back_to_recompute():
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=6,
                    preemption="swap", swap_num_blocks=1)
    prompt = torch.tensor([1, 2, 3])
    r = engine.add_request(prompt, max_new_tokens=4)
    while r.num_generated < 1:
        engine.step()
    drained = [engine.kv.pool.allocate()
               for _ in list(engine.kv.pool.free_blocks)]
    engine._make_room_for_next_tokens()
    assert r.status == "WAITING", "recompute preemption is the fallback"
    assert r.num_generated == 0, "recompute loses progress"
    for b in drained:
        engine.kv.pool.free(b)
    while engine.has_requests():
        engine.step()
    expected = greedy_reference(model, prompt, 4)
    assert engine.output(r)[len(prompt):] == expected


def test_swap_during_chunked_prefill_resumes_without_recompute():
    # cross-path test: a request swapped out MID-PREFILL (chunked mode)
    # restores its partial prefill progress and continues chunking.
    # pool must fit the full lifecycle (16 tokens = 8 blocks at bs=2)
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=8,
                    max_prefill_tokens=6, chunked_prefill=True,
                    preemption="swap", swap_num_blocks=8)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7, 7, 33, 21, 5, 8, 2])
    r = engine.add_request(prompt, max_new_tokens=4)
    steps = 0
    while r.num_prefilled < 6:      # one chunk of 6 of 12 tokens
        assert steps < 50, "request was never admitted (livelock)"
        engine.step()
        steps += 1
    assert 0 < r.num_prefilled < len(prompt)
    assert r.num_generated == 0

    drained = [engine.kv.pool.allocate()
               for _ in list(engine.kv.pool.free_blocks)]
    engine._make_room_for_next_tokens()
    assert r.status == "SWAPPED"
    assert r.num_prefilled == 6, "partial prefill progress is kept"
    for b in drained:
        engine.kv.pool.free(b)
    steps = 0
    while engine.has_requests():
        assert steps < 100, "engine failed to finish (livelock)"
        engine.step()
        steps += 1
    assert r.status == "FINISHED"
    assert r.num_prefilled == len(prompt)
    expected = greedy_reference(model, prompt, 4)
    assert engine.output(r)[len(prompt):] == expected


def test_swap_requires_budget_and_paged_model():
    with pytest.raises(ValueError):
        Engine(make_model(), preemption="swap")            # no swap blocks
    with pytest.raises(ValueError):
        Engine(make_model(), preemption="bogus")

    class _NoSwap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError):
        Engine(_NoSwap(), preemption="swap", swap_num_blocks=4)


def test_swapped_requests_block_engine_exit():
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=6,
                    preemption="swap", swap_num_blocks=8)
    prompt = torch.tensor([1, 2, 3])
    r = engine.add_request(prompt, max_new_tokens=4)
    while r.num_generated < 1:
        engine.step()
    drained = [engine.kv.pool.allocate()
               for _ in list(engine.kv.pool.free_blocks)]
    engine._make_room_for_next_tokens()
    assert r.status == "SWAPPED"
    assert engine.has_requests(), "swapped requests must keep the engine alive"
    for b in drained:
        engine.kv.pool.free(b)
