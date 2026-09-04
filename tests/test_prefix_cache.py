import pytest
import torch

from mini_vllm.engine import Engine
from mini_vllm.kv_cache import BlockPool, BlockTable, PrefixCache

from test_engine import greedy_reference, make_model


def run_on(engine, prompt, max_new_tokens):
    """Run one request to completion on an existing engine (cache persists)."""
    req = engine.add_request(prompt, max_new_tokens=max_new_tokens)
    while engine.has_requests():
        engine.step()
    assert req.status == "FINISHED"
    return req


# -- unit tests: PrefixCache semantics -------------------------------------

def _table_with_blocks(pool, num_blocks, num_tokens):
    table = BlockTable(pool)
    table.blocks = [pool.allocate() for _ in range(num_blocks)]
    table.num_tokens = num_tokens
    return table


def test_prefix_cache_match_acquire_release_evict():
    pool = BlockPool(num_blocks=4, block_size=2, num_heads=1, head_dim=4)
    pc = PrefixCache(pool)
    assert pc.match([1, 2, 3, 4, 5]) == ([], [], 0)

    # register 2 full blocks (5th token is a partial tail: never cached)
    table = _table_with_blocks(pool, 2, 5)
    pc.register(table, [1, 2, 3, 4, 5])
    assert len(pc) == 2

    # hit on both blocks, and the chain survives a different tail
    blocks, hashes, num_tok = pc.match([1, 2, 3, 4, 9, 9])
    assert num_tok == 4 and blocks == table.blocks

    # chain isolation: a different first block matches nothing
    assert pc.match([9, 9, 3, 4, 5])[2] == 0
    # only full blocks are ever visible
    assert pc.match([1, 2])[2] == 2
    assert pc.match([1])[2] == 0

    # refcounting: registered=1, acquire=2, one release leaves 1
    pc.acquire(hashes)
    pc.release_hash(hashes[0])
    pc.release_hash(hashes[1])
    assert pc.evict(1) == 0, "referenced blocks must not be evicted"
    pc.release_hash(hashes[0])
    pc.release_hash(hashes[1])
    assert pc.evict(1) == 1, "unreferenced blocks are evictable (LRU first)"
    assert len(pool.free_blocks) == 3

    # refcount must not go negative
    with pytest.raises(RuntimeError):
        pc.release_hash(hashes[1])


def test_prefix_cache_register_dedupes_identical_blocks():
    pool = BlockPool(num_blocks=4, block_size=2, num_heads=1, head_dim=4)
    pc = PrefixCache(pool)
    t1 = _table_with_blocks(pool, 2, 4)
    pc.register(t1, [1, 2, 3, 4])
    _, hashes, _ = pc.match([1, 2, 3, 4])
    pc.release_hash(hashes[0])
    pc.release_hash(hashes[1])
    assert len(pool.free_blocks) == 2, "released refs keep blocks cached"

    # a second table that computed the same tokens adopts the cached blocks
    t2 = _table_with_blocks(pool, 2, 4)
    pc.register(t2, [1, 2, 3, 4])
    assert pc.deduped_blocks == 2
    assert t2.blocks == t1.blocks, "duplicate compute must be collapsed"
    assert len(pool.free_blocks) == 2, "the duplicates were freed"


# -- engine integration ------------------------------------------------------

def test_prefix_cache_disabled_by_default():
    assert Engine(make_model()).prefix_cache is None


def test_init_rejects_unsupported_combinations():
    model = make_model()
    with pytest.raises(ValueError):
        Engine(model, use_cuda_graph=True, enable_prefix_cache=True)

    class _NoPrefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError):
        Engine(_NoPrefix(), enable_prefix_cache=True)


def test_cache_hit_reuses_prefix_and_matches_dense_reference():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16, enable_prefix_cache=True)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7, 7, 33])
    pc = engine.prefix_cache

    run_on(engine, prompt, max_new_tokens=4)
    blocks, hashes, num_tok = pc.match(prompt.tolist())
    assert num_tok == 8, "the first run must cache the prompt's full blocks"

    # second run: full-prompt match is capped one block short so the last
    # block is recomputed and still yields logits for the final position
    r2 = run_on(engine, prompt, max_new_tokens=4)
    assert pc.hits_tokens == 4 and pc.hits_blocks == 1
    expected = greedy_reference(model, prompt, 4)
    assert engine.output(r2)[len(prompt):] == expected
    # finished requests' blocks stay cached, not freed. The tail block is
    # partial forever: the last generated token's KV is never written (the
    # request stops before the decode step that would compute it)
    assert len(engine.kv.pool.free_blocks) + len(pc) == 16
    assert len(pc) == 2


def test_shared_prefix_family_matches_reference():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=32, enable_prefix_cache=True)
    common = [5, 11, 23, 8, 14, 2, 31, 19]
    p1 = torch.tensor(common + [1, 2])
    p2 = torch.tensor(common + [3, 4, 5, 6])
    p3 = torch.tensor(common + [7])

    run_on(engine, p1, max_new_tokens=5)          # populates the cache
    r2 = engine.add_request(p2, max_new_tokens=5)  # both hit the shared
    r3 = engine.add_request(p3, max_new_tokens=5)  # prefix in the same step
    while engine.has_requests():
        engine.step()
    pc = engine.prefix_cache
    assert pc.hits_tokens == 16, "8 cached tokens per later request"
    for p, r in ((p2, r2), (p3, r3)):
        expected = greedy_reference(model, p, 5)
        assert engine.output(r)[len(p):] == expected
    assert len(engine.kv.pool.free_blocks) + len(pc) == 32


def test_concurrent_identical_prompts_dedupe_blocks():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16, enable_prefix_cache=True)
    prompt = torch.tensor([9, 4, 4, 21, 13, 30, 2, 8])
    r1 = engine.add_request(prompt, max_new_tokens=4)
    r2 = engine.add_request(prompt, max_new_tokens=4)
    while engine.has_requests():
        engine.step()
    pc = engine.prefix_cache
    assert pc.deduped_blocks >= 2, \
        "same-batch identical prompts must collapse onto one block set"
    expected = greedy_reference(model, prompt, 4)
    for r in (r1, r2):
        assert engine.output(r)[len(prompt):] == expected
    assert len(engine.kv.pool.free_blocks) + len(pc) == 16


def test_cached_block_survives_one_owner_finishing():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16, enable_prefix_cache=True)
    prompt = torch.tensor([6, 18, 3, 27, 12, 9, 40, 5])
    short = engine.add_request(prompt, max_new_tokens=2)
    long_ = engine.add_request(prompt, max_new_tokens=6)
    while engine.has_requests():
        engine.step()
    assert short.status == "FINISHED" and long_.status == "FINISHED"
    expected = greedy_reference(model, prompt, 6)
    assert engine.output(long_)[len(prompt):] == expected
    # after both owners are gone the blocks stay cached, not freed
    assert len(engine.prefix_cache) > 0
    assert len(engine.kv.pool.free_blocks) + len(engine.prefix_cache) == 16


def test_waiting_queue_with_prefix_cache_matches_reference():
    # same setup as test_engine's queueing test, with the cache on
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=6, enable_prefix_cache=True)
    prompts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
    waited = False
    steps = 0
    while engine.has_requests():
        assert steps < 200, "engine failed to make progress (livelock)"
        engine.step()
        steps += 1
        waited = waited or any(r in engine.scheduler.waiting for r in reqs)
    assert waited, "test setup should queue the second request"
    for p, r in zip(prompts, reqs):
        expected = greedy_reference(model, p, 6)
        assert engine.output(r)[len(p):] == expected


def test_preempt_releases_shared_blocks_then_rematches():
    # drive _make_room_for_next_tokens directly: full-lifecycle admission
    # keeps actual usage under the pool, so an external drain is the honest
    # way to reach the preemption branch
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=6, enable_prefix_cache=True)
    prompt = torch.tensor([1, 2, 3])
    r = engine.add_request(prompt, max_new_tokens=4)
    while r.num_generated < 1:
        engine.step()
    pc = engine.prefix_cache
    # only the prompt's first full block is cached: the first generated
    # token's KV is written by the NEXT decode step, so cursor is still 3
    assert len(pc) == 1

    drained = [engine.kv.pool.allocate()
               for _ in list(engine.kv.pool.free_blocks)]
    engine._make_room_for_next_tokens()
    assert r.status == "WAITING", "the running request must be preempted"
    assert engine.output(r) == prompt.tolist(), "generation restarts"
    for b in drained:
        engine.kv.pool.free(b)
    while engine.has_requests():
        engine.step()
    assert r.status == "FINISHED"
    expected = greedy_reference(model, prompt, 4)
    assert engine.output(r)[len(prompt):] == expected
    assert pc.hits_tokens >= 0  # re-admission may or may not hit, both fine


def test_eviction_frees_blocks_for_new_admission():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=8, enable_prefix_cache=True)
    # 12-token prompts cache 4 blocks each (3 prompt + 1 completed decode
    # block), so three sequential requests overflow the 8-block pool
    prompts = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
               torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
               torch.tensor([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])]
    outs = []
    for p in prompts:
        r = run_on(engine, p, max_new_tokens=6)
        outs.append(engine.output(r))
    assert engine.prefix_cache.evicted_blocks >= 1, \
        "the tiny pool must evict cache blocks to admit later requests"
    for p, out in zip(prompts, outs):
        expected = greedy_reference(model, p, 6)
        assert out[len(p):] == expected


def test_warmup_does_not_pollute_cache():
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16, enable_prefix_cache=True)
    engine.warmup(batch_size=2, max_prompt_len=8, max_new_tokens=2)
    assert len(engine.prefix_cache) == 0
    assert len(engine.kv.pool.free_blocks) == 16


def test_cold_match_after_clear_matches_reference():
    # a request whose cached prefix was fully evicted between runs must
    # fall back to a full recompute and stay correct
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16, enable_prefix_cache=True)
    prompt = torch.tensor([2, 9, 17, 25, 33, 41, 8, 16])
    run_on(engine, prompt, max_new_tokens=4)
    engine.prefix_cache.clear()
    assert len(engine.kv.pool.free_blocks) == 16
    r2 = run_on(engine, prompt, max_new_tokens=4)
    assert engine.prefix_cache.hits_tokens == 0
    expected = greedy_reference(model, prompt, 4)
    assert engine.output(r2)[len(prompt):] == expected
