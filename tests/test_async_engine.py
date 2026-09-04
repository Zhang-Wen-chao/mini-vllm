"""Async engine tests: EngineCore process + streaming AsyncLLM frontend.

The oracle is still the dense greedy reference: a streamed token sequence
must be exactly what the synchronous engine (and the dense re-forward)
produces. Concurrency, abort and shutdown are exercised against the real
subprocess.
"""

import asyncio

import torch

from mini_vllm.async_engine import AsyncLLM
from mini_vllm.engine import Engine

from test_engine import greedy_reference, make_model


PROMPTS = [[3, 15, 27, 9, 42, 7], [11, 4, 5], [23, 8, 14, 2, 31, 19, 44]]
MAX_NEW = 6


def test_step_deltas_match_total_output():
    # the delta contract at the sync layer: concatenating step deltas
    # reproduces the full generated sequence, the last delta is marked
    # finished, and prefill-only steps emit nothing
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16)
    prompt = torch.tensor([3, 15, 27, 9, 42, 7])
    req = engine.add_request(prompt, max_new_tokens=5)
    tokens = []
    finished_flags = []
    while engine.has_requests():
        deltas = engine.step_deltas()
        for rid, new_tokens, finished in deltas:
            assert rid == req.request_id
            tokens.extend(new_tokens)
            finished_flags.append(finished)
    assert tokens == greedy_reference(model, prompt, 5)
    assert finished_flags[-1] is True
    assert all(not f for f in finished_flags[:-1])
    assert len(tokens) == 5


def test_abort_releases_request_from_any_queue():
    # abort a WAITING request before it is ever scheduled
    model = make_model()
    engine = Engine(model, block_size=4, num_blocks=16)
    r = engine.add_request(torch.tensor([3, 15, 27, 9]), max_new_tokens=4)
    assert engine.abort(r.request_id) is True
    assert not engine.has_requests()
    assert len(engine.kv.pool.free_blocks) == 16, "nothing was ever held"
    assert engine.abort(r.request_id) is False, "second abort is a no-op"

    # abort a RUNNING request mid-generation: blocks go back to the pool
    r2 = engine.add_request(torch.tensor([11, 4, 5]), max_new_tokens=8)
    while r2.num_generated < 2:
        engine.step()
    free_before = len(engine.kv.pool.free_blocks)
    assert engine.abort(r2.request_id) is True
    assert not engine.has_requests()
    assert len(engine.kv.pool.free_blocks) > free_before


def test_async_streaming_matches_dense_reference():
    async def main():
        llm = AsyncLLM(make_model, block_size=4, num_blocks=32,
                       enable_prefix_cache=True)
        try:
            async def one(prompt):
                return [tok async for tok in llm.generate(prompt, MAX_NEW)]

            outs = await asyncio.gather(*[one(p) for p in PROMPTS])
            assert llm.num_active_requests() == 0, "all requests finished"
        finally:
            llm.shutdown()
        return outs

    outs = asyncio.run(main())
    model = make_model()
    for prompt, out in zip(PROMPTS, outs):
        expected = greedy_reference(model, torch.tensor(prompt), MAX_NEW)
        assert out == expected, "stream must equal the dense reference"
        assert len(out) == MAX_NEW


def test_async_streams_progress_independently():
    # both streams run concurrently on one core: the short request finishes
    # first and the long one keeps streaming afterwards (per-request queues
    # decouple consumers; chunked prefill keeps the batch mixing)
    async def main():
        llm = AsyncLLM(make_model, block_size=4, num_blocks=32,
                       chunked_prefill=True, max_prefill_tokens=5)
        long_prompt = [3, 15, 27, 9, 42, 7, 7, 33, 21, 5, 8, 2]
        timeline = []   # (stream_name, token) in arrival order

        async def one(name, prompt, n):
            async for tok in llm.generate(prompt, n):
                timeline.append((name, tok))

        try:
            await asyncio.gather(one("short", [11, 4, 5], 2),
                                 one("long", long_prompt, 6))
        finally:
            llm.shutdown()
        return timeline

    timeline = asyncio.run(main())
    shorts = [tok for name, tok in timeline if name == "short"]
    longs = [tok for name, tok in timeline if name == "long"]
    assert len(shorts) == 2 and len(longs) == 6
    last_short = max(i for i, (name, _) in enumerate(timeline)
                     if name == "short")
    assert any(i > last_short for i, (name, _) in enumerate(timeline)
               if name == "long"), \
        "the long stream must keep producing after the short one finished"

    model = make_model()
    assert shorts == greedy_reference(model, torch.tensor([11, 4, 5]), 2)
    long_prompt = torch.tensor([3, 15, 27, 9, 42, 7, 7, 33, 21, 5, 8, 2])
    assert longs == greedy_reference(model, long_prompt, 6)


def test_async_abort_frees_core_request():
    async def main():
        llm = AsyncLLM(make_model, block_size=4, num_blocks=32)
        try:
            long_prompt = [3, 15, 27, 9, 42, 7, 7, 33, 21, 5, 8, 2]

            async def run_two_tokens():
                gen = llm.generate(long_prompt, 50)
                toks = []
                async for tok in gen:
                    toks.append(tok)
                    if len(toks) == 2:
                        break
                await gen.aclose()   # triggers ABORT in the core
                return toks

            aborted_tokens = await run_two_tokens()
            assert len(aborted_tokens) == 2

            # the core must not be stuck on the aborted 50-token request:
            # a fresh request still completes, and nothing stays active
            out = [tok async for tok in llm.generate([11, 4, 5], 3)]
            assert len(out) == 3
            for _ in range(50):
                if llm.num_active_requests() == 0:
                    break
                await asyncio.sleep(0.02)
            assert llm.num_active_requests() == 0, "aborted request leaked"
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_async_shutdown_is_idempotent():
    async def main():
        llm = AsyncLLM(make_model, block_size=4, num_blocks=16)
        out = [tok async for tok in llm.generate([3, 15, 27, 9], 2)]
        assert len(out) == 2
        llm.shutdown()
        llm.shutdown()
        assert not llm._proc.is_alive()
        return out

    asyncio.run(main())
