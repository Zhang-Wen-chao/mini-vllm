"""Async engine: an EngineCore process + an asyncio streaming frontend.

The vLLM v1 split, miniaturized:

- **EngineCore** (``_core_main``) runs in its OWN process and owns the model,
  the KV pool and the scheduler. It self-loops: each iteration drains
  control messages from the input queue (ADD / ABORT / STATUS / SHUTDOWN),
  then — while requests remain — runs one ``engine.step_deltas()`` and
  pushes the per-request token deltas to the output queue. The frontend
  never touches the model; requests cross the boundary as plain lists of
  token ids, exactly like vLLM's serialized ``EngineCoreRequest``.
- **AsyncLLM** is the asyncio frontend. A pump thread reads the output
  queue and routes each delta to the right per-request ``asyncio.Queue``
  via ``loop.call_soon_threadsafe`` (vLLM v1 instead reads a zmq socket
  from the event loop; the routing problem is the same). ``generate()``
  is an async generator that yields token ids one at a time; cancelling
  it (breaking out / ``aclose()``) sends ABORT so the core frees the
  request's KV instead of generating for a consumer that is gone.

Greedy sampling is deterministic, so no sampling parameters cross the
boundary and every stream observes exactly the synchronous engine's
output.
"""

import asyncio
import queue
import threading
import time
import multiprocessing as mp

import torch


def _core_main(in_q, out_q, model_factory, engine_kwargs):
    """EngineCore process body: own the engine, self-loop, stream deltas."""
    from .engine import Engine
    engine = Engine(model_factory(), **engine_kwargs)
    client_req = {}          # client_key -> engine request_id
    out_q.put(("READY", None))
    try:
        while True:
            shutdown = False
            while True:
                try:
                    cmd, payload = in_q.get_nowait()
                except queue.Empty:
                    break
                if cmd == "ADD":
                    key, prompt, max_new_tokens = payload
                    req = engine.add_request(torch.tensor(prompt, dtype=torch.long),
                                             max_new_tokens=max_new_tokens)
                    client_req[key] = req.request_id
                    out_q.put(("ADDED", (key, req.request_id)))
                elif cmd == "ABORT":
                    rid = client_req.pop(payload, None)
                    if rid is not None:
                        engine.abort(rid)
                        out_q.put(("ABORTED", rid))
                elif cmd == "STATUS":
                    out_q.put(("STATUS",
                               len(engine.scheduler.waiting) +
                               len(engine.scheduler.running) +
                               len(engine.scheduler.swapped)))
                elif cmd == "SHUTDOWN":
                    shutdown = True
            if shutdown:
                return
            if engine.has_requests():
                for rid, tokens, finished in engine.step_deltas():
                    out_q.put(("OUT", (rid, tokens, finished)))
                    if finished:
                        client_req = {k: v for k, v in client_req.items()
                                      if v != rid}
            else:
                time.sleep(0.0005)   # idle: no busy-spin
    finally:
        out_q.put(("EXIT", None))


class AsyncLLM:
    """Asyncio frontend over an EngineCore subprocess.

    Args:
        model_factory: zero-arg callable building the model — executed in
            the core process, so the parent never holds model weights.
        **engine_kwargs: forwarded to ``Engine`` in the core process
            (block_size, num_blocks, enable_prefix_cache, ...).
    """

    def __init__(self, model_factory, **engine_kwargs):
        ctx = mp.get_context("spawn")
        self._in_q = ctx.Queue()
        self._out_q = ctx.Queue()
        self._proc = ctx.Process(target=_core_main, daemon=True,
                                 args=(self._in_q, self._out_q,
                                       model_factory, engine_kwargs))
        self._proc.start()
        kind, _ = self._out_q.get(timeout=60)
        if kind != "READY":
            raise RuntimeError("EngineCore failed to start")
        self._loop = None
        self._pump = None
        self._pending = {}    # client_key -> asyncio.Queue (before ADDED)
        self._queues = {}     # engine request_id -> asyncio.Queue
        self._status = None   # (value, threading.Event) for STATUS replies
        self._next_key = 0    # client keys must be picklable (queue IPC)

    # -- public API ---------------------------------------------------------

    def generate(self, prompt_ids, max_new_tokens):
        """Async generator yielding generated token ids one by one.

        The final token arrives with the stream closing. Breaking out early
        aborts the request in the core (KV released, generation stops).
        """
        return self._generate(list(map(int, prompt_ids)),
                              int(max_new_tokens))

    async def _generate(self, prompt_ids, max_new_tokens):
        self._ensure_pump()
        q = asyncio.Queue()
        # an mp.Queue pickles payloads in its feeder thread; an unpicklable
        # key would fail SILENTLY there and the ADD would never arrive
        key = self._next_key
        self._next_key += 1
        self._pending[key] = q
        self._in_q.put(("ADD", (key, prompt_ids, max_new_tokens)))
        finished = False
        try:
            while True:
                token = await q.get()
                if token is None:      # sentinel: request finished
                    finished = True
                    return
                yield token
        finally:
            if not finished:
                # consumer went away: tell the core to free the request
                self._pending.pop(key, None)
                self._in_q.put(("ABORT", key))

    def num_active_requests(self):
        """Blocking introspection: requests the core still owns."""
        self._ensure_pump()
        event = threading.Event()
        self._status = (-1, event)
        self._in_q.put(("STATUS", None))
        event.wait(timeout=30)
        value = self._status[0]
        self._status = None
        return value

    def shutdown(self):
        """Stop the core process and the pump thread (idempotent)."""
        if self._proc.is_alive():
            self._in_q.put(("SHUTDOWN", None))
            self._proc.join(timeout=30)
            if self._proc.is_alive():
                self._proc.terminate()
        if self._pump is not None:
            self._pump.join(timeout=5)
            self._pump = None

    # -- internals ----------------------------------------------------------

    def _ensure_pump(self):
        if self._pump is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None   # sync caller (e.g. num_active_requests)
        self._pump = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump.start()

    def _pump_loop(self):
        """Drain the core's output queue; route to per-request queues."""
        while True:
            try:
                kind, payload = self._out_q.get(timeout=5)
            except queue.Empty:
                if not self._proc.is_alive():
                    return
                continue
            if kind == "EXIT":
                return
            if kind == "ADDED":
                key, rid = payload
                q = self._pending.pop(key, None)
                if q is not None and self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        self._queues.__setitem__, rid, q)
            elif kind == "OUT":
                rid, tokens, finished = payload
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        self._dispatch, rid, tokens, finished)
            elif kind == "ABORTED":
                rid = payload
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._queues.pop, rid, None)
            elif kind == "STATUS":
                # no loop hop needed: STATUS is read from sync code
                if self._status is not None:
                    _, event = self._status
                    self._status = (payload, event)
                    event.set()

    def _dispatch(self, rid, tokens, finished):
        q = self._queues.get(rid)
        if q is None:
            return
        for token in tokens:
            q.put_nowait(token)
        if finished:
            q.put_nowait(None)
            self._queues.pop(rid, None)
