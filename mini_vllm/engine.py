"""A minimal synchronous inference engine.

One ``step()`` is one forward pass over the whole running batch:

1. admit new WAITING requests (scheduler), preempting when the KV pool is
   too small to give every running request a block for its next token;
2. prefill every request that has produced no token yet (first token comes
   from the prefill logits);
3. decode one token for every other running request;
4. finish requests that hit ``max_new_tokens`` and release their KV blocks.

Generation is greedy (argmax) and deterministic, which is what makes the
engine's output directly comparable to a dense re-forward reference.
"""

import torch

from .kv_cache import KVBlockManager
from .scheduler import Scheduler


class Engine:
    def __init__(self, model, block_size=16, num_blocks=64,
                 max_prefill_tokens=256, max_running_tokens=512,
                 device=None, dtype=None, use_cuda_graph=False):
        self.model = model
        self.use_cuda_graph = use_cuda_graph
        if device is None:
            param = next(model.parameters())
            device = str(param.device)
        if dtype is None:
            dtype = getattr(model, "dtype", None)
            if dtype is None:
                dtype = next(model.parameters()).dtype
        self.kv = KVBlockManager(num_blocks, block_size,
                                 getattr(model, "n_kv_heads", model.n_heads),
                                 model.head_dim, num_layers=model.n_layers,
                                 device=device, dtype=dtype)
        self.scheduler = Scheduler(block_size=block_size,
                                   max_prefill_tokens=max_prefill_tokens,
                                   max_running_tokens=max_running_tokens)
        self._state = {}  # request_id -> {prompt_ids, generated, table}
        self._graphs = {}      # (batch_key) -> {bucket_blocks: buf}
        self._graph_buckets = None
        self._graph_key = None  # request ids captured in the graphs

    # -- public API -------------------------------------------------------

    def add_request(self, input_ids, max_new_tokens=16):
        req = self.scheduler.add_request(len(input_ids), max_new_tokens)
        self._state[req.request_id] = {
            "prompt_ids": input_ids,
            "generated": [],
            "table": self.kv.create_table(),
        }
        return req

    def has_requests(self):
        return (self.scheduler.has_running_requests() or
                bool(self.scheduler.waiting))

    def output(self, req):
        st = self._state[req.request_id]
        return list(st["prompt_ids"].tolist()) + st["generated"]

    # -- core loop --------------------------------------------------------

    def warmup(self, batch_size, max_prompt_len, max_new_tokens=4):
        """Pre-capture the CUDA graph ladders for a workload shape, the way
        vLLM captures its graphs at init. Graphs are keyed by batch size, so
        the real batch of the same size reuses them without re-capturing."""
        import torch as _torch
        for _ in range(batch_size):
            ids = _torch.zeros(max_prompt_len, dtype=_torch.long)
            self.add_request(ids, max_new_tokens=max_new_tokens)
        while self.has_requests():
            self.step()
        self._state.clear()
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.finished.clear()

    def step(self):
        self.scheduler.schedule(len(self.kv.pool.free_blocks))
        self._make_room_for_next_tokens()
        self._prefill_or_decode()
        return self._finish_completed()

    # -- internals ---------------------------------------------------------

    def _make_room_for_next_tokens(self):
        """Preempt until every running request can get its next KV block."""
        for req in list(self.scheduler.running):
            if req.status != "RUNNING":
                continue
            st = self._state[req.request_id]
            if (st["prompt_ids"].shape[0] + req.num_generated) % \
                    self.scheduler.block_size != 0:
                continue  # room left in the current block
            while not self.kv.pool.free_blocks:
                victim = self.scheduler.preempt()
                if victim is None:
                    break
                self.kv.release_table(self._state[victim.request_id]["table"])
                # recompute-based preemption: restart from the prompt
                self._state[victim.request_id]["generated"] = []
                victim.num_generated = 0

    def _prefill_or_decode(self):
        running = [r for r in self.scheduler.running if r.status == "RUNNING"]
        if not running:
            return
        new = [r for r in running if r.num_generated == 0]
        decode = [r for r in running if r.num_generated > 0]
        batchable = hasattr(self.model, "prefill_batch") and \
            hasattr(self.model, "decode_batch")
        if batchable and new:
            if self.use_cuda_graph and \
                    hasattr(self.model, "capture_prefill_graph"):
                self._prefill_with_graph(new)
            else:
                prompts = [self._state[r.request_id]["prompt_ids"] for r in new]
                tables = [self._state[r.request_id]["table"] for r in new]
                for r, logits in zip(new, self.model.prefill_batch(prompts, tables)):
                    self._sample(r, logits)
        else:
            for r in new:
                st = self._state[r.request_id]
                logits = self.model.prefill(st["prompt_ids"], st["table"])
                self._sample(r, logits)
        if batchable and decode:
            if self.use_cuda_graph and \
                    hasattr(self.model, "capture_decode_graph"):
                self._decode_with_graph(decode)
            else:
                tokens = [torch.tensor(self._state[r.request_id]["generated"][-1])
                          for r in decode]
                tables = [self._state[r.request_id]["table"] for r in decode]
                for r, logits in zip(decode, self.model.decode_batch(tokens, tables)):
                    self._sample(r, logits)
        else:
            for r in decode:
                st = self._state[r.request_id]
                logits = self.model.decode(
                    torch.tensor([st["generated"][-1]]), st["table"])
                self._sample(r, logits)

    def _sample(self, req, logits):
        token = int(torch.argmax(logits))
        st = self._state[req.request_id]
        st["generated"].append(token)
        req.num_generated += 1

    def _prefill_with_graph(self, new):
        """CUDA-graph prefill: bucket by prompt length, replay, sample."""
        tables = [self._state[r.request_id]["table"] for r in new]
        prompts = [self._state[r.request_id]["prompt_ids"] for r in new]
        max_len = max(p.shape[0] for p in prompts)
        buckets = getattr(self, "_prefill_buckets", None)
        if buckets is None or self._prefill_key != len(new):
            ladder = [32]
            while ladder[-1] < max_len:
                ladder.append(ladder[-1] * 2)
            self._prefill_buckets = {
                L: self.model.capture_prefill_graph(tables, L) for L in ladder
            }
            self._prefill_key = len(new)
        bucket = None
        for L in sorted(self._prefill_buckets):
            if L >= max_len:
                bucket = self._prefill_buckets[L]
                break
        logits = self.model.replay_prefill_graph(bucket, tables, prompts)
        for i, r in enumerate(new):
            self._sample(r, logits[i])
            tables[i].advance(prompts[i].shape[0])

    def _decode_with_graph(self, decode):
        """CUDA-graph decode: capture a ladder of KV-length buckets when the
        running batch changes, then replay the smallest bucket that fits."""
        key = len(decode)   # graphs are reusable for any batch of this size
        tables = [self._state[r.request_id]["table"] for r in decode]
        bs = self.scheduler.block_size
        if key != self._graph_key:
            total_blocks = max(
                (self._state[r.request_id]["prompt_ids"].shape[0] +
                 r.max_new_tokens + bs - 1) // bs + 1 for r in decode)
            # ladder of block caps, each doubling, up to the full reserve
            buckets = [4]
            while buckets[-1] < total_blocks:
                buckets.append(buckets[-1] * 2)
            self._graph_buckets = buckets
            self._graphs = {
                nb: self.model.capture_decode_graph(tables, nb)
                for nb in buckets
            }
            self._graph_key = key
        # Python-side block allocation for the token each request writes now
        for t in tables:
            if t.num_tokens >= len(t.blocks) * bs:
                t.blocks.append(self.kv.pool.allocate())
        # smallest bucket that still covers every running request's KV
        max_blocks = max(len(t.blocks) for t in tables)
        if max_blocks > self._graph_buckets[-1]:
            # workload outgrew the ladder; extend it (safety net)
            buckets = list(self._graph_buckets)
            while buckets[-1] < max_blocks:
                buckets.append(buckets[-1] * 2)
            self._graph_buckets = buckets
            self._graphs.update({
                nb: self.model.capture_decode_graph(tables, nb)
                for nb in buckets
            })
        nb = self._graph_buckets[0]
        for cand in self._graph_buckets:
            if cand >= max_blocks:
                nb = cand
                break
        buf = self._graphs[nb]
        tokens = [self._state[r.request_id]["generated"][-1] for r in decode]
        logits = self.model.replay_decode_graph(buf, tables, tokens)
        sampled = logits.argmax(dim=-1).tolist()   # one sync for the batch
        for i, r in enumerate(decode):
            self._state[r.request_id]["generated"].append(sampled[i])
            r.num_generated += 1
            tables[i].advance(1)   # mirror the KV this step wrote

    def _finish_completed(self):
        finished = []
        for req in list(self.scheduler.running):
            if req.num_generated >= req.max_new_tokens:
                self.scheduler.finish(req)
                self.kv.release_table(self._state[req.request_id]["table"])
                finished.append(req)
        return finished
