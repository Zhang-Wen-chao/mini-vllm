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
                 max_prefill_tokens=256, max_running_tokens=512):
        self.model = model
        self.kv = KVBlockManager(num_blocks, block_size, model.n_heads,
                                 model.head_dim, num_layers=model.n_layers)
        self.scheduler = Scheduler(block_size=block_size,
                                   max_prefill_tokens=max_prefill_tokens,
                                   max_running_tokens=max_running_tokens)
        self._state = {}  # request_id -> {prompt_ids, generated, table}

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
        for req in list(self.scheduler.running):
            if req.status != "RUNNING":
                continue
            st = self._state[req.request_id]
            if req.num_generated == 0:
                logits = self.model.prefill(st["prompt_ids"], st["table"])
            else:
                logits = self.model.decode(
                    torch.tensor([st["generated"][-1]]), st["table"])
            token = int(torch.argmax(logits[-1]))
            st["generated"].append(token)
            req.num_generated += 1

    def _finish_completed(self):
        finished = []
        for req in list(self.scheduler.running):
            if req.num_generated >= req.max_new_tokens:
                self.scheduler.finish(req)
                self.kv.release_table(self._state[req.request_id]["table"])
                finished.append(req)
        return finished
