"""Continuous batching scheduler.

A request starts in WAITING. Each engine step:

1. ``schedule()`` moves WAITING requests into the RUNNING batch while
   (a) the prefill token budget and (b) the KV block budget allow it.
2. RUNNING requests decode for one token; those reaching their stop
   condition are finished and their KV blocks are released.

When the running batch needs a new KV block but the pool is exhausted, the
engine preempts the most recently scheduled RUNNING requests: their KV
blocks are freed and they go back to WAITING (recompute-based preemption,
no CPU swap). The scheduler only *decides*; the engine executes block
allocation/free.
"""

from dataclasses import dataclass, field


@dataclass
class Request:
    request_id: int
    prompt_len: int
    max_new_tokens: int = 16
    prompt: str = ""
    num_generated: int = 0       # decode tokens produced so far
    status: str = "WAITING"      # WAITING | RUNNING | FINISHED
    kv_blocks: int = 0           # KV blocks currently held (kept by engine)
    arrival_order: int = field(default=0, compare=False)


class Scheduler:
    def __init__(self, block_size=16, max_prefill_tokens=256,
                 max_running_tokens=512):
        self.block_size = block_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_running_tokens = max_running_tokens
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._arrival = 0

    # -- request lifecycle ------------------------------------------------

    def add_request(self, prompt_len, max_new_tokens=16, prompt=""):
        req = Request(request_id=len(self.finished) + len(self.running) +
                      len(self.waiting) + 1,
                      prompt=prompt, prompt_len=prompt_len,
                      max_new_tokens=max_new_tokens,
                      arrival_order=self._arrival)
        self._arrival += 1
        self.waiting.append(req)
        return req

    def has_running_requests(self):
        return bool(self.running)

    # -- per-step scheduling ---------------------------------------------

    def schedule(self, free_blocks):
        """Admit as many WAITING requests as fit, respecting budgets.

        Args:
            free_blocks: number of KV blocks currently free in the pool.

        Returns:
            (new_requests, running_requests): the admitted requests plus the
            full running batch for this step.
        """
        new_requests = []
        for candidate in list(self.waiting):
            # full lifecycle blocks: prompt + max_new_tokens
            needed = self._blocks_for(candidate.prompt_len +
                                      candidate.max_new_tokens)
            if not self._fits(candidate, needed, free_blocks):
                continue
            self.waiting.remove(candidate)
            candidate.status = "RUNNING"
            candidate.kv_blocks = needed   # full-lifecycle block budget
            self.running.append(candidate)
            new_requests.append(candidate)
        return new_requests, list(self.running)

    def preempt(self):
        """Move the newest RUNNING request back to WAITING.

        Returns the preempted request, or None if nothing to preempt.
        """
        if not self.running:
            return None
        req = self.running.pop()
        req.status = "WAITING"
        req.kv_blocks = 0
        self.waiting.insert(0, req)
        return req

    def finish(self, request):
        self.running.remove(request)
        request.status = "FINISHED"
        self.finished.append(request)

    # -- helpers ----------------------------------------------------------

    def _blocks_for(self, num_tokens):
        return (num_tokens + self.block_size - 1) // self.block_size

    def _fits(self, request, needed_blocks, free_blocks):
        """Check that admitting `request` keeps the running batch in budget.

        `needed_blocks` covers the whole generation lifecycle (prompt +
        max_new_tokens), so requests that can never finish inside the pool
        are not admitted.
        """
        prefill = sum(r.prompt_len for r in self.running) + request.prompt_len
        if prefill > self.max_prefill_tokens:
            return False
        total = sum(r.prompt_len + r.num_generated for r in self.running)
        total += request.prompt_len
        if total > self.max_running_tokens:
            return False
        held = sum(r.kv_blocks for r in self.running) + needed_blocks
        return held <= free_blocks
