"""Continuous batching scheduler.

A request starts in WAITING. Each engine step:

1. ``schedule()`` moves WAITING requests into the RUNNING batch while
   (a) the prefill token budget and (b) the KV block budget allow it.
2. RUNNING requests decode for one token; those reaching their stop
   condition are finished and their KV blocks are released.

With ``chunked_prefill=True`` the prefill token budget becomes a per-step
compute budget (vLLM v1's ``max_num_batched_tokens`` semantics): a long
prompt is admitted with only the tokens that fit and continues across
steps, so one long prefill cannot stall the running batch's decodes.

When the running batch needs a new KV block but the pool is exhausted, the
engine preempts the most recently scheduled RUNNING requests. Two flavors:

- recompute (default, vLLM v1 semantics): blocks are freed, the request
  returns to WAITING and later restarts from the prompt;
- swap (vLLM v0 semantics): its KV moves to a CPU swap space, the request
  waits as SWAPPED keeping its progress, and is restored block by block
  once the pool has room.

The scheduler only *decides*; the engine executes block allocation/free.
"""

from dataclasses import dataclass, field


@dataclass
class Request:
    request_id: int
    prompt_len: int
    max_new_tokens: int = 16
    prompt: str = ""
    num_generated: int = 0       # decode tokens produced so far
    status: str = "WAITING"      # WAITING | RUNNING | SWAPPED | FINISHED
    kv_blocks: int = 0           # KV blocks currently held (kept by engine)
    arrival_order: int = field(default=0, compare=False)
    num_prefilled: int = 0       # prompt tokens whose KV is in the table
    chunk_tokens: int = 0        # prefill tokens to run in the current step


class Scheduler:
    def __init__(self, block_size=16, max_prefill_tokens=256,
                 max_running_tokens=512, chunked_prefill=False):
        self.block_size = block_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_running_tokens = max_running_tokens
        self.chunked_prefill = chunked_prefill
        # forward cost of one decode request per step: 1 for plain decode,
        # k+1 with speculative decoding (the verify forward runs k drafts
        # plus the last confirmed token). vLLM v1 counts draft tokens in
        # max_num_batched_tokens the same way.
        self.tokens_per_decode = 1
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.swapped: list[Request] = []
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
        """Admit WAITING requests and assign this step's prefill chunks.

        Args:
            free_blocks: number of KV blocks currently free in the pool.

        Returns:
            (new_requests, running_requests): the admitted requests plus the
            full running batch for this step.
        """
        budget = self.max_prefill_tokens
        if self.chunked_prefill:
            # vLLM v1 budget order: every running decode claims its token
            # first (k+1 with speculative verify), mid-prefill requests
            # continue with what is left (but always make >= 1 token of
            # progress per step)
            budget -= sum(self.tokens_per_decode for r in self.running
                          if r.num_prefilled >= r.prompt_len
                          and r.num_generated > 0)
        # requests still mid-prefill continue first; they own the budget
        for r in self.running:
            if r.num_prefilled < r.prompt_len:
                r.chunk_tokens = min(r.prompt_len - r.num_prefilled,
                                     max(budget, 1) if self.chunked_prefill
                                     else self.max_prefill_tokens)
                budget -= r.chunk_tokens
        budget = max(budget, 0)
        new_requests = []
        for candidate in list(self.waiting):
            # full lifecycle blocks: prompt + max_new_tokens
            needed = self._blocks_for(candidate.prompt_len +
                                      candidate.max_new_tokens)
            if self.chunked_prefill:
                chunk = min(candidate.prompt_len - candidate.num_prefilled,
                            budget)
                if chunk <= 0:
                    continue
            else:
                chunk = candidate.prompt_len
                if chunk > budget:
                    continue
            if not self._fits(candidate, needed, free_blocks, chunk):
                continue
            self.waiting.remove(candidate)
            candidate.status = "RUNNING"
            candidate.kv_blocks = needed   # full-lifecycle block budget
            candidate.chunk_tokens = chunk
            budget -= chunk
            self.running.append(candidate)
            new_requests.append(candidate)
        return new_requests, list(self.running)

    def preempt(self):
        """Move the newest RUNNING request back to WAITING (recompute).

        Returns the preempted request, or None if nothing to preempt.
        """
        if not self.running:
            return None
        req = self.running.pop()
        req.status = "WAITING"
        req.kv_blocks = 0
        req.num_prefilled = 0
        self.waiting.insert(0, req)
        return req

    def preempt_swap(self):
        """Move the newest RUNNING request to the SWAPPED queue (CPU swap).

        Progress (num_prefilled/num_generated) is kept; the engine moves
        the request's KV to the CPU swap space and frees its GPU blocks.
        """
        if not self.running:
            return None
        req = self.running.pop()
        req.status = "SWAPPED"
        req.kv_blocks = 0
        self.swapped.append(req)
        return req

    def swap_in(self, request):
        """Move a SWAPPED request back into the RUNNING batch."""
        self.swapped.remove(request)
        request.status = "RUNNING"
        self.running.append(request)

    def finish(self, request):
        self.running.remove(request)
        request.status = "FINISHED"
        self.finished.append(request)

    # -- helpers ----------------------------------------------------------

    def _blocks_for(self, num_tokens):
        return (num_tokens + self.block_size - 1) // self.block_size

    def _fits(self, request, needed_blocks, free_blocks, chunk=None):
        """Check that admitting `request` keeps the running batch in budget.

        `needed_blocks` covers the whole generation lifecycle (prompt +
        max_new_tokens), so requests that can never finish inside the pool
        are not admitted.
        """
        if chunk is None:
            chunk = request.prompt_len
        prefill = sum(r.chunk_tokens for r in self.running
                      if r.num_prefilled < r.prompt_len) + chunk
        if prefill > self.max_prefill_tokens:
            return False
        total = sum(r.prompt_len + r.num_generated for r in self.running)
        total += request.prompt_len
        if total > self.max_running_tokens:
            return False
        held = sum(r.kv_blocks for r in self.running) + needed_blocks
        return held <= free_blocks
