"""A minimal synchronous inference engine.

One ``step()`` is one forward pass over the whole running batch:

1. admit new WAITING requests (scheduler), preempting when the KV pool is
   too small to give every running request a block for its next token;
2. prefill every request that has produced no token yet (first token comes
   from the prefill logits);
3. decode one token for every other running request;
4. finish requests that hit ``max_new_tokens`` and release their KV blocks.

Default generation is greedy (argmax) and deterministic — directly
comparable to a dense re-forward reference. ``temperature``/``top_k``/
``top_p`` switch to the sampled chain (sampler.py); speculative steps
verify against that same target distribution, so sampled output is
distribution-exact, not an approximation.
"""

import torch

from .kv_cache import CpuSwapSpace, KVBlockManager, PrefixCache
from .scheduler import Scheduler
from .sampler import sample
from .spec_decode import NgramProposer, verify_drafts_sampled


class Engine:
    def __init__(self, model, block_size=16, num_blocks=64,
                 max_prefill_tokens=256, max_running_tokens=512,
                 device=None, dtype=None, use_cuda_graph=False,
                 temperature=0.0, top_p=1.0, top_k=None,
                 speculative_tokens=0,
                 spec_method="mtp", spec_proposer=None,
                 enable_prefix_cache=False, chunked_prefill=False,
                 preemption="recompute", swap_num_blocks=0,
                 kv_cache_dtype="auto"):
        self.model = model
        self.use_cuda_graph = use_cuda_graph
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if top_k is not None and int(top_k) < 1:
            raise ValueError("top_k must be >= 1")
        self.speculative_tokens = int(speculative_tokens)
        if self.speculative_tokens < 0:
            raise ValueError("speculative_tokens must be non-negative")
        if spec_method not in ("mtp", "ngram"):
            raise ValueError("spec_method must be 'mtp' or 'ngram'")
        self.spec_method = spec_method
        self._ngram_proposer = None
        if spec_method == "ngram" and self.speculative_tokens > 0:
            # vLLM v1 keeps the proposer pluggable; spec_proposer injects a
            # replacement (tests, or a smarter proposer later)
            self._ngram_proposer = spec_proposer or NgramProposer(
                match_len=3, k=self.speculative_tokens)
        # speculative decoding statistics (vLLM v1 spec-decode metrics):
        # drafted = proposed tokens, accepted = drafts the target agreed
        # with, bonus = full-accept extra tokens, fallbacks = no-draft steps
        self.spec_stats = {"steps": 0, "drafted": 0, "accepted": 0,
                           "bonus": 0, "full_accepts": 0, "fallbacks": 0}
        if preemption not in ("recompute", "swap"):
            raise ValueError("preemption must be 'recompute' or 'swap'")
        if kv_cache_dtype not in ("auto", "int8", "fp8"):
            raise ValueError("kv_cache_dtype must be 'auto', 'int8' or 'fp8'")
        if kv_cache_dtype != "auto" and preemption == "swap":
            raise ValueError(
                "quantized KV cache is not supported with swap preemption")
        if use_cuda_graph and getattr(model, "pp_size", 1) > 1:
            raise ValueError(
                "use_cuda_graph is not supported with pipeline parallelism "
                "(stage-to-stage p2p transfers cannot be captured)")
        if chunked_prefill and use_cuda_graph:
            raise ValueError(
                "chunked_prefill is not supported with use_cuda_graph "
                "(the prefill graph ladder replays full prompts)")
        if use_cuda_graph and self.spec_method == "ngram" \
                and self.speculative_tokens > 0:
            raise ValueError(
                "ngram speculative decoding is not supported with "
                "use_cuda_graph (the verify pass is an ungraphed prefill)")
        if enable_prefix_cache:
            # Phase-7 boundary: prefix caching shares blocks across
            # sequences. The CUDA-graph capture path does not model shared
            # blocks, so it is rejected up front instead of silently
            # misbehaving. Speculative decoding DOES compose (Phase 20):
            # the verify forward derives positions from the table cursor —
            # exactly the supports_prefix_cache contract — and the
            # post-verify truncate keeps whole-block granularity.
            if use_cuda_graph:
                raise ValueError(
                    "enable_prefix_cache is not supported with "
                    "use_cuda_graph yet")
            if not getattr(model, "supports_prefix_cache", False):
                raise ValueError(
                    "model does not support prefix caching (prefill must "
                    "derive positions from the block-table cursor)")
        if preemption == "swap":
            if swap_num_blocks <= 0:
                raise ValueError("swap preemption needs swap_num_blocks > 0")
            if not getattr(model, "supports_prefix_cache", False):
                raise ValueError(
                    "swap preemption requires a fully paged model (no "
                    "non-paged cache state such as GDN recurrent state)")
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
                                 device=device, dtype=dtype,
                                 kinds=getattr(model, "kv_kinds", 2),
                                 kv_cache_dtype=kv_cache_dtype)
        self.scheduler = Scheduler(block_size=block_size,
                                   max_prefill_tokens=max_prefill_tokens,
                                   max_running_tokens=max_running_tokens,
                                   chunked_prefill=chunked_prefill)
        if self._speculative_enabled:
            # a verify forward runs k+1 tokens (k drafts + the last
            # confirmed token), so it costs the batch budget k+1 "decode
            # tokens" — vLLM v1 counts draft tokens in
            # max_num_batched_tokens the same way. Reserving the worst
            # case (fallback steps cost only 1) keeps chunks conservative.
            self.scheduler.tokens_per_decode = 1 + self.speculative_tokens
        self.prefix_cache = PrefixCache(self.kv.pool) \
            if enable_prefix_cache else None
        self.preemption = preemption
        self.swap_space = CpuSwapSpace(swap_num_blocks) \
            if preemption == "swap" else None
        self._swap_handles = {}   # request_id -> retained CPU KV payload
        self._state = {}  # request_id -> {prompt_ids, generated, table}
        self._graphs = {}      # (batch_key) -> {bucket_blocks: buf}
        self._graph_buckets = None
        self._graph_key = None  # request ids captured in the graphs

    @property
    def _speculative_enabled(self):
        """Any speculative path active (used for feature-conflict checks)."""
        if self.speculative_tokens <= 0:
            return False
        return self.spec_method == "ngram" or self._mtp_enabled

    @property
    def _mtp_enabled(self):
        """Model-internal speculation (MTP): only on supporting models."""
        return (
            self.speculative_tokens > 0
            and self.spec_method == "mtp"
            and getattr(self.model, "supports_mtp", False)
            and hasattr(self.model, "speculative_decode")
        )

    @property
    def acceptance_rate(self):
        """Accepted drafts / proposed drafts (0.0 before any draft step)."""
        return (self.spec_stats["accepted"] / self.spec_stats["drafted"]
                if self.spec_stats["drafted"] else 0.0)

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
                bool(self.scheduler.waiting) or
                bool(self.scheduler.swapped))

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
        if self.prefix_cache is not None:
            # warmup prompts are synthetic; don't let them pollute the cache
            self.prefix_cache.clear()

    @torch.no_grad()
    def step(self):
        # Inference only, but K/V written into the long-lived block pool
        # carry grad_fn unless disabled — then every step's autograd graph
        # stays alive with the pool and memory grows without bound.
        if self.swap_space is not None:
            self._swap_in()
        if self.prefix_cache is not None:
            self._evict_for_admission()
        self.scheduler.schedule(len(self.kv.pool.free_blocks))
        self._make_room_for_next_tokens()
        self._prefill_or_decode()
        return self._finish_completed()

    def step_deltas(self):
        """``step()`` plus the per-request output deltas of this step.

        Returns a list of ``(request_id, new_token_ids, finished)`` tuples —
        the tokens generated during this step, whatever path sampled them
        (single prefill, batched decode, chunked prefill completion, or the
        speculative path). This is what a streaming consumer (the async
        EngineCore) forwards to the frontend. Prefill-only steps emit
        nothing.
        """
        before = {rid: len(st["generated"])
                  for rid, st in self._state.items()}
        finished = self.step()
        finished_ids = {r.request_id for r in finished}
        deltas = []
        for rid, old_len in before.items():
            st = self._state.get(rid)
            if st is None or len(st["generated"]) <= old_len:
                continue
            deltas.append((rid, st["generated"][old_len:],
                           rid in finished_ids))
        return deltas

    def abort(self, request_id):
        """Remove a request and release its KV (vLLM v1 abort semantics).

        Works from any queue (WAITING/RUNNING/SWAPPED). Returns False if the
        request is unknown or already finished. A SWAPPED request has no GPU
        blocks left — only its CPU swap handle is dropped.
        """
        req = None
        for queue in (self.scheduler.waiting, self.scheduler.running,
                      self.scheduler.swapped):
            for r in queue:
                if r.request_id == request_id:
                    req = r
                    queue.remove(r)
                    break
            if req is not None:
                break
        if req is None:
            return False   # finished (or unknown): nothing to release
        self._swap_handles.pop(request_id, None)
        if req.status != "SWAPPED":
            # swapped-out requests already released their GPU table
            st = self._state.pop(request_id, None)
            if st is not None:
                table = st["table"]
                self._reset_model_cache(table)
                self.kv.release_table(table)
        else:
            self._state.pop(request_id, None)
        return True

    # -- internals ---------------------------------------------------------

    def _make_room_for_next_tokens(self):
        """Make KV room for every running request's next block.

        Preference order when the pool is full: (1) evict unreferenced
        prefix-cache blocks, (2) swap the newest RUNNING request out to the
        CPU swap space (progress kept), (3) recompute-preempt it back to
        WAITING (progress lost). If nothing can free a block (total demand
        exceeds capacity), stop and retry next step — otherwise
        preempt->reschedule->prefill spins forever.
        """
        for req in list(self.scheduler.running):
            if req.status != "RUNNING":
                continue
            st = self._state[req.request_id]
            if (req.num_prefilled + req.num_generated) % \
                    self.scheduler.block_size != 0:
                continue  # room left in the current block
            attempts = 0
            while not self.kv.pool.free_blocks:
                # evicting an unreferenced cache block is always cheaper
                # than preempting a running sequence
                if self.prefix_cache is not None and \
                        self.prefix_cache.evict(1):
                    continue
                if self.swap_space is not None and self.scheduler.running:
                    victim = self.scheduler.running[-1]
                    table = self._state[victim.request_id]["table"]
                    need = (table.num_tokens + self.scheduler.block_size - 1) \
                        // self.scheduler.block_size
                    if self.swap_space.can_fit(need):
                        self.scheduler.preempt_swap()
                        handle = self.swap_space.swap_out(table)
                        self._reset_model_cache(table)
                        self._swap_handles[victim.request_id] = handle
                        self.kv.release_table(table)
                        continue
                    # swap space full: fall through to recompute preemption
                victim = self.scheduler.preempt()
                if victim is None:
                    break
                table = self._state[victim.request_id]["table"]
                self._reset_model_cache(table)
                self.kv.release_table(table)
                # recompute-based preemption: restart from the prompt
                self._state[victim.request_id]["generated"] = []
                victim.num_generated = 0
                victim.num_prefilled = 0
                attempts += 1
                if attempts >= len(self.scheduler.running) + 2:
                    # could not free a block: give up this round instead of
                    # spinning (the preempted request will be re-admitted
                    # next schedule() call).
                    break

    def _prefill_or_decode(self):
        running = [r for r in self.scheduler.running if r.status == "RUNNING"]
        if not running:
            return
        prefilling = [r for r in running if r.num_prefilled < r.prompt_len]
        decode = [r for r in running
                  if r.num_prefilled >= r.prompt_len and r.num_generated > 0]
        for r in prefilling:
            self._apply_prefix_match(r)
        batchable = hasattr(self.model, "prefill_batch") and \
            hasattr(self.model, "decode_batch")
        if batchable and prefilling:
            if self.use_cuda_graph and \
                    hasattr(self.model, "capture_prefill_graph"):
                self._prefill_with_graph(prefilling)
            else:
                suffixes = [self._prefill_chunk(r) for r in prefilling]
                tables = [self._state[r.request_id]["table"] for r in prefilling]
                for r, logits in zip(prefilling,
                                     self.model.prefill_batch(suffixes, tables)):
                    self._complete_prefill_chunk(r, logits)
        else:
            for r in prefilling:
                st = self._state[r.request_id]
                logits = self.model.prefill(self._prefill_chunk(r), st["table"])
                completed = self._complete_prefill_chunk(r, logits,
                                                         sample=False)
                if self._mtp_enabled:
                    tokens = self.model.speculative_decode(
                        st["prompt_ids"], st["table"],
                        min(self.speculative_tokens, r.max_new_tokens),
                    )
                    if tokens:
                        self._append_tokens(r, tokens)
                    elif completed:
                        self._sample(r, logits)
                elif completed:
                    self._sample(r, logits)
        if self._ngram_proposer is not None and decode:
            # n-gram speculative path: verify each request's draft in its
            # own forward (a batched verify would concatenate all drafts —
            # a good optimization, out of teaching scope)
            for r in decode:
                self._ngram_verify_step(r)
        elif batchable and decode:
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
                if hasattr(self.model, "decode_with_history"):
                    history = torch.cat(
                        [st["prompt_ids"], torch.tensor(st["generated"], device=st["prompt_ids"].device)]
                    )
                    if self._mtp_enabled:
                        tokens = self.model.speculative_decode(
                            history, st["table"],
                            min(self.speculative_tokens,
                                r.max_new_tokens - r.num_generated),
                        )
                        if tokens:
                            self._append_tokens(r, tokens)
                        else:
                            self._sample(r, self.model.decode_with_history(
                                history, st["table"]
                            ))
                        continue
                    logits = self.model.decode_with_history(history, st["table"])
                else:
                    logits = self.model.decode(
                        torch.tensor([st["generated"][-1]]), st["table"])
                self._sample(r, logits)

    def _sample(self, req, logits):
        # the sampler chain (temperature / top-k / top-p) lives in
        # sampler.py so the speculative verify judges against the SAME
        # target distribution
        token = sample(logits, self.temperature, self.top_k, self.top_p)
        self._append_tokens(req, [token])

    def _append_tokens(self, req, tokens):
        """Confirm generated tokens; the ONE registration point for prefix
        caching — plain decode, MTP and ngram-verify all funnel here, so
        blocks completed by speculatively-accepted tokens are cached too.
        The invariant register() relies on (the last confirmed token's KV
        is not yet written; table.num_tokens trails len(tokens) by one)
        holds identically on every path."""
        st = self._state[req.request_id]
        remaining = req.max_new_tokens - req.num_generated
        tokens = list(tokens[:remaining])
        st["generated"].extend(int(token) for token in tokens)
        req.num_generated += len(tokens)
        if self.prefix_cache is not None:
            # every appended token may complete a new full block
            self._register_prefix_blocks(req)

    def _prefill_with_graph(self, new):
        """CUDA-graph prefill: bucket by prompt length, replay, sample."""
        tables = [self._state[r.request_id]["table"] for r in new]
        prompts = [self._state[r.request_id]["prompt_ids"] for r in new]
        max_len = max(p.shape[0] for p in prompts)
        buckets = getattr(self, "_prefill_buckets", None)
        if buckets is None or self._prefill_key != len(new):
            if buckets is not None:
                for old in buckets.values():
                    self.kv.pool.free(old.get("scratch"))
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
        # make sure the pool can cover this prefill's blocks (preempt if not)
        bs = self.scheduler.block_size
        needed = sum((p.shape[0] + bs - 1) // bs for p in prompts)
        while len(self.kv.pool.free_blocks) < needed:
            victim = self.scheduler.preempt()
            if victim is None:
                break
            table = self._state[victim.request_id]["table"]
            self._reset_model_cache(table)
            self.kv.release_table(table)
            self._state[victim.request_id]["generated"] = []
            victim.num_generated = 0
        logits = self.model.replay_prefill_graph(bucket, tables, prompts)
        for i, r in enumerate(new):
            self._sample(r, logits[i])
            tables[i].advance(prompts[i].shape[0])
            r.num_prefilled = r.prompt_len   # graph prefills the full prompt

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
        sampled = [sample(logits[i], self.temperature, self.top_k,
                          self.top_p) for i in range(len(decode))]
        for i, r in enumerate(decode):
            self._state[r.request_id]["generated"].append(sampled[i])
            r.num_generated += 1
            tables[i].advance(1)   # mirror the KV this step wrote

    # -- n-gram speculative decoding (Phase 11) -------------------------------

    def _ngram_verify_step(self, req):
        """One speculative step: draft from history, verify in one forward.

        The streaming convention makes this clean: the last confirmed token
        t_last's KV is NOT yet written (it gets written when fed), so the
        verify forward over ``[t_last, d1..dk]`` produces, at position i,
        exactly the prediction that judges draft i+1 — and at the last
        position the bonus token when every draft is accepted.

        Greedy acceptance (``temperature <= 0``): accept the longest prefix
        of drafts that matches the target's argmax. Sampled acceptance:
        ``verify_drafts_sampled`` runs the Leviathan rejection test against
        the same sampler chain the plain decode uses, so the output
        distribution is the target's EXACTLY — for n-gram's point-mass
        proposals this is accept-with-prob p(d), resample the rest.

        On full accept the final position's logits yield a bonus token
        (k+1 tokens for one forward); on rejection the mismatching
        position's logits yield the correction token, so a step never
        regresses below plain decoding. Unconfirmed draft KV is rolled
        back with ``table.truncate``.
        """
        st = self._state[req.request_id]
        table = st["table"]
        remaining = req.max_new_tokens - req.num_generated
        k = min(self.speculative_tokens, remaining - 1)
        history = st["prompt_ids"].tolist() + st["generated"]
        drafts = self._ngram_proposer.propose(history, k) if k >= 1 else []
        stats = self.spec_stats
        if not drafts:
            # nothing to propose: fall back to a plain decode step
            # (decode returns (1, V); _sample wants one row)
            logits = self.model.decode(
                torch.tensor([st["generated"][-1]]), table).view(-1)
            self._sample(req, logits)
            stats["fallbacks"] += 1
            return
        cursor0 = table.num_tokens
        inputs = torch.tensor([history[-1]] + drafts,
                              dtype=st["prompt_ids"].dtype)
        logits = self.model.prefill(inputs, table)   # (k+1, V), KV written
        if self.temperature <= 0:
            pred = logits.argmax(dim=-1).tolist()
            m = 0
            while m < len(drafts) and drafts[m] == pred[m]:
                m += 1
            if m == len(drafts):
                new_tokens = drafts + [pred[m]]   # bonus token
            else:
                new_tokens = drafts[:m] + [pred[m]]   # correction token
        else:
            new_tokens, m = verify_drafts_sampled(
                logits, drafts, self.temperature, self.top_k, self.top_p)
        if m == len(drafts):
            stats["full_accepts"] += 1
            if len(new_tokens) <= remaining:
                stats["bonus"] += 1   # the extra token survived the cap
        accepted = min(len(new_tokens), remaining)
        self._append_tokens(req, new_tokens[:accepted])
        if table.num_tokens != cursor0 + accepted:
            table.truncate(cursor0 + accepted)   # drop rejected draft KV
        stats["steps"] += 1
        stats["drafted"] += len(drafts)
        stats["accepted"] += m

    def _finish_completed(self):
        finished = []
        for req in list(self.scheduler.running):
            if req.num_generated >= req.max_new_tokens:
                self.scheduler.finish(req)
                table = self._state[req.request_id]["table"]
                self._reset_model_cache(table)
                self.kv.release_table(table)
                finished.append(req)
        return finished

    # -- prefix caching (Phase 7) -------------------------------------------

    def _apply_prefix_match(self, req):
        """Seed a not-yet-prefilled request's table from the prefix cache.

        The matched blocks are adopted by reference (refcount +1); the
        matched length becomes the request's prefill progress, so only the
        unmatched suffix goes through the model. A full-prompt match is
        capped one block short so the prefill still produces logits for the
        final position.
        """
        pc = self.prefix_cache
        if pc is None or req.num_prefilled:
            return
        st = self._state[req.request_id]
        table = st["table"]
        if table.num_tokens:
            return  # defensive: request already holds KV
        prompt = st["prompt_ids"]
        blocks, hashes, num_tok = pc.match(prompt)
        if num_tok >= prompt.shape[0]:
            num_tok -= self.scheduler.block_size
            blocks, hashes = blocks[:-1], hashes[:-1]
        if num_tok <= 0:
            return
        pc.acquire(hashes)
        pc.hits_blocks += len(hashes)
        pc.hits_tokens += num_tok
        table.blocks = list(blocks)
        table.cache_hashes = list(hashes)
        table.cache = pc
        table.num_tokens = num_tok
        req.num_prefilled = num_tok

    def _prefill_chunk(self, req):
        """The token slice this request prefills in the current step."""
        st = self._state[req.request_id]
        chunk = min(req.chunk_tokens or req.prompt_len,
                    req.prompt_len - req.num_prefilled)
        return st["prompt_ids"][req.num_prefilled:req.num_prefilled + chunk]

    def _complete_prefill_chunk(self, req, logits, sample=True):
        """Bookkeep one prefilled chunk; sample the first token on the last.

        Returns True when the prompt is now fully prefilled (its final
        logits produced the first sampled token).
        """
        chunk = min(req.chunk_tokens or req.prompt_len,
                    req.prompt_len - req.num_prefilled)
        req.num_prefilled += chunk
        if req.num_prefilled >= req.prompt_len:
            req.num_prefilled = req.prompt_len
            if sample:
                self._sample(req, logits)
            return True
        return False

    def _register_prefix_blocks(self, req):
        """Cache newly-completed full blocks of a generating sequence."""
        st = self._state[req.request_id]
        tokens = st["prompt_ids"].tolist() + st["generated"]
        self.prefix_cache.register(st["table"], tokens)

    def _evict_for_admission(self):
        """Drop LRU cache blocks until the head WAITING request fits.

        Cached-but-unreferenced blocks are invisible to the pool's free
        list; without this loop a full cache could starve admission forever.
        Eviction only touches blocks with refcount 0, so running sequences
        are never affected.
        """
        head = None
        if self.scheduler.swapped:
            head = self.scheduler.swapped[0]
        elif self.scheduler.waiting:
            head = self.scheduler.waiting[0]
        if head is None:
            return
        bs = self.scheduler.block_size
        need = (head.prompt_len + head.max_new_tokens + bs - 1) // bs
        while len(self.kv.pool.free_blocks) < need:
            if not self.prefix_cache.evict(1):
                break

    # -- CPU swap preemption (Phase 8) ---------------------------------------

    def _swap_in(self):
        """Restore swapped-out requests while the pool has room.

        Physical blocks are re-allocated locally from the retained CPU
        payload — the same "block ids never cross a worker boundary" rule
        as the PD handoff. Progress (num_prefilled/num_generated) is kept,
        so no recomputation happens.
        """
        bs = self.scheduler.block_size
        for req in list(self.scheduler.swapped):
            need = (req.prompt_len + req.max_new_tokens + bs - 1) // bs
            while len(self.kv.pool.free_blocks) < need:
                if self.prefix_cache is None or \
                        not self.prefix_cache.evict(1):
                    return
            handle = self._swap_handles.pop(req.request_id)
            transfer = self.swap_space.swap_in(handle)
            table = self.kv.create_table()
            table.import_transfer(transfer)
            self._state[req.request_id]["table"] = table
            self.scheduler.swap_in(req)
            if self.prefix_cache is not None:
                # private copies of shared blocks dedupe back onto the cache
                self._register_prefix_blocks(req)

    def _reset_model_cache(self, table):
        """Let model adapters release non-paged state (e.g. Qwen3.5 GDN)."""
        reset = getattr(self.model, "reset_cache", None)
        if reset is not None:
            reset(table)
