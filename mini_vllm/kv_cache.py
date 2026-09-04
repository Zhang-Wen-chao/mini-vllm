"""Block-based KV cache: the storage layer behind PagedAttention.

A pre-allocated pool is divided into fixed-size blocks. Each sequence owns a
*block table* (a list of physical block ids) instead of a contiguous KV
buffer. Blocks are handed out on demand and returned to a free list when a
sequence finishes, which is how memory is shared and reused across requests.
"""

from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCacheTransfer:
    """Portable snapshot of one sequence's paged KV cache.

    Physical block ids deliberately do not cross a worker boundary: they are
    local implementation details of the source pool.  cache is a
    CPU-contiguous, block-major payload with layout
    [K/V, layer, logical_block, token_in_block, head, head_dim].  Keeping
    the payload on CPU makes the first PD implementation work between CPU
    processes as well as between different CUDA devices; a production system
    would replace this staged copy with CUDA IPC, P2P, or RDMA transport.
    """

    num_tokens: int
    block_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    cache: torch.Tensor

    @property
    def num_blocks(self):
        return self.cache.shape[2]

    @property
    def payload_bytes(self):
        return self.cache.numel() * self.cache.element_size()


class BlockPool:
    """Flat pre-allocated KV storage divided into fixed-size blocks.

    Underlying tensor layout:

        cache[kind][layer][block][token_in_block][head][head_dim]

    where kind=0 stores K and kind=1 stores V. Attention variants with a
    different per-token layout set ``kinds=1`` and reinterpret the single
    kind: MLA stores one per-token latent vector ``[c_KV ; k_R]`` as
    (heads=1, head_dim=latent+rope), so the compression is real storage,
    not bookkeeping.
    """

    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 num_layers=1, device="cpu", dtype=torch.float32, kinds=2):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.kinds = kinds
        self.cache = torch.zeros(
            kinds, num_layers, num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=dtype)
        self.free_blocks = deque(range(num_blocks))

    def allocate(self):
        """Take one free block off the free list."""
        if not self.free_blocks:
            raise RuntimeError("out of KV cache blocks")
        return self.free_blocks.pop()

    def free(self, block_id):
        """Zero out and return a block to the free list."""
        self.cache[:, :, block_id].zero_()
        self.free_blocks.append(block_id)

    def write(self, kind, layer, block_id, offset, value):
        """Write value of shape (T, H, D) into a block starting at `offset`."""
        self.cache[kind, layer, block_id, offset:offset + value.shape[0]] = value

    def gather(self, kind, layer, block_ids, num_tokens):
        """Collect one sequence's K or V across its block table.

        Returns (num_tokens, num_heads, head_dim); tokens that were never
        written are cut off via `num_tokens`.
        """
        if not block_ids:
            return self.cache.new_empty(0, self.num_heads, self.head_dim)
        blocks = torch.tensor(block_ids, dtype=torch.long, device=self.cache.device)
        gathered = self.cache[kind, layer].index_select(0, blocks)
        gathered = gathered.reshape(-1, self.num_heads, self.head_dim)
        return gathered[:num_tokens]

    def gather_block(self, kind, layer, block_id, num_tokens):
        """One physical block's rows [0, num_tokens), (num_tokens, H, D).

        The read path paged_attention uses per block. Quantized pools
        override the read family to dequantize; models and attention never
        see the storage dtype.
        """
        return self.cache[kind, layer, block_id][:num_tokens]

    def gather_batch(self, kind, layer, block_ids, num_tokens):
        """Padded batch gather: (B, nb) block ids -> (B, num_tokens, H, D).

        The read path of batched attention; rows past `num_tokens` are cut
        and masked by the caller.
        """
        b, nb = block_ids.shape
        got = self.cache[kind, layer].index_select(0, block_ids.flatten())
        return got.view(b, nb * self.block_size,
                        self.num_heads, self.head_dim)[:, :num_tokens]


class BlockTable:
    """Per-sequence mapping from logical tokens to physical KV blocks.

    Logical token position ``p`` lives in block ``blocks[p // block_size]``
    at offset ``p % block_size``.
    """

    def __init__(self, pool):
        self.pool = pool
        self.block_size = pool.block_size
        self.blocks = []          # physical block ids, in token order
        self.num_tokens = 0       # how many tokens of this sequence are stored
        # Prefix-cache bookkeeping, parallel to `blocks`:
        # cache_hashes[i] is the chain hash if block i is owned by a
        # PrefixCache (shared with other sequences), else None.
        self.cache_hashes = []
        self.cache = None         # owning PrefixCache, set on match/register

    def ensure_capacity(self, extra_tokens):
        """Allocate new blocks until `extra_tokens` more tokens fit."""
        needed = self.num_tokens + extra_tokens
        while len(self.blocks) * self.block_size < needed:
            self.blocks.append(self.pool.allocate())

    def append(self, layer, k, v):
        """Write (T, H, D) K/V for `layer` at the sequence's current position.

        Every layer of a transformer writes the same tokens, so ``append``
        does *not* advance the token cursor; call ``advance(T)`` once per
        token step after all layers have written.
        """
        num = k.shape[0]
        assert v is None or v.shape[0] == num
        self.ensure_capacity(num)
        start = self.num_tokens
        remaining = num
        k = k.clone()
        v = None if v is None else v.clone()
        while remaining > 0:
            block_id = self.blocks[start // self.block_size]
            offset = start % self.block_size
            take = min(remaining, self.block_size - offset)
            self.pool.write(0, layer, block_id, offset, k[:take])
            if v is not None:
                self.pool.write(1, layer, block_id, offset, v[:take])
            k = k[take:]
            v = None if v is None else v[take:]
            start += take
            remaining -= take

    def advance(self, num):
        """Mark `num` more tokens as stored for this sequence."""
        self.num_tokens += num

    def truncate(self, num_tokens):
        """Roll the table back to `num_tokens` (speculative KV rollback).

        The verify forward wrote K/V for draft tokens that were rejected;
        only the confirmed prefix may stay. Whole blocks that fall off are
        returned to the pool; the (possibly partially written) tail block is
        kept and simply re-filled from the new cursor.
        """
        if num_tokens > self.num_tokens:
            raise ValueError("truncate cannot extend a table")
        keep = (num_tokens + self.block_size - 1) // self.block_size
        for block_id in self.blocks[keep:]:
            self.pool.free(block_id)
        self.blocks = self.blocks[:keep]
        self.cache_hashes = self.cache_hashes[:keep]
        self.num_tokens = num_tokens

    def get_kv(self, layer):
        """Return (K, V) of shape (num_tokens, H, D) in token order."""
        k = self.pool.gather(0, layer, self.blocks, self.num_tokens)
        v = self.pool.gather(1, layer, self.blocks, self.num_tokens)
        return k, v

    def gather_kind(self, kind, layer, num_tokens=None):
        """Gather a single cache kind across the block table.

        For a standard MHA pool use :meth:`get_kv` (K and V together).
        Single-kind layouts — MLA's per-token latent vector — read their
        one and only kind through here.
        """
        if num_tokens is None:
            num_tokens = self.num_tokens
        return self.pool.gather(kind, layer, self.blocks, num_tokens)

    def export_transfer(self):
        """Copy this table into a worker-independent KV handoff payload.

        The returned transfer has no source block-id values.  A decode worker
        must allocate fresh local blocks and import it before reading K/V.  The
        clone is intentional: a handoff remains valid after the prefill worker
        releases its source table.
        """
        block_ids = torch.tensor(self.blocks, dtype=torch.long,
                                 device=self.pool.cache.device)
        if self.blocks:
            cache = self.pool.cache.index_select(2, block_ids)
        else:
            cache = self.pool.cache[:, :, :0]
        return KVCacheTransfer(
            num_tokens=self.num_tokens,
            block_size=self.block_size,
            num_layers=self.pool.num_layers,
            num_heads=self.pool.num_heads,
            head_dim=self.pool.head_dim,
            dtype=self.pool.cache.dtype,
            cache=cache.detach().to("cpu").contiguous().clone(),
        )

    def import_transfer(self, transfer):
        """Allocate local blocks and restore one KVCacheTransfer."""
        if self.blocks or self.num_tokens:
            raise RuntimeError("cannot import into a non-empty block table")
        expected = (self.block_size, self.pool.num_layers,
                    self.pool.num_heads, self.pool.head_dim)
        actual = (transfer.block_size, transfer.num_layers,
                  transfer.num_heads, transfer.head_dim)
        if actual != expected:
            raise ValueError("KV transfer shape is incompatible with destination pool")
        if transfer.cache.shape[0] != self.pool.kinds:
            raise ValueError("KV transfer kind layout is incompatible")
        expected_shape = (self.pool.kinds, self.pool.num_layers,
                          transfer.num_blocks,
                          self.block_size, self.pool.num_heads,
                          self.pool.head_dim)
        if tuple(transfer.cache.shape) != expected_shape:
            raise ValueError("KV transfer payload has an invalid shape")
        if transfer.num_tokens < 0 or transfer.num_tokens > transfer.num_blocks * self.block_size:
            raise ValueError("KV transfer has an invalid token count")
        if transfer.dtype != self.pool.cache.dtype:
            raise ValueError("KV transfer dtype is incompatible with destination pool")
        if len(self.pool.free_blocks) < transfer.num_blocks:
            raise RuntimeError("destination KV pool lacks blocks for transfer")
        for _ in range(transfer.num_blocks):
            self.blocks.append(self.pool.allocate())
        if self.blocks:
            target = torch.tensor(self.blocks, dtype=torch.long,
                                  device=self.pool.cache.device)
            self.pool.cache.index_copy_(2, target, transfer.cache.to(
                device=self.pool.cache.device, dtype=self.pool.cache.dtype))
        self.num_tokens = transfer.num_tokens

    def release(self):
        """Return all owned blocks to the pool and reset.

        Blocks owned by a PrefixCache are only dereferenced (refcount -= 1);
        the cache keeps them alive so a later request with the same prefix
        can reuse them. Everything else goes straight back to the free list.
        """
        for i, block_id in enumerate(self.blocks):
            h = self.cache_hashes[i] if i < len(self.cache_hashes) else None
            if h is not None and self.cache is not None:
                self.cache.release_hash(h)
            else:
                self.pool.free(block_id)
        self.blocks = []
        self.cache_hashes = []
        self.cache = None
        self.num_tokens = 0


class _CacheEntry:
    """One prefix-cached physical block.

    refcount counts the sequences currently *depending* on this block
    (the registering sequence included). When it drops to 0 the block is
    NOT freed — it stays resident as a cache candidate until LRU eviction
    needs the memory. That asymmetry is the whole point of prefix caching.
    """

    __slots__ = ("hash", "block_id", "refcount", "last_used")

    def __init__(self, block_hash, block_id, tick):
        self.hash = block_hash
        self.block_id = block_id
        self.refcount = 1
        self.last_used = tick


class PrefixCache:
    """Full-block prefix cache: chain hashing + refcounting + LRU eviction.

    A block's hash chains on its parent (hash((parent_hash, block_tokens))),
    so a cache hit on block i proves blocks 0..i match the prompt token by
    token. Only *full* blocks are cached: the tail block of a sequence keeps
    growing, so caching it would require copy-on-write. Full-block-only
    caching (the same choice vLLM V1 makes) makes COW unnecessary for
    greedy/sampled decode — the last block is simply never shared.

    Physical-block contents are trusted to be identical across sequences
    with identical prefix tokens: same tokens, same positions, same weights,
    deterministic kernels. That is exactly the invariant vLLM relies on.
    (Production systems use stronger hashes — e.g. xxhash of the token ids —
    and treat collisions the same way: as a correctness bug.)
    """

    def __init__(self, pool):
        self.pool = pool
        self.block_size = pool.block_size
        self._entries = {}     # hash -> _CacheEntry
        self._tick = 0         # LRU clock, bumped on every touch
        # counters for tests and benchmarks
        self.registered_blocks = 0   # new blocks inserted into the map
        self.deduped_blocks = 0      # duplicate computes collapsed onto a cached block
        self.hits_blocks = 0         # blocks adopted from cache at prefill
        self.hits_tokens = 0         # tokens served from cache at prefill
        self.evicted_blocks = 0      # blocks evicted (refcount was 0)

    # -- lookup ------------------------------------------------------------

    def match(self, token_ids):
        """Longest prefix of full blocks cached for `token_ids`.

        Returns (block_ids, hashes, num_tokens). Nothing is acquired here —
        the caller commits with :meth:`acquire` (usually after capping the
        match so at least one token still goes through the model for its
        logits).
        """
        tokens = token_ids.tolist() if torch.is_tensor(token_ids) \
            else list(token_ids)
        block_ids, hashes = [], []
        parent = None
        for i in range(len(tokens) // self.block_size):
            h = hash((parent, tuple(tokens[i * self.block_size:
                                           (i + 1) * self.block_size])))
            entry = self._entries.get(h)
            if entry is None:
                break
            block_ids.append(entry.block_id)
            hashes.append(h)
            parent = h
        return block_ids, hashes, len(hashes) * self.block_size

    def acquire(self, hashes):
        """Take a reference on each matched block (call once per sequence)."""
        for h in hashes:
            entry = self._entries[h]
            entry.refcount += 1
            self._tick += 1
            entry.last_used = self._tick

    # -- registration ------------------------------------------------------

    def register(self, table, tokens):
        """Cache every newly-completed full block of a sequence.

        `tokens` must cover the sequence's stored tokens (prompt + decoded
        output) and `table.num_tokens` marks how many of them are live.
        Blocks already registered (a matched prefix) are skipped; the hash
        chain continues from the last registered block, so registration is
        incremental — decode steps only hash the newest full block.
        """
        tokens = tokens[:table.num_tokens]
        hashes = table.cache_hashes
        while len(hashes) < len(table.blocks):
            hashes.append(None)
        registered = 0
        for h in hashes:
            if h is None:
                break
            registered += 1
        parent = hashes[registered - 1] if registered else None
        self._extend_chain(table, tokens, registered, parent)
        table.cache = self

    def _extend_chain(self, table, tokens, registered, parent):
        bs = self.block_size
        for i in range(registered, len(tokens) // bs):
            h = hash((parent, tuple(tokens[i * bs:(i + 1) * bs])))
            entry = self._entries.get(h)
            if entry is None:
                self._entries[h] = _CacheEntry(h, table.blocks[i], self._tick)
                self.registered_blocks += 1
            elif entry.block_id != table.blocks[i]:
                # Another sequence computed the same block first: adopt its
                # block, drop our duplicate. Identical tokens + positions
                # through identical weights produce identical K/V, so the
                # swap is invisible to attention.
                dup = table.blocks[i]
                table.blocks[i] = entry.block_id
                self.pool.free(dup)
                entry.refcount += 1
                self.deduped_blocks += 1
            else:
                entry.refcount += 1
            table.cache_hashes[i] = h
            self._tick += 1
            self._entries[h].last_used = self._tick
            parent = h

    # -- release / eviction ------------------------------------------------

    def release_hash(self, block_hash):
        """Drop one sequence's reference. The block stays cached."""
        entry = self._entries.get(block_hash)
        if entry is None:
            raise RuntimeError("released a hash that is not cached")
        entry.refcount -= 1
        if entry.refcount < 0:
            raise RuntimeError("prefix cache refcount went negative")
        self._tick += 1
        entry.last_used = self._tick

    def evict(self, max_blocks=1):
        """Free up to `max_blocks` LRU blocks nobody references.

        Returns how many blocks were actually freed. Cached-but-unreferenced
        blocks are invisible to the pool's free list, so the engine calls
        this before admitting new work — evicting cache is always cheaper
        than preempting a running sequence.
        """
        freed = 0
        for h, entry in sorted(self._entries.items(),
                               key=lambda kv: kv[1].last_used):
            if freed >= max_blocks:
                break
            if entry.refcount == 0:
                del self._entries[h]
                self.pool.free(entry.block_id)
                self.evicted_blocks += 1
                freed += 1
        return freed

    def clear(self):
        """Drop every entry (only safe when no sequence holds references)."""
        for entry in self._entries.values():
            if entry.refcount != 0:
                raise RuntimeError("clear() with live references")
            self.pool.free(entry.block_id)
        self._entries.clear()

    def __len__(self):
        return len(self._entries)


class CpuSwapSpace:
    """CPU-side swap space for preemption, sized in KV blocks.

    A swapped-out sequence's paged KV is copied to CPU as one compact
    payload (via ``BlockTable.export_transfer``) and retained until swap-in
    restores it into freshly allocated device blocks. Retaining whole
    payloads instead of compacting into a CPU pool trades some CPU memory
    for simplicity; the block budget still makes swap pressure explicit and
    testable — when ``can_fit`` fails, the engine falls back to recompute
    preemption.
    """

    def __init__(self, max_blocks):
        self.max_blocks = max_blocks
        self.used_blocks = 0
        self.swap_outs = 0
        self.swap_ins = 0

    def can_fit(self, num_blocks):
        return self.used_blocks + num_blocks <= self.max_blocks

    def swap_out(self, table):
        """Copy a table to CPU and reserve its blocks. Returns the handle."""
        transfer = table.export_transfer()
        if not self.can_fit(transfer.num_blocks):
            return None
        self.used_blocks += transfer.num_blocks
        self.swap_outs += 1
        return transfer

    def swap_in(self, handle):
        """Release a retained payload back for import (and free its budget)."""
        self.used_blocks -= handle.num_blocks
        self.swap_ins += 1
        return handle


class Int8KVBlockPool(BlockPool):
    """KV pool storing int8 values with per-(token, head) fp32 scales.

    vLLM's per-token KV quantization in miniature: every token's K (or V)
    vector is scaled by its own max-abs so the int8 grid spans the token's
    full dynamic range, and the scale rides next to the payload. Storage
    drops to 1 byte per value; :meth:`gather` dequantizes transparently, so
    attention and models are unchanged and only see fp32 numbers again.
    Quantization error per element is bounded by scale/2 (rounding to the
    nearest grid point).

    The trade-off being taught: scales at per-token granularity cost 4
    bytes per (token, head) — with head_dim=8 that still roughly halves
    memory (12 vs 32 bytes/token), but production fp8 KV uses the same
    trick with coarser scales for a better ratio.
    """

    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 num_layers=1, device="cpu", dtype=None, kinds=2):
        # storage dtype is int8 regardless of the model's compute dtype
        super().__init__(num_blocks, block_size, num_heads, head_dim,
                         num_layers=num_layers, device=device,
                         dtype=torch.int8, kinds=kinds)
        self.scale = torch.zeros(kinds, num_layers, num_blocks, block_size,
                                 num_heads, 1, device=device,
                                 dtype=torch.float32)

    def write(self, kind, layer, block_id, offset, value):
        v = value.float()
        s = (v.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-12)
        q = (v / s).round().clamp_(-127, 127).to(torch.int8)
        self.cache[kind, layer, block_id, offset:offset + q.shape[0]] = q
        self.scale[kind, layer, block_id, offset:offset + s.shape[0]] = s

    def gather(self, kind, layer, block_ids, num_tokens):
        if not block_ids:
            return torch.empty(0, self.num_heads, self.head_dim)
        blocks = torch.tensor(block_ids, dtype=torch.long,
                              device=self.cache.device)
        q = self.cache[kind, layer].index_select(0, blocks)
        s = self.scale[kind, layer].index_select(0, blocks)
        q = q.reshape(-1, self.num_heads, self.head_dim)[:num_tokens]
        s = s.reshape(-1, self.num_heads, 1)[:num_tokens]
        return q.float() * s

    def gather_block(self, kind, layer, block_id, num_tokens):
        q = self.cache[kind, layer, block_id][:num_tokens]
        s = self.scale[kind, layer, block_id][:num_tokens]
        return q.float() * s

    def gather_batch(self, kind, layer, block_ids, num_tokens):
        b, nb = block_ids.shape
        flat = block_ids.flatten()
        q = self.cache[kind, layer].index_select(0, flat)
        s = self.scale[kind, layer].index_select(0, flat)
        q = q.view(b, nb * self.block_size, self.num_heads,
                   self.head_dim)[:, :num_tokens]
        s = s.view(b, nb * self.block_size, self.num_heads, 1)[:, :num_tokens]
        return q.float() * s

    def free(self, block_id):
        self.cache[:, :, block_id].zero_()
        self.scale[:, :, block_id].zero_()
        self.free_blocks.append(block_id)


class Float8KVBlockPool(BlockPool):
    """KV pool storing e4m3 float8 values directly, without scales.

    The other half of the quantization story (vLLM's production
    ``kv_cache_dtype="fp8"``): e4m3 is a FLOATING format — 4 exponent bits
    give it a wide dynamic range (±448, normals down to 2^-6), so a
    value's rounding error is RELATIVE to its magnitude (half an ulp of
    the 3-bit mantissa, ≤ 2^-4 ≈ 6.25%) instead of int8's absolute
    s/2 bound. That removes the per-token scale bookkeeping entirely:
    1 byte per value, no side table. The trade is precision at the top of
    the range — values are clamped to ±448 before the cast (PyTorch would
    produce NaN on overflow), which is what a production scale factor
    prevents by keeping activations in range.

    The granularity contrast worth remembering: int8 buys range only by
    adding a scale per token (or block/tensor); e4m3's exponent bits ARE
    the per-value adaptive scale, already baked into the format.
    """

    FP8_MAX = 448.0   # largest finite e4m3 magnitude

    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 num_layers=1, device="cpu", dtype=None, kinds=2):
        # storage dtype is float8 regardless of the model's compute dtype
        super().__init__(num_blocks, block_size, num_heads, head_dim,
                         num_layers=num_layers, device=device,
                         dtype=torch.float8_e4m3fn, kinds=kinds)

    def write(self, kind, layer, block_id, offset, value):
        v = value.float().clamp_(-self.FP8_MAX, self.FP8_MAX)
        self.cache[kind, layer, block_id,
                   offset:offset + v.shape[0]] = v.to(torch.float8_e4m3fn)

    def gather(self, kind, layer, block_ids, num_tokens):
        if not block_ids:
            return torch.empty(0, self.num_heads, self.head_dim)
        blocks = torch.tensor(block_ids, dtype=torch.long,
                              device=self.cache.device)
        q = self.cache[kind, layer].index_select(0, blocks)
        return q.reshape(-1, self.num_heads,
                         self.head_dim)[:num_tokens].float()

    def gather_block(self, kind, layer, block_id, num_tokens):
        return self.cache[kind, layer, block_id][:num_tokens].float()

    def gather_batch(self, kind, layer, block_ids, num_tokens):
        b, nb = block_ids.shape
        flat = block_ids.flatten()
        q = self.cache[kind, layer].index_select(0, flat)
        return q.view(b, nb * self.block_size, self.num_heads,
                      self.head_dim)[:, :num_tokens].float()

    def free(self, block_id):
        self.cache[:, :, block_id].zero_()
        self.free_blocks.append(block_id)


class KVBlockManager:
    """Owns the block pool and hands out/recycles block tables."""

    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 num_layers=1, device="cpu", dtype=torch.float32, kinds=2,
                 kv_cache_dtype="auto"):
        pool_cls = {"int8": Int8KVBlockPool,
                    "fp8": Float8KVBlockPool}.get(kv_cache_dtype, BlockPool)
        self.pool = pool_cls(num_blocks, block_size, num_heads, head_dim,
                             num_layers=num_layers, device=device,
                             dtype=dtype, kinds=kinds)

    def create_table(self):
        return BlockTable(self.pool)

    def release_table(self, table):
        table.release()
