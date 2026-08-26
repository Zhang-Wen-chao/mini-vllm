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

    where kind=0 stores K and kind=1 stores V.
    """

    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 num_layers=1, device="cpu", dtype=torch.float32):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.cache = torch.zeros(
            2, num_layers, num_blocks, block_size, num_heads, head_dim,
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
        assert v.shape[0] == num
        self.ensure_capacity(num)
        start = self.num_tokens
        remaining = num
        k = k.clone()
        v = v.clone()
        while remaining > 0:
            block_id = self.blocks[start // self.block_size]
            offset = start % self.block_size
            take = min(remaining, self.block_size - offset)
            self.pool.write(0, layer, block_id, offset, k[:take])
            self.pool.write(1, layer, block_id, offset, v[:take])
            k = k[take:]
            v = v[take:]
            start += take
            remaining -= take

    def advance(self, num):
        """Mark `num` more tokens as stored for this sequence."""
        self.num_tokens += num

    def get_kv(self, layer):
        """Return (K, V) of shape (num_tokens, H, D) in token order."""
        k = self.pool.gather(0, layer, self.blocks, self.num_tokens)
        v = self.pool.gather(1, layer, self.blocks, self.num_tokens)
        return k, v

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
        expected_shape = (2, self.pool.num_layers, transfer.num_blocks,
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
        """Return all owned blocks to the pool and reset."""
        for block_id in self.blocks:
            self.pool.free(block_id)
        self.blocks = []
        self.num_tokens = 0


class KVBlockManager:
    """Owns the block pool and hands out/recycles block tables."""

    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 num_layers=1, device="cpu", dtype=torch.float32):
        self.pool = BlockPool(num_blocks, block_size, num_heads, head_dim,
                              num_layers=num_layers, device=device, dtype=dtype)

    def create_table(self):
        return BlockTable(self.pool)

    def release_table(self, table):
        table.release()
