"""Block-based KV cache: the storage layer behind PagedAttention.

A pre-allocated pool is divided into fixed-size blocks. Each sequence owns a
*block table* (a list of physical block ids) instead of a contiguous KV
buffer. Blocks are handed out on demand and returned to a free list when a
sequence finishes, which is how memory is shared and reused across requests.
"""

from collections import deque

import torch


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
