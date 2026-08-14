import pytest
import torch

from mini_vllm.kv_cache import BlockPool, BlockTable, KVBlockManager


def test_pool_allocate_free_reuse():
    pool = BlockPool(num_blocks=4, block_size=16, num_heads=2, head_dim=8)
    ids = [pool.allocate() for _ in range(4)]
    assert sorted(ids) == [0, 1, 2, 3]
    pool.free(ids[1])
    assert pool.allocate() == ids[1]


def test_pool_out_of_blocks_raises():
    pool = BlockPool(num_blocks=2, block_size=16, num_heads=1, head_dim=4)
    pool.allocate()
    pool.allocate()
    with pytest.raises(RuntimeError):
        pool.allocate()


def test_write_and_read_back_single_block():
    pool = BlockPool(num_blocks=1, block_size=16, num_heads=2, head_dim=8)
    block_id = pool.allocate()
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    pool.write(0, 0, block_id, 0, k)
    pool.write(1, 0, block_id, 0, v)
    assert torch.allclose(pool.gather(0, 0, [block_id], 5), k)
    assert torch.allclose(pool.gather(1, 0, [block_id], 5), v)


def test_write_spans_multiple_blocks():
    pool = BlockPool(num_blocks=4, block_size=4, num_heads=2, head_dim=8)
    table = BlockTable(pool)
    k = torch.randn(10, 2, 8)
    v = torch.randn(10, 2, 8)
    table.append(0, k, v)
    assert len(table.blocks) == 3          # 10 tokens / 4 per block
    assert table.num_tokens == 10
    k_read, v_read = table.get_kv(0)
    assert torch.allclose(k_read, k)
    assert torch.allclose(v_read, v)


def test_append_after_restart_mimics_prefill_then_decode():
    pool = BlockPool(num_blocks=4, block_size=4, num_heads=1, head_dim=4)
    table = BlockTable(pool)
    prefill_k = torch.randn(6, 1, 4)
    prefill_v = torch.randn(6, 1, 4)
    table.append(0, prefill_k, prefill_v)      # prefill: 6 tokens
    decode_k = torch.randn(1, 1, 4)
    decode_v = torch.randn(1, 1, 4)
    table.append(0, decode_k, decode_v)        # decode: +1 token
    k_read, v_read = table.get_kv(0)
    assert torch.allclose(k_read, torch.cat([prefill_k, decode_k]))
    assert torch.allclose(v_read, torch.cat([prefill_v, decode_v]))
    assert table.num_tokens == 7


def test_layers_are_isolated():
    pool = BlockPool(num_blocks=2, block_size=4, num_heads=1, head_dim=4,
                     num_layers=2)
    block_id = pool.allocate()
    pool.write(0, 1, block_id, 0, torch.ones(2, 1, 4))
    assert torch.allclose(pool.gather(0, 0, [block_id], 2),
                          torch.zeros(2, 1, 4))
    assert torch.allclose(pool.gather(0, 1, [block_id], 2),
                          torch.ones(2, 1, 4))


def test_k_and_v_storage_is_independent():
    pool = BlockPool(num_blocks=1, block_size=4, num_heads=1, head_dim=4)
    block_id = pool.allocate()
    pool.write(0, 0, block_id, 0, torch.ones(2, 1, 4))
    assert torch.allclose(pool.gather(1, 0, [block_id], 2),
                          torch.zeros(2, 1, 4))


def test_release_returns_blocks_to_pool():
    manager = KVBlockManager(num_blocks=6, block_size=4, num_heads=1, head_dim=4)
    t1 = manager.create_table()
    t1.append(0, torch.randn(9, 1, 4), torch.randn(9, 1, 4))  # 3 blocks
    t2 = manager.create_table()
    t2.append(0, torch.randn(5, 1, 4), torch.randn(5, 1, 4))  # 2 blocks
    assert len(manager.pool.free_blocks) == 1
    manager.release_table(t1)
    assert len(manager.pool.free_blocks) == 4
    assert t1.num_tokens == 0 and t1.blocks == []
    manager.release_table(t2)
    assert len(manager.pool.free_blocks) == 6


def test_freed_blocks_are_zeroed():
    pool = BlockPool(num_blocks=1, block_size=4, num_heads=1, head_dim=4)
    block_id = pool.allocate()
    pool.write(0, 0, block_id, 0, torch.ones(3, 1, 4))
    pool.free(block_id)
    reused = pool.allocate()
    assert reused == block_id
    assert torch.allclose(pool.gather(0, 0, [reused], 3),
                          torch.zeros(3, 1, 4))
