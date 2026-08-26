import copy
import os

import pytest
import torch

from mini_vllm.kv_cache import KVBlockManager
from mini_vllm.model_runner import TinyTransformer
from mini_vllm.pd import (
    DecodeWorker,
    LogicalPDEngine,
    PDRequestSpec,
    PrefillWorker,
    _handoff_from_wire,
    _handoff_to_wire,
    run_two_process_pd,
)


def make_model(seed=0):
    torch.manual_seed(seed)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=2, n_heads=4).eval()


def greedy_reference(model, prompt, max_new_tokens):
    tokens = prompt.clone()
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            generated.append(int(torch.argmax(model.dense_forward(tokens)[-1])))
            tokens = torch.cat((tokens, torch.tensor([generated[-1]])))
    return generated


def test_kv_transfer_restores_multilayer_cross_block_cache():
    source = KVBlockManager(
        num_blocks=8, block_size=2, num_heads=2, head_dim=3, num_layers=2
    )
    table = source.create_table()
    expected = []
    for layer in range(2):
        k = torch.arange(18, dtype=torch.float32).view(3, 2, 3) + 100 * layer
        v = k + 1000
        table.append(layer, k, v)
        expected.append((k, v))
    table.advance(3)
    transfer = table.export_transfer()
    assert transfer.num_blocks == 2
    assert transfer.num_tokens == 3
    source.release_table(table)
    assert len(source.pool.free_blocks) == 8

    destination = KVBlockManager(
        num_blocks=8, block_size=2, num_heads=2, head_dim=3, num_layers=2
    )
    restored = destination.create_table()
    restored.import_transfer(transfer)
    assert restored.num_tokens == 3
    for layer, (expected_k, expected_v) in enumerate(expected):
        actual_k, actual_v = restored.get_kv(layer)
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)
    destination.release_table(restored)
    assert len(destination.pool.free_blocks) == 8


def test_kv_transfer_rejects_incompatible_destination_without_leaking_blocks():
    source = KVBlockManager(
        num_blocks=8, block_size=2, num_heads=2, head_dim=3, num_layers=2
    )
    table = source.create_table()
    for layer in range(2):
        values = torch.ones(3, 2, 3)
        table.append(layer, values, values)
    table.advance(3)
    transfer = table.export_transfer()

    too_small = KVBlockManager(
        num_blocks=1, block_size=2, num_heads=2, head_dim=3, num_layers=2
    )
    destination = too_small.create_table()
    with pytest.raises(RuntimeError, match="lacks blocks"):
        destination.import_transfer(transfer)
    assert destination.blocks == []
    assert len(too_small.pool.free_blocks) == 1

    wrong_dtype = KVBlockManager(
        num_blocks=8, block_size=2, num_heads=2, head_dim=3, num_layers=2,
        dtype=torch.float16,
    )
    destination = wrong_dtype.create_table()
    with pytest.raises(ValueError, match="dtype"):
        destination.import_transfer(transfer)
    assert destination.blocks == []
    assert len(wrong_dtype.pool.free_blocks) == 8


def test_logical_pd_matches_dense_and_prioritizes_active_decode():
    model = make_model()
    engine = LogicalPDEngine(model, block_size=2, num_blocks=16)
    first_prompt = torch.tensor([3, 15, 27, 9])
    first = engine.add_request(first_prompt, max_new_tokens=5)
    engine.step()  # first request enters DECODE after prefill
    assert first.status == "DECODE"

    second_prompt = torch.tensor([42, 7, 7, 42, 33])
    second = engine.add_request(second_prompt, max_new_tokens=4)
    start = len(engine.events)
    engine.step()
    events = engine.events[start:]
    assert events.index(("DECODE", first.request_id)) < events.index(
        ("PREFILL", second.request_id)
    )

    while engine.has_requests():
        engine.step()
    assert engine.output(first)[len(first_prompt):] == greedy_reference(model, first_prompt, 5)
    assert engine.output(second)[len(second_prompt):] == greedy_reference(model, second_prompt, 4)
    assert len(engine.kv.pool.free_blocks) == engine.kv.pool.num_blocks


def test_logical_pd_cancel_releases_blocks_once():
    engine = LogicalPDEngine(make_model(), block_size=2, num_blocks=8)
    request = engine.add_request(torch.tensor([1, 2, 3]), max_new_tokens=5)
    engine.step()
    assert request.status == "DECODE"
    assert len(engine.kv.pool.free_blocks) < 8
    assert engine.cancel(request)
    assert request.status == "CANCELLED"
    assert len(engine.kv.pool.free_blocks) == 8
    assert not engine.cancel(request)


def test_logical_pd_defers_new_prefill_instead_of_preempting_active_decode():
    model = make_model()
    engine = LogicalPDEngine(model, block_size=2, num_blocks=4)
    first = engine.add_request(torch.tensor([1, 2, 3]), max_new_tokens=3)
    engine.step()
    assert first.status == "DECODE"

    second = engine.add_request(torch.tensor([4, 5, 6]), max_new_tokens=3)
    engine.step()
    assert first.status == "DECODE"
    assert second.status == "WAITING"
    assert ("PREFILL", second.request_id) not in engine.events

    while engine.has_requests():
        engine.step()
    assert first.status == second.status == "FINISHED"
    assert len(engine.kv.pool.free_blocks) == engine.kv.pool.num_blocks


def test_separate_workers_rebuild_kv_and_match_monolithic_engine():
    model = make_model()
    prompt = torch.tensor([9, 1, 4, 2, 6])
    max_new_tokens = 7
    prefill = PrefillWorker(copy.deepcopy(model), block_size=2, num_blocks=16)
    decode = DecodeWorker(copy.deepcopy(model), block_size=2, num_blocks=16)
    handoff = prefill.prefill(
        PDRequestSpec(request_id=7, input_ids=prompt, max_new_tokens=max_new_tokens)
    )
    assert len(prefill.kv.pool.free_blocks) == 16
    result = decode.decode(handoff)
    assert result.request_id == 7
    assert result.handoff_bytes == handoff.transfer.payload_bytes
    assert list(result.generated) == greedy_reference(model, prompt, max_new_tokens)
    assert len(decode.kv.pool.free_blocks) == 16


def test_wire_handoff_copies_kv_payload_without_source_block_ids():
    model = make_model()
    prefill = PrefillWorker(copy.deepcopy(model), block_size=2, num_blocks=16)
    handoff = prefill.prefill(
        PDRequestSpec(8, torch.tensor([4, 1, 9, 2, 3]), 4)
    )
    wire = _handoff_to_wire(handoff)
    assert wire.cache_bytes
    assert "block" not in wire.__dict__
    restored = _handoff_from_wire(wire)
    assert restored.transfer.cache.data_ptr() != handoff.transfer.cache.data_ptr()
    assert torch.equal(restored.transfer.cache, handoff.transfer.cache)
    assert restored.prefill_device == "cpu"


def test_two_process_cpu_pd_matches_monolithic_engine():
    model = make_model()
    specs = [
        PDRequestSpec(11, torch.tensor([2, 8, 1, 3, 5]), 5),
        PDRequestSpec(12, torch.tensor([4, 4, 9]), 6),
    ]
    results = run_two_process_pd(
        model, specs, block_size=2, num_blocks=16, timeout_seconds=30
    )
    assert [result.request_id for result in results] == [11, 12]
    for spec, result in zip(specs, results):
        assert list(result.generated) == greedy_reference(
            model, spec.input_ids, spec.max_new_tokens
        )
        assert result.handoff_bytes > 0
        assert result.transport == "cpu_staged"
        assert result.prefill_device == result.decode_device == "cpu"


def test_two_process_pd_rejects_unsupported_device_before_spawning_workers():
    with pytest.raises(ValueError, match="must be CPU or CUDA"):
        run_two_process_pd(make_model(), [], prefill_device="meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_two_process_same_gpu_pd_matches_monolithic_engine():
    """Exercise independently spawned CUDA workers and CPU-staged handoff.

    Both workers intentionally use cuda:0 here.  This is a GPU correctness
    check for separate model/KV-pool ownership, not a cross-GPU bandwidth or
    PD performance benchmark.
    """
    model = make_model()
    specs = [
        PDRequestSpec(21, torch.tensor([5, 1, 8, 2, 4]), 5),
        PDRequestSpec(22, torch.tensor([7, 3, 6]), 4),
    ]
    results = run_two_process_pd(
        model,
        specs,
        block_size=2,
        num_blocks=16,
        prefill_device="cuda:0",
        decode_device="cuda:0",
        timeout_seconds=60,
    )
    for spec, result in zip(specs, results):
        assert list(result.generated) == greedy_reference(
            model, spec.input_ids, spec.max_new_tokens
        )
        assert result.transport == "cpu_staged"
        assert result.prefill_device == result.decode_device == "cuda:0"


@pytest.mark.skipif(
    os.environ.get("RUN_CROSS_GPU_PD_TESTS") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason=(
        "set RUN_CROSS_GPU_PD_TESTS=1 on a host where cuda:0 and cuda:1 "
        "are explicitly reserved for this test"
    ),
)
def test_two_process_cross_gpu_cpu_staged_pd_matches_monolithic_engine():
    """Verify GPU 0 prefill -> CPU bytes -> GPU 1 decode semantics.

    This is deliberately opt-in so ordinary test runs never claim or consume
    a second GPU merely because it is visible.
    """
    model = make_model()
    specs = [
        PDRequestSpec(31, torch.tensor([1, 5, 9, 2, 6]), 5),
        PDRequestSpec(32, torch.tensor([4, 7, 3]), 4),
    ]
    results = run_two_process_pd(
        model,
        specs,
        block_size=2,
        num_blocks=16,
        prefill_device="cuda:0",
        decode_device="cuda:1",
        timeout_seconds=60,
    )
    for spec, result in zip(specs, results):
        assert list(result.generated) == greedy_reference(
            model, spec.input_ids, spec.max_new_tokens
        )
        assert result.transport == "cpu_staged"
        assert result.prefill_device == "cuda:0"
        assert result.decode_device == "cuda:1"
