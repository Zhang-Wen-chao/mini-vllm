"""Prefill-decode disaggregation primitives for the teaching engine.

There are two deliberately separate layers:

* LogicalPDEngine models independent prefill/decode queues in one process.
  It keeps a single KV pool, so moving a request is metadata-only.  This lets
  tests isolate scheduling and ownership semantics before transport exists.
* PrefillWorker / DecodeWorker model the real worker boundary.  Each owns a
  different KV pool.  Handoff exports K/V data without source physical block
  ids; the decode worker allocates fresh blocks and imports the payload.

The transport is intentionally CPU-staged for portability and testability.
It is real data movement, not a shared-memory shortcut, but it is not a
production CUDA IPC, P2P, RDMA, or network transport implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
from time import perf_counter

import torch

from .kv_cache import KVBlockManager, KVCacheTransfer


@dataclass
class PDRequest:
    """One request moving through WAITING -> PREFILL -> DECODE -> terminal."""

    request_id: int
    input_ids: torch.Tensor
    max_new_tokens: int
    generated: list[int]
    status: str = "WAITING"
    reserved_blocks: int = 0


@dataclass(frozen=True)
class PDRequestSpec:
    """Router-to-prefill-worker request payload."""

    request_id: int
    input_ids: torch.Tensor
    max_new_tokens: int


@dataclass(frozen=True)
class PDHandoff:
    """Prefill-to-decode payload with portable KV and generation metadata."""

    request_id: int
    max_new_tokens: int
    generated: tuple[int, ...]
    transfer: KVCacheTransfer
    prefill_seconds: float
    prefill_device: str


@dataclass(frozen=True)
class PDResult:
    """Final result returned by a decode worker."""

    request_id: int
    generated: tuple[int, ...]
    decode_seconds: float
    handoff_bytes: int
    transport: str
    prefill_device: str
    decode_device: str


@dataclass(frozen=True)
class PDHandoffWire:
    """Queue-safe handoff representation with an explicit byte payload.

    A torch Tensor sent directly through multiprocessing.Queue may use shared
    storage as an implementation optimization.  PD handoff instead converts
    the K/V payload to bytes before enqueueing, so source and destination
    workers have an auditable copy boundary even in the CPU test harness.
    """

    request_id: int
    max_new_tokens: int
    generated: tuple[int, ...]
    prefill_seconds: float
    prefill_device: str
    num_tokens: int
    block_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype_name: str
    cache_shape: tuple[int, ...]
    cache_bytes: bytes


def _model_cache_shape(model):
    return (
        getattr(model, "n_kv_heads", model.n_heads),
        model.head_dim,
        model.n_layers,
    )


def _infer_device_and_dtype(model, device=None, dtype=None):
    parameter = next(model.parameters())
    return (
        str(parameter.device) if device is None else str(device),
        parameter.dtype if dtype is None else dtype,
    )


def _normalize_pd_device(device):
    """Validate and normalize one worker device before spawning workers.

    The CPU-staged transport works across ``cuda:0 -> cuda:1`` because the
    wire payload is materialized as host bytes.  Validation happens in the
    parent process so a typo or unavailable GPU does not leave one child
    worker waiting indefinitely for the other.
    """
    parsed = torch.device(device)
    if parsed.type == "cpu":
        return "cpu"
    if parsed.type != "cuda":
        raise ValueError(f"PD worker device must be CPU or CUDA, got {device!r}")
    if not torch.cuda.is_available():
        raise ValueError(f"PD worker requested {device!r}, but CUDA is unavailable")
    index = 0 if parsed.index is None else parsed.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"PD worker requested cuda:{index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible"
        )
    return f"cuda:{index}"


def _sample(logits):
    return int(torch.argmax(logits))


_DTYPE_BY_NAME = {
    str(dtype): dtype
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
}


def _handoff_to_wire(handoff):
    """Serialize K/V as bytes before crossing a multiprocessing queue."""
    transfer = handoff.transfer
    cache = transfer.cache.detach().to("cpu").contiguous()
    return PDHandoffWire(
        request_id=handoff.request_id,
        max_new_tokens=handoff.max_new_tokens,
        generated=handoff.generated,
        prefill_seconds=handoff.prefill_seconds,
        prefill_device=handoff.prefill_device,
        num_tokens=transfer.num_tokens,
        block_size=transfer.block_size,
        num_layers=transfer.num_layers,
        num_heads=transfer.num_heads,
        head_dim=transfer.head_dim,
        dtype_name=str(transfer.dtype),
        cache_shape=tuple(cache.shape),
        cache_bytes=cache.view(torch.uint8).numpy().tobytes(),
    )


def _handoff_from_wire(wire):
    """Restore a fresh CPU K/V tensor owned by the decode process."""
    dtype = _DTYPE_BY_NAME.get(wire.dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported KV transfer dtype: {wire.dtype_name}")
    expected_bytes = (
        torch.empty((), dtype=dtype).element_size()
        * int(torch.tensor(wire.cache_shape).prod().item())
    )
    if len(wire.cache_bytes) != expected_bytes:
        raise ValueError("KV handoff payload byte count is invalid")
    raw = torch.frombuffer(bytearray(wire.cache_bytes), dtype=torch.uint8).clone()
    cache = raw.view(dtype).reshape(wire.cache_shape)
    return PDHandoff(
        request_id=wire.request_id,
        max_new_tokens=wire.max_new_tokens,
        generated=wire.generated,
        transfer=KVCacheTransfer(
            num_tokens=wire.num_tokens,
            block_size=wire.block_size,
            num_layers=wire.num_layers,
            num_heads=wire.num_heads,
            head_dim=wire.head_dim,
            dtype=dtype,
            cache=cache,
        ),
        prefill_seconds=wire.prefill_seconds,
        prefill_device=wire.prefill_device,
    )


class LogicalPDEngine:
    """Single-process logical PD scheduler with one shared block pool.

    Decode work is deliberately executed before new prefill work in every
    step.  That does not create parallel GPU execution, but it makes the
    scheduling policy explicit and ensures a new long prompt cannot jump
    ahead of an already active decode request.
    """

    def __init__(
        self,
        model,
        block_size=16,
        num_blocks=64,
        max_prefill_tokens=256,
        device=None,
        dtype=None,
    ):
        self.max_prefill_tokens = max_prefill_tokens
        device, dtype = _infer_device_and_dtype(model, device, dtype)
        self.model = model.to(device).eval()
        num_heads, head_dim, num_layers = _model_cache_shape(model)
        self.kv = KVBlockManager(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
        self.waiting: list[PDRequest] = []
        self.prefill: list[PDRequest] = []
        self.decode: list[PDRequest] = []
        self.finished: list[PDRequest] = []
        self.cancelled: list[PDRequest] = []
        self._state = {}
        self._next_request_id = 1
        self.events: list[tuple[str, int]] = []

    def add_request(self, input_ids, max_new_tokens=16):
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a non-empty rank-1 tensor")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        request = PDRequest(
            request_id=self._next_request_id,
            input_ids=input_ids.detach().cpu().clone(),
            max_new_tokens=max_new_tokens,
            generated=[],
        )
        self._next_request_id += 1
        self.waiting.append(request)
        self._state[request.request_id] = {"table": self.kv.create_table()}
        self.events.append(("WAITING", request.request_id))
        return request

    def has_requests(self):
        return bool(self.waiting or self.prefill or self.decode)

    def output(self, request):
        return request.input_ids.tolist() + request.generated

    def cancel(self, request):
        """Cancel a live request and return its one shared table exactly once."""
        if request.status in {"FINISHED", "CANCELLED"}:
            return False
        for queue in (self.waiting, self.prefill, self.decode):
            if request in queue:
                queue.remove(request)
        self.kv.release_table(self._state[request.request_id]["table"])
        request.status = "CANCELLED"
        self.cancelled.append(request)
        self.events.append(("CANCELLED", request.request_id))
        return True

    def step(self):
        """Run decode first, then admit and process an independent prefill batch."""
        self._decode_step()
        self._admit_prefill()
        self._prefill_step()

    def _blocks_for(self, request):
        return (request.input_ids.numel() + request.max_new_tokens +
                self.kv.pool.block_size - 1) // self.kv.pool.block_size

    def _reserved_blocks(self):
        return sum(
            request.reserved_blocks
            for request in self.prefill + self.decode
        )

    def _admit_prefill(self):
        used_tokens = 0
        for request in list(self.waiting):
            blocks = self._blocks_for(request)
            if used_tokens + request.input_ids.numel() > self.max_prefill_tokens:
                continue
            if self._reserved_blocks() + blocks > self.kv.pool.num_blocks:
                continue
            self.waiting.remove(request)
            request.status = "PREFILL"
            request.reserved_blocks = blocks
            self.prefill.append(request)
            used_tokens += request.input_ids.numel()
            self.events.append(("PREFILL", request.request_id))

    def _prefill_step(self):
        for request in list(self.prefill):
            table = self._state[request.request_id]["table"]
            logits = self.model.prefill(request.input_ids, table)
            request.generated.append(_sample(logits[-1]))
            self.prefill.remove(request)
            if len(request.generated) >= request.max_new_tokens:
                self._finish(request)
            else:
                request.status = "DECODE"
                self.decode.append(request)
                self.events.append(("HANDOFF", request.request_id))

    def _decode_step(self):
        for request in list(self.decode):
            table = self._state[request.request_id]["table"]
            token = torch.tensor([request.generated[-1]], dtype=torch.long)
            logits = self.model.decode(token, table)
            request.generated.append(_sample(logits))
            self.events.append(("DECODE", request.request_id))
            if len(request.generated) >= request.max_new_tokens:
                self._finish(request)

    def _finish(self, request):
        if request in self.decode:
            self.decode.remove(request)
        if request in self.prefill:
            self.prefill.remove(request)
        self.kv.release_table(self._state[request.request_id]["table"])
        request.status = "FINISHED"
        request.reserved_blocks = 0
        self.finished.append(request)
        self.events.append(("FINISHED", request.request_id))


class PrefillWorker:
    """Owns source KV blocks and exports a portable handoff payload."""

    def __init__(
        self,
        model,
        block_size=16,
        num_blocks=64,
        device=None,
        dtype=None,
    ):
        device, dtype = _infer_device_and_dtype(model, device, dtype)
        self.model = model.to(device).eval()
        num_heads, head_dim, num_layers = _model_cache_shape(model)
        self.kv = KVBlockManager(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

    def prefill(self, spec):
        """Return a handoff and release every source block before returning."""
        if spec.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        table = self.kv.create_table()
        started = perf_counter()
        try:
            logits = self.model.prefill(spec.input_ids, table)
            generated = (_sample(logits[-1]),)
            transfer = table.export_transfer()
            return PDHandoff(
                request_id=spec.request_id,
                max_new_tokens=spec.max_new_tokens,
                generated=generated,
                transfer=transfer,
                prefill_seconds=perf_counter() - started,
                prefill_device=str(self.kv.pool.cache.device),
            )
        finally:
            self.kv.release_table(table)


class DecodeWorker:
    """Owns destination KV blocks and reconstructs them from a PDHandoff."""

    def __init__(
        self,
        model,
        block_size=16,
        num_blocks=64,
        device=None,
        dtype=None,
    ):
        device, dtype = _infer_device_and_dtype(model, device, dtype)
        self.model = model.to(device).eval()
        num_heads, head_dim, num_layers = _model_cache_shape(model)
        self.kv = KVBlockManager(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

    def decode(self, handoff):
        """Import the handoff, finish greedy decode, then release local blocks."""
        if not handoff.generated:
            raise ValueError("handoff must include the prefill token")
        table = self.kv.create_table()
        started = perf_counter()
        try:
            table.import_transfer(handoff.transfer)
            generated = list(handoff.generated)
            while len(generated) < handoff.max_new_tokens:
                token = torch.tensor([generated[-1]], dtype=torch.long)
                logits = self.model.decode(token, table)
                generated.append(_sample(logits))
            return PDResult(
                request_id=handoff.request_id,
                generated=tuple(generated),
                decode_seconds=perf_counter() - started,
                handoff_bytes=handoff.transfer.payload_bytes,
                transport="cpu_staged",
                prefill_device=handoff.prefill_device,
                decode_device=str(self.kv.pool.cache.device),
            )
        finally:
            self.kv.release_table(table)


def _prefill_process(model, worker_kwargs, request_queue, handoff_queue, device):
    """Process target; kept module-level so spawn works on macOS and Linux."""
    model = model.to(device).eval()
    worker = PrefillWorker(model, device=device, **worker_kwargs)
    try:
        while True:
            spec = request_queue.get()
            if spec is None:
                handoff_queue.put(None)
                return
            handoff_queue.put(_handoff_to_wire(worker.prefill(spec)))
    except BaseException as exc:
        handoff_queue.put(("ERROR", repr(exc)))


def _decode_process(model, worker_kwargs, handoff_queue, result_queue, device):
    """Process target for independent decode ownership."""
    model = model.to(device).eval()
    worker = DecodeWorker(model, device=device, **worker_kwargs)
    try:
        while True:
            handoff = handoff_queue.get()
            if handoff is None:
                return
            if isinstance(handoff, tuple) and handoff[:1] == ("ERROR",):
                result_queue.put(handoff)
                return
            result_queue.put(worker.decode(_handoff_from_wire(handoff)))
    except BaseException as exc:
        result_queue.put(("ERROR", repr(exc)))


def run_two_process_pd(
    model,
    specs,
    *,
    block_size=16,
    num_blocks=64,
    prefill_device="cpu",
    decode_device="cpu",
    timeout_seconds=30,
):
    """Run ordered specs through separate prefill/decode worker processes.

    ``prefill_device`` and ``decode_device`` may be different CUDA devices.
    The data plane remains deliberately CPU-staged: K/V is copied to host
    bytes before the queue handoff, then copied into newly allocated decode
    blocks on the destination device.  This is a correctness transport, not
    a CUDA P2P/NCCL performance path.
    """
    prefill_device = _normalize_pd_device(prefill_device)
    decode_device = _normalize_pd_device(decode_device)
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    handoff_queue = context.Queue()
    result_queue = context.Queue()
    kwargs = {"block_size": block_size, "num_blocks": num_blocks}
    prefill = context.Process(
        target=_prefill_process,
        args=(model, kwargs, request_queue, handoff_queue, prefill_device),
    )
    decode = context.Process(
        target=_decode_process,
        args=(model, kwargs, handoff_queue, result_queue, decode_device),
    )
    prefill.start()
    decode.start()
    for spec in specs:
        request_queue.put(spec)
    request_queue.put(None)
    results = []
    try:
        for _ in specs:
            try:
                result = result_queue.get(timeout=timeout_seconds)
            except Empty as exc:
                raise RuntimeError("PD workers did not return before timeout") from exc
            if isinstance(result, tuple) and result[:1] == ("ERROR",):
                raise RuntimeError(f"PD worker failed: {result[1]}")
            results.append(result)
    finally:
        prefill.join(timeout=timeout_seconds)
        decode.join(timeout=timeout_seconds)
        if prefill.is_alive():
            prefill.terminate()
        if decode.is_alive():
            decode.terminate()
    if prefill.exitcode not in {0, None} or decode.exitcode not in {0, None}:
        raise RuntimeError(
            f"PD worker exited unexpectedly: prefill={prefill.exitcode}, "
            f"decode={decode.exitcode}"
        )
    return results
