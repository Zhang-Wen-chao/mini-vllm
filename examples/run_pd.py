"""Run the minimal prefill-decode disaggregation correctness harness.

Examples:

    python examples/run_pd.py
    python examples/run_pd.py --prefill-device cuda:0 --decode-device cuda:1

The two workers have independent model copies and independent KV pools.  The
handoff is currently CPU-staged K/V data, deliberately prioritizing an
auditable correctness contract over a production transport implementation.
When given different CUDA devices, the path is GPU -> CPU bytes -> GPU; it is
not CUDA P2P/NCCL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Match the README command: allow direct execution from a source checkout
# without requiring an editable package installation first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.model_runner import TinyTransformer
from mini_vllm.pd import PDRequestSpec, run_two_process_pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-device", default="cpu")
    parser.add_argument("--decode-device", default="cpu")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4
    ).eval()
    specs = [
        PDRequestSpec(1, torch.tensor([3, 15, 27, 9]), 6),
        PDRequestSpec(2, torch.tensor([42, 7, 7, 42, 33]), 5),
    ]
    results = run_two_process_pd(
        model,
        specs,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        prefill_device=args.prefill_device,
        decode_device=args.decode_device,
    )
    for result in results:
        print(
            f"request={result.request_id} generated={list(result.generated)} "
            f"handoff_bytes={result.handoff_bytes} "
            f"transport={result.transport} "
            f"devices={result.prefill_device}->{result.decode_device} "
            f"decode_ms={result.decode_seconds * 1e3:.2f}"
        )


if __name__ == "__main__":
    main()
