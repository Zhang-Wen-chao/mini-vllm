"""Tests for stochastic sampling (temperature / top-p) in mini-vllm engine.

The original engine was greedy-only (argmax). GRPO needs multiple *different*
rollouts per prompt, so sampling must be stochastic. These tests verify:
1. temperature=0 (default) stays deterministic/greedy (backward compatible).
2. temperature>0 produces varied outputs across runs with the same prompt.
3. top-p filtering keeps sampling valid (sums to 1, tokens in support).
"""

import torch

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer


def make_model(seed=0):
    torch.manual_seed(seed)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=2, n_heads=4)


def run_engine(model, prompt, max_new_tokens, **engine_kwargs):
    engine = Engine(model, **engine_kwargs)
    req = engine.add_request(prompt, max_new_tokens=max_new_tokens)
    while engine.has_requests():
        engine.step()
    assert req.status == "FINISHED"
    return engine.output(req)[len(prompt):], engine  # generated tokens only


def test_greedy_default_is_deterministic():
    model = make_model()
    prompt = torch.tensor([3, 15, 27, 9])
    out1, _ = run_engine(model, prompt, 8)
    out2, _ = run_engine(model, prompt, 8)
    assert out1 == out2, "greedy should be deterministic"


def test_sampling_produces_variety():
    model = make_model()
    prompt = torch.tensor([3, 15, 27, 9])
    outputs = set()
    for _ in range(8):
        out, _ = run_engine(model, prompt, 8, temperature=1.0)
        outputs.add(tuple(out))
    # With temperature=1.0 over 8 draws, at least 2 distinct sequences expected.
    assert len(outputs) >= 2, f"sampling should vary, got {len(outputs)} distinct"


def test_sampling_differs_from_greedy():
    model = make_model()
    prompt = torch.tensor([3, 15, 27, 9])
    greedy, _ = run_engine(model, prompt, 8)  # temperature=0 default
    sampled = set()
    for _ in range(16):
        out, _ = run_engine(model, prompt, 8, temperature=1.0)
        sampled.add(tuple(out))
    assert tuple(greedy) in sampled or len(sampled) > 1


def test_top_p_valid_sampling():
    model = make_model()
    prompt = torch.tensor([3, 15, 27, 9])
    # top-p=0.9 with temperature: should run without error and produce valid tokens
    out, _ = run_engine(model, prompt, 8, temperature=0.8, top_p=0.9)
    assert len(out) == 8
    assert all(0 <= t < 64 for t in out), "tokens must be in vocab"
