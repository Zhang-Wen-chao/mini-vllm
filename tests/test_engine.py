import torch

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer


def make_model(seed=0):
    torch.manual_seed(seed)
    return TinyTransformer(vocab_size=64, d_model=32, n_layers=2, n_heads=4)


def greedy_reference(model, prompt, max_new_tokens):
    """Ground truth: dense re-forward, greedy sampling, no paging."""
    model.eval()
    toks = prompt.clone()
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model.dense_forward(toks)
            nxt = int(torch.argmax(logits[-1]))
            generated.append(nxt)
            toks = torch.cat([toks, torch.tensor([nxt])])
    return generated


def run_engine(model, prompt, max_new_tokens, **engine_kwargs):
    engine = Engine(model, **engine_kwargs)
    req = engine.add_request(prompt, max_new_tokens=max_new_tokens)
    while engine.has_requests():
        engine.step()
    assert req.status == "FINISHED"
    return req, engine


def test_single_request_matches_dense_reference():
    model = make_model()
    prompt = torch.tensor([3, 15, 27, 9])
    req, engine = run_engine(model, prompt, max_new_tokens=10)
    assert req.num_generated == 10
    expected = greedy_reference(model, prompt, 10)
    assert engine.output(req)[len(prompt):] == expected


def test_two_requests_batch_matches_reference():
    model = make_model()
    engine = Engine(model)
    p1 = torch.tensor([3, 15, 27, 9])
    p2 = torch.tensor([42, 7, 7, 42, 33])
    r1 = engine.add_request(p1, max_new_tokens=8)
    r2 = engine.add_request(p2, max_new_tokens=8)
    while engine.has_requests():
        engine.step()
    e1 = greedy_reference(model, p1, 8)
    e2 = greedy_reference(model, p2, 8)
    assert engine.output(r1)[len(p1):] == e1
    assert engine.output(r2)[len(p2):] == e2


def test_kv_blocks_are_released_after_finish():
    model = make_model()
    engine = Engine(model, num_blocks=8)
    assert len(engine.kv.pool.free_blocks) == 8
    r = engine.add_request(torch.tensor([1, 2, 3]), max_new_tokens=4)
    while engine.has_requests():
        engine.step()
    assert len(engine.kv.pool.free_blocks) == 8
    assert engine.output(r) == [1, 2, 3] + greedy_reference(
        model, torch.tensor([1, 2, 3]), 4)


def test_preemption_preserves_correctness():
    # A small KV pool forces preemption mid-generation; greedy recompute from
    # the prompt must still reproduce the dense reference exactly.
    model = make_model()
    engine = Engine(model, block_size=2, num_blocks=6)
    prompts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    reqs = [engine.add_request(p, max_new_tokens=6) for p in prompts]
    preempted = False
    steps = 0
    while engine.has_requests():
        assert steps < 200, "engine failed to make progress (livelock)"
        engine.step()
        steps += 1
        for r in engine.scheduler.waiting:
            if r.request_id in [q.request_id for q in reqs]:
                preempted = True
    assert preempted, "test setup should force at least one preemption"
    for p, r in zip(prompts, reqs):
        expected = greedy_reference(model, p, 6)
        assert engine.output(r)[len(p):] == expected
