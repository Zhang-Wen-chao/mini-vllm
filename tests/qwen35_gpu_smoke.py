import torch

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from mini_vllm.kv_cache import KVBlockManager
from mini_vllm.engine import Engine
from mini_vllm.transformers_adapter import TransformersAdapter, _Qwen35MTP


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    config = Qwen3_5TextConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        mtp_num_hidden_layers=1,
    )
    torch.manual_seed(11)
    model = Qwen3_5ForCausalLM(config).to(device).eval()
    adapter = TransformersAdapter(model)
    manager = KVBlockManager(
        8, 4, adapter.n_kv_heads, adapter.head_dim, num_layers=adapter.n_layers
    )
    table = manager.create_table()
    prompt = torch.tensor([1, 2, 3, 4], device=device)
    adapter.prefill(prompt, table)
    history = [1, 2, 3, 4]
    errors = []
    for token in [5, 6, 7]:
        history.append(token)
        ids = torch.tensor(history, device=device)
        got = adapter.decode_with_history(ids, table)
        expected = model(input_ids=ids.unsqueeze(0), use_cache=False).logits[0, -1]
        errors.append((got - expected).abs().max().item())
    print(
        f"GPU={torch.cuda.get_device_name(0)} "
        f"transformers={__import__('transformers').__version__} "
        f"errors={errors} max_error={max(errors):.3e}"
    )
    assert max(errors) < 1e-4

    seed_mtp = _Qwen35MTP(adapter._text_model, adapter._lm_head, adapter.text_config)
    mtp_state = {
        "mtp." + name: value.detach().clone()
        for name, value in seed_mtp.state_dict().items()
        if name not in {"embed_tokens.weight", "lm_head.weight"}
    }
    adapter.enable_mtp(state_dict=mtp_state)
    table = manager.create_table()
    adapter.prefill(prompt, table)
    generated = adapter.speculative_decode(prompt, table, max_tokens=3)
    history = prompt.tolist()
    for token in generated:
        target = model(
            input_ids=torch.tensor(history, device=device).unsqueeze(0),
            use_cache=False,
        ).logits[0, -1]
        assert token == int(torch.argmax(target))
        history.append(token)
    print(f"mtp_generated={generated} mtp_tokens_verified={len(generated)}")

    engine = Engine(
        adapter,
        block_size=4,
        num_blocks=16,
        device=device,
        dtype=next(model.parameters()).dtype,
        speculative_tokens=2,
    )
    request = engine.add_request(prompt, max_new_tokens=4)
    while engine.has_requests():
        engine.step()
    engine_generated = engine.output(request)[len(prompt):]
    history = prompt.tolist()
    for token in engine_generated:
        target = model(
            input_ids=torch.tensor(history, device=device).unsqueeze(0),
            use_cache=False,
        ).logits[0, -1]
        assert token == int(torch.argmax(target))
        history.append(token)
    print(f"engine_mtp_generated={engine_generated}")


if __name__ == "__main__":
    main()
