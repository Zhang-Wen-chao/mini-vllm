from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn

from mini_vllm.kv_cache import KVBlockManager
from mini_vllm.transformers_adapter import TransformersAdapter, _Qwen35MTP


class _FakeCache:
    def __init__(self, tokens=0):
        self.tokens = tokens


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        text = SimpleNamespace(
            model_type="qwen3_5",
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=17,
            layer_types=["linear_attention", "full_attention"] * 2,
            mtp_num_hidden_layers=1,
        )
        self.config = SimpleNamespace(text_config=text)
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList())
        )
        self.calls = []

    def forward(self, input_ids, past_key_values=None, use_cache=True,
                attention_mask=None, cache_position=None, return_dict=True):
        del use_cache, attention_mask, cache_position, return_dict
        length = input_ids.shape[1]
        self.calls.append(length)
        cache = _FakeCache((past_key_values.tokens if past_key_values else 0) + length)
        logits = torch.arange(length * self.config.text_config.vocab_size,
                              dtype=torch.float32).view(1, length, -1)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_qwen35_adapter_uses_native_incremental_cache(monkeypatch):
    # Keep the test independent of an installed transformers package.
    cache_utils = ModuleType("transformers.cache_utils")
    cache_utils.DynamicCache = _FakeCache
    transformers = ModuleType("transformers")
    transformers.cache_utils = cache_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", cache_utils)

    model = _FakeModel()
    adapter = TransformersAdapter(model)
    manager = KVBlockManager(4, 4, adapter.n_kv_heads, adapter.head_dim,
                             num_layers=adapter.n_layers)
    table = manager.create_table()

    prompt = torch.tensor([1, 2, 3])
    adapter.prefill(prompt, table)
    adapter.decode_with_history(torch.tensor([1, 2, 3, 4]), table)

    assert adapter.is_qwen35
    assert adapter.layer_types[0] == "linear_attention"
    assert adapter.mtp_num_hidden_layers == 1
    assert model.calls == [3, 1]
    assert adapter._native_caches[id(table)]["cache"].tokens == 4

    adapter.reset_cache(table)
    assert id(table) not in adapter._native_caches


def test_qwen35_real_hybrid_cache_matches_dense_forward():
    transformers = pytest.importorskip("transformers")
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    except ImportError:
        pytest.skip("installed Transformers has no Qwen3.5 implementation")

    config = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=4, mtp_num_hidden_layers=1,
    )
    torch.manual_seed(11)
    model = Qwen3_5ForCausalLM(config).eval()
    adapter = TransformersAdapter(model)
    manager = KVBlockManager(8, 4, adapter.n_kv_heads, adapter.head_dim,
                             num_layers=adapter.n_layers)
    table = manager.create_table()
    prompt = torch.tensor([1, 2, 3, 4])
    adapter.prefill(prompt, table)
    history = prompt.tolist()
    for token in [5, 6, 7]:
        history.append(token)
        got = adapter.decode_with_history(torch.tensor(history), table)
        expected = model(input_ids=torch.tensor(history).unsqueeze(0),
                         use_cache=False).logits[0, -1]
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_qwen35_mtp_speculation_matches_greedy_target():
    transformers = pytest.importorskip("transformers")
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    except ImportError:
        pytest.skip("installed Transformers has no Qwen3.5 implementation")

    config = Qwen3_5TextConfig(
        vocab_size=53, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_conv_kernel_dim=4, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=4, mtp_num_hidden_layers=1,
    )
    torch.manual_seed(19)
    model = Qwen3_5ForCausalLM(config).eval()
    adapter = TransformersAdapter(model)
    # Build a random but complete MTP state dict, as a downloaded checkpoint
    # would provide.  The target embedding/lm_head are shared references.
    seed_mtp = _Qwen35MTP(adapter._text_model, adapter._lm_head, adapter.text_config)
    mtp_state = {
        "mtp." + name: value.detach().clone()
        for name, value in seed_mtp.state_dict().items()
        if name not in {"embed_tokens.weight", "lm_head.weight"}
    }
    adapter.enable_mtp(state_dict=mtp_state)
    manager = KVBlockManager(8, 4, adapter.n_kv_heads, adapter.head_dim,
                             num_layers=adapter.n_layers)
    table = manager.create_table()
    prompt = torch.tensor([1, 2, 3, 4])
    adapter.prefill(prompt, table)
    generated = adapter.speculative_decode(prompt, table, max_tokens=3)
    history = prompt.tolist()
    for token in generated:
        expected = int(torch.argmax(model(input_ids=torch.tensor(history).unsqueeze(0),
                                          use_cache=False).logits[0, -1]))
        assert token == expected
        history.append(token)
    assert generated
    more = adapter.speculative_decode(torch.tensor(history), table, max_tokens=3)
    for token in more:
        expected = int(torch.argmax(model(input_ids=torch.tensor(history).unsqueeze(0),
                                          use_cache=False).logits[0, -1]))
        assert token == expected
        history.append(token)
