# mini-vllm 实现计划

教学项目：用最少纯 PyTorch 代码复现 vLLM 的核心思想。
风格参考 mini-megatron（~800 行核心 + CPU 单测 + 独立 GitHub 仓库）。

## 定位

| 维度 | vLLM | mini-vllm |
|---|---|---|
| 代码量 | ~10 万行 | **~1,000 行核心** |
| 依赖 | Triton/CUDA 自定义 kernel | **纯 PyTorch + transformers（仅引擎演示用）** |
| 核心模块 | PagedAttention + 调度器 + 异步引擎 | **分块 KV cache + 连续批调度 + 同步引擎** |
| 目标 | 生产级推理框架 | 教学验证：读懂 KV cache 分页与 continuous batching |

## 核心模块

```
mini-vllm/
├── mini_vllm/
│   ├── kv_cache.py          # Phase 1: BlockPool + BlockTable + 分配器
│   ├── paged_attention.py   # Phase 2: 按块 online softmax 的注意力
│   ├── scheduler.py         # Phase 3: WAITING/RUNNING 队列 + 抢占
│   ├── engine.py            # Phase 4: 提交请求 → 调度 → prefill/decode → 采样
│   └── model_runner.py      # Phase 4: 加载 HF 小模型, 前向 + 采样
├── tests/                   # CPU 单测, pytest
│   ├── test_kv_cache.py
│   ├── test_paged_attention.py
│   └── test_scheduler.py
├── examples/                # 端到端 demo
└── README.md
```

## 设计决策

- **分块 KV cache**：预分配块池 `(num_blocks, block_size, 2, num_heads, head_dim)`，
  K/V 各一份；每个序列一张块表（逻辑块 → 物理块），按需分配、用完归还（自由链表）。
- **PagedAttention**：不写 CUDA kernel。按块逐块取出 K/V，做 **online softmax**
  逐块累积（保持 running max/sum），教学上最贴近真实 PagedAttention 的数值路径。
- **调度器**：连续批处理（continuous batching）——运行中的 batch 每步可增删；
  容量不足时**抢占最晚加入的 RUNNING 请求**回 WAITING（丢弃其 KV，重算式抢占，
  不做 CPU swap）。
- **引擎**：同步主循环 `step()`；prefill 阶段处理新请求，decode 阶段续生成。
- **不做**：CUDA kernel、异步引擎、speculative decoding、多卡 TP、CPU swap、
  CUDA graph。

## 分阶段验证

| 阶段 | 内容 | 验证方式 | 状态 |
|---|---|---|---|
| Phase 1 | 分块 KV cache（块池/块表/分配/归还） | CPU 单测 | ✅ 完成 |
| Phase 2 | PagedAttention（按块 online softmax） | 与稠密 attention 数值等价（allclose） | ✅ 完成 |
| Phase 3 | 调度器（WAITING/RUNNING/抢占） | CPU 单测：增删/容量/抢占场景 | ✅ 完成 |
| Phase 4 | 引擎 + 小模型端到端生成 | L20 上 gpt2/opt 跑通生成 | ✅ 完成 |
| Phase 5 | README + GitHub 发布 | 仓库私有已建，HF GPT-2 适配器验证 | ✅ 完成（待公开发布） |

## 实机验证结果（2026-08-14, L20）

- `distilgpt2`（6 层）：10 tokens 贪心生成，与 HF `generate` **逐 token 一致**
- `gpt2`（12 层）：15 tokens 生成，与 HF `generate` **逐 token 一致**
- 每 token 约 0.03s（未做批量化优化，纯教学）
- 调试中确认的关键坑：
  1. GPT-2 权重是 Conv1D 布局（`(in, out)` 不转置），`F.linear` 不能直接用
  2. `transpose(1,2).reshape(t, h, hd)` 会交换 head 与 token 顺序，必须用 `view/reshape` 直接切
  3. HF 5.x 的 `output_hidden_states` 末项是 `ln_f` 之后的输出

## 验收标准

- 核心模块（kv_cache/paged_attention/scheduler）零外部依赖，CPU 单测全绿
- PagedAttention 与稠密实现输出 allclose（atol=1e-4 量级）
- 引擎在 L20 上对同一 prompt 的生成结果与 HF 直接生成一致（同 seed 同采样参数）
