# mini-vllm

vLLM 核心思想的纯 PyTorch 教学实现：**分块 KV cache + PagedAttention + Continuous Batching**。

与 [mini-megatron](https://github.com/Zhang-Wen-chao/mini-megatron)（训练侧并行）、
[mini-deepspeed](https://github.com/Zhang-Wen-chao/mini-deepspeed)（ZeRO 分片）并列的
推理侧教学项目。

## 这是什么

| 维度 | vLLM | mini-vllm |
|---|---|---|
| 代码量 | ~10 万行 | **~700 行核心** |
| 依赖 | Triton / CUDA 自定义 kernel | **纯 PyTorch**（引擎 demo 另需 transformers） |
| 核心机制 | PagedAttention + 调度器 + 异步引擎 | 分块 KV cache + 按块 online softmax + 连续批调度 |
| 目标 | 生产级推理框架 | 读懂 KV cache 分页与 continuous batching |

## 核心模块

```
mini_vllm/
├── kv_cache.py          # BlockPool + BlockTable: 块池、块表、按需分配/归还
├── paged_attention.py   # 逐块读取 K/V + online softmax 累积 (FlashAttention 式)
├── scheduler.py         # WAITING/RUNNING 队列、预算检查、重算式抢占
├── engine.py            # 同步主循环: 调度 → prefill/decode → 采样 → 回收 KV
└── model_runner.py      # 微型 Transformer: 流式 prefill/decode + 稠密参考实现
```

### 1. 分块 KV cache

内存按固定大小块（block）预分配，每个序列通过**块表**（逻辑块 → 物理块）引用
KV，而不是一整段连续 buffer：

- 按需分配：生成到哪，块分配到哪，不预占最大长度
- 用完归还：序列结束后块回到自由链表，其他序列复用

### 2. PagedAttention

不写 CUDA kernel。逐块取出 K/V，用 **online softmax**（维护 running max/sum）
累积注意力输出——数值路径与真实 PagedAttention 一致，可用 `dense_attention`
（稠密参考实现）逐位验证。

### 3. Continuous Batching 调度器

- 每个 step 把 WAITING 请求收进 RUNNING 批（预算：prefill token 数 + KV 块数）
- RUNNING 请求每步 decode 一个 token
- KV 块不够时**抢占**最晚加入的 RUNNING 请求回 WAITING，其 KV 释放、
  从头重算（recompute 式抢占，无 CPU swap）

### 4. 引擎

一个 `step()` 完成一次完整前向：调度 → prefill（新请求，产出第一个 token）→
decode（老请求各一个 token）→ 完成回收。贪心采样、确定性输出，因此可以与
"每步全量重算"的稠密参考做逐 token 等价验证。

## 快速开始

```bash
pip install torch pytest
pytest                 # 30 个 CPU 单测, 全绿
```

端到端示例：

```bash
python examples/run_engine.py
```

## 与 vLLM 的对比（L20, 2026-08-14，公平版）

`gpt2` + 8 并发 + 贪心，fp16，vLLM 0.8.5 **V1 引擎**（修复了内存配置后满血），
双方都先 warmup（捕获成本与 init 一起排除），**在空闲 GPU 上实测**：

| 指标 | mini-vllm | vLLM V1 | 比值（mini/vllm） |
|---|---|---|---|
| TTFT（16-token prompt） | **1 ms** | 7-17 ms | 0.07-0.17x |
| TPOT | **0.2 ms** | 0.3-0.6 ms | 0.34-0.74x |
| 稳态吞吐 | **5342-5377 tok/s** | 1744-3826 tok/s | 1.40-3.07x |

**重要教训（benchmark 卫生）**：服务器 GPU 0 被其他任务占用 80%，之前所有
数字都在竞争环境下测得（vLLM 抖动 1744-4512、mini 出现 20-30ms 假慢步）。
换空闲 GPU 后 mini 数字纹丝不动（5342-5377），vLLM 仍波动——结论建立在 3 次
重复的稳定区间上。

**历史路径**（同一 workload 的演进）：35x → 5.8x → 4.4x → 4.2x → 2.9x → 1.16x
（V0+fp32，有偏差）→ **反超（V1+fp16，空闲 GPU）**。关键手段：批量前向 →
块表张量化 → 批量 scatter → CUDA graph decode（KV 长度分桶）→ **CUDA graph
prefill（prompt 长度分桶）+ 启动时 warmup 预捕获（图按 batch 大小复用）**。

**诚实保留意见**：
1. gpt2 是 124M 小模型，kernel 启动开销主导——CUDA graph 恰好吃满这个甜区；
   7B+ 模型上 vLLM 的融合 kernel 优势会显现
2. vLLM V1 跑在 NGC torch 2.6 容器里（非标准部署），可能有隐藏劣势
3. 只测了单卡、单模型、贪心采样；未覆盖大 batch、前缀共享、内存压力

**CUDA graph 的坑**（都已解决并记录）：捕获期污染 KV 池（快照/恢复）、fp16
bool mask 精度损失（必须 float(-inf)）、块表按 prompt+max_new 预留、图按
batch 大小而非请求 id 做键（否则 warmup 白做）。

对比脚本：`examples/bench_fair.py`（TTFT/TPOT/吞吐，双方同口径）。
vLLM 0.8.5 在此环境需 `VLLM_USE_V1=0` 已不需要——V1 修好可用；
`gpu_memory_utilization=0.9` 是 V1 正常工作的前提。

## 验证方法

| 层 | 验证 |
|---|---|
| KV cache | 块分配/归还/复用、跨块写入、K/V 独立、层隔离 |
| PagedAttention | 与稠密 attention 数值等价（fp64 紧公差 + fp32 松公差） |
| 调度器 | 预算边界、抢占顺序、重入 |
| 引擎 | 与"每步稠密重算"的贪心参考**逐 token 一致**（含抢占场景） |

## 设计边界（不做）

- 不写 CUDA kernel / Triton
- 无异步引擎、无 speculative decoding、无多卡 TP
- 无 CPU swap（抢占 = 丢弃 KV 重算）
- 贪心采样（无 top-k/top-p 采样器）

## 进度

- [x] Phase 1: 分块 KV cache + 单测
- [x] Phase 2: PagedAttention + 数值等价测试
- [x] Phase 3: Continuous batching 调度器 + 单测
- [x] Phase 4: 引擎 + 端到端等价验证（含抢占）
- [ ] Phase 5: HF 小模型 demo（L20 验证）
