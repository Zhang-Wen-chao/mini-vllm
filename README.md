# mini-vllm

vLLM 核心思想的纯 PyTorch 教学实现：**分块 KV cache + PagedAttention + Continuous Batching**。

> **这是一个教学项目，不是生产级推理框架**。它用 ~700 行纯 PyTorch 复现
> vLLM 的核心机制，用于理解 KV cache 分页、连续批处理和 CUDA graph 加速；
> 不是 vLLM 的替代品，也不应在生产环境中使用。

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

## 与 vLLM 的对比（L20, 2026-08-15，公平版）

fp16、V1 引擎（满血）、双方 warmup 后计时、空闲 GPU、TTFT/TPOT 同口径：

| 模型 | 指标 | mini-vllm | vLLM V1 | 比值（mini/vllm） |
|---|---|---|---|---|
| gpt2 (124M) | 稳态吞吐 | 5342-5377 tok/s | 1744-3826 tok/s | 1.40-3.07x |
| gpt2 | TTFT / TPOT | 1 ms / 0.2 ms | 7-17 ms / 0.3-0.6 ms | 0.07x / 0.34-0.74x |
| Qwen2.5-1.5B | 吞吐 | 991 tok/s | 1266 tok/s | 0.78x |
| Qwen2.5-1.5B | TTFT / TPOT | 8 ms / 1.0 ms | 7 ms / 0.8 ms | 1.13x / 1.29x |
| **Qwen2.5-7B** | 吞吐 (b=8) | **332 tok/s** | **333 tok/s** | **1.00x（打平）** |
| Qwen2.5-7B | TTFT / TPOT | 27 ms / 3.1 ms | 46 ms / 2.9 ms | **0.57x / 1.04x** |

**完整结论（三个模型尺寸的诚实图景）**：
- 小模型（124M）：启动开销主导，CUDA graph 甜区 → mini 大幅领先
- 中模型（1.5B）：vLLM 的 kernel 优势显现 → mini 落后 ~1.3x
- **大模型（7B）：双方逼近计算瓶颈 → 完全打平**（吞吐 1.00x，TTFT mini 快 2 倍）

"大模型 vLLM 一定赢"的直觉被证伪：7B 上 mini-vllm 的 CUDA graph + 合并投影
与 vLLM 的融合 kernel 打成平手。剩余差距（1.5B 的 1.3x）是微型 matmul 的
物理性低效（batch=8 的 decode matmul 只有 ~1% 利用率），双方同受其困。

**Qwen/Llama 适配器**（`examples/hf_llama.py`）：RMSNorm + RoPE + SwiGLU +
GQA + 分页 KV，复用 HF 的 rotary 保证数值一致；合并 qkv/gateup 投影（注意
Qwen2 的 attention_bias 必须带上）。0.5B 上 3/3 逐 token 与 HF 一致。

**踩过的坑**（全部已解决并记录）：
1. `0.0 × -inf = NaN`——静态因果 mask 不能乘出来，必须 `torch.where`
2. HF `apply_rotary_pos_emb` 要 `(B, H, S, D)` 布局且 q/k 一起转
3. 合并投影的权重**逐层不同**，且 Qwen2 的 qkv 带 bias
4. prefill 图里 padded 位置的 scatter 会覆盖真实 KV → scratch 块
5. 基准卫生：GPU 0 被占 80% 时所有数字作废；跑完必须释放显存再跑对方

## 已知限制与压力测试结论（2026-08-16）

压力测试（Qwen2.5-0.5B，动态 batch、混合 max_new、显存压力）结论：

| 检查 | 结果 |
|---|---|
| 动态 batch 正确性 | 5/6 逐 token 一致；1 个 **fp16 近邻平局翻转**（logits 差 <0.04，einsum 与 SDPA 舍入不同翻转 argmax，两路径都在 fp16 精度内） |
| 显存归还 | 511/512（1 个为 prefill 图的**活跃** scratch 块，非泄漏） |
| 抢占压力（8 块池） | 24 步完成，无崩溃，无泄漏 |

**已修复的真 bug**：
1. **抢占崩溃**：调度器准入只算 prefill 块数，生成中途把池子耗尽 → 崩溃。
   修复：准入按**全生命周期块数**（prompt + max_new_tokens）判定。
2. **scratch 块泄漏**：prefill 图重捕获时旧 scratch 块不释放 → 已修。

**已知且接受的行为**：
1. **fp16 近邻平局**：不同注意力实现（einsum vs SDPA）舍入差异可翻转平局
   argmax；工业方案是 kernel 内 fp32 累积（vLLM 的做法），mini 未做。
2. **动态 batch 重捕获停顿**：请求完成/加入改变 batch 大小 → 重捕获全部图
   （~200ms/次）。图按精确 batch 大小做键；vLLM 按 batch 分桶预捕获避免
   此开销。这是下一个明确优化项（batch 分桶 + 行填充）。

## 对比脚本

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

## 可复现信息

基准测试环境（2026-08-15/16）：

- 硬件：NVIDIA L20（48 GB，Ada Lovelace），4 卡服务器中的空闲单卡
- 软件：PyTorch 2.6.0+cu124，transformers 4.49.0，vLLM 0.8.5（V1 引擎，
  `gpu_memory_utilization=0.9`），CUDA 12.4
- 方法：双方 warmup 后计时（捕获/init 成本排除）；TTFT 用单请求
  max_tokens=1 测量；TPOT = 稳态 decode 每 token 时间；贪心采样；
  同一组 prompt、同一模型权重
- 复现命令：`python examples/bench_fair.py --model Qwen/Qwen2.5-7B --batch 8`
- 结果随源码版本演进：本仓库 git 历史记录每一步改动；基准数字对应
  2026-08 提交，不代表其他硬件/软件版本下的表现
