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

## 与 vLLM 的对比（L20, 2026-08-14）

`gpt2` + 8 个并发请求 + 贪心采样，mini-vllm vs vLLM 0.8.5（FP32）：

| 版本 | mini 吞吐 | vLLM 吞吐 | 差距 | 输出一致性 |
|---|---|---|---|---|
| 初版（逐请求串行前向） | 73.8 tok/s | ~2600 tok/s | 35x | 8/8 逐 token 一致 |
| 批量 prefill/decode | 257 tok/s | ~1500 tok/s | 5.8x | 8/8 |
| + 块表张量化 gather | 331 tok/s | ~1500 tok/s | 4.4x | 8/8 |
| + 批量 append scatter | 360 tok/s | ~1500 tok/s | 4.2x | 8/8 |
| **+ CUDA graph decode** | **522 tok/s** | ~1500 tok/s | **2.9x** | **8/8** |

**提速手段**（都已集成进引擎，全程保持逐 token 等价）：

1. **批量 prefill/decode**：整批请求合并成一次前向（每行掩码 + padded 注意力），
   73.8 → 257 tok/s（3.5x）
2. **块表张量化 gather**：`index_select` 一次取整批 KV（vLLM block_tables 同款设计）
3. **批量 append scatter**：decode 写入用一次高级索引散射代替逐请求循环
4. **CUDA graph decode**：静态缓冲 + 批量变化时重捕获；decode 步零启动开销

**CUDA graph 的三个真实坑**（都已解决并记录）：
- 捕获期（warmup+capture）会用垃圾缓冲**真实执行 scatter 污染 KV 池** → 捕获前后快照/恢复
- fp16 下 SDPA 的 **bool 型 attn_mask 有精度损失**（padding 区污染 softmax 统计），
  必须用 float(-inf) mask
- `torch.compile` 是死路（慢 100x）：动态形状 + Python 循环导致 graph 处处断点——
  这正说明了 vLLM 为何必须写自定义 PagedAttention kernel

**剩余 2.9x 差距 = 明确的优化清单**：融合 PagedAttention kernel（消除
gather/pad/mask 中间张量）、kernel 内联 softmax（消除 fp16 累积误差——fp16 下
7/8 一致，1 个近邻平局分歧由此而来）。

对比脚本：`examples/compare_vllm.py`（需要装了 vLLM 的容器；
vLLM 0.8.5 在此环境需 `VLLM_USE_V1=0` 且 transformers 固定 4.49）。

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
