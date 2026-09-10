# mini-vllm

vLLM 核心思想的纯 PyTorch 教学实现：**分块 KV cache + PagedAttention + Continuous Batching + 前缀缓存 + Chunked Prefill + Swap 抢占 + 张量并行/流水线并行（含微批流水线与 PP×TP）+ MoE/EP（共享/细粒度专家）+ MLA（RoPE）+ 异步引擎 + 投机解码（采样版 Leviathan verify）+ 采样链（temperature/top-k/top-p）+ KV 量化（int8/fp8）+ PD 分离原型**。

> **这是一个教学项目，不是生产级推理框架**。它用 ~4,700 行纯 PyTorch 逐机制
> 复现 vLLM 的核心设计，每个功能都锚定到官方对应模块；不是 vLLM 的替代品，
> 也不应在生产环境中使用。

与 [mini-megatron](https://github.com/Zhang-Wen-chao/mini-megatron)（训练侧并行）、
[mini-deepspeed](https://github.com/Zhang-Wen-chao/mini-deepspeed)（ZeRO 分片）并列的
推理侧教学项目。

## 这是什么

| 维度 | vLLM | mini-vllm |
|---|---|---|
| 代码量 | 运行包约 90 万行（2026-09 主干实测：2,326 个 Python 文件、897,549 物理行；v1 引擎 15.6 万行） | **~4,680 行核心（持续扩展中）** |
| 依赖 | Triton / CUDA 自定义 kernel | **纯 PyTorch**（引擎 demo 另需 transformers） |
| 核心机制 | PagedAttention + 调度器 + 异步引擎 + 多卡并行 + 投机解码 | 分块 KV cache + 按块 online softmax + 连续批调度 + TP/PP（微批 + PP×TP）+ MoE/EP + MLA + 异步引擎 + 采样链 + 投机解码 |
| 目标 | 生产级推理框架 | 逐机制吃透 vLLM：能亲手复现 + 能验证正确性 |

## 核心模块

```
mini_vllm/
├── kv_cache.py          # BlockPool + BlockTable: 块池、块表、按需分配/归还、
│                        #   前缀缓存哈希/引用计数/LRU、swap 导出导入、
│                        #   Int8KVBlockPool（逐 token scale）/Float8KVBlockPool（e4m3）
├── paged_attention.py   # 逐块读取 K/V + online softmax 累积 (FlashAttention 式)
├── scheduler.py         # WAITING/RUNNING 队列、三预算准入、chunked prefill 预算顺序
│                        #   （tokens_per_decode 预留 spec verify 的 k+1 token）
├── engine.py            # 同步主循环: 调度 → prefill/decode → 采样 → 回收/抢占/swap
├── sampler.py           # temperature/top-k/top-p 采样链（引擎与投机 verify 共用）
├── spec_decode.py       # NgramProposer + Leviathan rejection sampling verify
├── tp.py                # Megatron 式张量并行: 列并行/行并行 + all-reduce
│                        #   （group 作用域，供 PP×TP 按 stage 组归约）, KV 按头切分
├── pp.py                # 流水线并行: 按层切 stage, p2p 激活, 末级采样广播 token id;
│                        #   微批流水线（GPipe 前向调度）+ PP×TP（stage 即 TP 组）
├── moe.py               # top-k MoE FFN + EP: 按 expert 分片, all-reduce/p2p combine;
│                        #   DeepSeek 式共享专家 + 细粒度专家
├── mla.py               # Multi-head Latent Attention: 压缩 latent 分页缓存 +
│                        #   absorbed 路径 + 真 RoPE（写侧转 k_R）+ MLA×TP
├── async_engine.py      # EngineCore 独立进程 + AsyncLLM 流式前端 (vLLM v1 进程架构)
├── pd.py                # 逻辑 PD 队列 + 双进程 K/V handoff 协议 +
│                        #   decode worker 准入控制（PDOutOfBlocks 类型化抢占）
├── model_runner.py      # 微型 Transformer: 流式/批量 forward + 稠密参考实现
└── transformers_adapter.py  # Qwen2/Llama/Qwen3.5(GDN 混合缓存 + MTP) 适配
```

## 机制清单（每项都有对应单测，输出与稠密参考逐 token 一致）

| 机制 | 一句话设计 | vLLM 对应 |
|---|---|---|
| 分块 KV cache | 块池 + 块表，按需分配、用完归还（自由链表） | `v1/core/kv_cache_manager.py` + `block_pool.py` |
| PagedAttention | 逐块读 K/V + online softmax 滚动累积（running max/sum） | 原始 paged-attention kernel 的算法级复现（纯 PyTorch） |
| 连续批调度 | 每 step 按三预算（prefill token / running token / KV 块）准入 | `v1/core/sched/scheduler.py` |
| 抢占 | recompute（默认）与 CPU swap 双模式；swap 保进度、满则退回 recompute | vLLM v1 recompute / v0 swap 语义 |
| 前缀缓存 | 整块链式哈希 + 引用计数 + LRU 逐出，免 copy-on-write | `kv_cache_manager.py` prefix caching |
| Chunked prefill | `max_prefill_tokens` 变每步预算：decode 优先、长 prompt 分片推进 | v1 的 `max_num_batched_tokens` |
| 张量并行（TP） | Megatron 列并行/行并行，KV 池按本地头切分，控制面 SPMD 零通信 | `v1/worker/gpu_model_runner.py` + parallel state |
| 流水线并行（PP） | 按层切 stage，激活 p2p，末级采样后广播 token id，KV 池按层切分 | `vllm/distributed/` pipeline |
| 微批流水线 + PP×TP | 大 prefill 切微批流过 stage（GPipe 前向调度）；PP×TP：stage 即 TP 组，KV 池双重切分 | `v1/worker/gpu_model_runner.py`（micro-batch） |
| 异步引擎 | EngineCore 独立进程自循环 + AsyncLLM 流式前端 + ABORT 释放 KV | `v1/engine/core.py` + `async_llm.py` |
| 采样链 | temperature → top-k → top-p 逐级重归一；概率空间过滤（被滤质量严格为零） | `v1/sample/sampler.py` |
| n-gram 投机解码 | draft 与 verify 分离：历史 n-gram 提案，一次 verify forward 验证；采样版为 Leviathan rejection sampling（输出分布逐位等于目标） | `v1/spec_decode/`（ngram_proposer） |
| MoE + EP | top-k 路由按专家分组计算；EP 切专家、all-reduce / pairwise p2p combine；共享专家 + 细粒度专家（DeepSeek-V3 式） | `model/layers/moe.py` |
| MLA | 每 token 缓存一个 `[c_KV ; k_R]` 压缩向量；absorbed 吸收路径 = explicit 数学；真 RoPE（写侧转 k_R）；MLA×TP latent 池复制 | `v1/attention/backends/mla/` |
| KV 量化 | int8 + 每 (token, head) fp32 scale（绝对误差界）；fp8 e4m3 无 scale（相对误差界） | fp8 KV cache（生产同款格式） |
| PD 分离 | 双 worker 各持独立 KV 池，按逻辑块顺序导出/导入 K/V bytes；decode worker 准入控制（放不下→类型化 PREEMPTED） | P/D disaggregation 教学原型 |

## 快速开始

```bash
pip install torch pytest
pytest                 # 162 passed + 4 skipped (4 项跳过需 CUDA 或 transformers)
```

端到端示例（全部可在 CPU 上运行）：

```bash
python examples/run_engine.py        # 同步引擎：训练小模型 + 连续批生成
python examples/run_pd.py            # PD 分离（--prefill-device/--decode-device 可选 GPU）
python examples/run_async.py         # 异步引擎：EngineCore 进程 + 流式输出
python examples/run_tp.py            # TP=2 双进程 gloo（输出 == 稠密参考）
python examples/run_pp.py            # PP=2 流水线（`run_pp.py 4` 切 4 个 stage）
```

## 机制速览

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
- RUNNING 请求每步 decode 一个 token；长 prompt 走 chunked prefill（分片推进）
- KV 块不够时抢占：默认 **recompute**（最晚加入的 RUNNING 请求回 WAITING、
  KV 释放、从头重算）；`preemption="swap"` 时先逐出无引用缓存块、swap-out 到
  CPU（保留生成进度），swap 空间满则退回 recompute

### 4. 引擎

一个 `step()` 完成一次完整前向：调度 → prefill（新请求，产出第一个 token）→
decode（老请求各一个 token）→ 完成回收。默认贪心采样、确定性输出（可显式开启
temperature / top-p），因此可以与"每步全量重算"的稠密参考做逐 token 等价验证。

### 5. Prefill–Decode 分离（PD）

mini_vllm/pd.py 在原来的混合 Engine 之外提供两层 PD 教学实现：

    逻辑 PD（单进程）：
    WAITING -> PREFILL queue -> HANDOFF -> DECODE queue -> FINISHED

    真实 worker 边界（双进程）：
    PrefillWorker 的 BlockPool
      -> 导出 logical block 顺序的 K/V payload + request metadata
      -> CPU-staged handoff
      -> DecodeWorker 在自己的 BlockPool 重新分配 block、导入 K/V 后继续 decode

物理 block_id 从不跨 worker 传递：它只在所属 KV pool 内有效。handoff 带有
request_id、已生成首 token、max_new_tokens、token 数、block shape，以及每层
实际 K/V block 数据；源 worker 可在导出后立即释放自己的块，decode worker 则在
独立池中重建 block table。

CPU 双进程和单进程逻辑路径都以“与稠密参考逐 token 一致、handoff 后源/目的 KV 块都
正确归还”为验收。2026-08-26 又在 4090D 的独立 NGC PyTorch 24.04 容器中以 GPU 0
完成同卡双进程 smoke：`tests/test_pd.py` 9/9 通过，端到端示例也通过；容器退出后 GPU
显存回到 4 MiB。可运行：

    python examples/run_pd.py
    python examples/run_pd.py --prefill-device cuda:0 --decode-device cuda:1
    RUN_CROSS_GPU_PD_TESTS=1 pytest -q tests/test_pd.py -k cross_gpu

当前 transport 是为正确性设计的 CPU staged copy，**不是** CUDA IPC、P2P、RDMA、
网络服务或性能优化实现。上述容器 smoke 只验证独立进程/独立 KV pool 的语义，不能得出
跨卡传输或 PD 吞吐性能结论。
第二条命令会执行 GPU 0 -> CPU bytes -> GPU 1 的跨 GPU 正确性路径；测试默认显式跳过，
只有两张卡均已预约且设置 `RUN_CROSS_GPU_PD_TESTS=1` 才会占用 GPU 1。

逻辑 PD 的准入策略是 **decode 优先**：KV reservation 不足时，新 prefill 请求留在
WAITING，绝不为它抢占已运行的 decode 请求。原混合 Engine 的重算式抢占尚未迁移到
PD 路径；双 worker 目前也是按请求顺序的正确性 harness，没有 worker 内 dynamic
batching、弹性扩缩容或负载均衡。

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

注：以上数字对应 2026-08 的代码状态（当时尚无前缀缓存/TP/PP 等功能）。Phase 7–20
新增机制的收益复测已在 L20 完成，见下文"Phase 7–20 收益复测"。

### 复测（2026-09-10，vLLM 0.28.0，同机同卡同口径）

原始日志未留存是上一轮的缺口；本轮补齐（`experiments/bench_fair_2026-09-10_*.log`
+ nsys/kineto 归因日志），口径与 2026-08-15 完全一致（fp16、b=8、贪心、双方 warmup）：

| 模型 | 指标 | mini-vllm | vLLM 0.28.0 | 比值 |
|---|---|---:|---:|---:|
| gpt2 (124M) | 吞吐 | 5358 tok/s | 5443 tok/s | **0.98x（打平）** |
| gpt2 | TTFT / TPOT | 1 ms / 0.2 ms | 4 ms / 0.2 ms | **0.27x / 1.05x** |
| Qwen2.5-7B | 吞吐 | 327 tok/s | 380 tok/s | **0.86x** |
| Qwen2.5-7B | TTFT / TPOT | 25 ms / 3.1 ms | 23 ms / 2.6 ms | 1.09x / 1.18x |

两个事实，一喜一忧：

- **mini 的数字两个月纹丝不动**（gpt2 5342–5377 → 5358；7B 332 → 327）——
  同机同卡可复现，测量方法稳定。
- **vLLM 0.28 把小模型 decode 的 host 路径修好了**（gpt2 1744–3826 → 5443；
  7B 333 → 380）。上表的 1.40–3.07x / 1.00x 是对 0.8.5 的历史事实，对 0.28
  已不成立；引用请锁版本。

**归因（gpt2 decode 每 step，nsys + torch.profiler 双工具交叉验证）**：
mini 墙钟 1.51 ms、kernel 合计 1.03 ms、host 缺口 0.45 ms、约 252 个 kernel、
每步 1 次整图回放、20 次 `cudaStreamSynchronize`；vLLM 墙钟 1.42 ms、
kernel 0.76 ms、host 缺口 0.62 ms、约 139 个 kernel、1 次事件同步。
**mini 快在 host 路径（图回放把 launch 抹掉，TTFT 0.27x 即此），慢在 kernel
效率**：纯 PyTorch 未融合算子，kernel 数 1.8 倍、时长多 35%。采样是其中一处：
mini 逐请求 8 次 `torch.argmax`（Half kernel，均次 9.9 µs）加 8 次 GPU→CPU
同步；vLLM 0.28 用一个 Triton `_gumbel_sample_kernel` 整批完成（Gumbel-max
统一贪心与随机采样，temp=0 时不加噪声即纯 argmax），双方采样 GPU 时间
79.5 对 8.6 µs/步（占 kernel 时间 7.7% 对 1.1%；该路径不启用 FlashInfer，
启动日志里 "Using FlashInfer" 只是可用性提示）。gpt2 上 host 优势与 kernel
劣势恰好抵消 → 0.98x；7B 上双方 kernel 占比 98.5% / 99.5%，都钉在计算墙，
host 优势归零，0.86x 全是 kernel 效率差距。剖析脚本：
`experiments/profile_decode.py`（nsys 外壳需 `--cuda-graph-trace=node`，
vLLM 需 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，坑见脚本 docstring）与
`experiments/window_kernel_breakdown.py`（按 NVTX 窗口查 sqlite 拆 kernel 名。
**坑**：`cuda_gpu_kern_sum` 报表只统计直接发射的 kernel，图回放内部的 kernel
不在其分母里——拿它算占比会把 argmax 虚报到 ~50%；窗口查询才含图内 kernel，
`window_kernels_*_gpt2_2026-09-10.json` 即其输出）。

证据链全部入仓（双方同一 venv 同一 torch）：`experiments/env_vllm_2026-09-10.txt`
（环境锁档，含 pip freeze）、`nsys_*_gpt2_stats.csv`（kernel/API 统计导出）、
`nvtx_window_*_2026-09-10.txt`（NVTX 窗口切片）、
`window_kernels_*_gpt2_2026-09-10.json`（窗口内按 kernel 名拆分，含图内
kernel 的占比口径）与 6 份运行日志；二进制轨迹（.nsys-rep/.sqlite/chrome
trace 共 ~44MB）留测量机，`profile_decode.py` 可一键重采集。

**Qwen/Llama 适配器**（`examples/hf_llama.py`）：RMSNorm + RoPE + SwiGLU +
GQA + 分页 KV，复用 HF 的 rotary 保证数值一致；合并 qkv/gateup 投影（注意
Qwen2 的 attention_bias 必须带上）。0.5B 上 3/3 逐 token 与 HF 一致。

**Qwen3.5**（`mini_vllm/transformers_adapter.py`、`examples/hf_qwen35.py`）：
Qwen3.5 的 24 个 Gated DeltaNet 线性注意力层和 8 个 full-attention 层通过
Transformers 原生 `DynamicCache` 增量执行，cache 同时保存 GDN recurrent state
和 full-attention KV。该路径目前是正确性优先的单请求/逐请求 native-cache 路径，
尚未把混合 state 分页化，也未接入 PD handoff。Qwen3.5 checkpoint 的 `mtp.*`
权重会被 HF 模型类忽略；显式传入 `--speculative-tokens N` 时，mini-vllm 会单独
加载这些权重，执行 MTP draft + target verification，并在拒绝时完整恢复 hybrid cache。
运行该路径需要支持 Qwen3.5 的 Transformers 最新版本（当前测试为 5.16.1）；
示例默认使用公开的 `Qwen/Qwen3.5-4B`，这是当前本地官方 checkpoint 中最小的
Qwen3.5 档位，Qwen3-0.6B 不是 Qwen3.5。

**踩过的坑**（全部已解决并记录）：
1. `0.0 × -inf = NaN`——静态因果 mask 不能乘出来，必须 `torch.where`
2. HF `apply_rotary_pos_emb` 要 `(B, H, S, D)` 布局且 q/k 一起转
3. 合并投影的权重**逐层不同**，且 Qwen2 的 qkv 带 bias
4. prefill 图里 padded 位置的 scatter 会覆盖真实 KV → scratch 块
5. 基准卫生：GPU 0 被占 80% 时所有数字作废；跑完必须释放显存再跑对方
6. 批路径漏写块表 `advance` → 游标停在 0，decode 覆写 prompt 的 KV，输出
   "看着像对的"但前缀缓存注册为 0（靠 `hits_tokens` 断言抓出）
7. attention scores 是 (B,T,H,S)，掩码必须广播成 (B,T,1,S)——写成
   (B,1,T,S) 会把因果掩码错放到头轴上，T==H 时静默出错
8. 推理引擎必须关 autograd：`step()` 里没套 `no_grad` 时，K/V `clone()` 进
   长寿命块池的写入和 `index_select` 读回都带着 grad_fn，整张计算图随池
   常驻——每步 decode 留 ~0.5GB，100 步后在 L20 上 OOM（41.75GB 被占，
   单步显存估计只有几十 MB）。vLLM 全程 `@torch.inference_mode()` 不是
   装饰品
9. gloo 的 send/recv 绑裸指针、读不了 GPU 显存（collective 会经主机中转，
   p2p 不会）：CUDA 张量直接 `dist.send` 直接 `writev: Bad address`——p2p
   必须 host staging；反过来 NCCL 只收发 CUDA 张量，CPU broadcast 报
   "No backend type associated with device type cpu"。sampling-token 的
   broadcast 跟随 backend 选设备；另注意 `torch.empty(lead, device=...)`
   忘写 `dtype=torch.long` 会用 float32 buffer 收 int64 broadcast，
   gloo preamble 字节数对不上直接 abort

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

`examples/bench_fair.py` 与官方 vLLM 同机同负载公平对比（同模型同权重、同
prompt、同采样、同精度 fp16、双方 warmup 后计时、空闲 GPU）：

```bash
python examples/bench_fair.py --model Qwen/Qwen2.5-7B --batch 8
```

指标口径：TTFT 用单请求 `max_tokens=1` 测；TPOT = `(e2e - TTFT) / (总生成
token 数 - 请求数)`；吞吐 = 生成 token 总数 / 稳态 e2e 时间。

**Phase 7–20 收益复测（2026-09-04，L20 实测）**：两个专用基准脚本
（`experiments/bench_prefix_ttft.py`、`experiments/bench_tp_pp.py`）在
4×L20 48GB（Ada Lovelace，PCIe 4.0、无 NVLink）、NGC PyTorch 26.01 容器
（torch 2.10.0a0）上完成。所有配置的输出都先断言等于单卡稠密贪心参考，
再报数字——每一条耗时数据同时是一次正确性验证。

*前缀缓存 TTFT*（~187M fp32 tf32 模型，8 条请求×(1024 共享前缀+64 后缀)，
逐请求 `max_new_tokens=1`，墙钟即 TTFT）：

| 场景 | 前缀缓存开 | 关 | 收益 |
|---|---|---|---|
| 共享前缀 | **14.4 ms**（`hits_tokens=7168`） | 41.5 ms | **2.89×（−65%）** |
| 全新前缀 | 41.1 ms | 41.5 ms | 1.0×（零回归，hits=0） |

*TP/PP/微批吞吐*（d_model=1536×16 层×16 头，B=8×256 prompt+32 new，greedy；
prefill_only=每请求只生成 1 token 的整批计时，即微批/锁步 prefill 路径；
NCCL 在 1g-shm 容器内需 `NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1`）：

| 配置 | 后端 | prefill_only | generation | 吞吐 |
|---|---|---|---|---|
| dense 单卡 | — | 75.7 ms | 663.3 ms | 386.0 tok/s |
| TP=2 | gloo | 231.8 ms | 1101.7 ms | 232.4 tok/s |
| TP=2 | nccl | 146.9 ms | 869.6 ms | 294.4 tok/s |
| PP=2 锁步 | gloo | 110.1 ms | 786.2 ms | 325.6 tok/s |
| PP=2 锁步 | nccl | 113.5 ms | 1261.4 ms | 203.0 tok/s |
| PP=2 微批 2 | gloo | 104.6 ms | 740.1 ms | 345.9 tok/s |
| PP=2 微批 2 | nccl | 90.1 ms | 1270.1 ms | 201.6 tok/s |
| PP=2 微批 4 | gloo | 98.3 ms | 741.0 ms | 345.5 tok/s |
| PP=2 微批 4 | nccl | 86.8 ms | 1271.1 ms | 201.4 tok/s |
| PP×TP 2×2 | gloo | 360.3 ms | 1411.8 ms | 181.3 tok/s |
| PP×TP 2×2 | nccl | 231.7 ms | 2263.3 ms | 113.1 tok/s |

诚实读法（模型太小，通信主导，并行配置全部慢于单卡稠密，这本身就是结论）：
① 多卡收益的前提是"单卡计算时间 >> 通信时间"，d=1536 的微模型不满足，数字
全部为负收益——机制正确性与通信路径耗时是本次实测的价值，不是吞吐收益；
② TP 场景 NCCL all-reduce 原生 CUDA 明显快于 gloo（294 vs 232 tok/s）；
③ PP 场景相反：decode 每步 p2p 消息只有 ~49KB，NCCL 逐 op 启动开销 dominate，
gloo 经主机回环反而更快（326 vs 203 tok/s）——通信后端没有全面赢家，消息
尺寸决定选择；④ 微批把 prefill_only 从锁步的 110/113 ms 压到 98–105/87–90 ms
（GPipe 前向重叠减少了 stage 空闲），generation 不变（decode 设计上仍是
锁步）；⑤ PP×TP 最慢：每层 all-reduce × 每步 p2p 的 collective 次数最多。

## 验证方法

| 层 | 验证 |
|---|---|
| KV cache | 块分配/归还/复用、跨块写入、K/V 独立、层隔离、前缀哈希/逐出、swap 往返、int8 绝对误差界、fp8 相对误差界（≤2^-4） |
| PagedAttention | 与稠密 attention 数值等价（fp64 紧公差 + fp32 松公差） |
| 调度器 | 预算边界、抢占顺序、重入、chunked prefill 预算顺序、swap 保进度、spec verify 的 k+1 预算预留 |
| 引擎 | 与"每步稠密重算"的贪心参考**逐 token 一致**（含抢占/swap/前缀缓存/chunked/投机场景） |
| 采样/投机 | top-k 过滤质量严格为零、point-mass rejection sampling 输出分布逐位等于目标（2 万次试验） |
| TP/PP | 切片数学单测 → ws/pp=1 等价 → 真多进程 gloo 端到端（输出=稠密参考）；微批=整批；PP×TP 池双重切分断言 |
| MoE/EP | n_experts=1 与 dense 逐位相等锚点、EP 部分和=全量、双进程 gloo 一致（all-reduce 与 p2p 双形态） |
| MLA | absorbed=explicit 吸收恒等式（真旋转下保持）、缓存内容=真实 latent 向量（hook 验证）、MLA×TP latent 池不切分 |
| 异步引擎 | 流式增量拼接=全序列、abort 不泄漏 KV、多路并发流逐 token 一致 |
| 组合矩阵 | spec×前缀缓存、spec×chunked、量化×MLA、MLA×TP、微批 PP、PP×spec、PP×TP、PD 准入抢占 |

## 设计边界（不做）

- 不写 CUDA kernel / Triton；PagedAttention 是纯 PyTorch 逐块循环
- PD transport 仅 CPU staged copy；无 CUDA IPC/P2P、RDMA、跨机或服务化部署
- 投机解码提案仅 n-gram（point-mass）；无 EAGLE/Medusa proposer（接口已留 `spec_proposer`）
- 微批流水线是 GPipe 前向调度（推理无反向，不做 1F1B 的 backward 交错）；
  微批×TP 不组合（微批仅纯 PP）
- 显式拒绝的无效组合（构造时报错，不静默错值）：量化×swap、CUDA graph×PP、
  前缀缓存×CUDA graph、ngram verify×CUDA graph 等，完整清单见 plan.md"不做"
- 其余未做：custom all-reduce、NCCL 多 GPU 实测、fp8 生产级 per-tensor/per-block
  scale（教学版 clamp 兜底）、PD 抢占后的自动重试路由

## 进度

- [x] Phase 1: 分块 KV cache + 单测
- [x] Phase 2: PagedAttention + 数值等价测试
- [x] Phase 3: Continuous batching 调度器 + 单测
- [x] Phase 4: 引擎 + 端到端等价验证（含抢占）
- [x] Phase 5: README + 仓库（私有，待公开发布）
- [x] Phase 6: PD 原型（逻辑队列 + 双进程 KV handoff；CPU + 同卡 GPU 正确性）
- [x] Phase 7: 前缀缓存（链式哈希 + 引用计数 + LRU + 注册去重）
- [x] Phase 8: Chunked prefill + CPU swap 抢占
- [x] Phase 9: 张量并行（列/行并行 + all-reduce，TP=2 双进程 gloo 端到端）
- [x] Phase 10: 异步引擎（EngineCore 进程 + AsyncLLM 流式 + abort）
- [x] Phase 11: n-gram 投机解码（draft/verify 分离 + KV 回滚）
- [x] Phase 12: MoE FFN + EP（all-reduce combine，EP=2 双进程 gloo）
- [x] Phase 13: MLA（latent 分页 + absorbed 路径 + 压缩记账）
- [x] Phase 14: 流水线并行（PP=2/3 多进程 gloo）+ KV int8 量化
- [x] Phase 16: 采样链抽取（top-k）+ 投机采样（Leviathan rejection verify）
- [x] Phase 17: MLA RoPE 旋转 + MLA×TP（latent 复制）+ fp8 e4m3 KV 池 + 量化×MLA
- [x] Phase 18: MoE 共享专家/细粒度专家 + EP pairwise p2p combine
- [x] Phase 19: PP 微批流水线（GPipe）+ PP×TP（2×2，world 4）+ PP×spec
- [x] Phase 20: 组合矩阵清障（spec×前缀缓存、spec×chunked 解锁）+ PD 准入抢占
- [x] Phase 15: 收益复测（前缀缓存 TTFT 2.89×、TP/PP/微批/PP×TP × gloo/nccl 全矩阵，见"Phase 7–20 收益复测"）

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
- 单测口径（2026-09-04，CPU 本机）：`pytest -rN` → **162 passed + 4 skipped**
  （4 项跳过 = test_pd.py CUDA 同卡/跨卡 2 项 + transformers 环境依赖 2 项）
