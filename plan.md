# mini-vllm 实现计划

教学项目：用最少纯 PyTorch 代码复现 vLLM 的核心思想。
风格参考 mini-megatron（~800 行核心 + CPU 单测 + 独立 GitHub 仓库）。

## 定位

| 维度 | vLLM | mini-vllm |
|---|---|---|
| 代码量 | 运行包约 90 万行（2026-09 主干实测，v1 引擎 15.6 万行） | **~4,680 行核心（持续扩展中）** |
| 依赖 | Triton/CUDA 自定义 kernel | **纯 PyTorch + transformers（仅引擎演示用）** |
| 核心模块 | PagedAttention + 调度器 + 异步引擎 | **分块 KV cache + 连续批调度 + 同步/异步引擎 + TP/PP + MoE/EP + MLA + 投机解码** |
| 目标 | 生产级推理框架 | 教学验证：逐机制吃透 vLLM，每个功能锚定官方对应模块 |

## 核心模块

```
mini-vllm/
├── mini_vllm/
│   ├── kv_cache.py          # Phase 1/7/8/14/17: BlockPool + BlockTable + 分配器，
│   │                        #   前缀缓存哈希/引用计数/LRU、swap 导出导入、
│   │                        #   Int8KVBlockPool（逐 token scale）/Float8KVBlockPool（e4m3 无 scale）
│   ├── paged_attention.py   # Phase 2: 按块 online softmax 的注意力
│   ├── scheduler.py         # Phase 3/8/20: WAITING/RUNNING 队列 + 抢占 + chunked
│   │                        #   prefill 预算（tokens_per_decode 预留 spec verify 的 k+1）
│   ├── engine.py            # Phase 4+: 同步主循环（采样/抢占/swap/前缀/投机/校验门）
│   ├── model_runner.py      # Phase 4: 微型 Transformer, 流式/批量/稠密三套 forward
│   ├── sampler.py           # Phase 16: temperature/top-k/top-p 采样链（引擎与投机
│   │                        #   verify 共用同一目标分布定义）
│   ├── spec_decode.py       # Phase 11/16: NgramProposer + Leviathan rejection
│   │                        #   sampling verify（point-mass 提案退化形式）
│   ├── tp.py                # Phase 9/19: Megatron 列/行并行 + all-reduce
│   │                        #   （支持 group 作用域，PP×TP 按 stage 组归约）, KV 按头切分
│   ├── pp.py                # Phase 14/19: 按层切 stage, p2p 激活, 末级采样广播;
│   │                        #   微批流水线（GPipe 前向调度）+ PP×TP（stage 即 TP 组）
│   ├── moe.py               # Phase 12/18: top-k MoE FFN + 共享/细粒度专家 +
│   │                        #   EP combine（all-reduce / pairwise p2p 双形态）
│   ├── mla.py               # Phase 13/17: latent 分页缓存 + absorbed/explicit 双路径
│   │                        #   + 真 RoPE 旋转（写侧转 k_R）+ MLA×TP（latent 复制）
│   ├── async_engine.py      # Phase 10: EngineCore 进程 + AsyncLLM 流式前端
│   ├── pd.py                # Phase 6/20: 逻辑 PD + 双进程 K/V handoff +
│   │                        #   decode worker 准入控制（PDOutOfBlocks 类型化抢占）
│   └── transformers_adapter.py  # Qwen2/Llama/Qwen3.5(GDN 混合缓存 + MTP) 适配
├── tests/                   # CPU 单测, pytest（2026-09-04: 162 passed + 4 skipped）
├── examples/                # run_engine / run_pd / run_async / run_tp / run_pp / bench_fair
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
- **前缀缓存（Phase 7）**：只缓存**整块**（部分尾部块永不缓存 → 免 copy-on-write）；
  块哈希 = `hash((父块哈希, 本块 tokens))` 链式哈希，命中第 i 块即证明前 i 块逐 token
  相等；引用计数 + LRU 逐出（refcount=0 的块不释放、留在缓存中可复用）；注册时发现
  同哈希块直接收编并释放重复块（同 batch 相同 prompt 去重）。整 prompt 命中时截掉
  最后一个整块重算，保证 prefill 仍产出末位 logits。前提：prefill 位置码从块表
  cursor 起算（`supports_prefix_cache` 能力位）。与 CUDA graph / speculative decode
  互斥（构造时显式报错，不做静默降级）。
- **Chunked prefill（Phase 8）**：`max_prefill_tokens` 从"整 prompt 准入门槛"升级为
  **每步预填充 token 预算**（对应 vLLM v1 的 `max_num_batched_tokens`）。预算顺序：
  运行中 decode 先各占 1 token，未完成 prefill 的请求保证每步至少推进 1 token，
  新请求按 FIFO 拿剩余预算的部分 chunk（`num_prefilled` 跨 step 累积）。长 prompt
  不再阻塞运行批的 decode；等待中的新请求仍按 FIFO 排队（vLLM v1 同样如此）。
- **CPU swap 抢占（Phase 8，vLLM v0 语义）**：`preemption="swap"` 时缺块的处置顺序
  为 逐出无引用缓存块 → swap-out 到 CPU（保留 num_prefilled/num_generated 进度，
  payload 经 `export_transfer` 驻留 CPU，按块计预算）→ swap 空间满则退回 recompute
  抢占（vLLM v1 语义，丢进度重算，默认）。swap-in 在本地重新分配物理块后
  `import_transfer` 恢复——物理块 id 永不跨设备，与 PD handoff 同一规则。
- **张量并行（Phase 9，Megatron 式）**：q/k/v 与 w1 **列并行**（输出维按 rank 切：
  权重行切片，w1 bias 同切），wo/w2 **行并行**（输入维切：权重列切片，各一次
  all-reduce，w2 全量 bias 在 reduce 后加——先加会双计）。KV cache 随头切分：每 rank
  的块池只存本地头（vLLM per-worker paged KV）。引擎 SPMD：各 rank 跑同一调度，
  greedy 在位相同的 logits 上采样，控制面零通信。从 dense 权重切片构建 shard，
  `nn.Linear` 权重是 `(out, in)` 布局——列并行切行、行并行切列。验证分三层：
  切片数学单测（手工局部计算+求和）→ ws=1 引擎等价 → 真双进程 gloo TP=2。
- **异步引擎（Phase 10，vLLM v1 进程架构）**：EngineCore 独立进程持有模型/KV/调度器
  并自循环（每轮非阻塞取控制命令 → 有请求就 `step_deltas()` 推增量，空闲休眠）；
  AsyncLLM 前端用 pump 线程读输出队列，经 `call_soon_threadsafe` 路由到每请求的
  `asyncio.Queue`（vLLM 用 zmq + 事件循环直读，路由问题相同）。请求跨界即
  token id 纯数据；`generate()` 是异步生成器，提前退出经 GeneratorExit→finally 发
  ABORT，core 释放 KV（vLLM v1 abort 语义）。`step_deltas()` 用步前快照差分，
  与采样路径（单条/批量/chunked/speculative）解耦。坑：`mp.Queue` 的 pickle 在
  feeder 线程静默失败——不可 pickle 的请求 key 会让 ADD 永不送达（症状是流挂死）。
- **Speculative decoding v2（Phase 11，n-gram + 引擎级 verify）**：draft 与 verify
  分离（vLLM v1 架构）——`NgramProposer` 从请求自身历史查最右出现的 n-gram 提案
  （无 draft 模型权重），引擎用**一次 verify forward** 验证整段 draft。流式约定
  （最后确认 token 的 KV 未写入）使输入 `[t_last, d1..dk]` 的第 i 位 logits 恰好
  检验 d(i+1)、末位给出 bonus token：全接受一步 k+1 个 token，部分接受在第 m 位取
  correction token（每步 ≥1，绝不劣于普通 decode）。被拒 draft 的 KV 用
  `BlockTable.truncate` 回滚（整块归还池）。greedy 接受 = 最长一致前缀；采样版为
  Leviathan rejection sampling（Phase 16，见下）。proposer 可注入（`spec_proposer`），
  与 vLLM n-gram/EAGLE/Medusa 同一插拔位。统计口径：accepted/drafted 为接受率，
  bonus 仅在未被 max_new_tokens 截断时计数。
- **采样链 + 投机采样（Phase 16，temperature/top-k/top-p + Leviathan verify）**：
  采样链抽到 `sampler.py`（temperature → top-k → top-p，逐级重归一化），引擎
  `_sample` 与投机 verify **共用同一目标分布定义**——verify 判定的是「draft 是否
  来自采样器会采出的分布」，两者必须同源（vLLM 同构：logits processors 同时喂
  直接采样与 spec-decode）。过滤在**概率空间**做（scatter 置零后归一化），被滤
  token 的质量**严格为零**——过 log 域往返会残留 ~1e-20，配合 rejection sampling
  会变成「不该被接受的小概率被接受」。top-k=1 恰为 one-hot argmax（与 greedy 短路
  一致）。`verify_drafts_sampled`：n-gram 提案是确定性的，draft 分布 q 为 one-hot
  （point-mass），一般拒绝检验退化为——以 `min(1, p(d))` 接受；拒绝时从
  `norm(max(p - δ_d, 0))`（p 抹掉 draft 后重归一化）重采一个 correction；全接受则
  末位分布采 bonus。定理在退化形式下依然精确成立：`p(d)·δ_d +
  (1-p(d))·p(x≠d)/(1-p(d)) = p`——输出分布**逐位等于目标分布**，对任意 proposer q
  都成立（单测按此数值验证：经验频率 ≈ p，容差 0.02）。greedy（temperature≤0）
  自动退化为 argmax 接受规则，greedy 路径代码不变。坑：`model.decode` 契约返回
  (1,V)（批量维），直接喂采样器会把 topk 的 `values[-1]` 取成整行——采样边界必须
  压成 1-D；权重初始化消费 RNG，播种确定性测试要先 init 再 seed。
- **MoE + EP（Phase 12）**：FFN 换成 top-k MoE——router 对全部专家 softmax 后取
  top-k 并重归一化（Mixtral/DeepSeek 语义），计算按专家分组（每个专家对路由到它的
  token 只算一次，对应 vLLM fused MoE 的 grouped GEMM）。EP 切专家不切 token：
  router 复制、各 rank 只算自己专家的加权贡献、一次 all-reduce 合并
  `MoE(x)=Σ_ranks Σ_{e∈rank} p_e·E_e` ——与 vLLM all-to-all dispatch/combine 数学
  等价、但不需要 all_to_all 集合通信（gloo 不支持）。确定性锚点：n_experts=1 时
  router 权重恰为 1.0 且 expert 0 复用 dense FFN 权重 → 1 专家 MoE 与 dense 逐位
  相等。`layer.mlp(x)` 小重构（dense/TP/MoE 各自实现，前向代码零复制）。MoE 专家
  权重按 expert id 独立播种，保证跨 rank 构造顺序无关。
- **共享专家 + 细粒度专家 + p2p combine（Phase 18，DeepSeek-V3 式）**：
  **共享专家**（`n_shared`）不进 router、权重恒 1.0、对每个 token 必加——在 EP
  combine **之后**计算，复制才不会把共享项加倍（pre-combine 加会在 all-reduce/p2p
  中被 rank 数放大——这是本阶段最容易埋的错，测试用「EP=1 真参考」而非「rank 自身
  参考」才抓得住：rank 自身参考把同样的错也算进去了，自洽但错）。**细粒度专家**
  （`inter_dim`）：专家变多变窄（intermediate 远小于 4·d_model），用路由组合数换
  等激活参数预算；inter_dim 非 None 时 expert 0 不再复用 dense FFN 权重（宽度不
  匹配），dense 锚点只在默认宽度下成立。**p2p combine**（`ep_combine="p2p"`）：
  与每个 peer 成对 isend/irecv 再累加——vLLM EP all-to-all dispatch/combine 的
  **形状**（gloo 没有 all_to_all 集合通信），数值上与 all-reduce 是同一个和；
  `"none"` 模式零通信（单实例持有全部专家，或作 dist 内的参考 oracle）。坑：两个
  模型先后构造时 router 初始化吃 ambient RNG——谁先谁后权重不同，构造前各自
  pin seed 才能拿到同一 router。
- **MLA（Phase 13/17，Multi-head Latent Attention，多头潜在注意力）**：K/V 经共享潜在
  向量低秩分解——每 token 每层只缓存一个 `[c_KV ; k_R]`（d_latent + d_rope 维）：
  c_KV 跨头共享，k_R 是解耦键（DeepSeek 中 RoPE（Rotary Position Embedding，旋转位置编码）
  的旋转依赖绝对位置、无法吸收进 Q，所以必须单独缓存）。**RoPE 旋转是真实的（Phase 17）**：
  `apply_rope` 按绝对位置旋转特征对（偶数维），k_R 在**写侧旋转**后入缓存（vLLM MLA
  backend 同款约定——kernel 读到的是位置定型的向量），q_R 在查询侧按 query 位置旋转；
  两者内积只依赖相对位置，这就是 RoPE 做位置编码的原理。absorbed 恒等式不受影响——
  可吸收的是位置无关的线性（W_UK/W_UV 作用在 c_KV 上），旋转正是那个**不可吸收**的成分，
  这也是解耦设计存在的理由。注意力两条路径：**absorbed**
  （vLLM FlashMLA kernel 的形态）把 W_UK 吸收进 query（q_abs = W_UKᵀ q_nope）、W_UV 吸收
  到输出（线性所以能挪过 softmax：out = W_UV·Σ_t p_t·c_t），kernel 只读缓存里的小向量；
  **explicit**（对每个缓存 token 上投影出 k/v）作为数学参考，两者 allclose 验证吸收恒等式。
  权重约定：`nn.Linear` 权重 (out, in)，w_uk/w_uv 一律 `.view(H, d_h, d_c)`、einsum 下标
  `hdc`——写成 view(H, d_c, d_h) 是对内存的错读（已踩坑）。分页缓存泛化：BlockPool 加
  `kinds` 维（dense=2 存 K/V；MLA=1，H=1，D=latent+rope），压缩才是真实的
  （20 vs 64 float/token）。批处理路径逐行镜像 dense 版：注意 scores 是 (B,T,H,S)，
  掩码必须广播成 (B,T,1,S)——写成 (B,1,T,S) 会把因果掩码错放到头轴上，T==H 时静默出错
  （已踩坑，靠 batched=streaming 单测抓出）。
- **MLA×TP（Phase 17）**：只有**带头轴的投影**能切——w_uq/w_qr 列并行（权重行切片）、
  w_uk/w_uv 按头切片（其输出维本就逐头）、wo 行并行；w_dkv/w_kr 与 **latent 缓存池完全
  复制**——latent 向量没有头轴，TP 无处可切，每个 rank 存全量 latent 池
  （dense MHA 是池按头切、每 rank 只存本地头，对比即 MLA 的 TP 卖点：通信省了、
  缓存却复制了）。TP 层直接继承 `_MLALayer` 的 attend/latent 方法，只换权重来源——
  方法里 `w_uk.weight.view(self.n_heads, d_h, d_c)` 的 n_heads 是本地头数，切片天然兼容。
- **KV 量化·fp8 e4m3（Phase 17，`Float8KVBlockPool`）**：与 int8 相对的两极——e4m3 是
  **浮点**格式（4 位指数给 ±448 动态范围），误差**相对于幅值**（3 位尾数的半个 ulp，
  ≤ 2^-4 ≈ 6.25%），**无需任何 scale 旁表**：1 byte/value，无记账开销；写入前 clamp 到
  ±448（溢出会变 NaN——生产用 scale 因子把激活压进量程，这正是 fp8 配 scale 的原因）。
  对比记忆点：int8 用「逐 token scale」换动态范围（绝对误差界 s/2，12 byte/token），
  e4m3 让指数位**本身**成为逐值自适应 scale（相对误差界，8 byte/token，head_dim=8 时
  32→8）。量化×MLA 复合免费成立：latent 池经同一 write/gather 分层被透明量化
  （20 float/token → fp8 后 20 byte/token）。
- **流水线并行（Phase 14，PP，Pipeline Parallelism）**：按层切 stage——rank 0 持
  embedding + 前 L/P 层，末 rank 持末段层 + ln_f/lm_head；激活 p2p send/recv 逐
  stage 前传，只有末 stage 见 logits。控制面与 TP 同为 SPMD（各 rank 跑同一调度、
  同一请求列表），由此两点成立：stage 间激活形状可本地推算（无需元数据通道——vLLM
  用每步广播调度批次解决同一问题）；贪心采样只在末 stage 做一次，采出的 token id
  广播给所有 rank，非末 rank 返回 one-hot 代理张量、引擎的 `argmax(logits)` 采样
  契约不变（vLLM 跨进程传的同样是 token id 而非 logits）。KV 池按层切分
  （每 rank `pool.num_layers`=本地层数），是 TP 按头切分的层维度对应物。stage 切片
  是 dense 权重的共享视图（零复制）。踩坑：批处理路径（prefill_batch/decode_batch）
  漏写 `table.advance` ——cursor 停在 0，decode 覆写 prompt 的 KV，前缀缓存注册也
  看不到任何 token，输出可能"看着像对的"；靠 `hits_tokens == 4` 断言抓出。
- **PP 微批流水线 + PP×TP + PP×spec（Phase 19）**：**微批流水线**（`micro_batch_size`）
  把一个 prefill batch 切成 M 个微批，stage s 在第 m+s 槽算第 m 个微批（GPipe 前向
  调度）——气泡从 lockstep 的 (P-1)/P 降到 (P-1)/(M+P-1)。诚实定位：这是 GPipe
  不是 1F1B——1F1B 的 warmup 深度（`pp_size - pp_rank - 1` 个在途微批）是为了在
  **backward 交错前传**时封顶激活内存，推理没有反向，GPipe 调度就是推理调度（vLLM
  同样把大 prefill 切微批流过 stage）。工程细节：send 用 `isend`、缓冲区保活到
  drain（stage 0 才能跑到 stage 1 前面）；每微批各自 `table.advance`；token 广播
  放在 drain **之后**且**末 stage 也必须 join**（只有消费者调用 broadcast 是分布式
  死锁）。**PP×TP**（`PPTPTransformer`）：world = pp_size × tp_size、
  rank = pp_rank·tp_size + tp_rank，一个 stage 就是一个 TP 组——行并行的 all-reduce
  必须**按 stage 组作用域**（`group=None` 意味着全 world，会把不同 stage 的激活加在
  一起静默污染；tp_size=1 的单例 stage 组也必须建）；KV 池双重切分（层按 stage、
  头按 TP），断言 `pool.num_layers == n_layers//pp_size` 且
  `pool.num_heads == n_heads//tp_size`；跨 stage p2p 用全局 rank
  `(pp_rank±1)·tp_size + tp_rank`，每个 TP rank 都发送复制的激活；`new_group` 是
  全 world 操作——只建自己那组是经典挂死来源，所有 rank 建全部 stage 组。TP 侧
  `_attn_layer/_run_layers_batch` 的 reshape 是本地宽度 `n_heads·head_dim`（PP 基类
  按 d_model reshape 会在 tp>1 时静默错切），用类级别别名继承 TP 版本。
  **PP×spec 免费**：SPMD 控制面下 verify forward 就是又一次前传，one-hot 代理维持
  `argmax` 契约，引擎零改动。测试抓出 Phase 14 遗留真 bug：PP **流式**路径
  （prefill/decode）从未 `table.advance`——引擎一直走批处理路径掩盖了它，spec
  verify 的流式 forward 第一次踩中（症状是 `truncate` 报"不能延长表"）。
- **组合矩阵清障 + PD 抢占（Phase 20）**：Phase 7/8 埋的两把互斥锁是**保守门**而
  非根本冲突，逐个验证解锁条件后移除——**spec×前缀缓存**：verify forward 的位置码
  从块表 cursor 起算（正是 `supports_prefix_cache` 契约），verify 后的 truncate 保持
  整块粒度，且前缀注册统一收口到 `_append_tokens`（plain decode/MTP/ngram verify
  都汇入此处；spec 路径维持同一不变量"最后确认 token 的 KV 未写入"——
  `register` 以 `table.num_tokens` 截断，故 spec 接受的 token 产生的块也会入缓存）。
  **spec×chunked prefill**：verify forward 一次跑 k+1 个 token，调度器的每步预算
  必须为每个 decode 请求预留 k+1（`Scheduler.tokens_per_decode`，引擎在 spec 开启
  时置 1+k）——vLLM v1 同样把 draft token 计入 `max_num_batched_tokens`；预留取
  上界（fallback 步实际只花 1），保守但正确。**PD 路径抢占**：逻辑引擎靠准入
  预留（`reserved_blocks` 整生命周期）天然不会中途缺块，真正的洞在双进程路径——
  `DecodeWorker` 此前盲目接收 handoff，池不够会在 decode 中途崩溃。补上**准入
  控制**：只接收"传输 token + 剩余生成"整体放得下的 handoff（与逻辑引擎同一条
  预留规则），放不下抛 `PDOutOfBlocks`（在任何块分配/KV 导入**之前**，handoff
  原样可重试——recompute 式抢占，vLLM v1 语义）；进程包装把异常转成类型化
  `("PREEMPTED", request_id, needed, available)` 结果而非 worker 崩溃，router 侧
  显式报错提示扩池重试。
- **KV 量化（Phase 14，int8 + 逐 token scale）**：先做读取分层——BlockPool 新增
  `gather / gather_block / gather_batch` 读取族（paged_attention 与批路径此前直读
  `pool.cache`，属分层违规），`Int8KVBlockPool` 借此注入反量化：存 int8 +
  每 (token, head) fp32 scale（s = max|x|/127 对称量化，误差 ≤ s/2），
  gather 透明反量化，模型与注意力零改动。显存：int8 payload + fp32 scale ≈
  12 vs 32 byte/token（head_dim=8，规模减半以上）。与 swap 抢占互斥（swap payload
  不携带 scale，构造时显式报错，不做静默错值）。vLLM 生产用 fp8 e4m3 +
  更粗粒度 scale 换更高收益，机制相同（fp8 e4m3 版见 Phase 17）。
- **不做（截至 Phase 20）**：CUDA kernel、fused MoE kernel/token 排布优化、
  EP 的 token 级 dispatch（p2p combine 交换的是全隐状态，不按 token 路由）、
  MoE 负载均衡/辅助损失、MoE×TP 组合（EP 是 MoE 的并行位）、TP 中 GQA 头分组切分、
  zmq/socket IPC（用 mp.Queue 直连）、批量
  verify forward、nccl/多 GPU 实测、custom all-reduce、graph×前缀缓存组合、
  FlashMLA/分块 MLA kernel、fp8 生产级 per-tensor/per-block scale（教学版 clamp 兜底）、
  微批×TP 组合（微批只在纯 PP 下启用，批处理走 lockstep）、
  1F1B 的 backward 交错调度（推理无反向，GPipe 即推理调度）、
  stage 间 CUDA IPC/NCCL 传输、量化×swap（已显式拒绝）、
  PD 抢占后的自动重试路由（类型化 PREEMPTED 已打通，重试策略属 router 层）。
- **Qwen3.5 边界**：HF 原生 `DynamicCache` 已覆盖 GDN recurrent state +
  full-attention KV 的增量正确性路径；混合 state 尚未分页化或接入 PD。MTP
  通过独立 predictor、target verification 和 hybrid-cache 快照回滚实现，需显式
  开启且当前只支持 `mtp_num_hidden_layers=1`。
- **PD 原型**：逻辑路径拆为 WAITING → PREFILL → HANDOFF → DECODE → FINISHED，
  decode 优先；真实 worker 路径由两个独立进程、两个模型副本和两个 KV 块池构成。
  handoff 传输 request 元数据与按逻辑块顺序排列的每层 K/V bytes，decode worker
  在本地重新分配物理块后导入，绝不传 source block id。
- **PD 边界**：当前是 CPU-staged bytes transfer 的 correctness harness；没有 CUDA
  IPC/P2P/RDMA、网络服务、worker 内 dynamic batching 或弹性扩缩容。decode 路径
  缺块已有类型化准入抢占（`PDOutOfBlocks` → PREEMPTED，Phase 20），但抢占后无
  自动重试路由。
  可显式指定 `cuda:0 -> cuda:1`：实际数据路径仍为 GPU -> CPU bytes -> GPU；跨 GPU
  测试由 `RUN_CROSS_GPU_PD_TESTS=1` 显式开启，避免默认占用第二张卡。

## 分阶段验证

| 阶段 | 内容 | 验证方式 | 状态 |
|---|---|---|---|
| Phase 1 | 分块 KV cache（块池/块表/分配/归还） | CPU 单测 | ✅ 完成 |
| Phase 2 | PagedAttention（按块 online softmax） | 与稠密 attention 数值等价（allclose） | ✅ 完成 |
| Phase 3 | 调度器（WAITING/RUNNING/抢占） | CPU 单测：增删/容量/抢占场景 | ✅ 完成 |
| Phase 4 | 引擎 + 小模型端到端生成 | L20 上 gpt2/opt 跑通生成 | ✅ 完成 |
| Phase 5 | README + GitHub 发布 | 仓库私有已建，HF GPT-2 适配器验证 | ✅ 完成（待公开发布） |
| Phase 6 | PD 分离原型 | CPU：跨块/多层 KV 导入导出、逻辑队列、取消、独立双进程 bytes handoff 均逐 token 对齐；4090D 独立 NGC PyTorch 24.04 容器：GPU 0 同卡双进程 `test_pd.py` 9/9 通过 | ✅ 完成（正确性） |
| Phase 7 | 前缀缓存（链式哈希 + 引用计数 + LRU 逐出 + 注册去重） | 13 项单测：命中复用/共享块跨请求存活/同批去重/抢占重入/逐出扩容/warmup 不污染；输出与稠密参考逐 token 一致 | ✅ 完成（2026-09-04，59 passed + 4 skipped） |
| Phase 8 | Chunked prefill（decode 优先的每步预算）+ CPU swap 抢占 | 14 项单测：分块准入/预算顺序/长 prompt 与 decode 交错/swap 保进度/swap 满退回 recompute/mid-prefill swap 复合路径；输出与稠密参考逐 token 一致 | ✅ 完成（2026-09-04，73 passed + 4 skipped） |

## Phase 8–20 路线图（2026-09-04 起，目标：吃透 vLLM 全部核心机制）

| 阶段 | 内容 | 对照官方源码 | 验证方式 |
|---|---|---|---|
| Phase 8 | Chunked prefill（长 prompt 分片预填充）+ CPU swap 抢占 | `vllm/v1/core/sched/scheduler.py` 的 chunked prefill / swap 调度 | 切片与不切片输出逐 token 一致；swap 恢复正确 |
| Phase 9 | 推理侧张量并行（TP：列并行/行并行 + all-reduce） | `vllm/v1/worker/gpu_model_runner.py` + parallel state | 4 项单测：切片数学（列/行切分约定 + bias 语义）、ws=1 引擎等价、TP=2 双进程 gloo 端到端（前缀缓存/chunked prefill 复合）——输出与 dense 参考逐 token 一致；KV 池按本地头数断言。TP=2 GPU 实测已并入 Phase 15 复测（L20，gloo 232 / nccl 294 tok/s） |
| Phase 10 | 异步引擎（EngineCore 进程 + 流式输出） | `vllm/v1/engine/core.py` + `async_llm.py` | 6 项单测：step 增量契约拼接=全序列且末步 finished、abort 覆盖 WAITING/RUNNING、3 路并发流逐 token=dense 参考、长短流独立推进（chunked prefill）、aclose 触发 abort 不泄漏（num_active==0）、shutdown 幂等；全量 83 passed + 4 skipped |
| Phase 11 | Speculative decoding v2（bonus token、接受率统计、n-gram draft） | `vllm/v1/spec_decode/`（ngram_proposer） | 9 项单测：proposer 右most 出现/k 截断/不重叠约束、CycleModel 全接受（bonus+少步数+逐 token=参考）、注入错误 proposer 全拒绝（correction 兜底、输出不变、接受率 0）、真实模型混合路由、KV 回滚无泄漏、max_new 截断回滚、4 组组合校验；全量 92 passed + 4 skipped |
| Phase 12 | MoE FFN + Expert Parallelism（EP，all-reduce combine） | `vllm/model/layers/moe.py` | 8 项单测：n_experts=1 与 dense 逐位相等（锚点）、分组计算=逐 token 合并、引擎输出=自身 dense 参考、EP 部分和=全量（in-process）、EP=2 双进程 gloo 与 EP=1 参考 MoE 一致、MoE×ngram spec 复合、路由/分区校验；全量 100 passed + 4 skipped |
| Phase 13 | MLA（Multi-head Latent Attention，latent 分页 + absorbed 路径） | `vllm/v1/attention/backends/mla/` | 9 项单测：absorbed=explicit 吸收恒等式、paged prefill=dense_forward、batched=streaming、KV 压缩记账（20 vs 64 float/token，3.2×）、缓存内容=真实 latent 向量（hook 逐层验证）、引擎输出=dense 参考逐 token、前缀缓存/ngram spec/swap 三个复合场景；全量 109 passed + 4 skipped | ✅ 完成（2026-09-04） |
| Phase 14 | 流水线并行（PP）+ KV 量化（int8/fp8 + 逐块 scale） | `vllm/distributed/` + kv cache quantization | 12 项单测：PP stage 切分/校验、PP=1 进程内等价 dense、PP=2/PP=3 多进程 gloo 端到端（输出=dense 参考逐 token、KV 池按 stage 断言、前缀缓存复合、cuda graph 显式拒绝）；量化池往返误差界（≤s/2）、跨块块表往返、显存减半记账（12 vs 32 byte/token）、释放清零、引擎贪心输出=dense 参考、量化×前缀缓存、swap 显式拒绝；全量 121 passed + 4 skipped | ✅ 完成（2026-09-04） |
| Phase 15 | 全量回归 + 收益复测（前缀缓存 TTFT、TP/PP/微批/PP×TP 吞吐） | 与 vLLM 同负载对比 | 全量回归 162 passed + 4 skipped ✅；README/plan.md/面试稿同步更新 ✅；新增 CPU 可跑示例 run_tp/run_pp/run_async ✅。**GPU 复测完成（2026-09-04，L20×4，NGC torch 2.10.0a0）**：`experiments/bench_prefix_ttft.py` —— 共享 1024 前缀 TTFT 41.5→14.4 ms（2.89×，hits_tokens=7168），全新前缀零回归；`experiments/bench_tp_pp.py` —— dense/TP2/PP2 锁步/微批 2·4/PP×TP2×2 × gloo/nccl 全矩阵，每条配置先断言输出=dense 贪心参考再计时（gloo p2p host staging、NCCL broadcast 跟随后端选设备两处修复后全绿）。复测中发现并修复两处真 bug：engine `step()` 未关 autograd（K/V 经块池带 grad_fn 常驻，100 步 41.75GB OOM）与 pp.py CUDA p2p/broadcast 的后端适配（README 踩坑 8/9） |
| Phase 16 | top-k 采样链（sampler.py 抽取）+ 投机采样（Leviathan rejection verify） | `vllm/v1/sample/sampler.py` + `vllm/v1/spec_decode/` | 15 项单测：top-k 恰好保留 k 个 token 且被滤质量**严格为零**、top-k=1≡argmax、top-k×top-p 复合、point-mass 输出分布逐位等于目标（2 万次试验经验频率≈p，容差 0.02）、被拒 draft 永不重采、bonus 来自末位分布、多位置链式逐位等于目标、top-k 支撑集外 draft 必拒、greedy 退化为 argmax 规则、引擎级采样 spec=循环参考+种子确定性+step_deltas ≥1 token+KV 回滚无泄漏；全量 **136 passed + 4 skipped** | ✅ 完成（2026-09-04） |
| Phase 17 | MLA RoPE 旋转（写侧转 k_R）+ MLA×TP（latent 复制）+ fp8 e4m3 KV 池 + 量化×MLA | `vllm/v1/attention/backends/mla/` + kv_cache fp8 | 10 项单测：absorbed/explicit 吸收恒等式在真旋转下保持、分页=dense_forward（旋转后）、batched=streaming、缓存内容=旋转后 latent（hook 验证）、MLA 分片数学（头切片 partial 求和=全量）、ws=1 等价、TP=2 双进程 gloo 逐 token=dense 参考+**latent 池不切分**断言、fp8 往返相对误差界 ≤2^-4、存储 1/4 of fp32 且小于 int8 池、MLA×int8/fp8 引擎复合（int8 逐 token 相等、fp8 容忍至多 1 token 翻转）；全量 **146 passed + 4 skipped** | ✅ 完成（2026-09-04） |
| Phase 18 | MoE 共享专家/细粒度专家 + EP pairwise p2p combine（all-to-all 形状） | `vllm/model/layers/moe.py`（DeepSeek-V3 结构） | 6 项单测：共享专家=FFN 恒加一项（n_experts=1+n_shared=1 → 2×dense 逐位）、路由后恰好加一次（手工参考）、细粒度 inter_dim=16 权重形状+grouped=逐 token 合并+引擎端到端、默认宽度 dense 锚点不变、无 dist 时 p2p=恒等+非法 ep_combine 拒绝、EP=2 双进程 p2p == EP=1 真参考（ep_combine="none" oracle，共享项不加倍）；全量 **152 passed + 4 skipped** | ✅ 完成（2026-09-04） |
| Phase 19 | PP 微批流水线（GPipe 前向调度）+ PP×TP + PP×speculative | `vllm/v1/worker/gpu_model_runner.py`（微批） | 4 项单测：微批与整批输出一致（`micro_prefills≥1` 断言防 lockstep 兜底空转）、PP×TP=2×2（world 4）逐 token=dense 参考+KV 池双重切分断言（num_layers==n_layers//pp_size 且 num_heads==n_heads//tp_size）、PP=2×ngram spec 复合（注入 proposer 保证 verify 必跑、输出=dense 参考、引擎零改动）、PP×TP 无进程组/n_heads 不整除显式拒绝；顺带抓出并修复 Phase 14 遗留 bug（PP 流式路径漏 `table.advance`）；全量 **156 passed + 4 skipped** | ✅ 完成（2026-09-04） |
| Phase 20 | 组合矩阵清障：spec×前缀缓存、spec×chunked prefill、PD 抢占 | `vllm/v1/core/sched/` 组合校验 | 6 项单测：spec×prefix 共享前缀命中（hits_tokens==4）+ spec 步运行 + 输出=dense 参考、spec×chunked 长 prompt 分块+并发 decode+输出=dense 参考、调度器预算预留单元（k+1 计入 max_num_batched_tokens）、PD decode worker 准入拒绝（needed/available/不漏块/handoff 可重试成功）、双进程类型化 PREEMPTED、两把旧互斥锁翻转成"可构造"断言；全量 **162 passed + 4 skipped** | ✅ 完成（2026-09-04） |

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
