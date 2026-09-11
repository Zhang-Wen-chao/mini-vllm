# MTP 投机解码 sweep：Qwen3.8-27B 单卡（2026-09-11）

官方 vLLM 0.28.0 上对 Qwen3.8-27B（48 GDN + 16 full attention 混合架构，自带单层
MTP 头权重）做投机解码实测。**与 mini-vllm 仓库的关系**：本项目不涉及 mini-vllm
代码；本目录只作为该轮实验的证据归档位（官方框架实验统一收在 `experiments/`，
与 2026-09-10 的 `experiments/*_2026-09-10*` 同一惯例）。内部路径已脱敏：
`<model-dir>`（模型目录）、`<venv>`（Python venv）、`<out-dir>`（输出目录，
跨 sweep 引用带编号 `<out-dir-2/4/5>`）、`<lc-venv>`（标定 venv）、
`<chain-log>`/`<calib-log>`（编排/标定日志）。

## 问题

两个问题：
1. **性能**：MTP（`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`）
   在低并发/高并发下各值多少吞吐？和 fp8 KV 能不能叠加？
2. **正确性**：投机解码理论上必须无损（verify 用完整模型前向，贪心输出应与
   非 spec 路径逐 token 一致）——实测是否成立？

## 环境（锁档）

- 见 `sweep2/env_2026-09-11.txt`（nvidia-smi + pip freeze 全量）。
- Python 3.12.3 / vllm 0.28.0 / torch 2.13.0+cu130 / transformers 5.16.1 / triton 3.7.1；
  单卡 L20（48GB），与 2026-09-10 对比实验**同一 venv**（未变更）。
- 跑在容器内；时钟为 UTC（log 时间戳 01:xx-02:xx = 北京时间 09:xx-10:xx）。

## 协议

- 基线 = 单卡 fp8 权重 + **KV bf16** + 默认调度（8192）+ prefix caching ON
  （部署实验的"性能基线②"）。spec 臂只加 `--speculative-config`，其余零改动。
- 压测：`vllm bench serve --backend openai-chat`，1024 进 / 256 出，96 prompts，
  temperature 0，并发 16 与 48 两档；每臂独立起服务（冷缓存）。
- 臂序 = A → M1 → M2 → A2（sweep1）、K2 → C2 → A3（sweep2），
  首尾基线互为**锚点**（同类 ABBA 纪律：首尾基线复现则中间臂的差异归于旋钮）。
- sweep2 的 K2 = fp8 KV（无 spec）补测 @16（历史只有 @48）并回放 @48 作跨轮锚点；
  C2 = fp8 KV + MTP k=2（阶梯终点配置，boot 兼容性此前无人验证）。

### 公平清单（逐项）

| 维度 | 各臂取值 |
|---|---|
| venv / torch / vllm | 同一 `<venv>`（0.28.0 / 2.13.0） |
| 权重 / 精度 | 同一 fp8 权重目录；KV dtype 是臂变量（bf16 vs fp8） |
| 调度 | `--max-model-len 8192 --max-num-seqs 128`，其余默认 |
| workload | 同一 bench 命令、同 prompt 数/长度、temperature 0（双方都不设 top-k/top-p/seed） |
| 指标 | vLLM 自带 serve bench 输出，公式同源 |
| 优化默认项 | 双方都开 CUDA Graph、prefix caching（各自默认） |
| **已知 side-effect** | spec 臂被 vLLM 自动改写 `max_num_scheduled_tokens=2048`（启动 WARNING 原话见 `server_startup_lines.txt`），且 draft 权重吃掉 0.8-2.9 GiB → KV 池缩小（见下表）。二者都是"MTP 实现的一部分"，不是配置失误，但 @48 的 TTFT/吞吐劣势主要来自它们——已单独归因，不与 verify 收益混算 |

## 结果（输出吞吐 tok/s；A2/A3 为锚点）

### sweep1（KV bf16，spec 旋钮）

| 臂 | @16 | vs A | @48 | vs A |
|---|---:|---:|---:|---:|
| A 基线 | 219.37 | — | 307.14 | — |
| M1（k=1） | 269.66 | **+23%** | 287.62 | -6% |
| M2（k=2） | 278.11 | **+27%** | 282.31 | -8% |
| A2 锚点 | 219.55 (+0.08%) | | 307.58 (+0.14%) | |

TPOT @16：A 64.7ms → M1 53.0ms（-18%）→ M2 47.6ms（-26%）。
@48 TTFT：A 9.5s → M1 20.1s / M2 23.6s（≈2-2.5 倍，`max_num_scheduled_tokens`
被强制 2048 所致）。

### sweep2（fp8 KV + spec 叠加，阶梯终点）

| 臂 | 配置 | KV 池 tokens | @16 | @48 | @16 TPOT |
|---|---|---:|---:|---:|---:|
| A3 锚点 | fp8 权重，KV bf16 | 99,267 | 219.36 | 307.11 | 64.8ms |
| K2 | + fp8 KV | 136,533 | 219.53 | **351.99** | 67.3ms |
| C2 | + fp8 KV + MTP k=2 | 81,920 | **293.55** | 291.49 | **48.2ms** |

- **K2@48 = 351.99**，复现五天前同配置（fp8 权重+fp8 KV）的 352.9（±0.26%），
  且 KV 池 136,533 tokens / 16.67x 与历史完全一致——跨日复测通过；
- **fp8 KV 在 @16 零收益**（219.53 vs 219.37）：16 路并发无容量压力，
  KV 量化只在容量受限的 @48 兑现（+14.6%，351.99/307.14）；
- **C2@16 = 293.55**：单卡全栈比 M2（278.11）再 +5.5%——fp8 KV 单独无用，
  但与 MTP 叠加有正交互用（假设：verify 每步读 k+1 个位置的 KV，KV 读流量
  占比上升，fp8 KV 开始回本；未单独验证，记录为假设）；
- **阶梯（@16）**：219.4（fp8 权重）→ +27%（MTP）→ 278.1 → +5.5%（fp8 KV）
  → **293.6**；对照双卡 TP2 bf16 同窗 ABBA 值 284.9±6.5——超出均值 3.0%
  （TP2 自身波动 ±2.3%，属打平偏赢），TPOT 48.2 vs 49.5ms、TTFT 1.28s vs
  1.7-2.0s，省一半卡；
- **@48 最优仍是"fp8 KV 不加 MTP"**（351.99）：批量饱和后 verify 的多算
  换不回来（C2 291.49，比 K2 低 17%）。

### 接受率（分母显式）

- k=1（M1@16）：accepted 21,959 / drafts 27,093 = **81.0%**；
- k=2（M2@16）：drafts 20,647，draft tokens 41,294（每 draft 2 个位置），
  accepted 28,485；分位置 pos0 16,200/20,647 = **78.6%**、pos1 12,285/20,647 = **59.5%**；
- C2（fp8 KV + k=2）接受率与 M2 几乎不变（pos0 78.6% / pos1 59.9%）——
  C2 相对 M2 的 +5.5% 不是接受率贡献。

### sweep3（C2b：显式 sched 8192 排除法 + 容量墙实锤）

C2b 与 C2 唯一差异 = 显式 `--max-num-batched-tokens 8192`（消掉 spec 强制
`scheduled=2048` 的 prefill 节流假设）：

| 臂 | @48 | @48 TTFT | @16 | @16 TPOT |
|---|---:|---:|---:|---:|
| C2（sched 默认） | 291.49 | ~23s | 293.55 | 48.2ms |
| C2b（sched 8192） | 293.25 | 22.0s | 301.37 | 46.3ms |

- **@48 不动**（293.25 vs 291.49，+0.6%）：调度假设被排除；
  @16 +2.7%（301.37）为该臂最好成绩（超 TP2 284.9 达 +5.8%，TP2 波动带 ±2.3% 之外）；
- **容量墙实锤**（`sweep3/engine_stats_*.txt`，@48 采样）：spec 臂 Running 钉死
  **16-17 / Waiting 31 / KV usage 90-96%**，而 K2 同窗 Running 满载 48；
- C2b 的 ITL 114.8ms vs TPOT 48.9ms：每 verify 轮产出 ~2.35 token
  （118.5/50.4≈2.35 ≈ 接受率推算 2.38），verify 步仅 +4.6% 时间换 2.35 倍 token
  ——**decode 侧是赚的，@48 输在并发装不下，不在算力**。

### 容量墙机制（源码级，2026-09-11 补）

对 vLLM 0.28 源码的最终归因（两层修正后的定稿，见文末"修正轨迹"）：

- 池的块粒度被 mamba page 抬高：`kv_cache_interface.py:911` 强制
  **attention block size = 1,600 tokens**（对齐 mamba state page，padding 0.88%），
  混合架构里全注意力 KV 块与 GDN 状态槽共享同一块池；
- spec 请求的 mamba 运行时准入门（`single_type_kv_cache_manager.py` align 分支）：
  `cdiv(num_tokens_main_model, block_size) + num_speculative_blocks`——按**实际
  token 数**记账。1,280 token 的请求 = 1 + 2 = **3 块**（1 当前状态 + 2 个 spec
  回滚快照，`kv_cache_interface.py:694`：align = page×(2+spec)）；
- 对账：C2 池 81,920 tokens ÷ 1,600 ≈ **52 块**；52 ÷ 3 ≈ 17 路 ←→ 实测
  Running 16-17、usage 90-96% **逐位吻合**；
- `max_model_len` 只影响块表行长（元数据）与启动时 "Maximum concurrency"
  打印（81,920/8,192 = 10.00x），**不影响运行时每请求驻留**；
- 池从 136,533（K2）缩到 81,920（C2）的 40% 是 **draft 权重吃掉的显存**，
  叠加每请求 +2 快照块的准入税——两笔账都指向同一副药：**压目标模型权重，
  把省出的显存灌回块池**（int4 路线，见 sweep5 预注册）。

## sweep4 / sweep5 预注册（2026-09-11 12:5x 北京时间，C4 开跑前）

用户拍板新目标：**明确超越 TP2 基线**（@16 284.9±6.5 / @48 350.2±3.3 双卡），
"打平偏赢"不够。两条路线，预测与判据先落字：

- **C4 臂（sweep4，`mtp_bench4.sh`）**：fp8 KV + MTP k=2 + `max-model-len 4096`
  （+ sched 8192）。**预注册预测：阴性**——源码读出准入门按实际 token 记账
  （见上节），len 4096 不减每请求 3 块的驻留，Running 应仍 ~17、@48 应仍
  ~290。跑了三小时前的初始预测（"len 减半 → 每请求记账减半 → ~30 路"）在
  读源码后**于开跑前撤销**，保留于此作修正轨迹。若实测意外翻正，则说明
  还有未读到的 len 依赖，机制节需重写。判据：@48 ≥ 340 才算翻正。
- **W4 臂（sweep5，`int4_calib.py` + `mtp_bench5.sh`）**：GPTQ **W4A16**
  （int4 / group 128 / symmetric）混合精度——FFN+全注意力 QKV/O（~18B 参数）
  进 int4，**GDN 层（`*linear_attn*`）留 bf16**（递归状态误差累积风险，
  SGLang #22935 同源判断）、`mtp.*` 留 bf16（草稿精度即接受率）、lm_head/
  visual 留 bf16。产物 `<model-dir>-int4`，serving 自动识别 compressed-tensors，
  其余协议与 C2 逐项一致（fp8 KV + k=2 + len 8192 + sched 8192）——单旋钮 =
  权重精度。**预注册预测**：权重 ~30GB→~20GB，池 52 块→~150+ 块，@48
  Running 升到 ≥48，吞吐 **≥420 tok/s**（判据：超 TP2 350.2 达 +20% 才算
  "明确超越"；420 以下算部分兑现，需查块池实际增量）。风险预记：Marlin
  sm89 内核兼容、GDN bf16 与 int4 混载加载、贪心分歧进一步扩大（预期内，
  记录用）。
- ngram 投机解码**不跑**：本 bench 是 random token 数据集，prompt 无重复子串，
  ngram 接受率趋零，测出来只有纯开销——换 workload 才有意义，不破坏本轮
  单旋钮纪律（记录为弃选理由）。

### sweep4 结果（C4，2026-09-11 13:3x 北京时间）：**预注册阴性兑现**

| 臂 | @48 | @48 TTFT | @16 | @16 TPOT | 接受率 pos0/pos1 |
|---|---:|---:|---:|---:|---|
| C2b（len 8192） | 293.25 | 22.0s | 301.37 | 46.3ms | 78.6%/59.9% |
| C4（len 4096） | **298.21** | 21.9s | 297.57 | 46.7ms | 79.8%/61.7% |

- **判据裁决：@48 = 298.21 < 340 → 阴性确认**（vs C2b +1.7%，波动量级）；
  引擎统计与 C2b 完全同相：Running **16-17 / Waiting 31 / usage 90-96%**
  （`sweep4/engine_stats_C4_c48.txt`）——并发墙纹丝不动，机制定稿成立；
- **KV 显存字节两臂完全相同（9.28 GiB）**，但"token 容量"打印从 81,009
  （10.00x）变 48,605（11.87x）——group-aware 容量记账随 len 变，而实际
  Running 不随打印动：**"Maximum concurrency" 打印不是运行时闸门**的直接
  证据（打印 = 池 tokens ÷ max_model_len 的元数据算术）；
- **数值对照通过**：C2 vs C4 贪心探针 **8/8 逐 token 一致**（`greedy_diff_c2c4.py`），
  len/块表行长不改任何贪心 token——A vs C4 的 7/8 分歧全部来自 spec+fp8KV
  （与既有签名一致），进一步把贪心不稳定的归因收窄到量化+verify 前向，
  排除分页布局一类通用因素；
- 结论：**缩 max-model-len 不是解锁键**，与源码推演一致；W4（压权重）成为
  唯一在跑的超基线主牌。

### sweep5 结果（W4，2026-09-11 14:5x 北京时间）：**预注册判据未达，int4 负结果**

| 臂 | @48 | @48 TTFT | @16 | @16 TPOT | KV 池 | KV 显存 |
|---|---:|---:|---:|---:|---:|---:|
| C2b（fp8 全栈） | 293.25 | 22.0s | 301.37 | 46.3ms | 81,920 | 9.28 GiB |
| W4（int4 全栈） | **164.01** | 34.4s | 155.30 | 72.6ms | **124,700** | **14.23 GiB** |

- **判据裁决：@48 = 164.01 ≪ 420 → 大幅未达**，@16 也从 301.4 跌到 155.3
  （-48%）。int4 这张牌在本机型/内核栈上**打输**；
- **容量墙确实松了但没解**：池 81,920→124,700 tokens（KV 显存 9.28→14.23 GiB，
  +53%——省出的 ~5GB 显存兑现了），Running 从 16-17 升到 **27/48**，usage
  波动 54.9%↔98.9%——但 27 路仍不满 48，且每路更慢（见下）；
- **真正的死因是 decode 变慢，不是容量**：@16 TPOT 72.6ms vs C2b 46.3ms
  （+57%）。Marlin kernel 确认在用（启动日志原文
  `Using MarlinLinearKernel for CompressedTensorsWNA16`，无 fallback 警告），
  也就是说 **Marlin int4 GEMM 在 sm89/L20 这套 batch 形状下就是比 fp8 GEMM 慢**：
  低 batch decode 时 weight-only int4 的 dequant 开销吃掉了权重带宽红利
  （27B 模型 16 路 batch 的 GEMM 形状对 Marlin 的分块不友好）。@16 的 -48%
  全部来自这 +57% TPOT：155.3 ≈ 301.4 ÷ 1.57 × (48.2/46.3)；
- **贪心分歧谱：0/8 一致，但语义未崩**：全部 8 题与基线 A 分歧（分歧位置
  0-81，最早第 0 个 token 就翻）——比 fp8 KV 的 7/8 更宽；但输出检查是
  **语义保留的同义改写级**（数学题算到 340+51=391 对、翻译三语全对、riddle
  正确列方程）——int4 误差翻 logits 排序但不摧毁生成质量。这是数值不稳定
  谱系的又一档（bf16 基线 8/8 → fp8 KV 7/8 → spec 6/8 → int4 全翻但语义在）；
- **C-Eval 回归（补跑，坑 10/11 双踩后修通）**：**W4 = 0.7550**（151/200）——
  vs bf16 0.7950 净 -4.0 分、vs fp8 0.7750 净 -2.0 分；配对 McNemar
  w4 vs bf16 = 5:13（不显著，p≈0.096），与 fp8 轮同款结论。与贪心 0/8
  并看是本轮最反直觉的组合：**96-token 生成轨迹全翻，200 题单步 MCQ 只净漂
  4 题**（agree 179/200）——分歧是 logits 排序级的近并列翻转，不是知识损坏。
  W4 质量结论与性能结论合流：int4 在本机型的代价在速度，不在语义；
- **定稿结论**：W4 阴性。@48 的超基线路径在这套内核栈上断在 Marlin decode
  速度，不在容量——容量路线（int4→池+53%、Running 16→27）方向本身被证实，
  但每路 TPOT +57% 的代价抵消并发收益还倒贴。**@16 的「明确超越」答案保持
  C2b（301.4，fp8 全栈 + 显式 sched）**；@48 维持 fp8 KV 无 spec（352.0）。
  下一步若要攻 @48，方向是找比 Marlin 更快的 int4 执行路径（Machete/
  CUTLASS sm89 路径、或 GDN 层也压但换 W8A8），不是再压精度。

## 正确性发现：贪心等价性破了

探针 `mtp_greedy.py`（8 题固定双语 prompt，temperature 0，logprobs 逐 token），
比较脚本 `greedy_diff2.py` / `greedy_diff3.py`：

| 对照 | same / diverge | 解读 |
|---|---|---|
| A vs A2（同轮重启） | 8 / 0 | 探针零噪声 |
| A(sweep1) vs A3(sweep2) | 8 / 0 | 跨轮、跨日、跨服务重启，仍逐 token 一致 |
| A vs M1（spec k=1） | 4 / 4 | **投机解码改变贪心输出** |
| A vs M2（spec k=2） | 2 / 6 | 同上，更深 |
| A3 vs K2（纯 fp8 KV） | 1 / 7 | **KV 量化本身也改变贪心输出** |
| K2 vs C2 | 2 / 6 | 两种扰动互相也不同 |

- 分歧特征：**近并列 logits 的 argmax 翻转**——同一 prompt 在 M1/M2 翻在同一
  位置，翻后输出语义等价（两版都算对 17×23=391、代码都对）；K2 的翻转点与
  M 臂不同但特征相同（"total"→"sum"、标点级改写）；
- 与 C-Eval 口径自洽：fp8 单步 argmax 一致率 95.5%（191/200）→ 96 token
  贪心轨迹上至少一次翻转的概率趋近 1（7/8 实测）；
- 与 vLLM issue [#54928](https://github.com/vllm-project/vllm/issues/54928)
  （open）交叉印证：DFlash2 draft 在同模型上同样改变贪心输出（含 K=1 与
  enforce-eager）——两种不同 draft 机制都翻，共同因子是 verify 前向的
  batch 形状数值不变性（同类问题：vLLM #55238 的 GEMM padding 不变性）；
- **工程含义**：贪心输出对配置扰动不是不变量（量化 / spec / batch 形状各翻各的）。
  回归测试必须用轨迹相似度或 logprob 级比较，不能用精确匹配；生产上
  "temperature=0 保证可复现"只在同配置内成立。

## sweep6 预注册（2026-09-11 15:2x 北京时间，开跑前落字）

用户成功判据（原话意）：**同两卡预算** vs TP2 双卡——单请求更快、总吞吐更大、
精度掉不多 = 一次成功的部署。

**先修正一个口头汇报口误**（详见文末修正轨迹第 5 条）：W8A8 不是未测牌——
A 臂 `--quantization fp8` 即 W8A8 dynamic，fp8 GEMM 是整条阶梯的底座，每路
速度已在 fp8 顶。剩下唯一没打的牌是**部署架构：TP2 vs 数据并行副本**。
且 bf16 52GB 装不进单卡 48GB——**量化是副本架构的前置使能，不是可选加速器**。

臂设计（同一窗口顺序跑；GPU1/2/3——GPU0 被邻座 qserver 占 39GB 不用，
GPU1-3 另有 284MB 邻座足迹）：
- **TP2 锚点**（GPU2,3，历史原配：bf16/len 8192/seqs 128/sched 默认；
  served-model-name 统一成 qwen38-27b 使 bench 命令各臂逐字一致）：
  @48（96 prompts）+ @96（192）——邻座条件下同窗重测；历史值 350.2±3.3
  作对照，锚点偏离历史 >5% 判污染窗，只报同窗相对值；
- **R2b** = 2×（fp8 权重+fp8 KV 无 spec + sched 8192 = K2 配置）：
  @48total（2×[48p, c24]）+ @96total（2×[96p, c48]）；
- **R2c** = 2×（同上 + MTP k=1）：@48total——**新配置**（k=1 快照税减半：
  52 块÷2 = 26 路 ≥ 24，预期无墙）；
- **R2a** = 2×（同上 + MTP k=2 = C2b 配置）：@48total（每副本墙 17 路，
  c24 下 7 路排队）。

**预注册预测**：
- TP2 @48 ≈ 350±10（含邻座条件）；@96 ≈ 350-390（近饱和）；
- R2b @48total ≈ 540-620；@96total ≈ 680-720（2×K2@48=352）；
- R2c @48total ≈ 560-640（M1@16 269.7 基数 + 8 路增量，无墙假设下）；
- R2a @48total ≈ 560-600（每副本墙内饱和 ~290-300）；
- TPOT：TP2 @48 按吞吐推算 ~137ms（350.2/48）；副本对 46-90ms——
  预测单请求延迟也占优。TTFT：R2a 因 7 路排队会差（~5-12s），R2b/R2c 好。
- **判据**：@48total ≥ 420（超 TP2 +20%，与 W4 同口径）= 明确超越；
  精度沿用 C-Eval（副本 = 同引擎配置无新数值漂移；fp8 0.775 vs bf16
  0.795，McNemar n.s.），不新跑。

**公平性**：同总 prompt 数（@48total=96、@96total=192）、同 bench 工具/
workload/指标、TP2 历史原配复现、副本对内两卡同配。本对比是**部署层合成**
（架构+量化双变量）——单旋钮归因已在 sweep1-3 完成，此处回答「给定两张卡
怎么部署最好」；bf16 单卡装不下 → 2×bf16 副本对照组物理不存在，量化使能
副本这一因果在结论里必须明说。

### sweep6 结果（2 卡部署矩阵，2026-09-11 16:3x 北京时间）：**三个副本配置全部「明确超越」**

| 配置（两张卡） | @48total | vs TP2 同窗 | vs TP2 历史 | TPOT | TTFT | 每副本引擎态 |
|---|---:|---:|---:|---:|---:|---|
| TP2 bf16 锚点（同窗重测） | 372.07 | — | +6.2%（见下） | 104.2ms | 6.10s | Running 45-48 满载，usage 43.9% |
| TP2 锚点 @96（192p, c96） | 469.62 | — | — | 159.2ms | 10.87s | — |
| R2b：2×（fp8+KV fp8 无spec） | **548.17** | +47.3% | +56.5% | 66.5ms | 5.42s | Running 24/24，usage 50.8% |
| R2c：2×（同上+MTP k=1） | **618.76** | **+66.3%** | **+76.7%** | **55.6ms** | **4.90s** | Running 24/24，usage 94.4% |
| R2a：2×（同上+MTP k=2） | 578.49 | +55.4% | +65.2% | 44.5ms | 7.39s | Running 14-17+Waiting 7-8（墙） |
| R2b @96total（2×[96p,c48]） | 622.74 | +32.6% vs 469.62 | — | 101.0ms | 8.18s | 引擎态未采（见证据缺口） |

- **判据裁决：@48total ≥ 420——R2b 548 / R2c 619 / R2a 578 全部大幅通过**，
  且不等锚点选谁（同窗 372 或历史 350.2）结论都成立。用户成功判据
  （单请求更快 + 总吞吐更大 + 精度掉不多）**三配置全中，R2c 最优**：
  TPOT 55.6ms（TP2 的 53%@48 / 35%@96）、TTFT 4.90s（TP2 的 80%@48 /
  45%@96）、吞吐 +66%（同窗口径）、精度沿用 fp8 引擎 C-Eval 0.775
  （vs bf16 0.795，McNemar n.s.）；
- **TP2 锚点窗口标记**：372.07 vs 历史 350.2±3.3 = +6.2%，超预注册的 5%
  污染线 → 按预案**以同窗相对值为主**。注：本次窗口邻座 qserver（GPU0
  39GB）在全部 gpu_snapshots 里 util 0%——是驻留显存不是活跃算力，本次
  窗口可能比 09-06 历史窗（当时有活跃共享负载，TP2 跨窗波动 ±48% 有案）
  更干净；无论锚点取哪边，副本对都赢得不可逆；
- **预测对账**：R2b @48t 548（预测 540-620 ✓ 低段）、R2c @48t 619
  （预测 560-640 ✓ 正中）、R2a @48t 578（预测 560-600 ✓）、**R2b @96t
  622.7（预测 680-720 ✗ 低的诚实 miss）**——miss 机制：邻座 284MB 足迹
  把每副本 KV 池从 136,533（历史 K2）压到 129,706（15.83x），c48 档更贴
  容量天花板，叠加两副本+两 bench client 共享宿主 CPU；
- **机制逐位闭合（第三次）**：R2c usage 94.4% ↔ 24 路×2 块 = 48 块 / 池
  ~51 块；R2a 钉死 17 路×3 块 + Waiting 7-8 ↔ 与 sweep3 单卡墙同构——
  **k 是容量旋钮**：每个 k 加一块/请求的快照税，最优 k 取决于每副本并发
  离墙多远（c24 时 k=1 赢 k=2，c≤16 时 k=2 赢——C2b 单卡 @16 301 > k=1
  的 M1 269.7）。接受率 k=1 82.6-83.0%（两副本）与 sweep1 M1 的 81.0%
  同档；k=2 pos0 78.3-79.9% / pos1 58.7-61.6% 与 C2 的 78.6%/59.9% 同档；
- **bf16 副本对照物理不存在**：52GB bf16 装不进 48GB 单卡——量化是副本
  架构的前置使能，这层因果在结论里明说（公平性节预记）。

**证据缺口（诚实记录）**：R2b_c96t 的引擎态未采（脚本每相只调一次
engine_stats，`engine_stats_R2b_c48t.txt` 只覆盖 c48t 轮）；贪心探针
sweep6 未跑（R2b=K2、R2a=C2b 已有谱；k=1+fp8KV 组合未探，无结论依赖它）。

## sweep7 预注册（2026-09-11 北京时间，开跑前落字）

目标不是「再找一个优化」，是**把基线修对、把增益拆开、把天花板钉死**。
sweep1-6 把手段试遍了，但留了三个结构性缺陷，sweep7 先补这个。

### 0. 两条前置纪律（sweep1-6 违反过，本次修正）

1. **基线行必须同窗。** 速览 §0 表基线行现在 `@48：372`（sweep6 窗，09-11
   16:3x）+ `@16：284.9`（09-06 ABBA 窗——**那一窗的 @48 是 350.2±3.3**）。
   同一格两个数字来自两个窗口，正是本项目自己的 ABBA 纪律禁掉的做法。
   → sweep7 在基准窗补测 **TP2 @16**，整行改用同窗值。
2. **单卡线基线要有自己的名字。** §0 表「优化一 同卡 @48：307.6 → 352.9」的
   307.6 **不是**基线 372，是 **L1 =「fp8 权重 + KV bf16」单卡**。那张表实际是
   两条线拼的（双卡线 基线→优化三 / 单卡线 L1→L2→优化二），「每步只动一个
   变量」对自己不成立。→ 文档与口述必须点名 L1。

### 1. 优化手段总账（全量；每个手段只动三样之一：权重字节 / KV 字节 / 算子速度）

| 手段 | 权重字节 | KV 字节 | 算子速度 | 容量轴实测 | 裁决 |
|---|---|---|---|---|---|
| fp8 权重 W8A8 | 2B→1B | — | 原生 MMA，快 | 池 →136,533 | ✅ 已测（L1→L2）：吞吐 +14.7%、TPOT +12.5% |
| fp8 KV | — | 2B→1B | 略慢 | 99,267→136,533（**+37.5%，非 +100%**） | ✅ 已测（L2）；大头 GDN 状态压不动（⟶ 见 §2 口子 1，这条假设未验证） |
| int8 权重 | 1B = **与 fp8 相同** | — | L20 上 fp8 原生、int8 非 | **零增益** | ⛔ 不做（预注册阴性：容量轴等价 fp8） |
| int4 权重 GPTQ W4A16 | 1B→0.5B | — | **Marlin decode TPOT +57%**（46.3→72.6ms） | 81,920→124,700（**+53%**）、Running 16-17→**27** | ❌ 已测阴性：@48 164.0、C-Eval 0.7550 |
| int4 权重 AWQ | 同 int4 | — | **未知——唯一开放口子** | 同 int4 | ⏸ 先用 1 条 grep 判生死（§2 口子 6） |
| int4 KV | — | 无可用 kernel | — | — | ⛔ 不做（vLLM 0.28 `--kv-cache-dtype` 面只有 auto/fp8_*；需一条 `--help` 确认后写入档） |
| MTP k=1 | — | −（+1 回滚快照块/请求） | +（接受率 82.6-83.0%） | 单卡池 136,533→97,757 | ✅ 副本布局最优 |
| MTP k=2 | — | −（+2 块） | +（pos0 78.3-79.9% / pos1 58.7-61.6%） | 池 →81,920 | ✅ **k 是容量旋钮**：单卡 @16 赢（301），双副本 24 路输（578.5 vs 618.8） |
| MTP k=3 | — | −（+3 块） | 接受率必然更低 | — | ⛔ 不做（pos2 接受率低于 58.7% 且块税再 +1） |
| ngram / 外部 draft | — | — | — | — | ⛔ 已弃（random 数据集无重复子串，接受率趋零） |
| fp8 KV × MTP | — | — | — | — | ✅ **正交互用**：fp8 KV 单独 @16 零收益，叠加 MTP 再 +5.5%（机制：verify 每步读 k+1 位置 KV） |
| 前缀缓存 | — | — | — | — | ✅ 已测（@16 −74% 首字 / @48 零收益） |
| 调度预算 batched-tokens | — | — | − | — | ❌ 已否（TPOT −33% 换 TTFT 3×、吞吐 −11%） |
| max-model-len | — | — | — | **0**（不进运行时驻留） | ✅ 已裁决阴性（C4） |
| 副本 / TP / PP 布局 | — | — | — | — | ✅ 副本 > TP2 bf16（sweep6）；**TP2 fp8 未测**（§4 实验 1）；**PD 分离未测**（§2 口子 4） |

### 2. 六个「没被想到、但确实存在」的口子（本次新增，按价值排序）

**口子 1 ★ GDN 状态自身的 dtype。** 文档里「占大头的那部分状态**压不动**」
从未被验证，是假设。用 sweep2 两组数反推：

```
bf16 KV：9.89 GiB /  99,267 tok = 106,983 B/token
fp8  KV：9.58 GiB / 136,533 tok =  75,348 B/token
解得：GDN 状态 = 43,713 B/token（占池 58%），真 KV(bf16) = 63,270 B/token
```

KV 量化只动了那 42%。**若 vLLM 以 fp32 存 GDN 状态**（混合架构常见），压到
bf16 即状态 ×2 → 池 75,348 → 53,491 B/token，**容量 ×1.41**，不动权重、不动
算子、不碰精度敏感的 1/4 层。**先花 1 分钟读源码确认 dtype**，再决定是否排实验。
这一条若成立，容量轴最高优先级立即改判。

**口子 2 CUDA graph 捕获桶 vs 实际 batch 分布。** `cudagraph_capture_sizes` =
`[1,2,4,8,...,256]`；bench 要求 24 路但引擎态 Running 实际在 **18/20/24 间跳**。
批量落在两桶之间即退 eager/piecewise——纯性能损失，全程未看过。手段：显式指定
更密的桶（如 20-28 每 2 一格）。可解释 R2c TPOT 的非整数倍关系。

**口子 3 `--gpu-memory-utilization` 0.92→0.95。** 从未扫过。+0.01 ≈ +0.45 GiB
≈ 2 块；+0.03 = +1.35 GiB ≈ 7 块 → k=1 每请求 2 块 → **每副本 +3 路，双副本
48→54 路**，量级 +12%。最省事，风险是贴边界被邻座抢显存。

**口子 4 PD 分离（1 卡 prefill + 1 卡 decode）。** 两卡预算下**唯一未测的第三种
布局**，且正对我们最差的指标（TTFT 6.1s/4.9s）。混合架构额外有利：GDN 状态是
每序列固定大小、不随 token 长，跨卡传输量远小于纯 KV 模型。成本高（两引擎 +
KV 连接器），上窗口前先查 vLLM 0.28 对混合架构 PD 的支持程度。

**口子 5 `--enforce-eager`：把 CUDA graph 池还给 KV。** 0.55-1.13 GiB ≈ 3-6 块
≈ 每副本 +1.5-3 路。代价是每步重发射 kernel（mini-vllm 对比已证 host 开销在
小 batch 下占比可观）。**不是为了用它，是为了量出「graph 池值几路」。**

**口子 6 AWQ：1 条 grep 判生死，不要用窗口赌。** vLLM 在 sm80+ 会把 AWQ 自动
升级成 `awq_marlin`——若是，则我们测到的 +57% 不是「GPTQ 的问题」而是
**「Marlin 这一族 kernel 在 sm89 低 batch decode 上的 dequant 开销」**，这是更硬
的结论，AWQ 直接判死。若存在原生 awq kernel 路径才排实验，判据：**TPOT ≤ 50ms
（dequant 开销 < fp8 的 1.1 倍）且 @48 > 420**，两条都中才算翻案。

**附带（不是手段，是混杂）**：卡对差异。B0/B1 在 GPU2+3、R2 在 GPU1+2，且 GPU0
有邻座 43.8GB 常驻。**同配置在 (0,1)/(1,2)/(2,3) 上的差异从没测过**——这是唯一
能让 619 vs 372 失效的东西，跑一次当「卡对基线」。

### 3. 指标清单（五层；无分母的数字不许进表）

**A 端到端（`vllm bench serve`）**——对外只报这层：输出吞吐（**必同报成功请求数
+ Benchmark duration** 以便验算）、**TTFT mean/median/P99**、**TPOT
mean/P50/P99**（开 MTP 时必须与接受率同报，否则一次前向多 token 会稀释 TPOT，
单看 55.6ms 是假数）、Peak concurrent、Failed（>0 即整组作废）。

**B 引擎内部（`/metrics` + `loggers.py`）**：Running/Waiting、**GPU KV cache
usage %**、prefix cache hit rate（random 数据集恒 0，必须声明）、投机接受率
（**分位置**，分母显式）。**采样时点改为每 bench 点 3 次（前/中/后）**——sweep6
整相只采一次，R2b_c96t 的引擎态就是这么丢的。

**C 显存台账（启动日志 + 负载中 nvidia-smi）**：权重 / 峰值激活 / CUDA graph 池 /
KV+Mamba 池（`Available` GiB + `N tokens` + `X.XXx`），**四项加总必须闭合到
0.92 × 44.5**。sweep6 的 grep 漏了 `Available KV cache memory` /
`Estimated CUDA graph memory` / `model weights take` / `non-default args`，本次全采。

**D 数值**：C-Eval 200 题 + McNemar；贪心逐 token 一致率 n/8 探针。

**E 窗口卫生**：in-window 锚点首尾偏差 **>3% 整窗作废**；每相邻座 GPU 快照；
负载中 GPU 利用率快照。

### 4. 实验设计

窗口顺序 **`B0锚 → B1 → R2c → R2b扫描 → B0锚`**，全程同窗，B0 首尾偏差定窗口有效性。

| # | 臂 | GPU | 配置 | 动的旋钮 | 对照 | 成本 |
|---|---|---|---|---|---|---|
| 0 | **B0** | 2,3 | TP2 bf16（现基线） | — | — | 已有 |
| 0 | **B0-16** | 2,3 | 同上 @16 | 无（补测） | 修 §0-1 跨窗 | 1 bench |
| 0 | **B0-ledger** | 2,3 | 启动即采全台账 | 无（补采） | 修 §0-1 缺口 | 0 |
| 1 | **B1** | 2,3 | TP2 **fp8 权重 + fp8 KV** | **量化** | B0 → **拆因** | 1 serve/1 bench |
| 2 | **R2c** | 1,2 | 2× fp8 副本 + k=1 | 布局 | B1 @48 + 复现 618.8 | 已有/1 bench(@72t) |
| 2 | **R2b** | 1,2 | 2× fp8 副本 k=0 | spec 深度 | R2c | @120t/@144t |
| 3a | grep | — | AWQ marlin 升级条件 | — | 判口子 6 | 1 分钟 |
| 3b | R2c-16 | 1,2 | k=2 @16total | spec 深度（离墙远） | k=1 @16total | 1 bench |
| 3c | AWQ | 1,2 | 仅当 3a 判活 | 量化算子 | B1 | 1 serve/1 bench |
| 3d | 精度 | — | C-Eval 仅当 B1 为新配置 | — | fp8 0.775 | ~30min |

**拆因是本次核心**（§0-2 的缺陷 3）：`372 (TP2 bf16) --量化--> X (TP2 fp8)
--副本替代 TP--> 619 (R2c)`。它同时回答一个必被问、现在答不上的问题：
**「fp8 既然能装单卡，双卡为什么不量化完接着 TP2？」** 现在只能说「副本比
**bf16 的** TP2 好」，不能说「副本比 TP2 好」——B1 就是那个缺的数。

### 5. 明确不做（逐条理由，免得被问时现编）

int8（与 fp8 同 1 字节/参数，容量轴零增益）、int4 KV（该版本无 kernel）、
MTP k=3（接受率与块税双输）、TP4（PCIe 无 NVLink，TP2 已证通信受限）、
4 卡（与基线无关，另窗口独立问题）、TP2 @144（信息量低；TP2 的极限是延迟问题
不是容量问题——@48 时 usage 仅 43.9%）。

### 6. 公平清单

同总 prompt 数；同 bench 工具/workload（random 1024/256、temp 0）/指标；
每臂启动即采台账；每点 3 次引擎态；解析器对真实输出自测过（坑 12 的 nan）；
每相邻座快照；**卡对差异声明**（B0/B1 在 GPU2+3 vs R2 在 GPU1+2，若 B1 在两
卡对差异 >5% 则结论必须声明此混杂）；prefix cache 恒 0 声明。

### 7. 预注册预测与判据

- **B0-16**（基准窗 TP2 @16）≈ 270-310。sweep6 窗的 @48 比历史高 6.2%，
  @16 可能同向偏高。**锚点判据**：B0 首尾偏差 >3% → 整窗作废。
- **B1**（TP2 fp8）@48 ≈ **430-520**（权重读减半改善 TPOT，372→~480 量级）。
  **判据双向都有故事**：≥420 则「量化在双卡上单独成立」（且 619 的 +66% 可拆为
  量化 × 布局两段）；若 ≤372 则「量化在 TP2 布局上零收益甚至负收益」——
  那 372→619 全部来自布局，量化只是**使能器**，这同样是干净的结论。
- **R2b @120t/@144t**（60/72 路每副本，墙在 ~47）≈ 640-700 但 **TTFT >8s**
  （过墙排队）。**判据**：@144t 若 >650 且 TTFT <8s → 天花板不止 620，实验 2 翻案。
- **R2c @72t**（36 路每副本，超 24 路墙）≈ 600-650 且 **Waiting 暴涨**
  ——验证 94.4% 是真墙而非采样点巧合。
- **R2c-16（k=2 @16total）**：若 k=2 赢 k=1 → **动态 k 策略成立**（低负载 k=2、
  高负载退 k=1），这是可直接进生产的结论；若输 → 「k 是容量旋钮」进一步收敛为
  「墙远用 k=2、墙近用 k=1」的单调解。

### 8. 脚本改动清单（设计定了才动脚本）

相对 `mtp_bench6.sh`：①`startup_lines` 补采四项台账（§3-C）；②`summarize_dual`
正则改 `[^\n:]*:`（坑 12 的 nan，已在 bench7 草稿修好并自测）；③`engine_stats`
每点 3 次；④新增 B1 臂与 B0-16 补测；⑤4 卡相位改 opt-in（`RUN_4CARD=1`）——
基线是两张卡，4 卡不参与基线对比。

## 复现

### sweep7 运行手册（命令、相位、证据落点）

**一条命令 = 一臂。** 协议是「一次一个实验、记录完整、分析后回想遗漏，再决定下一步」，
所以脚本有相位闸门；**不设 `PHASES` 时仍是预注册的完整窗口**（`B0 B1 R2c R2b B0p`），
脚本的默认行为与预注册文档永远一致。
（下文的 `<root>` = 远端放脚本的工作目录，`<nvme-root>` = 缓存与证据落点，
`<venv>` / `<model-dir>` 同前。）

~~~bash
# 只跑基线臂（本次这么跑的）：
cd <root> && PHASES=B0 setsid nohup bash <root>/mtp_bench7.sh \
  > <root>/mtp7_run.log 2>&1 < /dev/null &

# 下一臂（分析完再决定跑哪几个）：
cd <root> && PHASES=B1   setsid nohup bash <root>/mtp_bench7.sh > <root>/mtp7_run.log 2>&1 < /dev/null &
PHASES="R2c R2b"         setsid nohup bash <root>/mtp_bench7.sh > <root>/mtp7_run.log 2>&1 < /dev/null &
# 全窗口 + 可选相位：
PHASES="B0 B1 R2c R2b B0p" RUN_EXTRA=1 RUN_4CARD=1 bash <root>/mtp_bench7.sh
~~~

**证据落点**（全部落在 <nvme-root>，不占容器 overlay，见坑 14）：

| 文件 | 内容 |
| --- | --- |
| `<out-dir-7>/run.log` | 时间线（**容器 UTC**，坑 3） |
| `<out-dir-7>/server_<arm>.log` | 每臂完整服务日志（台账来源） |
| `<out-dir-7>/server_startup_lines.txt` | 内存台账：KV 池 / CUDA graph / 权重字节 / non-default args（坑 12 修好的那批） |
| `<out-dir-7>/bench_<arm>[_pN].log` | `vllm bench serve` 原始输出（指标的**唯一**真源） |
| `<out-dir-7>/summary.txt` | 抓好的指标汇总（单卡臂 `summarize_single`，副本臂 `summarize_dual` 出 SUM/MEAN） |
| `<out-dir-7>/engine_stats_<arm>_s{1,2,3}.txt` | 每点 3 次引擎态（Running/Waiting/KV usage），sweep6 只采 1 次丢了 R2b_c96t |
| `<out-dir-7>/gpu_snapshots.txt` | 每相前后 GPU 快照（邻座证据） |

**记录规矩**：`bench_*.log` 是原始证据，任何时候都能重算；`summary.txt` 只是方便看，
**数字对不上以 `bench_*.log` 为准**。抓取正则有坑 12 的前科，新表必先对真实输出自测。

**脚本内含的全部启动/压测参数**（照抄即可，不用手敲）：

~~~bash
# 服务（两卡 TP2 臂；副本臂见 boot_replica）
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 <venv>/vllm serve <model-dir> \
  --port 8343 --served-model-name qwen38-27b --max-model-len 8192 \
  --tensor-parallel-size 2 --max-num-seqs 128 [--quantization fp8 --kv-cache-dtype fp8]

# 服务（单卡副本臂）
CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 <venv>/vllm serve <model-dir> \
  --port 834<1|2> --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

# 压测（每路副本各发一份，吞吐相加、延迟取均值）
<venv>/vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --tokenizer <model-dir> --dataset-name random --random-input-len 1024 \
  --random-output-len 256 --temperature 0 --num-prompts <N> \
  --max-concurrency <C> --host 127.0.0.1 --model qwen38-27b --port <port>

<venv>/python mtp_greedy.py <port> <ARM>   # 贪心等价探针（需要时）
~~~

## 坑（下次直接抄）

1. serve 必须 `setsid bash -c "..."` 包一层再后台：否则 stop 阶段按进程组
   kill 会把整个 sweep 脚本一起带走（sweep1 的 M1 臂赔过一次）；
2. V1 引擎退出常带 SIGABRT（exit 134/1）——JSON 已写完即算成功，各臂独立容错；
3. 容器时钟 UTC：run.log 里 02:08 ≠ 北京 02:08，对表先想清楚时区；
4. spec 配置会自动改写 `max_num_scheduled_tokens` 并缩 KV 池——归因时必须
   拆开（见公平清单 side-effect 行）；
5. 共享机 GPU 被他人批任务占用时，用"等卡 watcher"（轮询 mem+util，空闲即
   自动开跑），比人工盯守靠谱；GPU0 常驻一个 ~2.6GB 外部 daemon（pid 自
   2026-09-08 起），它的空闲阈值要放宽到 4,000 MiB，两卡标定才能起步；
6. AWQ 管线在 llmcompressor 0.13 里是 deprecated 工厂函数，transform 半边
   **没有 targets/ignore 过滤**（会碰 GDN 权重做 scale 改写）——混合精度
   要过滤层的量化走 GPTQModifier（同为 W4A16 格式）；
7. **compressed_tensors 的 targets/ignore 匹配语义是「精确名相等 或 `re:`
   前缀正则（re.match 锚定开头）」**（`compressed_tensors/utils/match.py`
   `match_name`）——glob 通配符（`model.visual.*`、`*linear_attn*`）**静默
   永不命中**，不报错不警告。首轮 int4 脚本因此差点把 GDN+mtp+visual 全
   量化掉（唯一护栏是 visual fc2 的 4304 列不被 128 整除触发的 divisibility
   报错）。两个教训：(a) ignore 写法必须是 `re:.*linear_attn.*` 这类；
   (b) **防呆校验必须复刻框架自己的匹配器**（脚本内 `_match_name` 副本），
   用另一套语义（如 fnmatch）校验等于自己骗自己。修正版断言还顺带排除了
   `self_attn.{q,k}_norm`（RMSNorm 非 Linear，名字含 self_attn 会被误数）；
8. 容器内 `ps` 看不到宿主机 PID（PID 命名空间隔离）——查他人 GPU 进程要
   从宿主机侧 `ps`，否则会把"进程不存在"误判成"任务已结束"；
9. llmcompressor save 的 checkpoint **不带视觉组件**（`preprocessor_config.json`
   / `video_preprocessor_config.json` / `vocab.json` / `merges.txt`）——起服务时
   `get_image_processor_dict` 直接 OSError（fp8 轮 §12.4 同款坑）。修法：从源
   模型目录抄四个文件进 checkpoint。bench5 首跑因此 ABORT 一次，重启即过；
10. **ceval_eval.py 只能在装了 `datasets` 的 venv 里跑**（lc-env），vllm-env 没有
   ——bench 脚本里内嵌 ceval 步骤会静默降级成错误。int4 轮的 C-Eval 因此
   未跑成；补跑：起 int4 服务后用 `<lc-venv>/bin/python ceval_eval.py
   --port 8331 --tag W4 --model-name qwen38-27b`（tag 用 w4 与 fp8 轮对齐）。
11. **补跑 ceval 时别把 `HF_HUB_OFFLINE=1` 带进去**：脚本靠 `setdefault`
   自配 `HF_ENDPOINT=hf-mirror` + 在线——`get_dataset_config_names` 需要到
   mirror 拉 C-Eval 科目列表（数据本体才走本地缓存）。强压离线后 config
   枚举退化为 `['default']`，而缓存里只有科目名 config → 5 秒即崩
   （`Couldn't find cache for ceval/ceval-exam for config 'default'`，
   错误信息里还把全部科目列出来误导你以为缓存坏了）。首次补跑因此白起了
   一次服务。
12. **从 bench 文本输出抓指标时，正则的冒号要跟在完整标签后**：行格式是
   `Output token throughput (tok/s):    164.01`——冒号在 `(tok/s)` 之后
   不在 key 之后，`key + r":"` 的正则静默失配返回 nan（sweep6 的
   summarize_dual 因此全输出 nan，总吞吐改为手工从 p1/p2 相加）。教训同
   match_name 坑：**校验解析器要对着真实输出格式测一次**，别对着想象中的
   格式写。
13. **`\"` 只在「双引号 `bash -c "..."` 内部」这一层是对的，提到顶层单引号
   赋值就变成字面反斜杠**。bench6 把 spec JSON 直接内联在 `bash -c "..."` 行
   里，所以 `'{\"method\": ...}'` 正确；bench7 把 k=1 的 config 提成顶层
   `SPEC1='...'`，照抄同一串就错了——变量展开不会二次处理转义，vLLM 收到
   `{\"method\": ...}` 直接 `JSONDecodeError` 拒绝启动。**检测方法**：拿一个
   打印 argv 的 stub 脚本顶替 `<venv>/vllm`，把 `boot_replica` 的引号结构原样
   跑一遍再 `json.loads`。不测的代价极不对称：B0（bf16，不带 spec）会正常跑
   完，只有 R2c/R2b 两个副本臂会在起服务时崩——**整窗白烧，而且看起来像
   「副本架构不行」而不是「脚本写错了」**。
14. **overlay 被写满是这条线的结构性风险，不是一次意外**。容器只有 overlay
   一个可写层，而 vLLM / torch / HF / pip / uv / Triton 默认全写 `$HOME` 和
   `/tmp`；这一线从来没设过 `VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` / `TMPDIR`
   / `XDG_CACHE_HOME`，于是每个权重下载、每个 venv、每份编译缓存都落在
   overlay 上。2026-09-11 撑到 894G/894G（剩 635MB），**症状是 vLLM 根本起不
   来**。宿主机 NVMe 是 bind mount、默认没有任何指针指向它。修法：脚本头部导
   出三个缓存变量 + `OUT` 一起指向 NVMe，模型目录走软链（路径零改动）。同机
   其他线（Qwen3-4B / Qwen2.5-Omni / relax-*）早就把 `models/ exps/ repos/`
   放在 NVMe 上了，只有本线堆在 overlay。
15. **共享机上只动自己这条线的目录和文件**。清理前必须按「归属」分三档，不是
   按「大小」：**(a) 本线自己造的、可重跑** → 可删（构建产物、编译缓存、临时
   checkpoint）；**(b) 别线的 venv / 模型 / 数据缓存** → 一律不碰，删掉要对方
   重装重下，而且可能正好打断别人挂在后台的 watcher（本机就有几处按空闲轮询
   自动开跑的 watcher 脚本）；**(c) 用途不明** → 只报不动。判断归属看
   创建时间 + 名字里有没有本线的标识，拿不准就归 (c)。
   **搬迁同理**：本线的模型目录可以移到共享盘再留软链（路径零改动、可逆），
   别人的模型目录即使腾出的空间更多也不动。

## 文件清单

| 文件 | 内容 |
|---|---|
| `mtp_bench.sh` / `mtp_bench2.sh` | 两轮 sweep 脚本（头部 docstring 写明臂定义与基线选择理由） |
| `mtp_bench3.sh` | C2b 臂（显式 sched 8192 排除法） |
| `mtp_bench4.sh` | C4 臂（len 4096 容量墙解锁阴性对照，docstring 含预注册预测） |
| `int4_calib.py` | GPTQ W4A16 混合精度标定（ignore 名单 + 防呆断言 + docstring 写选型理由） |
| `mtp_bench5.sh` | W4 臂 bench（int4 checkpoint + fp8 KV + MTP k=2） |
| `chain_int4.sh` | 串联编排：C4 完成 → 等两卡空闲 → 标定 → W4 bench |
| `ceval_w4.sh` | W4 C-Eval 补跑包装（起 int4 服务 + lc-env 跑 200 题；坑 10/11 的现场） |
| `mtp_greedy.py` | 贪心等价探针（8 题固定 prompt） |
| `greedy_diff2.py` / `greedy_diff3.py` | 逐 token 对照（含 within-arm 对照优先的设计） |
| `greedy_diff4.py` | A(sweep1) vs W4 对照（含首翻位置报告） |
| `greedy_diff_c2c4.py` | C2 vs C4 对照（len 数值中性的证据） |
| `sweep{1,2}/summary.txt` | 各臂各并发档的关键指标汇总 |
| `sweep{1,2}/run.log` | 臂启动/完成时间线（含锚点） |
| `sweep{1,2}/bench_*.log` | bench 原始输出 |
| `sweep{1,2}/specmetrics_*.txt` | /metrics 的 spec/accept/draft 指标 |
| `sweep{1,2}/greedy.txt` | 贪心探针原始 JSON（token 序列 + 文本） |
| `sweep{1,2}/server_startup_lines.txt` | 各臂启动配置/池大小/WARNING 摘录（全量 server 日志留测量机） |
| `sweep2/env_2026-09-11.txt` | 环境锁档 |
| `sweep3/summary.txt` + `bench_C2b_*.log` | C2b 臂结果（sched 排除法） |
| `sweep3/engine_stats_{K2,M2,C2b}_c48.txt` | **容量墙实锤**：spec 臂 Running 16-17 / usage 90-96% vs K2 满载 48 |
| `sweep4/`（C4 全套：summary/bench×2/engine_stats×2/specmetrics×2/greedy/startup_lines） | **len 4096 阴性对照**：@48 298.21（<340 判据）、Running 仍 16-17、KV 字节同 9.28 GiB、C2 vs C4 贪心 8/8 |
| `sweep5/` | W4 全套：summary/bench×2/engine_stats×2/specmetrics×2/greedy/startup_lines/**ceval_W4.log**（路径脱敏同上） |
| `mtp_bench6.sh` | sweep6 编排：TP2 锚点 + R2b/R2c/R2a 副本对四相（docstring 含臂设计与邻座条件） |
| `sweep6/` | 2 卡部署矩阵全套：summary（含 nan bug 现场）/bench×10/engine_stats×4/startup_lines（池打印 129,706 vs 历史 136,533 的邻座足迹证据）/gpu_snapshots×5/specmetrics×2 |

对照记录：TP2 bf16 基线数字来自部署实验同窗 ABBA（@16 284.9±6.5 / @48 350.2±3.3）。

## 修正轨迹（归因的自我纠错）

1. **verify 算力假设 → 排除**：初判 @48 MTP 亏损是"批量饱和后 verify 多算
   不划算"；TPOT/ITL 对账推翻——verify 步仅 +4.6% 时间换 2.35 倍 token；
2. **调度节流假设 → 排除**：改判 spec 强制 `scheduled=2048` 的 prefill 节流；
   C2b（显式 8192）@48 不动推翻；
3. **max_len 预留假设 → 开跑前撤销**：C4 臂初始预测"len 减半 → 记账减半"；
   读 `single_type_kv_cache_manager.py` 后发现准入门按实际 token 记账，
   预测改为阴性（见 sweep4/5 预注册节）；
4. **最终定稿**：容量墙 = draft 权重缩池（块数）+ 每请求 2 快照块（驻留），
   引擎统计与源码对账逐位闭合。
5. **W8A8 误判 → 口头汇报后撤销**（2026-09-11）：向用户汇报时把「W8A8
   fp8 dynamic」当成未测牌推荐——实际 A 臂 `--quantization fp8` 就是
   W8A8 dynamic（bf16 checkpoint 在线压 fp8，激活动态量化），整条阶梯
   一直跑在 fp8 GEMM 上，TPOT 46.3ms 已是这套内核栈的 fp8 速度顶。
   教训：给自己开「新牌」前先对一遍已有臂的 serve 标志位。
