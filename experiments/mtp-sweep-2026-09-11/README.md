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

## 复现

~~~bash
# 每臂（脚本内含全部参数；ARM/SPEC 由脚本轮转）：
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 <venv>/vllm serve <model-dir> \
  --port 8331 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 \
  [--kv-cache-dtype fp8] [--speculative-config '{"method":"mtp","num_speculative_tokens":2}']

<venv>/vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --tokenizer <model-dir> --dataset-name random --random-input-len 1024 \
  --random-output-len 256 --temperature 0 --num-prompts 96 \
  --max-concurrency {48,16} --host 127.0.0.1 --model qwen38-27b --port 8331

<venv>/python mtp_greedy.py 8331 <ARM>   # 贪心等价探针
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

对照记录：TP2 bf16 基线数字来自部署实验同窗 ABBA（@16 284.9±6.5 / @48 350.2±3.3）。

## 修正轨迹（归因的两次自我纠错）

1. **verify 算力假设 → 排除**：初判 @48 MTP 亏损是"批量饱和后 verify 多算
   不划算"；TPOT/ITL 对账推翻——verify 步仅 +4.6% 时间换 2.35 倍 token；
2. **调度节流假设 → 排除**：改判 spec 强制 `scheduled=2048` 的 prefill 节流；
   C2b（显式 8192）@48 不动推翻；
3. **max_len 预留假设 → 开跑前撤销**：C4 臂初始预测"len 减半 → 记账减半"；
   读 `single_type_kv_cache_manager.py` 后发现准入门按实际 token 记账，
   预测改为阴性（见 sweep4/5 预注册节）；
4. **最终定稿**：容量墙 = draft 权重缩池（块数）+ 每请求 2 快照块（驻留），
   引擎统计与源码对账逐位闭合。
