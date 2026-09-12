# 部署调优轮（Qwen3.8-27B × L20）：ABBA 同窗配对 / 量化精度土回归 / C-Eval

> **本目录是 2026-09-12 补档**。这批脚本与证据自 2026-09-06 产出后**只存在于测量机**，
> 从未归档到任何仓库（`experiments/mtp-sweep-2026-09-11/` 收的是 09-11 之后的 MTP 轮，
> 不含这一轮）。补档动作发生在测量机侧清理前，属于「先把唯一副本拿回来」。
> 详见 `experiments/mtp-sweep-2026-09-11/README.md` 的「归档缺口」一节。

## 这是什么

MTP 轮之前的那一轮部署调优（2026-09-06）。三件事：

1. **ABBA 交替配对 ×5**——把「单卡 fp8 权重+fp8 KV」与「TP2 bf16」放在**同一窗口**
   里交替跑，用配对抵消宿主机共享负载。这一轮之前的单轮结论是「TP2 @16 反超 +30%」，
   ABBA 把它推翻了。
2. **量化精度土回归（20 题，逐题保留原文）**——a/b/c 三组输出对照，回答
   「fp8 量化掉不掉精度」以及「标定出来的 scale 是否比 scale-1.0 更接近 bf16」。
3. **C-Eval 200 题正式回归**——用 `/v1/completions` + `logprobs` argmax +
   `allowed_token_ids` 绕开 chat/思考模板，四组各 200 题逐题 JSON。

## 目录

```
abba/                 ABBA 配对原始输出（10 个 round log + summary.txt）
qcheck/               20 题土回归逐题原文：a_*/b_*/c_* 各 20 份
ceval/                C-Eval 200 题逐题 JSON ×4 组 = 800 份
logs/                 9 份服务端/标定端原始日志（见下「服务端配置」）
abba_bench.sh         ABBA 编排（A=8321 单卡 fp8，B=8322 TP2 bf16，交替 5 轮）
quant_check.py        20 题并行打两个服务的采集端
fp8_calib.py          LLM Compressor 标定脚本（产出带 kv_cache_scheme 的 fp8 checkpoint）
qcheck_calib.py       同上（标定那一路的采集端）
analyze_calib.py      20 题三组对照 + 相似度统计（本目录证据可重算）
ceval_eval.py         C-Eval 200 题采集（logprobs argmax）
ceval_paired.py       三组配对分析 + McNemar
```

### 服务端配置（`logs/`）

**`abba_bench.sh` 只打 8321/8322 两个端口，不含怎么把这两个服务起起来**——
那些参数只在服务端日志里。补这批日志就是为了让「复现」这一步不断在最后一环：

| 日志 | 是什么 |
|---|---|
| `vllm_serve_fp8.log` | A 臂：单卡 8321，`--quantization fp8`（`non-default args` 那行原样记着 `max_model_len 8192 / max_num_seqs 128`） |
| `vllm_serve_tp2.log` | B 臂：TP2 bf16 8322（引擎 config 行含 `tensor_parallel_size=2`、`enable_prefix_caching=True`） |
| `vllm_serve_fp8kv.log` / `_noprefix.log` / `_sched.log` / `_32k.log` | fp8 KV 的四个变体（默认 / 关 prefix caching / 显式 sched / 32K 长上下文） |
| `vllm_serve_pp2.log` | PP2 臂（**只有服务端日志**，见下「本目录没有的」） |
| `vllm_serve_calib.log` | 标定 checkpoint 的服务（8323） |
| `fp8_calib.log` | `fp8_calib.py` 标定全过程原始输出 |

> `int4_calib.log` **不在这里**——它是 MTP 轮 sweep5 的 W4 臂标定，归
> `../mtp-sweep-2026-09-11/sweep5/`。

## 本目录没有的（别把它当成这一轮的完整证据）

补档只补回了**还在测量机上、且能被认出来的**那一部分。下列结论在文档里有记，
但**产出它们的证据不在本目录**，引用时不要指到这里：

| 结论 | 状态 |
|---|---|
| @16 三轮定稿：TP2 284.9 > 单卡 fp8 219.2 > PP2 173.5 | **无 bench 原始输出**（PP2 只留下了服务端日志） |
| KV fp8 只 +37.5%（Mamba 状态不量化） | 无 bench 原始输出 |
| 16384 预算 TTFT 3 倍恶化 = c1 差分定位排队侧 | 无 bench 原始输出 |
| prefix caching 收益边界（@16 −74% / @48 阴性）；32K 同卡并发优势 3.1x | 有 `logs/vllm_serve_fp8kv_noprefix.log`、`_32k.log` 两个配置，**无 bench 结果** |
| LLM Compressor 标定闭环踩的六条坑 | 有 `fp8_calib.py` + `fp8_calib.log`，**坑本身只写在文档里** |

**这一轮当时没有归档约定**（归档是从 09-11 的 MTP 轮才开始逐轮做的），
所以拿不回来的部分只能留在文档口径里。**本目录能独立重算的，就是下面这三块。**

## 可从本目录重算的数字（不依赖任何外部资源）

**ABBA（`abba/summary.txt`，96 条 / 客户端并发 48，1024 进 / 256 出，temperature 0）**

| 臂 | 配置 | 5 轮输出吞吐（tok/s） | 均值 | 单臂 5 轮极差 |
|---|---|---|---|---|
| A | 单卡 `--quantization fp8` + `--kv-cache-dtype fp8`（端口 8321） | 352.81 / 353.02 / 352.93 / 352.87 / 352.80 | **352.89** | **0.06%** |
| B | TP2 bf16（端口 8322） | 344.06 / 349.88 / 352.60 / 352.84 / 351.44 | **350.16** | **2.51%** |

→ **@48 两臂打平（A 比 B +0.78%）**。真正的差别在两边：
**单卡这条臂 5 轮极差只有 0.06%，TP2 那条是 2.51%——差 40 倍**。
同窗口同口径下，**TP2 的读数本身不稳**，这就是「宿主机共享负载污染所有 PCIe 方案」
的签名（TP2 走 PCIe 通信，单卡不跨卡）。
延时侧：Mean TTFT **A 5235.9 ms vs B 7876.6 ms**（单卡快 34%）、
Mean TPOT **A 113.6 ms vs B 105.7 ms**（TP2 略好）。
**@48 上「吞吐打平」是由「单卡省下三分之一的排队时间」换来的，不是算得更快。**

**20 题土回归（`qcheck/`，跑 `analyze_calib.py` 可重算）**

```
可比题数: a-b 18, c-b 18          （第 18/20 题被截断，不计入）
scale-1.0(a) vs bf16(b): mean 0.622
标定(c)     vs bf16(b): mean 0.682
逐题对比: c 更接近 b 11 题 | a 更接近 6 题 | 完全相同 1 题
```

→ **标定 scale 在小样本长生成上净改善**（0.622 → 0.682）。
**注意它的效力边界**：这是 20 题、18 题可比、用字符串相似度量的**土**回归，
不是精度集；下一节的 C-Eval 用的是单步 argmax，两者口径不同、结论也不同。

**C-Eval 预测一致率（`ceval/`，无需答案即可重算）**

| 对照 | 预测一致 | 比例 |
|---|---|---|
| `a2`(scale-1.0) vs `b2`(bf16) | 191/200 | **0.9550** |
| `c2`(标定) vs `b2`(bf16) | 187/200 | 0.9350 |
| `a2`(scale-1.0) vs `c2`(标定) | 186/200 | 0.9300 |
| `w4`(int4) vs `b2`(bf16) | 179/200 | 0.8950 |

四组的预测分布都在 A/B/C/D 之间大致均匀（各 40–59 题），**没有塌到某个字母**。
→ **fp8 的单步预测与 bf16 有 95.5% 重合**；标定把重合度降到 93.5%，
**但「更不一样」不等于「更不准」**——方向要答案才能判，见下。

## 需要数据集才能重算的数字（本目录只存了预测，没存答案）

`ceval_*.json` 只存 `pred` 与四选项的 `logprobs`，**答案键要从 `ceval/ceval-exam`
数据集重建**（`ceval_paired.py`：`random.Random(42).sample(rows, 200)` 固定抽样）。
所以下面这三个数**本目录不能独立重算**，它们是记录值：

```
bf16 0.7950 / fp8+scale-1.0 0.7750 / 标定 0.7700      （w4 = 0.7550）
McNemar 均不显著（a2 vs bf16 = 2:6）
```

结论：**fp8 在 MCQ 上没有显著回归；而标定 scale 在 MCQ 上没有增益**
（与上节土回归的 0.622→0.682 相反）。两条并存的解释是**口径差异**：
长生成轨迹对 scale 敏感，单步 argmax 不敏感。

> 要在别处重算：装 `datasets`、放开 `HF_HUB_OFFLINE`（脚本靠在线枚举科目列表，
> 强压离线会让 config 退化成 `'default'` 与缓存对不上，5 秒崩且错误信息误导），
> 再跑 `ceval_paired.py`。

## 脱敏

按 `experiments/mtp-sweep-2026-09-11/README.md` 的脱敏清单处理：模型目录、
虚拟环境、输出目录三类绝对路径换成 `<model-dir>` / `<venv>` / `<out-dir-*>`。

入库前逐文件扫过：**账号名 / 内网地址段 / 主机名 / 容器 ID / `/root/` 绝对路径
残留全部为 0**，非回环 IPv4 只剩 `0.0.0.0`（serve 的 bind 地址）。

**`logs/` 里有两处真地址，已打码**：`vllm_serve_tp2.log` 与 `vllm_serve_pp2.log`
各有一处 `mq_connect_ip=<host-ip>`（**只有走 multiprocess executor 的多卡臂才会打这行**，
TP=1 的单卡臂不打）——这与 `../mtp-sweep-2026-09-11/README.md` 记的是同一个来源，
按同一套约定换成 `<host-ip>`。

**代价同 MTP 那批**：脚本带占位符，**不能原样执行**——它们是协议记录，不是可运行
产物；复跑需先把占位符换回真路径（`analyze_calib.py` 换完可直接跑）。
