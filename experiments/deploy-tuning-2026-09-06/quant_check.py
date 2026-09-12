#!/usr/bin/env python3
# 量化精度土回归: 20 题并行打两个服务, b=TP2 bf16 参照, a=单卡 fp8权重+fp8KV
import json, urllib.request, concurrent.futures as cf

PROMPTS = [
    "用三句话解释什么是 TCP 拥塞控制，面向大一学生。",
    "9.11 和 9.8 哪个大？请一步步比较。",
    "把这句话翻译成英文：这个方案在低并发场景下收益有限，需要实测验证。",
    "写一个 Python 函数，判断一个字符串是否是回文，忽略大小写和标点。",
    "简述 Transformer 中 LayerNorm 的作用，以及 Post-LN 和 Pre-LN 的区别。",
    "一个水池，进水管每小时进 5 吨水，出水管每小时出 3 吨，水池容量 40 吨，初始 10 吨，多久灌满？",
    "用一句话概括《三体》第一部的主要情节。",
    "解释什么是大 O 记号，给出 O(n log n) 的一个常见例子。",
    "3 个红球 5 个蓝球，不放回摸两个，都是蓝球的概率是多少？给出分数。",
    "把 2026年9月6日 转换成英文的正式写法（如信件落款）。",
    "写 SQL：从 orders 表（user_id, amount, created_at）查 2026 年 8 月消费总额最高的前 5 名用户。",
    "为什么天空是蓝色的？用不超过 100 字回答。",
    "求 1 到 100 中所有能被 3 整除但不能被 5 整除的数之和。",
    "用面向对象的思想描述一个电梯调度系统的类设计（只列类名和核心方法）。",
    "strawberry 这个单词里有几个 r？",
    "简述 KV cache 在自回归解码中的作用，以及为什么能加速。",
    "把下面这段话改写成更正式的商务邮件语气：这个东西我们下周再说吧，我最近太忙了。",
    "12 个球有一个天平，其中一个球重量异常，最少称几次能找出坏球并判断轻重？",
    "解释 HTTP 401 和 403 的区别。",
    "用贪心策略说明找零钱问题（面额 1,5,10,25）为什么贪心能得到最优解。",
]

def ask(port, model, p, tag, i):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}],
                       "max_tokens": 1536, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    txt = out["choices"][0]["message"]["content"]
    with open(f"<out-dir-qcheck>/{tag}_{i:02d}.txt", "w") as f:
        f.write(txt)
    print(tag, i, "tokens:", out["usage"]["completion_tokens"], "finish:",
          out["choices"][0]["finish_reason"], flush=True)

def run(port, model, tag):
    with cf.ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(ask, port, model, p, tag, i + 1) for i, p in enumerate(PROMPTS)]
        for f in cf.as_completed(futs):
            f.result()

import os, time
os.makedirs("<out-dir-qcheck>", exist_ok=True)
t0 = time.time()
run(8322, "qwen38-27b-tp2", "b")
print("bf16 reference done", time.time() - t0, flush=True)
run(8321, "qwen38-27b", "a")
print("fp8 done", time.time() - t0, flush=True)
print("QC_DONE")
