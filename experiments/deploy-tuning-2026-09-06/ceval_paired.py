#!/usr/bin/env python3
# 三组 C-Eval 200 题配对分析: a2=scale1.0, b2=bf16参照, c2=标定scale
import json, itertools

def load(tag):
    preds = {}
    for i in range(1, 201):
        d = json.load(open(f"<out-dir-ceval>/{tag}_{i:03d}.json"))
        preds[i] = d["pred"]
    return preds

a, b, c = load("a2"), load("b2"), load("c2")
answers = {i: None for i in range(1, 201)}

# 正确性矩阵
import glob
# 从 a2 文件重建 answer 顺序不可行 — 用分数文件没有 answer。改为从数据集重建固定抽样。
import os, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset, get_dataset_config_names
rows = []
for cfg in get_dataset_config_names("ceval/ceval-exam"):
    ds = load_dataset("ceval/ceval-exam", cfg, split="val")
    rows += [{"answer": r["answer"], "subject": cfg} for r in ds]
rng = random.Random(42)
picked = rng.sample(rows, 200)
ans = [p["answer"] for p in picked]

acc = lambda p: sum(1 for i in range(200) if p[i + 1] == ans[i]) / 200
print(f"acc a2={acc(a):.4f} b2={acc(b):.4f} c2={acc(c):.4f}")

def mcnemar(x, y, nx, ny):
    b01 = sum(1 for i in range(200) if x[i + 1] == ans[i] and y[i + 1] != ans[i])
    b10 = sum(1 for i in range(200) if x[i + 1] != ans[i] and y[i + 1] == ans[i])
    agree = sum(1 for i in range(200) if x[i + 1] == y[i + 1])
    return b01, b10, agree

for (x, y, nx, ny) in [(a, b, "a2(scale1.0)", "b2(bf16)"),
                       (c, b, "c2(标定)", "b2(bf16)"),
                       (a, c, "a2(scale1.0)", "c2(标定)")]:
    b01, b10, agree = mcnemar(x, y, nx, ny)
    print(f"{nx} vs {ny}: 配对分歧 x对y错={b01} x错y对={b10} 预测一致率={agree}/200")
