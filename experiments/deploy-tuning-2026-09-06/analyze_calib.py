#!/usr/bin/env python3
# 对比三组 20 题输出: a=fp8+fp8KV scale-1.0, b=TP2 bf16 参照, c=fp8+fp8KV 标定scale
# 核心问题: 标定后 c 是否比 a 更接近 bf16 参照 b
import difflib, os

def final_answer(path):
    txt = open(path).read()
    if "</think>" in txt:
        return txt.split("</think>", 1)[1].strip(), False
    return "", True

rows = []
for i in range(1, 21):
    fa, ca = final_answer(f"<out-dir-qcheck>/a_{i:02d}.txt")
    fb, cb = final_answer(f"<out-dir-qcheck>/b_{i:02d}.txt")
    fc, cc = final_answer(f"<out-dir-qcheck>/c_{i:02d}.txt")
    rows.append((i,
                 difflib.SequenceMatcher(None, fa, fb).ratio() if fa and fb else None,
                 difflib.SequenceMatcher(None, fc, fb).ratio() if fc and fb else None,
                 ca, cc, len(fa), len(fb), len(fc)))

ab = [r[1] for r in rows if r[1] is not None]
cb = [r[2] for r in rows if r[2] is not None]
print(f"可比题数: a-b {len(ab)}, c-b {len(cb)}")
if ab:
    print(f"scale-1.0(a) vs bf16(b): mean {sum(ab)/len(ab):.3f}")
if cb:
    print(f"标定(c)     vs bf16(b): mean {sum(cb)/len(cb):.3f}")
both = [(r[1], r[2]) for r in rows if r[1] is not None and r[2] is not None]
if both:
    closer = sum(1 for x, y in both if y > x)
    tie = sum(1 for x, y in both if abs(y - x) < 1e-9)
    print(f"逐题对比: c 更接近 b {closer} 题 | a 更接近 {len(both)-closer-tie} 题 | 完全相同 {tie} 题")
print("截断: a", sum(r[3] for r in rows), "c", sum(r[4] for r in rows))
print("\n明细: 题 a-b相似 c-b相似 a截 c截 len(a)/len(b)/len(c)")
for r in rows:
    print(f"{r[0]:3d}  {r[1] if r[1] is not None else '—':>6} {r[2] if r[2] is not None else '—':>6}  {int(r[3])}{int(r[4])}  {r[5]}/{r[6]}/{r[7]}")
