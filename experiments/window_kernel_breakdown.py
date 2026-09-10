"""Steady-decode window kernel breakdown from an nsys sqlite export.

Why this exists (2026-09-10 correction): the plain ``cuda_gpu_kern_sum``
report only counts DIRECTLY launched kernels — kernels executed inside CUDA
graph replays are excluded from it. A mostly-graph engine (mini) therefore
gets a denominator made almost entirely of its out-of-graph kernels, and an
out-of-graph loop like per-request argmax reads as ~half of "GPU time" when
its true share is single-digit percent. Query the NVTX window in the sqlite
instead: CUPTI_ACTIVITY_KIND_KERNEL there includes graph-internal kernels
(run nsys with --cuda-graph-trace=node, see profile_decode.py).

Prints per window: wall, kernel count/total/share, top kernels, sampler
kernels (argmax / gumbel), and sync API counts — the numbers that survive
cross-checking against torch.profiler (nsys=kineto).

Usage (sqlite made by: nsys stats --report cuda_gpu_kern_sum ... or
``nsys export --type sqlite``):

    python experiments/window_kernel_breakdown.py \
        experiments/nsys_mini_gpt2.sqlite experiments/nsys_vllm_gpt2.sqlite
"""

import argparse
import json
import sqlite3


def _category(name):
    """Coarse kernel classification for the GEMM/attention/small-op split."""
    n = name.lower()
    if "gemm" in n or "splitkreduce" in n or ("cutlass" in n and "wmma" in n):
        return "gemm"
    if "fmha" in n or "flash" in n:  # attention + paged KV append
        return "attention"
    return "other"  # residual add / layernorm / gather / fill / argmax / ...


def analyze(path, steps):
    con = sqlite3.connect(path)
    cur = con.cursor()
    rows = cur.execute(
        'SELECT start, "end" FROM NVTX_EVENTS '
        "WHERE text = 'steady_decode' ORDER BY start DESC LIMIT 1"
    ).fetchall()
    if not rows:
        return {"path": path.split("/")[-1], "error": "no steady_decode range"}
    ws, we = rows[0]

    kcount, ksum = cur.execute(
        'SELECT COUNT(*), SUM("end" - start) FROM CUPTI_ACTIVITY_KIND_KERNEL '
        "WHERE start >= ? AND \"end\" <= ?",
        (ws, we),
    ).fetchone()
    top = cur.execute(
        "SELECT s.value, COUNT(*), SUM(k.\"end\" - k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id "
        "WHERE k.start >= ? AND k.\"end\" <= ? "
        "GROUP BY s.value ORDER BY 3 DESC LIMIT 10",
        (ws, we),
    ).fetchall()
    sampler = cur.execute(
        "SELECT s.value, COUNT(*), SUM(k.\"end\" - k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id "
        "WHERE k.start >= ? AND k.\"end\" <= ? AND (s.value LIKE '%gumbel%' "
        "OR s.value LIKE '%ArgMax%') "
        "GROUP BY s.value ORDER BY 3 DESC",
        (ws, we),
    ).fetchall()
    syncs = cur.execute(
        "SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id "
        "WHERE r.start >= ? AND r.\"end\" <= ? AND s.value LIKE '%Synchronize%' "
        "GROUP BY s.value",
        (ws, we),
    ).fetchall()
    all_rows = cur.execute(
        "SELECT s.value, COUNT(*), SUM(k.\"end\" - k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id "
        "WHERE k.start >= ? AND k.\"end\" <= ? GROUP BY s.value",
        (ws, we),
    ).fetchall()
    con.close()

    def kern_row(r):
        name, c, t = r
        return {
            "name": name[:80], "count": c, "total_ms": round(t / 1e6, 2),
            "avg_us": round(t / c / 1000, 1), "per_step_us": round(t / 1e6 / steps * 1000, 1),
        }

    sampler_us = sum(r[2] for r in sampler) / 1e6 / steps * 1000
    cats = {}
    for name, c, t in all_rows:
        cat = _category(name)
        ms, cnt = cats.get(cat, (0.0, 0))
        cats[cat] = (ms + t / 1e6, cnt + c)
    return {
        "file": path.split("/")[-1],
        "window_ms": round((we - ws) / 1e6, 1),
        "steps": steps,
        "kernels": kcount,
        "kernels_per_step": round(kcount / steps, 1),
        "kernel_ms": round((ksum or 0) / 1e6, 2),
        "kernel_us_per_step": round((ksum or 0) / 1e6 / steps * 1000, 0),
        "kernel_share_of_wall": round(100.0 * (ksum or 0) / (we - ws), 1),
        "categories": {
            k: {"total_ms": round(v[0], 2), "count": v[1],
                "us_per_step": round(v[0] / steps * 1000, 0)}
            for k, v in sorted(cats.items(), key=lambda kv: -kv[1][0])
        },
        "sampler_kernels_us_per_step": round(sampler_us, 1),
        "sampler_share_of_kernel": round(100.0 * sampler_us / ((ksum or 0) / 1e6 / steps * 1000), 1),
        "sync_per_step": {a: round(c / steps, 1) for a, c in syncs},
        "top_kernels": [kern_row(r) for r in top],
        "sampler_kernels": [kern_row(r) for r in sampler],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+", help="nsys sqlite exports to query")
    ap.add_argument("--steps", type=int, default=99,
                    help="decode steps inside the steady_decode window "
                         "(gpt2 b=8 max_new=100 -> 99; see profile log)")
    ns = ap.parse_args()
    for p in ns.sqlite:
        print(json.dumps(analyze(p, ns.steps), ensure_ascii=False))
