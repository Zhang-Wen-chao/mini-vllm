"""mtp5 greedy: W4 (int4 weights, fp8 KV + MTP k=2) vs A(sweep1 baseline).
W4 changes target-model numerics on two axes at once (GDN fp8->bf16,
FFN/attn fp8->int4), so token-level divergence vs A is expected to widen.
The interesting reads: (1) does the int4 arm still produce semantically
equivalent rewrites at tied-logit positions (same signature as M/K arms),
or does it produce real degradation; (2) does the W4 greedy output stay
stable against W4 itself across the two concurrency probes."""
import json

def load(path):
    arms = {}
    buf = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if line.endswith("]"):
            try:
                chunk = json.loads("".join(buf))
            except json.JSONDecodeError:
                continue
            for rec in chunk:
                arms.setdefault(rec["arm"], {})[rec["prompt"]] = rec
            buf = []
    return arms

mtp = load("<out-dir>/greedy.txt")
mtp5 = load("<out-dir-5>/greedy.txt")

def cmp(ra, rb, la, lb):
    same = diff = 0
    details = []
    for p in ra:
        x, y = ra[p], rb.get(p)
        if y is None or "error" in x or "error" in y:
            continue
        if x["tokens"] == y["tokens"]:
            same += 1
        else:
            # first divergence position within the token stream
            pos = next((i for i, (a, b) in enumerate(zip(x["tokens"], y["tokens"])) if a != b),
                       min(len(x["tokens"]), len(y["tokens"])))
            diff += 1
            details.append(f"{p[:14]}@{pos}")
    print(f"{la} vs {lb}: same={same} diverge={diff} {details}")

cmp(mtp["A"], mtp5["W4"], "A(sweep1)", "W4(int4+fp8KV+MTP2)")
