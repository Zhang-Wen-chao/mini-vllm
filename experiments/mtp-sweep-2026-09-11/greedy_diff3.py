"""mtp2 greedy controls: K2 (fp8 KV, no spec) and C2 (fp8 KV + MTP k=2) vs A3.
Cross-run anchor: A3 (this run) vs A (mtp run) must be identical too."""
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
mtp2 = load("<out-dir-2>/greedy.txt")

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
            diff += 1
            details.append(p[:20])
    print(f"{la} vs {lb}: same={same} diverge={diff} {details}")

cmp(mtp["A"], mtp2["A3"], "A(sweep1)", "A3(sweep2)")
cmp(mtp2["A3"], mtp2["K2"], "A3", "K2(fp8KV)")
cmp(mtp2["A3"], mtp2["C2"], "A3", "C2(fp8KV+MTP2)")
cmp(mtp2["K2"], mtp2["C2"], "K2", "C2")
