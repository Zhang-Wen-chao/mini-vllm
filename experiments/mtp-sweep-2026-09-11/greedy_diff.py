"""Greedy-equivalence diff for the MTP sweep: A vs M1/M2 token sequences."""
import json

arms = {}
buf = []
for line in open("<root>/mtp/greedy.txt"):
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

print("arms captured:", {a: len(v) for a, v in arms.items()})
base = arms.get("A", {})
same = diff = err = 0
for arm in ("M1", "M2"):
    for p, rec in arms.get(arm, {}).items():
        if "error" in rec or "error" in base.get(p, {"error": 1}):
            err += 1
            continue
        if base[p]["tokens"] == rec["tokens"]:
            same += 1
        else:
            diff += 1
            tb, tm = base[p]["tokens"], rec["tokens"]
            first = next((i for i, (x, y) in enumerate(zip(tb, tm)) if x != y), min(len(tb), len(tm)))
            print(f"DIVERGE {arm} @tok{first}: {p[:30]!r} len {len(tb)}/{len(tm)}")
print(f"\nsame={same} diverge={diff} error={err}")
