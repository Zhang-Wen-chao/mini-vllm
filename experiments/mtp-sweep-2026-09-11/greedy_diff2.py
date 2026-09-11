"""Within-arm control first: A vs A2 must be identical, then A vs M1/M2."""
import json

arms = {}
buf = []
for line in open("<out-dir>/greedy.txt"):
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

def cmp(a, b):
    same = diff = 0
    details = []
    for p in arms[a]:
        ra, rb = arms[a][p], arms[b].get(p)
        if rb is None or "error" in ra or "error" in rb:
            continue
        if ra["tokens"] == rb["tokens"]:
            same += 1
        else:
            diff += 1
            details.append(p[:28])
    print(f"{a} vs {b}: same={same} diverge={diff} {details}")

cmp("A", "A2")
cmp("A", "M1")
cmp("A", "M2")
