"""Show A vs M text at divergence points to characterize severity."""
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

for p in arms["A"]:
    a = arms["A"][p]
    for m in ("M1", "M2"):
        r = arms[m].get(p)
        if r is None or a["tokens"] == r["tokens"]:
            continue
        # find first diff position
        i = next((k for k in range(min(len(a["tokens"]), len(r["tokens"])))
                  if a["tokens"][k] != r["tokens"][k]), None)
        print(f"### {m} | pos={i} | {p[:24]}")
        print(f"  A  tail: ...{a['text'][-80:]}")
        print(f"  {m} tail: ...{r['text'][-80:]}")
        print()
