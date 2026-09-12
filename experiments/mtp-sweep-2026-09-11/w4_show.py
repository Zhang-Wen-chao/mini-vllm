import json
def load(path):
    arms={}; buf=[]
    for line in open(path):
        line=line.strip()
        if not line: continue
        buf.append(line)
        if line.endswith("]"):
            try: chunk=json.loads("".join(buf))
            except Exception: continue
            for rec in chunk: arms.setdefault(rec["arm"],{})[rec["prompt"]]=rec
            buf=[]
    return arms
w4=load("<root>/mtp5/greedy.txt")["W4"]
for p,r in list(w4.items())[:5]:
    print("PROMPT:", p[:34])
    print("  finish:", r.get("finish"))
    t=(r.get("text") or "").replace(chr(10)," | ")
    print("  text:", t[:160])
    print()
