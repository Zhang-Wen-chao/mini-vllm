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
m2=load("<out-dir-2>/greedy.txt"); c=load("<out-dir-4>/greedy.txt")
same=diff=0; det=[]
for p in m2["C2"]:
    x=m2["C2"][p]; y=c["C4"].get(p)
    if y is None or "error" in x or "error" in y: continue
    if x["tokens"]==y["tokens"]: same+=1
    else:
        diff+=1
        pos=next((i for i,(t1,t2) in enumerate(zip(x["tokens"],y["tokens"])) if t1!=t2), min(len(x["tokens"]),len(y["tokens"])))
        det.append(p[:14]+"@"+str(pos))
print("C2 vs C4 (len 8192 vs 4096, same dtype+spec): same=%d diverge=%d %s"%(same,diff,det))
