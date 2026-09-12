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
a=load("<root>/mtp/greedy.txt"); c=load("<root>/mtp4/greedy.txt")
same=diff=0; det=[]
for p in a["A"]:
    x=a["A"][p]; y=c["C4"].get(p)
    if y is None or "error" in x or "error" in y: continue
    if x["tokens"]==y["tokens"]: same+=1
    else:
        diff+=1
        pos=next((i for i,(t1,t2) in enumerate(zip(x["tokens"],y["tokens"])) if t1!=t2), min(len(x["tokens"]),len(y["tokens"])))
        det.append(p[:14]+"@"+str(pos))
print("A vs C4: same=%d diverge=%d %s"%(same,diff,det))
