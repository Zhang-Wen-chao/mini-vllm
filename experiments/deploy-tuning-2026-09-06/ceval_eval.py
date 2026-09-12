import argparse, json, os, random, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import urllib.request
import concurrent.futures as cf
from transformers import AutoTokenizer

TOK_DIR = "<model-dir>"
LETTERS = ["A", "B", "C", "D"]
PREFIX = "以下是单项选择题，请选出其中的正确答案。\n\n"
N_QUESTIONS = 200
SEED = 42

def build_prompt(q):
    lines = [PREFIX + q["question"]]
    for k in LETTERS:
        lines.append(f"{k}. {q[k]}")
    lines.append("答案：")
    return "\n".join(lines)

def load_questions():
    from datasets import load_dataset, get_dataset_config_names
    rows = []
    for cfg in get_dataset_config_names("ceval/ceval-exam"):
        ds = load_dataset("ceval/ceval-exam", cfg, split="val")
        rows += [{"question": r["question"], "A": r["A"], "B": r["B"],
                  "C": r["C"], "D": r["D"], "answer": r["answer"], "subject": cfg}
                 for r in ds]
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(N_QUESTIONS, len(rows)))
    return picked

def letter_token_ids(tok):
    ids = []
    for l in LETTERS:
        for form in (l, " " + l):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                ids.append(e[0])
    return sorted(set(ids))

def ask(port, model_name, prompt, allowed, tag, i, outdir):
    body = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 20,
        "allowed_token_ids": allowed,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    ch = out["choices"][0]
    pred = ch["text"].strip()[:1]
    lp = ch.get("logprobs", {}).get("top_logprobs", [{}])[0]
    def _lp(v):  # vLLM: {token: float}；OpenAI 风格: {token: {logprob: ...}}——两者兼容
        return v["logprob"] if isinstance(v, dict) else v
    scores = {t: _lp(v) for t, v in lp.items() if t.strip() in LETTERS}
    with open(f"{outdir}/{tag}_{i:03d}.json", "w") as f:
        json.dump({"pred": pred, "scores": scores}, f, ensure_ascii=False)
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--num", type=int, default=20, help="并发")
    args = ap.parse_args()

    outdir = "<out-dir-ceval>"
    os.makedirs(outdir, exist_ok=True)
    qs = load_questions()
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    allowed = letter_token_ids(tok)
    print(f"questions={len(qs)} allowed_ids={allowed} tok={tok.__class__.__name__}", flush=True)

    t0 = time.time()
    preds = [None] * len(qs)
    with cf.ThreadPoolExecutor(max_workers=args.num) as ex:
        futs = {ex.submit(ask, args.port, args.model_name, build_prompt(q), allowed, args.tag, i + 1, outdir): i
                for i, q in enumerate(qs)}
        for f in cf.as_completed(futs):
            preds[futs[f]] = f.result()

    answers = [q["answer"] for q in qs]
    correct = sum(1 for p, a in zip(preds, answers) if p == a)
    acc = correct / len(qs)
    # 分科目（粗粒度）
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for q, p, a in zip(qs, preds, answers):
        per[q["subject"]][1] += 1
        per[q["subject"]][0] += (p == a)
    print(f"TAG={args.tag} ACC={acc:.4f} ({correct}/{len(qs)}) time={time.time()-t0:.1f}s", flush=True)
    worst = sorted(per.items(), key=lambda x: x[1][0] / x[1][1])[:8]
    print("worst subjects:", [(s, f"{c}/{n}") for s, (c, n) in worst], flush=True)
    print(f"CEVAL_DONE {args.tag}", flush=True)

if __name__ == "__main__":
    main()
