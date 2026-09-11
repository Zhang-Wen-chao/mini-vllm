"""Greedy-equivalence probe for the MTP sweep (Qwen3.8-27B, port 8331).

Sends a fixed prompt set with temperature=0 to whichever arm is up, records the
generated token sequence (via chat completions logprobs tokens), so A / M1 / M2
can be compared offline. Speculative decoding must be lossless: any token-level
difference between baseline and MTP arms is a real finding, not noise.

Usage: python mtp_greedy.py <port> <arm>
"""

import json
import sys
import urllib.request

PROMPTS = [
    "用三句话解释 TCP 和 UDP 的区别。",
    "写一个迭代版斐波那契函数，直接给代码。",
    "17 * 23 等于多少？给出推理过程。",
    "把 the weather is nice today 翻译成法语、德语和日语。",
    "一个球拍和一个球共 1.10 元，球拍比球贵 1.00 元，球多少钱？",
    "List the first 10 prime numbers and explain why 1 is not prime.",
    "Write a haiku about GPUs.",
    "Summarize the plot of Hamlet in one paragraph.",
]

CONCURRENCY_NOTE = "probes run sequentially, no prefix-cache warmup between arms"


def probe(port: int, arm: str) -> None:
    out = []
    for p in PROMPTS:
        body = json.dumps(
            {
                "model": "qwen38-27b",
                "messages": [{"role": "user", "content": p}],
                "temperature": 0.0,
                "max_tokens": 96,
                "logprobs": True,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            choice = resp["choices"][0]
            lp = choice.get("logprobs") or {}
            toks = [t["token"] for t in (lp.get("content") or [])]
            out.append(
                {
                    "arm": arm,
                    "prompt": p,
                    "tokens": toks,
                    "text": choice["message"]["content"][:200],
                    "finish": choice.get("finish_reason"),
                }
            )
        except Exception as e:  # noqa: BLE001
            out.append({"arm": arm, "prompt": p, "error": str(e)[:200]})
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    probe(int(sys.argv[1]), sys.argv[2])
