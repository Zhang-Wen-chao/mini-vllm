"""End-to-end demo: train a tiny model to copy character strings, then serve
it through the mini-vllm engine with continuous batching.

No transformers dependency: a char-level tokenizer and a toy copy task
("a b c" -> "a b c") keep the whole demo runnable on CPU in seconds.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from mini_vllm.engine import Engine
from mini_vllm.model_runner import TinyTransformer


CHARS = "abcdefghijklmnopqrstuvwxyz "
VOCAB = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = pad/eos
IDS = {v: k for k, v in VOCAB.items()}


def encode(text):
    return torch.tensor([VOCAB[c] for c in text])


def decode(ids):
    return "".join(IDS.get(i, "?") for i in ids)


def make_copy_data(n_samples=400, max_len=8):
    rng = torch.Generator().manual_seed(7)
    samples = []
    for _ in range(n_samples):
        length = int(torch.randint(1, max_len + 1, (1,), generator=rng))
        chars = "".join(CHARS[int(torch.randint(0, len(CHARS), (1,),
                                                generator=rng))]
                        for _ in range(length))
        samples.append((chars, chars))  # copy task: input == target
    return samples


class CopyDataset(Dataset):
    def __init__(self, samples, max_len=10):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        x = encode(src)
        y = encode(tgt)
        return x, y


def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    pad = lambda t: F.pad(t, (0, max_len - len(t)), value=0)
    return torch.stack([pad(x) for x in xs]), torch.stack([pad(y) for y in ys])


def train(model, samples, steps=150, lr=3e-3):
    loader = DataLoader(CopyDataset(samples), batch_size=32,
                        collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(steps):
        total = 0.0
        for x, y in loader:
            logits = model.dense_forward(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size),
                                   y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"step {epoch + 1}: loss {total / len(loader):.4f}")
    model.eval()


def main():
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=len(VOCAB) + 1, d_model=64,
                            n_layers=2, n_heads=4)
    print("training a tiny copy model...")
    train(model, make_copy_data())

    engine = Engine(model, block_size=4, num_blocks=64)
    prompts = ["ab cd", "xyz hello", "a b c d"]
    for p in prompts:
        engine.add_request(encode(p), max_new_tokens=len(p) + 2)

    while engine.has_requests():
        engine.step()

    for r in engine.scheduler.finished:
        ids = engine.output(r)
        full = decode(ids)
        prompt_text = decode(ids[:len(prompts[engine.scheduler.finished.index(r)])])
        print(f"prompt={prompt_text!r} -> generated={full!r}")


if __name__ == "__main__":
    main()
