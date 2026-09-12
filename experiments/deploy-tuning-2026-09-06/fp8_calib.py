import os, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
t0 = time.time()
import torch
import transformers
print("transformers:", transformers.__version__, flush=True)

MODEL = "<model-dir>"
OUT = "<model-dir>-fp8-kv"
MAXLEN = 2048
N_SAMPLES = 128
BATCH = 4

from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
MAX_MEM = {0: "44GiB", 1: "44GiB"}  # 全部权重钉在 GPU 上，禁止 CPU offload（save 需要合并分片）
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", max_memory=MAX_MEM)
    print("loaded via AutoModelForImageTextToText", time.time() - t0, flush=True)
except Exception as e:
    print("VL load failed, fallback CausalLM:", e, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", max_memory=MAX_MEM)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("model+tokenizer ready", time.time() - t0, flush=True)

# ---- 自建标定 DataLoader（绕开 llmcompressor 数据管线：split/collator 坑已耗尽耐心）----
from datasets import load_dataset
from torch.utils.data import DataLoader

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
print("dataset rows:", len(ds), time.time() - t0, flush=True)

def tokenize_row(row):
    enc = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False,
        max_length=MAXLEN, truncation=True,
    )
    return enc["input_ids"]  # BatchEncoding -> 纯 list[int]

N = min(N_SAMPLES, len(ds))
tokenized = [tokenize_row(ds[i]) for i in range(N)]
print("tokenized", N, "rows, len range:", min(len(t) for t in tokenized), "-", max(len(t) for t in tokenized), flush=True)

def collate(features):
    B = len(features)
    ids = torch.zeros(B, MAXLEN, dtype=torch.long)
    am = torch.zeros(B, MAXLEN, dtype=torch.long)
    for i, f in enumerate(features):
        n = min(len(f), MAXLEN)
        ids[i, :n] = torch.tensor(f[:n])
        am[i, :n] = 1
    return {"input_ids": ids, "attention_mask": am}

loader = DataLoader(tokenized, batch_size=BATCH, shuffle=False, collate_fn=collate)
print("dataloader ready:", len(loader), "batches", time.time() - t0, flush=True)

# ---- 量化配方 ----
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs

# save 注入：跳过 original-format 还原（fp8 checkpoint 不需要；跨分片还原是已知炸点）
from transformers.modeling_utils import PreTrainedModel
_orig_save = PreTrainedModel.save_pretrained
def _patched_save(self, *args, **kwargs):
    kwargs["save_original_format"] = False
    kwargs["max_shard_size"] = "5GB"
    return _orig_save(self, *args, **kwargs)
PreTrainedModel.save_pretrained = _patched_save

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    kv_cache_scheme=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
    ignore=["lm_head"],
)
print("recipe ready", time.time() - t0, flush=True)

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=loader,
    recipe=recipe,
    output_dir=OUT,
    save_compressed=True,
)
print("CALIB_DONE", time.time() - t0, flush=True)
