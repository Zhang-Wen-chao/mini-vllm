import os, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
t0 = time.time()
import torch
import transformers
print("transformers:", transformers.__version__, flush=True)

MODEL = "<model-dir>"          # bf16 source (52GB)
OUT = "<model-dir>-int4"       # GPTQ W4A16 mixed-precision output
MAXLEN = 2048
N_SAMPLES = 128
BATCH = 4

from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
# CUDA_VISIBLE_DEVICES is set by chain_int4.sh to two free GPUs (0,3);
# masked indices renumber, so max_memory keys are 0/1. GPU0 carries a ~2.6GB
# foreign daemon -> keep its budget at 38GiB, GPU3 at 42GiB.
MAX_MEM = {0: "38GiB", 1: "42GiB"}  # weights pinned on GPU, no CPU offload (save needs merged shards)
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

# ---- self-built calibration DataLoader (same recipe as fp8_calib.py; the
# llmcompressor data-pipeline split/collator pitfalls are documented there) ----
from datasets import load_dataset
from torch.utils.data import DataLoader

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
print("dataset rows:", len(ds), time.time() - t0, flush=True)

def tokenize_row(row):
    enc = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False,
        max_length=MAXLEN, truncation=True,
    )
    return enc["input_ids"]

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

# ---- quantization recipe: GPTQ W4A16, mixed precision ----
# Why GPTQ and not the AWQ pipeline: llmcompressor 0.13's AWQModifier() is a
# deprecated factory splitting into an AWQ scale-search transform + quantizer,
# and the transform half exposes NO targets/ignore filter -- it would rescale
# GDN in_proj/out_proj weights (absorbed into upstream norms in exact math,
# but those feed custom GDN kernels, so drift there is unforced risk).
# GPTQModifier filters via targets/ignore directly and produces the same
# compressed-tensors W4A16 format (int4 / group 128 / symmetric / static).
# Mixed-precision policy (capacity-wall attack, not a quality cliff):
#   - FFN + full-attn QKV/O (~18B of 27B params) -> int4
#   - GDN layers stay bf16: int4 error would accumulate through the recursive
#     state; linear_attn on the ignore list
#   - mtp.* stays bf16: draft accuracy gates the acceptance rate
#   - lm_head + visual tower stay bf16 (visual is dead weight for text serving)
# PITFALL (cost one calibration run): compressed_tensors match_name semantics
# are EXACT string equality or "re:"-prefixed regex (re.match, anchored at
# start). Glob patterns like "model.visual.*" / "*linear_attn*" silently
# never match anything -- first attempt quantized GDN+mtp+visual and was only
# saved by the 4304-cols divisibility check erroring on visual fc2. The guard
# below replicates match_name so the check can't lie to itself again.
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

from transformers.modeling_utils import PreTrainedModel
_orig_save = PreTrainedModel.save_pretrained
def _patched_save(self, *args, **kwargs):
    kwargs["save_original_format"] = False
    kwargs["max_shard_size"] = "5GB"
    return _orig_save(self, *args, **kwargs)
PreTrainedModel.save_pretrained = _patched_save

IGNORE = ["re:lm_head", r"re:mtp\..*", r"re:.*linear_attn.*", r"re:model\.visual\..*"]

# guard: verify ignore patterns with the ACTUAL matcher (match_name replica:
# exact or re:-prefixed re.match), against module names derived from the
# weight index. targets="Linear" matches by CLASS, so every nn.Linear in the
# model is a quantization candidate -- the guard must therefore check the
# full candidate set (all weight-bearing modules), not a name-substring
# subset. Rule: everything under model.visual / mtp / *.linear_attn.* /
# lm_head must be ignored; language-model self_attn/mlp must stay quantized.
import json, re as _re
def _match_name(name, target):
    if target.startswith("re:"):
        return _re.match(target[3:], name) is not None
    return target == name

wm = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
mods = sorted({n[:-len(".weight")] for n in wm if n.endswith(".weight")})
hit = {n for n in mods if any(_match_name(n, pat) for pat in IGNORE)}

must_ignore = [n for n in mods if n.startswith(("mtp.", "model.visual."))
               or ".linear_attn." in n or n == "lm_head"]
missed = [n for n in must_ignore if n not in hit]
lang_lin = [n for n in mods if ".language_model." in n
            and ("self_attn" in n or ".mlp." in n)
            and not n.endswith("_norm")]  # q_norm/k_norm are RMSNorm, not Linear
lang_bad = [n for n in lang_lin if n in hit]
gdn_lin = [n for n in must_ignore if ".linear_attn." in n and
           ("proj" in n or ".fc" in n)]
print("ignore check (match_name replica): %d modules scanned; must-ignore=%d "
      "missed=%d; gdn Linear=%d; language Linear quantized=%d wrongly-ignored=%d"
      % (len(mods), len(must_ignore), len(missed), len(gdn_lin),
         len(lang_lin) - len(lang_bad), len(lang_bad)), flush=True)
assert not missed, "modules NOT covered by ignore patterns: %s" % missed[:5]
assert not lang_bad, "language-model self_attn/mlp unexpectedly ignored - abort"
assert len(gdn_lin) == 240, "GDN Linear count off (expect 240) - abort"
# language Linear quantized = 16 full-attn layers x 7 (qkvo+gate/up/down)
#                        + 48 GDN layers x 3 (mlp only; linear_attn is ignored)
assert len(lang_lin) - len(lang_bad) == 256, "language Linear count off (expect 256=16x7+48x3) - abort"

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)
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
