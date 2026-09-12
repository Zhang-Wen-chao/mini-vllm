import json, glob

# MTP weights in the 27B checkpoint
idx = json.load(open("<model-dir>/model.safetensors.index.json"))
wm = idx["weight_map"]
mtp = [k for k in wm if "mtp" in k.lower() or "nextn" in k.lower()]
print("MTP weight tensors:", len(mtp))
for k in mtp[:6]:
    print(" ", k)

cfg = json.load(open("<model-dir>/config.json"))
print("mtp_num_hidden_layers:", cfg.get("mtp_num_hidden_layers"))
print("mtp_use_dedicated_embeddings:", cfg.get("mtp_use_dedicated_embeddings"))
tc = cfg.get("text_config", {})
print("text_config mtp keys:", {k: v for k, v in tc.items() if "mtp" in k.lower()})

# installed vllm support for qwen3_5 mtp
import vllm, inspect
from vllm.config import speculative as spec_mod
src = inspect.getsource(spec_mod)
print("vllm", vllm.__version__, "| qwen3_5_mtp in speculative.py:", "qwen3_5_mtp" in src)
print("Qwen3_5MTP arch wired:", "Qwen3_5MTP" in src)
