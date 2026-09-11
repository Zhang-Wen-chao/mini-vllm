#!/bin/bash
# W4 C-Eval re-run: the bench5 ceval step failed (datasets not in <venv>).
# Starts the int4 server on GPU3 and runs the 200-question C-Eval harness
# from the lc venv (which has `datasets`). Same protocol as the fp8 round:
# tag w4, /v1/completions + logprobs argmax + allowed_token_ids.
# Reference numbers: bf16 0.7950 / fp8 scale1.0 0.7750 / calibrated 0.7700.
VENV=<venv>/bin
PORT=8331
MODEL=<model-dir>-int4
OUT=<out-dir-5>
ARM=W4

setsid bash -c "CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
--port $PORT --served-model-name qwen38-27b --max-model-len 8192 \
--max-num-seqs 128 --kv-cache-dtype fp8 \
--speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 2}' \
--max-num-batched-tokens 8192" > "$OUT/server_ceval.log" 2>&1 &
SRV_PID=$!
for i in $(seq 1 100); do
  sleep 10
  curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
done
echo "$(date +%H:%M:%S) ceval server up" >> "$OUT/run.log"

# NOTE: no HF_HUB_OFFLINE here — the script self-configures HF_ENDPOINT=hf-mirror
# + offline=0 via setdefault. Forcing offline makes get_dataset_config_names
# degrade to ['default'] and the cache (subject-named configs only) misses.
<lc-venv>/bin/python ceval_eval.py --port $PORT \
  --tag w4 --model-name qwen38-27b --num 20 > "$OUT/ceval_W4.log" 2>&1
tail -3 "$OUT/ceval_W4.log" >> "$OUT/run.log"
echo "$(date +%H:%M:%S) ceval done" >> "$OUT/run.log"

pgid=$(ps -o pgid= -p "$SRV_PID" | tr -d ' ')
[ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
wait "$SRV_PID" 2>/dev/null
echo "CEVAL_DONE $(date +%H:%M:%S)" >> "$OUT/run.log"
