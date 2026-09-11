#!/bin/bash
# W4 arm: int4 weights (GPTQ W4A16; GDN/mtp/visual/lm_head kept bf16 by the
# calibration ignore list) + fp8 KV + MTP k=2, max-model-len 8192 + explicit
# sched 8192. Protocol identical to arm C2 except weight precision -- the
# single new rung is "weights fp8 -> int4 mixed".
# This is the capacity-wall attack: draft weights shrank the shared block pool
# to ~52 blocks (~17 spec routes at @48, Running pinned 16-17 / usage 90-96%).
# int4 should free ~10GB -> pool grows several-fold -> @48 spec finally scales
# with full concurrency. Success criterion: @48 clearly above the no-spec
# single-card best 352.0 and the TP2 dual-card baseline 350.2.
# No --quantization flag: the checkpoint carries its compressed-tensors config
# and vLLM auto-detects. Greedy probe kept: W4 changes target-model numerics
# (GDN fp8 online -> bf16, FFN/attn fp8 -> int4), so token-level divergence
# vs arm A is expected to widen -- the interesting comparison is vs the
# spec-arm signature (M1/M2/K2 flips at tied-logit positions).
# Outputs to <out-dir-5>.

VENV=<venv>/bin
PORT=8331
MODEL=<model-dir>-int4
OUT=<out-dir-5>
mkdir -p "$OUT"

BENCH_COMMON="--backend openai-chat --endpoint /v1/chat/completions --tokenizer <model-dir> \
--dataset-name random --random-input-len 1024 --random-output-len 256 --temperature 0 \
--host 127.0.0.1 --model qwen38-27b --port $PORT"

SRV_PID=""

start_server() {
  local kv="$1" spec="$2" extra="$3"
  # setsid: the server must own its own process group, otherwise stop_server's
  # `kill -TERM -- -PGID` takes this sweep script down with it.
  local args="--port $PORT --served-model-name qwen38-27b --max-model-len 8192 \
--max-num-seqs 128"
  [ -n "$kv" ] && args="$args --kv-cache-dtype $kv"
  [ -n "$spec" ] && args="$args --speculative-config '$spec'"
  [ -n "$extra" ] && args="$args $extra"
  setsid bash -c "CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL $args" \
    > "$OUT/server_${ARM}.log" 2>&1 &
  SRV_PID=$!
  local i
  for i in $(seq 1 100); do
    sleep 10
    if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
      echo "$(date +%H:%M:%S) $ARM UP after $((i*10))s" >> "$OUT/run.log"
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "$(date +%H:%M:%S) $ARM SERVER_DIED" >> "$OUT/run.log"
      tail -25 "$OUT/server_${ARM}.log" >> "$OUT/run.log"
      return 1
    fi
  done
  echo "$(date +%H:%M:%S) $ARM TIMEOUT" >> "$OUT/run.log"
  return 1
}

stop_server() {
  [ -z "$SRV_PID" ] && return
  local pgid
  pgid=$(ps -o pgid= -p "$SRV_PID" | tr -d ' ')
  [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
  sleep 20
}

bench_arm() {
  local cc="$1"
  HF_HUB_OFFLINE=1 $VENV/vllm bench serve $BENCH_COMMON \
    --num-prompts 96 --max-concurrency "$cc" \
    > "$OUT/bench_${ARM}_c${cc}.log" 2>&1
  grep -E "Benchmark duration|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Mean ITL" \
    "$OUT/bench_${ARM}_c${cc}.log" | sed "s/^/$ARM c$cc /" >> "$OUT/summary.txt"
  grep "loggers.py:310" "$OUT/server_${ARM}.log" | tail -20 > "$OUT/engine_stats_${ARM}_c${cc}.txt"
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -i "spec\|accept\|draft" \
    > "$OUT/specmetrics_${ARM}_c${cc}.txt"
  echo "$(date +%H:%M:%S) $ARM bench c$cc done" >> "$OUT/run.log"
}

greedy_probe() {
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 $VENV/python mtp_greedy.py "$PORT" "$ARM" >> "$OUT/greedy.txt" 2>&1
  echo "$(date +%H:%M:%S) $ARM greedy done" >> "$OUT/run.log"
}

# pre-check: GPU3 must be free
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
if [ "$USED" -gt 2000 ]; then
  echo "GPU3 busy (${USED}MiB), abort" >> "$OUT/run.log"
  exit 1
fi
# pre-check: calibrated checkpoint must exist
if [ ! -f "$MODEL/config.json" ]; then
  echo "int4 checkpoint missing, abort" >> "$OUT/run.log"
  exit 1
fi

ARM="W4"
echo "=== ARM W4 weights=[int4-mixed] kv=[fp8] spec=[k=2] len=[8192] sched=[8192] $(date +%H:%M:%S) ===" >> "$OUT/run.log"
start_server "fp8" '{"method": "mtp", "num_speculative_tokens": 2}' \
  "--max-num-batched-tokens 8192" \
  || { stop_server; echo "ABORT" >> "$OUT/run.log"; exit 1; }
grep -E "max_num_scheduled|max_num_batched|KV cache size|Maximum concurrency|quantization=" \
  "$OUT/server_W4.log" | head -5 >> "$OUT/run.log" 2>/dev/null
bench_arm 48
bench_arm 16
greedy_probe
stop_server
echo "ALL_DONE $(date +%H:%M:%S)" >> "$OUT/run.log"
