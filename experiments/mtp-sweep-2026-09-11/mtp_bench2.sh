#!/bin/bash
# Combination sweep: the ladder endpoint for Qwen3.8-27B single-GPU (GPU3, port 8331).
# Arms: K2 (fp8 KV only, no spec) -> C2 (fp8 KV + MTP k=2) -> A3 (plain re-anchor).
# Why K2: historic fp8-KV data only has @48 (352.9); @16 was never measured, and the
# ladder (TP2 bf16 -> fp8 weights -> +fp8 KV -> +MTP) needs every level at BOTH
# concurrency points for attribution. K2@48 should also reproduce 352.9 = cross-run
# anchor. Why C2: the ladder endpoint — does MTP stack on top of fp8 KV, and does
# spec decode even boot with --kv-cache-dtype fp8 in 0.28.0 (unverified)?
# Baseline = 性能基线②: single-GPU fp8 weights + KV bf16 + default sched + prefix ON.
# A3 re-anchors against A/A2 from <out-dir> (219.37 @16 / 307.14 @48).
# Bench: openai-chat, 1024 in / 256 out, 96 prompts, temp 0. Per arm: @48 + @16.

VENV=<venv>/bin
PORT=8331
MODEL=<model-dir>
OUT=<out-dir-2>
mkdir -p "$OUT"

BENCH_COMMON="--backend openai-chat --endpoint /v1/chat/completions --tokenizer $MODEL \
--dataset-name random --random-input-len 1024 --random-output-len 256 --temperature 0 \
--host 127.0.0.1 --model qwen38-27b --port $PORT"

SRV_PID=""

start_server() {
  local kv="$1" spec="$2"
  # setsid: the server must own its own process group, otherwise stop_server's
  # `kill -TERM -- -PGID` takes this sweep script down with it.
  local args="--port $PORT --served-model-name qwen38-27b --max-model-len 8192 \
--quantization fp8 --max-num-seqs 128"
  [ -n "$kv" ] && args="$args --kv-cache-dtype $kv"
  [ -n "$spec" ] && args="$args --speculative-config '$spec'"
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
  tail -25 "$OUT/server_${ARM}.log" >> "$OUT/run.log"
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
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -i "spec\|accept\|draft" \
    > "$OUT/specmetrics_${ARM}_c${cc}.txt"
  echo "$(date +%H:%M:%S) $ARM bench c$cc done" >> "$OUT/run.log"
}

greedy_probe() {
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 $VENV/python mtp_greedy.py "$PORT" "$ARM" >> "$OUT/greedy.txt" 2>&1
  echo "$(date +%H:%M:%S) $ARM greedy done" >> "$OUT/run.log"
}

run_arm() {
  ARM="$1"; KV="$2"; SPEC="$3"
  echo "=== ARM $ARM kv=[$KV] spec=[$SPEC] $(date +%H:%M:%S) ===" >> "$OUT/run.log"
  start_server "$KV" "$SPEC" || { stop_server; return 1; }
  bench_arm 48
  bench_arm 16
  greedy_probe
  stop_server
}

# pre-check: GPU3 must be free
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
if [ "$USED" -gt 2000 ]; then
  echo "GPU3 busy (${USED}MiB), abort" >> "$OUT/run.log"
  exit 1
fi

run_arm K2 "fp8" ""
run_arm C2 "fp8" '{"method": "mtp", "num_speculative_tokens": 2}'
run_arm A3 "" ""
echo "ALL_DONE $(date +%H:%M:%S)" >> "$OUT/run.log"
