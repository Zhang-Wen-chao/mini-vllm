#!/bin/bash
# C4 arm: fp8 KV + MTP k=2 + max-model-len 4096 + explicit sched 8192.
# Purpose: capacity-wall unlock test. Engine logs showed spec arms @48 pin at
# Running 16-17 / KV usage 90-96% (pool 81,920 tokens ~= 52 blocks of 1,600).
# PRE-REGISTERED PREDICTION (corrected pre-run after source reading): NULL.
# The mamba admission gate (single_type_kv_cache_manager.py, align mode)
# computes cdiv(num_tokens_main_model, block_size) + num_speculative_blocks
# from ACTUAL tokens, not max_model_len -- a 1,280-token request holds
# 1 + 2 = 3 state blocks either way, so 52/3 ~= 17 routes stands at 4K too.
# max_model_len only sets the block-table row length (metadata) and the
# startup "Maximum concurrency" print. This arm therefore runs as the clean
# negative control that pins that mechanism; the initial pre-source
# prediction ("halving len halves per-request accounting -> ~30 routes") is
# retained in the evidence README's correction trail and is expected to be wrong.
# Greedy probe kept as numerics control: len change should NOT alter greedy
# outputs vs arm A (no weight or dtype change).
# Bench @48 + @16. Outputs to <out-dir-4>.

VENV=<venv>/bin
PORT=8331
MODEL=<model-dir>
OUT=<out-dir-4>
mkdir -p "$OUT"

BENCH_COMMON="--backend openai-chat --endpoint /v1/chat/completions --tokenizer $MODEL \
--dataset-name random --random-input-len 1024 --random-output-len 256 --temperature 0 \
--host 127.0.0.1 --model qwen38-27b --port $PORT"

SRV_PID=""

start_server() {
  local kv="$1" spec="$2" extra="$3"
  # setsid: the server must own its own process group, otherwise stop_server's
  # `kill -TERM -- -PGID` takes this sweep script down with it.
  local args="--port $PORT --served-model-name qwen38-27b \
--quantization fp8 --max-num-seqs 128"
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
  # engine scheduling stats: the direct capacity-wall observation
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

ARM="C4"
echo "=== ARM C4 kv=[fp8] spec=[k=2] len=[4096] sched=[8192] $(date +%H:%M:%S) ===" >> "$OUT/run.log"
start_server "fp8" '{"method": "mtp", "num_speculative_tokens": 2}' \
  "--max-model-len 4096 --max-num-batched-tokens 8192" \
  || { stop_server; echo "ABORT" >> "$OUT/run.log"; exit 1; }
grep -E "max_num_scheduled|max_num_batched|KV cache size|Maximum concurrency" \
  "$OUT/server_C4.log" >> "$OUT/run.log" 2>/dev/null
bench_arm 48
bench_arm 16
greedy_probe
stop_server
echo "ALL_DONE $(date +%H:%M:%S)" >> "$OUT/run.log"
