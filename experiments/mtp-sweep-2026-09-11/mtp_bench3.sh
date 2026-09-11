#!/bin/bash
# C2b arm: fp8 KV + MTP k=2 + explicit --max-num-batched-tokens 8192.
# Purpose: single-variable disambiguation of the @48 MTP throughput loss.
# TPOT/ITL reconciliation on mtp2 data shows decode-side verify is cheap
# (step time +4.6% for 2.35x tokens per round) while TTFT blows up 4x --
# pointing at the spec-forced max_num_scheduled_tokens=2048 prefill throttle,
# not verify compute. vLLM's own startup WARNING suggests raising
# max_num_batched_tokens to accommodate draft token slots; this arm does
# exactly that. If @48 throughput recovers toward ~350, the attribution
# holds; if it stays ~290, a verify compute wall is real after all.
# Protocol identical to mtp_bench2.sh arm C2 except the sched flag.
# Bench @48 + @16. No greedy probe (scheduling-only variable, numerics
# unchanged). Outputs to <out-dir>.

VENV=<venv>/bin
PORT=8331
MODEL=<model-dir>
OUT=<out-dir>
mkdir -p "$OUT"

BENCH_COMMON="--backend openai-chat --endpoint /v1/chat/completions --tokenizer $MODEL \
--dataset-name random --random-input-len 1024 --random-output-len 256 --temperature 0 \
--host 127.0.0.1 --model qwen38-27b --port $PORT"

SRV_PID=""

start_server() {
  local kv="$1" spec="$2" extra="$3"
  # setsid: the server must own its own process group, otherwise stop_server's
  # `kill -TERM -- -PGID` takes this sweep script down with it.
  local args="--port $PORT --served-model-name qwen38-27b --max-model-len 8192 \
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
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -i "spec\|accept\|draft" \
    > "$OUT/specmetrics_${ARM}_c${cc}.txt"
  echo "$(date +%H:%M:%S) $ARM bench c$cc done" >> "$OUT/run.log"
}

# pre-check: GPU3 must be free
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
if [ "$USED" -gt 2000 ]; then
  echo "GPU3 busy (${USED}MiB), abort" >> "$OUT/run.log"
  exit 1
fi

ARM="C2b"
echo "=== ARM C2b kv=[fp8] spec=[k=2] sched=8192-explicit $(date +%H:%M:%S) ===" >> "$OUT/run.log"
start_server "fp8" '{"method": "mtp", "num_speculative_tokens": 2}' "--max-num-batched-tokens 8192" \
  || { stop_server; echo "ABORT" >> "$OUT/run.log"; exit 1; }
# capture what the engine actually resolved sched/batched-tokens to
grep -E "max_num_scheduled|max_num_batched" "$OUT/server_C2b.log" >> "$OUT/run.log" 2>/dev/null
bench_arm 48
bench_arm 16
stop_server
echo "ALL_DONE $(date +%H:%M:%S)" >> "$OUT/run.log"
