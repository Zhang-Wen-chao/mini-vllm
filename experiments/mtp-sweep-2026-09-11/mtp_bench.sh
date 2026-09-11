#!/bin/bash
# MTP speculative-decode sweep for Qwen3.8-27B, single-GPU fp8 (GPU3, port 8331).
# Arms: A (baseline) -> M1 (mtp k=1) -> M2 (mtp k=2) -> A2 (baseline re-anchor).
# Baseline = 性能基线② from the deployment doc: single-GPU fp8 weights + KV bf16
# + default scheduling 8192 + prefix ON. NOT the final config (fp8 KV) — every
# historic knob was measured against ②, and "基线是滚动的" applies only after
# this round's winner is decided. MTP arms toggle ONLY speculative-config.
# Bench: openai-chat, 1024 in / 256 out, 96 prompts, temp 0. Per arm: @48 + @16.

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
  local spec="$1"
  # setsid: the server must own its own process group, otherwise stop_server's
  # `kill -TERM -- -PGID` takes this sweep script down with it (cost us arm M1
  # once: run.log just stops after "A greedy done", no SWEEP_EXIT line).
  if [ -n "$spec" ]; then
    setsid bash -c "CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
      --port $PORT --served-model-name qwen38-27b --max-model-len 8192 \
      --quantization fp8 --max-num-seqs 128 \
      --speculative-config '$spec'" \
      > "$OUT/server_${ARM}.log" 2>&1 &
  else
    setsid bash -c "CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
      --port $PORT --served-model-name qwen38-27b --max-model-len 8192 \
      --quantization fp8 --max-num-seqs 128" \
      > "$OUT/server_${ARM}.log" 2>&1 &
  fi
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
  ARM="$1"; SPEC="$2"
  echo "=== ARM $ARM spec=[$SPEC] $(date +%H:%M:%S) ===" >> "$OUT/run.log"
  start_server "$SPEC" || { stop_server; return 1; }
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

# A arm already collected in the first (doomed) run: 307.14 @48 / 219.37 @16,
# which matches the historic baseline ② within 0.2% — no need to redo it.
# Do NOT wipe summary.txt / greedy.txt, they hold arm A.
rm -f "$OUT/run.log"
run_arm M1 '{"method": "mtp", "num_speculative_tokens": 1}'
run_arm M2 '{"method": "mtp", "num_speculative_tokens": 2}'
run_arm A2 ""
echo "ALL_DONE $(date +%H:%M:%S)" >> "$OUT/run.log"
