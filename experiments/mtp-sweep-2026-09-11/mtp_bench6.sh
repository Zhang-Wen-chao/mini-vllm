#!/bin/bash
# sweep6: 2-card deployment matrix — TP2 bf16 anchor vs 2x fp8 single-card replicas.
# User's success criterion: same 2-card budget vs TP2 — faster per request,
# higher total throughput, accuracy drop small = a successful deployment.
#
# Arms (sequential, same window; GPUs 1/2/3 only — GPU0 has a foreign qserver):
#   TP2 : bf16 tensor-parallel 2 on GPU2+3, port 8343, historical flags
#         (len 8192 / seqs 128 / sched default; served-model-name unified to
#         qwen38-27b so the bench command is identical across arms).
#         Points: @48 (96 prompts, conc 48) + @96 (192 prompts, conc 96).
#   R2b : 2 replicas on GPU1+GPU2 (ports 8341/8342) — fp8 weights + fp8 KV,
#         no spec, sched 8192 (= K2 config). Points: @48total (2x [48p, c24])
#         + @96total (2x [96p, c48]).
#   R2c : same replica pair + MTP k=1 (new config: snapshot tax halves,
#         52 blocks / 2 = 26 routes >= 24, no wall expected). Point: @48total.
#   R2a : same replica pair + MTP k=2 (= C2b config; wall 17 routes/replica,
#         7 queued at c24). Point: @48total.
#
# Pre-registered predictions + criteria live in the evidence README (written
# BEFORE this launch). Neighbor condition at launch: foreign qserver ~39GB on
# GPU0 with 284MB footprints on GPU1-3 — TP2 anchor is re-measured in the same
# window so the comparison stays internally aligned; anchor-vs-history delta
# > 5% flags a polluted window (then report relative numbers only).
#
# Known-good patterns reused: setsid wrapper for serve (pitfall 1), PGID kill,
# SIGABRT-tolerant bench steps, UTC container clock (log timestamps = UTC).
# Outputs to <out-dir-6>: run.log timeline, summary.txt key metrics (+ totals),
# bench_*.log raw outputs, engine_stats_*.txt from server loggers lines,
# server_startup_lines.txt, gpu_snapshots.txt, specmetrics_*.txt.

VENV=<venv>/bin
MODEL=<model-dir>
OUT=<out-dir-6>
mkdir -p $OUT

log() { echo "$(date +%H:%M:%S) $*" >> $OUT/run.log; }

gpu_snap() { echo "--- $(date +%H:%M:%S)" >> $OUT/gpu_snapshots.txt
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $OUT/gpu_snapshots.txt; }

wait_up() { # $1 port
  for i in $(seq 1 120); do
    sleep 10
    curl -s "http://127.0.0.1:$1/health" > /dev/null 2>&1 && return 0
  done
  return 1
}

kill_srv() { # $1 pid
  local pgid=$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
  sleep 8
  [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  sleep 2
}

bench_one() { # $1 port $2 concurrency $3 num_prompts $4 logfile
  HF_HUB_OFFLINE=1 $VENV/vllm bench serve --backend openai-chat \
    --endpoint /v1/chat/completions --tokenizer $MODEL --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --temperature 0 \
    --num-prompts $3 --max-concurrency $2 --host 127.0.0.1 \
    --model qwen38-27b --port $1 > "$4" 2>&1
}

summarize_single() { # $1 tag $2 logfile
  echo "== $1 ==" >> $OUT/summary.txt
  grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TTFT|Mean TPOT|Peak concurrent" "$2" >> $OUT/summary.txt 2>/dev/null
}

summarize_dual() { # $1 tag — sums the two parts
  echo "== $1 (p1 + p2, TOTAL) ==" >> $OUT/summary.txt
  for f in $OUT/bench_${1}_p1.log $OUT/bench_${1}_p2.log; do
    grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TTFT|Mean TPOT|Peak concurrent" "$f" >> $OUT/summary.txt 2>/dev/null
  done
  $VENV/python - "$1" <<'EOF' >> $OUT/summary.txt
import re, sys
tag = sys.argv[1]
def val(part, key):
    m = re.search(key + r":\s+([0-9.]+)", open(f"<out-dir-6>/bench_{tag}_{part}.log").read())
    return float(m.group(1)) if m else float("nan")
for key in ["Output token throughput", "Mean TTFT", "Mean TPOT"]:
    a, b = val("p1", key), val("p2", key)
    print(f"{key}: p1={a:.2f} p2={b:.2f} TOTAL={a+b:.2f}")
EOF
}

engine_stats() { # $1 tag $2... server logs
  local tag=$1; shift
  for f in "$@"; do
    echo "--- $f" >> $OUT/engine_stats_${tag}.txt
    grep -a "loggers.py" "$f" | tail -40 >> $OUT/engine_stats_${tag}.txt
  done
}

startup_lines() { # $1... server logs
  for f in "$@"; do
    echo "--- $f" >> $OUT/server_startup_lines.txt
    grep -aE "KV cache size|Maximum concurrency|quantization|Using .*Kernel|speculative|max_num_scheduled" "$f" | head -12 >> $OUT/server_startup_lines.txt
  done
}

spec_metrics() { # $1 tag $2... ports
  local tag=$1; shift
  for p in "$@"; do
    curl -s "http://127.0.0.1:$p/metrics" 2>/dev/null | grep -iE "spec|draft|accept" >> $OUT/specmetrics_${tag}.txt
  done
}

log "sweep6 start: TP2 anchor + 2x replica matrix"
gpu_snap

# ---------- Phase 1: TP2 anchor (GPU2+3) ----------
setsid bash -c "CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8343 --served-model-name qwen38-27b --max-model-len 8192 \
  --tensor-parallel-size 2 --max-num-seqs 128" > $OUT/server_TP2.log 2>&1 &
TP2=$!
if wait_up 8343; then
  log "TP2 up (pid $TP2)"
  bench_one 8343 48 96 $OUT/bench_TP2_c48.log
  engine_stats TP2_c48 $OUT/server_TP2.log
  bench_one 8343 96 192 $OUT/bench_TP2_c96.log
  summarize_single TP2_c48 $OUT/bench_TP2_c48.log
  summarize_single TP2_c96 $OUT/bench_TP2_c96.log
  startup_lines $OUT/server_TP2.log
else
  log "ABORT TP2 boot failed"
fi
kill_srv $TP2
log "TP2 phase done"
gpu_snap

# ---------- Phase 2: R2b — K2-config replica pair (GPU1+GPU2) ----------
setsid bash -c "CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8341 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192" > $OUT/server_R2b_1.log 2>&1 &
RB1=$!
setsid bash -c "CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8342 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192" > $OUT/server_R2b_2.log 2>&1 &
RB2=$!
if wait_up 8341 && wait_up 8342; then
  log "R2b pair up ($RB1/$RB2)"
  bench_one 8341 24 48 $OUT/bench_R2b_c48t_p1.log &
  B1=$!
  bench_one 8342 24 48 $OUT/bench_R2b_c48t_p2.log &
  B2=$!
  wait $B1 $B2
  engine_stats R2b_c48t $OUT/server_R2b_1.log $OUT/server_R2b_2.log
  bench_one 8341 48 96 $OUT/bench_R2b_c96t_p1.log &
  B1=$!
  bench_one 8342 48 96 $OUT/bench_R2b_c96t_p2.log &
  B2=$!
  wait $B1 $B2
  summarize_dual R2b_c48t
  summarize_dual R2b_c96t
  startup_lines $OUT/server_R2b_1.log $OUT/server_R2b_2.log
else
  log "ABORT R2b boot failed"
fi
kill_srv $RB1; kill_srv $RB2
log "R2b phase done"
gpu_snap

# ---------- Phase 3: R2c — k=1 replica pair ----------
setsid bash -c "CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8341 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 1}'" > $OUT/server_R2c_1.log 2>&1 &
RC1=$!
setsid bash -c "CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8342 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 1}'" > $OUT/server_R2c_2.log 2>&1 &
RC2=$!
if wait_up 8341 && wait_up 8342; then
  log "R2c pair up ($RC1/$RC2)"
  bench_one 8341 24 48 $OUT/bench_R2c_c48t_p1.log &
  B1=$!
  bench_one 8342 24 48 $OUT/bench_R2c_c48t_p2.log &
  B2=$!
  wait $B1 $B2
  engine_stats R2c_c48t $OUT/server_R2c_1.log $OUT/server_R2c_2.log
  spec_metrics R2c_c48t 8341 8342
  summarize_dual R2c_c48t
  startup_lines $OUT/server_R2c_1.log $OUT/server_R2c_2.log
else
  log "ABORT R2c boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2c phase done"
gpu_snap

# ---------- Phase 4: R2a — k=2 (C2b-config) replica pair ----------
setsid bash -c "CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8341 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 2}'" > $OUT/server_R2a_1.log 2>&1 &
RA1=$!
setsid bash -c "CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
  --port 8342 --served-model-name qwen38-27b --max-model-len 8192 \
  --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 2}'" > $OUT/server_R2a_2.log 2>&1 &
RA2=$!
if wait_up 8341 && wait_up 8342; then
  log "R2a pair up ($RA1/$RA2)"
  bench_one 8341 24 48 $OUT/bench_R2a_c48t_p1.log &
  B1=$!
  bench_one 8342 24 48 $OUT/bench_R2a_c48t_p2.log &
  B2=$!
  wait $B1 $B2
  engine_stats R2a_c48t $OUT/server_R2a_1.log $OUT/server_R2a_2.log
  spec_metrics R2a_c48t 8341 8342
  summarize_dual R2a_c48t
  startup_lines $OUT/server_R2a_1.log $OUT/server_R2a_2.log
else
  log "ABORT R2a boot failed"
fi
kill_srv $RA1; kill_srv $RA2
log "R2a phase done"
gpu_snap

log "ALL_DONE"
