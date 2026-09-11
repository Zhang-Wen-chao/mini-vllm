#!/bin/bash
# sweep7: fix the baseline, split the gain, pin the ceiling.
# Implements the pre-registration in README ("sweep7 预注册"). Design first,
# script second — the arm list here is a transcription of that document, not a
# substitute for it.
#
# Window order (all one window): B0 -> B1 -> R2c -> R2b -> B0'
#   B0 and B0' bracket the window; |B0' - B0| > 3% voids the whole window.
#
# Arms:
#   B0   TP2 bf16 on GPU2+3, port 8343 — the baseline, historical flags.
#        Points @16 (NEW: fills the cross-window hole in the 速览 §0 table,
#        which currently pairs sweep6's @48 with the 09-06 @16) + @48 (anchor,
#        must reproduce 372.07 from sweep6's window).
#        Startup ledger captured BEFORE any load (sweep6 dropped it).
#   B1   same but --quantization fp8 --kv-cache-dtype fp8 on GPU2+3.
#        THE 拆因 ARM: 372 (bf16 TP2) --quantization--> X --replicas--> 619.
#        Answers "fp8 fits on one card, so why not quantize and keep TP2?".
#        Judged both ways: >=420 = quantization stands on its own on 2 cards;
#        <=372 = quantization buys nothing under TP2, and the whole gain is the
#        layout — the enabler story, which is just as publishable.
#   R2c  k=1 replica pair on GPU1+2, ports 8341/8342. @48total reproduces
#        618.8; @72total (36 lanes/replica vs a ~24-lane wall) tests that the
#        94.4% usage reading is a real wall and not a sampling artifact.
#   R2b  k=0 replica pair, same GPUs. @96total reproduces 622.7; @120t/@144t
#        bracket whether 622.7 is a knee or a cliff.
#
# Source-check casualties (recorded before launch, see README §2):
#   - GDN state dtype is a dead end: MambaDType stops at bfloat16 and
#     FUSED_GDN_STATE_DTYPES is (float32, bfloat16) — the 58% of the pool that
#     is mamba state is at its floor, now proven rather than assumed.
#   - AWQ is out on cost: it needs a fresh llmcompressor calibration and lands
#     in the same Marlin kernel family that already cost +57% TPOT on int4.
#
# Optional phases, both off by default so a plain run is the pre-registered
# 2-card design: RUN_EXTRA=1 (3b: k=1 vs k=2 at @16total — the dynamic-k
# question) and RUN_4CARD=1 (4 replicas; NOT a baseline, the baseline is 2 cards).
#
# Neighbour condition at launch: GPU0 held ~43.8GB by a foreign job with 284MB
# footprints on GPU1-3 — identical to sweep6's window, so B0's 372.07 anchor is
# directly comparable across the two windows.
#
# Known-good patterns reused: setsid wrapper, PGID kill, SIGABRT-tolerant bench,
# UTC container clock. Outputs to <out-dir-7>.

# Every heavy write is redirected off the container overlay and onto the host
# NVMe. On 2026-09-11 the overlay hit 894G/894G (635MB free) and vLLM could not
# start at all; the cause was that nothing had ever pointed VLLM_CACHE_ROOT /
# TRITON_CACHE_DIR / TMPDIR away from $HOME and /tmp, so every download and
# every compile cache from this whole line landed on the writable layer. The
# model dirs are symlinks into the same NVMe, so no path below changes.
export VLLM_CACHE_ROOT=<nvme-root>/cache/vllm
export TRITON_CACHE_DIR=<nvme-root>/cache/triton
export TMPDIR=<nvme-root>/tmp-run
VENV=<venv>/bin
MODEL=<model-dir>
OUT=<out-dir-7>
# Plain quotes here, NOT \" — this is a top-level assignment, so the backslash
# would stay literal and vLLM would get invalid JSON. The escaping is only
# correct one level in, inside boot_replica's double-quoted bash -c string.
SPEC1='{"method": "mtp", "num_speculative_tokens": 1}'
mkdir -p $OUT "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR"

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

# summarize_single: $1 tag $2 logfile
summarize_single() {
  echo "== $1 ==" >> $OUT/summary.txt
  grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Peak concurrent|Failed" "$2" >> $OUT/summary.txt 2>/dev/null
}

# summarize_dual: $1 tag $2 replica count. PITFALL 12 FIX: sweep6 used
# key + r":\s+" but the bench line is "Output token throughput (tok/s): 372.07"
# — text sits between key and colon, so every TOTAL row printed nan.
# Latency is reported as the MEAN across replicas, throughput as the SUM.
summarize_dual() {
  local tag=$1 n=$2
  echo "== $1 (x$n replicas, TOTAL) ==" >> $OUT/summary.txt
  for i in $(seq 1 $n); do
    grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Peak concurrent|Failed" \
      $OUT/bench_${tag}_p$i.log >> $OUT/summary.txt 2>/dev/null
  done
  $VENV/python - "$tag" "$n" "$OUT" <<'EOF' >> $OUT/summary.txt
import re, sys
tag, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def val(part, key):
    txt = open(f"{out}/bench_{tag}_p{part}.log").read()
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
for key in ["Output token throughput"]:
    v = [val(i, key) for i in range(1, n + 1)]
    print(f"{key}: " + " ".join(f"p{i}={x:.2f}" for i, x in enumerate(v, 1)) + f" TOTAL={sum(v):.2f}")
for key in ["Mean TTFT", "Mean TPOT"]:
    v = [val(i, key) for i in range(1, n + 1)]
    print(f"{key}: " + " ".join(f"p{i}={x:.2f}" for i, x in enumerate(v, 1)) + f" MEAN={sum(v)/len(v):.2f}")
EOF
}

engine_stats() { # $1 tag $2... server logs
  local tag=$1; shift
  for f in "$@"; do
    echo "--- $f" >> $OUT/engine_stats_${tag}.txt
    grep -a "loggers.py" "$f" | tail -40 >> $OUT/engine_stats_${tag}.txt
  done
}

# FULL_STARTUP — the whole memory ledger. sweep6's grep dropped
# "Available KV cache memory" / "Estimated CUDA graph memory" /
# "model weights take" / "non-default args", which is why TP2's ledger was
# missing and the four items never closed against 0.92 x 44.5 GiB.
startup_lines() { # $1... server logs
  for f in "$@"; do
    echo "--- $f" >> $OUT/server_startup_lines.txt
    grep -aE "Available KV cache memory|Estimated CUDA graph memory|CUDA graph pool memory|GPU KV cache size|Maximum concurrency|model weights take|Peak torch memory|non-default args|quantization|Using .*Kernel|speculative|max_num_scheduled" "$f" \
      | head -24 >> $OUT/server_startup_lines.txt
  done
}

spec_metrics() { # $1 tag $2... ports
  local tag=$1; shift
  for p in "$@"; do
    curl -s "http://127.0.0.1:$p/metrics" 2>/dev/null | grep -iE "spec|draft|accept" >> $OUT/specmetrics_${tag}.txt
  done
}

# bench_pair: $1 tag $2 conc $3 prompts $4 first port $5 replica count
# $6... server logs (for the 3 in-flight engine samples). Latency/throughput
# totals come from summarize_dual, called here so every point is self-contained.
bench_pair() {
  local tag=$1 conc=$2 np=$3 p0=$4 n=$5; shift 5
  local logs=("$@") i pids=() bp k
  for i in $(seq 0 $((n-1))); do
    bench_one $((p0+i)) $conc $np $OUT/bench_${tag}_p$((i+1)).log &
    pids+=($!)
  done
  bp=${pids[0]}
  # 3 engine samples while the load is live (sweep6 sampled once per phase and
  # lost R2b_c96t's engine state entirely).
  for k in 1 2 3; do
    sleep 8
    engine_stats ${tag}_s$k "${logs[@]}"
    kill -0 $bp 2>/dev/null || break
  done
  wait "${pids[@]}"
  gpu_snap
  summarize_dual $tag $n
}

# boot_replica sets the global LAST_PID. Not called via $(...) on purpose:
# a command-substitution subshell would have to outlive its own background job.
boot_replica() { # $1 port $2 gpu $3 logfile $4 extra flags
  setsid bash -c "CUDA_VISIBLE_DEVICES=$2 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
    --port $1 --served-model-name qwen38-27b --max-model-len 8192 \
    --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 $4" > $3 2>&1 &
  LAST_PID=$!
}

boot_tp2() { # $1 gpu-pair $2 logfile $3 quant flags ("" = bf16 baseline)
  setsid bash -c "CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
    --port 8343 --served-model-name qwen38-27b --max-model-len 8192 \
    --tensor-parallel-size 2 --max-num-seqs 128 $3" > $2 2>&1 &
  LAST_PID=$!
}

tp2_phase() { # $1 tag $2 gpu-pair $3 quant flags
  local tag=$1 gpus=$2 qf=$3 pid
  boot_tp2 "$gpus" $OUT/server_$tag.log "$qf"
  pid=$LAST_PID
  if wait_up 8343; then
    log "$tag up (pid $pid)"
    startup_lines $OUT/server_$tag.log
    bench_one 8343 16 32 $OUT/bench_${tag}_c16.log
    bench_one 8343 48 96 $OUT/bench_${tag}_c48.log
    summarize_single ${tag}_c16 $OUT/bench_${tag}_c16.log
    summarize_single ${tag}_c48 $OUT/bench_${tag}_c48.log
  else
    log "ABORT $tag boot failed"
  fi
  kill_srv $pid
  log "$tag phase done"
  gpu_snap
}

# Phase gate. The protocol is one arm at a time: run it, record every metric,
# analyse, reflect on what was missed, then decide the next. Unset = the full
# pre-registered window, so a plain `bash mtp_bench7.sh` still means what the
# pre-registration says it means.
#   PHASES=B0            bash mtp_bench7.sh   # just the baseline
#   PHASES="B0 B1"       bash mtp_bench7.sh
PHASES=${PHASES:-"B0 B1 R2c R2b B0p"}
want() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

log "sweep7 start: baseline fix + gain split + ceiling | PHASES=$PHASES"
gpu_snap

# ---------- P1: B0 baseline (TP2 bf16, GPU2+3) ----------
if want B0; then tp2_phase B0 "2,3" ""; fi

# ---------- P2: B1 拆因臂 (TP2 fp8 weights + fp8 KV, GPU2+3) ----------
if want B1; then tp2_phase B1 "2,3" "--quantization fp8 --kv-cache-dtype fp8"; fi

# ---------- P3: R2c k=1 pair (GPU1+2) ----------
if want R2c; then
boot_replica 8341 1 $OUT/server_R2c_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 2 $OUT/server_R2c_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2c pair up ($RC1/$RC2)"
  startup_lines $OUT/server_R2c_1.log $OUT/server_R2c_2.log
  bench_pair R2c_c48t 24 48 8341 2 $OUT/server_R2c_1.log $OUT/server_R2c_2.log
  spec_metrics R2c_c48t 8341 8342
  bench_pair R2c_c72t 36 72 8341 2 $OUT/server_R2c_1.log $OUT/server_R2c_2.log
else
  log "ABORT R2c boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2c phase done"
gpu_snap
fi

# ---------- P4: R2b k=0 pair, bracket the knee (GPU1+2) ----------
if want R2b; then
boot_replica 8341 1 $OUT/server_R2b_1.log ""; RB1=$LAST_PID
boot_replica 8342 2 $OUT/server_R2b_2.log ""; RB2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2b pair up ($RB1/$RB2)"
  startup_lines $OUT/server_R2b_1.log $OUT/server_R2b_2.log
  bench_pair R2b_c96t  48 96  8341 2 $OUT/server_R2b_1.log $OUT/server_R2b_2.log
  bench_pair R2b_c120t 60 120 8341 2 $OUT/server_R2b_1.log $OUT/server_R2b_2.log
  bench_pair R2b_c144t 72 144 8341 2 $OUT/server_R2b_1.log $OUT/server_R2b_2.log
else
  log "ABORT R2b boot failed"
fi
kill_srv $RB1; kill_srv $RB2
log "R2b phase done"
gpu_snap
fi

# ---------- P5: closing anchor B0' (TP2 bf16, GPU2+3) — window validity ----------
# Only meaningful when B0 also ran; a B0p alone has nothing to bracket.
if want B0p && want B0; then
tp2_phase B0p "2,3" ""
$VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import re, sys
out = sys.argv[1]
def val(tag, key):
    txt = open(f"{out}/bench_{tag}_c48.log").read()
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
a, b = val("B0", "Output token throughput"), val("B0p", "Output token throughput")
d = abs(b - a) / a * 100
print(f"== WINDOW CHECK ==\nB0={a:.2f} B0'={b:.2f} drift={d:.2f}% -> " + ("VALID" if d <= 3 else "VOID (>3%)"))
EOF
fi

# ---------- P6 (opt-in): dynamic-k question — k=1 vs k=2 at @16total ----------
if [ "${RUN_EXTRA:-0}" = "1" ]; then
  for K in 1 2; do
    SPEC='{\"method\": \"mtp\", \"num_speculative_tokens\": '$K'}'
    boot_replica 8341 1 $OUT/server_Rkd${K}_1.log "--speculative-config '$SPEC'"; P1=$LAST_PID
    boot_replica 8342 2 $OUT/server_Rkd${K}_2.log "--speculative-config '$SPEC'"; P2=$LAST_PID
    if wait_up 8341 && wait_up 8342; then
      log "R2kd${K} pair up (k=$K)"
      startup_lines $OUT/server_Rkd${K}_1.log $OUT/server_Rkd${K}_2.log
      bench_pair R2kd${K}_c16t 8 16 8341 2 $OUT/server_Rkd${K}_1.log $OUT/server_Rkd${K}_2.log
      spec_metrics R2kd${K}_c16t 8341 8342
    else
      log "ABORT R2kd${K} boot failed"
    fi
    kill_srv $P1; kill_srv $P2
    log "R2kd${K} phase done"
  done
fi

# ---------- P7 (opt-in, RUN_4CARD=1): 4 replicas — NOT a baseline ----------
# The baseline stays 2 cards. This phase answers the one question 2 cards
# cannot: does a replica keep its per-replica throughput when four of them
# share one host, or does the host penalty compound? @96total = 24 lanes per
# replica, identical load to R2c_c48t, so it is directly comparable.
if [ "${RUN_4CARD:-0}" = "1" ]; then
  GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [ "${GPU0_USED:-99999}" -gt 2048 ]; then
    log "SKIP P7: GPU0 busy (${GPU0_USED} MiB)"
  else
    R4=()
    for spec in "8341 0" "8342 1" "8343 2" "8344 3"; do
      set -- $spec
      boot_replica $1 $2 $OUT/server_R4c_$(( $1 - 8340 )).log "--speculative-config '$SPEC1'"
      R4+=($LAST_PID)
    done
    if wait_up 8341 && wait_up 8342 && wait_up 8343 && wait_up 8344; then
      log "R4c quad up (${R4[*]})"
      startup_lines $OUT/server_R4c_1.log $OUT/server_R4c_2.log $OUT/server_R4c_3.log $OUT/server_R4c_4.log
      bench_pair R4c_c96T 24 48 8341 4 $OUT/server_R4c_1.log $OUT/server_R4c_2.log $OUT/server_R4c_3.log $OUT/server_R4c_4.log
      spec_metrics R4c_c96T 8341 8342 8343 8344
    else
      log "ABORT R4c boot failed"
    fi
    for p in "${R4[@]}"; do kill_srv $p; done
    log "R4c phase done"
  fi
fi

gpu_snap
log "ALL_DONE"
