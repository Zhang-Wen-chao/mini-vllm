#!/bin/bash
# sweep7: fix the baseline, split the gain, pin the ceiling.
# Implements the pre-registration in README ("sweep7 预注册"). Design first,
# script second — the arm list here is a transcription of that document, not a
# substitute for it.
#
# Window order (all one window): B0 -> B1 -> B1b -> R2c -> R2b -> B1c -> B0'
#   B0 and B0' bracket the window; |B0' - B0| > 3% voids the whole window.
#   B1b is the amendment added after B1 exposed pitfall 19 (see below).
#
# Extension window (amendment 3, run after the P5 anchor): B0' -> R2b48 -> R2c0 -> B0''
#   B0' (374.74) opens, B0'' closes. The P5 check already measured B0 -> B0' drift
#   at 0.09%, so this second window inherits the same comparability. It exists to
#   stop borrowing sweep6's R2b@48t as the shared denominator of two factors.
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
# The replica card pair. The pre-registered pin is GPU1+2 — every replica arm in
# sweep7 ran there — so an unset environment reproduces the registered design
# exactly. It is overridable so a contended night can use whichever pair the
# neighbour leaves free instead of losing the window entirely.
# What the override does and does not cost: every phase's OWN comparisons happen
# within one boot on one pair, so the pair cancels out of them (R2bZ's ladder,
# R2cZ's noise floor and crossover, R2cABr's three points are all untouched).
# It only matters for CROSS-phase links — the crossover against the recorded
# R2b_c120t/144t, and FV's 布局 rung against R2b48z. B1c priced the pair effect
# at 2.21%, so those links carry a <=2.2% caveat that must be stated wherever
# the number appears. Pairs must not be MIXED within a phase: both replicas of a
# boot always take RPA and RPB.
RPA=${RPA:-1}
RPB=${RPB:-2}
mkdir -p $OUT "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR"

log() { echo "$(date +%H:%M:%S) $*" >> $OUT/run.log; }

# gpu_snap: memory + util ALONE cannot tell "the machine got faster" from "our
# config changed" — that ambiguity cost a whole diagnostic round on the B0
# anchor. SM clock / power / temperature catch clock and thermal drift; the
# host loadavg catches a neighbour taking CPU on this shared box (loadavg ran
# 16/10/9 during B0, which no earlier sweep had ever recorded).
gpu_snap() { echo "--- $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/gpu_snapshots.txt
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv,noheader >> $OUT/gpu_snapshots.txt
  # PITFALL 22 — the original query carried clocks.sm but NOT clocks.mem, and no
  # throttle-reason field. A uniform +9% per-decode-step slowdown (R2c_c48t ran
  # 6.04 ms/step against 5.55 for the same config) is exactly what memory-clock
  # throttling or a power cap produces, and with those fields absent the record
  # could not tell "the machine was slow" from "our config changed" — the same
  # blind spot that cost a diagnostic round on the B0 anchor. Both fields exist
  # on this driver (checked before use: clocks.mem 9000 MHz, power.limit 350 W,
  # clocks_throttle_reasons.active 0x0), so they go in.
  nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv,noheader >> $OUT/gpu_snapshots.txt 2>/dev/null || true; }

# load_snap — loadavg DURING the load, not just before/after it. gpu_snap only
# brackets a bench point, so a neighbour that spikes for 30s mid-run is invisible
# in it. Amendment 4 needs the loadavg that a point was actually measured AT,
# because the k=1 arms swing ~9% and the one high-load sample is the prime suspect.
load_snap() { echo "$(date +%H:%M:%S) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/loadavg_trace.txt; }

declare -A BOOT_PID=()   # port -> pid we spawned; wait_up must ask about THIS pid

wait_up() { # $1 port
  for i in $(seq 1 120); do
    sleep 10
    curl -s "http://127.0.0.1:$1/health" > /dev/null 2>&1 && return 0
    # Past the GPU deadline no server was ever launched on this port, so do not
    # burn the full 20-minute budget waiting for one. The phase still logs its
    # ABORT line, so the failure stays visible in run.log either way.
    [ "${GPU_GATE_FAILED:-0}" = "1" ] && return 1
    # PITFALL 25b — a DEAD server is not a SLOW server. When a boot loses the
    # CUDA-allocation race it exits within ~2 minutes with "Engine core
    # initialization failed"; the old loop then sat here for the full 20 minutes
    # waiting on a port nothing would ever bind. Check whether the process we
    # launched still exists (grace of 4 polls = 40s so the python start-up does
    # not count as death) and bail out immediately if it does not.
    #
    # PITFALL 26 — this check used to be `pgrep -f -- "--port $1 "`, which asks
    # "does ANY process carry this port", not "is MY process alive". An orphaned
    # server from an earlier run carries the same --port and answers /health
    # perfectly, so the phase silently inherited a stale config (see port_precheck
    # below). Ask about the pid we actually spawned.
    if [ "$i" -gt 4 ]; then
      local mine="${BOOT_PID[$1]:-}"
      if [ -n "$mine" ]; then
        kill -0 "$mine" 2>/dev/null || {
          log "wait_up $1: our pid $mine died after $((i*10))s — boot failed (bind conflict or OOM; see server log)"
          return 1; }
      elif ! pgrep -f -- "--port $1 " > /dev/null 2>&1; then
        log "wait_up $1: no process for this port after $((i*10))s — boot died (check server log for OOM)"
        return 1
      fi
    fi
  done
  return 1
}

# PITFALL 26 — a port that is already serving does not belong to us, and booting
# onto it does NOT fail loudly. On 2026-09-11 an orphaned k=0 server (survived a
# kill that orphaned it to ppid=1) held 8341 from 16:36 onward. Two phases in a
# row booted "successfully" onto it: the new server died with
# `OSError: [Errno 98] Address already in use` into a 40-line log, `wait_up`
# returned 0 because the ORPHAN answered /health, and every request went to the
# stale process. R2cZ spent its whole phase measuring a k=0 server while the
# phase under test was k=1 — and the contamination was invisible for the exact
# reason it mattered: the numbers looked plausible (274 tok/s/replica is a real
# k=0 rate), only the *config* was wrong.
# So: refuse to boot onto an occupied port, loudly, before spawning anything.
port_precheck() { # $1 port
  if curl -s --max-time 2 "http://127.0.0.1:$1/health" > /dev/null 2>&1; then
    log "PORT_CONFLICT $1 already answering /health — refusing to boot onto it (stale server?)"
    echo "$(date +%H:%M:%S) CONFLICT $1 busy" >> $OUT/port_conflict.txt
    ps -eo pid,ppid,etime,args | grep -- "--port $1 " | grep -v grep >> $OUT/port_conflict.txt 2>/dev/null
    return 1
  fi
  return 0
}

kill_srv() { # $1 pid
  local pgid=$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
  sleep 8
  [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  sleep 2
}

# PITFALL 16 FIX — per-point seed.
# `--dataset-name random` is DETERMINISTIC under a fixed seed, and the seed
# defaulted to 0 for every point. So a 96-prompt run at conc 48 contains the
# 32 prompts of an earlier 32-prompt run at conc 16 verbatim (32/96 = 33%), and
# the second bench reads a prefix cache the first bench warmed. Measured:
# B0_c16 showed "Prefix cache hit rate: 0.0%" throughout while B0_c48 — the very
# next bench on the same boot — showed 29-32%, and its Median TPOT fell to
# 88.19ms against sweep6's 116.24ms for the same configuration. Throughput came
# out +14.4% over sweep6's TP2 anchor on byte-identical server flags, an
# identical KV pool (264,071 tok / 32.24x) and an identical client namespace.
# Every bench point now takes its own seed, so prompt pools never overlap and
# the measurement is order-independent.
#
# PITFALL 21 — THAT FIX WAS INCOMPLETE, and the reproducibility audit caught it.
# `SEED=$((SEED + 1))` looked per-point, but SEED is a SCRIPT-LEVEL global:
# every separate `bash mtp_bench7.sh` invocation restarted at 700, so tags
# measured in different invocations collided. Extracted from each bench log's
# own `Namespace(... seed=N ...)` header (sweep7/seeds.txt): 701 used 6x,
# 702 used 6x, 703 used 2x, 0 used 2x.
#
# Contamination did NOT follow. Every collision is ACROSS boots — each phase
# boots a fresh server, so no prompt pool was ever re-read from a warm KV cache
# — and 349/349 engine samples read `Prefix cache hit rate: 0.0%`. What it did
# cost was COMPARABILITY: the 调度步长 segment pits B1_c48 (seed 702) against
# B1b_c48 (seed 701), two different prompt pools.
#
# The fix is a FROZEN table, NOT a new derivation. Re-deriving by hash would
# hand every future re-run a different prompt set from the run it exists to
# reproduce — the opposite of the point. Tags already measured keep the seed
# they were actually measured with; anything not listed falls through to a
# tag-keyed hash, so new tags are invocation-independent and cannot collide
# with a recorded one. FORCE_SEED still overrides everything.
PHASE_ID=""   # set by each bench entry point; see seed_snap()
SEED_TABLE=$(cat <<'TBL'
B0_c16 701
B0_c48 702
B0p_c16 701
B0p_c48 702
B0pp_c48 703
B1_c16 701
B1_c48 702
B1b_c48 701
B1b_c96 702
B1c_c48 701
R2b48_c48t_p1 701
R2b48_c48t_p2 702
R2b_c120t_p1 708
R2b_c120t_p2 709
R2b_c144t_p1 710
R2b_c144t_p2 711
R2b_c96t_p1 706
R2b_c96t_p2 707
R2c0_c48t_p1 0
R2c0_c48t_p2 0
R2c0b_c48t_p1 0
R2c0b_c48t_p2 0
R2cAB00_c48t_p1 0
R2cAB00_c48t_p2 0
R2c_c48t_p1 702
R2c_c48t_p2 703
R2c_c72t_p1 704
R2c_c72t_p2 705
TBL
)

# seed_for — $1 point tag (the name in bench_<tag>.log) -> client seed.
# Table hit wins so recorded tags reproduce byte for byte; anything else gets a
# tag-keyed hash. Stable across invocations, and the 1000+ floor keeps new tags
# clear of the 0-711 range the frozen table occupies, so a new tag can never
# silently land on a recorded one.
seed_for() {
  local s
  s=$(awk -v t="$1" '$1==t {print $2; exit}' <<<"$SEED_TABLE")
  if [ -n "$s" ]; then echo "$s"; else
    printf '%s' "$1" | cksum | awk '{print 1000 + ($1 % 8000)}'
  fi
}

# seed_snap — every point writes its own tag->seed pair to seeds_used.txt, so
# the map is auditable with a grep rather than decoded out of each bench log's
# 2000-character Namespace dump. The collision alarm is the part the old
# counter structurally could not have: 坑 21 was invisible precisely because
# nothing ever compared one point's seed against another's.
#
# It deliberately does NOT fire on legitimate alias pairs (B0_c16/B0p_c16/
# B1_c16 all share 701 by measurement). Those are different phases, each on a
# fresh boot with a cold KV cache, so sharing a pool cannot contaminate
# anything — 坑 16 needed the SAME server to serve two benches to bite. The
# alarm exists for the case that WOULD be new: two points inside one phase.
seed_snap() { # $1 point tag $2 seed
  local prev
  # FORCE_SEED shares one seed across both replicas of an arm ON PURPOSE, so the
  # alarm must not fire on it — R2cAB's first run reported exactly that pair and
  # it was the design working, not a defect.
  #
  # CORRECTION (found while auditing R2cAB): an earlier version of this comment
  # claimed the replicas take DISJOINT halves of the pool. They do not. In
  # bench_pair, `p0+i` is the PORT, not a prompt offset — bench_one's 3rd arg is
  # np for every replica, so each replica serves the SAME 48 prompts on its own
  # server (Total input tokens 51668 = 2 x 25834 confirms it). The sharing is
  # safe for a different reason: two servers, two KV caches, so the same prompt
  # is never read from a cache warmed by its twin — which is precisely what
  # 坑 16 needed to bite. Cross-replica pool identity is harmless; the alarm is
  # for the case that is NOT deliberate, namely two points in one phase that
  # reached the same seed by arithmetic, which is what 坑 21 produced.
  if [ -z "${FORCE_SEED:-}" ]; then
    prev=$(awk -v s="$2" -v t="$1" -v p="$PHASE_ID" \
               '$2==s && $1!=t && $3==p {print $1; exit}' $OUT/seeds_used.txt 2>/dev/null)
    [ -n "$prev" ] && log "SEED COLLISION (same phase): $1 seed=$2 already used by $prev"
  fi
  echo "$1 $2 $PHASE_ID" >> $OUT/seeds_used.txt
}

# seed_audit — dump the whole frozen table at startup, with every alias group
# printed explicitly. The aliases are the audit's actual finding, not noise:
# each one is a place where two recorded points share a prompt pool, and the
# reader is entitled to see the list rather than trust that it is empty.
seed_audit() {
  { echo "== SEED AUDIT $(date +%H:%M:%S) =="
    echo "$SEED_TABLE" | awk '
      {n[$2] = n[$2] " " $1}
      END {for (s in n) {c = split(n[s], a, " ")
                         if (c > 1) printf "  alias seed %-4s -> %s\n", s, n[s]}}'
  } >> $OUT/summary.txt
}
# Every metric line `vllm bench serve` prints. Both summarizers grep this one
# pattern, so the single-card and replica paths can never drift apart — and the
# per-point summary carries the full latency distribution (median/P99 TPOT, the
# ITL trio, token totals), not just the handful sweep6's summary kept. The raw
# bench_*.log always had them; the summarizer was throwing them away.
METRICS_RE="Successful requests|Failed requests|Benchmark duration|Maximum request concurrency|Peak concurrent|Request throughput|Output token throughput|Peak output token throughput|Total token throughput|Total input tokens|Total generated tokens|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL"
bench_one() { # $1 port $2 concurrency $3 num_prompts $4 logfile $5 seed
  HF_HUB_OFFLINE=1 $VENV/vllm bench serve --backend openai-chat \
    --endpoint /v1/chat/completions --tokenizer $MODEL --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --temperature 0 \
    --seed ${5:-0} \
    --num-prompts $3 --max-concurrency $2 --host 127.0.0.1 \
    --model qwen38-27b --port $1 > "$4" 2>&1
}

# summarize_single: $1 tag $2 logfile
summarize_single() {
  echo "== $1 ==" >> $OUT/summary.txt
  grep -E "$METRICS_RE" "$2" >> $OUT/summary.txt 2>/dev/null
}

# summarize_dual: $1 tag $2 replica count. PITFALL 12 FIX: sweep6 used
# key + r":\s+" but the bench line is "Output token throughput (tok/s): 372.07"
# — text sits between key and colon, so every TOTAL row printed nan.
# Latency is reported as the MEAN across replicas, throughput as the SUM.
summarize_dual() {
  local tag=$1 n=$2
  echo "== $1 (x$n replicas, TOTAL) ==" >> $OUT/summary.txt
  for i in $(seq 1 $n); do
    grep -E "$METRICS_RE" $OUT/bench_${tag}_p$i.log >> $OUT/summary.txt 2>/dev/null
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
#
# PITFALL 18 — the ledger's ONLY complete line is gpu_worker.py:804, and its
# wording changed across versions: older vLLM says "model weights take X GiB",
# 0.28.0 says "Actual usage is X GiB for consumed memory (weights + non-torch),
# Y GiB for peak activation, and Z GiB for CUDAGraph memory". The first sweep7
# run matched the OLD wording and therefore captured NO ledger line at all —
# silently, with no error, into an evidence file that looked complete. The
# numbers were sitting in server_<arm>.log the whole time. So: a missing ledger
# item must be confirmed against the raw server log, never against summary.txt.
startup_lines() { # $1... server logs
  for f in "$@"; do
    echo "--- $f" >> $OUT/server_startup_lines.txt
    grep -aE "Available KV cache memory|Estimated CUDA graph memory|CUDA graph pool memory|GPU KV cache size|Maximum concurrency|consumed memory|peak activation|CUDAGraph memory|Model loading took|Graph capturing finished|model weights take|Peak torch memory|non-default args|equivalent to --gpu-memory-utilization|No available shared memory broadcast" "$f" \
      | head -40 >> $OUT/server_startup_lines.txt
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
  local logs=("$@") i pids=() bp k s
  PHASE_ID=$tag
  for i in $(seq 0 $((n-1))); do
    # FORCE_SEED pins the client seed for BOTH replicas (seed 0 reproduces
    # sweep6's R2c_c48t prompt set byte for byte). Otherwise the seed comes from
    # seed_for() keyed on the per-REPLICA point tag, so two replicas of one arm
    # get different pools and a re-run of the arm gets the same pools it had.
    # Resolved in the PARENT: doing it inside the background subshell would
    # evaluate against a copy and the per-replica tags would drift.
    if [ -n "${FORCE_SEED:-}" ]; then s=$FORCE_SEED; else s=$(seed_for "${tag}_p$((i+1))"); fi
    seed_snap "${tag}_p$((i+1))" "$s"
    bench_one $((p0+i)) $conc $np $OUT/bench_${tag}_p$((i+1)).log $s &
    pids+=($!)
  done
  bp=${pids[0]}
  # 3 engine samples while the load is live (sweep6 sampled once per phase and
  # lost R2b_c96t's engine state entirely).
  for k in 1 2 3; do
    sleep 8
    engine_stats ${tag}_s$k "${logs[@]}"
    load_snap
    kill -0 $bp 2>/dev/null || break
  done
  wait "${pids[@]}"
  gpu_snap
  summarize_dual $tag $n
}

# boot_replica sets the global LAST_PID. Not called via $(...) on purpose:
# a command-substitution subshell would have to outlive its own background job.
# wait_gpus — block until every listed card is free, then return.
#
# This is a SHARED box and the night of 2026-09-11 proved it matters: at 16:07 a
# foreign job elsewhere on the host (16.7 GB resident on each of GPU1/2/3) took
# all three cards my whole design runs on. 别人的进程不能动 — we
# never kill it. We wait for it instead, and we RECORD the wait, because a
# neighbour holding a card is exactly what would explain a slow window later and
# I have already lost one diagnostic round to a stale GPU snapshot.
#
# The gate lives inside the two boot functions rather than at the ten call sites:
# every phase allocates its cards through one of these two, so two insertions
# cover the whole script and there is no way for a future phase to forget it.
#
# A card counts as busy at >2 GiB used. Our own servers claim ~44 GiB, a
# neighbour's footprint is ~16.7 GB, and an idle card reads 295 MiB — the
# threshold has an order of magnitude of headroom on both sides.
#
# PITFALL 25 — ONE FREE SAMPLE IS NOT FREE (2026-09-11 16:14, cost: one boot and
# a scare). The first version released the moment a single poll read free. That
# poll passed at 16:14:45 and both replicas started; the neighbour launched a new
# 41 GiB job 35 seconds later, my GPU1 replica died with "Engine core
# initialization failed" (CUDA OOM), and for a moment my 40 GiB allocation and
# theirs were racing for the same card. My job lost, which is the harmless
# direction — but on a SHARED box the coin can land the other way, and OOMing
# someone else's run is not a thing this script may ever do.
# Fix: FREE_HOLD consecutive free polls (default 10 x 30s = 5 min) before
# releasing, so a card that just went idle because a neighbour's job ended does
# not get handed to us in the gap before their next launch. It cannot make the
# race impossible (their launches are uncorrelated with us) — it makes the
# window we are exposed to small instead of 30 seconds wide.
FREE_HOLD=${FREE_HOLD:-10}
#
# The deadline exists so an overnight run terminates and reports rather than
# blocking forever if the neighbour never leaves.
WAIT_DEADLINE=${WAIT_DEADLINE:-$(($(date +%s) + 21600))}   # default 6h
GPU_GATE_FAILED=0
wait_gpus() { # $1 comma list, e.g. "1,2"
  local want=$1 g used busy n=0 held=0
  while :; do
    busy=""
    for g in ${want//,/ }; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ -n "$used" ] && [ "$used" -gt 2000 ] && busy="$busy $g:${used}MiB"
    done
    if [ -z "$busy" ]; then
      held=$((held+1))
      if [ "$held" -ge "$FREE_HOLD" ]; then
        [ "$n" -gt 0 ] && { log "GPUS_FREE $want after $n polls (held $held)"; \
          echo "$(date +%H:%M:%S) FREE $want waited $((n*30))s held ${held}x30s" >> $OUT/gpu_wait.txt; }
        return 0
      fi
      # Free but not yet proven stable — say so once, so a 5-minute hold does not
      # look like the script has hung.
      [ "$held" -eq 1 ] && log "GPUS_IDLE $want — holding $FREE_HOLD polls before boot"
    else
      if [ "$held" -gt 0 ]; then
        log "GPUS_FREE_ABORTED $want:$busy after $held free polls — neighbour returned"
        echo "$(date +%H:%M:%S) aborted-hold $want:$busy after ${held}x30s" >> $OUT/gpu_wait.txt
      fi
      held=0
    fi
    n=$((n+1))
    if [ $((n % 20)) -eq 1 ]; then
      log "WAIT_GPU $want:${busy:-idle-holding} (poll $n)"
      echo "$(date +%H:%M:%S) busy $want:${busy:-none} held=$held" >> $OUT/gpu_wait.txt
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $OUT/gpu_wait.txt
    fi
    if [ "$(date +%s)" -gt "$WAIT_DEADLINE" ]; then
      log "WAIT_GPU DEADLINE hit waiting for $want:${busy:-idle} — skipping this boot"
      echo "$(date +%H:%M:%S) DEADLINE $want:${busy:-none}" >> $OUT/gpu_wait.txt
      GPU_GATE_FAILED=1
      return 1
    fi
    sleep 30
  done
}

boot_replica() { # $1 port $2 gpu $3 logfile $4 extra flags
  port_precheck "$1" || { LAST_PID=""; return 1; }
  wait_gpus "$2" || { LAST_PID=""; return 1; }
  setsid bash -c "CUDA_VISIBLE_DEVICES=$2 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
    --port $1 --served-model-name qwen38-27b --max-model-len 8192 \
    --quantization fp8 --max-num-seqs 128 --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 $4" > $3 2>&1 &
  LAST_PID=$!
  BOOT_PID[$1]=$LAST_PID
}

boot_tp2() { # $1 gpu-pair $2 logfile $3 quant flags ("" = bf16 baseline)
  port_precheck 8343 || { LAST_PID=""; return 1; }
  wait_gpus "$1" || { LAST_PID=""; return 1; }
  setsid bash -c "CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 $VENV/vllm serve $MODEL \
    --port 8343 --served-model-name qwen38-27b --max-model-len 8192 \
    --tensor-parallel-size 2 --max-num-seqs 128 $3" > $2 2>&1 &
  LAST_PID=$!
  BOOT_PID[8343]=$LAST_PID
}

# bench_single_sampled: the TP2 path used to call bench_one bare and collect NO
# engine state at all — sweep6 got TP2's Running/Waiting/KV-usage by luck, and
# without it B0's @48 cannot be put next to R2c's 94.4% wall. Same 3-sample
# protocol as bench_pair so the two paths are symmetric.
bench_single_sampled() { # $1 tag $2 conc $3 np $4 port $5 server log
  local tag=$1 conc=$2 np=$3 port=$4 slog=$5 bp k s
  PHASE_ID=$tag
  if [ -n "${FORCE_SEED:-}" ]; then s=$FORCE_SEED; else s=$(seed_for "$tag"); fi
  seed_snap "$tag" "$s"
  bench_one $port $conc $np $OUT/bench_$tag.log $s &
  bp=$!
  for k in 1 2 3; do
    sleep 8
    engine_stats ${tag}_s$k "$slog"
    load_snap
    kill -0 $bp 2>/dev/null || break
  done
  wait $bp
  gpu_snap
  summarize_single $tag $OUT/bench_$tag.log
}

tp2_phase() { # $1 tag $2 gpu-pair $3 quant flags $4 concurrency list (default "16 48")
  local tag=$1 gpus=$2 qf=$3 concs=${4:-"16 48"} pid c
  boot_tp2 "$gpus" $OUT/server_$tag.log "$qf"
  pid=$LAST_PID
  if wait_up 8343; then
    log "$tag up (pid $pid)"
    startup_lines $OUT/server_$tag.log
    # Every point stays at 96 prompts: sweep1-5 measured every @16 with
    # `--num-prompts 96 --max-concurrency 16` (Total input tokens 103333 in all
    # ten arms), and the R2 arms' @96total used 96 prompts too — so 96 keeps the
    # whole matrix on one prompt pool. B0 run #2 used 32 at @16 and landed at
    # 237.07 vs 96's 239.61 (1.1%): at @16 the client concurrency, not the
    # prompt count, sets the rate (see README pitfall 17). 96 is the convention,
    # not a magic number — the point is that it is the SAME everywhere.
    for c in $concs; do
      bench_single_sampled ${tag}_c${c} $c 96 8343 $OUT/server_$tag.log
    done
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
PHASES=${PHASES:-"B0 B1 B1b R2c R2b B1c B0p R2b48 R2c0 B0pp R2c0b R2cH B0ppp R2cAB R2b48z SB"}
want() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

log "sweep7 start: baseline fix + gain split + ceiling | PHASES=$PHASES"
seed_audit
: > $OUT/seeds_used.txt   # per-run; seed_snap appends the tag->seed->phase map
gpu_snap

# ---------- P1: B0 baseline (TP2 bf16, GPU2+3) ----------
if want B0; then tp2_phase B0 "2,3" ""; fi

# ---------- P2: B1 拆因臂 (TP2 fp8 weights + fp8 KV, GPU2+3) ----------
if want B1; then tp2_phase B1 "2,3" "--quantization fp8 --kv-cache-dtype fp8"; fi

# ---------- P2b: B1b 桥接臂 (PITFALL 19) ----------
# vLLM 0.28.0's default max_num_batched_tokens is context-keyed:
# {LLM_CLASS: 8192, OPENAI_API_SERVER: 2048}. `vllm serve` = 2048, so B0 and B1
# ran at 2048 while sweep6's replica arms passed 8192 EXPLICITLY — the +66%
# headline spans a 4x prefill-step-cap difference that was never in the
# fairness list. B1b adds the missing cell: same arm as B1 plus the flag, so
# B1b vs R2b/R2c isolates LAYOUT at equal quant + equal cap.
# @96 answers the other open question: B1 sat at only 13.6% KV usage @48, so its
# 488 may be load-limited rather than ceiling-limited.
if want B1b; then tp2_phase B1b "2,3" "--quantization fp8 --kv-cache-dtype fp8 --max-num-batched-tokens 8192" "48 96"; fi

# ---------- P2c: B1c 卡对效应控制臂 (B1b 的配置换到 GPU1+2, 只跑 @48) ----------
# B0/B1/B1b live on GPU2+3 and every R2 arm lives on GPU1+2 — a cross-card-pair
# difference is asserted in the fairness list but was never measured, and the
# layout factor it would confound is only +7.5%. One point settles it.
if want B1c; then tp2_phase B1c "1,2" "--quantization fp8 --kv-cache-dtype fp8 --max-num-batched-tokens 8192" "48"; fi

# ---------- P3: R2c k=1 pair (GPU1+2) ----------
if want R2c; then
boot_replica 8341 $RPA $OUT/server_R2c_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2c_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
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
boot_replica 8341 $RPA $OUT/server_R2b_1.log ""; RB1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2b_2.log ""; RB2=$LAST_PID
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
# The gate tests the ANCHOR ARTIFACT, not the PHASES list. Gating on `want B0`
# meant `PHASES=B0p` alone silently did nothing — the phase was skipped with no
# error, which is the same failure mode as pitfall 18 (a check that can only
# fail silently). What the window check needs is bench_B0_c48.log on disk; if
# it is there, the bracket is meaningful regardless of how it was produced.
if want B0p && [ -s $OUT/bench_B0_c48.log ]; then
tp2_phase B0p "2,3" ""
$VENV/python - "$OUT" "B0" "B0p" <<'EOF' >> $OUT/summary.txt
import re, sys
out, t1, t2 = sys.argv[1], sys.argv[2], sys.argv[3]
def val(tag, key):
    txt = open(f"{out}/bench_{tag}_c48.log").read()
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
a, b = val(t1, "Output token throughput"), val(t2, "Output token throughput")
d = abs(b - a) / a * 100
print(f"== WINDOW CHECK {t1} vs {t2} ==\n{t1}={a:.2f} {t2}={b:.2f} drift={d:.2f}% -> " + ("VALID" if d <= 3 else "VOID (>3%)"))
EOF
fi

# ---------- P8/P9 + closing anchor: amendment 3 extension window ----------
# The 布局 and 投机 factors both rest on ONE borrowed denominator — sweep6's
# R2b@48t = 548.17. Their PRODUCT is already same-window (R2c@48t / B1b =
# 600.40/509.83 = +17.77%), so what is unknown is only how to SPLIT it. This
# extension buys the split directly instead of borrowing it across windows.
# Window: B0p (the P5 anchor, already measured) opens, B0pp closes.

# ---------- P8: R2b48 — the k=0 @24-lane cell, same window as the k=1 one ----------
if want R2b48; then
boot_replica 8341 $RPA $OUT/server_R2b48_1.log ""; RB1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2b48_2.log ""; RB2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2b48 pair up ($RB1/$RB2)"
  startup_lines $OUT/server_R2b48_1.log $OUT/server_R2b48_2.log
  bench_pair R2b48_c48t 24 48 8341 2 $OUT/server_R2b48_1.log $OUT/server_R2b48_2.log
else
  log "ABORT R2b48 boot failed"
fi
kill_srv $RB1; kill_srv $RB2
log "R2b48 phase done"
gpu_snap
fi

# ---------- P15: SB — 调度步长 on ONE prompt pool (closes the 坑 21 hole) ----------
# The reproducibility audit found the 调度步长 row was the only one in the
# four-factor table whose two ends sat on DIFFERENT prompt pools: B1_c48 was
# measured at seed 702 and B1b_c48 at seed 701, because SEED was a per-invocation
# global (坑 21). Every other row is safe — the k=0 pool-sensitivity bound is
# <=0.25% by direct measurement (B0p 702->B0pp 703 = 0.18%, R2b48 cross-window
# 0.25%, R2b_c96t 0.03%) — but "bounded by a measurement made elsewhere" is a
# weaker claim than "measured directly", and this is the row the whole +4.5%
# rests on. Two boots (the flag needs a different server), both FORCE_SEED=702,
# both @48: the ONLY difference is --max-num-batched-tokens.
if want SB; then
FORCE_SEED=702 tp2_phase SB2048 "2,3" "--quantization fp8 --kv-cache-dtype fp8" "48"
FORCE_SEED=702 tp2_phase SB8192 "2,3" "--quantization fp8 --kv-cache-dtype fp8 --max-num-batched-tokens 8192" "48"
$VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import re, sys
out = sys.argv[1]
def val(tag, key):
    txt = open(f"{out}/bench_{tag}_c48.log").read()
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
a, b = val("SB2048", "Output token throughput"), val("SB8192", "Output token throughput")
print(f"== SB same-pool 调度步长 A/B (both seed 702) ==\n"
      f"2048={a:.2f} 8192={b:.2f} gain={(b-a)/a*100:+.2f}%\n"
      f"compare recorded B1 488.01 -> B1b 509.83 = +4.47% (different pools)")
EOF
fi

# ---------- P16: R2b48z — the k=0 @48t denominator on seed 0 ----------
# Same reason as SB, applied to the headline row. The 投机 factor currently
# divides R2c(seed 0) by R2b48(seeds 701/702), so numerator and denominator sit
# on different pools. Re-measuring the denominator at seed 0 makes the ratio
# same-pool as well as same-window, which is what the +74% claim needs.
if want R2b48z; then
boot_replica 8341 $RPA $OUT/server_R2b48z_1.log ""; RZ1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2b48z_2.log ""; RZ2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2b48z pair up ($RZ1/$RZ2)"
  startup_lines $OUT/server_R2b48z_1.log $OUT/server_R2b48z_2.log
  FORCE_SEED=0 bench_pair R2b48z_c48t 24 48 8341 2 $OUT/server_R2b48z_1.log $OUT/server_R2b48z_2.log
else
  log "ABORT R2b48z boot failed"
fi
kill_srv $RZ1; kill_srv $RZ2
log "R2b48z phase done"
gpu_snap
fi

# ---------- P17: R2cABr — three points on one boot: 0, 702, 0 ----------
# R2cAB (seed702 645.39 then seed0 661.29, +2.46%) has THREE readings and they
# cannot be told apart from one pair:
#   (i)  a genuine prompt-pool effect;
#   (ii) an ORDER effect — the first bench on a fresh boot may be the slow one;
#   (iii) ordinary noise. TTFT alone swung 1912 / 2415 / 2430 ms across three
#        seed-0 points on three boots, i.e. ~20%, and r0 vs r1 sit 40s apart.
# Running 0, 702, 0 in that order separates all three: if the two seed-0 points
# agree and 702 sits below both, the pool effect is real and order is excluded;
# if point 1 and point 3 straddle 702, it was order; if all three scatter ~2%,
# then 2% IS the noise floor here and the amendment-4 band must be widened.
# Three benches back to back, no anchor — the points bracket each other.
if want R2cABr; then
boot_replica 8341 $RPA $OUT/server_R2cABr_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2cABr_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2cABr pair up ($RC1/$RC2)"
  startup_lines $OUT/server_R2cABr_1.log $OUT/server_R2cABr_2.log
  FORCE_SEED=0   bench_pair R2cABr0a_c48t 24 48 8341 2 $OUT/server_R2cABr_1.log $OUT/server_R2cABr_2.log
  FORCE_SEED=702 bench_pair R2cABr70_c48t 24 48 8341 2 $OUT/server_R2cABr_1.log $OUT/server_R2cABr_2.log
  FORCE_SEED=0   bench_pair R2cABr0b_c48t 24 48 8341 2 $OUT/server_R2cABr_1.log $OUT/server_R2cABr_2.log
  spec_metrics R2cABr0b_c48t 8341 8342
else
  log "ABORT R2cABr boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2cABr phase done"
gpu_snap
fi

# ---------- P18: R2bZ — the k=0 ladder on ONE pool (seed 0) ----------
# Three things at once, all on one boot and one prompt pool:
#   * 108t/132t (54/66 lanes) bracket the 682.96 peak at 120t (60 lanes) from
#     both sides, so "the ceiling is at 60 lanes" stops resting on a single
#     sample flanked by 96t and 144t, which are 12 lanes away on each side;
#   * 120t/144t re-measured at seed 0 give the k=1 crossover a denominator on
#     the SAME pool as its numerator — currently R2b_c120t/144t used seeds
#     708-711 while R2cH used seed_for values, so the crossover compared two
#     different pools;
#   * the whole ladder at one seed on one boot is internally consistent, which
#     the existing ladder (seeds 708-711, three separate boots) is not.
if want R2bZ; then
boot_replica 8341 $RPA $OUT/server_R2bZ_1.log ""; ZB1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2bZ_2.log ""; ZB2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2bZ pair up ($ZB1/$ZB2)"
  startup_lines $OUT/server_R2bZ_1.log $OUT/server_R2bZ_2.log
  for c in 108 120 132 144; do
    FORCE_SEED=0 bench_pair R2bZ_c${c}t $((c/2)) $c 8341 2 $OUT/server_R2bZ_1.log $OUT/server_R2bZ_2.log
  done
  spec_metrics R2bZ_c144t 8341 8342
  # Print the ladder so the peak bracketing exists as text next to the raw logs
  # (same reason as R2cZ's block: the number should not have to be recomputed by
  # hand later, and a printed line is greppable while a hand-computed one is not).
  $VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import os, re, sys
out = sys.argv[1]
def total(tag):
    s = 0.0
    for i in (1, 2):
        f = f"{out}/bench_{tag}_p{i}.log"
        if not os.path.exists(f):
            return float("nan")   # partial ladder -> nan, never a silent half-sum
        m = re.search(re.escape("Output token throughput") + r"[^\n:]*:\s+([0-9.]+)", open(f).read())
        s += float(m.group(1)) if m else float("nan")
    return s
print("== R2bZ k=0 ladder, ONE pool (seed 0), ONE boot ==")
lad = [(c, total(f"R2bZ_c{c}t")) for c in (108, 120, 132, 144)]
ok = [(c, v) for c, v in lad if v == v]          # drop nan before max(): nan
peak_c, peak_v = max(ok, key=lambda kv: kv[1]) if ok else (None, float("nan"))
for c, v in lad:
    tag = ""
    if c == peak_c:
        tag = "   <- peak"
    elif v != v:
        tag = "   <- MISSING"
    print(f"   @{c}t ({c//2} lanes/rep): {v:.2f}" + tag)
print(f"   peak = @{peak_c}t {peak_v:.2f}; vs recorded R2b_c120t 682.96 = {(peak_v-682.96)/682.96*100:+.2f}%")
print( "   NOTE: every point below is the SAME pool (seed 0) on ONE boot, so the")
print( "   SHAPE of this ladder is comparable to itself; the recorded 96t/120t/144t")
print( "   ladder used seeds 706-711 across three boots and is not.")
EOF
else
  log "ABORT R2bZ boot failed"
fi
kill_srv $ZB1; kill_srv $ZB2
log "R2bZ phase done"
gpu_snap
fi

# ---------- P19: R2cZ — k=1 repeats and the crossover, one pool (seed 0) ----------
# The audit's biggest hole was that NO k=1 arm had ever been measured twice at
# the same setting, so every k=1 number carried an unquantified error bar — and
# after the 600.40 anomaly that error bar is the thing that matters most. Three
# back-to-back c48t points on one boot measure it directly. Then 120t/144t at
# the same pool as R2bZ give the crossover as a same-pool, same-window number.
if want R2cZ; then
boot_replica 8341 $RPA $OUT/server_R2cZ_1.log "--speculative-config '$SPEC1'"; ZC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2cZ_2.log "--speculative-config '$SPEC1'"; ZC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2cZ pair up ($ZC1/$ZC2)"
  startup_lines $OUT/server_R2cZ_1.log $OUT/server_R2cZ_2.log
  for r in a b c; do
    FORCE_SEED=0 bench_pair R2cZ_c48t_${r} 24 48 8341 2 $OUT/server_R2cZ_1.log $OUT/server_R2cZ_2.log
    spec_metrics R2cZ_c48t_${r} 8341 8342
  done
  FORCE_SEED=0 bench_pair R2cZ_c120t 60 120 8341 2 $OUT/server_R2cZ_1.log $OUT/server_R2cZ_2.log
  FORCE_SEED=0 bench_pair R2cZ_c144t 72 144 8341 2 $OUT/server_R2cZ_1.log $OUT/server_R2cZ_2.log
  spec_metrics R2cZ_c144t 8341 8342
  # Report the k=1 noise floor and the same-pool crossover here, so the numbers
  # exist as text next to the raw logs rather than being recomputed by hand later.
  $VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import os, re, sys, statistics as st
out = sys.argv[1]
def g(f, key):
    if not os.path.exists(f):
        return float("nan")
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", open(f).read())
    return float(m.group(1)) if m else float("nan")
def total(tag):
    return sum(g(f"{out}/bench_{tag}_p{i}.log", "Output token throughput") for i in (1, 2))
# The noise floor prints FIRST and the crossover is guarded: if R2bZ was skipped
# (deadline, failed boot) its files are absent, and an unguarded open() would
# kill the block AFTER the noise floor was printed but BEFORE anything else --
# losing the crossover silently in the summary. Missing files now read as nan.
rep = [total(f"R2cZ_c48t_{r}") for r in "abc"]
print("== R2cZ k=1 @48t noise floor (3 back-to-back, one boot, seed 0) ==")
print("   " + "  ".join(f"{v:.2f}" for v in rep))
if all(v == v for v in rep):
    print(f"   mean={st.mean(rep):.2f} sd={st.pstdev(rep):.2f} spread={(max(rep)-min(rep))/st.mean(rep)*100:.2f}%")
print("== R2cZ vs R2bZ crossover (SAME pool, both seed 0) ==")
for c in (108, 120, 132, 144):
    k0, k1 = total(f"R2bZ_c{c}t"), total(f"R2cZ_c{c}t")
    if k0 != k0 or k1 != k1:
        print(f"   @{c}t ({c//2} lanes/rep): k=0 {k0}  k=1 {k1}  -> n/a (incomplete)")
    else:
        print(f"   @{c}t ({c//2} lanes/rep): k=0 {k0:.2f}  k=1 {k1:.2f}  -> {(k1-k0)/k0*100:+.2f}%")
EOF
else
  log "ABORT R2cZ boot failed"
fi
kill_srv $ZC1; kill_srv $ZC2
log "R2cZ phase done"
gpu_snap
fi

# ---------- P21: Z2 — the crossover's missing rung (@132t), both ladders WARM ----------
# Two holes, one phase.
# (1) R2bZ and R2cZ sample 108/120/144 but NEVER 132 — yet that is exactly where
#     k=0's own maximum turned out to sit (703.26, +2.97% over the recorded peak).
#     So the k=0/k=1 crossover could only be localized to the unsampled (60,72]
#     lane band. This phase measures the missing rung on both sides.
# (2) The night's other finding: the FIRST measured point after a boot is ~1.5%
#     slow and the rest plateau. The cold ladders are therefore not comparable
#     position-for-position (R2bZ's @120t was its 2nd point, a re-run's would be
#     its 4th). Here BOTH ladders discard one warm-up point first and then take
#     the four rungs at identical positions, so any remaining k=0/k=1 difference
#     is the thing under test rather than a position artefact.
# Bonus: the discarded warm-up is the same shape and pool as @120t (conc 60,
# 120 prompts, seed 0), so w0 -> c120t is a controlled pos1-vs-pos3 measurement
# of the ramp itself, on both arms.
if want Z2; then
  for spec in 0 1; do
    if [ "$spec" = "1" ]; then sp="--speculative-config '$SPEC1'"; pfx=R2cZw
    else sp=""; pfx=R2bZw; fi
    boot_replica 8341 $RPA $OUT/server_${pfx}_1.log "$sp"; Z21=$LAST_PID
    boot_replica 8342 $RPB $OUT/server_${pfx}_2.log "$sp"; Z22=$LAST_PID
    if wait_up 8341 && wait_up 8342; then
      log "$pfx pair up ($Z21/$Z22)"
      startup_lines $OUT/server_${pfx}_1.log $OUT/server_${pfx}_2.log
      FORCE_SEED=0 bench_pair ${pfx}_w0 60 120 8341 2 $OUT/server_${pfx}_1.log $OUT/server_${pfx}_2.log
      for c in 108 120 132 144; do
        FORCE_SEED=0 bench_pair ${pfx}_c${c}t $((c/2)) $c 8341 2 $OUT/server_${pfx}_1.log $OUT/server_${pfx}_2.log
      done
      spec_metrics ${pfx}_c132t 8341 8342
    else
      log "ABORT $pfx boot failed"
    fi
    kill_srv $Z21; kill_srv $Z22
    gpu_snap
  done
  log "Z2 phase done"
  $VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import os, re, sys
out = sys.argv[1]
def g(f, key):
    if not os.path.exists(f):
        return float("nan")
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", open(f).read())
    return float(m.group(1)) if m else float("nan")
def total(tag):
    return sum(g(f"{out}/bench_{tag}_p{i}.log", "Output token throughput") for i in (1, 2))
_miss = [t for t in [f"{p}_{x}" for p in ("R2bZw", "R2cZw")
                     for x in ("w0", "c108t", "c120t", "c132t", "c144t")]
         if total(t) != total(t)]
print("== Z2 warm ladders, ONE pool (seed 0), ONE boot each, SAME positions ==")
print(f"  completeness: {'ALL TEN PRESENT' if not _miss else 'INCOMPLETE -> ' + ', '.join(_miss)}")
print("  both ladders discard one warm-up point first, so position i means the same")
print("  thing on both sides -- which the cold R2bZ/R2cZ ladders did NOT.")
for pfx, name in (("R2bZw", "k=0"), ("R2cZw", "k=1")):
    vals = [(c, total(f"{pfx}_c{c}t")) for c in (108, 120, 132, 144)]
    ok = [(c, v) for c, v in vals if v == v]
    pc = max(ok, key=lambda kv: kv[1])[0] if ok else None
    for c, v in vals:
        mark = "   <- peak" if c == pc else ("   <- MISSING" if v != v else "")
        print(f"   {name} @{c}t ({c//2} lanes/rep): {v:.2f}{mark}")
    w0, c120 = total(f"{pfx}_w0"), total(f"{pfx}_c120t")
    if w0 == w0 and c120 == c120:
        print(f"   {name} ramp w0->c120t (same shape+pool, pos1->pos3): "
              f"{w0:.2f} -> {c120:.2f} = {(c120-w0)/w0*100:+.2f}%")
print("== crossover, point-for-point, both ladders warm ==")
for c in (108, 120, 132, 144):
    k0, k1 = total(f"R2bZw_c{c}t"), total(f"R2cZw_c{c}t")
    if k0 != k0 or k1 != k1:
        print(f"   @{c}t ({c//2} lanes/rep): k=0 {k0}  k=1 {k1}  -> n/a (incomplete)")
    else:
        print(f"   @{c}t ({c//2} lanes/rep): k=0 {k0:.2f}  k=1 {k1:.2f}  -> "
              f"{(k1-k0)/k0*100:+.2f}%  ({'k=0' if k0 > k1 else 'k=1'} wins)")
print("   NOTE: @132t is the rung that did not exist before this phase. The")
print("   crossover is now bracketed by MEASURED points, not by an unsampled band.")
print("   Do NOT report it as a single lane count: the peak of each arm sits at a")
print("   different abscissa, and these four rungs only bracket where it crosses.")
EOF
fi

# ---------- P20: FV — the whole four-factor chain on ONE seed ----------
# The decomposition currently mixes pools: 量化 compares B0(702) to B1(702) —
# fine — but 调度步长 compares B1(702) to B1b(701), and 布局 compares B1b(701) to
# R2b48(701/702). That is a COMPARABILITY defect, not a closure defect: the
# product of the four ratios is an identity that closes for any node values
# (each row is next/prev, so it telescopes to end/start). What the mixed pools
# actually cost is ~1.2pp of ambiguity in where the boundary between 调度步长
# and 布局 falls (bounded by SB's same-pool +5.66% vs the recorded +4.47%).
# Re-measuring the three TP2 rungs at seed 0 — the seed R2b48z (549.20) and
# R2cZ already use — makes the whole chain same-pool AND same-window, which is
# the strongest form the headline can take: every individual RATIO is then a
# same-pool number, and the split is pinned instead of bounded.
# Three boots because the flag needs its own server; each is one bench point.
if want FV; then
FORCE_SEED=0 tp2_phase FV0 "2,3" "" "48"
FORCE_SEED=0 tp2_phase FV1 "2,3" "--quantization fp8 --kv-cache-dtype fp8" "48"
FORCE_SEED=0 tp2_phase FV2 "2,3" "--quantization fp8 --kv-cache-dtype fp8 --max-num-batched-tokens 8192" "48"
$VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import os, re, sys
out = sys.argv[1]
def pick(f, key):
    if not os.path.exists(f):
        return float("nan")
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", open(f).read())
    return float(m.group(1)) if m else float("nan")
def val(tag, key):
    # SINGLE point (tp2_phase writes bench_<tag>_c48.log, one replica).
    return pick(f"{out}/bench_{tag}_c48.log", key)
def dual(pt):
    # DUAL point: the bench files are bench_<point>_p1.log / _p2.log, i.e. the
    # port index — NOT bench_<point>_p1_c48.log. Two separate bugs lived here and
    # both produced nan rather than an error, which is why this block is now
    # dry-run against real logs before any run depends on it:
    #   (1) an earlier draft built the tag by appending "_c48t" to tags that
    #       already carried it;
    #   (2) even after that fix, this function still handed the point tag to
    #       val(), which appends "_c48" — so 布局 and 投机 read nonexistent files
    #       and printed nan while the block "succeeded". Val/dual are now split.
    return sum(pick(f"{out}/bench_{pt}_p{i}.log", "Output token throughput")
               for i in (1, 2))
b0 = val("FV0", "Output token throughput")
b1 = val("FV1", "Output token throughput")
b2 = val("FV2", "Output token throughput")
layout = dual("R2b48z_c48t")
spec   = dual("R2cZ_c48t_a")
print("== FV four-factor chain, ALL on seed 0 (one pool) ==")
# Loud failure beats quiet nan: three of these five values come from files this
# block does not itself create, so an incomplete run must say so on its face
# rather than emit "nan" in a table someone reads a week later.
_missing = [n for n, v in (("FV0", b0), ("FV1", b1), ("FV2", b2),
                           ("R2b48z_c48t", layout), ("R2cZ_c48t_a", spec)) if v != v]
print(f"  completeness: {'ALL FIVE PRESENT' if not _missing else 'INCOMPLETE -> ' + ', '.join(_missing)}")
print(f"  量化   bf16->fp8 w+kv : {b0:.2f} -> {b1:.2f}  {(b1-b0)/b0*100:+.2f}%")
print(f"  调度   2048->8192     : {b1:.2f} -> {b2:.2f}  {(b2-b1)/b1*100:+.2f}%")
print(f"  布局   TP2->2 replica : {b2:.2f} -> {layout:.2f}  {(layout-b2)/b2*100:+.2f}%")
print(f"  投机   k=0 -> k=1     : {layout:.2f} -> {spec:.2f}  {(spec-layout)/layout*100:+.2f}%")
prod = (b1/b0)*(b2/b1)*(layout/b2)*(spec/layout)
# `prod` is printed for completeness ONLY. It is an identity (the four ratios
# telescope), so "it closed" is not evidence and must never be quoted as such.
# The evidence in this block is that each RATIO above is now a same-pool number.
print(f"  product {prod:.5f} vs end/start {spec/b0:.5f}  -> total {(spec-b0)/b0*100:+.2f}%")
print( "  NOTE: product==end/start is an IDENTITY (the ratios telescope). Do not")
print( "  cite the agreement; cite that every ratio here is same-pool/same-window.")
EOF
fi

# ---------- P9: R2c0 — same config as R2c, but seed 0 = sweep6's exact prompt set ----------
# Two independent questions, one measurement:
#   same-window A/B: R2c0(seed 0) vs R2c_c48t(seed 708/709) — differs ONLY in the
#     prompt set, so the gap is the pure prompt-set / MTP-acceptance effect.
#   cross-window A/B: R2c0(sweep7) vs 618.76(sweep6) — same prompt set, differs
#     ONLY in the window, so the gap is pure window drift.
# If the same-window gap comes out ~0, the acceptance explanation is FALSIFIED.
if want R2c0; then
boot_replica 8341 $RPA $OUT/server_R2c0_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2c0_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2c0 pair up ($RC1/$RC2)"
  startup_lines $OUT/server_R2c0_1.log $OUT/server_R2c0_2.log
  FORCE_SEED=0 bench_pair R2c0_c48t 24 48 8341 2 $OUT/server_R2c0_1.log $OUT/server_R2c0_2.log
  spec_metrics R2c0_c48t 8341 8342
else
  log "ABORT R2c0 boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2c0 phase done"
gpu_snap
fi

# ---------- P10: extension closing anchor B0'' ----------
if want B0pp && [ -s $OUT/bench_B0p_c48.log ]; then
tp2_phase B0pp "2,3" "" "48"
$VENV/python - "$OUT" "B0p" "B0pp" <<'EOF' >> $OUT/summary.txt
import re, sys
out, t1, t2 = sys.argv[1], sys.argv[2], sys.argv[3]
def val(tag, key):
    txt = open(f"{out}/bench_{tag}_c48.log").read()
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
a, b = val(t1, "Output token throughput"), val(t2, "Output token throughput")
d = abs(b - a) / a * 100
print(f"== WINDOW CHECK {t1} vs {t2} ==\n{t1}={a:.2f} {t2}={b:.2f} drift={d:.2f}% -> " + ("VALID" if d <= 3 else "VOID (>3%)"))
EOF
fi

# ---------- P11: R2c0b — pure run-to-run variance of the k=1 path (amendment 4) ----------
# Same window, same config, same boot params, same prompt set (seed 0) as R2c0.
# Identical inputs, so ANY gap is run-to-run variance — the one thing the k=1
# arms have never had measured. It also discriminates the two live hypotheses:
# low gap => R2c_c48t's 600.40 was a load artifact; high gap => k=1 is simply
# not reproducible to better than ~5% and the 投机 factor must be a RANGE.
if want R2c0b; then
boot_replica 8341 $RPA $OUT/server_R2c0b_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2c0b_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2c0b pair up ($RC1/$RC2)"
  startup_lines $OUT/server_R2c0b_1.log $OUT/server_R2c0b_2.log
  FORCE_SEED=0 bench_pair R2c0b_c48t 24 48 8341 2 $OUT/server_R2c0b_1.log $OUT/server_R2c0b_2.log
  spec_metrics R2c0b_c48t 8341 8342
else
  log "ABORT R2c0b boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2c0b phase done"
gpu_snap
fi

# ---------- P12: R2cH — the k=1 lane bracket at k=0's exact lane counts ----------
# R2b_c120t (60 lanes) = 682.96 and R2b_c144t (72 lanes) = 653.79 already exist.
# Running k=1 at the SAME lanes makes the crossover a direct measurement instead
# of a comparison across different lane counts, which is all we have now.
if want R2cH; then
boot_replica 8341 $RPA $OUT/server_R2cH_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2cH_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2cH pair up ($RC1/$RC2)"
  startup_lines $OUT/server_R2cH_1.log $OUT/server_R2cH_2.log
  bench_pair R2cH_c120t 60 120 8341 2 $OUT/server_R2cH_1.log $OUT/server_R2cH_2.log
  bench_pair R2cH_c144t 72 144 8341 2 $OUT/server_R2cH_1.log $OUT/server_R2cH_2.log
  spec_metrics R2cH_c144t 8341 8342
else
  log "ABORT R2cH boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2cH phase done"
gpu_snap
fi

# ---------- P13: closing anchor B0''' for the amendment-4 window ----------
if want B0ppp && [ -s $OUT/bench_B0pp_c48.log ]; then
tp2_phase B0ppp "2,3" "" "48"
$VENV/python - "$OUT" "B0pp" "B0ppp" <<'EOF' >> $OUT/summary.txt
import re, sys
out, t1, t2 = sys.argv[1], sys.argv[2], sys.argv[3]
def val(tag, key):
    txt = open(f"{out}/bench_{tag}_c48.log").read()
    m = re.search(re.escape(key) + r"[^\n:]*:\s+([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
a, b = val(t1, "Output token throughput"), val(t2, "Output token throughput")
d = abs(b - a) / a * 100
print(f"== WINDOW CHECK {t1} vs {t2} ==\n{t1}={a:.2f} {t2}={b:.2f} drift={d:.2f}% -> " + ("VALID" if d <= 3 else "VOID (>3%)"))
EOF
fi

# ---------- P14: R2cAB — the decisive seed A/B, ONE boot, back to back ----------
# Why this exists. R2c_c48t (13:19, seeds 702/703) = 600.40, but R2c0 and
# R2c0b (14:49 / 15:02, seed 0) = 652.98 / 651.60. Three suspects have already
# been eliminated by data already on disk:
#   * run-to-run variance — R2c0 vs R2c0b differ 0.21%;
#   * host load — R2c0b ran at loadavg 15.10, HIGHER than the 9.22 blamed for
#     the slow 600.40, and was still fast;
#   * "seed 70x is a slow prompt pool" — R2b48_c48t_p2 ran seed 702 at k=0 and
#     landed 274.03 against its seed-701 sibling's 275.52 (-0.54%), and
#     R2c_c48t's own two replicas used DIFFERENT pools (702, 703) and agreed
#     to 0.14%. A pool that is normal at k=0 and indistinguishable from a
#     neighbouring pool cannot be what moves k=1 by 9%.
# What survives is (a) a one-off on that single 13:19 k=1 run, or (b) an effect
# specific to the k=1 × prompt-pool interaction that seed 0 happens to win.
# Two bench_pair calls on ONE boot, ~40s apart, same server, same window, same
# client namespace: the ONLY thing that differs is the seed. No anchor is
# needed — the two points bracket each other, which is a tighter control than
# any window check. FORCE_SEED=702 goes first so that if the second bench dies
# the rarer datum is already on disk.
if want R2cAB; then
boot_replica 8341 $RPA $OUT/server_R2cAB_1.log "--speculative-config '$SPEC1'"; RC1=$LAST_PID
boot_replica 8342 $RPB $OUT/server_R2cAB_2.log "--speculative-config '$SPEC1'"; RC2=$LAST_PID
if wait_up 8341 && wait_up 8342; then
  log "R2cAB pair up ($RC1/$RC2)"
  startup_lines $OUT/server_R2cAB_1.log $OUT/server_R2cAB_2.log
  FORCE_SEED=702 bench_pair R2cAB70_c48t 24 48 8341 2 $OUT/server_R2cAB_1.log $OUT/server_R2cAB_2.log
  spec_metrics R2cAB70_c48t 8341 8342
  FORCE_SEED=0   bench_pair R2cAB00_c48t 24 48 8341 2 $OUT/server_R2cAB_1.log $OUT/server_R2cAB_2.log
  spec_metrics R2cAB00_c48t 8341 8342
  # The spec counters are cumulative from server boot and the second snapshot
  # therefore contains the first point's drafts too. Print the delta here so
  # the per-point step count is on disk next to the throughput, not left as an
  # arithmetic exercise for whoever reads the evidence later.
  $VENV/python - "$OUT" <<'EOF' >> $OUT/summary.txt
import re, sys
out = sys.argv[1]
def tot(tag, key):
    txt = open(f"{out}/specmetrics_{tag}_c48t.txt").read()
    return sum(float(m) for m in re.findall(rf"^{key}\{{[^}}]*\}}\s+([0-9.e+]+)", txt, re.M))
a = tot("R2cAB70", "vllm:spec_decode_num_drafts_total")
b = tot("R2cAB00", "vllm:spec_decode_num_drafts_total")
print(f"== R2cAB drafts (cumulative from boot) ==\nafter seed702={a:.0f} after seed0={b:.0f} -> seed0 own drafts={b-a:.0f}")
EOF
else
  log "ABORT R2cAB boot failed"
fi
kill_srv $RC1; kill_srv $RC2
log "R2cAB phase done"
gpu_snap
fi

# ---------- P6 (opt-in): dynamic-k question — k=1 vs k=2 at @16total ----------
if [ "${RUN_EXTRA:-0}" = "1" ]; then
  for K in 1 2; do
    SPEC='{\"method\": \"mtp\", \"num_speculative_tokens\": '$K'}'
    boot_replica 8341 $RPA $OUT/server_Rkd${K}_1.log "--speculative-config '$SPEC'"; P1=$LAST_PID
    boot_replica 8342 $RPB $OUT/server_Rkd${K}_2.log "--speculative-config '$SPEC'"; P2=$LAST_PID
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
