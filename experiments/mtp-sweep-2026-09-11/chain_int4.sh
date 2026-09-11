#!/bin/bash
# Orchestrates the baseline-beating run: (1) wait for mtp_bench4.sh (C4 arm)
# to finish, (2) wait for GPUs 0 and 3 free, (3) run int4 calibration, (4) if
# a checkpoint was produced, run the W4 arm benchmark.
# GPU0 permanently carries a ~2.6GB foreign daemon (pid 2961183 since Sep 8),
# so its free-memory threshold is 4000 MiB instead of GPU3's 2000.
# Logs to <chain-log>.

LOG=<chain-log>
echo "$(date +%H:%M:%S) chain started" >> $LOG

# 1) wait for C4 to finish (or abort -- then bail out loudly)
while ! grep -qE "ALL_DONE|ABORT" <out-dir-4>/run.log 2>/dev/null; do sleep 60; done
if grep -q ABORT <out-dir-4>/run.log; then
  echo "$(date +%H:%M:%S) C4 ABORTED - chain stops" >> $LOG
  exit 1
fi
echo "$(date +%H:%M:%S) C4 done" >> $LOG

free_gpu() {
  local idx="$1" thr="$2"
  local m u
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx")
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx")
  [ "$m" -lt "$thr" ] && [ "$u" -lt 20 ]
}

# 2) wait for GPUs 0 and 3 free
while ! { free_gpu 0 4000 && free_gpu 3 2000; }; do sleep 120; done
echo "$(date +%H:%M:%S) GPUs 0+3 free, starting calibration" >> $LOG

# 3) calibrate int4 (two-GPU sharded load of the bf16 source)
CUDA_VISIBLE_DEVICES=0,3 <lc-venv>/bin/python int4_calib.py > <calib-log> 2>&1
if ! grep -q CALIB_DONE <calib-log>; then
  echo "$(date +%H:%M:%S) CALIB FAILED - abort" >> $LOG
  tail -5 <calib-log> >> $LOG
  exit 1
fi
echo "$(date +%H:%M:%S) calibration done" >> $LOG
sleep 30

# 4) wait for GPU3 free again (calibration released it), then bench
while ! free_gpu 3 2000; do sleep 60; done
bash mtp_bench5.sh
echo "$(date +%H:%M:%S) CHAIN_DONE" >> $LOG
