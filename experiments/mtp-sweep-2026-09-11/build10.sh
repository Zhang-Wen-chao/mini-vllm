#!/bin/sh
# sweep10 runner = mtp_bench7.sh, mechanically, same recipe as bench8/bench9.
set -e
cd <script-dir>
head -n -2 mtp_bench7.sh > mtp_bench10.sh
sed -i '63s#.*#OUT=<out-dir-10>#' mtp_bench10.sh
sed -i '569s#.*#PHASES=${PHASES:-"CGS TP2F"}#' mtp_bench10.sh
sed -i '572s#.*#log "sweep10 start: cudagraph capture table + quantization gain on TP2 | PHASES=$PHASES"#' mtp_bench10.sh
cat <phase-block-10> >> mtp_bench10.sh
echo "=== syntax check ==="
bash -n mtp_bench10.sh && echo "syntax OK"
echo "=== changed-line count (expect 3) ==="
diff mtp_bench7.sh mtp_bench10.sh | grep -c '^<'
echo "=== the three changed lines ==="
diff mtp_bench7.sh mtp_bench10.sh | grep '^[<>]' | head -8
echo "=== tail ==="
tail -3 mtp_bench10.sh
