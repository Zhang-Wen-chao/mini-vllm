#!/bin/sh
# sweep9 runner = mtp_bench7.sh, mechanically, exactly the way mtp_bench8.sh was made:
# drop the old tail, retarget three lines, append the new phase block. Nothing else
# is touched, and the diff against bench7 is printed as proof.
set -e
cd <script-dir>
head -n -2 mtp_bench7.sh > mtp_bench9.sh

# 63: OUT -> mtp9
sed -i '63s#.*#OUT=<out-dir-9>#' mtp_bench9.sh
# 569: PHASES default -> the four sweep9 phases
sed -i '569s#.*#PHASES=${PHASES:-"PAIR EG GMU MB"}#' mtp_bench9.sh
# 572: banner
sed -i '572s#.*#log "sweep9 start: card pair / enforce-eager / gmu / mamba block size | PHASES=$PHASES"#' mtp_bench9.sh

cat <phase-block-9> >> mtp_bench9.sh

echo "=== syntax check ==="
bash -n mtp_bench9.sh && echo "syntax OK"
echo "=== line count: bench7 -> bench9 ==="
wc -l mtp_bench7.sh mtp_bench9.sh
echo "=== diff vs bench7: expect exactly 3 changed lines + one appended block ==="
diff mtp_bench7.sh mtp_bench9.sh | head -40
echo "=== changed-line count (should be 3) ==="
diff mtp_bench7.sh mtp_bench9.sh | grep -c '^<'
echo "=== last 4 lines ==="
tail -4 mtp_bench9.sh
