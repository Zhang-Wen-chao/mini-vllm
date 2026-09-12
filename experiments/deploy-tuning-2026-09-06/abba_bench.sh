#!/bin/bash
# ABBA 交替配对 x5: A=单卡fp8权重+fp8KV(8321)  B=TP2 bf16(8322)
# 口径: 1024进/256出定长, temperature 0, 96条, 客户端并发48
mkdir -p <out-dir-abba>
BENCH=<venv>/bin/vllm
COMMON="--backend openai-chat --endpoint /v1/chat/completions --tokenizer <model-dir> --dataset-name random --random-input-len 1024 --random-output-len 256 --temperature 0 --num-prompts 96 --max-concurrency 48 --host 127.0.0.1"
rm -f <out-dir-abba>/summary.txt
for i in 1 2 3 4 5; do
  if [ $((i % 2)) -eq 1 ]; then ORDER="A B"; else ORDER="B A"; fi
  for M in $ORDER; do
    if [ "$M" = "A" ]; then PORT=8321; NAME=qwen38-27b; else PORT=8322; NAME=qwen38-27b-tp2; fi
    HF_HUB_OFFLINE=1 $BENCH bench serve --model $NAME --port $PORT $COMMON > <out-dir-abba>/round${i}_${M}.log 2>&1
    grep -E "Benchmark duration|Output token throughput|Total token throughput|Mean TTFT|P99 TTFT|Mean TPOT" <out-dir-abba>/round${i}_${M}.log | sed "s/^/r$i $M /" >> <out-dir-abba>/summary.txt
  done
done
echo ALL_DONE >> <out-dir-abba>/summary.txt
