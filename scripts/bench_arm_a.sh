#!/usr/bin/env bash
# Arm A measurements on the running llama.cpp serve (scripts/llama_serve.sh start), run on the head:
# health/props/clocks/memory snapshots, spark-bench decode (1 and 4 streams) + prefill depths that fit 32K,
# then the GSM8K quality anchor (greedy, reasoning effort low). Detached:
#   ssh -n <head> 'cd <recipe dir> && setsid nohup scripts/bench_arm_a.sh > logs/bench-arm-a.log 2>&1 </dev/null & disown'
# Knobs: SUITES (default decode,depth), DEPTHS, DECODE_TOKENS, GSM8K_N (0 skips the evals), EVAL_CONC, SB.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_hosts HEAD WORKER
API_PORT="${API_PORT:-8892}"
URL="http://127.0.0.1:$API_PORT"
OUT="${OUT:-results/A-llamacpp-$GGUF_QUANT}"
SB="${SB:-$HOME/muse/spark-bench}"
SUITES="${SUITES-decode,depth}"   # SUITES= (empty) skips spark-bench
DEPTHS="${DEPTHS:-4096,8192,16384,28672}"
DECODE_TOKENS="${DECODE_TOKENS:-512}"
GSM8K_N="${GSM8K_N:-200}"
EVAL_CONC="${EVAL_CONC:-4}"
MODEL="Muse-Spark-Open-$GGUF_QUANT"
mkdir -p "$OUT"

snap() {   # snap <tag>: health, props, slots, metrics, clocks and memory on both nodes
  curl -s "$URL/health" > "$OUT/health-$1.json"
  curl -s "$URL/props" > "$OUT/props-$1.json"
  curl -s "$URL/slots" > "$OUT/slots-$1.json"
  curl -s "$URL/metrics" > "$OUT/metrics-$1.txt"
  { echo "# $1 $(date -Is)"
    echo "## $HEAD"; nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu --format=csv
    echo "## $WORKER"; ssh "$WORKER" nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu --format=csv
  } >> "$OUT/clocks.txt"
  { echo "# $1 $(date -Is)"; echo "## $HEAD"; free -m; echo "## $WORKER"; ssh "$WORKER" free -m; } >> "$OUT/memory.txt"
}

log "snapshot before"
snap before
if [[ -n "$SUITES" ]]; then
  log "spark-bench $SUITES (decode tokens $DECODE_TOKENS, depths $DEPTHS)"
  /usr/bin/python3 "$SB/scripts/bench.py" --base-url "$URL" --model "$MODEL" --suites "$SUITES" \
    --concurrency 1,4 --decode-tokens "$DECODE_TOKENS" --depths "$DEPTHS" --depth-samples 2 \
    --label "2x DGX Spark, llama.cpp RPC (layer split), $GGUF_QUANT" --no-chart --out "$OUT/spark-bench"
  snap mid
fi
if [[ "$GSM8K_N" != 0 ]]; then
  log "GSM8K $GSM8K_N items, greedy, reasoning effort low, concurrency $EVAL_CONC"
  /usr/bin/python3 scripts/eval_tasks.py "$URL" "$MODEL" results/evals --tag "A-llamacpp-$GGUF_QUANT" \
    --tasks gsm8k --limit "$GSM8K_N" --concurrency "$EVAL_CONC" --reasoning-effort low
  cp "results/evals/A-llamacpp-$GGUF_QUANT-gsm8k.score.json" "$OUT/gsm8k.score.json"
fi
log "snapshot after"
snap after
log "done: $OUT"
