#!/usr/bin/env bash
# Zig-engine arm measurements on the running ./start.sh serve, run on the head, with arm A's settings so the arms
# compare (scripts/bench_arm_a.sh): health/models/clocks/memory snapshots on both nodes, spark-bench decode (1 and 4
# streams, 512 tokens) + prefill depths that fit 32K (2 samples), then the GSM8K quality anchor (200 items, greedy,
# reasoning effort low, 4 at a time) and, with MMLU_N > 0, MMLU-Pro on the same settings. Detached:
#   ssh -n <head> 'cd <recipe dir> && setsid nohup scripts/bench_zig.sh > logs/bench-zig.log 2>&1 </dev/null & disown'
# Output (OUT): results/B-zig-<pack>-plain without the drafter, results/C-zig-<pack>-dflash2 with it (read from the serve's
# logs/serve-config.txt). Knobs: SUITES (default decode,depth), DEPTHS, DECODE_TOKENS, GSM8K_N (0 skips), MMLU_N
# (default 0), EVAL_CONC, EVAL_MAX_TOKENS (overrides the per-task budget), TAG (the eval files' tag), SB, OUT.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_hosts HEAD WORKER
URL="http://127.0.0.1:$PORT"
arm=plain
grep -q '^drafter /' logs/serve-config.txt 2>/dev/null && arm=dflash2
case $arm in plain) letter=B ;; *) letter=C ;; esac
OUT="${OUT:-results/$letter-zig-$PACK_NAME-$arm}"
SB="${SB:-$HOME/muse/spark-bench}"
SUITES="${SUITES-decode,depth}"   # SUITES= (empty) skips spark-bench
DEPTHS="${DEPTHS:-4096,8192,16384,28672}"
DECODE_TOKENS="${DECODE_TOKENS:-512}"
GSM8K_N="${GSM8K_N:-200}"
MMLU_N="${MMLU_N:-0}"
EVAL_CONC="${EVAL_CONC:-4}"
TAG="${TAG:-$letter-zig-$PACK_NAME-$arm}"
mkdir -p "$OUT"
curl -fsS -m 5 "$URL/health" | grep -q '"ok"' || die "no serve on $URL: ./start.sh first"

snap() {   # snap <tag>: health, models and the serve config, clocks and memory on both nodes
  curl -s "$URL/health" > "$OUT/health-$1.json"
  curl -s "$URL/v1/models" > "$OUT/models-$1.json"
  { echo "# $1 $(date -Is)"
    echo "## $HEAD"; nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu --format=csv
    echo "## $WORKER"; ssh "$WORKER" nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu --format=csv </dev/null
  } >> "$OUT/clocks.txt"
  { echo "# $1 $(date -Is)"; echo "## $HEAD"; free -m; echo "## $WORKER"; ssh "$WORKER" free -m </dev/null; } >> "$OUT/memory.txt"
}

cp logs/serve-config.txt "$OUT/serve-config.txt" 2>/dev/null || true
git -C "$ZIG_DIR" rev-parse HEAD > "$OUT/engine-commit.txt" 2>/dev/null || echo "unknown" > "$OUT/engine-commit.txt"
echo "tree $ZIG_TREE (scripts/config.sh)" >> "$OUT/engine-commit.txt"
log "arm $TAG: snapshot before"
snap before
if [[ -n "$SUITES" ]]; then
  log "spark-bench $SUITES (decode tokens $DECODE_TOKENS, depths $DEPTHS)"
  /usr/bin/python3 "$SB/scripts/bench.py" --base-url "$URL" --model "$SERVED_NAME" --suites "$SUITES" \
    --concurrency 1,4 --decode-tokens "$DECODE_TOKENS" --depths "$DEPTHS" --depth-samples 2 \
    --label "2x DGX Spark, TensorFold Zig TP=2, $PACK_NAME, $arm" --no-chart --out "$OUT/spark-bench"
  snap mid
fi
for spec in "gsm8k:$GSM8K_N" "mmlu_pro:$MMLU_N"; do
  task=${spec%%:*}; n=${spec#*:}
  [[ "$n" == 0 ]] && continue
  [[ -f "results/evals/items-$task.json" ]] || /usr/bin/python3 scripts/eval_data.py results/evals --n 200 --tasks "$task"
  log "$task $n items, greedy, reasoning effort low, concurrency $EVAL_CONC"
  /usr/bin/python3 scripts/eval_tasks.py "$URL" "$SERVED_NAME" results/evals --tag "$TAG" \
    --tasks "$task" --limit "$n" --concurrency "$EVAL_CONC" --reasoning-effort low ${EVAL_MAX_TOKENS:+--max-tokens $EVAL_MAX_TOKENS} ${EVAL_EXTRA:-}
  cp "results/evals/$TAG-$task.score.json" "$OUT/$task.score.json"
done
log "snapshot after"
snap after
docker logs "$ZIG_CONTAINER" > "$OUT/server-rank0.log" 2>&1 || true
log "done: $OUT"
