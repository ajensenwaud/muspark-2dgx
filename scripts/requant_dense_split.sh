#!/usr/bin/env bash
# Dense requantization split over both Sparks: the dense requantization of scripts/requant_dense.sh with layers
# 0..L-1 on SRC_NODE (node A, reached as A_HOST: has the MXFP4 source) and layers L..61 on the head (node B, this
# node), then one pack. Run on the head, detached:
#   TAG=<tag> setsid nohup scripts/requant_dense_split.sh 3.0 32 > logs/requant-dense-split-3.0.log 2>&1 </dev/null &
# Node B starts at layer L from SEED_A, a checkpoint on node A whose state is the calibration rows at the input of
# layer L. Default: P2.0's own conversion checkpoint $WORK_ROOT/quant-work/2.0/ckpt (next module 34 = layer 32: the rows
# through P2.0's quantized layers 0-31; keep it until this has run). Without a matching checkpoint, ADVANCE_BASE=1
# makes node B compute it (layers 0..L-1 with the base pack's tensors, ~1 min a layer). Node B's dense weights are
# fetched from node A by tools/dense/fetch_dense.py (range server, ~0.57 GB a layer). Caveat as in the split conversion: node B's layers
# are calibrated on states from P2.0's layers 0..L-1 (attention/latent at 4.0), not from the new 3.0 ones.
# Steps: range server on node A -> A launched there (detached) -> fetch dense weights + seed here -> B here ->
# wait for A -> B's qtensors to node A -> dense_pack.py assemble + check + verify_exl3_pack.py on node A (the pack is
# written there: 212 GB). Deletes nothing.
# Knobs: TAG (required), OVERRIDE, BATCH, A_HOST (default WORKER), A_IP (default WORKER_IP), SEED_A, ADVANCE_BASE, PACK_OUT, CHECK_SAMPLE.
# Rerun the same command to resume: each half resumes its own job, finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_hosts HEAD SRC_NODE
bits="${1:?usage: requant_dense_split.sh <dense bits> <L>}"; L="${2:?usage: requant_dense_split.sh <dense bits> <L>}"
TAG="${TAG:?set TAG to the muse-tf image tag with patches-exl3 0005}"
A="${A_HOST:-$WORKER}"                                  # node A (SRC_NODE) over CX7
A_IP="${A_IP:-$WORKER_IP}"
[[ -n "$A" && -n "$A_IP" ]] || die "node A unknown: set WORKER (or A_HOST and A_IP) in scripts/local.sh, see scripts/local.sh.example"
BASE="${BASE:-$MODELS_ROOT/Muse-Spark-Open-exl3-2.0bpw}"
out="${PACK_OUT:-$MODELS_ROOT/Muse-Spark-Open-exl3-2.0e-${bits}d}"
wa="$WORK_ROOT/quant-work/dense-$bits-A"                 # on node A (job in $wa/work)
wb="$WORK_ROOT/quant-work/dense-$bits-B"                 # here
SEED_A="${SEED_A:-$WORK_ROOT/quant-work/2.0/ckpt}"
IDS_A="${IDS_A:-$WORK_ROOT/quant-work/2.0/ckpt/original_input_ids.safetensors}"
PORT="${RANGE_PORT:-8899}"
POLL="${POLL:-120}"
NUM_LAYERS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["text_config"]["num_hidden_layers"])' "$BASE/config.json")
EXPECT=$(( NUM_LAYERS * 775 + 1 ))
envs="TAG=$TAG BATCH=${BATCH:-10}${OVERRIDE:+ OVERRIDE='$OVERRIDE'}"
ra() { ssh -n -o ConnectTimeout=20 "$A" "$@"; }

# Checks on both nodes: image, base pack, free GPU, recipe files
docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1 || die "no $IMAGE:$TAG here"
ra "docker image inspect $IMAGE:$TAG >/dev/null 2>&1" || die "no $IMAGE:$TAG on $A"
ra "test -x ~/muse/recipe/scripts/requant_dense.sh && test -f ~/muse/recipe/tools/exl3/dense_pack.py" \
  || die "$A lacks the recipe scripts (rsync scripts/ tools/ to ~/muse/recipe there)"
for f in "$BASE/config.json" "$BASE/quantization_config.json"; do [[ -f "$f" ]] || die "no $f here"; ra "test -f $f" || die "no $f on $A"; done
[[ ! -e "$WORK_ROOT/gpu.lock" ]] || die "GPU locked here: $(cat "$WORK_ROOT/gpu.lock")"
ra "test ! -e $WORK_ROOT/gpu.lock" || ra "test -f $wa/work/requant-done.json" \
  || ra "docker ps --format '{{.Names}}' | grep -qx muse-requant-dense-$bits" || die "GPU locked on $A: $(ra "cat $WORK_ROOT/gpu.lock")"
free_b=$(df -B1G --output=avail "$WORK_ROOT" | tail -1); free_a=$(ra "df -B1G --output=avail $MODELS_ROOT | tail -1")
(( free_b >= 45 )) || die "$HEAD has $free_b GB free, node B needs ~40 (seed 16.8 + dense weights 17 + qtensors 3)"
(( free_a >= 260 )) || die "$A has $free_a GB free, needs ~255 (pack 212 + node A's work 37)"

# Seed for node B: a checkpoint on node A at layer L, else ADVANCE_BASE
seed_idx=$(ra "python3 -c 'import json; print(json.load(open(\"$SEED_A/job.json\"))[\"next_module_idx\"])'" 2>/dev/null || echo none)
if [[ "$seed_idx" == "$((2 + L))" ]]; then
  use_seed=1; log "seed for layer $L: $A:$SEED_A (next module $seed_idx)"
elif [[ "${ADVANCE_BASE:-0}" == 1 ]]; then
  use_seed=0; log "no seed at layer $L on $A ($SEED_A: $seed_idx); node B advances layers 0..$((L - 1)) with the base pack"
else
  die "$A:$SEED_A is at module $seed_idx, not $((2 + L)) (layer $L): pick L=$(( seed_idx - 2 )) or set ADVANCE_BASE=1"
fi

# 1. Node A (detached; its own GPU lock), layers 0..L-1, from the calibration data (rows checked against P2.0's)
if ! ra "test -f $wa/work/requant-done.json"; then
  ra "docker ps --format '{{.Names}}' | grep -qx muse-requant-dense-$bits" && log "node A already running on $A" || {
    ra "cd ~/muse/recipe && mkdir -p ~/muse/logs && setsid nohup env $envs WORK=$wa/work LAYERS=0-$((L - 1)) ASSEMBLE=0 \
        IDS_REF=$IDS_A scripts/requant_dense.sh $bits > ~/muse/logs/requant-dense-$bits-A.log 2>&1 </dev/null & disown"
    log "node A launched on $A (log ~/muse/logs/requant-dense-$bits-A.log there)"
    sleep 90
    ra "docker ps --format '{{.Names}}' | grep -qx muse-requant-dense-$bits" \
      || die "node A did not start on $A: $(ra "tail -5 ~/muse/logs/requant-dense-$bits-A.log")"; }
fi

# 2. Node B inputs: dense weights of layers L..61 (range server on node A), the seed
if [[ ! -f "$wb/requant-done.json" ]]; then
  mkdir -p "$wb/src" "$wb/meta"
  if (( $(ls "$wb/src" | grep -c '^dense-.*\.safetensors$' || true) < NUM_LAYERS - L )); then
    ra "docker ps --format '{{.Names}}' | grep -qx muse-range-dense" || ra "docker run -d --rm --name muse-range-dense --network host \
      --user \$(id -u):\$(id -g) -v $SRC_DIR:/src:ro -v ~/muse/recipe/tools:/rtools:ro --entrypoint python3 $IMAGE:$TAG \
      /rtools/g3/range_server.py --root /src --bind $A_IP --port $PORT >/dev/null"
    rsync -a "$A:$SRC_DIR/model.safetensors.index.json" "$wb/meta/"
    sleep 2
    python3 tools/dense/fetch_dense.py --url "http://$A_IP:$PORT" --index "$wb/meta/model.safetensors.index.json" \
      --out "$wb/src" --layers "$L-$((NUM_LAYERS - 1))"
    ra "docker stop -t 2 muse-range-dense >/dev/null" || true
  fi
  if [[ "$use_seed" == 1 && ! -f "$wb/seed/job.json" ]]; then
    rsync -a --partial "$A:$SEED_A/" "$wb/seed.part/" && mv "$wb/seed.part" "$wb/seed"
  fi
  # 3. Node B here: no intermediate checkpoints (disk: seed + 2 state copies would not fit beside the dense source
  # here); an interrupted run restarts from the seed
  if [[ "$use_seed" == 1 ]]; then
    TAG=$TAG BATCH=${BATCH:-10} OVERRIDE="${OVERRIDE:-}" WORK="$wb/work" SRC="$wb/src" LAYERS="$L-$((NUM_LAYERS - 1))" \
      SEED="$wb/seed" CPI=-1 ASSEMBLE=0 BASE="$BASE" scripts/requant_dense.sh "$bits"
  else
    TAG=$TAG BATCH=${BATCH:-10} OVERRIDE="${OVERRIDE:-}" WORK="$wb/work" SRC="$wb/src" LAYERS="$L-$((NUM_LAYERS - 1))" \
      ADVANCE_BASE=1 CPI=-1 ASSEMBLE=0 BASE="$BASE" scripts/requant_dense.sh "$bits"
  fi
  cp "$wb/work/requant-done.json" "$wb/requant-done.json"
fi
log "node B done: $(cat "$wb/requant-done.json" | tr -d '\n ')"

# 4. Wait for node A
while ! ra "test -f $wa/work/requant-done.json"; do
  if ! ra "docker ps --format '{{.Names}}' | grep -qx muse-requant-dense-$bits"; then
    sleep 30
    ra "test -f $wa/work/requant-done.json" && break
    die "node A on $A is not running and has not finished: ssh $A tail ~/muse/logs/requant-dense-$bits-A.log (resume: rerun this script)"
  fi
  sleep "$POLL"
done
log "node A done: $(ra "cat $wa/work/requant-done.json" | tr -d '\n ')"

# 5. Merge on node A: B's qtensors there, assemble, check, verify
rsync -a --mkpath "$wb/work/qtensors/" "$A:$wa-from-$HEAD/qtensors/"
rsync -a "$wb/work/report.jsonl" "$A:$wa-from-$HEAD/"
ra "cd ~/muse/recipe && test -z \"\$(ls -A $out 2>/dev/null)\"" || die "$A:$out is not empty"
ra "cd ~/muse/recipe && python3 tools/exl3/dense_pack.py assemble --base $BASE --qtensors $wa/work/qtensors $wa-from-$HEAD/qtensors \
    --out $out --note 'dense linears requantized at $bits${OVERRIDE:+ (override $OVERRIDE)}, split at layer $L ($SRC_NODE 0-$((L - 1)), $HEAD $L-$((NUM_LAYERS - 1)))' \
    && python3 tools/exl3/dense_pack.py check --base $BASE --pack $out --qtensors $wa/work/qtensors $wa-from-$HEAD/qtensors --sample ${CHECK_SAMPLE:-0} \
    && python3 tools/exl3/verify_exl3_pack.py $out $EXPECT" || die "assemble/check/verify failed on $A"
mkdir -p "results/G6/dense-$bits"
cat "$wb/work/report.jsonl" > "results/G6/dense-$bits/report-$HEAD.jsonl"
ra "cat $wa/work/report.jsonl" > "results/G6/dense-$bits/report-$SRC_NODE.jsonl"
log "pack $A:$out verified ($EXPECT EXL3 matrices). Next: room on $HEAD (old work dirs, by hand), then
  rsync -a $A:$out/ $out/ && python3 tools/exl3/verify_exl3_pack.py $out $EXPECT"
