#!/usr/bin/env bash
# Requantize only the dense linears of an EXL3 pack (attention q/k/v/o/gate and the latent
# pre/post expert projections; 7 a layer) at <bits>, keeping every routed expert, the embedding, norms, router,
# residual gates, expert score bias and head of the base pack byte for byte. One node; ~3.5 h for Muse Spark's 62
# layers on one GB10 (projected) against ~64 h for a full conversion.
#   TAG=<tag> setsid nohup scripts/requant_dense.sh 3.0 > logs/requant-dense-3.0.log 2>&1 </dev/null &
# Steps: exllamav3's requant_dense (patches-exl3 0005) in the recipe image -> <work>/qtensors (dense tensors only)
# -> tools/exl3/dense_pack.py assemble (base pack + new dense tensors -> new pack) -> dense_pack.py check (every kept
# tensor byte-identical to the base pack) -> tools/exl3/verify_exl3_pack.py (48,051 EXL3 matrices).
# Knobs (env): TAG (muse-tf image tag, required), BASE (base pack, default P2.0), SRC (unquantized source of the
#   dense weights: the MXFP4 source dir, or a dir of tools/dense/fetch_dense.py files), OVERRIDE (e.g.
#   "*.k_proj=4,*.v_proj=4"), PACK_OUT (default $MODELS_ROOT/Muse-Spark-Open-exl3-2.0e-<bits>d), WORK
#   (default $WORK_ROOT/quant-work/dense-<bits>), LAYERS (a-b; default all), SEED (checkpoint dir to start from),
#   ADVANCE_BASE=1, IDS_REF (calibration-row check), BATCH (10), CPI (checkpoint interval s, 600; <0 none),
#   ASSEMBLE (1; 0 = stop after the qtensors, for split runs), CHECK_SAMPLE (0 = compare every kept tensor),
#   GPU_LOCK_HELD=1 (a caller holds $WORK_ROOT/gpu.lock).
# Resumable: rerun the same command (the job resumes with -r from its last checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
bits="${1:?usage: scripts/requant_dense.sh <dense bits>}"
TAG="${TAG:?set TAG to the muse-tf image tag with patches-exl3 0005 (requant_dense)}"
BASE="${BASE:-$MODELS_ROOT/Muse-Spark-Open-exl3-2.0bpw}"
SRC="${SRC:-$SRC_DIR}"
out="${PACK_OUT:-$MODELS_ROOT/Muse-Spark-Open-exl3-2.0e-${bits}d}"
work="${WORK:-$WORK_ROOT/quant-work/dense-$bits}"
lock="$WORK_ROOT/gpu.lock"
cache="$WORK_ROOT/scratch/home"
P=model.language_model

docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1 || die "no $IMAGE:$TAG image here"
docker run --rm --entrypoint python3 "$IMAGE:$TAG" -c 'import importlib.util as u; assert u.find_spec("exllamav3.conversion.requant_dense")' \
  2>/dev/null || die "$IMAGE:$TAG has no exllamav3.conversion.requant_dense (patches-exl3 0005)"
[[ -f "$BASE/config.json" && -f "$BASE/quantization_config.json" ]] || die "no base pack at $BASE"
[[ -d "$SRC" ]] || die "no dense source at $SRC"
read -r NUM_LAYERS NUM_EXPERTS < <(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["text_config"]
print(c["num_hidden_layers"], c["num_local_experts"])' "$BASE/config.json")
EXPECT=$(( NUM_LAYERS * (7 + 3 * NUM_EXPERTS) + 1 ))      # Muse Spark Open: 62 x 775 + 1 = 48,051

if [[ "${GPU_LOCK_HELD:-0}" != 1 ]]; then
  ( set -o noclobber; echo "G6 requant dense $bits ${LAYERS:-all} $(date -Is)" > "$lock" ) 2>/dev/null || die "GPU locked: $(cat "$lock")"
  trap 'rm -f "$lock"' EXIT
fi
mkdir -p "$work" "$cache"

if [[ ! -f "$work/requant-done.json" ]]; then
  if [[ -f "$work/args.json" ]]; then
    targs=(-r)
    log "dense $bits: resuming $work"
  else
    targs=(-b /base -s /src --bits "$bits")
    [[ -n "${OVERRIDE:-}" ]] && targs+=(--override "$OVERRIDE")
    [[ -n "${LAYERS:-}" ]] && targs+=(--layers "$LAYERS")
    [[ -n "${SEED:-}" ]] && targs+=(--seed /seed)
    [[ "${ADVANCE_BASE:-0}" == 1 ]] && targs+=(--advance_base)
    [[ -n "${IDS_REF:-}" ]] && targs+=(--ids_ref /ids_ref.safetensors)
    log "dense $bits: $BASE + dense weights from $SRC -> $work (layers ${LAYERS:-all}, override '${OVERRIDE:-}', image $IMAGE:$TAG)"
  fi
  mounts=(-v "$BASE:/base:ro" -v "$SRC:/src:ro" -v "$work:/work" -v "$cache:/cache")
  [[ -n "${SEED:-}" ]] && mounts+=(-v "$SEED:/seed:ro")
  [[ -n "${IDS_REF:-}" ]] && mounts+=(-v "$IDS_REF:/ids_ref.safetensors:ro")
  t0=$(date +%s)
  docker run --rm --name "muse-requant-dense-$bits" --gpus all --ipc=host --shm-size 16g --user "$(id -u):$(id -g)" \
    -e HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "${mounts[@]}" \
    --entrypoint env "$IMAGE:$TAG" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 -m exllamav3.conversion.requant_dense \
    -w /work "${targs[@]}" --batch "${BATCH:-10}" -cpi "${CPI:-600}"
  [[ -f "$work/requant-done.json" ]] || die "dense $bits: the requantization exited without requant-done.json"
  log "dense $bits: requantized in $(( $(date +%s) - t0 )) s ($(ls "$work/qtensors" | wc -l) qtensors files, $(du -sh "$work/qtensors" | cut -f1))"
fi

[[ "${ASSEMBLE:-1}" == 1 ]] || { log "dense $bits: ASSEMBLE=0, stopping after the qtensors"; exit 0; }
n=$(ls "$work/qtensors" | grep -cE "^$P\.layers\.[0-9]+\.safetensors$" || true)
(( n == NUM_LAYERS )) || die "$n layer qtensors files in $work/qtensors, expected $NUM_LAYERS (split run: assemble with scripts/requant_dense_split.sh)"
[[ -z "$(ls -A "$out" 2>/dev/null)" ]] || die "$out is not empty"
log "dense $bits: assembling $out"
python3 tools/exl3/dense_pack.py assemble --base "$BASE" --qtensors "$work/qtensors" --out "$out" \
  --note "dense linears requantized at $bits${OVERRIDE:+ (override $OVERRIDE)}, experts and all other tensors from $(basename "$BASE")"
python3 tools/exl3/dense_pack.py check --base "$BASE" --pack "$out" --qtensors "$work/qtensors" --sample "${CHECK_SAMPLE:-0}"
python3 tools/exl3/verify_exl3_pack.py "$out" "$EXPECT"
mkdir -p "results/G6/dense-$bits" && cp "$work/report.jsonl" "results/G6/dense-$bits/report.jsonl"
log "dense $bits: pack $out: $(du -sh "$out" | cut -f1), verified ($EXPECT EXL3 matrices)"
