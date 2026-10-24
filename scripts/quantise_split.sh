#!/usr/bin/env bash
# Split conversion, second node (the head): quantise layers L..61, the final norm and the head of the same pack that
# the first node's scripts/quantise.sh job quantises from layer 0, then stop without compiling (--no_compile).
# The job is the one tools/g3/state_gen.py wrote into $SPLIT/work: the first node's own args.json and calibration
# rows, the calibration state at the input of layer L made by the UNQUANTISED source layers 0..L-1, next module L.
# The converter resumes it with -r. Source: the per-layer files of tools/g3/fetch_layers.py (layers L..61 + head)
# beside the source's config/tokenizer files. Merge and compile: tools/g3/merge_split.sh on the first node.
#   setsid nohup scripts/quantise_split.sh 2.0 32 > logs/quantise-split-2.0.log 2>&1 </dev/null &
# Holds $WORK_ROOT/gpu.lock as "G3 quantise split L..61 <ts> SHARED-OK" (small correctness jobs may run beside it while
# MemAvailable stays >= 45 GiB; never speed measurements). Resumable: rerun the same command.
# Per-tensor recipe jobs: RECIPE_JOB=1 resumes with only -r --no_compile (bits, recipe and calibration from the job's
# args.json, which is the first node's); <bits> is then just the job's name (e.g. 1.8s).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
bits="${1:?usage: scripts/quantise_split.sh <bits> <L>}"; L="${2:?usage: scripts/quantise_split.sh <bits> <L>}"
TAG="${TAG:?set TAG to the muse-tf image tag with patches-exl3 0003+ (--no_compile, layer-61 fold)}"
split="${SPLIT:-$WORK_ROOT/quant-work/split-$bits}"
work="$split/work"
src="${SPLIT_SRC:-$MODELS_ROOT/Muse-Spark-Open-MXFP4-split}"
out="$split/out-unused"                 # the converter wants an output dir; --no_compile never writes it
lock="$WORK_ROOT/gpu.lock"
cache="$WORK_ROOT/scratch/home"

docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1 || die "no $IMAGE:$TAG image here"
docker run --rm --entrypoint python3 "$IMAGE:$TAG" -c 'import exllamav3.conversion.convert_model as c; assert "no_compile" in c.parser.format_help()' \
  || die "$IMAGE:$TAG has no --no_compile (patches-exl3 0003)"
[[ -f "$work/args.json" && -f "$work/ckpt/job.json" ]] || die "no split job in $work (run tools/g3/run_state_gen.sh $L first)"
idx=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_module_idx"])' "$work/ckpt/job.json")
(( idx >= 2 + L )) || die "job in $work is at module $idx, expected >= $((2 + L)) (layer $L)"
for l in $(seq "$L" 61); do [[ -f "$src/layer-$(printf %02d "$l").safetensors" ]] || die "missing $src/layer-$(printf %02d "$l").safetensors"; done
[[ -f "$src/head.safetensors" ]] || die "missing $src/head.safetensors"
msg="G3 quantise split $L..61 $(date -Is) SHARED-OK"
if [[ -f "$lock" ]] && grep -q '^G3 quantise split' "$lock"; then
  echo "$msg" > "$lock.tmp" && mv -f "$lock.tmp" "$lock"      # our own state-generation lock: hand over
else
  ( set -o noclobber; echo "$msg" > "$lock" ) 2>/dev/null || die "GPU locked: $(cat "$lock")"
fi
trap 'rm -f "$lock"' EXIT

if [[ "${RECIPE_JOB:-0}" == 1 ]]; then qargs=(); else qargs=(-b "$bits" -hb 6 -hq -cr 250 -cc 2048 -cb mul1); fi
mkdir -p "$out" "$cache"
log "$bits bpw split: layers $L..61 + norm + head from module $idx, $src -> $work/qtensors (image $IMAGE:$TAG)"
t0=$(date +%s)
docker run --rm --name "muse-quant-split-$bits" --gpus all --ipc=host --shm-size 16g --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "$src:/src:ro" -v "$work:/work" -v "$out:/out" -v "$cache:/cache" \
  --entrypoint env "$IMAGE:$TAG" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /tools/convert.py \
  -i /src -w /work -o /out "${qargs[@]}" -r --no_compile
n=$(ls "$work/qtensors" | grep -cE '^model\.language_model\.layers\.[0-9]+\.safetensors$' || true)
[[ -f "$work/qtensors/lm_head.safetensors" && -f "$work/qtensors/model.language_model.norm.safetensors" ]] \
  || die "the converter exited without the head/norm qtensors"
(( n == 62 - L )) || die "$n layer qtensors files in $work/qtensors, expected $((62 - L))"
log "$bits bpw split: layers $L..61 + norm + head done in $(( $(date +%s) - t0 )) s; $(du -sh "$work/qtensors" | cut -f1)"
