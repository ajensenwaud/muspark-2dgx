#!/usr/bin/env bash
# Quantise our own EXL3 pack from the MXFP4 source with the patched exllamav3 (muse-tf image) on the node that holds
# the source (SRC_NODE). Resumable: rerun the same command and the job continues from its work dir (exllamav3 -r).
#   setsid nohup scripts/quantise.sh 2.0 > logs/quantise-2.0.log 2>&1 </dev/null &
# Settings follow the small-model dry run: experts at <bits>, attention + latent projections at 4 (-hq), head 6, mul1 codebook,
# 250 x 2048 calibration. Holds $WORK_ROOT/gpu.lock for the whole run (taken atomically; refuses if another job holds it).
# Per-tensor recipe (sc_optimize.py): RECIPE=<recipe.yaml> PACK_OUT=<dir> scripts/quantise.sh <name> runs
# `--recipe` instead of -b/-hq (head 6, the same calibration); a resumed job takes everything from its args.json.
# Image: $IMAGE:${TAG:-latest} (per-tensor recipes: TAG=a78c6fbfa1, patches-exl3 0001-0005: --no_compile, fp32 router input).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
bits="${1:?usage: scripts/quantise.sh <bits>}"
out="${PACK_OUT:-$MODELS_ROOT/Muse-Spark-Open-exl3-${bits}bpw}"
work="${QUANT_WORK:-$WORK_ROOT/quant-work/$bits}"
lock="$WORK_ROOT/gpu.lock"
cache="$WORK_ROOT/scratch/home"

TAG="${TAG:-latest}"
docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1 || die "no $IMAGE:$TAG image here: scripts/build-image.sh"
[[ -f "$SRC_DIR/config.json" ]] || die "no source checkpoint at $SRC_DIR"
( set -o noclobber; echo "G3 quantise $bits $(date -Is)" > "$lock" ) 2>/dev/null || die "GPU locked: $(cat "$lock")"
trap 'rm -f "$lock"' EXIT

mkdir -p "$work" "$out" "$cache"
resume=(); [[ -f "$work/args.json" ]] && resume=(-r)
rmount=()
if [[ -n "${RECIPE:-}" ]]; then
  [[ -f "$RECIPE" ]] || die "no recipe at $RECIPE"
  rmount=(-v "$(readlink -f "$RECIPE"):/recipe.yaml:ro")
  if [[ -f "$work/args.json" ]]; then qargs=(-r); else qargs=(--recipe /recipe.yaml -hb 6 -cr 250 -cc 2048 -cb mul1); fi
else
  qargs=(-b "$bits" -hb 6 -hq -cr 250 -cc 2048 -cb mul1 "${resume[@]}")
fi
log "$bits bpw: converting ${resume[*]:+(resuming) }$SRC_DIR -> $out (work $work)"
t0=$(date +%s)
docker run --rm --name "muse-quant-$bits" --gpus all --ipc=host --shm-size 16g --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "$SRC_DIR:/src:ro" -v "$work:/work" -v "$out:/out" -v "$cache:/cache" "${rmount[@]}" \
  --entrypoint env "$IMAGE:$TAG" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /tools/convert.py \
  -i /src -w /work -o /out "${qargs[@]}"
[[ -f "$out/quantization_config.json" ]] || die "$bits bpw: the converter exited without writing a pack"
log "$bits bpw: done in $(( $(date +%s) - t0 )) s; $(du -sh "$out" | cut -f1)"
