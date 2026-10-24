#!/usr/bin/env bash
# Quantise RadixArk's DFlash2 drafter (bf16, 9.8 GB) to an EXL3 pack for the Zig engine, on the node that holds it
# (SRC_NODE or the head), when its GPU is free. exllamav3 builds the drafter as DFlash2DraftModel (checked on this config: 47 linears,
# uncalibrated quantisation, no tokenizer or calibration data): fc, q/k/v/o, gate/up/down and both kernel projections
# of each layer become EXL3 (mul1, the codebook the Zig kernels read); the selector's hidden projection stays fp16;
# norms, conv base kernels and the selector codebooks are copied as they are. 4.0 bpw -> ~2.5 GB.
#   setsid nohup scripts/quantise_drafter.sh 4.0 > logs/quantise-drafter.log 2>&1 </dev/null &
# Holds $WORK_ROOT/gpu.lock (taken atomically, refused when held) unless GPU_LOCK_HELD=1 (a caller holds it).
# Both nodes need the pack at the same path: scripts/prepare.sh quantises it on the head and copies it to the worker.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
bits="${1:-$DRAFTER_BITS}"
out="${DRAFTER_OUT:-$MODELS_ROOT/Muse-Spark-Open-DFlash2-exl3-${bits}bpw}"
work="${QUANT_WORK:-$WORK_ROOT/quant-work/drafter-$bits}"
lock="$WORK_ROOT/gpu.lock"
cache="$WORK_ROOT/scratch/home"
img="${QUANT_IMAGE:-$IMAGE:latest}"

docker image inspect "$img" >/dev/null 2>&1 || die "no $img image here: scripts/build-image.sh"
[[ -f "$DRAFTER_SRC_DIR/config.json" && -f "$DRAFTER_SRC_DIR/model.safetensors" ]] || die "no drafter at $DRAFTER_SRC_DIR"
[[ -f "$out/quantization_config.json" ]] && { log "already converted: $out"; exit 0; }
if [[ "${GPU_LOCK_HELD:-0}" == 1 && "$(cat "$lock" 2>/dev/null)" == "G6 queue"* ]]; then
  log "running under the G6 queue's lock"
else
  ( set -o noclobber; echo "G6 quantise drafter $bits $(date -Is)" > "$lock" ) 2>/dev/null || die "GPU locked: $(cat "$lock")"
  trap 'rm -f "$lock"' EXIT
fi
docker ps --format '{{.Names}}' | grep -q '^muse-quant' && die "a muse-quant container is running"
mkdir -p "$work" "$out" "$cache"
resume=(); [[ -f "$work/args.json" ]] && resume=(-r)
log "$bits bpw: converting ${resume[*]:+(resuming) }$DRAFTER_SRC_DIR -> $out (work $work, image $img)"
t0=$(date +%s)
docker run --rm --name "muse-quant-drafter" --gpus all --ipc=host --shm-size 16g --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -v "$DRAFTER_SRC_DIR:/src:ro" -v "$work:/work" -v "$out:/out" -v "$cache:/cache" \
  --entrypoint env "$img" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /tools/convert.py \
  -i /src -w /work -o /out -b "$bits" -cb mul1 "${resume[@]}"
[[ -f "$out/quantization_config.json" ]] || die "the converter exited without writing a pack"
# 46 EXL3 matrices (1 fc + 5 x 9); header-only check of every shard
python3 tools/exl3/verify_exl3_pack.py "$out" 46
cp "$DRAFTER_SRC_DIR/README.md" "$out/README-source.md" 2>/dev/null || true
log "$bits bpw: done in $(( $(date +%s) - t0 )) s; $(du -sh "$out" | cut -f1)"
