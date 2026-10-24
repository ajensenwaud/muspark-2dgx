#!/usr/bin/env bash
# One-time setup on the head (rank 0), idempotent: everything except the model pack (scripts/fetch_pack.sh).
#   1. the CUDA base image (pulled once if it is not here) and the Zig build image, both copied to the worker
#   2. the engine: TensorFold ZIG_TF_TAG + patches-zig/, tree checked against ZIG_TREE, built in a container,
#      zig-out/ copied to the worker (scripts/zig_prepare.sh)
#   3. the DFlash2 drafter: RadixArk's bf16 original fetched at DRAFTER_REV, quantised to EXL3 4.0 bpw on this node's
#      GPU (a few minutes; needs the exllamav3 image, scripts/build-image.sh, ~30-60 min the first time), copied to
#      the worker. The drafter is not redistributed by this recipe; SKIP_DRAFTER=1 skips this step (then DRAFTER=0).
# Nothing is installed on the hosts. Disk: ~10 GB for the images and build, ~12 GB for the drafter (source + pack).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_hosts WORKER

copy_image() {   # image: present on the worker with the same layers as here
  local img=$1 hl wl
  if ! ssh "$WORKER" "docker image inspect $img >/dev/null 2>&1" </dev/null; then
    log "copying $img to $WORKER"
    docker save "$img" | ssh "$WORKER" docker load >/dev/null
  fi
  hl=$(docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' "$img")
  wl=$(ssh "$WORKER" "docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' $img" </dev/null)
  [[ "$hl" == "$wl" ]] || die "$img differs between this node and $WORKER"
}

# 1. images
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && ! docker image inspect tensorfold-glm53:v0.6.0 >/dev/null 2>&1; then
  log "pulling the CUDA base image (once, ~20 GB): $BASE_IMAGE"
  docker pull -q "$BASE_IMAGE" >/dev/null
fi

# 2. the engine (builds ZIG_BUILD_IMAGE from the base when it is missing)
scripts/zig_prepare.sh
copy_image "$ZIG_RUN_IMAGE"

# 3. the drafter
if [[ "${SKIP_DRAFTER:-0}" == 1 ]]; then
  log "SKIP_DRAFTER=1: no drafter; serve with DRAFTER=0"
elif [[ -f "$DRAFTER_DIR/quantization_config.json" ]] && ssh "$WORKER" "test -f '$DRAFTER_DIR/quantization_config.json'" </dev/null; then
  log "drafter pack present on both nodes: $DRAFTER_DIR"
else
  if [[ ! -f "$DRAFTER_DIR/quantization_config.json" ]]; then
    docker image inspect "$IMAGE:latest" >/dev/null 2>&1 || scripts/build-image.sh
    [[ -f "$DRAFTER_SRC_DIR/model.safetensors" ]] || HF_IMAGE="$IMAGE:latest" scripts/fetch.sh drafter
    scripts/quantise_drafter.sh "$DRAFTER_BITS"
  fi
  log "copying the drafter pack to $WORKER"
  ssh "$WORKER" "mkdir -p '$DRAFTER_DIR'" </dev/null
  rsync -a "$DRAFTER_DIR/" "$WORKER:$DRAFTER_DIR/"
  ssh "$WORKER" python3 - "$DRAFTER_DIR" 46 < tools/exl3/verify_exl3_pack.py >/dev/null || die "the worker's drafter copy does not verify"
fi
log "ready: scripts/fetch_pack.sh (the model, 182 GiB), then ./start.sh"
