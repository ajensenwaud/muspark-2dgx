#!/usr/bin/env bash
# Split conversion, second node: the state generation for the split conversion (tools/g3/state_gen.py) in the recipe
# image, GPU, TF32 override unset as for the converter. Writes the resumable job into $SPLIT/work.
#   tools/g3/run_state_gen.sh <L> [extra state_gen.py args]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/config.sh
L="${1:?usage: run_state_gen.sh <L> [args]}"; shift
SPLIT="${SPLIT:-$WORK_ROOT/quant-work/split-2.0}"
SPLIT_SRC="${SPLIT_SRC:-$MODELS_ROOT/Muse-Spark-Open-MXFP4-split}"
SRC_URL="${SRC_URL:-http://${WORKER_IP:-$WORKER}:8899}"
TAG="${TAG:-latest}"
mkdir -p "$SPLIT/work" "$SPLIT/stream" "$WORK_ROOT/scratch/home"
docker run --rm --name muse-stategen --gpus all --ipc=host --network host --shm-size 16g --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -v "$SPLIT_SRC:/src:ro" -v "$SPLIT/meta:/meta:ro" -v "$SPLIT/seed:/seed:ro" -v "$SPLIT/work:/work" \
  -v "$SPLIT/stream:/stream" -v "$SPLIT:/out" -v "$WORK_ROOT/scratch/home:/cache" -v "$PWD/tools:/rtools:ro" \
  --entrypoint env "$IMAGE:$TAG" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /rtools/g3/state_gen.py \
  --src /src --url "$SRC_URL" --index /meta/model.safetensors.index.json --seed /seed --work /work \
  --upto "$L" --stream-dir /stream --stats /out/state-gen-stats.json "$@"
