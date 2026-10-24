#!/usr/bin/env bash
# Download one pinned Hugging Face repo into a directory, in a container (no host installs), resumable.
#   scripts/fetch.sh source|gguf|drafter
# Run detached on the node that should hold the files, e.g.:
#   setsid nohup scripts/fetch.sh source > logs/fetch-source.log 2>&1 </dev/null &
# Beside a live serve (unified memory): FETCH_DOCKER_ARGS="--memory 768m -e HF_HUB_DISABLE_XET=1" FETCH_WORKERS=2 caps the
# container (its page cache included), so the download cannot take the serve's headroom.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
what="${1:?usage: scripts/fetch.sh source|gguf|drafter}"
case "$what" in
  source)  repo=$SRC_REPO;     rev=$SRC_REV;     dir=$SRC_DIR;         pat='None' ;;
  gguf)    repo=$GGUF_REPO;    rev=$GGUF_REV;    dir=$GGUF_DIR;        pat="['$GGUF_QUANT/*', '*.md']" ;;
  drafter) repo=$DRAFTER_REPO; rev=$DRAFTER_REV; dir=$DRAFTER_SRC_DIR; pat='None' ;;
  *) die "unknown target $what" ;;
esac
mkdir -p "$dir"
log "$repo @ ${rev:0:12} -> $dir"
# shellcheck disable=SC2086
docker run --rm --name "muse-fetch-$what" --network host --user "$(id -u):$(id -g)" ${FETCH_DOCKER_ARGS:-} \
  -e HF_HOME=/out/.hf-home -e HF_XET_HIGH_PERFORMANCE=1 -v "$dir:/out" \
  --entrypoint python3 "$HF_IMAGE" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('$repo', revision='$rev', local_dir='/out', allow_patterns=$pat, max_workers=${FETCH_WORKERS:-8})"
rm -rf "$dir/.hf-home"
log "done: $(du -sh "$dir" | cut -f1)"
