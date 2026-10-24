#!/usr/bin/env bash
# Download the published pack instead of quantising it (the split conversion took ~3 days on both Sparks): Hugging
# Face on the head, pinned to PACK_REV, inside a container as you (no host installs, no root-owned files); every shard
# checked against its header; copied to the worker over the CX7 link and checked there.
#   scripts/fetch_pack.sh                  the whole pack (182 GiB: ~195 GB free disk on the head and the worker)
#   ALLOW='["config.json", "LICENSE.md"]' scripts/fetch_pack.sh   only matching files, no copy (a quick test)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
[[ -n "$PACK_REV" ]] || die "no PACK_REV pinned in scripts/config.sh"
[[ -n "${ALLOW:-}" ]] || need_hosts WORKER
img="$HF_IMAGE"; docker image inspect "$img" >/dev/null 2>&1 || img="$ZIG_RUN_IMAGE"
docker run --rm --entrypoint python3 "$img" -c "import huggingface_hub" 2>/dev/null \
  || die "no image with huggingface_hub here ($HF_IMAGE or $ZIG_RUN_IMAGE): scripts/prepare.sh first"
mkdir -p "$MODEL_DIR"
allow="${ALLOW:-None}"
log "$PACK_REPO @ ${PACK_REV:0:12} -> $MODEL_DIR${ALLOW:+ (only $ALLOW)}"
docker run --rm --name muse-fetch-pack --network host --user "$(id -u):$(id -g)" -e HF_HOME=/out/.hf-home \
  ${HF_TOKEN:+-e HF_TOKEN} -v "$MODEL_DIR:/out" --entrypoint python3 "$img" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('$PACK_REPO', revision='$PACK_REV', local_dir='/out', allow_patterns=$allow, max_workers=8)"
rm -rf "$MODEL_DIR/.hf-home" "$MODEL_DIR/.cache"
if [[ -n "${ALLOW:-}" ]]; then
  log "partial download done (ALLOW set): $(ls "$MODEL_DIR" | tr '\n' ' ')"
  exit 0
fi
python3 tools/exl3/verify_exl3_pack.py "$MODEL_DIR" "$PACK_MATRICES" || die "the download does not verify"
log "copying to $WORKER over the CX7 link"
ssh "$WORKER" "mkdir -p '$MODEL_DIR'" </dev/null
rsync -a --delete "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
ssh "$WORKER" python3 - "$MODEL_DIR" "$PACK_MATRICES" < tools/exl3/verify_exl3_pack.py || die "the worker's copy does not verify"
log "pack ready on both nodes: ./start.sh"
