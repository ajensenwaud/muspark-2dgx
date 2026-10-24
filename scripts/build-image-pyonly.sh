#!/usr/bin/env bash
# Python-only image update (docker/Dockerfile.pyonly): FROM_TAG (an existing muse-tf image on this node) + the
# patches-exl3/ files after the ones it was built with. Tags muse-tf:<first 10 of EXL3_PATCHED_SHA>, checks the sha
# baked into the image, and leaves muse-tf:latest alone (a running converter may use it).
#   FROM_TAG=81a81acbb5 scripts/build-image-pyonly.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
FROM_TAG="${FROM_TAG:?set FROM_TAG}"
from_sha=$(docker run --rm --entrypoint cat "$IMAGE:$FROM_TAG" /opt/exllamav3.patched-sha)
tag="${EXL3_PATCHED_SHA:0:10}"
log "building $IMAGE:$tag from $IMAGE:$FROM_TAG (exllamav3 ${from_sha:0:10}) + $(ls patches-exl3/*.patch | wc -l) patches total"
docker build -f docker/Dockerfile.pyonly --build-arg "FROM_IMAGE=$IMAGE:$FROM_TAG" --build-arg "FROM_SHA=$from_sha" \
  --build-arg "EXL3_PATCHED_SHA=$EXL3_PATCHED_SHA" \
  --build-arg "RECIPE_COMMIT=$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)" -t "$IMAGE:$tag" .
got=$(docker run --rm --entrypoint cat "$IMAGE:$tag" /opt/exllamav3.patched-sha)
[[ "$got" == "$EXL3_PATCHED_SHA" ]] || die "image has exllamav3 $got, expected $EXL3_PATCHED_SHA"
log "done: $IMAGE:$tag, exllamav3 $got"
