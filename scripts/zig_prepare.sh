#!/usr/bin/env bash
# Build the Zig engine: TensorFold ZIG_TF_TAG plus patches-zig/, compiled in a build container (nothing is installed
# on the hosts), then copied to the worker. Run on the head (rank 0); CPU only, safe beside a running serve (the build
# container is capped at ZIG_BUILD_MEM). The engine base and its patches: docs/ZIG-MUSE.md.
#   scripts/zig_prepare.sh            clone/patch, build image if missing, build, copy zig-out/ to $WORKER
#   NO_WORKER=1 scripts/zig_prepare.sh   the same without the copy
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE=$(pwd)
source scripts/config.sh
[[ -n "${NO_WORKER:-}" ]] || need_hosts WORKER

# 1. the source: TensorFold at the pinned release, the patches on a fresh branch, the tree hash checked
if [[ ! -d "$ZIG_DIR/.git" ]]; then
  log "cloning TensorFold into $ZIG_DIR"
  git clone -q "$TF_REPO" "$ZIG_DIR"
fi
cd "$ZIG_DIR"
git fetch -q --tags origin
[[ "$(git rev-parse "$ZIG_TF_TAG^{commit}")" == "$ZIG_TF_COMMIT" ]] || die "$ZIG_TF_TAG is not $ZIG_TF_COMMIT"
git checkout -q -B muse-zig "$ZIG_TF_COMMIT"
log "applying $(ls "$RECIPE"/patches-zig/*.patch | wc -l) patches to $ZIG_TF_TAG"
GIT_COMMITTER_NAME=recipe GIT_COMMITTER_EMAIL=recipe@localhost git am -q --3way "$RECIPE"/patches-zig/*.patch \
  || { git am --abort; die "patches-zig/ did not apply"; }
tree=$(git rev-parse HEAD^{tree})
[[ "$tree" == "$ZIG_TREE" ]] || die "patched tree is $tree, config.sh pins $ZIG_TREE"
log "tree $tree"

# 2. the build image: a local TensorFold CUDA base (never pulled) plus Zig (zig/docker/Dockerfile.build)
if ! docker image inspect "$ZIG_BUILD_IMAGE" >/dev/null 2>&1; then
  base=""
  for b in "$BASE_IMAGE" tensorfold-glm53:v0.6.0; do  # the GLM recipe's base by digest, else its local tag
    if docker image inspect "$b" >/dev/null 2>&1; then base=$b; break; fi
  done
  [[ -n "$base" ]] || die "no local TensorFold CUDA base image ($BASE_IMAGE or tensorfold-glm53:v0.6.0)"
  log "building $ZIG_BUILD_IMAGE on $base"
  docker build -q --pull=false -t "$ZIG_BUILD_IMAGE" --build-arg BASE_IMAGE="$base" \
    -f "$RECIPE/zig/docker/Dockerfile.build" "$RECIPE/zig/docker" >/dev/null
fi

# 3. compile (a few minutes the first time): kernels for GB10 (sm_121), the TP mailbox kernel, the tools, the server
log "compiling ($ZIG_BUILD_JOBS jobs, $ZIG_BUILD_MEM)"
rm -rf "$ZIG_DIR/zig-out"
docker run --rm --name muse-zig-build --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/work/.zig-global \
  --memory="$ZIG_BUILD_MEM" --memory-swap="$ZIG_BUILD_MEM" --cpus="$ZIG_BUILD_JOBS" \
  -v "$ZIG_DIR:/work" -w /work "$ZIG_BUILD_IMAGE" bash -c "
    set -e
    NVCC=\$(which nvcc)
    zig build -j$ZIG_BUILD_JOBS -Dnvcc=\$NVCC -Dtp-nvcc=\$NVCC -Dsm=121 --prefix /work/zig-out
    zig build native -j$ZIG_BUILD_JOBS -Dnvcc=\$NVCC -Dtp-nvcc=\$NVCC -Dsm=121 --prefix /work/zig-out"
for f in bin/tf-tp-test bin/tf-cuda-test bin/tf-mimo bin/tf-muse bin/tensorfold native/bin/tensorfold-native; do
  [[ -x "$ZIG_DIR/zig-out/$f" ]] || die "build incomplete: no $f"
done
"$ZIG_DIR/zig-out/bin/tf-tp-test" kernel >/dev/null 2>&1 || die "tf-tp-test has no mailbox kernel (-Dtp-nvcc)"
(cd "$ZIG_DIR/zig-out" && sha256sum bin/* native/bin/* > SHA256SUMS)

# 4. the same build on the worker
if [[ -z "${NO_WORKER:-}" ]]; then
  log "copying the build to $WORKER:$ZIG_DIR/zig-out"
  ssh "$WORKER" "mkdir -p $ZIG_DIR" </dev/null
  rsync -a --delete "$ZIG_DIR/zig-out/" "$WORKER:$ZIG_DIR/zig-out/"
fi
log "built $(git rev-parse --short HEAD) (tree ${tree:0:12}) in $ZIG_DIR/zig-out"
