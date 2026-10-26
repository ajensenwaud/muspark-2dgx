# Muse Spark Open (816B-A42B) on two DGX Sparks via TensorFold's Zig engine: shared settings. Override any of them
# in scripts/local.sh (gitignored; start from scripts/local.sh.example).
# shellcheck disable=SC2034

# --- nodes -------------------------------------------------------------------------------------------------------
# No defaults: set these in scripts/local.sh. Scripts that reach the other node call need_hosts (below), which stops
# with a message when one is empty.
HEAD="${HEAD:-}"                              # rank 0 (serves the API), e.g. spark-a
WORKER="${WORKER:-}"                          # rank 1 as ssh reaches it over the CX7 link, e.g. user@192.0.2.2
MASTER="${MASTER:-}"                          # rank 0's CX7 address (rendezvous + NCCL), e.g. 192.0.2.1
SRC_NODE="${SRC_NODE:-}"                      # holds the source checkpoint and runs the conversion (disk)
WORKER_IP="${WORKER_IP:-}"                    # the worker's CX7 address; empty: WORKER's host part

# --- checkpoints -------------------------------------------------------------------------------------------------
# Source for our own EXL3 packs: AMD Quark MXFP4 export of Meta's BF16 (ungated). Experts are MXFP4 (e2m1, e8m0
# scales every 32) except layers' experts listed bf16 in its quantization_config; attention, latent projections,
# router and head are bf16. Meta's BF16 (1.64 TB) and NVFP4 (533 GB) repos are gated (manual approval).
SRC_REPO="amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
SRC_REV="4885eb2a901c2638854ef35cf6833e6447622afb"
# Baseline: Unsloth's dynamic GGUF on llama.cpp (Muse support: ggml-org/llama.cpp PR #30616, unmerged at pin time).
GGUF_REPO="unsloth/Muse-Spark-Open-816B-A42B-GGUF"
GGUF_REV="c00670568b6ce5e56a247f67091247984df2a01e"
GGUF_QUANT="${GGUF_QUANT:-UD-IQ2_XXS}"       # 225.6 GB: the largest Unsloth quant that fits two Sparks
# Drafter: RadixArk's DFlash2 speculator (5 layers, block 8), bf16, trained against Meta's NVFP4 target.
DRAFTER_REPO="RadixArk/Muse-Spark-Open-DFlash2"
DRAFTER_REV="46c90f412549c472941a68382d91a99e5e6372a0"

MODELS_ROOT="${MODELS_ROOT:-$HOME/models/muse-spark}"
SRC_DIR="${SRC_DIR:-$MODELS_ROOT/Muse-Spark-Open-MXFP4}"
GGUF_DIR="${GGUF_DIR:-$MODELS_ROOT/gguf-$GGUF_QUANT}"
DRAFTER_SRC_DIR="${DRAFTER_SRC_DIR:-$MODELS_ROOT/Muse-Spark-Open-DFlash2}"
WORK_ROOT="${WORK_ROOT:-$HOME/muse}"

# Recipe image (docker/Dockerfile, scripts/build-image.sh): a TensorFold GB10 runtime plus exllamav3 at EXL3_COMMIT
# with patches-exl3/ applied (muse_spark architecture). EXL3_PATCHED_TREE is the tree those patches produce (commit
# hashes depend on who applies them; trees do not); update it whenever patches-exl3/ changes.
IMAGE="${IMAGE:-muse-tf}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529}"
EXL3_COMMIT="058b8630fd081f3ff46fce6246955b157c3a2e3b"
EXL3_PATCHED_SHA="8e7581ae9bde58ffda320e1e3eb0bc9ab10f001b"   # an image label only
EXL3_PATCHED_TREE="d11926cec947ef5981a033397f1075c2b2a30131"  # git tree of EXL3_COMMIT + patches-exl3/ (build-image.sh checks)

# Image used only as a Python + huggingface_hub runtime for downloads (no host installs).
HF_IMAGE="${HF_IMAGE:-mimo-tf:latest}"

# --- Zig engine (scripts/zig_prepare.sh) -------------------------------------------------------------------------
# TensorFold's Zig engine at an upstream main commit plus patches-zig/ (the muse_spark family and its build wiring,
# the kernel-contract registrations, and the server's recipient-reply split). The build runs in ZIG_BUILD_IMAGE:
# BASE_IMAGE (local, never pulled) plus Zig.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold}"
ZIG_TF_COMMIT="40c6406b3deb290aa3ec8ced310bc1c8896d160c"   # main, 26 Oct 2026 (no tag); .zig-version 0.17.0
ZIG_TREE="71a9f224a5304a7af39c6652ed5b5e5e833d64af"   # git tree of ZIG_TF_COMMIT with patches-zig/ applied (zig_prepare.sh checks)
ZIG_DIR="${ZIG_DIR:-$HOME/muse/tensorfold-zig}"      # the patched tree and its build (zig-out/), the same path on both nodes
ZIG_BUILD_IMAGE="${ZIG_BUILD_IMAGE:-muse-zig-build:0.17.0}"   # FROM BASE_IMAGE above (local, never pulled) + Zig
ZIG_BUILD_MEM="${ZIG_BUILD_MEM:-8g}"          # the build container's memory cap (a serve may hold the rest)
ZIG_BUILD_JOBS="${ZIG_BUILD_JOBS:-4}"

# --- the published pack (scripts/fetch_pack.sh) ------------------------------------------------------------------
# P1.8s: experts 1.80 bpw on average, allocated per tensor by measured sensitivity; dense 4.0, head 6 (182 GiB,
# 27 shards, 48,051 EXL3 matrices). PACK_REV is the Hugging Face commit the release was verified against.
PACK_REPO="${PACK_REPO:-ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw}"
PACK_REV="${PACK_REV:-efcdf2ec4222c4597e291f1b07edd19c3c02c652}"
PACK_MATRICES=48051                           # EXL3 matrices in the pack (tools/exl3/verify_exl3_pack.py)

# --- Zig serve (./start.sh, ./stop.sh, scripts/bench_zig.sh) ------------------------------------------------------
# tensorfold-native serve on the head (TP rank 0), `tf-muse follow` on the worker (rank 1); the pack, the drafter
# pack and zig-out at the same paths on both nodes.
PACK_NAME="${PACK_NAME:-Muse-Spark-Open-exl3-1.8s}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/$PACK_NAME}"
DRAFTER="${DRAFTER:-1}"                       # 1: the DFlash2 drafter (drafted replies byte-identical to plain); 0: off
DRAFTER_BITS="${DRAFTER_BITS:-4.0}"
DRAFTER_DIR="${DRAFTER_DIR:-$MODELS_ROOT/Muse-Spark-Open-DFlash2-exl3-${DRAFTER_BITS}bpw}"   # scripts/quantise_drafter.sh
SERVED_NAME="${SERVED_NAME:-Muse-Spark-Open}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"                          # API port
ZIG_TP_PORT="${ZIG_TP_PORT:-29681}"           # rank rendezvous
# Serve profiles (PROFILE=long|multi ./start.sh). Both: fp16 KV, the prompt cache and the drafter. The context per
# slot is what start.sh's memory estimate allows on the MemAvailable it reads at start: with CONTEXT_AUTO=1 a request
# that does not fit is cut to the largest that does.
#   long   1 slot; asks for 1,048,576 tokens, gets ~450K-470K on two idle Sparks (3 GiB a rank kept free)
#   multi  4 slots; asks for 196,608 tokens each, gets ~110K
PROFILE="${PROFILE:-long}"
case "$PROFILE" in
  long)  _ctx=1048576; _par=1 ;;
  multi) _ctx=196608;  _par=4 ;;
  *) echo "config.sh: unknown PROFILE=$PROFILE (long|multi)" >&2; exit 1 ;;
esac
ZIG_CONTEXT="${ZIG_CONTEXT:-$_ctx}"           # tokens of context per stream slot
PARALLEL="${PARALLEL:-$_par}"                 # stream slots
CONTEXT_AUTO="${CONTEXT_AUTO:-1}"
unset _ctx _par
# Memory guard (start.sh): refuse a context x slots that tools/mem/estimate.py says leaves a rank under
# MEM_MIN_FREE_GIB, where MEM_FIXED_GIB is the measured per-rank rest beyond weights, drafter and KV; while serving,
# tools/mem/mem_guard.sh on each node removes both ranks under GUARD_MIB MemAvailable.
MEM_MIN_FREE_GIB="${MEM_MIN_FREE_GIB:-3}"
MEM_FIXED_GIB="${MEM_FIXED_GIB:-9.2}"         # measured, includes the 1 GiB prompt cache; at the chosen contexts both
                                              # ranks idled at 5.0-6.7 GiB MemAvailable (long and multi)
GUARD_MIB="${GUARD_MIB:-512}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
ZIG_COMM="${ZIG_COMM:-roce}"                  # roce (one-shot RoCE mailbox) or nccl: identical bytes
# Engine settings, both ranks. The family's speed and memory levers are built in on this engine (no switches); the
# KV cache is fp16. NCCL at 4 channels: 1.2 GiB a rank of pinned host buffers -> 78 MiB, prompt reading unchanged.
ZIG_LEVERS="${ZIG_LEVERS:--e NCCL_MAX_NCHANNELS=4}"
# Prompt cache: kept prompt states let a follow-up turn skip the history it shares with an earlier request (replies
# byte-identical to a cache-off serve on 115 of 115 test requests). On by default in the engine for this family:
# 1 GiB a rank, inside MEM_FIXED_GIB, with the engine's 3 GiB reserve. PROMPT_CACHE=0 turns it off
# (--prompt-cache-gib 0); PROMPT_CACHE_GIB=N passes another budget.
PROMPT_CACHE="${PROMPT_CACHE:-1}"
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"
ZIG_CONTAINER="${ZIG_CONTAINER:-muse-zig}"    # rank 1: $ZIG_CONTAINER-follow on the worker
ZIG_RUN_IMAGE="${ZIG_RUN_IMAGE:-$ZIG_BUILD_IMAGE}"   # the CUDA runtime the binaries run in (present on both nodes)
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f0np0}"
NCCL_HCAS="${NCCL_HCAS:-rocep1s0f0,roceP2p1s0f0}"
NCCL_GID="${NCCL_GID:-3}"
# Request sampling defaults when a request omits them: the model's generation_config.json (temperature 1, top_p 1,
# top_k 64), as RadixArk's drafter numbers were taken with
SAMPLING_ARGS="${SAMPLING_ARGS:---temperature 1.0 --top-p 1.0 --top-k 64}"

[[ -f "$(dirname "${BASH_SOURCE[0]}")/local.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/local.sh"
[[ -z "$WORKER_IP" && -n "$WORKER" ]] && WORKER_IP="${WORKER#*@}"

log() { printf '[%s %s] %s\n' "$(date +%H:%M:%S)" "$(basename "$0")" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
# need_hosts VAR...: stop unless every named node setting is set (HEAD, WORKER, MASTER, SRC_NODE, WORKER_IP).
need_hosts() {
  local v
  for v in "$@"; do
    [[ -n "${!v:-}" ]] || die "$v is empty: set HEAD, WORKER, MASTER (and SRC_NODE for conversions) in scripts/local.sh, see scripts/local.sh.example"
  done
}
