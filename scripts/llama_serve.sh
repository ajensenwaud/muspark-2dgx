#!/usr/bin/env bash
# Arm A baseline: Unsloth's GGUF on llama.cpp (PR #30616) across both Sparks over RPC.
#   default roles: rpc-server on the worker, llama-server on the head (holds the GGUF)
#   ROLES=swap:    rpc-server on the head, llama-server on the worker (holds the GGUF in WORKER_GGUF_DIR);
#                  the API is then at the worker's address (http://$WORKER_IP:$API_PORT)
#   Either way the server's layers are split between its local GPU and the RPC GPU, all on GPU.
#
#   scripts/llama_serve.sh start|stop|status          (run on the head, from the recipe dir)
#
# Containers: muse-llama-rpc, muse-llama-server. LOCK=1 writes $WORK_ROOT/gpu.lock on both nodes (0: none).
# Knobs (env): LLAMA_IMAGE, GGUF_QUANT, RPC_PORT, API_PORT, CTX, PARALLEL, KV_UNIFIED (1: one KV pool of CTX for all
# slots; 0: CTX / PARALLEL per slot), TENSOR_SPLIT, RPC_CACHE (1: rpc-server -c, tensor cache on its disk), EXTRA_ARGS.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

LLAMA_IMAGE="${LLAMA_IMAGE:-muse-llama:335916a}"
ROLES="${ROLES:-default}"
WORKER_SSH="${WORKER_SSH:-$WORKER}"
need_hosts WORKER MASTER WORKER_IP
if [[ "$ROLES" == swap ]]; then
  RPC_HOST="${RPC_HOST:-$MASTER}"; RPC_ON=local; SRV_ON="$WORKER_SSH"
  SRV_GGUF_DIR="${WORKER_GGUF_DIR:-}"
else
  RPC_HOST="${RPC_HOST:-$WORKER_IP}"; RPC_ON="$WORKER_SSH"; SRV_ON=local; SRV_GGUF_DIR="$GGUF_DIR"
fi
RPC_PORT="${RPC_PORT:-50052}"
API_PORT="${API_PORT:-8892}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-4}"
KV_UNIFIED="${KV_UNIFIED:-1}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,1}"            # order follows llama.cpp's device list (see the server log)
RPC_CACHE="${RPC_CACHE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOCK="${LOCK:-1}"
LOG_DIR="$WORK_ROOT/logs"

GPU_OPTS="--gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1"

on() { local where=$1; shift; if [[ "$where" == local ]]; then bash -c "$*"; else ssh -n "$where" "$*"; fi; }
lock() { [[ "$LOCK" == 1 ]] || return 0
  printf 'G1 baseline (llama.cpp) %s\n' "$(date -Is)" | tee "$WORK_ROOT/gpu.lock" | ssh "$WORKER_SSH" "cat > ~/muse/gpu.lock"; }

case "${1:-}" in
  start)
    [[ -n "$SRV_GGUF_DIR" ]] || die "ROLES=swap needs WORKER_GGUF_DIR (the GGUF folder on the worker)"
    SHARD1="$(on "$SRV_ON" "ls $SRV_GGUF_DIR/$GGUF_QUANT/*-00001-of-*.gguf 2>/dev/null | head -1" || true)"
    [[ -n "$SHARD1" ]] || die "no first shard under $SRV_GGUF_DIR/$GGUF_QUANT on $SRV_ON"
    lock
    cache=""; [[ "$RPC_CACHE" == 1 ]] && cache="-c"
    log "rpc-server on $RPC_ON ($RPC_HOST:$RPC_PORT)"
    on "$RPC_ON" "mkdir -p \$HOME/muse/rpc-cache && docker rm -f muse-llama-rpc >/dev/null 2>&1; \
      docker run -d --name muse-llama-rpc --user \$(id -u):\$(id -g) $GPU_OPTS -v \$HOME/muse/rpc-cache:/cache -e LLAMA_CACHE=/cache \
      $LLAMA_IMAGE rpc-server -H $RPC_HOST -p $RPC_PORT $cache >/dev/null"
    for _ in $(seq 60); do
      on "$RPC_ON" "docker logs muse-llama-rpc 2>&1" | grep -q "endpoint" && break; sleep 2
    done
    on "$RPC_ON" "docker logs muse-llama-rpc 2>&1 | tail -8"
    kvu=""; [[ "$KV_UNIFIED" == 1 ]] && kvu="--kv-unified"
    log "llama-server on $SRV_ON :$API_PORT, model $(basename "$SHARD1"), -c $CTX -np $PARALLEL $kvu, split $TENSOR_SPLIT"
    on "$SRV_ON" "docker rm -f muse-llama-server >/dev/null 2>&1; \
      docker run -d --name muse-llama-server --user \$(id -u):\$(id -g) $GPU_OPTS -v $SRV_GGUF_DIR:/models:ro $LLAMA_IMAGE \
      llama-server -m /models/$GGUF_QUANT/$(basename "$SHARD1") --alias Muse-Spark-Open-$GGUF_QUANT \
      --rpc $RPC_HOST:$RPC_PORT -ngl 999 -fa on -c $CTX -np $PARALLEL $kvu \
      --tensor-split $TENSOR_SPLIT -lm dio -fit off --cache-ram 0 --jinja --metrics \
      --host 0.0.0.0 --port $API_PORT $EXTRA_ARGS >/dev/null"
    log "loading; poll /health on port $API_PORT; docker logs -f muse-llama-server (on $SRV_ON)"
    ;;
  stop)
    ts=$(date +%Y%m%d-%H%M%S)
    mkdir -p "$LOG_DIR"
    for c in muse-llama-server muse-llama-rpc; do
      where=$SRV_ON; [[ $c == muse-llama-rpc ]] && where=$RPC_ON
      if [[ "$where" == local ]]; then docker logs "$c" > "$LOG_DIR/$c-$ts.log" 2>&1 || true; docker rm -f "$c" >/dev/null 2>&1 || true
      else ssh -n "$where" "mkdir -p ~/muse/logs; docker logs $c > ~/muse/logs/$c-$ts.log 2>&1; docker rm -f $c >/dev/null 2>&1" || true; fi
    done
    if [[ "$LOCK" == 1 ]]; then ssh -n "$WORKER_SSH" "rm -f ~/muse/gpu.lock" || true; rm -f "$WORK_ROOT/gpu.lock"; fi
    log "stopped; logs in ~/muse/logs/*-$ts.log on the node that ran each container"
    ;;
  status)
    docker ps --filter name=muse-llama --format '{{.Names}} {{.Status}}'
    ssh -n "$WORKER_SSH" "docker ps --filter name=muse-llama --format '{{.Names}} {{.Status}}'"
    if [[ "$SRV_ON" == local ]]; then curl -s "localhost:$API_PORT/health"; else curl -s "$WORKER_IP:$API_PORT/health"; fi; echo
    ;;
  *) die "usage: [ROLES=swap WORKER_GGUF_DIR=...] scripts/llama_serve.sh start|stop|status" ;;
esac
