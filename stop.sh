#!/usr/bin/env bash
# Stop the Zig engine on both Sparks, keeping both ranks' logs (per-request stats: tok/s, drafting) in logs/.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh
need_hosts WORKER
mkdir -p logs
ts=$(date +%Y%m%d-%H%M%S)
if docker ps -a --format '{{.Names}}' | grep -qx "$ZIG_CONTAINER"; then
  docker logs "$ZIG_CONTAINER" > "logs/$ZIG_CONTAINER-$ts.log" 2>&1 || true
  docker rm -f "$ZIG_CONTAINER" >/dev/null 2>&1 && log "stopped rank 0 (log logs/$ZIG_CONTAINER-$ts.log)"
fi
if ssh "$WORKER" "docker ps -a --format '{{.Names}}'" </dev/null | grep -qx "$ZIG_CONTAINER-follow"; then
  ssh "$WORKER" "docker logs $ZIG_CONTAINER-follow" </dev/null > "logs/$ZIG_CONTAINER-follow-$ts.log" 2>&1 || true
  ssh "$WORKER" "docker rm -f $ZIG_CONTAINER-follow >/dev/null 2>&1" </dev/null && log "stopped rank 1 on $WORKER"
fi
log "stopped"
