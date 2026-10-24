#!/usr/bin/env bash
# In-run memory watchdog for one node of the two-Spark serve. start.sh runs one on each node beside its rank;
# it exits on its own once that rank's container is gone (stop.sh), and is never needed by hand.
#   mem_guard.sh LOCAL_CONTAINER [PEER PEER_CONTAINER]
# Every GUARD_EVERY s (2) it reads this node's MemAvailable; under GUARD_MIB (512) it saves the rank's log tail,
# removes the local rank container (docker rm -f: the process dies, its unified memory returns at once), then the
# peer's over ssh (best effort, with a timeout; the peer's own guard and TP fail-fast stop it anyway). With a PEER it
# also reads the peer's MemAvailable (ssh, short timeouts) and trips when that is under GUARD_MIB or the peer has not
# answered GUARD_MISS (5) times in a row: a node that thrashes stops answering before it stops entirely.
# Writes GUARD_DIR/mem_guard-<container>.{pid,log} and, when it trips, GUARD_DIR/mem_guard.tripped.
set -uo pipefail
LOCAL=${1:?local container}
PEER=${2:-}
PEER_C=${3:-}
GUARD_MIB=${GUARD_MIB:-512}
GUARD_EVERY=${GUARD_EVERY:-2}
GUARD_MISS=${GUARD_MISS:-5}
GUARD_DIR=${GUARD_DIR:-$HOME/muse}
LOG=$GUARD_DIR/mem_guard-$LOCAL.log
echo $$ > "$GUARD_DIR/mem_guard-$LOCAL.pid"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
avail() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
pavail() { timeout 6 ssh -n -o ConnectTimeout=4 -o BatchMode=yes "$PEER" "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo" 2>/dev/null; }
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$LOCAL" 2>/dev/null)" == true ]]; }
trip() {
  say "TRIP: $*: removing $LOCAL${PEER:+ and $PEER_C on $PEER}"
  docker logs --tail 20 "$LOCAL" >> "$LOG" 2>&1
  docker rm -f "$LOCAL" >/dev/null 2>&1
  [[ -n "$PEER" ]] && timeout 20 ssh -n -o ConnectTimeout=5 -o BatchMode=yes "$PEER" "docker rm -f $PEER_C" >/dev/null 2>&1
  touch "$GUARD_DIR/mem_guard.tripped"
  say "removed; MemAvailable now $(avail) MiB"
  exit 2
}
say "watching $LOCAL${PEER:+ and $PEER} (trip under $GUARD_MIB MiB${PEER:+ or $GUARD_MISS missed reads})"
for _ in $(seq 1 30); do running && break; sleep 1; done   # the container may still be starting
miss=0; low=999999; plow=999999; n=0
while running; do
  h=$(avail)
  (( h < low )) && low=$h
  (( h < GUARD_MIB )) && trip "MemAvailable $h MiB here"
  if [[ -n "$PEER" ]]; then
    w=$(pavail)
    if [[ -z "$w" ]]; then
      miss=$((miss + 1)); say "peer $PEER did not answer ($miss)"
      (( miss >= GUARD_MISS )) && trip "peer $PEER silent $miss times"
    else
      miss=0; (( w < plow )) && plow=$w
      (( w < GUARD_MIB )) && trip "MemAvailable $w MiB on $PEER"
    fi
  fi
  n=$((n + 1)); (( n % 150 == 0 )) && say "ok: lowest MemAvailable here $low MiB${PEER:+, $PEER $plow MiB}"
  sleep "$GUARD_EVERY"
done
say "$LOCAL gone; exiting (lowest MemAvailable here $low MiB${PEER:+, $PEER $plow MiB})"
