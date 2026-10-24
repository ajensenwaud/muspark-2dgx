#!/usr/bin/env bash
# Start Muse Spark Open on TensorFold's Zig engine across both Sparks: rank 1 (`tf-muse follow`) on the worker, then
# the OpenAI-compatible server here (rank 0) on :$PORT. The pack ($MODEL_DIR) and the build ($ZIG_DIR/zig-out, from
# scripts/prepare.sh) sit at the same paths on both nodes. Waits for /health, then sends one warm-up request.
#   ./start.sh                  PROFILE=long: 1 slot, the longest context that fits (~905K tokens)
#   PROFILE=multi ./start.sh    4 slots x 196,608 tokens
#   ./start.sh restart          ./stop.sh first (saves both ranks' logs), then start
# Both profiles serve the DFlash2 drafter (DRAFTER=0: off), the 8-bit KV cache (KV8=0: fp16) and the prompt cache
# (PROMPT_CACHE=0: off); ZIG_CONTEXT= / PARALLEL= override a profile's numbers (scripts/config.sh).
# Engine levers for both ranks: ZIG_LEVERS (scripts/config.sh; ZIG_LEVERS=" " for none); extra environment:
# ZIG_ENV="-e TF_MUSE_DEPTH=geometric ..."; another build: ZIG_BIN=dir.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh
need_hosts HEAD WORKER MASTER

[[ "${1:-}" == restart ]] && ./stop.sh
release_env=""
[[ "$KV8" == 1 ]] && release_env+=" -e TF_MUSE_KV8=1"
[[ "$PROMPT_CACHE" == 1 ]] && release_env+=" -e TF_MUSE_PROMPT_CACHE=1 -e TF_MUSE_PROMPT_CACHE_GIB=$PROMPT_CACHE_GIB"
ZIG_ENV="$release_env ${ZIG_ENV:-}"
log "profile $PROFILE: $PARALLEL slot(s) x $ZIG_CONTEXT tokens asked, KV $([[ "$KV8" == 1 ]] && echo int8 || echo fp16), prompt cache $([[ "$PROMPT_CACHE" == 1 ]] && echo on || echo off), drafter $([[ "$DRAFTER" == 1 ]] && echo on || echo off)"
BIN="${ZIG_BIN:-$ZIG_DIR/zig-out}"
[[ -x "$BIN/native/bin/tensorfold-native" && -x "$BIN/bin/tf-muse" ]] || die "no Zig build in $BIN: scripts/prepare.sh"
[[ -f "$MODEL_DIR/config.json" ]] || die "no pack at $MODEL_DIR"
ssh "$WORKER" "test -f '$MODEL_DIR/config.json' && test -x '$BIN/bin/tf-muse'" </dev/null \
  || die "the worker ($WORKER) lacks the pack or the build at $MODEL_DIR / $BIN"
docker ps --format '{{.Names}}' | grep -qx "$ZIG_CONTAINER" && die "$ZIG_CONTAINER is running: ./stop.sh first (or ./start.sh restart)"
ss -ltn "( sport = :$PORT )" | grep -q LISTEN && die "port $PORT is taken: $(ss -ltnp "( sport = :$PORT )" 2>/dev/null | tail -1)"
for n in "" "$WORKER"; do   # another GPU job (a conversion, a test) holds a node's GPU
  lock=$(if [[ -z "$n" ]]; then cat "$WORK_ROOT/gpu.lock" 2>/dev/null; else ssh "$n" "cat ~/muse/gpu.lock 2>/dev/null" </dev/null; fi) || true
  [[ -z "$lock" || "$lock" == "G6 queue"* ]] || die "GPU lock on ${n:-$HEAD}: $lock"
done

draft_env=""; draft_arg="--no-drafts"
if [[ "$DRAFTER" == 1 ]]; then
  [[ -f "$DRAFTER_DIR/config.json" ]] || die "DRAFTER=1 but no drafter pack at $DRAFTER_DIR (scripts/quantise_drafter.sh)"
  ssh "$WORKER" "test -f '$DRAFTER_DIR/config.json'" </dev/null || die "the worker lacks the drafter pack at $DRAFTER_DIR"
  draft_env="-e TF_MUSE_DRAFTER=/drafter -v $DRAFTER_DIR:/drafter:ro"; draft_arg=""
fi

# Memory check before anything loads (unified memory: an oversized KV allocation does not fail, it takes the host
# down): tools/mem/estimate.py sizes a rank (weights, drafter, KV for the context x slots, the measured fixed rest)
# against both nodes' MemAvailable now and refuses when a rank would keep under MEM_MIN_FREE_GIB. CONTEXT_AUTO=1
# serves the largest context that fits instead; MEM_CHECK=0 skips the check (the in-run guard below still runs).
if [[ "${MEM_CHECK:-1}" == 1 ]]; then
  env_all="$ZIG_LEVERS ${ZIG_ENV:-}"
  kv8=0; [[ "$env_all" =~ TF_MUSE_KV8=[12] ]] && kv8=1
  eh=0; [[ "$env_all" =~ TF_MUSE_EMBED_HOST=1 ]] && eh=1
  ha=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  wa=$(ssh "$WORKER" "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo" </dev/null) || die "the worker did not report MemAvailable"
  rc=0
  est=$(/usr/bin/python3 tools/mem/estimate.py "$MODEL_DIR" $([[ "$DRAFTER" == 1 ]] && echo "--drafter $DRAFTER_DIR") \
        --context "$ZIG_CONTEXT" --slots "$PARALLEL" --kv8 "$kv8" --embed-host "$eh" --avail-mib "$ha" "$wa" \
        --min-free-gib "$MEM_MIN_FREE_GIB" --fixed-gib "$MEM_FIXED_GIB") || rc=$?
  log "$(head -1 <<< "$est")"
  if (( rc != 0 )); then
    best=$(awk '/^VERDICT/ {print $3}' <<< "$est")
    if [[ "${CONTEXT_AUTO:-0}" == 1 ]] && (( ${best:-0} >= 4096 )); then
      log "CONTEXT_AUTO=1: context $ZIG_CONTEXT -> $best"; ZIG_CONTEXT=$best
    else
      die "refused before loading: context $ZIG_CONTEXT x $PARALLEL slots would leave a rank under $MEM_MIN_FREE_GIB GiB (largest that fits now: ${best:-none}; CONTEXT_AUTO=1 serves that, MEM_CHECK=0 skips this check)"
    fi
  fi
fi
rm -f "$WORK_ROOT/mem_guard.tripped"
ssh "$WORKER" "rm -f ~/muse/mem_guard.tripped; mkdir -p ~/muse/bin && cat > ~/muse/bin/mem_guard.sh && chmod +x ~/muse/bin/mem_guard.sh" \
  < tools/mem/mem_guard.sh || die "could not copy the memory guard to the worker"

zig_args() {   # RANK: docker run arguments for one rank
  echo "--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK" \
       "--ulimit memlock=-1 --ulimit stack=67108864 --user $(id -u):$(id -g) -e HOME=/tmp" \
       "-e NCCL_SOCKET_IFNAME=$NCCL_IFNAME -e NCCL_IB_HCA=$NCCL_HCAS -e NCCL_IB_GID_INDEX=$NCCL_GID" \
       "-e TF_TP_WORLD=2 -e TF_TP_RANK=$1 -e TF_TP_MASTER=$MASTER -e TF_TP_PORT=$ZIG_TP_PORT" \
       "-e TF_COMM_BACKEND=$ZIG_COMM -e TF_TP_ROCE_HCA=$NCCL_HCAS -e TF_TP_ROCE_FAST=1 $draft_env $ZIG_LEVERS ${ZIG_ENV:-}" \
       "-v $MODEL_DIR:/model:ro -v $BIN:/zig:ro"
}

mkdir -p logs
log "rank 1 on $WORKER ($PACK_NAME, drafter $([[ "$DRAFTER" == 1 ]] && basename "$DRAFTER_DIR" || echo off))"
ssh "$WORKER" "docker rm -f $ZIG_CONTAINER-follow >/dev/null 2>&1;
  docker run -d --name $ZIG_CONTAINER-follow $(zig_args 1) --entrypoint /zig/bin/tf-muse $ZIG_RUN_IMAGE follow /model >/dev/null" </dev/null
# In-run memory guard on each node (tools/mem/mem_guard.sh): under GUARD_MIB MemAvailable on either node, or the
# worker silent, both ranks are removed so neither host goes out of memory; it ends itself when the serve stops.
# GUARD=0 runs without it (never for a context that has not served before).
if [[ "${GUARD:-1}" == 1 ]]; then
  ssh "$WORKER" "GUARD_MIB=$GUARD_MIB setsid nohup ~/muse/bin/mem_guard.sh $ZIG_CONTAINER-follow >/dev/null 2>&1 </dev/null &" </dev/null
fi
log "rank 0 here, :$PORT"
docker rm -f "$ZIG_CONTAINER" >/dev/null 2>&1 || true
# shellcheck disable=SC2046,SC2086
docker run -d --name "$ZIG_CONTAINER" $(zig_args 0) --entrypoint /zig/native/bin/tensorfold-native "$ZIG_RUN_IMAGE" \
  serve /model --host "$HOST" --port "$PORT" --name "$SERVED_NAME" --parallel "$PARALLEL" --context "$ZIG_CONTEXT" \
  --max-tokens "$MAX_TOKENS" $SAMPLING_ARGS $draft_arg >/dev/null
if [[ "${GUARD:-1}" == 1 ]]; then
  GUARD_MIB=$GUARD_MIB GUARD_DIR=$WORK_ROOT setsid nohup tools/mem/mem_guard.sh "$ZIG_CONTAINER" "$WORKER" "$ZIG_CONTAINER-follow" \
    >/dev/null 2>&1 </dev/null &
fi
{ echo "# $(date -Is) PROFILE=$PROFILE ./start.sh"; echo "pack $MODEL_DIR"; echo "build $BIN ($(cat "$BIN/SHA256SUMS" 2>/dev/null | grep native | cut -c1-16))"
  echo "drafter $([[ "$DRAFTER" == 1 ]] && echo "$DRAFTER_DIR" || echo off)"; echo "levers $ZIG_LEVERS ${ZIG_ENV:-}"; echo "rank0 $(zig_args 0)"
  echo "serve --parallel $PARALLEL --context $ZIG_CONTEXT --max-tokens $MAX_TOKENS $SAMPLING_ARGS $draft_arg"; } > logs/serve-config.txt

log "loading (a few minutes): docker logs -f $ZIG_CONTAINER"
for i in $(seq 1 240); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    log "up after $((i * 5)) s; warm-up request"
    t0=$(date +%s.%N)
    out=$(curl -fsS -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$SERVED_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}],
           \"max_tokens\": 64, \"temperature\": 0, \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}") \
      || die "warm-up request failed: docker logs $ZIG_CONTAINER"
    log "warm-up: $(python3 -c 'import json,sys; j=json.loads(sys.argv[1]); print(j["usage"])' "$out" 2>/dev/null) in $(python3 -c "print(round($(date +%s.%N) - $t0, 1))") s"
    log "serving $SERVED_NAME (Zig engine, TP=2) at http://$(hostname):$PORT/v1"
    exit 0
  fi
  [[ -f "$WORK_ROOT/mem_guard.tripped" ]] && die "the memory guard removed the serve: $(tail -3 "$WORK_ROOT/mem_guard-$ZIG_CONTAINER.log" 2>/dev/null)"
  ssh "$WORKER" "test -f ~/muse/mem_guard.tripped" </dev/null && { ./stop.sh >/dev/null 2>&1; die "the worker's memory guard removed rank 1"; }
  docker inspect -f '{{.State.Running}}' "$ZIG_CONTAINER" 2>/dev/null | grep -q true || {
    docker logs --tail 30 "$ZIG_CONTAINER"; die "rank 0 exited"; }
  if (( i % 6 == 0 )); then
    ssh "$WORKER" "docker inspect -f '{{.State.Running}}' $ZIG_CONTAINER-follow 2>/dev/null" </dev/null | grep -q true || {
      ssh "$WORKER" "docker logs --tail 30 $ZIG_CONTAINER-follow" </dev/null; die "rank 1 exited"; }
  fi
  sleep 5
done
die "not up after 20 min: docker logs $ZIG_CONTAINER"
