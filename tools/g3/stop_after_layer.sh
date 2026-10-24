#!/usr/bin/env bash
# Split conversion, first node: end the running single-node conversion once it has finished layer L-1, so the second
# node's layers L..61 are used instead. Polls the job's checkpoint; when ckpt/job.json says the next module is layer L
# (module index 2 + L), the qtensors file of layer L-1 is complete and the log shows the job has moved on to loading
# layer L (so the checkpoint rename is over), it stops the container by name. The checkpoint after layer L-1 stays
# intact: `scripts/quantise.sh <bits>` resumes from it if the split has to be abandoned.
#
# Safety: it only stops the job when the go-flag file exists (created by hand once the second node's conversion has
# finished its first layer cleanly). Without the flag it logs and lets the single-node job carry on.
#   setsid nohup tools/g3/stop_after_layer.sh <bits> <L> > logs/stop-after.log 2>&1 </dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/config.sh
bits="${1:?usage: stop_after_layer.sh <bits> <L>}"; L="${2:?}"
work="${WORK:-$WORK_ROOT/quant-work/$bits}"
qlog="${QLOG:-$WORK_ROOT/logs/quantise-$bits.log}"
ctr="muse-quant-$bits"
go="${GO:-$WORK_ROOT/quant-work/split-$bits.go}"
want=$((2 + L))                      # embed_tokens, embed_norm, then layer 0 at module index 2
qt="$work/qtensors/model.language_model.layers.$((L - 1)).safetensors"
POLL="${POLL:-240}"
log "watching $ctr: stop after layer $((L - 1)) (checkpoint next_module_idx $want) if $go exists"
warned=0
while true; do
  docker ps --format '{{.Names}}' | grep -qx "$ctr" || { log "container $ctr is not running; nothing to do"; exit 0; }
  idx=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_module_idx"])' "$work/ckpt/job.json" 2>/dev/null || echo -1)
  if (( idx >= want )) && [[ -f "$qt" ]] && grep -aqE "Loading unquantized module: model\.language_model\.layers\.$L[[:space:]]*$" "$qlog"; then
    if [[ ! -f "$go" ]]; then
      (( warned )) || log "layer $((L - 1)) done (next module $idx) but no $go: leaving the single-node job running"
      warned=1
    else
      st=$(stat -c %s "$work/ckpt/state.safetensors")
      log "layer $((L - 1)) done: next module $idx, $(basename "$qt") $(stat -c %s "$qt") bytes, ckpt state $st bytes"
      docker stop -t 30 "$ctr" >/dev/null
      log "stopped $ctr; checkpoint kept at next module $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_module_idx"])' "$work/ckpt/job.json")"
      exit 0
    fi
  fi
  sleep "$POLL"
done
