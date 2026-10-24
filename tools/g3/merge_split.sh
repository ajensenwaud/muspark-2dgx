#!/usr/bin/env bash
# Split conversion, merge on the first node: one pack from the two halves of a split conversion.
#   first node:  $WORK_ROOT/quant-work/<bits>/qtensors          embed_tokens, embed_norm, layers 0..L-1 (scripts/quantise.sh,
#                                                              stopped after layer L-1 by tools/g3/stop_after_layer.sh)
#   second node: $WORK_ROOT/quant-work/split-<bits>/from-second-node/   layers L..61, norm, lm_head + its ckpt/job.json, copied
#                                                              here from the second node (scripts/quantise_split.sh)
# Builds $WORK_ROOT/quant-work/<bits>-merged (hard links, no copies) holding every module's qtensors and a job state
# past the last module, then resumes the converter on it: it skips every module and compiles the pack into the usual
# output dir ($MODELS_ROOT/Muse-Spark-Open-exl3-<bits>bpw). Verifies the pack (tools/exl3/verify_exl3_pack.py).
#   tools/g3/merge_split.sh <bits> <L>
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/config.sh
bits="${1:?usage: merge_split.sh <bits> <L>}"; L="${2:?}"
TAG="${TAG:?set TAG to the muse-tf image tag used for the split}"
a="${MERGE_A:-$WORK_ROOT/quant-work/$bits}"
b="${MERGE_B:-$WORK_ROOT/quant-work/split-$bits/from-second-node}"
m="${MERGE_DIR:-$WORK_ROOT/quant-work/$bits-merged}"
out="${PACK_OUT:-$MODELS_ROOT/Muse-Spark-Open-exl3-${bits}bpw}"
lock="$WORK_ROOT/gpu.lock"
cache="$WORK_ROOT/scratch/home"
P=model.language_model
read -r NUM_LAYERS NUM_EXPERTS < <(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["text_config"]
print(c["num_hidden_layers"], c["num_local_experts"])' "$SRC_DIR/config.json")
PER_LAYER=$(( 5 + 2 + 3 * NUM_EXPERTS ))              # q k v o gate, latent in/out, experts gate/up/down
LAST=$(( 2 + NUM_LAYERS + 2 ))                        # module count: embed, embed_norm, layers, norm, lm_head
EXPECT=$(( NUM_LAYERS * PER_LAYER + 1 ))              # Muse Spark Open: 62 x 775 + 1 = 48,051 EXL3 matrices

docker ps --format '{{.Names}}' | grep -qx "muse-quant-$bits" && die "muse-quant-$bits is still running"
[[ -e "$m" ]] && die "$m exists already"
[[ -z "$(ls -A "$out" 2>/dev/null)" ]] || die "$out is not empty"
files_a=("$P.embed_tokens" "$P.embed_tokens.embed_norm"); for l in $(seq 0 $((L - 1))); do files_a+=("$P.layers.$l"); done
files_b=(); for l in $(seq "$L" $((NUM_LAYERS - 1))); do files_b+=("$P.layers.$l"); done; files_b+=("$P.norm" lm_head)

# EXL3 matrices per module file (header reads): PER_LAYER per layer, 1 for lm_head, 0 otherwise
count() { python3 -c 'import json,struct,sys
f=open(sys.argv[1],"rb"); n=struct.unpack("<Q",f.read(8))[0]; h=json.loads(f.read(n))
import os; end=max([v["data_offsets"][1] for k,v in h.items() if k!="__metadata__"] or [0])
assert os.path.getsize(sys.argv[1])==8+n+end, "truncated"
print(sum(k.endswith(".trellis") for k in h))' "$1"; }
check() {   # dir name
  local d="$1" f="$2" want=0 got
  [[ -f "$d/qtensors/$f.safetensors" ]] || die "missing $d/qtensors/$f.safetensors"
  [[ "$f" == "$P.layers."* ]] && want=$PER_LAYER
  [[ "$f" == lm_head ]] && want=1
  got=$(count "$d/qtensors/$f.safetensors") || die "$d/qtensors/$f.safetensors: unreadable or truncated"
  (( got == want )) || die "$d/qtensors/$f.safetensors: $got EXL3 matrices, expected $want"
}
for f in "${files_a[@]}"; do check "$a" "$f"; done
for f in "${files_b[@]}"; do check "$b" "$f"; done
log "inputs ok: ${#files_a[@]} module files from $a, ${#files_b[@]} from $b"

if [[ "${MERGE_LOCK:-1}" == 1 ]]; then    # MERGE_LOCK=0: tests beside a job that holds the lock
  ( set -o noclobber; echo "G3 merge split $bits $(date -Is)" > "$lock" ) 2>/dev/null || die "GPU locked: $(cat "$lock")"
  trap 'rm -f "$lock"' EXIT
fi
mkdir -p "$m/qtensors" "$m/ckpt" "$out" "$cache"
for f in "${files_a[@]}"; do ln "$a/qtensors/$f.safetensors" "$m/qtensors/"; done
for f in "${files_b[@]}"; do ln "$b/qtensors/$f.safetensors" "$m/qtensors/"; done
cp "$a/args.json" "$m/args.json"
python3 - "$a/ckpt/job.json" "$b/ckpt/job.json" "$m/ckpt/job.json" "$LAST" <<'EOF'
import json, sys
ja, jb = (json.load(open(p)) for p in sys.argv[1:3])
last = int(sys.argv[4])
if jb["next_module_idx"] != last:   # checkpoints are time-gated (-cpi); the qtensors files were checked above
    print(f" !! second node's last checkpoint is at module {jb['next_module_idx']} (not {last}); bad_rows from it")
json.dump({"next_module_idx": last, "q_strategy": None,
           "bad_rows": sorted(set(ja.get("bad_rows") or []) | set(jb.get("bad_rows") or []))},
          open(sys.argv[3], "w"), indent = 4)
EOF
# The resumed job loads a state it never uses (no module is left): a one-element placeholder
docker run --rm --user "$(id -u):$(id -g)" -v "$m/ckpt:/ck" --entrypoint python3 "$IMAGE:$TAG" -c '
import torch; from exllamav3.loader.safetensors_alt import save_file
save_file({"tensor.0": torch.zeros(1)}, "/ck/state.safetensors")
save_file({"tensor.0": torch.zeros((1, 1), dtype = torch.long)}, "/ck/original_input_ids.safetensors")'
log "merged work dir $m: $(ls "$m/qtensors" | wc -l) module files; compiling into $out"
docker run --rm --name "muse-merge-$bits" --gpus all --ipc=host --shm-size 16g --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -v "$SRC_DIR:/src:ro" -v "$m:/work" -v "$out:/out" -v "$cache:/cache" \
  --entrypoint env "$IMAGE:$TAG" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /tools/convert.py \
  -i /src -w /work -o /out -r            # every other setting comes from the job's args.json
[[ -f "$out/quantization_config.json" ]] || die "compile did not write $out/quantization_config.json"
python3 tools/exl3/verify_exl3_pack.py "$out" "$EXPECT"
log "pack $out: $(du -sh "$out" | cut -f1), verified ($EXPECT EXL3 matrices)"
