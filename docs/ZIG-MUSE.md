# The `muse_spark` family in TensorFold's Zig engine

TensorFold has no Python family for Muse Spark, so this one is written directly in Zig and CUDA and checked against
exllamav3's forward on the same EXL3 pack (exllamav3's `muse_spark` architecture, added by `patches-exl3/`). The bar
is exllamav3's own noise, not bit-exactness, because no bit-exact reference exists: logits KL against exllamav3's
prefix-vs-full noise floor, true-token NLL, greedy continuations, and per-layer errors relative to each layer's own
update.

## Engine base and build

- TensorFold v1.0.11 (commit `3a6d1f14`, Zig 0.17.0) plus `patches-zig/` 0001-0060. The patched tree is `bf62c813`
  (`ZIG_TREE` in `scripts/config.sh`).
- 0001-0037: generic CUDA infrastructure (keyed graph cache, device arena, tensor parallelism across processes with
  NCCL and a RoCE mailbox all-gather, a plan link between ranks, fail-fast, session memory, round planner, server
  drain and SSE keepalives, family chat-rule hooks) and the MiMo-V2 family, whose EXL3 pack helpers, GEMV and grouped
  expert kernels and DFlash kernels the Muse family reuses. Authors and licences: `CREDITS.md`. None of these
  commits is in upstream v1.0.11.
- 0038-0060: the Muse family and the server changes it needed (recipient replies, generation_config stop tokens,
  the family's own prompt-cache hooks, the keyed sampler fix).

`scripts/zig_prepare.sh` (run by `scripts/prepare.sh`) clones the tag, applies the patches with `git am`, refuses
unless the tree hash equals `ZIG_TREE`, builds the build image (`BASE_IMAGE` plus Zig, `zig/docker/Dockerfile.build`)
if it is missing, then compiles in a container capped at `ZIG_BUILD_MEM` (8 GiB) and `ZIG_BUILD_JOBS` CPUs:

```bash
zig build        -j4 -Dnvcc=$(which nvcc) -Dtp-nvcc=$(which nvcc) -Dsm=121 --prefix zig-out
zig build native -j4 -Dnvcc=$(which nvcc) -Dtp-nvcc=$(which nvcc) -Dsm=121 --prefix zig-out
```

It checks for `bin/tensorfold`, `bin/tf-muse`, `bin/tf-mimo`, `bin/tf-cuda-test`, `bin/tf-tp-test` (with its mailbox
kernel) and `native/bin/tensorfold-native`, writes `zig-out/SHA256SUMS`, and copies `zig-out/` to the worker at the
same path (`NO_WORKER=1` skips the copy). The cap is there so a build can run beside a serve. Build time is a few
minutes.

Checks for the base:

- Host tests (no GPU), in the build image: `zig build test` on v1.0.11 with 0001-0037 passed 871 of 877 tests (6
  skipped); `zig build test-muse` runs the family's own host tests (config parsing, TP cuts, prompt-cache chunk
  cuts, slot choice, plan words).
- The two-node fabric: `tf-tp-test` through the `run_hosts.sh` runner described in `zig/src/fabric/tp/README.md` of
  the patched tree. On two Sparks with `--network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit
  memlock=-1`, `TF_TP_ROCE_HCA=rocep1s0f0,roceP2p1s0f0`, `NCCL_IB_GID_INDEX=3`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`:
  1,000 RoCE gathers of 1 B to 256 KiB bit-exact, the RoCE suite 378/378 and the NCCL suite 189/189 on each rank,
  and every injected failure (fatal or kill, either rank) ends the other rank with exit 70.

## The model

Muse Spark Open's text model: 62 layers, 16 full attention (no RoPE) and 46 sliding-window (2,047 keys, RoPE),
head_dim 64, 128 query heads and 16 KV heads (GQA 8:1), a sigmoid output gate per attention, a latent MoE (8192 ->
4096 before the experts, back after), 8 of the routed experts per token with a sigmoid router plus score bias, a
norm per expert output, gated residuals, and a softcapped head (202,048 tokens, padded to 202,112 rows).
`config.zig` refuses configs outside what the kernels handle: dense MLPs, shared experts, attention bias, tied
embeddings, head_dim other than 64, GQA other than 8:1, non-default RoPE, codebooks other than mul1.

### Files (`zig/src/families/muse_spark`, `zig/kernels/cuda/muse_spark`)

| File | What it does |
|---|---|
| `config.zig` | Parses `text_config`; a rank's view; the TP=2 cuts (`layerCuts`, `headCut`) |
| `load.zig` | Loads a rank's share of the pack: EXL3 matrices through the MiMo-V2 pack helpers; norm weights as `1 + w` (centred and input-scaled norms) or `w` (final norm); residual gates as fp32 `alpha, beta`; the fp32 router kept as bf16 on the GPU when every value is exactly a bf16 (halves its reads) |
| `forward.zig` | The decode forward for a window of rows over any slots and positions, eager or graphed, with capture hooks for per-layer checks and teacher forcing |
| `prompt.zig` | The prompt path (below) |
| `engine.zig` | The model, its kernels and TP session; the plan link's words (`hello`, `forward`, `prompt`, `cache`, `stop`); rank 0 sends each window to rank 1 before launching it |
| `native.zig` | `model_type = "muse_spark"` behind `tensorfold-native serve`: the lanes backend, keyed sampling, device argmax when greedy, costs timed at load, per-rank weight admission |
| `kept.zig` | The prompt cache's state copies, slot pins and chunk cuts |
| `dflash2.zig`, `dflash2_config.zig` | The DFlash2 drafter |
| `pair.zig` | A test-only collective that runs both TP ranks as two threads on one GPU |
| `m1_main.zig` and the other `*_main.zig` | `tf-muse`: test drivers (`m1`, `prompt`, `dflash`, `drafted`, `timing`) and `follow` (rank 1 behind the server) |
| `glue.cu` | Embedding norm, RMS norms, residual gate + post norm + next pre-norm in one kernel, fp32 router, sigmoid + bias top-8, per-expert norm and weighted sum, softcap head, argmax, RoPE tables. Built without FMA contraction and with IEEE division |
| `attention.cu` | QK norm + NeoX RoPE + cache write; decode attention partials; chunk merge + `o_norm` + sigmoid gate; the 8-bit cache's store and load |
| `dflash2.cu` | The drafter's selector projection, causal convolutions, SiLU-mul, per-rank top 16 and the candidate walk |

Build wiring: `zig/build/cuda.zig` (the fatbins, `tf-muse`, `zig build test-muse`) and `cuda/kernels.zig`.
`server/recipient_reply.zig` handles Muse's reply format (below).

### Why hand-written CUDA for attention

TensorFold's AOT Triton route exists so that a Zig port can load a Python family's own cubins and match it bit for
bit. Muse has no Python family, so that gain is absent, and Triton would add a second toolchain for kernels that are
short at Muse's shapes. Head_dim 64 maps exactly onto a warp (a lane holds dims `l` and `l + 32`, which are also
NeoX's rotation pairs) and GQA 8:1 onto 8 warps sharing one K/V tile. The decode attention kernel is one block per
(row, KV head, key chunk): eight query heads as warps over shared K/V tiles of 32 keys (K rows padded to 33 words for
conflict-free dot products), online softmax in fp32, and a second kernel that merges the chunks and applies `o_norm`
and the gate. Everything builds with the engine's normal `zig build`, and graph capture is plain.

### Numerics: following the reference's roundings

| Step | exllamav3 (`muse_spark`) | Zig |
|---|---|---|
| Embedding | bf16 row -> fp32, weightless RMS (eps 1e-5), fp32 residual | same |
| Pre norms | centred RMS `x * rsqrt(mean x^2 + eps) * (1 + w)` -> fp16 | same, fp32 inside |
| q, k, v, gate | EXL3 -> fp16 | EXL3 GEMV -> fp16 |
| QK norm, RoPE | weightless per-head RMS + NeoX RoPE fused, one fp16 rounding (RoPE layers); norm -> fp16 (no-RoPE layers) | same in one kernel; RoPE tables in fp64 then fp32 |
| Attention | flash attention, window (2047, 2047) on sliding layers, `sm_scale = 5.473 / 8` | fp32 online softmax, keys `[pos - 2047, pos]` or `[0, pos]`, fp16 out |
| o_norm, gate | weightless RMS per head -> fp16; sliding: `o * fp16(sigmoid(g))`; full: `o * sigmoid(g)` | same, per layer type |
| o_proj | EXL3 -> fp32 | same |
| Post-attention | weightless RMS (eps 1e-8) fp32; `x = x * alpha + y * beta` | fused with the pre-ffn norm |
| Router | `F.linear(x.float(), W_fp32) * 0.1768` on the fp32 normed input, sigmoid, + bias for choice, top 8, `/ (sum + 1e-15)` | same in fp32 on the unrounded norm output; picks in increasing expert id |
| Latent | pre_expert_proj -> fp16, centred RMS -> fp16 | same |
| Experts | gate, up -> fp16; `fp16(silu(g) * u)`; down -> fp32; input-scaled norm `bf16(y * (1 + w))`, RMS eps 1e-8; `* weight`, summed in expert order | grouped EXL3 experts with an fp32 SwiGLU epilogue (no fp16 rounding of g, u); then one kernel: per slot the input-scaled norm, the weight, the sum in expert order -> fp16 |
| post_expert_proj | EXL3 -> fp32 | same |
| Post-ffn | input-scaled RMS (bf16 rounding, eps 1e-8), gate | fused with the next layer's input norm |
| Head | final RMS (w) -> fp16, EXL3 head fp16, `* 0.1768`, softcap 20 in fp16 | EXL3 head fp32, `* 0.1768`, `20 tanh(x / 20)` in fp32 (one rounding fewer) |

## Tensor parallelism across the two Sparks

Cuts (`config.layerCuts`): q, gate and o by query heads (64 a rank), k and v by KV heads (8 a rank), pre_expert_proj
by latent outputs (2,048 a rank), post_expert_proj by latent inputs, experts by width (2,048 a rank: gate and up by
outputs, down by inputs), the head in 128-row blocks (790 and 789 blocks, so the two slices differ by one block).
Embedding, norms, gates and router are replicated.

Exchanges per layer (fp32 unless noted; per row): o_proj partials (32 KiB), the latent halves (4 KiB fp16), expert
down partials (8 slots x 4096, 128 KiB; the per-expert norm needs each expert's whole output, so this cannot be
folded into the residual sum), post_expert_proj partials (32 KiB); plus the head's softcapped slices for the rows
whose logits are wanted. All are all-gathers. Every sum of partials is taken in rank order inside the kernel that
consumes it, on both ranks, so both ranks hold identical residuals and route identically without a broadcast.

The latent projections are split rather than replicated because replicating pre_expert_proj would add about 1 GB of
reads per token per rank (about 4.7 ms) to save one 4 KiB gather a layer (0.6-0.9 ms a token). The router stays
replicated: splitting it saves about 0.6 ms a token for one more gather a layer. The transport is `TF_COMM_BACKEND`
(`ZIG_COMM`): `roce` (the one-shot RoCE mailbox, the default) or `nccl`; both give identical bytes.

## Decode graphs and window invariance

`Model.graphed` captures a CUDA graph per (rows in the window, key bucket); the release sets one bucket
(`TF_MUSE_GRAPH_BUCKET=1048576`), so one graph a width. A full layer's key chunk width depends only on each row's own
position (256 keys up to a 16K bucket, then bucket / 64 rounded up to 256), and empty chunks write `max = -inf,
sum = 0` and are skipped by the merge. So a row's bits do not depend on its window, its neighbours or the bucket:
graphed logits are byte-equal to eager ones, and windows of 8 rows x 2 slots or 32 rows x 4 slots give the same
logits as one row at a time. The lanes backend relies on this (`exact_width = 8`), and so do the prompt path, the
drafter's verify windows and the prompt cache.

## Prompt path

`TF_MUSE_PROMPT=1` (on in the release) runs prompts of at least `TF_MUSE_PROMPT_MIN` rows (33) as chunks of
`TF_MUSE_PROMPT_ROWS` (1,024) rows: wide windows for the dense layers, experts in grouped windows of
`TF_MUSE_PROMPT_EWIN` (512) rows, and the head on the last row only. It gives the same bits as decode rows, which the
prompt cache depends on. `TF_MUSE_PROMPT_EXACT=0` uses unsplit gate/up projections; its prompt rows differ from
decode rows, so the prompt cache refuses to start with it. Shorter prompts, and every prompt without the switch, run
as decode windows of up to 128 rows.

Full-attention layers still read keys with the decode-row kernel, so cold prompt reading slows with depth: 224
tok/s at 4K tokens, 179 at 31K, 145 at 61K, 108 at 122K, 70 at 244K, 41 at 488K. A flash-style prompt attention
kernel is planned for v1.1.

## DFlash2 drafter

RadixArk's Muse-Spark-Open-DFlash2 (bf16, 9.8 GB) is quantised to a 4.0 bpw EXL3 pack by
`scripts/quantise_drafter.sh` (46 EXL3 matrices plus the selector's fp16 projection, uncalibrated; about 2.5 GB,
1.52 GiB a rank at TP=2). The semantics follow the trainer's reference, z-lab/dflash `dflash/model.py` @ 07ebd93d
(MIT, vendored as `tools/g6/dflash_ref_model.py`); exllamav3's `architecture/dflash2.py` agrees.

- Inputs: the target's residual after layers 1, 16, 30, 44 and 59, concatenated (5 x 8192), through `fc` and
  `hidden_norm`. Each context row is projected once into every drafter layer's K and V (k_norm, RoPE at the row's
  position) and cached; the cache keeps accepted rows only.
- A draft is a block of 8 rows: the pending token, then 7 copies of mask token 200002, as raw target embedding rows.
- 5 Qwen3-style layers, each wrapping attention and MLP in grouped dynamic causal convolutions over the block's rows
  (kernel 2, groups of 16 channels): a base kernel plus a per-row dynamic weight from one `kernel_projection`.
- Attention: 64 query and 8 KV heads of 128, q_norm and k_norm, NeoX RoPE, not causal inside the block; context keys
  within 2,048 positions.
- Output: final norm, rows 1-7 through the target's head (`* 0.1768`, softcap 20), top 16 per row, then the
  candidate selector walks the rows greedily with the predecessor and successor codebooks. The engine drafts greedily
  and verifies with the target's own sampling (accept while the drafted token equals the token the target samples),
  so the output distribution is exact; at temperature 1.0 that costs some acceptance against the reference's
  rejection sampling.
- TP: split like the target (heads, KV heads, MLP width, `fc` outputs; norms, conv kernels, selector and codebooks
  replicated): about 1.15 GB of reads a rank a pass instead of 2.2 GB replicated, for 11 small all-gathers (about
  0.15 ms). Draft passes are captured as CUDA graphs per stream count, up to 16 streams x 8 rows.
- On: `TF_MUSE_DRAFTER=<drafter pack>` on every rank and drafts on in the server; `start.sh` does both when
  `DRAFTER=1` (the default).

Measured on the full 2.0 bpw development pack at TP=2, greedy, 5 prompts x 128 steps: mean accepted tokens of 7 were 3.9 (coding), 3.2
(code edit), 5.0 (reasoning), 0.8 (prose) and 2.9 (tool calls). Against the reference DFlash2 model on the same taps
the port matched (3.155 vs 3.158 with the EXL3 weights, first-draft agreement 1.000), and the 4-bit drafter cost
nothing measurable against the bf16 weights (3.155 vs 3.122). Drafted replies are byte-identical to plain ones.

## Serving

- `./start.sh` starts `tf-muse follow` (rank 1) on the worker, then `tensorfold-native serve` (rank 0) on the head
  (API on `PORT`, rendezvous on `ZIG_TP_PORT`), waits for `/health`, sends a warm-up request and writes
  `logs/serve-config.txt`. `./stop.sh` saves both ranks' logs to `logs/`.
- Chat: Muse's template (`<|start|>role<|message|>...<|eot|>`, `<|eom|>` after a tool call) runs in the server's
  template engine. Stop tokens are config.json's EOS plus `generation_config.json`'s `eos_token_id` list (which
  includes `<|eot|>`).
- Replies are messages to recipients (`to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>...`).
  `server/recipient_reply.zig` maps `self` to reasoning, `user` and `commentary` to the answer, and any other
  recipient to a tool call: its `<atem:function_calls>` block becomes OpenAI `tool_calls` with schema-typed
  arguments. Streaming holds back unfinished headers and call messages.
- Streams share forwards: the family reports width 8 with or without the drafter, so up to 8 rows of different
  streams run in one window.
- Sampling: the keyed sampler (patch 0060) no longer sorts the whole vocabulary when top_k and top_p cut nothing; it
  makes one pass over the row and draws the same token as the sorted path. At temperature 1.0 with top_k off this took
  one-stream decode from 14.7 to 23.2 tok/s with the drafter and from 13.3 to 18.2 without it.

## Prompt cache

Every chat or agent turn re-sends the whole conversation. Without a cache the server re-reads all of it (a 22.7K-token
history takes about 2 minutes); with it, a turn resumes from the longest kept prefix and reads only what is new.
`TF_MUSE_PROMPT_CACHE=1` on rank 0 turns it on (`PROMPT_CACHE=1`, the release default); without it the serve takes
the plain path.

### What is kept

The server's store (upstream's `core/prompt_cache.zig`) decides where states are kept: at the conversation's history
(everything before the assistant's generation header), before the last user message, at shared system blocks, and at
the end of a background warm pass that re-reads the reply after each chat turn while the server is idle. The family
keeps the state at each mark through the `snap_*` hooks:

| Part of a state after `at` tokens | Where it lives | Bytes a rank |
|---|---|---|
| 16 full-attention layers, K/V rows 0..at-1 | in place in the stream's slot: pinned, never copied | none extra |
| 46 sliding layers, each K/V ring (2,176 rows: the 2,047-key window + the row + a 128-row window) | a device buffer of the state | 195.5 MiB |
| The drafter's context ring | the same buffer | 21.3 MiB (with the drafter) |

A kept state costs about 0.2 GiB a rank whatever its length. `TF_MUSE_PROMPT_CACHE_GIB` caps the store (engine default
2; the release sets 1 through `PROMPT_CACHE_GIB`). A state stays alive while no later pass on its slot rewrites rows
under `at`: the next turn of the same conversation writes only past `at` and keeps it; another conversation that
takes the slot ends the states there, and a request that would have used them runs fresh.

### Slots and resumes

- A new stream takes a free slot that no live state pins; when all are pinned, the one whose newest state was used
  longest ago. With no more conversations than slots, nothing is evicted by slot reuse.
- A resume takes the state's own slot when it is free: the windows are copied back (about 0.2 GiB on the device) and
  the pass starts at `at`. When that slot is busy (two requests of one conversation at once, an agent fanning out, a
  shared system block) it takes another free slot and copies the full-layer rows 0..at-1 there first (128K tokens is
  about 4 GiB a rank, tens of ms).
- Prefill cuts its chunks at the store's marks so the pass stands exactly at each mark when the state is saved.
  Background warm passes give way when a request arrives.

### Why a resumed reply equals a fresh one

A row's bits do not depend on how a pass is cut (window invariance above, and the prompt path's bits equal decode
rows'). So the state at `at` is the same whether one pass built it from 0, several chunks cut at a mark, or decoded
reply rows, and a pass resumed there continues with the same bits as a fresh one. The first sampled token is drawn at
the prompt's end with the keyed sampler, as in a fresh pass.

### TP=2 and the server hook

Rank 0 drives everything: each cache step (save, restore, copy prefix, drop) goes to rank 1 over the plan link before
rank 0 runs it, and rank 1 replays it on its own stream in plan order with its own copies (each rank holds its half
of the KV heads). The native server used to build the store from Nemotron's `snap_*` declarations only; patch 0058
lets any family supply them, with optional `snap_budget`, `snap_lookahead` and `snap_ready`. Nemotron's behaviour is
unchanged.

### Measured

On P1.8s at TP=2 (4 slots x 32K, fp16 KV cache), 115 requests (multi-turn chats on the model's own replies, a
sampled conversation, an agent session whose tool outputs grow the history, a 22.7K-token document conversation,
four conversations at once, four resumes of one history at once, six conversations round robin on four slots, and
resume points across the sliding window's edge, the ring, prompt chunks and decode windows): every reply
byte-identical to a cache-off serve, with and without the drafter; 63 requests resumed, 155,416 of 285,822 prompt
tokens served from the cache. Follow-up turns of the document conversation started in 0.29 s instead of 119.5 s.

## Memory

Each Spark has 121.6 GiB shared by CPU and GPU, and an oversized allocation takes the host down instead of failing,
so memory is planned before loading (`tools/mem/estimate.py`, called by `start.sh`) and watched while serving
(`tools/mem/mem_guard.sh`). `MEM_FIXED_GIB` (9.2) is the measured per-rank rest beyond weights, drafter and KV cache.

8-bit KV cache (`TF_MUSE_KV8=1`, `KV8=1` in the release): the 16 full-attention layers store int8 values with an
fp32 scale per token and KV head (absmax / 127), dequantised per tile inside the attention kernel; about half the
bytes of fp16, roughly 17 GiB a Spark for 1M tokens. The sliding layers keep fp16 rings. Unset, the cache is fp16.
`TF_MUSE_KV8=2` selects e4m3 and is untested. The engine's start-up line says "e4m3" for mode 1 too; it is stale.

Memory levers in the release, each byte-identical to the serve without it:

| Lever | Effect a rank |
|---|---|
| `TF_MUSE_EMBED_HOST=1` | the embedding is read from the pack's file mapping: 3.3 GB less GPU memory |
| `TF_LOAD_ARENA_MIB=64` | first-fit 64 MiB arena blocks instead of 1 GiB ones: 1.9 GiB of block tails recovered |
| `TF_LOAD_FREE_HOST=1` | the host copy of each uploaded matrix is freed: 4.8 GB of host memory that otherwise sat in swap |
| `TF_MUSE_DROP_CACHE=1` | the pack's pages leave the process and the page cache after each layer uploads |
| `TF_MUSE_GRAPH_BUCKET=1048576` | one decode graph a width instead of one a width and key bucket (about 53 MiB a graph; 32 graphs instead of up to 128 at 32K) |
| `NCCL_MAX_NCHANNELS=4` | NCCL's pinned host buffers from 1.2 GiB to 78 MiB; prefill unchanged |

`TF_MUSE_MEMLOG=1` prints a rank's memory at each load stage and graph capture (arena bytes handed out and held, KV
bytes, process RSS, swap, pinned).

## Environment switches

Engine switches (set on both ranks; `start.sh` passes `ZIG_LEVERS`, the `KV8` / `PROMPT_CACHE` / `DRAFTER`
settings, and anything in `ZIG_ENV`):

| Switch | Engine default | Release | What it does |
|---|---|---|---|
| `TF_MUSE_PROMPT` | 0 | 1 | the prompt path |
| `TF_MUSE_PROMPT_ROWS`, `_EWIN`, `_MIN` | 1024, 512, 33 | same | prompt chunk rows, expert window rows, shortest prompt on the prompt path |
| `TF_MUSE_PROMPT_EXACT` | 1 | 1 | 0: unsplit gate/up on the prompt path (other bits; disables the prompt cache) |
| `TF_MUSE_KV8` | unset (fp16) | 1 | 1: int8 full-layer KV cache; 2: e4m3, untested |
| `TF_MUSE_PROMPT_CACHE`, `_GIB` | 0, 2 | 1, 1 | the prompt cache and its budget a rank |
| `TF_MUSE_DRAFTER` | unset | the drafter pack | turns the drafter on (with drafts on in the server) |
| `TF_MUSE_DEPTH` | unset | unset | `geometric`: a draft depth whose round cost grows with context |
| `TF_MUSE_FAST` | on | on | launch levers (q/k/v/gate in one launch, router blocks, cluster kernels, 32-row passes, 32-member expert tiles), all byte-identical; 0: the original launches |
| `TF_MUSE_OVERLAP` | 0 | 0 | router on a side stream beside the latent gather; measured no gain |
| `TF_MUSE_EMBED_HOST`, `TF_LOAD_ARENA_MIB`, `TF_LOAD_FREE_HOST`, `TF_MUSE_DROP_CACHE`, `TF_MUSE_GRAPH_BUCKET` | off | on | memory levers (above) |
| `TF_MUSE_GRAPHS` | 0 (no limit) | 0 | LRU cap on decode graphs |
| `TF_MUSE_TRIM` | 0 | 0 | `malloc_trim` after load and graph captures |
| `TF_MUSE_MEMLOG` | 0 | 0 | memory ledger in the log |
| `TF_COMM_BACKEND` | | `roce` | `roce` or `nccl` (`ZIG_COMM`) |

Recipe switches for `./start.sh` (defaults in `scripts/config.sh`, overrides in `scripts/local.sh` or the
environment): `PROFILE` (`long` or `multi`), `ZIG_CONTEXT`, `PARALLEL`, `CONTEXT_AUTO`, `KV8`, `PROMPT_CACHE`,
`PROMPT_CACHE_GIB`, `DRAFTER`, `ZIG_LEVERS` (`" "` for none), `ZIG_ENV` (extra `-e` flags), `ZIG_BIN` (another
build), `MEM_MIN_FREE_GIB`, `MEM_FIXED_GIB`, `MEM_CHECK=0` (skip the estimate), `GUARD_MIB`, `GUARD=0` (no in-run
guard), `SAMPLING_ARGS` (the server's defaults when a request omits them: temperature 1.0, top_p 1.0, top_k 64).

## Correctness checks

These checks were run during development and the numbers below come from those runs. The building blocks are generic; the wrapper scripts under
`tools/g5`, `tools/g6` and `tools/pc` were written for the development cluster and assume its paths and a GPU lock
file, so read them as worked examples.

Parity with exllamav3:

1. Answer key: `python3 tools/g6/muse_ref_stream.py MODEL OUT_DIR [--texts tools/g5/muse_texts.json]` in Muspark's
   image on one GPU (about 10 GB; the full pack is streamed module by module because it does not fit one Spark).
   It writes the ids, fp32 logits at every position and the noise floor (forwards over the first T/2 and T-17
   tokens against the full forward). `tools/g5/muse_texts.json` holds English prose, Chinese prose, code and an agent
   turn with a tool call. The long_mixed figure (2,600 tokens, past the sliding window) was measured on an internal
   text that is not shipped; pass any long text of your own with `--texts`.
2. Engine: `tf-muse m1 MODEL IN_DIR OUT_DIR`, teacher-forced decode rows over the same ids, at TP=2 with the
   `TF_TP_*` variables on both nodes (rank 0 and rank 1), or both ranks on one GPU with `MUSE_PAIR=1` for packs that
   fit. Knobs: `MUSE_ROWS`, `MUSE_SLOTS`, `MUSE_LAYERS`, `MUSE_CONTEXT`, `MUSE_GREEDY`, `MUSE_CAPTURE` (per-layer
   dumps), `MUSE_TEACHER`, `MUSE_GRAPHS`.
3. Score: `python3 tools/g5/muse_check.py REF_DIR ZIG_DIR --json out.json` (CPU). Pass: KL within 1.5x of
   exllamav3's floor, NLL within 1.5 %, top-1 at least 0.9.

The shipped 1.8 bpw pack at TP=2:

| Text | Tokens | KL (max) | exllamav3 floor | Ratio | Top-1 | NLL exllamav3 / Zig |
|---|---|---|---|---|---|---|
| English prose | 157 | 0.00066 (0.0068) | 0.00098 | 0.67 | 0.9745 | 1.7108 / 1.7129 (+0.12 %) |
| Chinese prose | 138 | 0.00023 (0.0022) | 0.00019 | 1.24 | 0.9928 | 1.0008 / 1.0010 (+0.02 %) |
| Code | 214 | 0.00039 (0.0151) | 0.00041 | 0.96 | 0.9907 | 0.5744 / 0.5779 (+0.61 %) |
| Agent turn | 239 | 0.00339 (0.2260) | 0.00422 | 0.80 | 0.9623 | 2.3440 / 2.3421 (-0.08 %) |
| Long mixed | 2600 | 0.00361 (0.5163) | 0.00336 | 1.07 | 0.9754 | 2.0569 / 2.0572 (+0.01 %) |

Per-layer errors on a 2-layer pack (one sliding, one full layer) were flat along the 2,600-token text, through the
point where the sliding window starts dropping keys and the ring wraps.

Bit-identity checks (each must be byte-equal):

- Graphs vs eager: `tf-muse m1` with `MUSE_GRAPHS=1` against an eager run of the same texts (9 of 9 cases).
- Prompt path vs decode rows: `tf-muse prompt MODEL IN_DIR OUT_DIR`.
- Drafter: `tf-muse dflash` (the fixed protocol, taps dumped, compared with `tools/g6/dflash_ref.py`) and
  `tf-muse drafted` (drafted vs plain greedy) with `TF_MUSE_DRAFTER` set; on a serve,
  `python3 tools/g6/drafted_exact.py http://127.0.0.1:8888 Muse-Spark-Open out.json --sb <spark-bench checkout>`
  sends spark-bench's 12 decode prompts with and without drafting, serial and 4 at a time (12 of 12 identical).
- Prompt cache: record on a cache-off serve, replay on a cache-on one (and with the drafter):

  ```bash
  PROMPT_CACHE=0 ./start.sh
  python3 tools/pc/pc_contracts.py record --base http://127.0.0.1:8888 --ref pc/ref.jsonl
  ./stop.sh && ./start.sh
  python3 tools/pc/pc_contracts.py replay --base http://127.0.0.1:8888 --ref pc/ref.jsonl --out pc/cache.jsonl
  ```

  The chat template prints the date, so compare runs from the same day only.

Quality on tasks (GSM8K, MMLU-Pro at Meta's sampling): `scripts/eval_data.py` and `scripts/eval_tasks.py`; results
are in the README.

## Pitfalls and known limits

- Routing near-ties: top-8 margins around 1e-5 on about 0.2 % of tokens, so a per-layer comparison on deep packs sees
  a few tokens routed differently from exllamav3. `muse_check.py` reports per-token medians, p99 and outliers, and the
  error without them, to tell flips from a wiring bug.
- NCCL refuses two ranks on one GPU, so the two-rank arithmetic is checked on one GPU with `MUSE_PAIR=1`.
- exllamav3 computes the head and softcap in fp16; the engine uses fp32. The gap is far below the noise floor, but
  exllamav3's logits move in fp16 steps.
- The padded head (202,112 rows) is not a multiple of 256, so the vocabulary split is uneven and the gather pads rank
  1's slice.
- The router reads the fp32 normed input. exllamav3 cast it to fp16 before `patches-exl3/` 0004; answer keys made
  with an older image do not match.
- A NaN row from the head could leave the drafter's top-16 ids at their sentinel; patch 0049 skips NaN logits.
- Prompt cache: with more live conversations than slots, states end when another conversation takes their slot and
  the hit rate falls. Ended states keep their budget charge until the store drops them, which can evict a live state
  early under heavy churn (fewer hits, never wrong replies). A rank-1 allocation failure while saving a state stops the
  serve; the budget is part of `MEM_FIXED_GIB`. The warm pass adds prefill work on a busy serve. States are allocated
  with `cuMemAlloc` per state (no pool). Keeping states in host RAM gains nothing on GB10, where device memory is host
  memory.
- Cold prompts are slow at depth (above). Greedy decoding can loop on this pack: use temperature 1.0.
