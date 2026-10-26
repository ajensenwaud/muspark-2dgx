# Changelog

## v1.1 (October 2026)

Engine move: Muspark now runs on TensorFold's main line instead of a 60-patch v1.0.11. The pack does not change.

- Engine: TensorFold main at `40c6406b` plus `patches-zig/` 0001-0005 (tree `71a9f224`, checked by
  `scripts/zig_prepare.sh`, which now builds from a pinned commit): the Muse family on TensorFold's shared EXL3
  operators and tensor parallelism with its kernel-contract registrations (upstream pull request #1295), and the
  server's recipient-reply split (#1336). The sampler fix, the tensor-parallel and RoCE work and the other generic
  patches v1.0 carried are in main.
- Same replies: logits byte-equal to the v1.0 engine with an fp16 KV cache (decode rows, drafted windows, prompt
  path); drafted == plain on 12 of 12 prompts; resumed == fresh on 115 of 115 requests (49 resumed).
- KV cache fp16 only: main takes no lossy KV cache, so the 8-bit KV cache and `KV8` are gone. Context: `PROFILE=long`
  ~450K-470K tokens for one stream (was ~905K), `PROFILE=multi` 4 x ~110K (was 4 x 192K).
- Speed (two GB10s, greedy, 512-token replies): 1 stream 30.1 tok/s (v1.0: 31.6), 4 streams 43.1 tok/s total (v1.0:
  50.9); cold prompt 209.0 tok/s at 3.9K tokens and 196.4 at 15.4K (v1.0: 224.0 and 200.4).
- Settings: the drafter and the prompt cache are serve flags (`--drafter`, `--prompt-cache-gib`); the prompt cache is
  on by default at 1 GiB a Spark with a 3 GiB reserve (`PROMPT_CACHE=0` turns it off, `PROMPT_CACHE_GIB=N` sets
  another budget). The `TF_MUSE_*` switches are gone: the levers v1.0 switched on are built in. `ZIG_LEVERS` keeps
  only `NCCL_MAX_NCHANNELS=4`.
- `start.sh` prints the engine's start-up line (memory, streams, context, prompt cache).

Known limits: as v1.0 (cold prompts slow at depth, greedy can loop, text only), plus the smaller context above.

## v1.0 (October 2026)

First release.

- Pack: P1.8s (`Muse-Spark-Open-exl3-1.8s`), published as
  [ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw](https://huggingface.co/ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw)
  at the revision pinned in `scripts/config.sh` (`PACK_REV`). Experts at 1.80 bpw on average, set per tensor by
  measured sensitivity; dense weights at 4.0, the head at 6; 182 GiB.
- Engine: TensorFold Zig v1.0.11 plus `patches-zig/` 0001-0060 (tree `bf62c813`, checked by `scripts/zig_prepare.sh`).
- Serve profiles: `PROFILE=long` (the default: 1 slot, the longest context that fits, about 905K tokens on two idle
  Sparks) and `PROFILE=multi` (4 slots x 196,608 tokens). Both serve the DFlash2 drafter.
- 8-bit KV cache for the 16 full-attention layers: int8 with an fp32 scale per token and KV head, about half the bytes
  of fp16 (`KV8=1`, the default). The engine's start-up line calls it e4m3; that is mode 2, which the release does not
  use.
- Prompt cache: a follow-up turn reuses the kept state of the conversation and reads only the new tokens
  (`PROMPT_CACHE=1`, the default; 1 GiB a Spark).
- Memory guard: `start.sh` estimates each Spark's memory before loading and refuses, or with `CONTEXT_AUTO=1` shrinks,
  a context that would leave less than 3 GiB free; a watchdog on each node stops both ranks under 512 MiB free.
- Sampler fix (patch 0060): at temperature 1.0 with top_k off the server no longer sorts the whole vocabulary for
  every sampled token. One-stream decode with the drafter went from 14.7 to 23.2 tok/s; the tokens are the same.
- `scripts/prepare.sh` (images, engine, drafter) and `scripts/fetch_pack.sh` (the pack from Hugging Face).
- API on port 8888.
- Node names and addresses have no defaults: set them in `scripts/local.sh` (see `scripts/local.sh.example`).

Known limits:

- Cold prompts are slow: 145 tok/s at 61K tokens, falling to 41 tok/s at 488K (3.3 h to the first token). A faster
  prompt-attention kernel is planned for v1.1 (engine only; the pack does not change).
- Greedy decoding can loop on this pack; use temperature 1.0 (the server's default).
- Text only.

## Before v1 (development)

- A release candidate ran a 2.0 bpw pack (P2.0) with 4 slots x 32,768 tokens and an fp16 KV cache. P1.8s replaced it
  for memory: P2.0 holds about 101 GiB a Spark and leaves no room for long context.
