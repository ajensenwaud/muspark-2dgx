# Changelog

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
