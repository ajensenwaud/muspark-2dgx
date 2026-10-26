# Muspark

Muse Spark Open on 2x DGX Spark with TensorFold.

Meta's Muse Spark Open (816B parameters, MoE, about 42B active per token; 62 layers, 16 of them full
attention and 46 sliding-window) served across two DGX Sparks nodes with tensor parallelism. The model runs on
TensorFold (main, plus Muspark's Muse family, submitted upstream) from our own EXL3 quantisation at 1.8 bits per
weight (bpw) on average. RadixArk's DFlash2 drafter is used for speculative decoding.

Currently text only, vision to be added later. Muse Spark is a heavy model for Sparks to run so don't expect high speed. The default profile serves one stream with a ~460K token context, a second profile serves four at ~110K each.

v1.1 moves Muspark from a 60-patch TensorFold v1.0.11 onto TensorFold's main line with five patches. The price: main
takes no 8-bit KV cache, so the context is about half of v1.0's (~460K instead of ~905K for one stream, ~110K instead
of 192K for four), and four-stream decode is lower (43.1 tok/s instead of 50.9). Quality carries over: v1.1 computes
the same logits as v1.0 did with an fp16 KV cache, byte for byte.

The comparison point throughout is llama.cpp running Unsloth's UD-IQ2_XXS GGUF (225.6 GB) across the same two
Sparks over RPC: the most direct way to run this model on this hardware before Muspark.

## Speed and quality

Speed, measured with [spark-bench](https://github.com/ajensenwaud/spark-bench) on the release configuration,
both GPUs free, no other client, greedy, 512-token replies, reasoning effort low:

| | llama.cpp, UD-IQ2_XXS | Muspark v1.0 | Muspark v1.1 |
|---|---|---|---|
| Decode, 1 stream (tok/s, mean of 4 workloads) | 7.4 | 31.6 (29.4-33.7) | 30.1 (28.7-32.1) |
| Decode, 1 stream at temperature 1.0 (Meta's sampling, top_k off) | not measured | 23.2 (21.9-24.0); 25.4 at the server's default top_k 64 | not re-measured |
| Decode, 4 streams, total (multi profile) | 17.4 | 50.9 (48.4-53.6) | 43.1 (41.6-45.5) |
| Prompt read, 4K tokens (tok/s) | 237.4 | 224.0 | 209.0 |
| Prompt read, 16K tokens (tok/s) | 206.2 | 200.4 | 196.4 |
| Decode after a 32K prompt (tok/s) | 6.8 (28K) | 21.3 | not re-measured |
| Context per conversation, as run | 32K (4 slots) | ~905K (1 slot) or 192K (4 slots) | ~450K-470K (1 slot) or ~110K (4 slots) |

v1.0 ran the 8-bit KV cache; v1.1 runs fp16. The v1.1 numbers come from a later session on the same Sparks, not a
back-to-back run.

Quality, at Meta's recommended sampling (temperature 1.0, top_p 1.0, top_k off, request seed 1234, reasoning effort
low):

| Arm | GSM8K | MMLU-Pro | Replies that looped to the cap |
|---|---|---|---|
| llama.cpp, Unsloth UD-IQ2_XXS | 193/200 | not measured | 0 |
| **Muspark, 1.8 bpw pack, fp16 KV cache (v1.1)** | **195/200** | **167/200** | 0 |
| Muspark, 1.8 bpw pack, 8-bit KV cache (v1.0) | 196/200 | 167/200 | 0 |

The fp16 row was measured on the v1.0 engine; v1.1 gives the same logits byte for byte, so it carries over.

Use temperature 1.0 (Meta's default); greedy decoding can loop on this model at these bit rates.

Reading a new long prompt is slow: about 19 minutes before the first token at 122K tokens, around 3 hours at the
full context. Follow-up turns reuse the cached conversation and start in under a second. Faster prompt reading is
the next engine step.

Paired on identical items (McNemar; "only A / only B" = items only one arm got right):

| Comparison | Task | Only A / only B | p |
|---|---|---|---|
| llama.cpp vs 8-bit KV (v1.0) | GSM8K | 0 / 3 | 0.08 (exact 0.25) |
| 2.0 bpw vs 1.8 bpw (fp16 KV) | MMLU-Pro | 4 / 1 | 0.18 (exact 0.38) |
| 2.0 bpw vs 1.8 bpw (fp16 KV) | GSM8K | 2 / 2 | 1.00 |
| fp16 vs 8-bit KV, both packs pooled (400 pairs) | MMLU-Pro | 7 / 3 | 0.21 (exact 0.34) |
| fp16 vs 8-bit KV, both packs pooled (400 pairs) | GSM8K | 1 / 3 | 0.32 |

## What you need

- Two DGX Sparks joined by their CX7 ports, Docker with the NVIDIA runtime on both, SSH from the
  head (rank 0) to the worker (rank 1). Hosts, addresses and network interface names: `scripts/config.sh` (override
  in `scripts/local.sh`).
- About 200 GB of free disk on each node for the pack, plus ~25 GB for the images, the engine build and the drafter.
- Nothing else serving on either GPU: the release uses all but 3 GiB of each Spark's memory.
- Ports: 8888 (API) and 29681 (rendezvous between the ranks).

## Run it

```bash
scripts/prepare.sh       # images, the engine (TensorFold main + patches-zig/, built in a container), the drafter
scripts/fetch_pack.sh    # the 1.8 bpw pack from Hugging Face (182 GiB, pinned revision) to both nodes, verified
./start.sh               # PROFILE=long: OpenAI-compatible API on :8888, model id Muse-Spark-Open
./stop.sh
```

The pack: [ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw](https://huggingface.co/ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw).
`prepare.sh` also fetches RadixArk's drafter at a pinned revision and quantises it on your own GPU (a few minutes,
after a one-time 30-60 minute image build); Muspark does not redistribute it. `SKIP_DRAFTER=1 scripts/prepare.sh`
skips that, and `DRAFTER=0 ./start.sh` then serves without it (about 17.5 tok/s for one stream instead of 30.1 with it).

Making the pack yourself instead of downloading it took about three days with the conversion split across both
Sparks by layers (`scripts/quantise_split.sh`), and needs the 509 GB AMD MXFP4 source.

Loading takes about 6 minutes.

## Profiles

| | `PROFILE=long` (default) | `PROFILE=multi` |
|---|---|---|
| Conversations at once | 1 | 4 |
| Context per conversation | ~450K-470K tokens (the most that fits) | ~110K tokens (110,592) |
| Decode, one stream | 30.1 tok/s | not measured |
| Decode, four streams (total) | n/a | 43.1 tok/s |
| Free memory after loading (head / worker) | 5.5 / 5.1 GiB | 4.9 / 5.5 GiB |

Both use the drafter, an fp16 KV cache and the prompt cache. `start.sh` estimates each Spark's memory before it
loads anything and, with `CONTEXT_AUTO=1` (the default), cuts the context to the largest that leaves 3 GiB free on
both; on two idle Sparks that is about 450K-470K tokens for one slot and about 110K each for four. Pick your own with `ZIG_CONTEXT=` and `PARALLEL=`;
the estimate refuses settings that do not fit (`CONTEXT_AUTO=0`) or shrinks them.

## Memory

Each Spark has 121.6 GiB shared between CPU and GPU. Per Spark, the release holds the pack's half (88.9 GiB), the
drafter (1.2 GiB), the KV cache, and about 9 GiB of CUDA context, kernels, graphs, scratch and buffers. The KV cache
is fp16: about 14 GiB a Spark at 470K tokens. v1.0's 8-bit KV cache held about twice the context in the same memory;
TensorFold's main line takes no lossy KV cache, so v1.1 does without it.

Unified memory does not fail an oversized allocation; it takes the host down. Two guards stop that: `start.sh`
refuses a context its estimate says will leave a Spark under 3 GiB, and a watchdog on each node removes both ranks if
free memory drops under 512 MiB. If the watchdog fires, `./start.sh` says so; restart with a smaller `ZIG_CONTEXT`.

## How it was built

- Engine: TensorFold's Zig engine had no Muse family. Muspark adds one (`patches-zig/`, 5 patches on TensorFold
  main, submitted upstream as pull requests #1295 and #1336): the model's forward on TensorFold's shared EXL3
  operators, split across the two Sparks, CUDA graphs for decode, the DFlash2 drafter, a prompt path and the prompt
  cache. It was checked against exllamav3's reference forward within exllamav3's own noise; v1.1 gives the same logits
  as v1.0 with an fp16 KV cache byte for byte, and drafted and cache-resumed replies are byte-identical to plain ones.
- Pack: AMD's MXFP4 export of Meta's weights (Meta's own repositories are gated), converted with exllamav3 plus
  `patches-exl3/`. Each expert projection got its own bit rate from a measured sensitivity (exllamav3's `sc_measure`
  and `sc_optimize`): 1.80 bits on average, attention and other dense weights at 4 bits, the output head at 6. The
  conversion ran split by layers across both Sparks (one converts layers 0-31, the other 32-61 and the head). The
  second half was calibrated on states that passed through unquantised layers 0-31, not quantised ones; the full-pack
  checks above were run on the finished pack.
- Drafter: RadixArk's DFlash2 (5 layers, blocks of 8), quantised to 4 bits by `scripts/quantise_drafter.sh`. Drafted
  replies are byte-identical to plain ones.

## Credits 

Thanks to Meta for Muse Spark Open; AMD for the MXFP4 export it was quantised from; RadixArk for the DFlash2 drafter;
Ash Hart ([@ashxhart](https://x.com/ashxhart)) and TensorFold's contributors for
[TensorFold](https://github.com/ashhart/TensorFold); Jay Leaton ([@jayleaton](https://github.com/jayleaton)) for the
tensor-parallel and RoCE work (now part of TensorFold) and the method this port follows; turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3); Unsloth for the GGUF used as the baseline. Full list:
[CREDITS.md](CREDITS.md).

## License

Muspark's own code is under the BSD 3-Clause License (`LICENSE`). The engine and quantiser patches in
`patches-zig/` and `patches-exl3/` modify TensorFold (Apache-2.0) and exllamav3 (MIT) and stay under those projects'
licences. The model remains under Meta's own Muse License.

Built with Muse Spark Open.
