# Muspark

Muse Spark Open on 2x DGX Spark with TensorFold.

Meta's Muse Spark Open (816B parameters, MoE, about 42B active per token; 62 layers, 16 of them full
attention and 46 sliding-window) served across two DGX Sparks nodes with tensor parallelism. The model runs on
TensorFold (v1.0.11) from our own EXL3 quantisation at 1.8 bits per weight (bpw) on
average. RadixArk's DFlash2 drafter is used for speculative decoding.

Currently text only, vision to be added later. Muse Spark is a heavy model for Sparks to run so don't expect high speed. The default profile serves one stream with a 905K token context, a second profile serves four at 192K each.

The comparison point throughout is llama.cpp running Unsloth's UD-IQ2_XXS GGUF (225.6 GB) across the same two
Sparks over RPC: the most direct way to run this model on this hardware before Muspark.

## Speed and quality

Speed, measured with [spark-bench](https://github.com/ajensenwaud/spark-bench) on the release configuration,
both GPUs free, no other client, greedy, 512-token replies, reasoning effort low:

| | llama.cpp, UD-IQ2_XXS | Muspark |
|---|---|---|
| Decode, 1 stream (tok/s, mean of 4 workloads) | 7.4 | 31.6 (29.4-33.7) |
| Decode, 1 stream at temperature 1.0 (Meta's sampling, top_k off) | not measured | 23.2 (21.9-24.0); 25.4 at the server's default top_k 64 |
| Decode, 4 streams, total (multi profile) | 17.4 | 50.9 (48.4-53.6) |
| Prompt read, 4K tokens (tok/s) | 237.4 | 224.0 |
| Prompt read, 16K tokens (tok/s) | 206.2 | 200.4 |
| Decode after a 32K prompt (tok/s) | 6.8 (28K) | 21.3 |
| Context per conversation, as run | 32K (4 slots) | ~905K (1 slot) or 192K (4 slots) |

Quality, at Meta's recommended sampling (temperature 1.0, top_p 1.0, top_k off, request seed 1234, reasoning effort
low):

| Arm | GSM8K | MMLU-Pro | Replies that looped to the cap |
|---|---|---|---|
| llama.cpp, Unsloth UD-IQ2_XXS | 193/200 | not measured | 0 |
| Muspark, 1.8 bpw pack, fp16 KV cache | 195/200 | 167/200 | 0 |
| **Muspark, 1.8 bpw pack, 8-bit KV cache (the release)** | **196/200** | **167/200** | 0 |

Use temperature 1.0 (Meta's default); greedy decoding can loop on this model at these bit rates.

Reading a new long prompt is slow: about 19 minutes before the first token at 122K tokens, 3.3 hours at 488K.
Follow-up turns reuse the cached conversation and start in under a second. Faster prompt reading is coming in v1.1.

Paired on identical items (McNemar; "only A / only B" = items only one arm got right):

| Comparison | Task | Only A / only B | p |
|---|---|---|---|
| llama.cpp vs the release | GSM8K | 0 / 3 | 0.08 (exact 0.25) |
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
scripts/prepare.sh       # images, the engine (TensorFold v1.0.11 + patches-zig/, built in a container), the drafter
scripts/fetch_pack.sh    # the 1.8 bpw pack from Hugging Face (182 GiB, pinned revision) to both nodes, verified
./start.sh               # PROFILE=long: OpenAI-compatible API on :8888, model id Muse-Spark-Open
./stop.sh
```

The pack: [ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw](https://huggingface.co/ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw).
`prepare.sh` also fetches RadixArk's drafter at a pinned revision and quantises it on your own GPU (a few minutes,
after a one-time 30-60 minute image build); Muspark does not redistribute it. `SKIP_DRAFTER=1 scripts/prepare.sh`
skips that, and `DRAFTER=0 ./start.sh` then serves without it (about 18 tok/s for one stream instead of 31.6 with it).

Making the pack yourself instead of downloading it took about three days with the conversion split across both
Sparks by layers (`scripts/quantise_split.sh`), and needs the 509 GB AMD MXFP4 source.

Loading takes about 6 minutes.

## Profiles

| | `PROFILE=long` (default) | `PROFILE=multi` |
|---|---|---|
| Conversations at once | 1 | 4 |
| Context per conversation | ~905K tokens (the most that fits) | 196,608 tokens |
| Decode, one stream | 31.6 tok/s | 31.9 tok/s |
| Decode, four streams (total) | n/a | 50.9 tok/s |
| Lowest free memory while measured (head / worker) | 4.89 / 6.23 GiB | 5.72 / 7.10 GiB |

Both use the drafter, the 8-bit KV cache and the prompt cache. `start.sh` estimates each Spark's memory before it
loads anything and, with `CONTEXT_AUTO=1` (the default), cuts the context to the largest that leaves 3 GiB free on
both; on two idle Sparks that is about 905K tokens for one slot. Pick your own with `ZIG_CONTEXT=` and `PARALLEL=`;
the estimate refuses settings that do not fit (`CONTEXT_AUTO=0`) or shrinks them.

## Memory

Each Spark has 121.6 GiB shared between CPU and GPU. Per Spark, the release holds the pack's half (88.9 GiB), the
drafter (1.2 GiB), the KV cache, and about 9 GiB of CUDA context, kernels, graphs, scratch and buffers. The 8-bit KV
cache stores the 16 full-attention layers in int8 with an fp32 scale per token and head, about half the bytes of
fp16: roughly 17 GiB a Spark for 1M tokens, which is what makes a ~905K context fit at all.

Unified memory does not fail an oversized allocation; it takes the host down. Two guards stop that: `start.sh`
refuses a context its estimate says will leave a Spark under 3 GiB, and a watchdog on each node removes both ranks if
free memory drops under 512 MiB. If the watchdog fires, `./start.sh` says so; restart with a smaller `ZIG_CONTEXT`.

## How it was built

- Engine: TensorFold's Zig engine had no Muse family. Muspark adds one (`patches-zig/`, 60 patches on v1.0.11):
  the model's forward on EXL3 packs, split across the two Sparks, CUDA graphs for decode, the DFlash2 drafter,
  a prompt path, the 8-bit KV cache, the prompt cache and the memory reductions. It is checked against exllamav3's
  reference forward within exllamav3's own noise.
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
tensor-parallel and RoCE work carried in `patches-zig/` and the method this port follows; turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3); Unsloth for the GGUF used as the baseline. Full list:
[CREDITS.md](CREDITS.md).

## License

Muspark's own code is under the BSD 3-Clause License (`LICENSE`). The engine and quantiser patches in
`patches-zig/` and `patches-exl3/` modify TensorFold (Apache-2.0) and exllamav3 (MIT) and stay under those projects'
licences. The model remains under Meta's own Muse License.

Built with Muse Spark Open.
