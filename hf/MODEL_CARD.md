---
license: other
license_name: muse-model-license
license_link: LICENSE.md
base_model: amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
base_model_relation: quantized
pipeline_tag: text-generation
library_name: exllamav3
tags:
- exl3
- muse_spark
- moe
- dflash
- tensorfold
- dgx-spark
---

# Muse Spark Open 816B-A42B, EXL3 1.8 bpw, optimised for 2 x DGX Spark

Meta's [Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) (816B parameters, MoE, about
42B active per token) quantised to EXL3 at 1.8 bits per weight on average, sized to run across two NVIDIA DGX Sparks
with a context of about 905K tokens. It is served by
[Muspark](https://github.com/ajensenwaud/muspark-2dgx) on TensorFold (v1.0.11).

Currently text only, vision to be added in a future version.

| | |
|---|---|
| Size | 182 GiB in 27 shards (88.9 GiB per Spark) |
| Bit rates | experts 1.80 bpw on average, set per expert projection by measured sensitivity; attention and other dense weights 4.0; output head 6.0 |
| Source | [amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4) @ `4885eb2a` (Meta's own repositories are gated) |
| Quantiser | exllamav3 1.6.2 with Muspark's `patches-exl3/` (upstream exllamav3 does not have the `muse_spark` architecture yet) |
| Drafter | not included; Muspark fetches RadixArk's [DFlash2 drafter](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2) and quantises it locally |

## Quality

At Meta's recommended sampling (temperature 1.0, top_p 1.0, top_k off, reasoning effort low), 200 items per task:

| | GSM8K | MMLU-Pro | Replies that looped to the cap |
|---|---|---|---|
| llama.cpp, Unsloth UD-IQ2_XXS | 193/200 | not measured | 0 |
| This pack, fp16 KV cache | 195/200 | 167/200 | 0 |
| **This pack, 8-bit KV cache (Muspark's default)** | **196/200** | **167/200** | 0 |

Differences are within noise at 200 items.

Use temperature 1.0; greedy decoding can loop on this model at these bit rates.

## Speed

Two DGX Sparks, Muspark's release serve, measured with
[spark-bench](https://github.com/ajensenwaud/spark-bench):

| | llama.cpp, UD-IQ2_XXS | Muspark |
|---|---|---|
| Decode, 1 stream, greedy (tok/s) | 7.4 | 31.6 |
| Decode, 1 stream, temperature 1.0 (tok/s) | not measured | 23.2 |
| Decode, 4 streams, total (tok/s) | 17.4 | 50.9 |
| Prompt read, 4K tokens (tok/s) | 237.4 | 224.0 |
| Context per conversation | 32K | ~905K (1 stream) or 192K (4 streams) |

Don't expect this model to be fast. It is a highly capable frontier model running on bandwidth constrained hardware.

Reading a new long prompt is slow: about 19 minutes before the first token at 122K tokens, 3.3 hours at 488K.
Follow-up turns reuse the cached conversation and start in under a second. Faster prompt reading is coming in v1.1.

## Use

With Muspark on two Sparks: `scripts/prepare.sh`, `scripts/fetch_pack.sh`, `./start.sh`. The pack has only been
evaluated with Muspark's engine. At 182 GiB it does not fit on one Spark.

## License

Muse Model License, as the source model. The licence and the Muse Spark Open Usage Policy are in `LICENSE.md` and
apply to this pack and anything built with it.

Built with Muse Spark Open.

## Credits

Thanks to Meta for Muse Spark Open; AMD for the MXFP4 export it was quantised from; RadixArk for the DFlash2 drafter;
Ash Hart and TensorFold's contributors for [TensorFold](https://github.com/ashhart/TensorFold); Jay Leaton for the
tensor-parallel work the engine port uses; turboderp for [exllamav3](https://github.com/turboderp-org/exllamav3);
Unsloth for the GGUF used as the baseline. Full list in Muspark's
[CREDITS.md](https://github.com/ajensenwaud/muspark-2dgx/blob/main/CREDITS.md).
