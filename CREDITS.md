# Credits

Muspark is built on other people's work. Thank you to:

- **Meta** for [Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) (816B-A42B), the
  model Muspark serves, under the Muse Model License and its Usage Policy (the text ships in the pack as
  `LICENSE.md`). Built with Muse Spark Open.
- **AMD** for [amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4)
  (revision `4885eb2a`, Muse Model License), the MXFP4 export of Meta's weights that our EXL3 pack was quantised from.
  Quality figures in Muspark are relative to that artefact, not to Meta's BF16 weights.
- **RadixArk** for [RadixArk/Muse-Spark-Open-DFlash2](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2)
  (revision `46c90f41`), the DFlash2 speculative-decoding drafter. Its repository states no licence, so Muspark
  does not redistribute it: `scripts/prepare.sh` fetches the original at that revision and quantises it locally
  (`scripts/quantise_drafter.sh`).
- **TensorFold** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) by Ash Hart
  ([@ashxhart](https://x.com/ashxhart)) and its contributors. The engine, its native server, lanes scheduler, kernels
  and tooling are TensorFold's; Muspark is a set of patches on top of release v1.0.11 (`patches-zig/`).
- **Jay Leaton** ([@jayleaton](https://github.com/jayleaton)). Patches 0001-0002 and 0004-0034 in `patches-zig/` are
  his commits (cherry-picked with `-x`): the keyed CUDA graph cache and device arena, tensor parallelism across CUDA
  processes, the one-shot RoCE exchange between the Sparks, the server's stop and drain handling and the family hooks.
  The Muse port follows the method of his DeepSeek-V4.1 port to the Zig engine.
- **Jonathan Regan-Beasley** ([@jregan-beasley-smc](https://github.com/jregan-beasley-smc)), author of patch 0003
  (the RDMA VMM capability fix).
- **b12x "RoCEnante"** (Apache-2.0): the RoCE all-gather in the patches adapts its protocol.
- **turboderp** for [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format, the quantiser
  (with `patches-exl3/` adding the `muse_spark` architecture), the sensitivity tools `sc_measure` / `sc_optimize` that
  set the per-tensor bit rates, and the reference forward and perplexity method the engine is checked against.
- **Unsloth** for [unsloth/Muse-Spark-Open-816B-A42B-GGUF](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF)
  (UD-IQ2_XXS, revision `c0067056`), the baseline Muspark is compared with, served by
  [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) with its Muse support from pull request #30616.
- **Mia's AI Lab** ([MiaAI-Lab](https://github.com/MiaAI-Lab)) for the two-Spark recipe format and the GLM-5.3
  TensorFold image used as the CUDA base for the build and run images.
- **Zig** ([ziglang.org](https://ziglang.org), MIT), 0.17.0, the toolchain inside the build image.

Evaluation data:

- **GSM8K** ([openai/grade-school-math](https://github.com/openai/grade-school-math), MIT), Cobbe et al., 2021.
- **MMLU-Pro** ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), MIT), Wang et al., 2024.
- **WikiText-2** ([Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), CC BY-SA 3.0), Merity et
  al., 2016, for perplexity.
- Speed was measured with [spark-bench](https://github.com/ajensenwaud/spark-bench); its prompts are its own.

Any errors in Muspark are mine, not theirs.
