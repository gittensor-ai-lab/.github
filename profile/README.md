<p align="center">
  <img src="https://raw.githubusercontent.com/gittensor-ai-lab/.github/main/profile/gittensor-ai-lab-banner.png" alt="gittensor-ai-lab" width="100%">
</p>

# gittensor-ai-lab

**We build SPARKINFER: the fastest MoE/LLM inference runtime for consumer and edge NVIDIA Blackwell GPUs.**

SPARKINFER is a Blackwell-native C++/CUDA runtime for **local AI agents**, **edge AI**, and **robotics** on RTX 50xx, RTX PRO 6000, RTX Spark / GB10, and Jetson Thor. It is optimized through **SN74 on Gittensor**, where every speed claim is rebuilt from source, checked for correctness, and measured on real RTX 5090 hardware.

[Live dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) · [Runtime repo](https://github.com/gittensor-ai-lab/sparkinfer) · [Eval logs](https://gittensor-ai-lab.github.io/sparkinfer-log/) · [Miner guide](https://github.com/gittensor-ai-lab/sparkinfer/blob/main/docs/miner-guide.md)

---

## What We Build

SPARKINFER targets the gap between cloud inference engines and portable baselines:

- **Fast local decode.** Batch-size-1 MoE/LLM inference for personal agents where latency, power, and memory decide usability.
- **Consumer and edge Blackwell.** `sm_120` RTX 5090 / RTX PRO 6000 and `sm_121` RTX Spark / GB10 / Jetson Thor. It is not a B200/GB200 datacenter runtime.
- **Small native runtime.** C++/CUDA, no Python service stack required for the core runtime path.
- **Correctness-gated speed.** Optimizations count only when token-match and KL stay within the eval thresholds.
- **Source-rebuilt evaluation.** PRs are built from source on the evaluator. Prebuilt binaries are only a run convenience, never a submission format.

## Current RTX 5090 Frontier

Qwen3-30B-A3B, 128 generated tokens, batch size 1.

| context | sparkinfer<br>GGUF Q4_K_M | llama.cpp<br>GGUF Q4_K_M | vLLM<br>GPTQ Int4 | SGLang<br>GPTQ Int4 | TensorRT-LLM<br>NVFP4 |
|---:|---:|---:|---:|---:|---:|
| 128 | **493.56 tok/s** | 365.85 tok/s | 280.83 tok/s | 241.21 tok/s | 99.00 tok/s |
| 512 | **469.58 tok/s** | 342.59 tok/s | 270.86 tok/s | 239.82 tok/s | 98.59 tok/s |
| 4k | **392.65 tok/s** | 292.99 tok/s | 202.65 tok/s | 234.67 tok/s | failed |
| 16k | **266.14 tok/s** | 245.53 tok/s | 81.89 tok/s | 226.12 tok/s | not run |

sparkinfer and llama.cpp use the same RTX 5090, same Qwen3-30B-A3B Q4_K_M GGUF, and same 128 generated tokens. vLLM, SGLang, and TensorRT-LLM use the fastest successful quantized HF path from the same competitor run because they do not load GGUF. Full commands, model IDs, caveats, and raw artifact paths are in [`bench/competitors/latest-results.md`](https://github.com/gittensor-ai-lab/sparkinfer/blob/main/bench/competitors/latest-results.md).

Runtime footprint, excluding model weights and launcher scripts:

| runtime | measured artifact | size | sparkinfer is |
|---|---|---:|---:|
| sparkinfer | native runtime binary | **2.5 MB** | baseline |
| llama.cpp | CUDA runtime executable + shared libs | 80 MB | 33x smaller |
| vLLM | runtime package | 605 MB | 243x smaller |
| SGLang | runtime + native kernel packages | 1.9 GB | 743x smaller |
| TensorRT-LLM | runtime package | 3.6 GB | 1,430x smaller |

Quality is checked separately from speed. The current 196-item quality suite keeps sparkinfer in the same range as llama.cpp/vLLM while the runtime stays much smaller and faster on the tracked decode path.

## Why It Exists

Most LLM inference engines were built for datacenter GPUs and cloud serving. On consumer GPUs they can be hard to install, heavy, power hungry, and slow to adapt to new MoE models or decode algorithms because their codebases are large and multi-target.

SPARKINFER is built for the opposite use case:

- **Local-first AI.** Your data stays on your machine.
- **Agent-native decode.** Optimized for single-stream, low-latency token generation.
- **Power-aware Blackwell kernels.** Designed for cards people actually own, not only datacenter GPUs.
- **Fast-moving MoE support.** Quantized experts, paged KV cache, flash-decode, CUDA graphs, and sync-free MoE dispatch are first-class runtime features.
- **Small enough to audit.** The core runtime is measured in megabytes, not gigabytes.

## How SN74 Keeps It Honest

SN74 rewards verified marginal speedup, not claims in a PR description.

1. A contributor opens a PR with source changes and benchmark evidence.
2. The bot builds `main` and the PR from source on the same RTX 5090.
3. Correctness is checked with token-match and KL against the reference path.
4. Decode guards run at 128, 512, 4k, and 16k context.
5. A real improvement above the significance gate gets an `eval:<label>` score.
6. Regressions are marked explicitly with `regression-*` labels.
7. Public artifacts go to the dashboard and eval log.

The eval path is trust-hardened: held-out prompts reduce overfitting, model weights and llama.cpp references are pinned, GPU clock metadata is recorded, and every frontier advance is immutably logged. Sub-2% gains are never aggregated across contexts. Tooling, docs, refactors, and tests are welcome, but SN74 score is speedup-only.

## Repository Map

| repo | purpose |
|---|---|
| [`sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer) | Main runtime monorepo: `kernels/`, `runtime/`, `moe/`, `bench/`, eval tooling, docs |
| [`sparkinfer-log`](https://github.com/gittensor-ai-lab/sparkinfer-log) | Immutable public eval log for reproducible PR runs |
| [`sparkinfer-bench`](https://github.com/gittensor-ai-lab/sparkinfer-bench) | Standalone reproducible benchmark work |
| [`sparkinfer-kernels`](https://github.com/gittensor-ai-lab/sparkinfer-kernels) | Kernel-focused component history |
| [`sparkinfer-runtime`](https://github.com/gittensor-ai-lab/sparkinfer-runtime) | Runtime-focused component history |
| [`sparkinfer-moe`](https://github.com/gittensor-ai-lab/sparkinfer-moe) | MoE-focused component history |

The main work now happens in [`sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer).

## Quickstart

On an NVIDIA Blackwell box with CUDA 12.8+:

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer
cd sparkinfer

# Decode throughput.
bench/scripts/bench.sh --download

# Head-to-head vs llama.cpp on the same GGUF and GPU.
bench/scripts/bench.sh --download --compare

# Accuracy gate: token-match, KL, perplexity.
bench/scripts/accuracy.sh --download
```

The scripts auto-detect the GPU arch, use the newest matching prebuilt binary when available, and fall back to a source build when needed.

## Roadmap

**Milestone 1 - RTX 5090 proof of concept and v1.0.** Make `sm_120` RTX 5090 the proof platform for Qwen3.6 MoE: fastest TPS and TTFT across tracked context sizes, DFlash3 as the default decode path, SOTA decode algorithms implemented as first-class runtime features, power/thermals optimized, and the v1.0 release target ready to ship.

**Milestone 2 - PRO 6000 / RTX Spark v2.0.** Extend the same runtime across RTX 50xx, RTX PRO 6000, and unified-memory Blackwell systems such as RTX Spark / GB10 and Jetson Thor (`sm_121`). The v2.0 target is a production-ready local runtime for personal AI agents.

**Milestone 3 - Physical AI v3.0.** Deploy SOTA VLA and world foundation models on edge Blackwell to accelerate robotics: low-latency perception-action loops, on-device planning, multimodal memory, and runtime support for physical AI agents that must operate locally and safely.
