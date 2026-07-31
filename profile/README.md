<p align="center">
  <img src="https://raw.githubusercontent.com/gittensor-ai-lab/.github/main/profile/gittensor-ai-lab-banner-2.png" alt="Gittensor AI Lab — Optimize LLM Inference" width="100%">
</p>

# Gittensor AI Lab

**We build SPARKINFER — inference runtimes optimized by open competition.**

One thing, done properly: making language models run faster on the hardware they actually run on. Inference speed first, and on top of it fine-tuning, distillation, and routing.

> **Fewer models. Deeper optimization. Faster evolution.**

## Why speed

Speed compounds in a way most other optimizations do not. A model that runs five times faster on the same hardware is functionally the same as owning five times the compute — more responsive agents, more tokens per dollar, more intelligence out of the same box.

The product that falls out of it is deliberately plain: a fast API endpoint you point your agent at. No chat interface, no dashboard, nothing to sign up for beyond a key.

## What we are building

Two runtimes, one lineage. Same C++/CUDA core, same correctness-gated evaluation, opposite ends of the hardware range.

### ⚡ [SparkInfer](https://github.com/gittensor-ai-lab/sparkinfer) — Blackwell edge

Agentic inference on the GPUs people own. Blackwell-native from the start: `sm_120` and `sm_121`, not datacenter `sm_100`.

| | |
|---|---|
| Hardware | RTX 5090 · RTX PRO 6000 · RTX Spark GB10 · DGX Spark |
| Frontier | **+71% vs llama.cpp** — 473 tok/s on Qwen3.6-35B-A3B, RTX 5090, same GGUF, bs=1 |
| Footprint | **2.5 MB** native binary — 33× smaller than llama.cpp's CUDA runtime |

### 🔷 [SparkInfer-K3](https://github.com/gittensor-ai-lab/sparkinfer-k3) — frontier scale

Kimi K3 on a single 8× H200 node — the largest open-weight model anyone can actually run.

| | |
|---|---|
| Model | 2.8T params · 896 routed experts · hybrid KDA + MLA · 1M context |
| Status | Runs end to end, all 93 layers across 8 GPUs, tensor-parallel and pipeline |
| Parity | **top-1 100%** against llama.cpp on identical weights and ids |
| Open problem | Speed. This is the live competition |

At the edge the bottleneck is one card's memory bandwidth. At frontier scale it becomes expert residency across a node. Working both ends is what full-stack inference optimization means here.

## How it gets faster

SparkInfer is optimized by competition on **[SN74](https://gittensor.io/)** and by Kernel Design Agents. Nobody is asked to trust a number in a PR description.

1. A contributor submits source changes with benchmark evidence.
2. The evaluator builds `main` and the PR from source on the same hardware.
3. Correctness is checked against llama.cpp — token match and KL — before speed counts at all.
4. Decode guards run across every tracked context size.
5. A verified speedup above the significance gate scores; regressions are labelled explicitly.
6. Every run is sealed into a public, immutable log.

Held-out prompts, pinned weights and references, recorded GPU clocks. Tooling, docs, and tests are welcome — the score itself is speedup-only.

**If you want to contribute, read the repo before you write anything.** The scope is narrow and the evaluation baseline is specific.

## Where this goes

**A single endpoint.** You send a prompt, we route it on the back end to whichever of our models answers it best and fastest, and you get a response. No model picker, no configuration.

Asking a user to select a model pushes our job onto them, and most of the time they will pick wrong — spending a frontier model's tokens on a request a much smaller model would have answered instantly. If the routing is good, the choice should not exist.

Further out this points at tuning and training our own models, and eventually hosting only models we built ourselves. Not this month's work, but it is the direction the stack is pointed.

## Repos

| repo | purpose |
|---|---|
| [`sparkinfer-k3`](https://github.com/gittensor-ai-lab/sparkinfer-k3) | Kimi K3 on 8× H200 — runtime, eval harness, contributor guide |
| [`sparkinfer-k3-log`](https://github.com/gittensor-ai-lab/sparkinfer-k3-log) | Immutable eval log — one sealed receipt per run |
| [`sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer) | Blackwell edge runtime — kernels, MoE, bench, docs |
| [`sparkinfer-log`](https://github.com/gittensor-ai-lab/sparkinfer-log) | Immutable eval log for the edge runtime |

---

[Site](https://gittensor.io/) · [Live competition](https://github.com/gittensor-ai-lab/sparkinfer-k3) · [Subnet repo](https://github.com/entrius/gittensor) · [Docs](https://docs.gittensor.io/)
