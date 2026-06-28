<p align="center">
  <img src="https://raw.githubusercontent.com/gittensor-ai-lab/.github/main/profile/gittensor-ai-lab-banner.png" alt="gittensor-ai-lab" width="100%">
</p>

# gittensor-ai-lab

**SN74 on Gittensor** — a **Blackwell-native** MoE/LLM inference runtime, pushed toward the hardware ceiling by human engineers and AI agents. Real kernel and runtime work, rewarded by **verified, source-required speedups — not benchmark gaming**.

[**Live dashboard**](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) · [**sparkinfer**](https://github.com/gittensor-ai-lab/sparkinfer) · [**releases**](https://github.com/gittensor-ai-lab/sparkinfer/releases)

---

## Why Blackwell-only

vLLM, SGLang, and llama.cpp must spread across vendors and GPU generations; we go the other way — **deep on NVIDIA Blackwell** (`sm_120` / `sm_121`): general across *models* (Qwen, Gemma), uncompromising on one *architecture*. Blackwell is the substrate of the next personal computer — the **RTX Spark** (GB10, 128 GB unified) brings full 70B+ MoE inference to a desk, Jetson Thor to robots, RTX 5090 / PRO 6000 to every workstation. Local inference (privacy, latency, cost, offline) is where AI is heading, and whoever runs MoE fastest on local Blackwell owns that layer. **RTX Spark is our flagship.**

## Proven

**Qwen3-30B-A3B (Q4_K_M)** runs end-to-end on Blackwell, output verified correct vs llama.cpp:

- **RTX PRO 6000** (sm_120) — decode **0.60 → 134 tok/s** across 6 source-verifiable passes ([live chart](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/)), within **1.8×** of llama.cpp, **21.7 GB** resident (experts kept quantized, vs ~57 GB bf16).
- **RTX 5090** (sm_120, CUDA 13) — frontier **388.68 tok/s** (`v0.3.0`) at **21.4 GB**, fits a 32 GB card; `ctest` 5/5, compute-sanitizer clean. **Now past llama.cpp** — **+4.5%** at 128-tok decode (~parity at long context), the first **kernel-level** decode win on this model: same GGUF, same Q4_K_M precision, same greedy bs=1 decode — no speculative decoding or attention shortcut. Each PR is scored against `main` on the **same** GPU, so the delta is hardware-independent.
- **Accuracy** — **98%** top-1 token agreement, **KL ≈ 0.14**, perplexity 6.16. Every PR is gated against this bar, so no kernel change silently regresses quality. ([details](https://github.com/gittensor-ai-lab/sparkinfer/blob/main/bench/results/accuracy_qwen3-30b-a3b_q4km.md))

## Try it

On any Blackwell box (CUDA 12.8+) — auto-detects the GPU, fetches prebuilt binaries (or builds from source), downloads the model:

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer && cd sparkinfer
bench/scripts/bench.sh --download              # decode tok/s
bench/scripts/bench.sh --download --compare    # head-to-head vs llama.cpp
bench/scripts/accuracy.sh --download           # token-match / KL / perplexity vs llama.cpp
```

## Target hardware & models

Consumer/workstation Blackwell is `sm_120` / `sm_121` — **not** `sm_100` (datacenter B200/GB200, binary-incompatible). Requires CUDA 12.8+.

| Device | Memory | Bandwidth | Arch |
|---|---|---|---|
| **RTX Spark** (GB10) — flagship | 128 GB LPDDR5X unified | ~273 GB/s | sm_121 |
| **RTX PRO 6000** (GB202) — bring-up | 96 GB GDDR7 | ~1.79 TB/s | sm_120 |
| **RTX 5090** (GB202) | 32 GB GDDR7 | ~1.79 TB/s | sm_120 |
| **Jetson Thor** | unified | — | sm_121 |

Two deliberately **distinct** MoEs — a win must generalize across both, or it's overfitting and doesn't count:

| Model | Experts | top-k | The hard part |
|---|---|---|---|
| **Qwen3.5-35B-A3B** | 256 | 8 + 1 | extreme expert count, skinny GEMM, GQA `head_dim=128` |
| **Gemma 4 26B-A4B** | 128 | 8 + 1 | `head_dim=512` globals (no public kernel), local-SWA/global interleave, dual RoPE |

> Proven today via **Qwen3-30B-A3B (Q4_K_M)** as the GGUF proxy for the 35B target; Gemma 4 is the next forcing function.

## How it's built

Native **C++/CUDA, no Triton**. The production path is **portable CUDA** (`sm_120` / `sm_121` today); the **[CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)** tensor-core path (TMA, WGMMA, persistent kernels) is opt-in for the ceiling.

- **Sync-free MoE** — router counts stay on-device (no `cudaMemcpy` / `cudaDeviceSynchronize` between routing and dispatch), so the whole MoE forward is captured as one CUDA graph.
- **Fused expert FFN** — dequant only the routed experts on-read, SwiGLU in registers before any DRAM write; variable tokens-per-expert handled on-device.
- **CUDA-graph decode**, paged KV cache, byte-exact on-GPU Q4_K/Q6_K dequant (experts kept quantized resident).
- *Roadmap:* Blackwell warp specialization — producer/consumer warpgroups pipelining TMA+WGMMA through 228 KB smem for memory-peak overlap on RTX Spark's LPDDR5X.

## SN74 — rewards for engineering, not gaming

The reward model pays **real, verified speedups** (the lesson from SN14):

- **Source-required, validator-rebuilt** — you submit source; the validator compiles your PR, so the measured artifact *is* your code and copying earns nothing. (Release binaries are a user convenience, not a submission format.)
- **Frontier-delta** — paid for the **marginal speedup over the current best**, not rank; copy-the-leader-plus-ε pays ≈ ε. Labeled **XL/L/M/S/XS** by the eval loop, scored the same in any subsystem (no per-area budget). **Non-speedup PRs are welcome but score 0.**
- **Correctness-gated** — speed counts only if output is right: per-kernel match to an fp64 reference (dequant bit-exact), ≥99% teacher-forced next-token agreement + bounded KL, perplexity within ε — on a fixed set **plus secret holdout / fuzzed shapes**, and **both basket models must pass**.
- **Auditable** — deterministic pass/fail; every frontier advance → `(Δ, author, commit)` on a public ledger.

Labels are **bands of % speedup over the frontier** (`XS` 2–3.5% … `XL` >18%; a sub-2% gain is within noise → `none`). They scale with the frontier, so every tier stays reachable as decode speed grows and the subnet keeps paying real progress instead of stalling.

**Automated & hardened.** A bot evaluates each *greenlit* PR (tick *Tested on RTX 5090*, or a maintainer adds `test-on-5090`): builds from source on an RTX 5090, gates correctness, scores the frontier-delta, posts an `eval:<label>` verdict, and streams it to the [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) — never auto-merging. The scoring harness is **maintainer-only** (CODEOWNERS + a sensitive-path gate, graded pinned to `origin/main`); a **denylist** auto-closes flagged accounts and **copycat/sybil detection** catches re-submitted diffs. See [`sparkinfer/eval`](https://github.com/gittensor-ai-lab/sparkinfer/tree/main/eval).

## Repos & roadmap

Single monorepo — **[sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer)** (`kernels/` · `runtime/` · `moe/` · `bench/`); the NCU-driven **[sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent)** autotuner is a separate repo.

- **Shipped** — portable CUDA kernels (flash-decode incl. `head_dim=512`, KV-split, decode GEMV, sync-free MoE FFN, GEMM, RMSNorm, RoPE); CUDA-graph decode + paged KV + native GGUF; Qwen3-30B-A3B end-to-end; **automated eval loop + live dashboard + anti-gaming**; **`v0.2.0`** (prebuilt sm_120/CUDA13 binaries).
- **Next** — Gemma 4 26B-A4B end-to-end (`head_dim=512`, dual RoPE, SWA/global); head_dim-general flash-decode; long-context + batched prefill; held-out/fuzzed correctness + clock-pinned ledger.
- **Then** — CuTe DSL tensor-core GroupGEMM + SwiGLU, FP8/FP4 weights + KV; expert prefetch/eviction + RTX Spark unified-memory streaming; NCU autotuning loop feeding SN74.
