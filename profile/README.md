<p align="center">
  <img src="https://raw.githubusercontent.com/gittensor-ai-lab/.github/main/profile/gittensor-ai-lab-banner.png" alt="gittensor-ai-lab" width="100%">
</p>

# gittensor-ai-lab

**SN74 on Gittensor** — human engineers and AI agents co-designing and continuously improving kernels, memory systems, and routing for RTX 5090, RTX PRO 6000, RTX Spark, and Jetson Thor.

Real kernel and runtime engineering: reproducible, hardware-level inference-speed gains beyond existing serving stacks — rewarded by **verified, source-required contribution, not benchmark gaming**.

---

## Proven

**Qwen3-30B-A3B (Q4_K_M GGUF) runs end-to-end on an RTX PRO 6000 Blackwell (sm_120)**, decode optimized **0.60 → 134 tok/s (≈220×)** across 6 source-verifiable passes — **within 1.8× of llama.cpp** on the same model + GPU, output verified correct, **21.7 GB resident** (experts kept quantized, vs ~57 GB bf16).

| Pass | Optimization | tok/s |
|---|---|--:|
| baseline | dequantize all 128 experts | 0.60 |
| 1 | fused selected-expert Q4_K_M FFN — dequant only the routed experts on-read | 32.7 |
| 2 | decode GEMV — coalesced `[out,in]`, replaces M=1 tiled GEMM | 84.4 |
| 3 | CUDA-graph decode — capture once, replay per token | 118.7 |
| 4 | flash-decoding — KV-split + log-sum-exp combine | 133 |
| 5 | fused residual + RMSNorm | 134 |

Each pass is correctness-gated and reproducible from source — the unit the subnet rewards.

---

## The problem we are solving

Existing MoE inference stacks (vLLM, SGLang, llama.cpp) were designed for datacenter multi-GPU serving. They leave significant performance on the table on single-device unified-memory hardware:

- **RTX Spark** (128 GB LPDDR5X, ~273 GB/s, sm_121): no public attention kernel handles `head_dim=512` (Gemma 4 global layers); no serving stack exploits a full CUDA graph across routing + dispatch + FFN
- **RTX PRO 6000 Blackwell** (96 GB GDDR7, ~1.79 TB/s, sm_120): datacenter-class capacity *and* bandwidth on one card — a full 35B MoE plus all experts and a large KV cache stay resident, no paging
- **RTX 5090** (32 GB GDDR7, ~1.79 TB/s, sm_120): sync barriers between router and GroupGEMM break CUDA graph capture, adding launch overhead per MoE layer

SN74 rewards engineers who close these gaps with source-verifiable kernel contributions.

---

## Repos & emission weights

Org-level weights split the org's SN74 emission share across repos. *Within* each repo, contributions are scored on merit — kernels / runtime / moe by **verified speedup** (frontier-delta), bench / agent by **code quality**. Weights are dynamic and revisited as repos mature.

| Repo | Weight | What it does |
|---|--:|---|
| [sparkinfer-kernels](https://github.com/gittensor-ai-lab/sparkinfer-kernels) | **0.40** | Native C++/CUDA kernels — flash decode (hd128/256/512, GQA), decode GEMV, fused quantized MoE expert FFN, GEMM, RMSNorm, RoPE, GGUF dequant. Where inference speed is actually won. |
| [sparkinfer-runtime](https://github.com/gittensor-ai-lab/sparkinfer-runtime) | **0.25** | Scheduler, paged KV cache, CUDA-graph decode engine, native GGUF loading, MoE dispatch, model forward. |
| [sparkinfer-moe](https://github.com/gittensor-ai-lab/sparkinfer-moe) | **0.20** | Sync-free MoE routing + expert dispatch: on-device token counts, no CPU readback, end-to-end CUDA-graph compatible. |
| [sparkinfer-bench](https://github.com/gittensor-ai-lab/sparkinfer-bench) | **0.10** | Reproducible benchmarks + the eval harness: source-required builds, frozen weights, hardware-level metrics. |
| [sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent) | **0.05** | NCU-driven autonomous kernel optimization agent: profile → propose → compile → benchmark. |
| | **1.00** | |

---

## Target hardware

| Device | Memory | Bandwidth | Architecture |
|---|---|---|---|
| **RTX Spark** (GB10) | 128 GB LPDDR5X unified | ~273 GB/s | Blackwell sm_121 |
| **RTX PRO 6000** (GB202) | 96 GB GDDR7 ECC | ~1.79 TB/s | Blackwell sm_120 |
| **RTX 5090** (GB202) | 32 GB GDDR7 | ~1.79 TB/s | Blackwell sm_120 |
| **Jetson Thor** | unified | — | Blackwell sm_121 |

Primary target is RTX Spark — the first consumer device with enough unified memory to run full 70B+ MoE models without expert eviction. RTX PRO 6000 is the high-end discrete target and our current bring-up GPU. (Consumer/workstation Blackwell is `sm_120` / `sm_121` — *not* `sm_100`, which is datacenter B200/GB200 and binary-incompatible. Requires CUDA 12.8+.)

---

## Target models

Two architecturally **distinct** MoEs, deliberately chosen so optimizations must generalize — a win on one but not the other is overfitting, and doesn't count.

| Model | Experts | top-k | Why it matters |
|---|---|---|---|
| **Qwen3.5-35B-A3B** | 256 | 8 + 1 shared | Extreme expert count; skinny GEMM; 8:1 GQA, `head_dim=128`; uniform full attention |
| **Gemma 4 26B-A4B** | 128 | 8 + 1 shared | `head_dim=512` global layers (no public kernel handles this); interleaved local-SWA / global attention, dual RoPE |

> Today's proven path uses **Qwen3-30B-A3B (Q4_K_M)** as the available GGUF proxy for the 35B target. Gemma 4 is next — its head_dim=512 / sliding-window / dual-RoPE attention is the generality forcing function.

---

## Kernel engineering focus

**No Triton.** Native C++ and CUDA. The **production path is portable CUDA** — runs on `sm_120` / `sm_121` today (it's what powers the proven Qwen3 result above). The **[CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)** (CUTLASS tensor-layout DSL) path is opt-in for the tensor-core ceiling — TMA, WGMMA, persistent kernels, epilogue visitor composition.

**Sync-free MoE.** Router token counts stay in GPU memory. No `cudaMemcpy`, no `cudaDeviceSynchronize` between routing and dispatch — the entire MoE forward (routing → expert FFN → SwiGLU epilogue) is captured in a single CUDA graph.

**Fused MoE expert FFN.** Inspired by [NVIDIA's MoE fusion kernel work](https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/): dequantize only the routed experts on-read, apply SwiGLU in the register file before any global-memory write — no intermediate `[T, 2F]` tensor touches DRAM. Unlike dense epilogue fusion (e.g. CODA), the MoE case must handle variable token counts per expert — solved by tracking counts on-device.

**Blackwell warp specialization (roadmap).** Persistent kernels split warpgroups into producer (TMA bulk copy) and consumer (WGMMA compute) roles, pipelining tiles through the 228 KB shared-memory bank so compute and memory fully overlap — the difference between running at memory peak and stalling on load latency, especially on RTX Spark's LPDDR5X.

---

## Roadmap

**Shipped**
- Portable CUDA kernels — flash-decode (incl. `head_dim=512`) + **flash-decoding KV-split**, RoPE, MoE router + sync-free expert FFN, **decode GEMV**, GEMM, RMSNorm, quant — each verified to compile for `sm_120` and `sm_121`
- Sync-free MoE engine (on-device counts, CUDA-graph-ready)
- Runtime: paged KV cache, scheduler, device query, **CUDA-graph decode**, greedy `generate()`
- **Native GGUF loading** — mmap parser + on-GPU **byte-exact Q4_K/Q6_K dequant**; experts kept quantized resident
- **Qwen3-30B-A3B Q4_K_M end-to-end on RTX PRO 6000 — decode 0.60 → 134 tok/s, within 1.8× of llama.cpp**, output verified
- Correctness validated against a double-precision reference (full model to ~1e-8)
- `v0.1.0` tagged across all repos

**Next**
- **Gemma 4 26B-A4B end-to-end** (interleaved local/global, `head_dim=512`, dual RoPE) — the second basket model
- Head_dim-general flash-decode (128 / 256 / 512) + sliding-window mask
- Long-context decode benchmark (8K–32K); batched prefill; top-p / temperature sampling
- The inference-opt **evaluation loop**: source-required build → correctness gate → frontier-delta scoring → public ledger

**Then**
- Tensor-core CuTe DSL GroupGEMM + SwiGLU (TMA, `sm_120a` / `sm_121a`); FP8 / FP4 weights + KV cache
- Expert prefetch / eviction for models exceeding VRAM; RTX Spark unified-memory expert streaming
- NCU-driven autotuning loop ([sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent)) feeding SN74 reproducible scoring

---

## SN74 — how contributions are rewarded

This org is the inference-optimization arm of SN74 on Gittensor. The reward model is built to pay **real engineering, not benchmark gaming** — the lesson from SN14:

1. **Source-required, validator-rebuilt** — no pre-built binaries, no Docker-image submission. The validator compiles your PR from source, so the measured artifact *is* your code, and copying earns nothing.
2. **Frontier-delta rewards** — you're paid for the **verified marginal speedup you add over the current best**, not your rank. Copy-the-leader-plus-ε pays ≈ ε.
3. **Correctness-gated** — speed counts only if the output is *right*. The reference is a frozen, SHA-pinned **exact fp32/fp64 evaluation of the same quantized weights** (so quantization loss is shared and only kernel errors surface). On a fixed prompt set **plus secret holdout / fuzzed shapes**, a submission must clear three layers:
   - *per-kernel* — numerical match to an fp64 reference (dequant **bit-exact**; GEMV/attention within a calibrated fp tolerance);
   - *end-to-end, teacher-forced* — **≥ 99.x% next-token argmax agreement** + bounded logit KL (teacher-forcing avoids single-token cascade divergence);
   - *aggregate* — **perplexity within ε** of the reference on a frozen corpus.

   Tolerances are calibrated above the FP noise floor between two correct implementations; holdout inputs block prompt special-casing; **both basket models (Qwen + Gemma) must pass**. The verdict is a deterministic pass/fail, so validators converge on it.
4. **Multi-model basket** — wins must hold across Qwen3-MoE *and* Gemma 4; a single-model optimization is overfitting and doesn't count.
5. **Reproducible metric** — normalized roofline % + end-to-end tok/s on a canonical GPU class, N-run median + significance gate, so independent validators converge.
6. **Public ledger** — every frontier advance → `(Δ, author, commit)` is auditable.
