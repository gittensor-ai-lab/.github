<p align="center">
  <img src="https://raw.githubusercontent.com/gittensor-ai-lab/.github/main/profile/gittensor-ai-lab-banner.png" alt="gittensor-ai-lab" width="100%">
</p>

# gittensor-ai-lab

**SN74 on Gittensor** — a **Blackwell-native** MoE/LLM inference runtime, co-designed and continuously improved by human engineers and AI agents to reach the hardware ceiling on NVIDIA's next generation of personal AI computers.

Real kernel and runtime engineering: reproducible, hardware-level inference-speed gains beyond existing serving stacks — rewarded by **verified, source-required contribution, not benchmark gaming**.

---

## Vision

**Blackwell only — by design.** We don't spread thin across vendors and GPU generations the way vLLM, SGLang, and llama.cpp must. We specialize on **NVIDIA Blackwell** (`sm_120` / `sm_121`) and drive it to the hardware ceiling — unified memory, GDDR7 / LPDDR5X bandwidth, the new SM. Generalists go wide; we go deep: **general across _models_ (Qwen, Gemma, …), uncompromising on one _architecture_.**

**Blackwell is the substrate of the next computer.** The **RTX Spark** (GB10) is NVIDIA's personal AI supercomputer — 128 GB of unified memory brings full **70B+ MoE inference to a desk**, no datacenter required. Jetson Thor puts it in robots; RTX 5090 and RTX PRO 6000 put it in every workstation. The future personal machine ships with a Blackwell-class accelerator running large local models — and it needs a runtime built for exactly that silicon.

**The demand is local AI.** Privacy, latency, cost, offline capability, sovereignty — inference is moving on-device, and fast. Whoever runs MoE/LLMs fastest on local Blackwell hardware owns that layer. **RTX Spark is our flagship**; the runtime that extracts the most from it becomes core infrastructure for the next generation of computing.

---

## Proven

**Qwen3-30B-A3B (Q4_K_M GGUF) runs end-to-end on an RTX PRO 6000 Blackwell (sm_120)**, decode optimized **0.60 → 134 tok/s (≈220×)** across 6 source-verifiable passes — **within 1.8× of llama.cpp** on the same model + GPU, output verified correct, **21.7 GB resident** (experts kept quantized, vs ~57 GB bf16).

Independently **verified on an RTX 5090** (sm_120, CUDA 13): clean build, `ctest` **5/5**, compute-sanitizer **0 errors**, same correct output, **187.61 tok/s** decode (the live `v0.2.0` frontier, up from 163.88 at `v0.1.0`) at **21.4 GB** — fits a 32 GB consumer card. On the same card `llama.cpp` runs **365.73 tok/s** (a **1.95×** gap, narrowing — our launch-bound decode doesn't fully ride the consumer boost clock yet; fusion is the lever). ([live dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) · [results](https://github.com/gittensor-ai-lab/sparkinfer/blob/main/bench/results/qwen3-30b-a3b_q4km_rtx5090.md))

| Pass | Optimization | tok/s |
|---|---|--:|
| baseline | dequantize all 128 experts | 0.60 |
| 1 | fused selected-expert Q4_K_M FFN — dequant only the routed experts on-read | 32.7 |
| 2 | decode GEMV — coalesced `[out,in]`, replaces M=1 tiled GEMM | 84.4 |
| 3 | CUDA-graph decode — capture once, replay per token | 118.7 |
| 4 | flash-decoding — KV-split + log-sum-exp combine | 133 |
| 5 | fused residual + RMSNorm | 134 |

Each pass is correctness-gated and reproducible from source — the unit the subnet rewards.

## Try it

On any Blackwell box (CUDA 12.8+) — the scripts auto-detect your GPU, fetch prebuilt binaries (or build from source), and download the model:

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer && cd sparkinfer
bench/scripts/bench.sh --download              # decode tok/s
bench/scripts/bench.sh --download --compare    # head-to-head vs llama.cpp, same GPU
bench/scripts/accuracy.sh --download           # accuracy: token-match / KL / perplexity vs llama.cpp
```

## Accuracy — verified against llama.cpp

Speed only counts if the output stays right. `accuracy.sh` teacher-forces a fixed text through **both engines on the same GGUF** and compares them position-by-position. On the RTX 5090 (Qwen3-30B-A3B Q4_K_M):

| metric | result | |
|---|---|---|
| top-1 token agreement vs llama.cpp | **98%** | (bar ≥ 90%) |
| mean KL(llama ‖ sparkinfer) | **0.14 nats** | distributions ~identical |
| perplexity (sparkinfer, exact) | **6.16** | matches the reference |

Same greedy choice as llama.cpp at **98 of 100** positions, KL ≈ 0.14 — well above the 90% bar. The same tool gates **every** PR against the live frontier (the auto-eval rejects any drop below the bar), so no kernel change can silently regress quality. ([how it works + full results](https://github.com/gittensor-ai-lab/sparkinfer/blob/main/bench/results/accuracy_qwen3-30b-a3b_q4km.md))

---

## The problem we are solving

Existing MoE inference stacks (vLLM, SGLang, llama.cpp) were designed for datacenter multi-GPU serving. They leave significant performance on the table on single-device unified-memory hardware:

- **RTX Spark** (128 GB LPDDR5X, ~273 GB/s, sm_121): no public attention kernel handles `head_dim=512` (Gemma 4 global layers); no serving stack exploits a full CUDA graph across routing + dispatch + FFN
- **RTX PRO 6000 Blackwell** (96 GB GDDR7, ~1.79 TB/s, sm_120): datacenter-class capacity *and* bandwidth on one card — a full 35B MoE plus all experts and a large KV cache stay resident, no paging
- **RTX 5090** (32 GB GDDR7, ~1.79 TB/s, sm_120): sync barriers between router and GroupGEMM break CUDA graph capture, adding launch overhead per MoE layer

SN74 rewards engineers who close these gaps with source-verifiable kernel contributions.

---

## Repos & scoring

The runtime is a single monorepo — **[sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer)** (kernels + MoE engine + runtime + benchmarks); the [agent](https://github.com/gittensor-ai-lab/sparkinfer-agent) autotuner stays separate. **Scoring is speedup-only:** SN74 pays each merged PR for its **verified frontier-delta speedup** — labeled **XL / L / M / S / XS** by the eval loop (or **BASELINE** for the first verified entry on a new model/target), scored the same wherever it lands, with **no per-subsystem budget**. **Non-speedup PRs — tooling, benchmarks, docs, refactors — are welcome but score 0.** The eval/scoring harness is **maintainer-owned** (changes there are gated to maintainers for scoring integrity).

| Path (in `sparkinfer`) | What |
|---|---|
| `kernels/` | CUDA kernels — flash-decode (hd128/256/512), decode GEMV, fused quantized MoE expert FFN, GEMM, RMSNorm, RoPE, GGUF dequant. Where inference speed is won. |
| `runtime/` | Scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward. |
| `moe/` | Sync-free MoE router + expert dispatch — on-device counts, CUDA-graph-ready. |
| `bench/` | Reproducible benchmarks + eval harness (the eval/scoring scripts are maintainer-owned). |

Separate repo: [`sparkinfer-agent`](https://github.com/gittensor-ai-lab/sparkinfer-agent) (NCU-driven autotuning) carries its own small org share.

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
- **Qwen3-30B-A3B Q4_K_M end-to-end on RTX PRO 6000 — decode 0.60 → 134 tok/s, within 1.8× of llama.cpp**, output verified; **RTX 5090 frontier 187.61 tok/s** (`v0.2.0`)
- Correctness validated against a double-precision reference (full model to ~1e-8)
- **Automated evaluation loop — live** — a bot builds each PR from source on an RTX 5090, gates correctness, scores the frontier-delta, and posts an `eval:<label>` verdict; results stream to a [public dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/)
- **Anti-gaming controls** — sensitive-path merge gate (eval/scoring harness maintainer-only), contributor denylist + sybil/copycat detection, opt-in on-device eval (maintainer greenlight)
- **`v0.2.0`** tagged — prebuilt `sm_120` / CUDA 13 binaries attached, with source-build fallback

**Next**
- **Gemma 4 26B-A4B end-to-end** (interleaved local/global, `head_dim=512`, dual RoPE) — the second basket model
- Head_dim-general flash-decode (128 / 256 / 512) + sliding-window mask
- Long-context decode benchmark (8K–32K); batched prefill; top-p / temperature sampling
- Held-out/fuzzed correctness shapes + a verifiable frontier ledger (secret holdout, clock-pinned tps)

**Then**
- Tensor-core CuTe DSL GroupGEMM + SwiGLU (TMA, `sm_120a` / `sm_121a`); FP8 / FP4 weights + KV cache
- Expert prefetch / eviction for models exceeding VRAM; RTX Spark unified-memory expert streaming
- NCU-driven autotuning loop ([sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent)) feeding SN74 reproducible scoring

---

## SN74 — how contributions are rewarded

This org is the inference-optimization arm of SN74 on Gittensor. The reward model is built to pay **real engineering, not benchmark gaming** — the lesson from SN14:

1. **Source-required, validator-rebuilt** — you submit source, never binaries or Docker images. The validator compiles your PR itself, so the measured artifact *is* your code, and copying earns nothing. (The prebuilt binaries in our [releases](https://github.com/gittensor-ai-lab/sparkinfer/releases) are a run convenience for users — not a submission format.)
2. **Frontier-delta rewards** — you're paid for the **verified marginal speedup you add over the current best**, not your rank. Copy-the-leader-plus-ε pays ≈ ε.
3. **Correctness-gated** — speed counts only if the output is *right*. The reference is a frozen, SHA-pinned **exact fp32/fp64 evaluation of the same quantized weights** (so quantization loss is shared and only kernel errors surface). On a fixed prompt set **plus secret holdout / fuzzed shapes**, a submission must clear three layers:
   - *per-kernel* — numerical match to an fp64 reference (dequant **bit-exact**; GEMV/attention within a calibrated fp tolerance);
   - *end-to-end, teacher-forced* — **≥ 99.x% next-token argmax agreement** + bounded logit KL (teacher-forcing avoids single-token cascade divergence);
   - *aggregate* — **perplexity within ε** of the reference on a frozen corpus.

   Tolerances are calibrated above the FP noise floor between two correct implementations; holdout inputs block prompt special-casing; **both basket models (Qwen + Gemma) must pass**. The verdict is a deterministic pass/fail, so validators converge on it.
4. **Multi-model basket** — wins must hold across Qwen3-MoE *and* Gemma 4; a single-model optimization is overfitting and doesn't count.
5. **Reproducible metric** — normalized roofline % + end-to-end tok/s on a canonical GPU class, N-run median + significance gate, so independent validators converge.
6. **Public ledger** — every frontier advance → `(Δ, author, commit)` is auditable.

### Impact labels — and how they age

Performance PRs (kernels / runtime / moe) are bucketed **XL · L · M · S · XS** by the **verified speedup over the live frontier**, assigned by the eval loop — not by hand. Each label carries a weight, superlinear early so breakthroughs dominate.

**These weights are not fixed — they mature with the runtime.** While there's large headroom, the reward sits in XL (new-SOTA / big wins). As the runtime approaches the hardware ceiling — where 10× wins no longer exist and a 3% gain is a genuine breakthrough — improvement is scored against **remaining headroom (roofline %)**, so a small absolute gain near the ceiling still maps to a high label, and org governance **rebalances the weights toward the smaller buckets**. The subnet keeps paying real progress instead of stalling once the easy wins are gone.

### Automated, on every PR

Evaluation runs itself. A bot polls open PRs every ~30 minutes and, for each greenlit PR, builds it
**from source** on an RTX 5090, gates **correctness** (token-match / KL vs llama.cpp), benchmarks
**decode speed**, and posts the **`eval:<label>`** verdict (XL · L · M · S · XS · BASELINE · none · REJECT) as
a PR comment — a **deterministic** function of the measurements, so independent validators converge
on it. Verdicts and the optimization journey stream to a [live dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/).
Non-speedup PRs (tooling, docs, refactors) are welcome but score 0. It **never merges** (manual, after review).

The eval runs **opt-in** — a maintainer (or the PR template's *Tested on RTX 5090* checkbox) greenlights a
PR with `test-on-5090`; everything else is `not-tested` and never reaches the GPU. The harness is hardened
against gaming: the scoring code is **maintainer-only** (CODEOWNERS + a sensitive-paths status check) and the
bot grades with it pinned to `origin/main`, a **denylist** auto-closes flagged accounts, and **copycat/sybil
detection** flags PRs that re-submit another author's diff. See
[`sparkinfer/eval`](https://github.com/gittensor-ai-lab/sparkinfer/tree/main/eval).
