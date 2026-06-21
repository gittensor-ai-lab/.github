# gittensor-ai-lab

**SN74 on Gittensor** — where human engineers and AI agents co-design and continuously improve kernels, memory systems, and routing for RTX Spark, RTX 5090, and Jetson Thor.

We build real kernel and runtime engineering: reproducible, hardware-level latency and efficiency gains that go beyond existing serving stacks.

---

## The problem we are solving

Existing MoE inference stacks (vLLM, SGLang, llama.cpp) were designed for datacenter multi-GPU serving. They leave significant performance on the table on single-device unified-memory hardware:

- **RTX Spark** (128 GB LPDDR5X, 273 GB/s, sm_121): no attention kernel handles `head_dim=512` (Gemma 4 global layers); no serving stack exploits the full CUDA graph pipeline across routing + dispatch + FFN
- **RTX PRO 6000 Blackwell** (96 GB GDDR7, ~1.79 TB/s, sm_120): datacenter-class capacity *and* bandwidth on one card — a full 35B MoE plus all experts and a large KV cache stay resident, no paging
- **RTX 5090** (32 GB GDDR7, 1.79 TB/s, sm_120): sync barriers between router and GroupGEMM break CUDA graph capture, adding kernel launch overhead per MoE layer

SN74 rewards engineers who close these gaps with source-verifiable kernel contributions.

---

## Repos

| Repo | What it does |
|---|---|
| [sparkinfer-kernels](https://github.com/gittensor-ai-lab/sparkinfer-kernels) | Native C++/CUDA + CuTe DSL kernels: flash decode (hd128/256/512, GQA), sync-free GroupGEMM + SwiGLU, quantization |
| [sparkinfer-runtime](https://github.com/gittensor-ai-lab/sparkinfer-runtime) | Scheduler, KV cache manager, CUDA graph engine, paged memory, MoE dispatch layer |
| [sparkinfer-moe](https://github.com/gittensor-ai-lab/sparkinfer-moe) | Sync-free MoE routing and expert dispatch: no CPU readback, end-to-end CUDA graph compatible |
| [sparkinfer-bench](https://github.com/gittensor-ai-lab/sparkinfer-bench) | Reproducible benchmarks: source-required builds, frozen model weights, hardware-level metrics |
| [sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent) | NCU-driven autonomous kernel optimization agent: profile → propose → compile → benchmark |

---

## Target hardware

| Device | Memory | Bandwidth | Architecture |
|---|---|---|---|
| **RTX Spark** (GB10) | 128 GB LPDDR5X unified | ~273 GB/s | Blackwell sm_121 |
| **RTX PRO 6000** (GB202) | 96 GB GDDR7 ECC | ~1.79 TB/s | Blackwell sm_120 |
| **RTX 5090** (GB202) | 32 GB GDDR7 | 1.79 TB/s | Blackwell sm_120 |
| Jetson Thor | unified | — | Blackwell sm_121 |

Primary target is RTX Spark — the first consumer device with enough unified memory to run full 70B+ MoE models without expert eviction. RTX PRO 6000 is the high-end discrete target: 96 GB at full GDDR7 bandwidth. (Consumer/workstation Blackwell is `sm_120`/`sm_121` — *not* `sm_100`, which is datacenter B200/GB200 and binary-incompatible.)

---

## Target models

| Model | Experts | top-k | Why it matters |
|---|---|---|---|
| **Qwen3.5-35B-A3B** | 256 | 8 + 1 shared | Extreme expert count; skinny GEMM 2048→512; 2 KV heads (8:1 GQA, hd=128) |
| **Gemma 4 26B-A4B** | 128 | 8 + 1 shared | `head_dim=512` in global layers — no public kernel handles this; interleaved local/global attention |

---

## Kernel engineering focus

**No Triton.** Native C++ and CUDA with [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) (CUTLASS tensor layout DSL) throughout. CuTe exposes TMA, WGMMA, persistent kernels, and epilogue visitor composition — the primitives that reach the performance ceiling on Blackwell (sm_120 / sm_121).

**Sync-free MoE.** Token counts from the router stay in GPU memory. No `cudaMemcpy`, no `cudaDeviceSynchronize` between routing and dispatch. The entire MoE forward pass — routing → GroupGEMM → SwiGLU epilogue — is captured in a single CUDA graph.

**Fused MoE GroupGEMM + SwiGLU.** Inspired by [NVIDIA's MoE fusion kernel work](https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/): gate and input projections fused into a single GroupGEMM pass with interleaved weight layout; SwiGLU applied in the accumulator register file before any global memory write. No intermediate [T, 2F] tensor touches DRAM. Unlike dense epilogue fusion (e.g. CODA), the MoE case must handle variable token counts per expert — solved by tracking counts in GPU memory and computing tile assignments on-device.

**Blackwell warp specialization.** Persistent kernels split warpgroups into dedicated producer (TMA async bulk copy) and consumer (WGMMA compute) roles. Producers pipeline the next tile into the 228 KB shared memory bank while consumers execute the current tile — compute and memory fully overlapped. On RTX Spark's bandwidth-constrained LPDDR5X this is the difference between running at memory peak and stalling on load latency.

---

## Roadmap

**Shipped**
- Five repos; portable CUDA kernels — flash-decode (incl. `head_dim=512`), RoPE, MoE router + sync-free SwiGLU expert FFN, GEMM, RMSNorm, quant — each verified to compile for `sm_120` (RTX 5090 / PRO 6000) and `sm_121` (RTX Spark / Jetson Thor)
- Sync-free MoE engine (on-device token counts, CUDA-graph-ready)
- Runtime: paged KV cache, scheduler, device query
- **Qwen3.5-35B-A3B end-to-end** — QK-norm, RoPE, paged GQA decode, routed top-8 + shared expert, LM head, greedy `generate()`; HF → sparkinfer weight converter
- Correctness validated against a double-precision reference (full model to ~1e-8)

**Next**
- Hardware bring-up on RTX 5090 / RTX PRO 6000 (native build, on-device runs, NCU profiles)
- Byte-exact Q4_K_M GGUF dequant; batched prefill; top-p / temperature sampling
- End-to-end CUDA graph capture of the decode step

**Then**
- Tensor-core CuTe DSL GroupGEMM + SwiGLU (TMA, `sm_120a`/`sm_121a`); FP8 / FP4 weights + KV cache
- Gemma 4 26B-A4B end-to-end (interleaved local/global, `head_dim=512`)
- Expert prefetch / eviction for models exceeding VRAM; RTX Spark unified-memory expert streaming
- NCU-driven autotuning loop ([sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent)) feeding SN74 reproducible scoring

---

## SN74

This organization is the engineering arm of SN74 on the Gittensor network. Validators score contributions by:

1. **Source diff** — kernel changes must be in version control, compilable from source
2. **NCU delta** — before/after Nsight Compute metrics (bandwidth utilization, tensor core throughput, occupancy)
3. **Benchmark delta** — measured latency improvement on reproducible hardware + model configuration
4. **No benchmark gaming** — frozen model weights (SHA256 pinned), no hidden training, no pre-built binaries
