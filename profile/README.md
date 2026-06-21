# gittensor-ai-lab

**SN74 on Gittensor** — where human engineers and AI agents co-design and continuously improve kernels, memory systems, and routing for RTX Spark, RTX 5090, and Jetson Thor.

We build real kernel and runtime engineering: reproducible, hardware-level latency and efficiency gains that go beyond existing serving stacks.

---

## The problem we are solving

Existing MoE inference stacks (vLLM, SGLang, llama.cpp) were designed for datacenter multi-GPU serving. They leave significant performance on the table on single-device unified-memory hardware:

- **RTX Spark** (128 GB LPDDR5X, 273 GB/s, sm_100): no attention kernel handles `head_dim=512` (Gemma 4 global layers); no serving stack exploits the full CUDA graph pipeline across routing + dispatch + FFN
- **RTX 5090** (32 GB GDDR7, 1.79 TB/s, sm_100): sync barriers between router and GroupGEMM break CUDA graph capture, adding kernel launch overhead per MoE layer

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
| **RTX Spark** | 128 GB LPDDR5X unified | ~273 GB/s | Blackwell sm_100 |
| **RTX 5090** | 32 GB GDDR7 | 1.79 TB/s | Blackwell sm_100 |
| Jetson Thor | unified | — | Blackwell |

Primary target is RTX Spark — the first consumer device with enough unified memory to run full 70B+ MoE models without expert eviction.

---

## Target models

| Model | Experts | top-k | Why it matters |
|---|---|---|---|
| **Qwen3.5-35B-A3B** | 256 | 8 + 1 shared | Extreme expert count; skinny GEMM 2048→512; 2 KV heads (8:1 GQA, hd=128) |
| **Gemma 4 26B-A4B** | 128 | 8 + 1 shared | `head_dim=512` in global layers — no public kernel handles this; interleaved local/global attention |

---

## Kernel engineering focus

**No Triton.** Native C++ and CUDA with [CuTe DSL](https://github.com/NVIDIA/cutlass) (CUTLASS tensor layout DSL) throughout. CuTe exposes TMA, WGMMA, persistent kernels, and epilogue visitor composition — the primitives that reach the performance ceiling on sm_100.

**Sync-free MoE.** Token counts from the router stay in GPU memory. No `cudaMemcpy`, no `cudaDeviceSynchronize` between routing and dispatch. The entire MoE forward pass — routing → GroupGEMM → SwiGLU epilogue — is captured in a single CUDA graph.

**CODA-style epilogue fusion.** Gate and input projections computed in one WGMMA pass with interleaved weight layout; SwiGLU applied in the accumulator register file before any global memory write. No intermediate [T, 2F] tensor touches DRAM.

---

## SN74

This organization is the engineering arm of SN74 on the Gittensor network. Validators score contributions by:

1. **Source diff** — kernel changes must be in version control, compilable from source
2. **NCU delta** — before/after Nsight Compute metrics (bandwidth utilization, tensor core throughput, occupancy)
3. **Benchmark delta** — measured latency improvement on reproducible hardware + model configuration
4. **No benchmark gaming** — frozen model weights (SHA256 pinned), no hidden training, no pre-built binaries
