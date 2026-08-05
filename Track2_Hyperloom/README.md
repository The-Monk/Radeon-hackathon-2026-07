# Hyperloom — an autonomous ROCm kernel-optimizer agent for AMD Radeon

**AMD AI DevMaster Hackathon · Track 2: Agentic AI**

Hyperloom is an AI agent that makes AMD Radeon GPUs faster at LLM inference — on its own. Point it at a Radeon GPU and it detects the exact silicon, maps the real instruction-set capability with the assembler (not marketing claims), builds a measured roofline, finds where kernels leave performance on the table, and **writes and tunes correctness-gated HIP kernels** to close the gap. It reasons across the whole stack — silicon → driver → runtime → compiler → kernel → format → serving — and isolates a fault at its *real* layer instead of guessing.

This isn't a demo agent. Hyperloom already produced **multiple correctness-clean, measured, shippable optimizations** on a Radeon AI PRO R9700 (gfx1201 / RDNA4), across the full inference stack:

| Stage | Optimization | Result (MEASURED, correctness-gated) |
|---|---|---|
| **Decode** | production int4 GEMV `k_mmvq_dot8_iu4` (block-per-row + LDS reduction, native `v_dot8_i32_iu4`) | **96–97% of the measured 631 GB/s DRAM roofline** — memory-bound and saturating, correctness-gated vs CPU reference. (Instruction-only A/B, dot8 vs dp4a at fixed shape: **2.2×**.) |
| **Prefill** | int4 2:4-sparse SWMMAC (`v_swmmac_i32_16x16x64_iu4`) full GEMM | **3.67×** vs int8 WMMA @ K=8192 — 88–95% of the raw ISA ceiling, `max_abs_err=0`, `test-backend-ops` 1141/1141 PASS |
| **Comms** | INT6 inline-compressed all-reduce (dual-GPU tensor-parallel) | exact-integer reduce (~5e-7 drift), ~2.5× compression, bypasses the RCCL gfx1201 tuning gap |

Every number is reproduced from source in `benchmarks/`, on real gfx1201 hardware, gated against a CPU reference before it was trusted.

---

## Why this fits "Agentic AI"

The track asks for **intelligent agents with reasoning and planning**. Hyperloom is exactly that, applied to a hard, verifiable domain where you can't fake the result — a kernel is either numerically correct and faster, or it isn't.

**The agent's method (encoded in `skill/SKILL.md`):**

1. **Measure, never assume.** Every claim is tagged MEASURED (proven this run, on the actual target) or ASSUMED (theory / a result from a different path). An assumption *finds* a path; it never *closes* one. This discipline repeatedly overturned "obvious" answers — e.g. a "washes-out" verdict for 2:4-sparse int4 turned into a shippable 3.67× once actually measured on the right kernel.
2. **Layered fault-isolation** (the OSI method, applied to the GPU stack). Name the layer a fault lives in, verify with *that* layer's ground-truth tool, work bottom-up. This is why the agent scans the silicon *first* — it nails L1 truth (via `llvm-mc`, which the assembler can't fake) before any higher layer can lie to it. It caught, for real: a perf-level-gated counter mistaken for "broken hardware," a compiler false-positive mistaken for a silicon feature, and an env-var trap mistaken for a broken model.
3. **The mission loop:** DETECT → MAP → FIND-GAPS → TUNE → FIX → VALIDATE, run per capability and per gap, until measured ≥ peak or the silicon's real limit is proven with numbers.

---

## What's in this repo

```
skill/        The agent itself — a standards-compliant AMD Agent Skill (passes amd/skills' own validator)
toolkit/      The agent's instruments: ISA census, disasm verification, unused-instruction sweep, kernel eval
kernels/      The three optimizations, as standalone correctness-gated benchmarks
  decode/       int4 dot8 vs dp4a decode GEMV
  prefill/      int4 2:4-sparse K64 SWMMAC full GEMM
  allreduce/    INT6 compressed all-reduce (numpy reference + design)
benchmarks/   One-command reproductions of every number in the table
results/       The gfx1201 ISA capability map + the unused-instruction audit
demo/          Watch the agent optimize a kernel end-to-end, live
```

## Practical value beyond the demo

- **Installable.** Hyperloom is packaged as an [Agent Skill](skill/SKILL.md) in AMD's own `amd/skills` standard — it passes `validate_skills.py` with zero errors. Any compatible coding agent can load it and optimize a Radeon kernel. It maps directly to AMD's planned `hyperloom-kernel-optimizer` catalog slot.
- **Real ecosystem contribution.** While building this, Hyperloom's method surfaced a genuine RDNA4 gap in AMD's own kernel-evaluation tool and filed it upstream with a fix: **AMD-AGI/Magpie#70** (compare-mode perf ranking weights CDNA metrics that read zero on RDNA4).
- **Honest about limits.** Where the silicon has a real floor (e.g. a fixed ~2.7% sparsity-decode cost that survives every software lever), Hyperloom reports the number and stops — no inflated claims.

## Reproduce it

Hardware: an AMD Radeon GPU on ROCm (validated on Radeon AI PRO R9700, gfx1201). See `benchmarks/README.md` for one-command reproductions. Every kernel correctness-gates against a CPU reference and prints its own throughput; the ISA claims are verified by disassembly, not assertion.

*Built on 2× Radeon AI PRO R9700 (gfx1201) + ROCm. Every optimization measured, correctness-gated, and — where it matters — shippable.*
