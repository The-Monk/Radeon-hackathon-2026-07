# Hyperloom — Project Specification

**AMD AI DevMaster Hackathon · Track 2: Development & Local Deployment of Private AI Agents**

Hyperloom is a private, locally-deployed AI agent that makes AMD Radeon GPUs faster at LLM
inference — autonomously. It detects the exact silicon, maps that silicon's real instruction-set
capability using the assembler as ground truth, finds where shipped kernels leave performance
unused, writes and tunes correctness-gated HIP kernels to close the gap, and measures the result
on the card.

Its core inference runs entirely on an AMD Radeon GPU. Nothing about its reasoning loop depends
on a cloud API.

---

## 1. Application scenarios

**Who this is for.** Anyone running LLM inference on AMD Radeon hardware who is leaving
performance on the table and does not have a kernel engineer on staff.

| Scenario | The problem today | What Hyperloom does |
|---|---|---|
| **A new Radeon part ships** | Vendor kernel libraries lag the silicon. RDNA4 (gfx1201) has instructions no shipped kernel uses, and library support is scoped to data-center parts. | Scans the ISA with the assembler, finds the unexploited instruction, writes a kernel that uses it. |
| **Local/private LLM serving** | Users who cannot send data to a cloud must self-host, and self-hosted throughput on prosumer cards is often far below the hardware's ceiling. | Raises measured decode/prefill throughput on the user's own card, with correctness gates so speed never costs accuracy. |
| **A performance claim needs auditing** | Speedup claims routinely measure the wrong thing — cache instead of DRAM, a strawman baseline, a dead code path. | Owns the measurement: prints working-set size, compares against a measured roofline, and flags its own invalid readings. |
| **Multi-GPU tensor parallelism** | RCCL has no gfx1201 tuning index, so the stock all-reduce falls back to unrelated tuning and stalls. | Routes around it by compressing the payload (INT6 with shared scale, exact-integer reduce). |

**Why it matters.** The prosumer/workstation long tail is exactly where vendor optimization
effort arrives last. An agent that can do this work unattended turns "wait for the vendor" into
"close it yourself tonight."

---

## 2. Agent architecture

```
                    ┌────────────────────────────────────────────────┐
                    │  CORE INFERENCE — 100% LOCAL, AMD Radeon GPU   │
                    │  Lemonade (:13305) serving a tool-calling      │
                    │  model on the R9700. No cloud in the loop.     │
                    └───────────────────────┬────────────────────────┘
                                            │
   ┌────────────────────────────────────────▼─────────────────────────────────────┐
   │                          MISSION LOOP (reason → plan → act)                  │
   │                                                                              │
   │   DETECT ──► SCAN ──► FIND GAP ──► WRITE ──► CORRECTNESS GATE ──► MEASURE    │
   │      │         │          │           │              │              │        │
   │      │         │          │           │              │              ▼        │
   │      │         │          │           │              │      ┌──────────────┐ │
   │      │         │          │           │              └─FAIL─┤  VERIFY THE  │ │
   │      │         │          │           │                     │ MEASUREMENT  │ │
   │      │         │          │           └─────────────────────┤ vs roofline, │ │
   │      │         │          │                                 │ working set  │ │
   │      │         │          │                                 └──────┬───────┘ │
   │      │         │          └── layered fault isolation               │        │
   │      │         │              L1 silicon → L2 driver → L3 runtime   │        │
   │      │         │              → L4 compiler → L5 kernel → L6 format │        │
   │      │         │              → L7 serving                          │        │
   │      │         │                                                    ▼        │
   │      │         └── ISA ground truth (llvm-mc: the assembler cannot lie)      │
   │      └── hardware.py: 18 AMD targets (RDNA2-next, CDNA1-4, Zen, XDNA, APU)   │
   └───────────────────────────────┬──────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                           ▼
  ┌───────────┐            ┌──────────────┐            ┌──────────────┐
  │  TOOLS    │            │   MEMORY     │            │  ESCALATION  │
  │ scan-isa  │            │ persists     │            │ stuck-detect │
  │ disasm    │            │ across       │            │ → structured │
  │ hipcc     │            │ attempts,    │            │   packet     │
  │ magpie    │            │ so a closed  │            │ → AMD-hosted │
  │ profile   │            │ path is not  │            │   assist     │
  │ bench     │            │ re-explored  │            │ → RE-GATED   │
  └───────────┘            └──────────────┘            │   LOCALLY    │
                                                       └──────────────┘
```

**The escalation path is deliberately shaped.** When the agent is stuck (N correctness-gated
failures, degenerate looping, or explicit self-report), it builds a *structured* packet — which
isolation layer, verbatim evidence, what was already tried — and asks for help. The answer is
treated as **assumed until it passes a local correctness gate on the actual Radeon GPU**. Core
inference never leaves the card; assistance is advisory and must earn its way in by measurement.

---

## 3. Core capabilities

**1. Silicon-accurate ISA mapping.** `toolkit/scan-isa-gfx.sh` probes the target with `llvm-mc`
and classifies every candidate instruction as REAL / EMULATED / REJECTED. Marketing claims and
header definitions both lie; the assembler does not. A 351-builtin sweep on gfx1201 found 54 real,
and three genuinely unexploited: `v_dot8_i32_iu4`, `v_dot2_f32_f16`, `v_dot4_f32_fp8`.

**2. Correctness before speed, without exception.** Every kernel is gated bit-exact against a CPU
reference *before* any timing is reported. A fast-but-wrong kernel exits non-zero. In the prefill
benchmark this is 3 kernels × 4 shapes, all `max_abs_err=0`, before a single number prints.

**3. Measurement integrity — the agent audits itself.** This is the capability we consider most
important, because it is the one most optimizers lack. The harness prints its own working-set size
and compares throughput against a *measured* roofline. A reading above roofline means cache was
measured, not DRAM — so the harness says so:

```
N=14336 K=4096  |  31.5 MiB (CACHE-RESIDENT!) | 792 GB/s | 126% of roofline
N=65536 K=4096  | 144.0 MiB (DRAM-honest)     | 604 GB/s |  96% of roofline
```

This caught a real error in our own submission: the decode benchmark had been measuring a
cache-resident working set and reporting 122% of roofline. An optimizer you can trust has to be
able to prove itself wrong.

**4. Layered fault isolation.** Faults are diagnosed at their real layer (silicon → driver →
runtime → compiler → kernel → format → serving) rather than guessed at across the stack.

**5. Portable across the AMD substrate.** `hardware.py` covers 18 AMD AI targets — RDNA2-next
through CDNA1-4, Zen CPU, XDNA NPU, and APUs — so the method is not gfx1201-specific even though
gfx1201 is where it is proven.

---

## 4. Model and local deployment plan

**Brain.** A tool-calling model served locally by **Lemonade** on `:13305`, pinned to one card so
it fits in 32 GB VRAM. The agent talks to it over an OpenAI-compatible endpoint at
`http://localhost:13305/v1`. Swapping the brain is a config change (`LEMONADE_URL`, model name),
not a code change.

**Everything generative is local.** The demo materials in this repository were produced on the
same single R9700 — narration by `kokoro-v1`, imagery by `Flux-2-Klein-9B` — through the same
Lemonade endpoint. The artifact is produced by the stack it documents.

**Deployment.**

```bash
git clone https://github.com/The-Monk/rocky-hackathon && cd rocky-hackathon
# 1. ROCm + hipcc for your Radeon target
hipcc --version
# 2. the mission loop, end to end on the card
./demo/run_demo.sh
# 3. reproduce any individual claim
cd kernels/decode && hipcc --offload-arch=gfx1201 -O3 decode_mmvq_iu4.hip -o d && ./d
```

Full environment configuration, startup guide and dependency list are in `README.md`;
per-claim reproduction commands are in `benchmarks/README.md`. A container build
(`container/Containerfile`) is provided for a pinned toolchain.

**Hardware validated on:** AMD Radeon AI PRO R9700 (gfx1201 / RDNA4), 32 GB, ROCm 7.x.

---

## 5. Inference-speed optimization on AMD Radeon GPU

Three measured, correctness-gated results across the inference stack. Every number below is
reproducible from a clean clone of the public repository, on the card.

| Stage | Optimization | Measured result |
|---|---|---|
| **Decode** | `k_mmvq_dot8_iu4` — native `v_dot8_i32_iu4`, one block per row, coalesced K-stride, shared-memory reduction | **604–613 GB/s = 96–97% of the measured 631 GB/s DRAM roofline.** Memory-bound and saturating. `max_rel_err=1.1e-04` PASS |
| **Prefill** | int4 2:4-sparse SWMMAC (`v_swmmac_i32_16x16x64_iu4`) full GEMM | **3.67× vs int8 K16 WMMA at K=8192** (90.5 vs 25.0 TOP/s) — 88–95% of the 3.90× raw-instruction ISA ceiling. All correctness gates `max_abs_err=0` |
| **Comms** | INT6 inline-compressed all-reduce, shared scale | **2.46× payload reduction vs fp16**, mean `rel_l2 = 0.0239`, routing around the RCCL gfx1201 tuning gap |

**On the decode number specifically.** The naive route — one thread per row — looks 2.2× faster
with the native instruction, and that A/B is real *as an instruction comparison*. But it is
uncoalesced: adjacent lanes land K/2 bytes apart. That is invisible while the weights fit in the
64 MiB Infinity Cache and catastrophic once they do not — past the cache, both routes collapse to
~17 GB/s and the advantage disappears entirely. The win is the **access pattern**, not the
instruction. We report the DRAM-honest production number and state the collapse plainly rather
than leaving a judge to discover it.

**Upstream contributions.** Work from this project has been contributed back:
ROCm/composable_kernel **#3759** (3-bug 2:4-sparse SWMMAC correctness fix, `err 352 → 0` on
gfx1201) and AMD-AGI/Magpie **#70**.

---

## 6. Compliance with track rules

- **Core inference runs locally on an AMD Radeon GPU.** The agent's brain is a local model served
  by Lemonade on the R9700. There is no cloud model in the reasoning loop.
- **Not dependent on a closed-source agent platform.** The mission loop, tools, memory and
  escalation logic are in this repository (`agent/`).
- **All tooling is ROCm-native** — hipcc, llvm-mc, rocprof/metrix, Magpie.
