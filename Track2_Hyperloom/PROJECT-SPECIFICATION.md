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
  │  TOOLS    │            │   MEMORY     │            │ ESCALATION*  │
  │ scan-isa  │            │ persists     │            │ stuck-detect │
  │ disasm    │            │ across       │            │ → structured │
  │ hipcc     │            │ attempts,    │            │   packet     │
  │ magpie    │            │ so a closed  │            │ → AMD-hosted │
  │ profile   │            │ path is not  │            │   assist     │
  │ bench     │            │ re-explored  │            │ → RE-GATED   │
  └───────────┘            └──────────────┘            │   LOCALLY    │
                                                       └──────────────┘
```

\* **The escalation path is designed but not wired in.** `agent/escalation.py` exists and its
stuck-detector and packet-builder are unit-tested offline, but nothing in the agent loop calls it
and no cloud endpoint is configured. It is shown here because it is the intended shape, not
because it runs today.

**Why it is shaped that way.** When the agent is stuck (N correctness-gated
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

**Brain.** `Qwen-AgentWorld-35B-A3B` (35B total, ~3B active per token — a mixture-of-experts
model), quantized to **Q4_K_XL GGUF**, ~20.8 GB on disk and ~27 GB resident in VRAM once loaded.
Served locally by **Lemonade** on `:13305`, pinned to one card so it fits in 32 GB. It is chosen
for reliable tool-calling rather than raw size: an earlier 35B dense candidate failed the same
agentic fixture this model passed. The agent talks to it over an OpenAI-compatible endpoint at
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

Full environment configuration, startup guide and dependency list are in `README.md` (section "Environment configuration, startup, and dependencies");
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

---

## 7. Where to find each thing the Track 2 criteria ask for

This section exists so a reviewer can locate evidence directly rather than infer it.
Every row names a file in this repository or a timestamp in the demo video.

### 7.1 Minimum functional requirements (the track asks for at least 2 of 5)

Hyperloom implements **four**.

| Capability | Implemented | Where |
|---|---|---|
| **Local knowledge retrieval (RAG)** | Yes | `agent/optimizer-agent/memory.py` — `ingest_knowledge()` builds a corpus from documents the agent reads and from its own logs; `search_knowledge()` retrieves the most relevant chunks before it decides. Storage is plain JSONL under `memory/`, durable across runs and auditable. The corpus is not handed to the agent; it grows it. |
| **Tool invocation** | Yes | `agent/optimizer-agent/agent.py` drives a tool-call loop. Toolkit in `toolkit/`: `scan-isa-gfx.sh` (ISA ground truth via `llvm-mc`), `disasm-gfx.sh` (verify emitted instructions), `profile-datapath-gfx.sh`, `magpie.sh`, plus `hipcc` builds and benchmark execution. **Demo video 0:25–1:24** shows the agent declining to answer a hardware question from memory and emitting a real `scan_isa` tool call instead. |
| **Multi-step task planning** | Yes | The mission loop — DETECT → SCAN → FIND-GAPS → TUNE → FIX → VALIDATE → **AUDIT THE MEASUREMENT** — encoded in `skill/SKILL.md` and visible end to end in the demo video. Fault diagnosis is planned across seven layers (silicon → driver → runtime → compiler → kernel → format → serving) rather than guessed at. |
| **Local multi-turn memory** | Yes | `memory.py` — `remember(note, tags)` / `recall(query)`. `agent.py:run_attempt()` starts each attempt with a fresh context while **memory persists on disk across attempts**, and a diagnosis step between attempts is seeded from what previous attempts recorded. That is what stops it re-exploring a path it already closed. |
| Permission control & privacy | Partial | Private by construction: the agent's brain runs locally on the Radeon GPU, and **no inference — core or otherwise — leaves the machine**. The agent does carry optional `web_search` / `web_fetch` lookup tools; they are **disabled by default** and require `HYPERLOOM_ALLOW_WEB=1`, so a default run makes no outbound request at all. An escalation path exists (`agent/escalation.py`) and is deliberately gated — any external answer is treated as *assumed* until it passes a local correctness gate on the actual GPU. There is no separate permissions UI, so this is claimed as partial. |

### 7.2 Evaluation criteria

| Criterion | Evidence |
|---|---|
| **Task positioning & creative scenarios** | The agent operates in a domain where results cannot be faked: a kernel is either numerically correct and faster, or it is not. It targets the prosumer/workstation long tail, where vendor optimization arrives last — a new Radeon part ships, the kernel libraries lag it, and this closes the gap without waiting. |
| **Core capabilities — task decomposition, tool invocation, RAG, memory** | All four present; see 7.1 for file-level locations. |
| **Multi-turn interaction** | Demonstrated directly in the demo video: a four-turn conversation in one context, where the agent calls tools, is corrected by one of them, and refers back to what earlier turns established. Underneath, the same loop runs unattended: repeated attempts against the same objective, each with a fresh context, with durable memory and a grounded diagnosis step between them (`agent.py:run_attempt`, `diagnose`). Stated plainly — this is a multi-turn *work* loop, not a chat interface. |
| **Core inference on AMD Radeon GPU** | **Demo video 0:25–1:24**: the 35B tool-calling brain loads onto the card, Radeon VRAM goes from ~740 MB to ~27.7 GB, and the tool call is issued from it. Served locally by Lemonade. No remote API is involved in any core function. |
| **Targeted optimization for inference speed** | Three measured, correctness-gated results across the inference stack — decode at 96–97% of the measured 631 GB/s DRAM roofline, prefill 3.67× via int4 2:4-sparse SWMMAC, and an INT6 compressed all-reduce for the dual-GPU path. Reproduction commands in `benchmarks/README.md`; the routing layer and its measured abstain threshold are shown in the video at **1:24–2:07**. |

### 7.3 What this submission does not claim

Stated because a claim a reviewer can disprove is worth less than one they can check.

- There is **no graphical chat UI**. Interaction is conversational but terminal-based — see the multi-turn session in the demo video — and the agent also runs as an unattended work loop.
- Permission control is **partial** — private by construction, but without a dedicated permissions layer.
- The Radeon **cloud** bonus is **not claimed**. `agent/escalation.py` was written against that bonus and its docstring still says so, but the module is not wired into the agent loop and no cloud endpoint is configured; core inference runs entirely on local Radeon hardware.
- The prefill routing layer is **opt-in and off by default**, and validation of the individual routes is ongoing. The three headline results above are measured on the default path and reproduce from a clean clone.

---

## 8. The kernel work itself, and how it was validated

The three results in §5 are not one-off benchmarks. They live in a working fork of
llama.cpp targeting gfx1201, on a branch (`roc8`) that carries the shipping kernels,
the routing layer, and the test harness used to gate them. This section describes what
is in it and — more importantly — how it was checked, because the checking is where
most of the engineering went.

### 8.1 What is on the branch

| Area | Work |
|---|---|
| **Decode** | `k_mmvq_dot8_iu4` — native `v_dot8_i32_iu4`, one block per row, coalesced K-stride, shared-memory reduction |
| **Prefill routing** | Per-format hipBLASLt routes for Q1_0, Q2_0, Q4_K, Q8_0, F8E4M3, MXFP8, MXFP6, IU4, F16 — each opt-in, each soft-failing back to the stock kernel, each gated on a measured M threshold |
| **Comms** | INT6 inline-compressed all-reduce for the dual-GPU tensor-parallel path |
| **Sparsity** | int4 2:4-sparse SWMMAC GEMM (`v_swmmac_i32_16x16x64_iu4`) |

### 8.2 Two defects found by validation, not by use

Both were invisible to the testing that existed before, and each is a lesson about
what a given test can and cannot see.

**A Q4_K decode kernel that produced garbage.** A micro-optimisation replaced a
per-thread activation sum with the block sum that `q8_1` already stores. Those are not
the same quantity at that call site: each thread owns a quarter of the block, so all
four threads added the whole block's sum and the min term came out **4× too large**.
Every Q4_K model generated garbage at decode — `"The capital of France is"` produced
`|  Term???????????????`.

It had passed validation because the validation was perplexity, and
`llama-perplexity` runs at `n_batch=512`, so every matmul goes through the `mmq` path
and the broken `mmvq` decode path is **never executed**. The measurement was structurally
incapable of seeing the bug. Fix: revert the hunk; `test-backend-ops -o MUL_MAT` goes
from 1113 OK / 28 q4_K FAIL to **1143 OK / 0 FAIL**, and the model says `Paris.`

**A weight cache that returned another model's weights.** Every hipBLASLt route cached
converted weights keyed on the weight tensor's device address, with no invalidation
anywhere. Once a buffer was freed, a later allocation at the same address hit a stale
entry. It never fires in single-model single-process use — which is exactly what
perplexity and ordinary generation exercise — but it fires on multi-model benchmarks and
on server model swaps. Fix: the routes register an invalidator and the CUDA backend calls
it on buffer teardown, so the cost is zero on the hot path. Q8_0 int8 with the cache
enabled goes from **35 OK / 12 FAIL to 46 OK / 0 FAIL**; the same for Q1_0 and IU4.

### 8.3 How claims are gated

- **A clean-tree build gate.** `verify-roc8-promotion.sh` builds the branch from a git
  worktree containing committed content only, then runs `test-backend-ops`. This exists
  because a local build can quietly compile untracked files that no clean clone has —
  which had already happened once.
- **Correctness before throughput, always.** Every kernel checks itself against a CPU
  reference and exits non-zero if it fails, before printing a single timing.
- **Measurement audited against a measured roofline.** Working-set size is printed
  beside every bandwidth figure, and a reading above the DRAM roofline is reported as
  cache-residency rather than as a result.
- **Independent adversarial review.** Both defects above were found by a review whose
  brief was to *refute* the existing conclusions rather than confirm them. One of them
  overturned a hypothesis the author was confident in.

### 8.4 What is not claimed

The prefill routes are **opt-in and off by default**, and validation of the individual
routes is ongoing. A full sweep across all nine found real problems that are not yet
fixed: the F16 route is a net regression at production shapes, the Q8_0 fp8 mode fails
its own correctness tolerance in production configuration, and two routes alter generated
text while having no coverage in `test-backend-ops` at all.

So the branch ships them disabled. The three headline results in §5 are measured on the
**default** path and reproduce from a clean clone; the routing layer is presented as work
in progress, because that is what it is.

---

## 9. Serving throughput: auto-tuned continuous batching

Kernel work raises the ceiling for one stream. Most of the value of a locally-deployed
agent, though, shows up when several requests are in flight — and there the lever is
continuous batching, whose optimal parallelism is **per-model, VRAM-coupled, and not
monotonic**. So Hyperloom measures it rather than picking a number.

`scripts/auto-batch-serve.sh` (in the llama.cpp fork) computes the KV-bounded search
space from the model's own dimensions, sweeps `-np` under pinned clocks, detects the
throughput knee, caches the answer per `(model, context)`, and launches the server with
that parallelism.

### Measured — Ternary-Bonsai-8B-Q2_0, gfx1201, `npp=32 ntg=128`

| streams (`-np`) | decode throughput | vs single stream |
|---|---|---|
| 1 | 154.9 t/s | 1.00× |
| 8 | 509.3 t/s | 3.29× |
| 16 | 889.9 t/s | 5.75× |
| 32 | 1333.8 t/s | 8.61× |
| 48 | 1917.7 t/s | 12.39× |
| 64 | 1859.6 t/s | 12.01× |
| 96 | 2503.8 t/s | 16.17× |
| 128 | 2481.3 t/s | 16.02× |
| 160 | 2683.5 t/s | 17.33× |
| **192** | **2764.1 t/s** | **17.85×** |

**Read this correctly: it is aggregate throughput, not per-request speed.** One stream
generates ~155 tokens/s; ninety-six streams together generate ~2489, which is ~26 t/s
each. Batching buys you *total work done*, and it costs per-request latency. For an agent
serving several concurrent tasks — the case this track is about — throughput is the
figure that matters, but it should never be quoted as though a single reply got 16×
faster.

**Why the sweep is not optional — including when it misleads you.** The curve is not
monotonic: 64 streams (1859.6) is *slower* than 48 (1917.7) before recovering. A guessed
`-np` can land in that trough. And the trough moves with the model — the same script picks
`-np 16` for Ternary-Bonsai-27B at 4096 context per slot, where VRAM binds long before
throughput does.

It also caught an error in our own earlier work. A previous sweep concluded there was a
throughput *cliff* at 128 streams and treated 96 as the peak. Re-running with a larger KV
budget (`-c 65536` rather than `32768`) shows no cliff at all: 128 is flat within run-to-run
variance, and throughput keeps climbing to **17.85× at 192 streams**. The original "cliff"
was the context budget binding, not the hardware — a limit of the measurement, read as a
property of the GPU.

We did not test beyond 192, so the true ceiling is unknown; 17.85× is a floor, not a peak.
Per-stream latency degrades as expected across the range — at 192 streams each sees about
14 t/s against 155 for a lone stream.
