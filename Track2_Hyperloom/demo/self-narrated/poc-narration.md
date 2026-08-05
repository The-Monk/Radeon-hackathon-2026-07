# POC video — narration script

Voice: `kokoro-v1`, `af_sky`, generated locally on the R9700 via Lemonade.
Each section is timed to the terminal footage it sits over.

Rule for this script: **do not narrate a number the screen does not show.** The
previous demo hardcoded "1.49x" into a slide after the measurement had changed.
Every figure below appears on screen in the same shot.

---

## 1 · COLD OPEN  (Flux card, ~14s)

> This video was produced and narrated entirely on one AMD Radeon AI PRO R9700.
> The narration, the imagery, and every number you are about to see were generated
> on that card. Nothing here touched a cloud service.

## 2 · THE THESIS  (Flux card / title, ~20s)

> Hyperloom is an agent that makes AMD Radeon GPUs faster at large language model
> inference, on its own. It reads the silicon's real instruction set, finds what
> shipped kernels never use, writes the kernel to close the gap, and then does the
> part most optimizers skip. It audits its own measurement.

## 3 · THE BRAIN COMES UP ON THE CARD  (seg-launch, ~45s)

> First, the agent's brain. A thirty-five billion parameter tool-calling model,
> served locally by Lemonade. Watch Radeon memory: seven hundred megabytes before,
> thirty-one gigabytes after. That is the agent's reasoning engine, resident on the
> GPU. Core inference runs on the card, which is what this track requires.

## 4 · IT REFUSES TO GUESS  (seg-launch, ~40s)

> Now the important part. We ask it a hardware question it could easily bluff.
> And it does not answer. It calls a tool instead.
>
> That restraint is the whole design. Asked about this same instruction, two smaller
> local models answered confidently and wrongly — one called four-bit integers
> eight-bit, the other sixteen. Model recall is not evidence. So the agent runs the
> assembler, and the assembler cannot lie.

## 5 · WRITING THE KERNEL  (seg-kernel, ~60s)

> Here is the naive kernel. One thread per row. It uses the native eight-wide int four
> dot product, so it looks like a win — and in a narrow test, it is one.
>
> But look at the addresses instead of the instruction. Each thread owns an entire row,
> so neighbouring lanes read memory two kilobytes apart. That is uncoalesced. It stays
> invisible while the weights fit inside the sixty-four megabyte Infinity Cache.
>
> The fix is structural. One block per row, threads striding along K so neighbouring
> lanes read neighbouring data, then a shared-memory reduction. Same instruction,
> different access pattern. The access pattern was always the real story.

## 6 · PROVE IT, THEN MEASURE IT  (seg-kernel, ~35s)

> Claiming an instruction is used is not evidence either. So we disassemble and look
> for it. Then the kernel checks itself against a CPU reference. A kernel that is fast
> and wrong exits non-zero before it is ever allowed to report a number.

## 7 · THE AUDIT  (demo-operation step 5, ~40s)

> And now the part that matters most. Run the same benchmark on a working set small
> enough to fit in cache, and it reports a hundred and twenty-five percent of the
> memory roofline. That is not a record. It is proof that cache was measured instead
> of memory.
>
> This caught a real error in this submission. The decode benchmark had been reporting
> a hundred and twenty-two percent of roofline as though it were a result. An optimizer
> you can trust has to be able to prove itself wrong.

## 8 · THE REAL NUMBERS  (demo-operation steps 6–8, ~35s)

> With a working set too large to hide in cache: ninety-seven percent of the measured
> memory roofline, correctness passed. Memory-bound and saturating — there is almost
> nothing left in this kernel.
>
> Prefill, three point six seven times faster using two-four sparse swim-mac, every
> correctness gate at zero error. And communications, an int six compressed all-reduce
> that routes around a gap in RCCL's support for this GPU.

## 9 · CLOSE  (Flux card, ~18s)

> Detect the silicon. Read its real instruction set. Find the gap. Write the kernel.
> Gate it for correctness. Measure it — then check the measurement.
>
> Every number in this video is reproducible from a clean clone of the public
> repository, on the card. Hyperloom, for AMD Radeon.
