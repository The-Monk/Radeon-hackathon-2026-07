# Track 1 self-narrated demo - RESULT (produced locally on one AMD Radeon R9700)

The artifact is produced BY the model stack it demonstrates - the Track 2 dual-play claim.
Every generative stage ran locally via Lemonade (:13305). No cloud.

## What ran locally
| stage | model / tool | result |
|-------|--------------|--------|
| VO script (4 sections) | Qwen-AgentWorld-35B-A3B (the Track 2 agent brain) | agent wrote the narration from CAMPAIGN-STATUS.md |
| Narration x4 | **kokoro-v1** (82M, on the card) | section1-4.mp3, 13.1/13.7/15.7/12.9s = 55.5s total, valid 24kHz |
| Title image | **Flux-2-Klein-9B** | card1.png (opening visual) |
| Assembly | ffmpeg | seg1 (Flux image + VO) + seg2-4 (title cards + VO) -> concat -> demo.mp4 |

## Deliverable
`demo.mp4` - 55.5s, 1280x720, ~912 KB. Section 1 = the Flux-generated card + narration;
sections 2-4 = captioned cards + narration.

## Honest note on the drive
The local agent (260s run) scripted the VO, narrated all 4 sections with kokoro, and generated
the Flux image - all locally. It did not land the final ffmpeg concat itself (attempted it, 31
ffmpeg calls in the transcript, but stopped before a valid demo.mp4). The ffmpeg assembly - pure
mechanical stitching, not a model task - was completed by the harness. The narration you HEAR and
the image you SEE are the local model stack (kokoro + Flux) on the R9700; the same silicon and
model family the Track 2 agent runs on.
