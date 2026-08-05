# Track 1 (self-narrated demo) OF Track 2 (agentic optimization) - combined task

## GOAL
Produce `demo.mp4` - a short self-narrated demo of the Track 2 agentic-optimization work,
generated and narrated ENTIRELY LOCALLY on the AMD Radeon R9700 via Lemonade (`:13305`). The
artifact is produced BY the model stack it demonstrates - that dual-play IS the submission claim
(core inference runs locally on the Radeon GPU).

## THE SUBSTANCE (Track 2) you are demoing - it is REAL and already done
The win-reproduction campaign: a local agent reproduced shipped gfx1201 kernel/route/policy/
measurement wins BLIND from spec, under an integrity-owning harness - all 4 grade classes proven.
Read the story to narrate from:
`/home/jmonk/gfx1201-native-model/bakeoff/contest/reproduction/CAMPAIGN-STATUS.md`
Key beats: (1) thesis = "it's instructions, not skill"; (2) 4 grade classes reproduced blind
(G1 kernel, G3 policy, G4 measurement, G2 route); (3) the honest G2 boundary - correctness
reproduced but perf failed until the perf instruction was added, then it hit 2.72x; (4) the
harness caught a cheat (agent read the reference) and the fix revealed the true result.

## PRODUCE (Track 1 pipeline - every stage LOCAL via Lemonade :13305)
1. **Script the voiceover** (use the agent LLM). Write a tight ~4-section narration (title,
   the thesis, the 4-classes result, the honest G2 boundary+fix) from CAMPAIGN-STATUS.md. Each
   section 2-3 sentences - kokoro narrates short clips best.
2. **Narrate** each section to MP3 with kokoro (proven on the card):
   `bash /home/jmonk/rocky-hackathon/demo/self-narrated/narrate.sh "<section text>" sectionN.mp3`
3. **Visuals** - generate 1-2 title/section cards with Flux-2-Klein via the Lemonade image API
   (`POST :13305/api/v1/images/generations`, model `Flux-2-Klein-9B-GGUF`). If Flux is slow or
   unavailable, fall back to plain ffmpeg title cards - do not block the whole demo on it.
4. **Assemble** - stitch the narration MP3s + visuals into `demo.mp4` with `ffmpeg` (use
   `build-demo-video.sh` as the skeleton, or ffmpeg directly: one video segment per section, its
   narration as the audio track, concatenated).

## DELIVERABLE
`/home/jmonk/rocky-hackathon/demo/self-narrated/demo.mp4` + a `DEMO-RESULT.md` listing exactly
what ran locally (which models, section count, each mp3 size, final mp4 duration/size). Every
stage on the R9700 - no cloud. An HONEST partial (e.g. narration + title-card video assembled,
Flux skipped) with a note on what worked and what didn't is a VALID result - do not fake a video.
