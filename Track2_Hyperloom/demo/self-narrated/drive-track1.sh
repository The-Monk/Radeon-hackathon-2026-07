#!/bin/bash
# Drive the LOCAL agent (Gwen-Agent-Core / agentworld) to produce the Track-1 self-narrated
# demo OF the Track-2 optimization work - every stage local on the R9700 via Lemonade.
set -u
D=/home/jmonk/rocky-hackathon/demo/self-narrated
MODEL="${1:-Lemonade/agentworld}"; TIMEOUT="${2:-3000}"
OPENCODE=/home/jmonk/.opencode/bin/opencode
export HIP_VISIBLE_DEVICES=0; unset ROCR_VISIBLE_DEVICES
STAMP=$(date +%Y%m%d-%H%M%S); LOG=$D/drive-$STAMP; mkdir -p "$LOG"
rm -f "$D/demo.mp4" "$D/DEMO-RESULT.md" "$D"/section*.mp3
{
  echo "You are the local Omni agent stack on one AMD Radeon R9700. Do the following task"
  echo "end-to-end using ONLY local models via Lemonade http://localhost:13305 (no cloud)."
  echo "Use your bash tool to run narrate.sh, curl the Lemonade image API, and ffmpeg."
  echo; echo "=== TASK ==="; echo
  cat "$D/TRACK1-WITH-TRACK2-TASK.md"
  echo; echo "=== EXECUTE NOW ==="
  echo "Work in $D. Script the VO, narrate each section with narrate.sh, make visuals (Flux or"
  echo "title cards), assemble demo.mp4 with ffmpeg, then write DEMO-RESULT.md with what ran"
  echo "locally. If a stage fails, log it honestly and continue - a partial real demo beats a fake one."
} > "$LOG/prompt.txt"
echo "== DRIVE Track1+2 self-narrated demo =="
echo "model=$MODEL timeout=${TIMEOUT}s logdir=$LOG"
START=$(date +%s)
timeout "$TIMEOUT" "$OPENCODE" run --agent hyperloom-primary -m "$MODEL" --format json \
    --dir /home/jmonk "$(cat "$LOG/prompt.txt")" >"$LOG/transcript.json" 2>"$LOG/run.err"
echo "== agent done rc=$? elapsed=$(( $(date +%s)-START ))s =="
echo "== artifacts produced =="
ls -la "$D"/demo.mp4 "$D"/section*.mp3 2>/dev/null | awk '{print "  "$5, $NF}'
echo "== agent's DEMO-RESULT.md =="
[ -f "$D/DEMO-RESULT.md" ] && cat "$D/DEMO-RESULT.md" || echo "  (none)"
echo "logdir: $LOG"
