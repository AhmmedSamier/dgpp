#!/bin/bash
# The prefill re-measurement (PLAN's M6 tail: "the prefill behind the time
# to first token — first re-measure it"): glm_gen_check across the four
# nodes with the same natural-text prompt cut to several lengths, two decode
# steps each, reading rank 0's "prefill: N tokens in X ms" line. Prompts are
# token ids (--prompt), so the lengths are exact and the arguments survive
# the ssh hop; the ids come from IDS_FILE (one line, comma-separated — a
# tokens file's prompt line from a world-1 run of the teacher text).
#
# Usage: scripts/fabric_prefill_check.sh IDS_FILE [OUT_DIR] [LENGTHS]
#   LENGTHS defaults to "64 256 1024 2048".
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IDS_FILE=$1
OUT=${2:-$ROOT/build-ci/fabric-runs/prefill_$(date +%Y-%m-%d_%H%M%S)}
LENGTHS=${3:-"64 256 1024 2048"}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
ALL=$(tr -d ' \n' < "$IDS_FILE")
TOTAL=$(echo "$ALL" | tr ',' '\n' | grep -c .)
echo "=== $TOTAL prompt ids available in $IDS_FILE"
STAGE=""
for n in $LENGTHS; do
  if [ "$n" -gt "$TOTAL" ]; then echo "skip $n: only $TOTAL ids"; continue; fi
  ids=$(echo "$ALL" | tr ',' '\n' | head -n "$n" | paste -sd, -)
  echo "=== prompt of $n tokens"
  scripts/fabric_run.sh $STAGE --log-dir "$OUT/n$n" -- --model unsloth/GLM-5.3-Flash-FP8 \
    --prompt "$ids" --steps 2 > "$OUT/n$n.out" 2>&1
  STAGE="--no-stage"
  grep -h "prefill:\|step 1:\|rank consistency" "$OUT/n$n/r0.log" "$OUT/n$n.out" 2>/dev/null | sed 's/^[0-9-]* [0-9:.]* INFO  //' | cut -c1-140 | head -4
done
echo "=== summary (rank 0)"
for n in $LENGTHS; do
  f="$OUT/n$n/r0.log"
  [ -f "$f" ] || continue
  python3 - "$f" "$n" <<'PY'
import re, sys
log, n = open(sys.argv[1], errors="replace").read(), int(sys.argv[2])
m = re.search(r"prefill: (\d+) tokens in (\d+)ms", log)
if m:
    toks, ms = int(m.group(1)), int(m.group(2))
    print(f"  {toks:>5} tokens: {ms:>7} ms prefill = {ms / toks:6.1f} ms/token")
else:
    print(f"  n{n}: no prefill line")
PY
done
echo "=== prefill check done: $OUT"
