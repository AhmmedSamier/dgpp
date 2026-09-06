#!/bin/bash
# Kept in scripts/ since 2026-09-06 (the closure pass ran it from a scratch directory).
# The steady-state prefill ritual: fabric_prefill_repeat.sh OUT_DIR LEN...
# For each LEN: the first LEN of the hard-prose ids (tiled x4 above 2100),
# glm_gen_check on the four nodes with --prefill-repeat 3 (the timed prefill
# is the third), two decode steps; prints the prefill line and the 4-way
# generated-ids verdict.
set -u
ROOT=/home/user/workspace/dgpp
IDS=$ROOT/build-ci/fabric-runs/prefill_ids/hard_ids.csv
OUT=${1:?OUT_DIR}; shift
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
cd "$ROOT" || exit 1
for n in "$@"; do
  prompt=$(python3 - "$IDS" "$n" <<'PY'
import sys
ids = open(sys.argv[1]).read().strip().replace("\n", ",").split(",")
ids = [i for i in ids if i]
n = int(sys.argv[2])
print(",".join((ids * 20)[:n]))
PY
)
  "$ROOT/scripts/fabric_run.sh" --force --log-dir "$OUT/n$n" -- \
    --model unsloth/GLM-5.3-Flash-FP8 --prompt "$prompt" --steps 2 \
    --prefill-repeat 3 > "$OUT/n$n.out" 2>&1
  echo "== $n tokens"
  grep -h "rank 0 prefill:" "$OUT/n$n/r0.log" | tail -1
  grep -h "rank consistency\|rc=" "$OUT/n$n.out" | tail -2
  [ -f "$OUT/n$n/r0.log" ] || tail -5 "$OUT/n$n.out"
done
