#!/bin/bash
# Kept in scripts/ since 2026-09-06 (the closure pass ran it from a scratch directory).
# The prefix cache's capacity/hit curve sweep: for each arena budget (GiB)
# and conversation count C, a fresh service boot and serve_prefix_curve.py.
# Usage: prefix_curve_sweep.sh OUT_DIR "GIB..." "C..."
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${1:?OUT_DIR}; GIBS=${2:-"0.25 1.5"}; CS=${3:-"8 32 64"}
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"; cd "$ROOT" || exit 1
for gib in $GIBS; do
  for c in $CS; do
    export DGPP_SERVE_KNOBS="--max-concurrency 2 --kv-capacity 16384 --default-max-tokens 256 --queue-limit 8 --decode-graph --mtp --prefix-cache-gib $gib"
    export DGPP_SERVE_LOG="$OUT/gib${gib}_c${c}"
    "$ROOT/scripts/serve_run.sh" up > "$OUT/gib${gib}_c${c}.up.log" 2>&1
    if ! grep -q READY "$OUT/gib${gib}_c${c}.up.log"; then echo "FAIL: boot gib $gib c $c"; tail -3 "$OUT/gib${gib}_c${c}.up.log"; continue; fi
    python3 "$ROOT/scripts/serve_prefix_curve.py" 192.0.2.11 18080 "$c" "gib${gib}_C${c}" | tee -a "$OUT/curve.txt"
    "$ROOT/scripts/serve_run.sh" down > "$OUT/gib${gib}_c${c}.down.log" 2>&1
    grep -h "serve_rank" "$OUT/gib${gib}_c${c}.down.log" | awk '{print $1}' | sort -u | wc -l | sed 's/^/  distinct op-stream md5s: /'
  done
done
