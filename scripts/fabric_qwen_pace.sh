#!/bin/bash
# The quick single-stream pace reading (2026-09-09, the decode session):
# the world up, two 200-token serve_bench requests (the second is the
# reading), rank 0's steady stats line, the world down. ~90 s per config;
# DGPP_* knobs in the environment reach every rank (dgpp-cluster forwards
# them), so a knob A/B is  KNOB=v scripts/fabric_qwen_pace.sh CONFIG OUT.
#   fabric_qwen_pace.sh CONFIG OUT_DIR [--knobs "FLAGS"] [--tokens N]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
KNOBS=""; TOKENS=200
while [[ $# -gt 0 ]]; do
  case "$1" in
    --knobs) KNOBS="$2"; shift 2;;
    --tokens) TOKENS="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
HOST=$(jq -r '.nodes[0]' "$CONFIG"); PORT=$(jq -r '.ports.http' "$CONFIG")
cd "$ROOT" || exit 1
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world")
[[ -n "$KNOBS" ]] && up+=(--knobs "$KNOBS")
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
env | grep '^DGPP_' | sort | tr '\n' ' ' > "$OUT/env.txt"; echo >> "$OUT/env.txt"
python3 scripts/serve_bench.py "$HOST" "$PORT" "$TOKENS" warm > /dev/null 2>&1
python3 scripts/serve_bench.py "$HOST" "$PORT" "$TOKENS" pace 2>&1 | grep 'decode pace' | tee "$OUT/pace.log"
python3 scripts/serve_bench.py "$HOST" "$PORT" "$TOKENS" pace2 2>&1 | grep 'decode pace' | tee -a "$OUT/pace.log"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" > "$OUT/down.log" 2>&1
grep -h 'stats: rank 0' "$OUT/world/serve_r0.log" | grep 'prefill 1 prompt' | sed 's/^.*| decode/decode/' | cut -c1-90 | tail -2
grep -h 'op streams' "$OUT/down.log"
