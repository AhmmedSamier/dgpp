#!/bin/bash
# The decode profile ritual (docs/qwen38_flash_next_plan.md Q8, 2026-09-09):
# the world up with rank 0 under nsys (dgpp-cluster's --head-wrap), one
# warm request, then a 400-token single-stream request (the profiled
# steps), the world down (nsys writes the report on rank 0's SIGINT), the
# per-step kernel breakdown from the report.
# --prefill LEN: the profiled request is one LEN-token prompt through
# serve_prefill_probe.py instead (the breakdown of the last burst).
#   fabric_qwen_profile.sh CONFIG OUT_DIR [--marker KERNEL] [--knobs "FLAGS"] [--prefill LEN]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
MARKER=spec_commit_kernel; KNOBS=""; PREFILL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --marker) MARKER="$2"; shift 2;;
    --knobs) KNOBS="$2"; shift 2;;
    --prefill) PREFILL="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
HOST=$(jq -r '.nodes[0]' "$CONFIG"); PORT=$(jq -r '.ports.http' "$CONFIG")
cd "$ROOT" || exit 1
WRAP="nsys profile -t cuda --cuda-graph-trace=node -o $OUT/r0 --force-overwrite true"
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world" --head-wrap "$WRAP")
[[ -n "$KNOBS" ]] && up+=(--knobs "$KNOBS")
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up under nsys ($CONFIG)"
grep -h 'model constructed\|graph variants' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-160
echo "== warm + profiled requests"
if [[ -n "$PREFILL" ]]; then
  python3 scripts/serve_prefill_probe.py "$HOST" "$PORT" "$PREFILL" --repeat 2 2>&1 | tee "$OUT/bench.log"
else
  python3 scripts/serve_bench.py "$HOST" "$PORT" 64 warm 2>&1 | tail -1
  python3 scripts/serve_bench.py "$HOST" "$PORT" 400 profiled 2>&1 | tail -2 | tee "$OUT/bench.log"
fi
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -3
for _ in $(seq 1 120); do [[ -f "$OUT/r0.nsys-rep" ]] && break; sleep 1; done
[[ -f "$OUT/r0.nsys-rep" ]] || { echo "no nsys report at $OUT/r0.nsys-rep"; exit 1; }
echo "== step breakdown"
mode=(--marker "$MARKER"); [[ -n "$PREFILL" ]] && mode=(--burst)
python3 scripts/nsys_step_breakdown.py "$OUT/r0.nsys-rep" "${mode[@]}" --top 45 2>&1 | tee "$OUT/breakdown.txt"
