#!/bin/bash
# Any family's decode profile on the fabric (2026-09-14; the Qwen procedure generalized):
# the world up with rank 0 under nsys (dgpp-cluster's --head-wrap), a warm request, the
# profiled client (default: one 400-token single-stream request; --client for another, e.g.
# a serve_load sweep at a concurrency), the world down (nsys writes the report on rank 0's
# SIGINT), and the per-step kernel breakdown from the report between instances of the
# once-per-replay marker kernel (publish_seq_kernel: the verify verdict's publication, one
# per replay whatever the batch; spec_commit_kernel is one per REQUEST in a batch).
#   fabric_profile.sh CONFIG OUT_DIR [--knobs "FLAGS"] [--marker KERNEL] [--client "ARGS after HOST PORT"]
#   e.g. --client "scripts/serve_load.py HOST PORT --concurrency 6 --classes code --max-tokens 320"
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
MARKER=publish_seq_kernel; KNOBS=""; CLIENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --marker) MARKER="$2"; shift 2;;
    --knobs) KNOBS="$2"; shift 2;;
    --client) CLIENT="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
HOST=$(jq -r '.http.bind_host // empty' "$CONFIG"); [[ -z "$HOST" || "$HOST" == "0.0.0.0" ]] && { HOST=$(dgpp_client_host) || exit 1; }
PORT=$(jq -r '.http.port // empty' "$CONFIG"); [[ -z "$PORT" ]] && { PORT=$(dgpp_http_port) || exit 1; }
cd "$ROOT" || exit 1
WRAP="nsys profile -t cuda --cuda-graph-trace=node -o $OUT/r0 --force-overwrite true"
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world" --head-wrap "$WRAP")
[[ -n "$KNOBS" ]] && up+=("--knobs=$KNOBS")
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up under nsys ($CONFIG) knobs: ${KNOBS:-none}"
grep -h 'model constructed\|graph variants warm\|scheduled verify depth on' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-160
echo "== warm + profiled requests"
python3 scripts/serve_bench.py "$HOST" "$PORT" 64 warm 2>&1 | tail -1
if [[ -n "$CLIENT" ]]; then
  # shellcheck disable=SC2086
  eval "python3 ${CLIENT//HOST PORT/$HOST $PORT}" 2>&1 | tee "$OUT/bench.log" | grep -E "aggregate|isolation|tok/s" | cut -c1-160
else
  python3 scripts/serve_bench.py "$HOST" "$PORT" 400 profiled 2>&1 | tail -2 | tee "$OUT/bench.log"
fi
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -2
for _ in $(seq 1 120); do [[ -f "$OUT/r0.nsys-rep" ]] && break; sleep 1; done
[[ -f "$OUT/r0.nsys-rep" ]] || { echo "no nsys report at $OUT/r0.nsys-rep"; exit 1; }
echo "== step breakdown (marker $MARKER)"
python3 scripts/nsys_step_breakdown.py "$OUT/r0.nsys-rep" --marker "$MARKER" --top 45 2>&1 | tr '\r' '\n' | grep -vE "Processing|^\s*$" | tee "$OUT/breakdown.txt"
