#!/bin/bash
# The vLLM DGX Spark recipe's own benchmark (tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark,
# bench/v41bench.py: prompt set v1, temperature 0, thinking off, streaming, decode tok/s after the
# first token, the server's usage counts) run against a dgpp world — the like-for-like comparison
# with the numbers that recipe reports. Boots the world from a cluster config (--knobs for engine
# flags), runs the client at the given concurrency levels and prefill targets, then takes the world
# down. The client script is not vendored: pass its path (--bench), fetched from the recipe's repo.
#   fabric_v41bench.sh CONFIG OUT_DIR --bench PATH/v41bench.py [--levels 1,2,6] [--prefill 2000,8000]
#                      [--label NAME] [--knobs "FLAGS"]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
BENCH=""; LEVELS="1,2,6"; PREFILL="2000,8000"; LABEL="dgpp"; KNOBS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2;;
    --levels) LEVELS="$2"; shift 2;;
    --prefill) PREFILL="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --knobs) KNOBS="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -f "$BENCH" ]] || { echo "--bench PATH/v41bench.py is required" >&2; exit 2; }
mkdir -p "$OUT"
HOST=$(jq -r '.http.bind_host // empty' "$CONFIG"); [[ -z "$HOST" || "$HOST" == "0.0.0.0" ]] && { HOST=$(dgpp_client_host) || exit 1; }
PORT=$(jq -r '.http.port // empty' "$CONFIG"); [[ -z "$PORT" ]] && { PORT=$(dgpp_http_port) || exit 1; }
MODEL=$(jq -r '.model' "$CONFIG")
cd "$ROOT" || exit 1
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world")
[[ -n "$KNOBS" ]] && up+=("--knobs=$KNOBS")
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up ($CONFIG) knobs: ${KNOBS:-none}"
grep -h 'model constructed\|graph variants warm\|scheduled verify depth on\|memory plan total' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
echo "== v41bench levels $LEVELS prefill ${PREFILL:-none}"
python3 "$BENCH" --base "http://$HOST:$PORT/v1" --model "$MODEL" --label "$LABEL" --out "$OUT" --levels "$LEVELS" --prefill "$PREFILL" 2>&1 | tee "$OUT/v41bench.log"
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -3 | tee "$OUT/down.log"
grep -h 'scheduled verify depth summary' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-300
grep -h 'stats: rank 0' "$OUT/world/serve_r0.log" | grep -v 'decode 0.0 tok/s' | tail -1 | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-220
