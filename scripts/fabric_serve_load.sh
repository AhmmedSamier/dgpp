#!/bin/bash
# Any family's served decode measures on the fabric (2026-09-14, the scheduled
# verify depth's A/B): boot the world from a cluster config (with --knobs for the
# engine flags under test), the greedy transcripts (compared against a reference
# when given), the MTP acceptance per prompt class (the scheduler's retired
# lines), the concurrency probe (serve_load at the given concurrencies), the
# world down with its op-stream md5s, and the engine's scheduled-depth summary.
# One world at a time on the fabric.
#   fabric_serve_load.sh CONFIG OUT_DIR [--compare REF_TRANSCRIPTS.json] [--knobs "FLAGS"]
#                        [--concurrency 1,2] [--isolation C] [--no-classes] [--no-transcripts]
#   --isolation C: serve_load's transcript-isolation check (prompt 0 alone, then beside C-1 others)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
COMPARE=""; KNOBS=""; CONC="1,2"; CLASSES=1; TRANSCRIPTS=1; ISOLATION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compare) COMPARE="$2"; shift 2;;
    --knobs) KNOBS="$2"; shift 2;;
    --concurrency) CONC="$2"; shift 2;;
    --no-classes) CLASSES=0; shift;;
    --no-transcripts) TRANSCRIPTS=0; shift;;
    --isolation) ISOLATION="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
HOST=$(jq -r '.http.bind_host // empty' "$CONFIG"); [[ -z "$HOST" || "$HOST" == "0.0.0.0" ]] && { HOST=$(dgpp_client_host) || exit 1; }
PORT=$(jq -r '.http.port // empty' "$CONFIG"); [[ -z "$PORT" ]] && { PORT=$(dgpp_http_port) || exit 1; }
cd "$ROOT" || exit 1
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world")
[[ -n "$KNOBS" ]] && up+=("--knobs=$KNOBS")  # the = form: a knob that starts with -- is a value, not a flag
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up ($CONFIG) knobs: ${KNOBS:-none}"
grep -h 'model constructed\|graph variants warm\|scheduled verify depth on\|listening' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
rc=0
if [[ $TRANSCRIPTS -eq 1 ]]; then
  echo "== greedy transcripts"
  cmp=(); [[ -n "$COMPARE" ]] && cmp=(--compare "$COMPARE")
  python3 scripts/serve_greedy_transcript.py "$HOST" "$PORT" --out "$OUT/transcripts.json" "${cmp[@]}" 2>&1 | tee "$OUT/transcripts.log" || rc=1
fi
if [[ $CLASSES -eq 1 ]]; then
  echo "== MTP acceptance per class (greedy, 300 tokens; the scheduler's retired lines)"
  python3 scripts/serve_mtp_classes.py "$HOST" "$PORT" --out "$OUT/mtp_classes.json" --log "$OUT/world/serve_r0.log" 2>&1 | tee "$OUT/mtp_classes.log"
fi
if [[ -n "$CONC" ]]; then
  echo "== concurrency (serve_load, greedy, 320 tokens, classes prose,code,json,math,chat)"
  iso=(); [[ "$ISOLATION" -gt 1 ]] && iso=(--isolation "$ISOLATION")
  python3 scripts/serve_load.py "$HOST" "$PORT" --concurrency "$CONC" --max-tokens 320 --classes prose,code,json,math,chat "${iso[@]}" 2>&1 | tee "$OUT/load.log" | grep -E "aggregate|concurrency|class|tok/s|isolation" | cut -c1-200
fi
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -3 | tee "$OUT/down.log"
grep -h 'scheduled verify depth summary' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-260
grep -h 'stats: rank 0' "$OUT/world/serve_r0.log" | grep -v 'decode 0.0 tok/s' | tail -1 | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-220
exit $rc
