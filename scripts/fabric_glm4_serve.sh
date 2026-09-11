#!/bin/bash
# The GLM-4.7 serving gates on the fabric:
# boot the world from a cluster config (deploy/cluster_glm47.json: MTP +
# the decode graph; deploy/cluster_glm47_t1.json: plain T=1), then the
# API check, the greedy transcripts (compared against a reference file
# when given), the client-side pace, and the world down with its op-stream
# md5s. One world at a time on the fabric.
#   fabric_glm4_serve.sh CONFIG OUT_DIR [--compare REF_TRANSCRIPTS.json] [--eval] [--knobs "FLAGS"]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
COMPARE=""; EVAL=0; KNOBS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compare) COMPARE="$2"; shift 2;;
    --eval) EVAL=1; shift;;
    --knobs) KNOBS="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
HOST=$(jq -r '.nodes[0]' "$CONFIG"); PORT=$(jq -r '.ports.http' "$CONFIG")
cd "$ROOT" || exit 1
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world")
[[ -n "$KNOBS" ]] && up+=(--knobs "$KNOBS")
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up ($CONFIG)"; grep -h 'model constructed\|graph variants\|memory plan total\|listening' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
rc=0
echo "== greedy transcripts"
cmp=(); [[ -n "$COMPARE" ]] && cmp=(--compare "$COMPARE")
python3 scripts/serve_greedy_transcript.py "$HOST" "$PORT" --out "$OUT/transcripts.json" "${cmp[@]}" 2>&1 | tee "$OUT/transcripts.log" || rc=1
echo "== pace (serve_bench, 200 tokens)"
python3 scripts/serve_bench.py "$HOST" "$PORT" 200 glm4 2>&1 | tail -4 | tee "$OUT/bench.log"
echo "== api check"
python3 scripts/serve_api_check.py "$HOST" "$PORT" > "$OUT/api_check.log" 2>&1 && echo "api check: OK" || { echo "api check: FAILED (see $OUT/api_check.log)"; tail -5 "$OUT/api_check.log"; rc=1; }
if [[ $EVAL -eq 1 ]]; then
  # GLM-4.7 ignores reasoning_effort and thinks to the token cap (2,048
  # tokens of reasoning per gsm8k problem at 150 s each): the eval runs
  # with the template's thinking switched off (chat_template_kwargs.
  # enable_thinking = false), the model's non-thinking mode.
  echo "== eval (gsm8k 60, humaneval 40, extract 30; thinking off)"
  python3 scripts/serve_eval.py "$HOST" "$PORT" --out "$OUT/eval" --tasks gsm8k --limit 60 --concurrency 4 --no-think 2>&1 | tail -3
  python3 scripts/serve_eval.py "$HOST" "$PORT" --out "$OUT/eval" --tasks humaneval --limit 40 --concurrency 4 --no-think 2>&1 | tail -3
  python3 scripts/serve_eval.py "$HOST" "$PORT" --out "$OUT/eval" --tasks extract --limit 30 --concurrency 4 --no-think 2>&1 | tail -3
fi
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -6 | tee "$OUT/down.log"
grep -h 'throughput\|steps/s\|ms/step\|per step' "$OUT/world/serve_r0.log" | tail -3 | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
exit $rc
