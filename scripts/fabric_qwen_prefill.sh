#!/bin/bash
# The Qwen prefill procedure (docs/qwen38_flash_next_plan.md Q7, 2026-09-09):
# the world up from a cluster config, serve_prefill_probe.py at the prompt
# lengths (512 / 2048 / 8192 by default), rank 0's per-request prefill
# lines, the world down. Every DGPP_* knob in this environment reaches all
# ranks (dgpp-cluster forwards them), so an A/B is
#   DGPP_QWEN_MOE_PREFILL=host scripts/fabric_qwen_prefill.sh ...
#   fabric_qwen_prefill.sh CONFIG OUT_DIR [--knobs "FLAGS"] [LEN...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
KNOBS=""; LENS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --knobs) KNOBS="$2"; shift 2;;
    *) LENS+=("$1"); shift;;
  esac
done
[[ ${#LENS[@]} -eq 0 ]] && LENS=(512 2048 8192)
mkdir -p "$OUT"
HOST=$(dgpp_client_host) || exit 1
PORT=$(dgpp_http_port) || exit 1
cd "$ROOT" || exit 1
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world")
[[ -n "$KNOBS" ]] && up+=(--knobs "$KNOBS")
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up ($CONFIG) knobs='$KNOBS' DGPP_QWEN_MOE_PREFILL=${DGPP_QWEN_MOE_PREFILL:-}"
grep -h 'model constructed\|memory plan total' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-160
echo "== prefill probe (${LENS[*]})"
python3 scripts/serve_prefill_probe.py "$HOST" "$PORT" "${LENS[@]}" 2>&1 | tee "$OUT/probe.log"
echo "== rank 0 retire lines"
grep -h "retired" "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* INFO  sched: //' | sed 's/; decode.*//' | tail -n $(( ${#LENS[@]} * 3 + 1 ))
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -3 | tee "$OUT/down.log"
