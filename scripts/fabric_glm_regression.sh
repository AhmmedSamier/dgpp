#!/bin/bash
# The GLM-5.3-Flash no-regression procedure (2026-09-09, the Qwen sessions):
# the same three readings as build-ci/fabric-runs/glm_baseline_2026-09-09
# — MTP on the chat class (fabric_mtp_classes.sh), T=1 on the chat prompt
# with fabric_step_times.py, steady-state prefill at 512 / 2048 / 8192
# (fabric_prefill_repeat.sh) — so a diff against the baseline is one
# command. Runs ~4 minutes; the fabric must be idle.
#   fabric_glm_regression.sh [--config CLUSTER.json] OUT_DIR
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=$(dgpp_config) || exit 1
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
OUT=${1:?OUT_DIR}
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"; cd "$ROOT" || exit 1
MODEL=$(jq -r '.model' "$CONFIG")
NODE_LIST=$(dgpp_nodes --config "$CONFIG") || exit 1
read -r -a NODES <<< "$NODE_LIST"

CHAT="Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs."
date
echo "== MTP chat"
scripts/fabric_mtp_classes.sh --config "$CONFIG" "$OUT/mtp" chat
echo "== T=1"
scripts/fabric_run.sh --force --log-dir "$OUT/t1" -- \
  --model "$MODEL" --chat "$CHAT" --steps 300 --decode-graph > "$OUT/t1.out" 2>&1
grep -h "rank consistency" "$OUT/t1.out" | tail -1
python3 scripts/fabric_step_times.py "$OUT/t1"
python3 scripts/fabric_step_times.py "$OUT/mtp/chat"
grep -h "speculative summary" "$OUT/mtp/chat/r0.log" | tail -1 | sed 's/^.*INFO  //' | cut -c1-260
echo "== prefill"
scripts/fabric_prefill_repeat.sh --config "$CONFIG" "$OUT/prefill" 512 2048 8192
date
