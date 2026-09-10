#!/bin/bash
# The Q0 measurement of docs/qwen38_flash_next_plan.md (2026-09-09): what one
# more collective per residual site costs the decode step on the fabric.
# glm_gen_check --gr-probe N issues one extra [T x 10240] all-reduce after the
# attention fold of the first N layers of every decode row (a scratch buffer
# nobody reads; GlmBoundaryReducer::probe), so (probe - base) / N is the
# marginal cost of a recorded collective node. N is bounded by the graph's
# node budget (kBusMaxGraphGens 128 less the step's own nodes).
#   fabric_gr_probe.sh [--config CLUSTER.json] OUT_DIR [PROBE_LAYERS]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/deploy/cluster.json"
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[[ -f "$CONFIG" ]] || { echo "no cluster config at $CONFIG" >&2; exit 1; }
MODEL=$(jq -r '.model' "$CONFIG")
NODES=($(jq -r '.nodes[]' "$CONFIG"))
export DGPP_FABRIC_HEAD="${NODES[0]}"
export DGPP_FABRIC_PEERS="${NODES[*]:1}"
export DGPP_FABRIC_USER="$(jq -r '.ssh_user // env.USER' "$CONFIG")"
OUT=${1:?OUT_DIR}; shift
N=${1:-32}
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"; cd "$ROOT" || exit 1
PROMPT="Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs."
run() {
  local name=$1; shift
  "$ROOT/scripts/fabric_run.sh" --force --fetch-logs --log-dir "$OUT/$name" -- \
    --model "$MODEL" --chat "$PROMPT" --steps 300 --decode-graph "$@" \
    > "$OUT/$name.out" 2>&1
  echo "== $name (exit $?)"
  grep -h 'ms/step avg\|speculative summary' "$OUT/$name/r0.log" | tail -1 | sed 's/^.*INFO  //' | cut -c1-300
  grep -h 'rank consistency' "$OUT/$name.out" | tail -1
}
run t1_base
run t1_probe$N --gr-probe "$N"
run mtp_base --mtp
run mtp_probe$N --mtp --gr-probe "$N"
