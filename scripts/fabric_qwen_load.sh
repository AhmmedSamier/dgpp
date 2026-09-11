#!/bin/bash
# Q2's real-checkpoint load gate (docs/qwen38_flash_next_plan.md, 2026-09-09):
# qwen_load_check on the first WORLD nodes of the cluster config, one rank
# per node, in parallel and WITHOUT a bus — the loader needs none. Stages
# the binary, runs every rank, fetches the logs, prints each rank's
# formulas, totals and digest so a mismatch is visible at a glance.
#   fabric_qwen_load.sh [--config CLUSTER.json] OUT_DIR WORLD [extra qwen_load_check args...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=$(dgpp_config) || exit 1
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
USER_=$(dgpp_ssh_user) || exit 1
STAGE="$DGPP_STAGE_DIR"
OUT=${1:?OUT_DIR}; WORLD=${2:?WORLD}; shift 2
NODE_LIST=$(dgpp_nodes --world "$WORLD") || exit 1
read -r -a NODES <<< "$NODE_LIST"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
BIN="${DGPP_BUILD_DIR:-$ROOT/build-ci}/qwen_load_check"
MODEL=${DGPP_QWEN_MODEL:-$(dgpp_model)}
pids=()
for ((r=0; r<WORLD; r++)); do
  ip=${NODES[$r]}
  if [[ $r -eq 0 ]]; then
    ( dgpp_run_rank 0 "$BIN" --model "$MODEL" --world "$WORLD" --rank 0 "$@" > "$OUT/w${WORLD}_r0.log" 2>&1 ) &
    pids+=($!)
  else
    ssh -o BatchMode=yes "$USER_@$ip" "mkdir -p $STAGE" >/dev/null 2>&1
    RANK_ENV=$(dgpp_rank_prefix "$r") || exit 1
    scp -q "$BIN" "$USER_@$ip:$STAGE/qwen_load_check"
    ( ssh -o BatchMode=yes "$USER_@$ip" "$RANK_ENV $STAGE/qwen_load_check --model $MODEL --world $WORLD --rank $r $*" > "$OUT/w${WORLD}_r$r.log" 2>&1 ) &
    pids+=($!)
  fi
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "world $WORLD: exit $rc"
for ((r=0; r<WORLD; r++)); do
  echo "== rank $r"
  grep -h 'formulas\|digest\|n-gram table rows\|layers .* resident\|ERROR' "$OUT/w${WORLD}_r$r.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-260
done
exit $rc
