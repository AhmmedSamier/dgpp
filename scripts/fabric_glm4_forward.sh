#!/bin/bash
# The GLM-4.7 real-checkpoint TP forward gate (docs/glm47_plan.md,
# 2026-09-09): glm4_forward_check on the first WORLD nodes of the cluster
# config over the fabric bus, one rank per node — the same token ids on
# every rank; every rank's per-layer hyper-state digests must agree
# bitwise, and the merged argmax (the highest logit across the ranks'
# vocab slices) must equal the world-1 argmax.
#   fabric_glm4_forward.sh [--config CLUSTER.json] OUT_DIR WORLD IDS [extra args...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=$(dgpp_config) || exit 1
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
USER_=$(dgpp_ssh_user) || exit 1
STAGE="$DGPP_STAGE_DIR"
OUT=${1:?OUT_DIR}; WORLD=${2:?WORLD}; IDS=${3:?IDS}; shift 3
NODE_LIST=$(dgpp_nodes --world "$WORLD") || exit 1
read -r -a NODES <<< "$NODE_LIST"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
BIN="${DGPP_BUILD_DIR:-$ROOT/build-ci}/glm4_forward_check"
MODEL=${DGPP_GLM4_MODEL:-$(dgpp_model)}
PORT=${DGPP_GLM4_PORT:-$DGPP_FABRIC_PORT}
HEAD=${NODES[0]}
pids=()
for ((r=0; r<WORLD; r++)); do
  ip=${NODES[$r]}
  if [[ $r -eq 0 ]]; then
    ( dgpp_run_rank 0 "$BIN" --model "$MODEL" --ids "$IDS" --world "$WORLD" --rank 0 --port "$PORT" "$@" > "$OUT/w${WORLD}_r0.log" 2>&1 ) &
    pids+=($!)
  else
    ssh -o BatchMode=yes "$USER_@$ip" "mkdir -p $STAGE" >/dev/null 2>&1
    RANK_ENV=$(dgpp_rank_prefix "$r") || exit 1
    scp -q "$BIN" "$USER_@$ip:$STAGE/glm4_forward_check"
    ( ssh -o BatchMode=yes "$USER_@$ip" "$RANK_ENV $STAGE/glm4_forward_check --model $MODEL --ids $IDS --world $WORLD --rank $r --peer $HEAD --port $PORT $*" > "$OUT/w${WORLD}_r$r.log" 2>&1 ) &
    pids+=($!)
  fi
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "world $WORLD: exit $rc"
for ((r=0; r<WORLD; r++)); do
  echo "== rank $r"
  grep -h 'digest\|argmax\|boot\|ERROR\|glm4_forward_check:' "$OUT/w${WORLD}_r$r.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
done
# Cross-rank digests must agree; the merged argmax is the per-position max over the ranks' argmax logits.
python3 "$ROOT/scripts/forward_check_report.py" "$OUT" "$WORLD"
