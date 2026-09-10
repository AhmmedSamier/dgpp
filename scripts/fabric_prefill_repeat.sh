#!/bin/bash
# Kept in scripts/ since 2026-09-06 (the closure pass ran it from a scratch directory).
# The steady-state prefill ritual:
#   fabric_prefill_repeat.sh [--config CLUSTER.json] OUT_DIR LEN...
# For each LEN: the first LEN of the hard-prose ids (tiled x4 above 2100),
# glm_gen_check on the four nodes with --prefill-repeat 3 (the timed prefill
# is the third), two decode steps; prints the prefill line and the 4-way
# generated-ids verdict.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The checkpoint and the fabric come from a cluster config (dgpp-cluster's
# file: deploy/cluster.json by default, or --config F as the first argument)
# — its `model`, `nodes` (node 0 is the head, where this runs) and `ssh_user`
# — never from the environment or a hardcoded id.
CONFIG="$ROOT/deploy/cluster.json"
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[[ -f "$CONFIG" ]] || { echo "no cluster config at $CONFIG" >&2; exit 1; }
MODEL=$(jq -r '.model' "$CONFIG")
NODES=($(jq -r '.nodes[]' "$CONFIG"))
export DGPP_FABRIC_HEAD="${NODES[0]}"
export DGPP_FABRIC_PEERS="${NODES[*]:1}"
export DGPP_FABRIC_USER="$(jq -r '.ssh_user // env.USER' "$CONFIG")"
IDS=$ROOT/build-ci/fabric-runs/prefill_ids/hard_ids.csv
OUT=${1:?OUT_DIR}; shift
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
cd "$ROOT" || exit 1
for n in "$@"; do
  prompt=$(python3 - "$IDS" "$n" <<'PY'
import sys
ids = open(sys.argv[1]).read().strip().replace("\n", ",").split(",")
ids = [i for i in ids if i]
n = int(sys.argv[2])
print(",".join((ids * 20)[:n]))
PY
)
  "$ROOT/scripts/fabric_run.sh" --force --log-dir "$OUT/n$n" -- \
    --model "$MODEL" --prompt "$prompt" --steps 2 \
    --prefill-repeat 3 > "$OUT/n$n.out" 2>&1
  echo "== $n tokens"
  grep -h "rank 0 prefill:" "$OUT/n$n/r0.log" | tail -1
  grep -h "rank consistency\|rc=" "$OUT/n$n.out" | tail -2
  [ -f "$OUT/n$n/r0.log" ] || tail -5 "$OUT/n$n.out"
done
