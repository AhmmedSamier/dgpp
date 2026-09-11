#!/bin/bash
# The bus-level half of the Qwen plan's Q0 measurement (2026-09-09): the cost
# of one recorded all-reduce node at a given world size, from bus_check's
# allreduce mode — a graph of GENS collectives replayed ITERS times, on the
# first WORLD nodes of the cluster config. Prints every rank's GRAPH-PROBE
# line (per-collective min/p50 µs, and the eager p50 for comparison).
#   fabric_bus_probe.sh [--config CLUSTER.json] OUT_DIR WORLD [GENS] [ITERS]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=$(dgpp_config) || exit 1
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
USER_=$(dgpp_ssh_user) || exit 1
STAGE="$DGPP_STAGE_DIR"
OUT=${1:?OUT_DIR}; WORLD=${2:?WORLD}; GENS=${3:-32}; ITERS=${4:-200}
NODE_LIST=$(dgpp_nodes --world "$WORLD") || exit 1
read -r -a NODES <<< "$NODE_LIST"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
BIN="${DGPP_BUILD_DIR:-$ROOT/build-ci}/bus_check"
HEAD=${NODES[0]}
PORT="$DGPP_FABRIC_PORT"
for ((r=1; r<WORLD; r++)); do
  ip=${NODES[$r]}
  ssh -o BatchMode=yes "$USER_@$ip" "mkdir -p $STAGE" >/dev/null 2>&1
  scp -q "$BIN" "$USER_@$ip:$STAGE/bus_check"
done
# rank 0 first (it listens), then the peers.
( dgpp_run_rank 0 "$BIN" allreduce --world "$WORLD" --rank 0 --port $PORT --iters "$ITERS" --graph-gens "$GENS" \
    > "$OUT/w${WORLD}_r0.log" 2>&1 ) &
HEADPID=$!
sleep 2
for ((r=1; r<WORLD; r++)); do
  ip=${NODES[$r]}
  RANK_ENV=$(dgpp_rank_prefix "$r") || exit 1
  ssh -o BatchMode=yes "$USER_@$ip" "$RANK_ENV nohup $STAGE/bus_check allreduce --peer $HEAD --world $WORLD --rank $r --port $PORT --iters $ITERS --graph-gens $GENS > $STAGE/bus_probe_r$r.log 2>&1 < /dev/null &"
done
wait $HEADPID
echo "rank 0 exit $?"
for ((r=1; r<WORLD; r++)); do
  ip=${NODES[$r]}
  scp -q "$USER_@$ip:$STAGE/bus_probe_r$r.log" "$OUT/w${WORLD}_r$r.log"
done
grep -h 'GRAPH-PROBE\|ALLREDUCE.*p50\|stopped cleanly' "$OUT"/w${WORLD}_r*.log | sed 's/^.*INFO  //' | cut -c1-220
