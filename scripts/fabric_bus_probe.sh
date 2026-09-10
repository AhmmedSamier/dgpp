#!/bin/bash
# The bus-level half of the Qwen plan's Q0 measurement (2026-09-09): the cost
# of one recorded all-reduce node at a given world size, from bus_check's
# allreduce mode — a graph of GENS collectives replayed ITERS times, on the
# first WORLD nodes of the cluster config. Prints every rank's GRAPH-PROBE
# line (per-collective min/p50 µs, and the eager p50 for comparison).
#   fabric_bus_probe.sh [--config CLUSTER.json] OUT_DIR WORLD [GENS] [ITERS]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/deploy/cluster.json"
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
NODES=($(jq -r '.nodes[]' "$CONFIG"))
USER_=$(jq -r '.ssh_user // env.USER' "$CONFIG")
STAGE=$(jq -r '.paths.stage_dir // "/tmp/bus4"' "$CONFIG")
OUT=${1:?OUT_DIR}; WORLD=${2:?WORLD}; GENS=${3:-32}; ITERS=${4:-200}
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
BIN="$ROOT/build-ci/bus_check"
HEAD=${NODES[0]}
PORT=29650
for ((r=1; r<WORLD; r++)); do
  ip=${NODES[$r]}
  ssh -o BatchMode=yes "$USER_@$ip" "mkdir -p $STAGE && pkill -x bus_check; true" >/dev/null 2>&1
  scp -q "$BIN" "$USER_@$ip:$STAGE/bus_check"
done
# rank 0 first (it listens), then the peers.
( "$BIN" allreduce --world "$WORLD" --rank 0 --port $PORT --iters "$ITERS" --graph-gens "$GENS" \
    > "$OUT/w${WORLD}_r0.log" 2>&1 ) &
HEADPID=$!
sleep 2
for ((r=1; r<WORLD; r++)); do
  ip=${NODES[$r]}
  ssh -o BatchMode=yes "$USER_@$ip" "nohup $STAGE/bus_check allreduce --peer $HEAD --world $WORLD --rank $r --port $PORT --iters $ITERS --graph-gens $GENS > $STAGE/bus_probe_r$r.log 2>&1 < /dev/null &" 
done
wait $HEADPID
echo "rank 0 exit $?"
for ((r=1; r<WORLD; r++)); do
  ip=${NODES[$r]}
  scp -q "$USER_@$ip:$STAGE/bus_probe_r$r.log" "$OUT/w${WORLD}_r$r.log"
done
grep -h 'GRAPH-PROBE\|ALLREDUCE.*p50\|stopped cleanly' "$OUT"/w${WORLD}_r*.log | sed 's/^.*INFO  //' | cut -c1-220
