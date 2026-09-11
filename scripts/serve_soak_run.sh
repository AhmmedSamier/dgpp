#!/bin/bash
# Four-node serving soak: boot with the production knobs, start the
# node probes on every node, run serve_soak.py for MINUTES, stop, collect.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
dgpp_require_world 4 || exit 1
MIN=${1:-60}
OUT=${2:-${DGPP_BUILD_DIR:-$ROOT/build-ci}/fabric-runs/soak_$(date -u +%Y%m%d-%H%M%S)}
PEER_LIST=$(dgpp_peers) || exit 1
read -r -a PEERS <<< "$PEER_LIST"
SSH=(-o BatchMode=yes -o ConnectTimeout=5)
# The shared SSH login comes from .env (or an exported override).
SSH_USER=$(dgpp_ssh_user) || exit 1
mkdir -p "$OUT"
export DGPP_SERVE_KNOBS="--max-concurrency 4 --kv-capacity 8192 --default-max-tokens 256 --queue-limit 8 --decode-graph --mtp"
export DGPP_SERVE_LOG="$OUT/serve"
STAGE=$(dgpp_stage_dir) || exit 1
cd "$ROOT" || exit 1
"$ROOT/scripts/serve_run.sh" up > "$OUT/up.log" 2>&1
grep -q READY "$OUT/up.log" || { echo "FAIL: boot"; tail -5 "$OUT/up.log"; exit 1; }
# Node probes at 1 Hz on every node for the run.
python3 "$ROOT/scripts/cluster_process.py" launch --state "$OUT/probe.process.json" --log "$OUT/probe_r0.log" --cwd "$ROOT" -- bash "$ROOT/scripts/node_probe.sh" soak_probe
for i in 1 2 3; do
  h=${PEERS[$((i - 1))]}
  timeout 20 scp -q "${SSH[@]}" "$ROOT/scripts/node_probe.sh" "$SSH_USER@$h:$STAGE/node_probe.sh"
  timeout 20 ssh "${SSH[@]}" "$SSH_USER@$h" "cd $STAGE && python3 cluster_process.py launch --state probe.process.json --log probe_r$i.log --cwd . -- bash node_probe.sh soak_probe" || echo "WARN: probe on $h"
done
echo "=== soak: $MIN min from $(date +%T)"
python3 "$ROOT/scripts/serve_soak.py" "$(dgpp_client_host)" "$(dgpp_http_port)" "$MIN" "$OUT/client" 2>&1 | tee "$OUT/soak_client.log"
echo "=== soak client done at $(date +%T); stopping the world"
python3 "$ROOT/scripts/cluster_process.py" signal --state "$OUT/probe.process.json" --signal TERM
for i in 1 2 3; do
  h=${PEERS[$((i - 1))]}
  timeout 20 ssh "${SSH[@]}" "$SSH_USER@$h" "cd $STAGE && python3 cluster_process.py signal --state probe.process.json --signal TERM" >/dev/null 2>&1
  timeout 20 scp -q "${SSH[@]}" "$SSH_USER@$h:$STAGE/probe_r$i.log" "$OUT/probe_r$i.log" 2>/dev/null
done
"$ROOT/scripts/serve_run.sh" down > "$OUT/down.log" 2>&1
echo "=== op-stream md5s:"; grep -h "serve_rank" "$OUT/down.log"
echo "=== STALLED lines per rank log:"
for f in "$OUT/serve/serve_r0.log" "$OUT/serve/serve_r1.log" "$OUT/serve/serve_r2.log" "$OUT/serve/serve_r3.log"; do
  [ -f "$f" ] || { for i in 1 2 3; do h=${PEERS[$((i - 1))]}; timeout 20 scp -q "${SSH[@]}" "$SSH_USER@$h:$STAGE/serve_r$i.log" "$OUT/serve/serve_r$i.log" 2>/dev/null; done; }
done
for f in "$OUT"/serve/serve_r[0-3].log; do echo "  $(basename $f): $(grep -c STALLED "$f" 2>/dev/null) stalled, $(grep -c 'ENGINE FAILURE\|divergence' "$f" 2>/dev/null) failures"; done
echo "=== node probes: allocstall / pgmajfault / pswpin+out sums per node, max throttle mask"
for f in "$OUT"/probe_r[0-3].log; do
  python3 "$ROOT/scripts/node_probe_report.py" "$f"
done
echo "=== done: $OUT"
