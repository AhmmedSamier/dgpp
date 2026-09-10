#!/bin/bash
# Kept in scripts/ since 2026-09-06 (the closure pass ran it from a scratch directory).
# The M9 soak on the four nodes: boot with the production knobs, start the
# node probes on every node, run serve_soak.py for MINUTES, stop, collect.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh"
MIN=${1:-60}
OUT=${2:-$ROOT/build-ci/fabric-runs/soak_2026-09-05}
PEERS=($(dgpp_peers))
SSH=(-o BatchMode=yes -o ConnectTimeout=5)
# The peers' ssh login: DGPP_FABRIC_USER, else the config's, else the caller's.
SSH_USER="${DGPP_FABRIC_USER:-$(dgpp_ssh_user)}"
mkdir -p "$OUT"
export DGPP_SERVE_KNOBS="--max-concurrency 4 --kv-capacity 8192 --default-max-tokens 256 --queue-limit 8 --decode-graph --mtp"
export DGPP_SERVE_LOG="$OUT/serve"
cd "$ROOT" || exit 1
rm -f "$ROOT/serve_rank0.ops"
"$ROOT/scripts/serve_run.sh" up > "$OUT/up.log" 2>&1
grep -q READY "$OUT/up.log" || { echo "FAIL: boot"; tail -5 "$OUT/up.log"; exit 1; }
# Node probes at 1 Hz on every node for the run.
setsid nohup "$ROOT/scripts/node_probe.sh" soak_probe > "$OUT/probe_r0.log" 2>&1 < /dev/null &
echo $! > "$OUT/probe_r0.pid"
for i in 1 2 3; do
  h=${PEERS[$((i - 1))]}
  timeout 20 scp -q "${SSH[@]}" "$ROOT/scripts/node_probe.sh" "$SSH_USER@$h:/tmp/bus4/node_probe.sh"
  timeout 20 ssh "${SSH[@]}" "$SSH_USER@$h" "cd /tmp/bus4 && chmod +x node_probe.sh && setsid nohup ./node_probe.sh soak_probe > probe_r$i.log 2>&1 < /dev/null &" || echo "WARN: probe on $h"
done
echo "=== soak: $MIN min from $(date +%T)"
python3 "$ROOT/scripts/serve_soak.py" "$(dgpp_head)" 18080 "$MIN" "$OUT/client" 2>&1 | tee "$OUT/soak_client.log"
echo "=== soak client done at $(date +%T); stopping the world"
kill "$(cat "$OUT/probe_r0.pid")" 2>/dev/null
for i in 1 2 3; do
  h=${PEERS[$((i - 1))]}
  timeout 20 ssh "${SSH[@]}" "$SSH_USER@$h" "pkill -f 'node_probe.sh soak_probe'" >/dev/null 2>&1
  timeout 20 scp -q "${SSH[@]}" "$SSH_USER@$h:/tmp/bus4/probe_r$i.log" "$OUT/probe_r$i.log" 2>/dev/null
done
"$ROOT/scripts/serve_run.sh" down > "$OUT/down.log" 2>&1
echo "=== op-stream md5s:"; grep -h "serve_rank" "$OUT/down.log"
echo "=== STALLED lines per rank log:"
for f in "$OUT/serve/serve_r0.log" "$OUT/serve/serve_r1.log" "$OUT/serve/serve_r2.log" "$OUT/serve/serve_r3.log"; do
  [ -f "$f" ] || { for i in 1 2 3; do h=${PEERS[$((i - 1))]}; timeout 20 scp -q "${SSH[@]}" "$SSH_USER@$h:/tmp/bus4/serve_r$i.log" "$OUT/serve/serve_r$i.log" 2>/dev/null; done; }
done
for f in "$OUT"/serve/serve_r[0-3].log; do echo "  $(basename $f): $(grep -c STALLED "$f" 2>/dev/null) stalled, $(grep -c 'ENGINE FAILURE\|divergence' "$f" 2>/dev/null) failures"; done
echo "=== node probes: allocstall / pgmajfault / pswpin+out sums per node, max throttle mask"
for f in "$OUT"/probe_r[0-3].log; do
  python3 - "$f" <<'PY'
import re, sys
alloc = major = swap = scan = compact = 0; masks = set(); n = 0
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\d\d:\d\d:\d\d (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) \|", line)
    if not m:
        continue
    a, mj, si, so, sc, cs = (int(x) for x in m.groups())
    alloc += a; major += mj; swap += si + so; scan += sc; compact += cs; n += 1
    t = re.search(r"throttle=(\S+)", line)
    if t: masks.add(t.group(1))
print(f"  {sys.argv[1].split('/')[-1]}: {n} samples, allocstall {alloc}, pgmajfault {major}, swap {swap}, pgscan_direct {scan}, compact_stall {compact}, throttle masks {sorted(masks)}")
PY
done
echo "=== done: $OUT"
