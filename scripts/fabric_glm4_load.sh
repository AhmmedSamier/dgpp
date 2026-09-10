#!/bin/bash
# The GLM-4.7 concurrency reading on the fabric (2026-09-10, the batched
# depth-2 study): the world up from a cluster config with a 5-second stats
# line, serve_load.py's fixed-concurrency phases (1, 2 and 4 live requests
# by default: distinct long-answer prompts, greedy, thinking off), then the
# world down and rank 0's stats lines inside each phase (ms/step,
# tok/step/req, the acceptance) beside the client's reading. One world at
# a time on the fabric; ~3 minutes per config.
#   fabric_glm4_load.sh CONFIG OUT_DIR [--concurrency 1,2,4] [--tokens N] [--knobs "FLAGS"]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
CONC="1,2,4"; TOKENS=320; KNOBS="--stats-interval-s 5"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) CONC="$2"; shift 2;;
    --tokens) TOKENS="$2"; shift 2;;
    --knobs) KNOBS="$KNOBS $2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
HOST=$(jq -r '.nodes[0]' "$CONFIG"); PORT=$(jq -r '.ports.http' "$CONFIG")
cd "$ROOT" || exit 1
python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world" --knobs "$KNOBS" > "$OUT/up.log" 2>&1 \
  || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up ($CONFIG)"
grep -h 'decode rows\|graph variants\|row batch\|memory plan total' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
echo "== load"
python3 scripts/serve_load.py "$HOST" "$PORT" --concurrency "$CONC" --max-tokens "$TOKENS" 2>&1 | tee "$OUT/load.log"
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -2
echo "== rank 0 stats lines inside each phase"
python3 - "$OUT" <<'EOF'
import re, sys, os
out = sys.argv[1]
load = open(os.path.join(out, "load.log")).read().splitlines()
phases = []
for l in load:
    m = re.match(r"== concurrency (\d+): (\S+) \.\. (\S+)", l)
    if m: phases.append((int(m.group(1)), m.group(2), m.group(3)))
stats = []
for l in open(os.path.join(out, "world", "serve_r0.log")):
    m = re.match(r"\S+ (\S+) INFO  stats: rank 0 \| (\S+) s \| (.*)", l)
    if m: stats.append((m.group(1), float(m.group(2)), m.group(3)))
def secs(t):
    h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)
for c, a, b in phases:
    ta, tb = secs(a), secs(b)
    print(f"-- concurrency {c} ({a} .. {b})")
    for stamp, period, rest in stats:
        te = secs(stamp); ts = te - period
        if ts >= ta - 0.5 and te <= tb + 0.5:
            print("   " + re.sub(r" \| prefill.*", "", rest)[:150])
EOF
