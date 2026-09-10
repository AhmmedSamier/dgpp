#!/bin/bash
# Drain-on-stop on the fabric (M6 6c): boots the four-node world, starts a
# long streamed request, and stops the world while it is mid-generation.
# The expected picture: the client's stream ends with the server_shutdown
# error event and [DONE] after the tokens it got; rank 0 logs the drain
# pass; every peer logs "stop record — exiting" and "exited cleanly"; no
# rank logs an ERROR, a stall or transport-retry-exceeded; the op streams
# agree on all four ranks (the interrupted request retired at the same
# quantum everywhere).
#
# Usage: scripts/serve_stop_check.sh [OUT_DIR]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/cluster_env.sh"
OUT=${1:-$ROOT/build-ci/fabric-runs/stop_gate_$(date +%Y-%m-%d_%H%M%S)}
export DGPP_SERVE_LOG=$OUT/serve
export DGPP_SERVE_KNOBS="${DGPP_SERVE_KNOBS:---max-concurrency 2 --kv-capacity 4096 --default-max-tokens 512 --queue-limit 8 --decode-graph --mtp --seed 20260904}"
HOST=${DGPP_SERVE_HOST:-$(dgpp_head)}
mkdir -p "$DGPP_SERVE_LOG"
cd "$ROOT" || exit 1
scripts/serve_run.sh up || exit 1
sleep 2
URL=http://$HOST:18080/v1/chat/completions
M=unsloth/GLM-5.3-Flash-FP8
# The long stream, in the background; a second request sits queued behind
# the first's slot pair? No — two slots are free; it runs too. Both are
# mid-generation when the stop lands.
curl -s --max-time 600 -N "$URL" -H 'Content-Type: application/json' -d '{"model":"'$M'","messages":[{"role":"user","content":"Write a long essay on the history of lighthouses."}],"stream":true,"max_tokens":1024}' > "$OUT/stream.sse" &
CURL1=$!
curl -s --max-time 600 "$URL" -H 'Content-Type: application/json' -d '{"model":"'$M'","messages":[{"role":"user","content":"Explain tides to a child, at length."}],"max_tokens":1024}' > "$OUT/oneshot.json" &
CURL2=$!
sleep 6
echo "=== stopping the world with two requests mid-generation"
scripts/serve_run.sh down
wait $CURL1 $CURL2
echo "--- the stream's tail"
tail -c 700 "$OUT/stream.sse"; echo
python3 - "$OUT/stream.sse" <<'PY'
import json, sys
n = 0; err = None; done = False; finish = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith('data: '): continue
    p = line[6:]
    if p == '[DONE]': done = True; continue
    d = json.loads(p)
    if 'error' in d: err = d['error']; continue
    n += 1
    for ch in d.get('choices', []):
        if ch.get('finish_reason'): finish = ch['finish_reason']
print(f"stream: {n} chunks, error={err}, done={done}, finish_reason={finish}")
PY
echo "--- the one-shot"
head -c 400 "$OUT/oneshot.json"; echo
echo "--- rank 0's drain line"
grep -h "serve: stop\|drain" "$DGPP_SERVE_LOG/serve_r0.log" | tail -3
echo "--- every rank's exit"
for r in 0 1 2 3; do grep -h "exited cleanly\|stop record\|stopped cleanly" "$DGPP_SERVE_LOG/serve_r$r.log" 2>/dev/null | tail -2 | sed "s/^/rank $r: /"; done
echo "--- errors on any rank (expect none)"
grep -h "ERROR\|STALL\|retry-exceeded\|retry exceeded" "$DGPP_SERVE_LOG"/serve_r*.log | grep -v generation_config | head -5
md5sum "$DGPP_SERVE_LOG"/serve_rank*.ops
echo "=== stop check done: $OUT"
