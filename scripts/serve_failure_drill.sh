#!/bin/bash
# The failure drill (M8's exit criterion "injected rank failure leaves
# committed state unchanged"; M9's "kill -9 a random rank under load";
# built 2026-09-05): kill -9 one rank of the four-node service while
# clients stream, and check the v1 failure semantics end to end:
#   1. every client that was streaming got the tokens the service had
#      committed, then — rank 0 alive — the engine_failure error event and
#      [DONE] (no finish chunk); with rank 0 the victim the connection
#      simply closes (nobody is left to write an event);
#   2. rank 0 left nonzero within the bound (its log: ENGINE FAILURE, then
#      "exiting with status 2"), and every peer was gone within the bound
#      (its log: the in-tick watch's "status 3", or the read loop's EOF);
#   3. `serve_run.sh up` serves again, and the same prompts at temperature
#      0 reproduce every client's committed tokens as a PREFIX of the fresh
#      answer — nothing uncommitted ever reached a client, nothing
#      committed was lost.
# Usage: serve_failure_drill.sh <victim rank 0..3> [clients (1..3, default 3)]
# Set DGPP_SERVE_KNOBS as for serve_run.sh (the same knobs on every rank).
# Everything lands under $DGPP_DRILL_DIR (default build-ci/fabric-runs/
# failure_drill_<stamp>/): the SSE captures, the logs, the ops files, the
# fresh answers, and summary.txt.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh"
SR=$ROOT/scripts/serve_run.sh
LOG="${DGPP_SERVE_LOG:-$HOME/dgpp/log}"
PEERS=($(dgpp_peers))
RANK0=$(dgpp_head)
HTTP="http://$RANK0:18080"
PEER_DIR=/tmp/bus4
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)
# The peers' ssh login: DGPP_FABRIC_USER, else the config's, else the caller's.
SSH_USER="${DGPP_FABRIC_USER:-$(dgpp_ssh_user)}"
VICTIM=${1:?usage: $0 <victim rank 0..3> [clients]}
CLIENTS=${2:-3}
STAMP=$(date +%Y-%m-%d_%H%M%S)
DRILL="${DGPP_DRILL_DIR:-$ROOT/build-ci/fabric-runs/failure_drill_${STAMP}}"
mkdir -p "$DRILL"
SUMMARY="$DRILL/summary.txt"
: > "$SUMMARY"
say() { echo "$*" | tee -a "$SUMMARY"; }

PROMPTS=(
  "Write a long, detailed history of the Roman Republic from its founding to the rise of Augustus, one era per paragraph."
  "Explain, step by step and at length, how a CUDA graph is captured and replayed, and what can go wrong with device-side pointers."
  "Tell a long story about a lighthouse keeper who discovers that the sea has started to speak, in at least ten paragraphs."
)
MAX_TOKENS=1500  # long enough that the live streams outlast the kill

peer_ssh() { timeout 20 ssh "${SSH_OPTS[@]}" "$SSH_USER@$1" "${2:-true}"; }

body_for() {  # body_for INDEX STREAM(true|false)
  local i=$1 stream=$2
  python3 - "$i" "$stream" "$MAX_TOKENS" "${PROMPTS[$i]}" <<'PY'
import json, sys
i, stream, n, prompt = int(sys.argv[1]), sys.argv[2] == "true", int(sys.argv[3]), sys.argv[4]
print(json.dumps({"model": "unsloth/GLM-5.3-Flash-FP8", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": n, "temperature": 0, "stream": stream}))
PY
}

# Every token the client received, concatenated: the reasoning deltas and
# the content deltas, in order (the committed text the prefix check is over).
sse_content() {
  python3 - "$1" <<'PY'
import json, sys
out = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    try:
        obj = json.loads(line[6:])
    except json.JSONDecodeError:
        continue
    for ch in obj.get("choices", []):
        d = ch.get("delta", {})
        if d.get("reasoning_content"):
            out.append(d["reasoning_content"])
        if d.get("content"):
            out.append(d["content"])
sys.stdout.write("".join(out))
PY
}

say "=== failure drill: victim rank $VICTIM, $CLIENTS streaming client(s), knobs: ${DGPP_SERVE_KNOBS:-<serve_run.sh default>}"
say "=== artifacts: $DRILL"

# 1. Boot.
"$SR" up 2>&1 | tee "$DRILL/up1.log" | tail -3
grep -q "READY" "$DRILL/up1.log" || { say "FAIL: the world did not come up"; exit 1; }

# 2. Load: the streaming clients.
CURL_PIDS=()
for ((i = 0; i < CLIENTS; ++i)); do
  body_for "$i" true > "$DRILL/client_$i.body"
  curl -sN --max-time 600 "$HTTP/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$DRILL/client_$i.body" > "$DRILL/client_$i.sse" 2> "$DRILL/client_$i.err" &
  CURL_PIDS+=($!)
done
# Streaming: at least two clients (the service's concurrency here) with six
# token deltas (reasoning or content) each; a third waits in the queue.
deltas() { grep -o '"\(reasoning_content\|content\)":"[^"]' "$1" 2>/dev/null | wc -l; }
need=$(( CLIENTS < 2 ? CLIENTS : 2 ))
for _ in $(seq 1 600); do
  live=0
  for ((i = 0; i < CLIENTS; ++i)); do [ "$(deltas "$DRILL/client_$i.sse")" -ge 6 ] && live=$((live + 1)); done
  [ "$live" -ge "$need" ] && break
  sleep 0.1
done
for ((i = 0; i < CLIENTS; ++i)); do
  say "client $i: $(deltas "$DRILL/client_$i.sse") token deltas before the kill"
done
sleep "$(python3 -c 'import random; print(round(random.uniform(0.2, 1.5), 2))')"  # a random moment under load

# 3. The kill.
T_KILL=$(date +%s.%N)
if [ "$VICTIM" -eq 0 ]; then
  pid=$(cat "$LOG/r0.pid")
  say "=== kill -9 rank 0 (pid $pid) at $(date +%T.%N)"
  kill -9 "$pid"
else
  h=${PEERS[$((VICTIM - 1))]}
  say "=== kill -9 rank $VICTIM on $h at $(date +%T.%N)"
  peer_ssh "$h" "pkill -9 -x dgpp-serve"
fi

# 4. The ranks leave. Rank 0: gone within 30 s of the kill.
r0_gone=""
for _ in $(seq 1 300); do
  if ! kill -0 "$(cat "$LOG/r0.pid")" 2>/dev/null; then
    r0_gone=$(python3 -c "import time; print(round(time.time() - $T_KILL, 2))"); break
  fi
  sleep 0.1
done
if [ -n "$r0_gone" ]; then say "rank 0: gone ${r0_gone}s after the kill"; else say "FAIL: rank 0 still alive 30 s after the kill"; fi
for ((r = 1; r <= 3; ++r)); do
  h=${PEERS[$((r - 1))]}
  gone=""
  for _ in $(seq 1 300); do
    if ! peer_ssh "$h" "pgrep -x dgpp-serve" >/dev/null 2>&1; then
      gone=$(python3 -c "import time; print(round(time.time() - $T_KILL, 2))"); break
    fi
    sleep 0.1
  done
  if [ -n "$gone" ]; then say "rank $r: gone ${gone}s after the kill"; else say "FAIL: rank $r still alive 30 s after the kill"; fi
done

# 5. The clients' streams end.
for p in "${CURL_PIDS[@]}"; do
  for _ in $(seq 1 600); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
  kill -0 "$p" 2>/dev/null && { say "WARN: a client stream did not end within 60 s; killing curl"; kill "$p"; }
done
interrupted=0
for ((i = 0; i < CLIENTS; ++i)); do
  f="$DRILL/client_$i.sse"
  sse_content "$f" > "$DRILL/client_$i.committed.txt"
  chunks=$(deltas "$f")
  if grep -q '"finish_reason":"' "$f"; then
    say "client $i: completed before the kill ($chunks token deltas, finish $(grep -o '"finish_reason":"[a-z_]*"' "$f" | tail -1)) — not interrupted"
    continue
  fi
  if [ "$VICTIM" -ne 0 ]; then
    if grep -q '"code":"engine_failure"' "$f" && grep -q '^data: \[DONE\]' "$f"; then
      interrupted=$((interrupted + 1))
      if [ "$chunks" -eq 0 ]; then
        say "client $i: queued at the kill — the engine_failure event and [DONE], no tokens — OK"
      else
        say "client $i: $chunks token deltas, then the engine_failure event and [DONE], no finish chunk — OK"
      fi
    else
      say "FAIL: client $i did not end with engine_failure + [DONE] (see $f)"
    fi
    # No token after the error event.
    if python3 - "$f" <<'PY'
import sys
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
at = raw.find('"error"')
tail = raw[at:] if at >= 0 else ""
sys.exit(0 if '"content":"' not in tail and '"reasoning_content":"' not in tail else 1)
PY
    then :; else say "FAIL: client $i received a token after the error event"; fi
  else
    interrupted=$((interrupted + 1))
    say "client $i: $chunks token deltas, then the connection closed (rank 0 was the victim) — $(grep -c '"error"' "$f") error events"
  fi
done
[ "$interrupted" -ge 1 ] || say "FAIL: no client was interrupted by the kill (every stream completed first — the drill measured nothing)"

# 6. The logs and the ops files.
cp "$LOG/serve_r0.log" "$DRILL/serve_r0.log" 2>/dev/null
grep -h "ENGINE FAILURE\|exiting with status\|rank 0's journal closed\|stream ended\|exited cleanly" "$DRILL/serve_r0.log" | tail -3 | sed 's/^/  rank 0: /' | tee -a "$SUMMARY"
for ((r = 1; r <= 3; ++r)); do
  h=${PEERS[$((r - 1))]}
  timeout 20 scp -q "${SSH_OPTS[@]}" "$SSH_USER@$h:$PEER_DIR/serve_r$r.log" "$DRILL/serve_r$r.log" 2>/dev/null
  grep -h "exiting with status\|rank 0's journal closed\|stream ended\|exited cleanly\|serve: " "$DRILL/serve_r$r.log" 2>/dev/null | tail -2 | sed "s/^/  rank $r: /" | tee -a "$SUMMARY"
  if [ "$r" -eq "$VICTIM" ]; then say "  rank $r: the victim — no op stream this run"; continue; fi
  timeout 20 scp -q "${SSH_OPTS[@]}" "$SSH_USER@$h:$PEER_DIR/serve_rank$r.ops" "$DRILL/serve_rank$r.ops" 2>/dev/null || say "  rank $r: no ops file (died before writing)"
done
cp "$ROOT/serve_rank0.ops" "$DRILL/serve_rank0.ops" 2>/dev/null || say "  rank 0: no ops file"
say "=== ops files (a survivor's stream is rank 0's up to the failure):"
(cd "$DRILL" && wc -l serve_rank*.ops 2>/dev/null | tee -a "$SUMMARY")
# Every survivor's op stream must be a prefix-consistent view: identical
# lines up to the shortest file.
python3 - "$DRILL" "$VICTIM" <<'PY' | tee -a "$SUMMARY"
import glob, os, sys
d = sys.argv[1]
victim = f"serve_rank{sys.argv[2]}.ops"  # killed -9: it wrote nothing this run
files = sorted(f for f in glob.glob(os.path.join(d, "serve_rank*.ops"))
               if os.path.getsize(f) > 0 and os.path.basename(f) != victim)
texts = {os.path.basename(f): open(f, encoding="utf-8", errors="replace").read().splitlines() for f in files}
if len(texts) < 2:
    print("  (fewer than two ops files; no cross-rank comparison)")
else:
    n = min(len(v) for v in texts.values())
    base = next(iter(texts.values()))[:n]
    bad = [k for k, v in texts.items() if v[:n] != base]
    print(f"  op streams agree over the first {n} lines" if not bad else f"  FAIL: op streams disagree within the first {n} lines: {bad}")
PY

# 7. Sweep whatever is left (nothing should be), then restart and replay.
"$SR" down > "$DRILL/down1.log" 2>&1 || true
"$SR" up 2>&1 | tee "$DRILL/up2.log" | tail -2
if ! grep -q "READY" "$DRILL/up2.log"; then say "FAIL: the world did not come back"; exit 1; fi
T_UP=$(grep -o "rank 0 serving: ok ([0-9]*s)" "$DRILL/up2.log" | head -1)
say "=== restarted: $T_UP"
for ((i = 0; i < CLIENTS; ++i)); do
  body_for "$i" true > "$DRILL/replay_$i.body"
  curl -sN --max-time 600 "$HTTP/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$DRILL/replay_$i.body" > "$DRILL/replay_$i.sse" 2>/dev/null
  sse_content "$DRILL/replay_$i.sse" > "$DRILL/replay_$i.txt"
  if [ ! -s "$DRILL/client_$i.committed.txt" ]; then
    say "client $i: no committed tokens (queued at the kill) — nothing to compare"
    continue
  fi
  if python3 - "$DRILL/client_$i.committed.txt" "$DRILL/replay_$i.txt" <<'PY'
import sys
a = open(sys.argv[1], encoding="utf-8").read()
b = open(sys.argv[2], encoding="utf-8").read()
sys.exit(0 if a and b.startswith(a) else 1)
PY
  then
    say "client $i: the $(wc -c < "$DRILL/client_$i.committed.txt")-byte committed text is a prefix of the fresh $(wc -c < "$DRILL/replay_$i.txt")-byte answer — OK"
  else
    say "FAIL: client $i's committed text is NOT a prefix of the fresh answer (client_$i.committed.txt vs replay_$i.txt)"
  fi
done
"$SR" down > "$DRILL/down2.log" 2>&1 || true
say "=== drill done: $(grep -c '^FAIL' "$SUMMARY") failure line(s); artifacts in $DRILL"
grep -q '^FAIL' "$SUMMARY" && exit 1
exit 0
