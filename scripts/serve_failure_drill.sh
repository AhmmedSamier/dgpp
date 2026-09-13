#!/bin/bash
# The failure drill (M8's exit criterion "injected rank failure leaves
# committed state unchanged"; M9's "kill -9 a random rank under load";
# built 2026-09-05): kill -9 one rank of the serving world while
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
# Usage: serve_failure_drill.sh <victim rank> [clients (1..3, default 3)]
# The victim is a rank of the selected deployment's world (0..world-1).
# Set DGPP_SERVE_KNOBS as for serve_run.sh (the same knobs on every rank).
# Everything lands under $DGPP_DRILL_DIR (default build-ci/fabric-runs/
# failure_drill_<stamp>/): the SSE captures, the logs, the ops files, the
# fresh answers, and summary.txt.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
dgpp_require_peers || exit 1
WORLD=$(dgpp_world) || exit 1
SR=$ROOT/scripts/serve_run.sh
LOG=$(dgpp_log_dir) || exit 1
LOG=${LOG/#\~/$HOME}
PEER_LIST=$(dgpp_peers) || exit 1
read -r -a PEERS <<< "$PEER_LIST"
RANK0=$(dgpp_head) || exit 1
HTTP="http://$(dgpp_client_host):$(dgpp_http_port)"
PEER_DIR=$(dgpp_stage_dir) || exit 1
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)
# The shared SSH login comes from .env (or an exported override).
SSH_USER=$(dgpp_ssh_user) || exit 1
VICTIM=${1:?usage: $0 <victim rank> [clients]}
CLIENTS=${2:-3}
[[ "$VICTIM" =~ ^[0-9]+$ && "$VICTIM" -lt "$WORLD" && "$CLIENTS" =~ ^[1-3]$ ]] ||
  { echo "victim must be 0..$((WORLD - 1)) and clients 1..3" >&2; exit 2; }
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
  python3 "$ROOT/scripts/failure_report.py" request-body "$stream" "$MAX_TOKENS" "${PROMPTS[$i]}" "$MODEL"
}

# Every token the client received, concatenated: the reasoning deltas and
# the content deltas, in order (the committed text the prefix check is over).
sse_content() {
  python3 "$ROOT/scripts/serve_streams.py" text "$1"
}

say "=== failure drill: victim rank $VICTIM, $CLIENTS streaming client(s), knobs: ${DGPP_SERVE_KNOBS:-<serve_run.sh default>}"
say "=== artifacts: $DRILL"

# 1. Boot.
"$SR" up 2>&1 | tee "$DRILL/up1.log" | tail -3
grep -q "READY" "$DRILL/up1.log" || { say "FAIL: the world did not come up"; exit 1; }
MODEL=$(dgpp_served_model) || exit 1

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
sleep "$(python3 "$ROOT/scripts/run_helpers.py" random-delay)"  # a random moment under load

# 3. The kill.
T_KILL=$(date +%s.%N)
say "=== kill -9 recorded rank $VICTIM at $(date +%T.%N)"
signal_args=(signal --rank "$VICTIM" --signal KILL)
[[ -n "${DGPP_SERVE_LOG:-}" ]] && signal_args+=(--log-dir "$DGPP_SERVE_LOG")
python3 "$ROOT/scripts/dgpp-cluster" "${signal_args[@]}" || exit 1

# 4. The ranks leave. Rank 0: gone within 30 s of the kill.
r0_gone=""
for _ in $(seq 1 300); do
  if ! python3 "$ROOT/scripts/cluster_process.py" status --state "$LOG/rank0.process.json" >/dev/null; then
    r0_gone=$(python3 "$ROOT/scripts/run_helpers.py" elapsed "$T_KILL"); break
  fi
  sleep 0.1
done
if [ -n "$r0_gone" ]; then say "rank 0: gone ${r0_gone}s after the kill"; else say "FAIL: rank 0 still alive 30 s after the kill"; fi
for ((r = 1; r < WORLD; ++r)); do
  h=${PEERS[$((r - 1))]}
  gone=""
  for _ in $(seq 1 300); do
    if ! peer_ssh "$h" "python3 $PEER_DIR/cluster_process.py status --state $PEER_DIR/rank$r.process.json" >/dev/null 2>&1; then
      gone=$(python3 "$ROOT/scripts/run_helpers.py" elapsed "$T_KILL"); break
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
    if python3 "$ROOT/scripts/serve_streams.py" no-tokens-after-error "$f"
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
for ((r = 1; r < WORLD; ++r)); do
  h=${PEERS[$((r - 1))]}
  timeout 20 scp -q "${SSH_OPTS[@]}" "$SSH_USER@$h:$PEER_DIR/serve_r$r.log" "$DRILL/serve_r$r.log" 2>/dev/null
  grep -h "exiting with status\|rank 0's journal closed\|stream ended\|exited cleanly\|serve: " "$DRILL/serve_r$r.log" 2>/dev/null | tail -2 | sed "s/^/  rank $r: /" | tee -a "$SUMMARY"
  if [ "$r" -eq "$VICTIM" ]; then say "  rank $r: the victim — no op stream this run"; continue; fi
  timeout 20 scp -q "${SSH_OPTS[@]}" "$SSH_USER@$h:$PEER_DIR/serve_rank$r.ops" "$DRILL/serve_rank$r.ops" 2>/dev/null || say "  rank $r: no ops file (died before writing)"
done
# Rank 0 streams its op stream into its cwd, the deployment's log dir
# (the launcher's namespace under DGPP_LOG_DIR, or DGPP_SERVE_LOG's).
R0_LOG=$(dgpp_log_dir) && R0_LOG=${R0_LOG/#\~/$HOME}
cp "$R0_LOG/serve_rank0.ops" "$DRILL/serve_rank0.ops" 2>/dev/null || say "  rank 0: no ops file ($R0_LOG)"
say "=== ops files (a survivor's stream is rank 0's up to the failure):"
(cd "$DRILL" && wc -l serve_rank*.ops 2>/dev/null | tee -a "$SUMMARY")
# Every survivor's op stream must be a prefix-consistent view: identical
# lines up to the shortest file.
python3 "$ROOT/scripts/failure_report.py" ops "$DRILL" "$VICTIM" | tee -a "$SUMMARY"

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
  if python3 "$ROOT/scripts/failure_report.py" prefix "$DRILL/client_$i.committed.txt" "$DRILL/replay_$i.txt"
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
