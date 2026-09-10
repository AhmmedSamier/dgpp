#!/usr/bin/env bash
# fabric_run: launch glm_gen_check across the fabric (this node as rank 0
# head + the peer ranks via ssh) as ONE atomic job.
#
# WHY THIS EXISTS: the bus rendezvous is a 120s window that opens when rank 0
# starts listening, and peers reach bus-start ~1s after launch. Manual
# attempts that launched peers first, or took minutes to fumble ssh, burned
# the window and died with "rendezvous connect: Connection refused" /
# "rendezvous: table read failed" (the 2026-09-01 Stage 3b smoke needed
# three tries to get the ORDER right). The working order, encoded here:
#   0. refuse if any glm_gen_check is already running on any rank
#      (--force kills them first)
#   1. stage the binary to the peers — scp BEFORE the window opens
#   2. pick a free rendezvous port (probe; --port overrides) so rebinds
#      never collide with TIME_WAIT or a stale listener
#   3. launch rank 0, wait for "rendezvous listening" in its log
#   4. launch ranks 1..N by ssh, fire-and-forget — ssh sessions may
#      linger even with -n and </dev/null, so peers are verified by
#      OBSERVING the remote process (pgrep), never by ssh's exit code
#   5. monitor rank 0 until it exits; a --timeout watchdog kills every
#      rank on the head's behalf if it wedges
#   6. collect: EOS line, per-rank generated-ids md5 (the log line's
#      rank prefix stripped — the 2026-09-01 lesson: hash the payload,
#      not the line), rank-consensus verdict, bus stats
#
# The model resolves from each node's own HF cache (--model ID), so the
# only artifact the peers need is the staged binary.
#
# Usage (from the repo root, ON THE HEAD NODE — rank 0 is this box):
#   scripts/fabric_run.sh [options] -- APP_ARGS...
#
#   APP_ARGS are passed verbatim to glm_gen_check on EVERY rank; the
#   script appends --world/--rank/--peer/--port itself (do not pass
#   them yourself).
#
# Options:
#   --app PATH      binary to stage + run (default $BUILD/glm_gen_check)
#   --port N        rendezvous port (default: first free of 29970..29989)
#   --timeout SECS  head watchdog; 0 disables (default 1800)
#   --no-stage      skip the peer scp (peers already carry this binary)
#   --stage-file F  ALSO scp F to the peers and rewrite the peers'
#                  --requests / --teacher-file argument to the copy (rank 0 keeps
#                  the local path — the app's manifest-hash log line is
#                  the cross-rank identity check). Stages even under
#                  --no-stage: the manifest changes more often than the
#                  binary and costs nothing to ship.
#   --force         pkill -x glm_gen_check on all ranks before starting
#   --log-dir DIR   logs land here (default $BUILD/fabric-runs/<UTC ts>)
#   --head-wrap CMD prefix rank 0's command line with CMD (word-split) —
#                  the profiling hook: e.g. --head-wrap "nsys profile
#                  --trace=cuda -o /tmp/r0" traces rank 0's kernels while
#                  the peers run bare. The wrapper must pass stdout
#                  through (the launch waits on the "rendezvous
#                  listening" log line) and exit with the app.
#   --fetch-logs    after the run, scp every peer's log into LOG_DIR as
#                  rN.log next to r0.log — the cross-rank analysis tools
#                  (scripts/fabric_xrank.py) read the directory as a unit.
#                  Cross-rank correlation by wall clock is hopeless (the
#                  boxes disagree by hours); by bus generation it is exact.
#   --node-probe    run scripts/node_probe.sh on EVERY node for the run's
#                  duration (1 Hz /proc/vmstat + meminfo deltas into the
#                  node's $PEER_DIR/probe_rN.log; fetched with the logs).
#                  The 2026-09-02 jitter hunt's question was "is the box
#                  reclaiming memory under us?" — this answers it per node.
#   --node-cmd CMD  run CMD (a shell string) on every node in the
#                  background for the run's duration, killed at the end —
#                  the knob hook for "what if the node did X during the
#                  run" (e.g. a drop_caches loop). Runs as $FABRIC_USER;
#                  passwordless sudo is available on the lab fabric.
#
# Examples:
#   scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
#       --chat "The capital of France is" \
#       --system "You are a concise assistant." --steps 64
#   scripts/fabric_run.sh --no-stage -- \
#       --model unsloth/GLM-5.3-Flash-FP8 --text "Hello" --steps 2
#   scripts/fabric_run.sh --stage-file build-ci/sched_smoke.jsonl -- \
#       --model unsloth/GLM-5.3-Flash-FP8 --requests build-ci/sched_smoke.jsonl \
#       --max-concurrency 2 --kv-capacity 256
#
# Fabric layout via env (defaults are the lab fabric):
#   DGPP_FABRIC_HEAD   head's fabric IP as peers --peer it (192.0.2.11)
#   DGPP_FABRIC_PEERS  space-separated peer IPs (192.0.2.12..14)
#   DGPP_FABRIC_USER   ssh user (default: the caller's own login)
#   DGPP_PEER_DIR      staging dir on peers (/tmp/bus4)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${DGPP_BUILD_DIR:-$ROOT/build-ci}"
APP="$BUILD/glm_gen_check"
FABRIC_HEAD="${DGPP_FABRIC_HEAD:-192.0.2.11}"
FABRIC_PEERS=(${DGPP_FABRIC_PEERS:-192.0.2.12 192.0.2.13 192.0.2.14})
FABRIC_USER="${DGPP_FABRIC_USER:-$(id -un)}"
PEER_DIR="${DGPP_PEER_DIR:-/tmp/bus4}"
SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=8)
MONITOR_TIMEOUT=1800
STAGE=1
STAGE_FILE=""
FORCE=0
LOG_DIR=""
HEAD_WRAP=()
FETCH_LOGS=0
NODE_PROBE=0
NODE_CMD=""

die() { echo "fabric_run: $*" >&2; exit 1; }
peer_ssh() { timeout 20 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$1" "${2:-true}"; }

# ---------------------------------------------------------------- options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --timeout) MONITOR_TIMEOUT="$2"; shift 2 ;;
    --no-stage) STAGE=0; shift ;;
    --stage-file) STAGE_FILE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --head-wrap) read -r -a HEAD_WRAP <<< "$2"; shift 2 ;;
    --fetch-logs) FETCH_LOGS=1; shift ;;
    --node-probe) NODE_PROBE=1; shift ;;
    --node-cmd) NODE_CMD="$2"; shift 2 ;;
    --) shift; break ;;
    *) die "unknown option $1 (app args go after --)" ;;
  esac
done
APP_ARGS=("$@")
[[ ${#APP_ARGS[@]} -gt 0 ]] || die "no app args after -- (e.g. -- --model ID --chat TEXT)"
[[ -x "$APP" ]] || die "app not found/executable: $APP (build it, or pass --app)"
# The staged binary keeps its name on the peers; every pkill/pgrep below
# matches THAT name, so --app works for any fabric app, not just the
# default (the 2026-09-02 hunt wanted bus_check and a synthetic decode
# loop through the same launcher).
APP_NAME="$(basename "$APP")"
WORLD=$((1 + ${#FABRIC_PEERS[@]}))
LOG_DIR="${LOG_DIR:-$BUILD/fabric-runs/$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
export DGPP_LOG_LEVEL="${DGPP_LOG_LEVEL:-info}"  # render/eos lines are INFO
PROBE_SCRIPT="$ROOT/scripts/node_probe.sh"
ALL_NODES=(127.0.0.1 "${FABRIC_PEERS[@]}")  # index == rank

# ------------------------------------------------------------- cleanup
kill_all() {
  pkill -x "$APP_NAME" 2>/dev/null || true
  for ip in "${FABRIC_PEERS[@]}"; do
    timeout 15 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$ip" \
      "pkill -x $APP_NAME" 2>/dev/null || true
  done
  stop_node_helpers
}
# Per-node helpers (probe sampler, --node-cmd) are tagged by a marker in
# their command line so they can be killed by pattern without touching
# anything else; the marker carries the run's log dir name for uniqueness.
HELPER_TAG="fabric_run_helper_$(basename "$LOG_DIR")"
stop_node_helpers() {
  [[ $NODE_PROBE -eq 1 || -n "$NODE_CMD" ]] || return 0
  # The head's helpers are killed locally: ssh to our own address is not
  # guaranteed to work (host keys), and a dozen runs' worth of probe loops
  # were found still running here for exactly that reason.
  pkill -f "$HELPER_TAG" >/dev/null 2>&1 || true
  for ip in "${FABRIC_PEERS[@]}"; do
    timeout 15 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$ip" \
      "pkill -f $HELPER_TAG" >/dev/null 2>&1 || true
  done
}
trap 'kill_all' INT TERM
# (pkill -x matches the process NAME exactly — it can never kill this script)

# ------------------------------------------------------- stale / staging
if [[ $FORCE -eq 1 ]]; then
  echo "fabric_run: --force — killing any stale $APP_NAME on all ranks"
  kill_all
  sleep 2
fi
pgrep -x "$APP_NAME" >/dev/null && die "a $APP_NAME already runs here (--force to kill)"
for ip in "${FABRIC_PEERS[@]}"; do
  peer_ssh "$ip" "pgrep -x $APP_NAME" >/dev/null 2>&1 \
    && die "a $APP_NAME already runs on $ip (--force to kill)"
done

if [[ $STAGE -eq 1 ]]; then
  for ip in "${FABRIC_PEERS[@]}"; do
    echo "fabric_run: staging $APP -> $FABRIC_USER@$ip:$PEER_DIR/"
    scp -o BatchMode=yes -o ConnectTimeout=8 "$APP" \
      "$FABRIC_USER@$ip:$PEER_DIR/" >/dev/null || die "staging to $ip failed"
  done
fi
if [[ $NODE_PROBE -eq 1 || -n "$NODE_CMD" ]]; then
  [[ -x "$PROBE_SCRIPT" ]] || die "node probe script missing: $PROBE_SCRIPT"
  for ip in "${FABRIC_PEERS[@]}"; do
    scp -o BatchMode=yes -o ConnectTimeout=8 "$PROBE_SCRIPT" \
      "$FABRIC_USER@$ip:$PEER_DIR/" >/dev/null || die "staging probe to $ip failed"
  done
fi
if [[ -n "$STAGE_FILE" ]]; then
  [[ -f "$STAGE_FILE" ]] || die "--stage-file not found: $STAGE_FILE"
  for ip in "${FABRIC_PEERS[@]}"; do
    scp -o BatchMode=yes -o ConnectTimeout=8 "$STAGE_FILE" \
      "$FABRIC_USER@$ip:$PEER_DIR/" >/dev/null \
      || die "staging $STAGE_FILE to $ip failed"
  done
  echo "fabric_run: staged $STAGE_FILE (peers read it as $PEER_DIR/$(basename "$STAGE_FILE"))"
fi

# ------------------------------------------------------------ the port
if [[ -z "${PORT:-}" ]]; then
  for p in $(seq 29970 29989); do
    # connect_ex == 0 means something answers: not free. Probe, don't bind —
    # rank 0 owns the listener, TIME_WAIT on peers' sides is not our problem.
    if ! python3 -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',$p))==0 else 1)" "$p"; then
      PORT="$p"; break
    fi
  done
fi
[[ -n "${PORT:-}" ]] || die "no free port in 29970..29989 (pass --port)"

# ------------------------------------------------------------- rank 0
echo "fabric_run: world $WORLD, port $PORT, logs in $LOG_DIR"

# Per-node helpers start BEFORE the ranks so the probe's first sample is
# the pre-load baseline. Rank 0's helpers write next to r0.log; peers'
# into $PEER_DIR (fetched with --fetch-logs). The tag makes them killable.
# The lab boxes' clocks disagree by hours (no NTP on two of them), and the
# probes stamp with THEIR clock. Record each node's offset from the head
# (peer_now - head_now, seconds; half an ssh round trip of error, well
# under the probes' 1 s grain) so fabric_xrank.py can line probe rows up
# with rank 0's log.
record_clock_offsets() {
  : > "$LOG_DIR/clock_offsets.txt"
  for i in "${!ALL_NODES[@]}"; do
    ip="${ALL_NODES[$i]}"
    local t0 t1 peer
    t0=$(date -u +%s.%N)
    peer=$( [[ $i -eq 0 ]] && date -u +%s.%N || peer_ssh "$ip" "date -u +%s.%N" ) || peer=""
    t1=$(date -u +%s.%N)
    [[ -n "$peer" ]] || { echo "r$i ?" >> "$LOG_DIR/clock_offsets.txt"; continue; }
    awk -v r="$i" -v p="$peer" -v a="$t0" -v b="$t1" \
      'BEGIN { printf "r%s %.3f\n", r, p - (a + b) / 2 }' >> "$LOG_DIR/clock_offsets.txt"
  done
}

start_node_helpers() {
  [[ $NODE_PROBE -eq 1 ]] && record_clock_offsets
  for i in "${!ALL_NODES[@]}"; do
    ip="${ALL_NODES[$i]}"
    local dir="$PEER_DIR" runner="peer_ssh $ip"
    [[ $i -eq 0 ]] && { dir="$LOG_DIR"; runner="bash -c"; }
    if [[ $NODE_PROBE -eq 1 ]]; then
      local probe="$dir/node_probe.sh"; [[ $i -eq 0 ]] && probe="$PROBE_SCRIPT"
      $runner "nohup $probe $HELPER_TAG > $dir/probe_r$i.log 2>&1 < /dev/null &" || true
    fi
    if [[ -n "$NODE_CMD" ]]; then
      # sh -c with the tag as $0: pgrep -f sees the tag, the command runs verbatim.
      $runner "nohup sh -c $(printf '%q' "$NODE_CMD") $HELPER_TAG > $dir/nodecmd_r$i.log 2>&1 < /dev/null &" || true
    fi
  done
}
start_node_helpers

nohup "${HEAD_WRAP[@]}" "$APP" "${APP_ARGS[@]}" --world "$WORLD" --rank 0 --port "$PORT" \
  > "$LOG_DIR/r0.log" 2>&1 < /dev/null &
HEAD_PID=$!
# plain nohup (no setsid): the head stays our child so `wait` reaps its code

for _ in $(seq 1 60); do
  grep -q "rendezvous listening" "$LOG_DIR/r0.log" 2>/dev/null && break
  kill -0 "$HEAD_PID" 2>/dev/null \
    || { tail -5 "$LOG_DIR/r0.log"; die "rank 0 died before listening"; }
  sleep 0.5
done
grep -q "rendezvous listening" "$LOG_DIR/r0.log" \
  || die "rank 0 never listened (30s) — see $LOG_DIR/r0.log"
echo "fabric_run: rank 0 listening on :$PORT"

# ------------------------------------------------------- peers 1..N-1
# The peers' args: verbatim, except a staged --requests / --teacher-file
# path points at the peer's copy (rank 0 reads the local file — identical
# bytes, and the app logs the file's hash on every rank as the identity
# check).
REMOTE_ARGS=""
prev=""
for a in "${APP_ARGS[@]}"; do
  if [[ -n "$STAGE_FILE" && ( "$prev" == "--requests" || "$prev" == "--teacher-file" ) ]]; then
    REMOTE_ARGS+="$(printf '%q ' "$PEER_DIR/$(basename "$STAGE_FILE")")"
  else
    REMOTE_ARGS+="$(printf '%q ' "$a")"
  fi
  prev="$a"
done
# Every DGPP_* knob in our environment reaches the peers too (the app's
# knobs: DGPP_MLOCK, DGPP_RESIDENT_CACHE*, DGPP_LOG_LEVEL, ...) — except
# this script's own, which describe the head's side of the world.
REMOTE_ENV=""
while IFS='=' read -r name value; do
  case "$name" in
    DGPP_PEER_DIR|DGPP_BUILD_DIR|DGPP_FABRIC_*) continue ;;
    DGPP_*) REMOTE_ENV+="$name=$(printf '%q' "$value") " ;;
  esac
done < <(env)
for i in "${!FABRIC_PEERS[@]}"; do
  rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
  # Fire-and-forget on purpose (see header): the remote side is fully
  # detached; whether THIS ssh returns is irrelevant to the launch.
  ( timeout 25 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$ip" \
      "cd $PEER_DIR && $REMOTE_ENV nohup ./$APP_NAME $REMOTE_ARGS \
       --world $WORLD --rank $rank --peer $FABRIC_HEAD --port $PORT \
       > $PEER_DIR/fabric_r$rank.log 2>&1 < /dev/null &" \
      >/dev/null 2>&1 ) &
done
for i in "${!FABRIC_PEERS[@]}"; do
  rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
  up=0
  for _ in $(seq 1 20); do
    peer_ssh "$ip" "pgrep -x $APP_NAME" >/dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  if [[ $up -ne 1 ]]; then
    peer_ssh "$ip" "tail -5 $PEER_DIR/fabric_r$rank.log" || true
    kill_all
    die "peer rank $rank ($ip) never started (20s) — everything killed"
  fi
  echo "fabric_run: rank $rank up on $ip"
done

# ------------------------------------------------------------- monitor
elapsed=0
while kill -0 "$HEAD_PID" 2>/dev/null; do
  if [[ $MONITOR_TIMEOUT -gt 0 && $elapsed -ge $MONITOR_TIMEOUT ]]; then
    kill_all
    die "watchdog: rank 0 still running after ${MONITOR_TIMEOUT}s — everything killed; logs in $LOG_DIR"
  fi
  sleep 10; elapsed=$((elapsed + 10))
done
HEAD_RC=0; wait "$HEAD_PID" || HEAD_RC=$?

# A failed head leaves the peers wedged in collectives forever (they
# wait for a rank that will never post again — the 2026-09-01 hunt
# measured 39s+ stalls and manual pkill on every box). Kill them here so
# a failed run needs no cleanup; the peers' logs were already flushed by
# their own nohup redirection.
if [[ $HEAD_RC -ne 0 ]]; then
  echo "fabric_run: head failed — killing wedged peers"
  kill_all
fi
stop_node_helpers

# ------------------------------------------------------------ collect
echo "fabric_run: rank 0 exited rc=$HEAD_RC — collecting verdicts"
# md5 over EVERY "generated ids" line (scheduler mode logs one per
# request, in deterministic order) — the payload after the colon, so the
# embedded rank prefix cannot differ. Missing lines mean failure.
md5_of_generated() { grep "generated ids" "$1" | sed 's/^.*generated ids: //' | md5sum | awk '{print $1}'; }
head_md5="$(md5_of_generated "$LOG_DIR/r0.log" || true)"
echo "  head : ${head_md5:-<no generated line>}"
for i in "${!FABRIC_PEERS[@]}"; do
  rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
  peer_md5="$(peer_ssh "$ip" \
    "grep 'generated ids' $PEER_DIR/fabric_r$rank.log | sed 's/^.*generated ids: //' | md5sum | awk '{print \$1}'" 2>/dev/null || true)"
  echo "  rank $rank: ${peer_md5:-<no generated line>}"
  [[ -n "$peer_md5" && "$peer_md5" == "$head_md5" ]] || HEAD_RC=1
done
echo "fabric_run: rank consistency $([[ -n "$head_md5" ]] && echo "$head_md5" || echo 'n/a') — $([[ $HEAD_RC -eq 0 ]] && echo 'ALL RANKS IDENTICAL' || echo 'MISMATCH or missing')"
grep -m1 "eos stop" "$LOG_DIR/r0.log" || true
grep "retired (" "$LOG_DIR/r0.log" || true
grep -m1 -E "steps in|step cap in" "$LOG_DIR/r0.log" || true

if [[ $FETCH_LOGS -eq 1 ]]; then
  for i in "${!FABRIC_PEERS[@]}"; do
    rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
    scp -q -o BatchMode=yes -o ConnectTimeout=8 \
      "$FABRIC_USER@$ip:$PEER_DIR/fabric_r$rank.log" "$LOG_DIR/r$rank.log" \
      || echo "fabric_run: could not fetch rank $rank's log"
    # One scp per file: OpenSSH's sftp-backed scp expands no remote globs.
    [[ $NODE_PROBE -eq 1 ]] && scp -q -o BatchMode=yes -o ConnectTimeout=8 \
        "$FABRIC_USER@$ip:$PEER_DIR/probe_r$rank.log" "$LOG_DIR/" 2>/dev/null || true
    [[ -n "$NODE_CMD" ]] && scp -q -o BatchMode=yes -o ConnectTimeout=8 \
        "$FABRIC_USER@$ip:$PEER_DIR/nodecmd_r$rank.log" "$LOG_DIR/" 2>/dev/null || true
  done
  # The cross-rank verdict the hunt lives on: per-rank step distribution,
  # inter-step host gaps, and every gen-0 stall with all ranks' views.
  if [[ -x "$ROOT/scripts/fabric_xrank.py" ]]; then
    python3 "$ROOT/scripts/fabric_xrank.py" "$LOG_DIR" --summary || true
  fi
fi
echo "fabric_run: full logs in $LOG_DIR $([[ $FETCH_LOGS -eq 1 ]] && echo '(r0..rN.log fetched)' || echo '(r0.log local, fabric_rN.log on peers)')"
exit "$HEAD_RC"
