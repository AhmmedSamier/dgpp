#!/bin/bash
# M6 Stage 4b: the fabric serving launcher — rank 0 (HTTP ingress +
# the admission journal + resident TP model) + world-1 peers, with the
# fabric_run.sh discipline: head first, the 120s rendezvous window,
# timeout-wrapped ssh, observe by pgrep (never ssh's exit code), and
# peers swept on head death. The 2026-09-01 soak lessons are baked in:
# spawn peers IN PARALLEL (sequential ssh fumbling burned rendezvous
# windows), never trust a bare `ssh 'nohup ... &'` (session hangs
# don't honor ConnectTimeout).
#
# Usage:
#   serve_run.sh up     — stage + boot the world; waits for readiness
#   serve_run.sh down   — SIGINT rank 0 (stop record releases peers),
#                        sweeps strays, fetches + md5s the per-rank
#                        op streams (the §11 evidence)
#   serve_run.sh status — liveness snapshot of every rank
set -u
ROOT=/home/user/workspace/dgpp
BIN=$ROOT/build-ci/glm_serve
PEER_DIR=/tmp/bus4
PEERS=(192.0.2.12 192.0.2.13 192.0.2.14)
RANK0=192.0.2.11
WORLD=4
HTTP_PORT=18080
FABRIC_PORT=29970
JOURNAL_PORT=29971
MODEL=unsloth/GLM-5.3-Flash-FP8
# DGPP_SERVE_KNOBS overrides the engine knobs on EVERY rank (the graph
# modes: "--max-concurrency 1 --kv-capacity 4096 --default-max-tokens 256
# --queue-limit 8 --decode-graph --mtp"); the peers must run the same
# flags as rank 0, which is why this is one string for the whole world.
KNOBS="${DGPP_SERVE_KNOBS:---max-concurrency 2 --kv-capacity 4096 --default-max-tokens 32 --queue-limit 8}"
LOG="${DGPP_SERVE_LOG:-/tmp/opencode/serve_fabric}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

mkdir -p "$LOG"

peer_ssh() {  # timeout-wrapped: a hung session must not stall the launch
  timeout 20 ssh "${SSH_OPTS[@]}" "user@$1" "${2:-true}"
}

stage() {
  echo "=== staging glm_serve to peers"
  for h in "${PEERS[@]}"; do
    # The staging directory lives in /tmp (a reboot empties it), and a
    # previous run's op stream must never be hashed as this run's evidence
    # (a rank killed before it writes one would leave the old file behind).
    peer_ssh "$h" "mkdir -p $PEER_DIR && rm -f $PEER_DIR/serve_rank*.ops" \
      || { echo "STAGE FAILED to $h (ssh)"; return 1; }
    timeout 30 scp -q "${SSH_OPTS[@]}" "$BIN" "user@$h:$PEER_DIR/glm_serve" \
      || { echo "STAGE FAILED to $h"; return 1; }
  done
}

boot_head() {
  echo "=== rank 0 (HTTP :$HTTP_PORT, fabric :$FABRIC_PORT, journal :$JOURNAL_PORT)"
  cd "$ROOT" || exit 1
  # Rank 0 writes its op stream to its CWD at shutdown — sweep any stale
  # file so the §11 ritual can never hash a previous run's evidence.
  rm -f "$ROOT/serve_rank0.ops"
  DGPP_LOG_LEVEL=info setsid nohup "$BIN" --model "$MODEL" $KNOBS \
    --world "$WORLD" --rank 0 --port "$HTTP_PORT" \
    --fabric-port "$FABRIC_PORT" --journal-port "$JOURNAL_PORT" \
    > "$LOG/serve_r0.log" 2>&1 < /dev/null &
  echo $! > "$LOG/r0.pid"
  echo "head pid $(cat "$LOG/r0.pid")"
}

wait_log() {  # wait_log FILE NEEDLE TIMEOUT_S LABEL
  local file=$1 needle=$2 timeout_s=$3 label=$4 waited=0
  while ! grep -q "$needle" "$file" 2>/dev/null; do
    sleep 2; waited=$((waited + 2))
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "TIMEOUT waiting for $label (${timeout_s}s)"; return 1
    fi
  done
  echo "$label: ok (${waited}s)"
}

head_alive() { kill -0 "$(cat "$LOG/r0.pid" 2>/dev/null)" 2>/dev/null; }

sweep_peers() {  # best-effort: any glm_serve left on a peer dies now
  for h in "${PEERS[@]}"; do
    peer_ssh "$h" "pkill -x glm_serve" >/dev/null 2>&1 || true
  done
}

peers_gone() {
  for h in "${PEERS[@]}"; do
    if peer_ssh "$h" "pgrep -x glm_serve" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

cmd_up() {
  stage || { sweep_peers; exit 1; }
  boot_head
  # Head first: the bus rendezvous listener appears after ~20s of CUDA
  # context init; peers connect against it.
  if ! wait_log "$LOG/serve_r0.log" "rendezvous listening" 90 "bus rendezvous"; then
    tail -5 "$LOG/serve_r0.log"; sweep_peers; exit 1
  fi
  # Peers IN PARALLEL (the soak's sequential-spawn lesson), each spawn
  # timeout-wrapped, observed by pgrep — never by ssh's exit code.
  echo "=== peers ${PEERS[*]}"
  local spawn_pids=()
  for i in 1 2 3; do
    h=${PEERS[$((i - 1))]}
    ( timeout 25 ssh "${SSH_OPTS[@]}" "user@$h" \
        "cd $PEER_DIR && DGPP_LOG_LEVEL=info nohup ./glm_serve --model $MODEL $KNOBS \
         --world $WORLD --rank $i --peer $RANK0 --fabric-port $FABRIC_PORT \
         --journal-port $JOURNAL_PORT > serve_r$i.log 2>&1 &" \
        >/dev/null 2>&1 ) &
    spawn_pids+=($!)
  done
  # Wait ONLY for the spawn subshells: a bare `wait` would also collect
  # rank 0's setsid child (without job control setsid does not fork, so
  # glm_serve stays a direct child) and block until the whole SERVE
  # exits — the 2026-09-01 boot appeared to hang on exactly this.
  for p in "${spawn_pids[@]}"; do
    wait "$p" 2>/dev/null || true
  done
  for i in 1 2 3; do
    h=${PEERS[$((i - 1))]}
    peer_ssh "$h" "pgrep -x glm_serve" >/dev/null 2>&1 \
      || echo "WARN: peer $i (rank $i) not observed on $h"
  done
  # Readiness = rank 0's HTTP line: journal world complete + resident
  # materialization (~2-3 min warm) + tokenizer + listen.
  if ! wait_log "$LOG/serve_r0.log" "listening on :$HTTP_PORT" 600 "rank 0 serving"; then
    tail -10 "$LOG/serve_r0.log"; sweep_peers; kill -9 "$(cat "$LOG/r0.pid")" 2>/dev/null
    exit 1
  fi
  echo "READY — curl http://$RANK0:$HTTP_PORT/v1/models  (stop: $0 down)"
}

cmd_down() {
  local pid
  pid=$(cat "$LOG/r0.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "=== SIGINT rank 0 ($pid) — the stop record releases the peers"
    kill -INT "$pid"
    # 240s, not 60: a SIGINT that lands mid-PREFILL (a cold first-touch
    # pass runs 2s/fold x ~90 folds) is honored only at the pass
    # boundary — the 2026-09-01 catch killed rank 0 at 60s mid-collective,
    # the peers ate transport-retry-exceeded, and the sweep had to shoot
    # them (the drain-on-stop debt, one entry in the record). Give the
    # engine time to reach a tick boundary; the SIGKILL remains for the
    # genuinely wedged.
    for _ in $(seq 1 240); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && { echo "rank 0 wedged; SIGKILL"; kill -9 "$pid"; }
  fi
  echo "=== waiting for peers to drain"
  for _ in $(seq 1 30); do
    peers_gone && break
    sleep 2
  done
  peers_gone || { echo "sweeping wedged peers"; sweep_peers; }
  # The §11 evidence: every rank's engine event stream, md5-compared.
  echo "=== op-stream identity (rank 0 + peers)"
  for i in 1 2 3; do
    h=${PEERS[$((i - 1))]}
    timeout 20 scp -q "${SSH_OPTS[@]}" "user@$h:$PEER_DIR/serve_rank$i.ops" \
      "$LOG/serve_rank$i.ops" 2>/dev/null \
      || echo "WARN: no serve_rank$i.ops on $h"
  done
  # Rank 0's op stream lands in its CWD (the repo root) at shutdown —
  # ALWAYS refresh the log-dir copy before hashing: a stale copy from a
  # previous run is exactly the false "identical" the ritual exists to
  # prevent (the 2026-09-01 catch: `down` hashed Stage 4b's file).
  if [ -f "$ROOT/serve_rank0.ops" ]; then
    cp "$ROOT/serve_rank0.ops" "$LOG/"
  else
    echo "WARN: no fresh serve_rank0.ops in $ROOT (rank 0 never wrote one?)"
  fi
  md5sum "$LOG"/serve_rank*.ops 2>/dev/null || echo "(no op streams)"
  echo "=== logs in $LOG"
}

cmd_status() {
  head_alive && echo "rank 0: alive ($(cat "$LOG/r0.pid"))" || echo "rank 0: DOWN"
  tail -2 "$LOG/serve_r0.log" 2>/dev/null
  for h in "${PEERS[@]}"; do
    echo -n "$h: "
    peer_ssh "$h" "pgrep -x glm_serve | wc -l" 2>/dev/null || echo "?"
  done
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  *) echo "usage: $0 up|down|status"; exit 2 ;;
esac
