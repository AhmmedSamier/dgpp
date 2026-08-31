#!/bin/bash
# soak: the M5 exit-gate mixed-class bus soak as ONE re-runnable command
# (the procedure that passed gate 2 on 2026-08-31; see
# benchmarks/results/2026-08-29-bus-m5.md).
#
# Cycles N fresh (serve, ping --contend --soak-ms) sessions between this
# node (rank 0, the receiver) and a peer (rank 1, the sender), each a
# duration-bounded mix of a bulk flood and decode-class latency probes
# running CONCURRENTLY. The rendezvous listener is one-shot by design,
# so every phase is a fresh bus session — the repeated setup/teardown is
# the harder test anyway.
#
# Gate verdict (exit 0): every phase's ping exits 0 with zero failures.
# After the run, verify the logs for the rest of the criteria:
#   grep "SOAK"     -> per-phase bulk/lat tails + throughput
#   grep "lane peer" -> lane contribution (multi-stripe phases must show
#                      both lanes; single-stripe bulk is lane-0 by the
#                      striping rule, latency class is lane-0 by contract)
#   grep -E "ERROR|watchdog" -> must find nothing
#
# Sizes rotate by design: multi-stripe bulk exercises both lanes;
# including one slot-sized (single-stripe) phase documents the
# by-design lane-0 pinning. The last phase repeats the first size —
# the across-the-hour drift check.
#
# Usage (from the repo root, on the RANK 0 node):
#   scripts/soak.sh PEER_USER@PEER_HOST [PHASE_MINUTES] [BIN_DIR]
#
#   PEER_USER@PEER_HOST   the rank-1 node; expects the bus_check binary
#                         at /tmp/bus4/bus_check there (the standard
#                         staging location) and key-based ssh
#   PHASE_MINUTES         per-phase soak duration (default 15)
#   BIN_DIR               local build dir holding bus_check (default
#                         build-ci)
#
# Examples:
#   scripts/soak.sh user@192.0.2.12              # the gate run: 4x15 min
#   scripts/soak.sh user@192.0.2.12 3             # a 4x3 min smoke
set -u

if [ $# -lt 1 ]; then
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
PEER=$1
PHASE_MIN=${2:-15}
BIN_DIR=${3:-build-ci}

BIN="$BIN_DIR/bus_check"
LOG=/tmp/opencode/soak
PORT_BASE=29980
SOAK_MS=$((PHASE_MIN * 60 * 1000))

[ -x "$BIN" ] || { echo "soak: $BIN not found (build it first)" >&2; exit 2; }

# Rotate sizes; the last repeats the first (drift check). Odd phase
# counts still end on a repeat if the count is even — adjust below if
# you change this.
PHASES=(1048576 262144 4194304 1048576)

mkdir -p "$LOG"
rm -f "$LOG"/*.log 2>/dev/null
overall_rc=0
phase=0

for bytes in "${PHASES[@]}"; do
  port=$((PORT_BASE + phase))
  slog="$LOG/serve_p$phase.log"
  plog="$LOG/ping_p$phase.log"

  echo "== phase $phase: bulk=$bytes soak=${PHASE_MIN}m port=$port"

  # Fresh serve (rank 0) on this node; duration caps a wedged phase.
  setsid nohup "$BIN" serve --port "$port" --world 2 \
      --duration-ms $((SOAK_MS + 180000)) > "$slog" 2>&1 &
  serve_pid=$!

  up=0
  for _ in $(seq 1 90); do
    grep -q "rendezvous listening" "$slog" 2>/dev/null && { up=1; break; }
    sleep 2
  done
  if [ "$up" != 1 ]; then
    echo "soak: phase $phase serve never listened" | tee -a "$LOG/driver.log"
    kill "$serve_pid" 2>/dev/null
    overall_rc=1
    phase=$((phase + 1))
    continue
  fi

  ssh -o BatchMode=yes "$PEER" "cd /tmp/bus4 && ./bus_check ping \
      --peer $(hostname -I | awk '{print $1}') --port $port --contend \
      --soak-ms $SOAK_MS --bytes $bytes > ping_p$phase.log 2>&1; \
      echo exit=\$? >> ping_p$phase.log"
  rc=$?
  scp -o BatchMode=yes -q "$PEER:/tmp/bus4/ping_p$phase.log" "$plog" 2>/dev/null

  if [ $rc -ne 0 ] || ! grep -q "exit=0" "$plog" 2>/dev/null; then
    echo "soak: phase $phase FAILED (rc=$rc)" | tee -a "$LOG/driver.log"
    overall_rc=1
  fi
  grep -E "SOAK (bulk|lat):" "$plog" 2>/dev/null | tail -2
  echo "phase $phase done (bytes=$bytes)" >> "$LOG/driver.log"

  kill "$serve_pid" 2>/dev/null
  pkill -x bus_check 2>/dev/null
  sleep 3
  phase=$((phase + 1))
done

echo "soak complete rc=$overall_rc (logs in $LOG)"
exit $overall_rc
