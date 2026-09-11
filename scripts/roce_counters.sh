#!/bin/bash
# RoCE hardware counters across the fabric's nodes — the wire-side view of a
# collective run: retransmissions, sequence errors and congestion
# notifications per active RoCE device, plus the NIC's own ingress
# discards and pause frames. Snapshot mode prints one line per (node,
# device) with the cumulative counters; diff mode subtracts two snapshots
# and prints the deltas, so a run's own retransmits and CNPs are legible
# against the since-boot totals.
#
# Usage: scripts/roce_counters.sh snapshot > before.txt
#        ... run ...
#        scripts/roce_counters.sh snapshot > after.txt
#        scripts/roce_counters.sh diff before.txt after.txt
# Nodes come from .env, limited to the selected deployment's world_size.
# Devices with no traffic at all are skipped.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
# The shared SSH login comes from .env (or an exported override).
SSH_USER=$(dgpp_ssh_user) || exit 1
NODES=$(dgpp_nodes) || exit 1
mode=${1:-snapshot}
case "$mode" in
  snapshot)
    for h in $NODES; do
      ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_USER@$h" bash -s 2>/dev/null <<'REMOTE' | sed "s/^/$h /"
for d in /sys/class/infiniband/*/ports/1; do
  dev=$(basename "$(dirname "$(dirname "$d")")")
  data=$(cat "$d/counters/port_xmit_data" 2>/dev/null || echo 0)
  [ "$data" = "0" ] && continue
  line="$dev xmit_data=$data"
  for c in packet_seq_err out_of_sequence roce_adp_retrans roce_adp_retrans_to local_ack_timeout_err rnr_nak_retry_err implied_nak_seq_err np_cnp_sent rp_cnp_handled np_ecn_marked_roce_packets rx_icrc_encapsulated; do
    [ -f "$d/hw_counters/$c" ] && line="$line $c=$(cat "$d/hw_counters/$c")"
  done
  ifc=$(ls "/sys/class/infiniband/$dev/device/net" 2>/dev/null | head -1)
  if [ -n "$ifc" ]; then
    # The NIC's own view: packets discarded at ingress for lack of buffer
    # (the PCIe drain slower than the wire) and pause frames sent/seen.
    stats=$(ethtool -S "$ifc" 2>/dev/null)
    for c in rx_discards_phy rx_buffer_passed_thres_phy tx_pause_ctrl_phy rx_pause_ctrl_phy; do
      val=$(echo "$stats" | awk -v k="$c:" '$1 == k {print $2}')
      [ -n "$val" ] && line="$line $c=$val"
    done
  fi
  echo "$line"
done
REMOTE
    done ;;
  diff)
    python3 "$ROOT/scripts/roce_report.py" "$2" "$3"
    ;;
  *) echo "usage: $0 snapshot | diff BEFORE AFTER" >&2; exit 2 ;;
esac
