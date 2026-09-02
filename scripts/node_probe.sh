#!/usr/bin/env bash
# node_probe: 1 Hz memory-pressure sampler for one fabric node.
#
# Prints one line per second with the DELTAS of the kernel counters that
# say "this box is reclaiming memory in someone's allocation path":
#   allocstall   direct-reclaim stalls (a thread blocked in page reclaim
#                because kswapd fell behind — the caller eats the latency)
#   pgmajfault   major page faults (a page read back from disk/swap)
#   pswpin/out   swap traffic
#   pgscan_direct pages scanned by direct reclaim
#   compact_stall direct compaction stalls
# plus absolute MemFree / MemAvailable / Shmem / SwapFree in MiB. The
# 2026-09-02 jitter hunt found the decode boxes at the watermark (82 GiB
# model + the checkpoint's page cache = the whole box); this is the
# per-node witness for "was reclaim running while we were decoding".
#
# The GPU column group (when nvidia-smi is present): SM MHz, power W,
# temperature C, and the clocks-event (throttle) reason bitmask — 0x4 is SW
# power capping, the GB10's habitual one. The 2026-09-02 run after the
# memory fix showed ONE rank at a time ~30% slower for 5-10 s stretches
# (every generation of every window, rotating between peers); this is the
# per-node witness for "was that box's GPU throttled right then".
#
# Usage: node_probe.sh TAG   (TAG is an inert marker fabric_run.sh kills by)
set -u
GPU_QUERY="clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
read_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "gpu=n/a"; return; }
  nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | awk -F', *' '{printf "sm=%s W=%s C=%s throttle=%s", $1, $2, $3, $4}'
}
read_vm() {
  awk '/^(allocstall_normal|allocstall_movable|pgmajfault|pswpin|pswpout|pgscan_direct |pgsteal_direct |compact_stall)/ {v[$1]=$2}
       END {printf "%d %d %d %d %d %d\n", v["allocstall_normal"]+v["allocstall_movable"], v["pgmajfault"], v["pswpin"], v["pswpout"], v["pgscan_direct"], v["compact_stall"]}' /proc/vmstat
}
read_mem() {
  awk '/^(MemFree|MemAvailable|Shmem|SwapFree|Cached):/ {v[$1]=$2}
       END {printf "free=%d avail=%d cached=%d shmem=%d swapfree=%d", v["MemFree:"]/1024, v["MemAvailable:"]/1024, v["Cached:"]/1024, v["Shmem:"]/1024, v["SwapFree:"]/1024}' /proc/meminfo
}
echo "# $(hostname) $(date -u +%FT%TZ) columns: time d_allocstall d_majfault d_swpin d_swpout d_pgscan_direct d_compact_stall | MiB | gpu"
read -r p_stall p_maj p_in p_out p_scan p_comp <<< "$(read_vm)"
while true; do
  sleep 1
  read -r stall maj in out scan comp <<< "$(read_vm)"
  printf "%s %d %d %d %d %d %d | %s | %s\n" "$(date -u +%T)" \
    $((stall - p_stall)) $((maj - p_maj)) $((in - p_in)) $((out - p_out)) \
    $((scan - p_scan)) $((comp - p_comp)) "$(read_mem)" "$(read_gpu)"
  p_stall=$stall; p_maj=$maj; p_in=$in; p_out=$out; p_scan=$scan; p_comp=$comp
done
