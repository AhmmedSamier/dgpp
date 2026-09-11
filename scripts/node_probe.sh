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
# memory fix showed one rank at a time ~30% slower for 5-10 s stretches
# (every generation of every window, rotating between peers); this is the
# per-node witness for "was that box's GPU throttled right then".
#
# The CPU column group: whole-box busy / iowait / irq+softirq percentages
# from /proc/stat, then the top three processes by CPU over the last second
# (comm:percent, from per-pid utime+stime deltas — USER_HZ is 100, so ticks
# per second ARE percent of one core). The rotating slow rank shows no GPU
# clock or throttle signal, so the next suspect is something ELSE on that
# box eating memory bandwidth or a core the engine wants; this is the
# per-node witness for "who else was running right then".
#
# Usage: node_probe.sh TAG   (TAG is an inert marker fabric_run.sh kills by)
set -u
PREV_PIDS="$(mktemp -t node_probe_pids.XXXXXX)"
trap 'rm -f "$PREV_PIDS"' EXIT
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
# Whole-box CPU ticks: busy (user nice system irq softirq steal), iowait,
# irq+softirq, and total — the caller turns deltas into percentages.
read_cpu() {
  awk '$1 == "cpu" {busy=$2+$3+$4+$7+$8+$9; total=busy+$5+$6;
       printf "%d %d %d %d\n", busy, $6, $7+$8, total}' /proc/stat
}
# Top three processes by CPU ticks since the previous call. One awk pass
# over every /proc/PID/stat (cat'd, so a process that exits between the
# glob and the open is a skipped file, not a failed pass); the previous
# ticks live in $PREV_PIDS. The comm field is parenthesised and may hold
# spaces, so the record is split on the LAST ')' rather than on whitespace.
read_top() {
  cat /proc/[0-9]*/stat 2>/dev/null | awk -v prev="$PREV_PIDS" '
    BEGIN { while ((getline line < prev) > 0) { split(line, f, " "); was[f[1]] = f[2] } close(prev) }
    {
      i = index($0, "("); j = 0
      for (k = length($0); k > 0; k--) if (substr($0, k, 1) == ")") { j = k; break }
      pid = substr($0, 1, i - 2); comm = substr($0, i + 1, j - i - 1)
      split(substr($0, j + 2), r, " ")           # r[12]=utime r[13]=stime
      ticks = r[12] + r[13]
      now[pid] = ticks; name[pid] = comm
      if (pid in was) d[pid] = ticks - was[pid]
    }
    END {
      for (pid in now) print pid, now[pid] > prev
      n = asorti(d, order, "@val_num_desc")
      out = ""
      for (k = 1; k <= n && k <= 3; k++) { pid = order[k]; if (d[pid] <= 0) break
        out = out (out == "" ? "" : " ") name[pid] ":" d[pid] }
      print (out == "" ? "-" : out)
    }'
}
echo "# $(hostname) $(date -u +%FT%TZ) columns: time d_allocstall d_majfault d_swpin d_swpout d_pgscan_direct d_compact_stall | MiB | gpu | cpu busy% iowait% irq% top3(comm:%)"
read -r p_stall p_maj p_in p_out p_scan p_comp <<< "$(read_vm)"
read -r p_busy p_iow p_irq p_total <<< "$(read_cpu)"
read_top >/dev/null
while true; do
  sleep 1
  read -r stall maj in out scan comp <<< "$(read_vm)"
  read -r busy iow irq total <<< "$(read_cpu)"
  dt=$((total - p_total)); [ "$dt" -gt 0 ] || dt=1
  printf "%s %d %d %d %d %d %d | %s | %s | cpu %d %d %d %s\n" "$(date -u +%T)" \
    $((stall - p_stall)) $((maj - p_maj)) $((in - p_in)) $((out - p_out)) \
    $((scan - p_scan)) $((comp - p_comp)) "$(read_mem)" "$(read_gpu)" \
    $(( (busy - p_busy) * 100 / dt )) $(( (iow - p_iow) * 100 / dt )) \
    $(( (irq - p_irq) * 100 / dt )) "$(read_top)"
  p_stall=$stall; p_maj=$maj; p_in=$in; p_out=$out; p_scan=$scan; p_comp=$comp
  p_busy=$busy; p_iow=$iow; p_irq=$irq; p_total=$total
done
