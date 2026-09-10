#!/usr/bin/env python3
"""Steady-state decode step time from a fabric run's rank-0 log.

    fabric_step_times.py RUN_DIR... [--skip N]

Reads r0.log's '[gen] rank 0 step S:' lines (one per replay), takes the
timestamp deltas after the first --skip steps (default 20: the graph's
warm-up and the prefill's tail), and prints mean / p50 / p99 ms per step.
The MTP summary line, when present, is echoed for the tokens-per-step
denominator."""
import re, sys, statistics
from datetime import datetime

args = [a for a in sys.argv[1:] if not a.startswith("--")]
skip = 20
for i, a in enumerate(sys.argv[1:]):
    if a == "--skip":
        skip = int(sys.argv[i + 2])
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) INFO  \[gen\] rank 0 step (\d+):")
for run in args:
    times = []
    summary = None
    with open(f"{run}/r0.log", errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if m:
                times.append((int(m.group(2)), datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")))
            elif "speculative summary" in line:
                summary = line.rstrip().split("INFO  ")[-1]
    times.sort()
    deltas = [(b[1] - a[1]).total_seconds() * 1000 for a, b in zip(times, times[1:]) if a[0] >= skip]
    if not deltas:
        print(f"{run}: no steps"); continue
    d = sorted(deltas)
    print(f"{run}: {len(deltas)} steps  mean {statistics.mean(deltas):.2f}  p50 {d[len(d)//2]:.2f}  "
          f"p99 {d[int(0.99*(len(d)-1))]:.2f} ms/step")
    if summary:
        print("   " + summary[:220])
