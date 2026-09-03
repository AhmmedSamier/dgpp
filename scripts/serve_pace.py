#!/usr/bin/env python3
"""Per-request decode pace from a glm_serve rank log.

Reads the scheduler's INFO lines
  <ts> INFO  sched: request 'ID' admitted to slot ... — first token T
  <ts> INFO  sched: request 'ID' step N: token T
  <ts> INFO  sched: request 'ID' retired (...)
and reports, per request: tokens, the decode pace between the first
decode token and the last (ms/token), the replay count (tokens whose
stamps are within 10 ms of the previous one rode the same replay — a
replay is >= 20 ms apart), and the resulting tokens/replay (MTP's
acceptance in service clothing). ms timestamps, so treat the per-step
figures as +-1 ms.
"""
import re
import sys
from datetime import datetime

LINE = re.compile(r"^(\S+ \S+) INFO  sched: request '([^']+)' (admitted to slot|step (\d+): token|retired)")


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def main(path):
    reqs = {}
    order = []
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        ts = parse_ts(m.group(1))
        rid = m.group(2)
        kind = m.group(3)
        r = reqs.setdefault(rid, {"admit": None, "stamps": [], "retire": None})
        if rid not in order:
            order.append(rid)
        if kind.startswith("admitted"):
            r["admit"] = ts
        elif kind.startswith("step"):
            r["stamps"].append(ts)
        else:
            r["retire"] = ts
    print(f"{'request':<28} {'tok':>4} {'replays':>7} {'tok/rep':>7} {'decode ms/tok':>13} {'ms/replay':>9} {'p50':>6} {'p99':>6} {'max':>6}")
    for rid in order:
        r = reqs[rid]
        st = r["stamps"]
        n = len(st)
        if n < 2:
            print(f"{rid:<28} {n:>4}  (too short)")
            continue
        # The prefill pick is step 1; decode stamps are steps 2..n.
        dec = st[1:]
        gaps = [b - a for a, b in zip(dec, dec[1:])]
        replays = 1 + sum(1 for g in gaps if g >= 0.010)
        rep_gaps = sorted(g * 1e3 for g in gaps if g >= 0.010)
        span = (dec[-1] - dec[0]) * 1e3
        pace = span / (len(dec) - 1) if len(dec) > 1 else float("nan")
        per_replay = span / (replays - 1) if replays > 1 else float("nan")
        p50 = rep_gaps[len(rep_gaps) // 2] if rep_gaps else float("nan")
        p99 = rep_gaps[min(len(rep_gaps) - 1, int(len(rep_gaps) * 0.99))] if rep_gaps else float("nan")
        mx = rep_gaps[-1] if rep_gaps else float("nan")
        print(f"{rid:<28} {n:>4} {replays:>7} {len(dec) / replays:>7.2f} {pace:>13.2f} {per_replay:>9.2f} {p50:>6.1f} {p99:>6.1f} {mx:>6.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
