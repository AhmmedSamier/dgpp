#!/usr/bin/env python3
"""Per-request decode pace from a glm_serve rank log.

Reads the scheduler's per-token lines (DEBUG since 2026-09-06: run the
service with DGPP_LOG_LEVEL=debug to record them, and the bus's graph
timeline lines with them)
  <ts> INFO  sched: request 'ID' admitted to slot ... — first token T
  <ts> INFO  sched: request 'ID' step N: token T
  <ts> INFO  sched: request 'ID' retired (...)
and reports, per request: tokens, the decode pace between the first
decode token and the last (ms/token), the replay count (tokens whose
stamps are within 10 ms of the previous one rode the same replay — a
replay is >= 20 ms apart), and the resulting tokens/replay (MTP's
acceptance in service clothing). ms timestamps, so treat the per-step
figures as +-1 ms. With `--waves`, it also reports each contiguous request
wave's peak-occupancy window: from the admission that reaches the peak
through the first retirement. Graph timeline lines provide the physical
replay clock, so serialized prefill stalls are excluded from that table.
"""
import re
import sys
from datetime import datetime

LINE = re.compile(r"^(\S+ \S+) (?:INFO|DEBUG) +sched: request '([^']+)' (admitted to slot|step (\d+): token|retired)")
STAMP = re.compile(r"^(\S+ \S+)")


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def print_waves(path):
    waves = []
    active = set()
    wave = None
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        is_replay = "graph window timeline:" in line
        if not m and not is_replay:
            continue
        stamp = STAMP.match(line)
        if not stamp:
            continue
        ts = parse_ts(stamp.group(1))
        if wave is not None and is_replay:
            wave["replays"].append(ts)
        if not m:
            continue
        rid = m.group(2)
        kind = m.group(3)
        if kind.startswith("admitted"):
            if not active:
                wave = {"ids": [], "peak": 0, "peak_start": None,
                        "peak_end": None, "replays": [], "steps": []}
            active.add(rid)
            wave["ids"].append(rid)
            if len(active) > wave["peak"]:
                wave["peak"] = len(active)
                wave["peak_start"] = ts
        elif kind.startswith("step"):
            if wave is not None:
                wave["steps"].append((ts, rid, int(m.group(4))))
        else:
            if wave is not None and len(active) == wave["peak"] and \
                    wave["peak_end"] is None:
                wave["peak_end"] = ts
            active.discard(rid)
            if wave is not None and not active:
                waves.append(wave)
                wave = None

    print()
    print(f"{'wave':>4} {'live':>4} {'requests':>8} {'replays':>7} "
          f"{'tokens':>6} {'tok/rep':>7} {'ms/replay':>9} {'tok/s':>7} "
          f"{'p50':>6} {'p99':>6} {'max':>6}")
    for index, w in enumerate(waves, 1):
        start = w["peak_start"]
        end = w["peak_end"]
        reps = [t for t in w["replays"] if start < t <= end]
        ids = set(w["ids"])
        tokens = sum(1 for t, rid, step in w["steps"]
                     if rid in ids and step > 1 and start < t <= end)
        if len(reps) < 2:
            print(f"{index:>4} {w['peak']:>4} {len(ids):>8} "
                  f"{len(reps):>7} {tokens:>6}  (too short)")
            continue
        gaps = sorted((b - a) * 1e3 for a, b in zip(reps, reps[1:]))
        per_replay = tokens / len(reps)
        replay_ms = (reps[-1] - reps[0]) * 1e3 / (len(reps) - 1)
        rate = per_replay * 1e3 / replay_ms
        p50 = gaps[len(gaps) // 2]
        p99 = gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))]
        print(f"{index:>4} {w['peak']:>4} {len(ids):>8} {len(reps):>7} "
              f"{tokens:>6} {per_replay:>7.3f} {replay_ms:>9.3f} "
              f"{rate:>7.2f} {p50:>6.1f} {p99:>6.1f} {gaps[-1]:>6.1f}")


def main(path, show_waves=False):
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
    if show_waves:
        print_waves(path)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or \
            (len(sys.argv) == 3 and sys.argv[2] != "--waves"):
        raise SystemExit("usage: serve_pace.py LOG [--waves]")
    main(sys.argv[1], len(sys.argv) == 3)
