#!/usr/bin/env python3
"""Cross-rank step correlation for fabric decode logs (r0.log .. rN.log).

The per-step and per-window lines this reads are DEBUG since 2026-09-06
(the throughput line replaced them at INFO): record the run with
DGPP_LOG_LEVEL=debug.

    fabric_xrank.py LOGDIR [--summary] [--show N] [--thresh-ms 3] [--skip 100]

Ranks are aligned by the bus generation counter (identical on every rank),
never by wall clock (the lab boxes disagree by hours). For each rank:

  steps     the per-step wall time distribution (median/p90/p99/max, >=48 ms)
  gaps      the untimed host seam between steps, reconstructed from the log
            timestamps: ts([gen] of step k+1) - ts(phases of step k) minus
            step k+1's timed phases. The 2026-09-02 hunt lived here: ~10 ms
            gaps on the peers, in lockstep, invisible to every bus stamp.
  gen0      windows whose slowest generation was the window's FIRST
            collective, split into handshake-dominated (every peer late =>
            THIS rank arrived early / the peers were held up) and
            skew-dominated (one peer late).

--show N prints N stall windows with every rank's view (window max, phases,
the previous step's two pick collectives).

--laggard prints a per-second table from rank 0's view: median step ms, the
rank that arrived LAST at most of that second's generations (the
"laggard", when one rank takes >= 2/3 of them) with its average lag, and —
when LOGDIR holds probe_rN.log + clock_offsets.txt — that laggard node's
probe row (GPU clock/power/throttle, CPU busy/iowait, top processes). The
rotating ~30% slow rank of 2026-09-02 lives here: this is the join between
"who was slow" and "what else was that box doing".
"""
import glob
import os
import re
import sys
from collections import defaultdict

GRAPH_GENS = 90

re_tl = re.compile(
    r"graph window timeline: rank (\d+) gens (\d+) avg us: total ([\d.]+) = copy ([\d.]+) \+ handshake ([\d.]+) \+ skew ([\d.]+) \+ fold ([\d.]+); engine post ([\d.]+); max handshake ([\d.]+) skew ([\d.]+) total ([\d.]+) \(gen (\d+)\)")
re_peers = re.compile(r"^\S+ (\S+) .*graph window peers: rank (\d+) .*?:(.*)$")
re_lag = re.compile(r"rank(\d+) lag (-?\d+)us last (\d+)x")
re_ph = re.compile(
    r"^\S+ (\S+) .*rank (\d+) step phases ms: launch ([\d.]+) sync ([\d.]+) finish ([\d.]+) collect ([\d.]+) pick ([\d.]+); launch->gpu_start (-?[\d.]+)(.*)$")
re_gapinfo = re.compile(r"gap ([\d.]+) = log ([\d.]+) \+ top ([\d.]+); sched run ([\d.]+) wait ([\d.]+) majflt (\d+) minflt (\d+) nvcsw (\d+) nivcsw (\d+) cpu (\d+)->(\d+)")
re_pick = re.compile(
    r"^\S+ (\S+) .*allreduce: rank (\d+) seq (\d+) elems (\d+) done status=(\d+) total ([\d.]+)us = queue ([\d.]+) \+ claim ([\d.]+) \+ launch_call ([\d.]+) \+ launch->start ([\d.]+) \+ copy ([\d.]+) \+ handshake ([\d.]+) \+ skew ([\d.]+) \+ fold ([\d.]+) \+ done->notice ([\d.]+); gate\(waits=(\d+) spins=(\d+)\); peer lags us:(.*)$")
re_step = re.compile(r"^\S+ (\S+) .*rank (\d+) step (\d+)ms \(graph replay")
re_gen = re.compile(r"^\S+ (\S+) .*\[gen\] rank (\d+) step (\d+):")


def ts_ms(hms):
    h, m, s = hms.split(":")
    return (int(h) * 3600 + int(m) * 60 + float(s)) * 1000.0


def parse(path):
    rec = {"tl": [], "peers": [], "ph": [], "pick": {}, "step": [], "gen": []}
    with open(path, errors="replace") as f:
        for line in f:
            m = re_tl.search(line)
            if m:
                g = m.groups()
                rec["tl"].append({"max_hs": float(g[8]), "max_skew": float(g[9]),
                                  "max_total": float(g[10]), "max_gen": int(g[11])})
                continue
            m = re_peers.search(line)
            if m:
                rec["peers"].append({int(r): (int(l), int(n)) for r, l, n in re_lag.findall(m.group(3))})
                rec.setdefault("peers_ts", []).append(ts_ms(m.group(1)))
                continue
            m = re_ph.search(line)
            if m:
                g = m.groups()
                ph = {"ts": ts_ms(g[0]), "launch": float(g[2]), "sync": float(g[3]),
                      "finish": float(g[4]), "collect": float(g[5]), "pick": float(g[6]),
                      "l2g": float(g[7]), "gapinfo": None}
                gi = re_gapinfo.search(g[8])
                if gi:
                    ph["gapinfo"] = gi.groups()
                rec["ph"].append(ph)
                continue
            m = re_pick.search(line)
            if m:
                g = m.groups()
                rec["pick"][int(g[2])] = {
                    "total": float(g[5]), "queue": float(g[6]), "claim": float(g[7]),
                    "launch_call": float(g[8]), "hs": float(g[11]), "skew": float(g[12]),
                    "lags": g[17].strip()}
                continue
            m = re_step.search(line)
            if m:
                rec["step"].append(int(m.group(3)))
                continue
            m = re_gen.search(line)
            if m:
                rec["gen"].append(ts_ms(m.group(1)))
    return rec


def pct(sorted_vals, f):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(f * len(sorted_vals)))]


def host_gaps(rec):
    """gap before step i+2 (index i): reconstructed from log timestamps."""
    ph, gen = rec["ph"], rec["gen"]
    n = min(len(ph), len(gen)) - 1
    return [gen[i + 1] - ph[i]["ts"] - (ph[i + 1]["launch"] + ph[i + 1]["sync"] +
                                         ph[i + 1]["finish"] + ph[i + 1]["collect"])
            for i in range(n)]


def load_probes(logdir):
    """probe_rN.log rows keyed by the HEAD's second (offset-corrected)."""
    offsets = {}
    try:
        with open(os.path.join(logdir, "clock_offsets.txt")) as f:
            for line in f:
                r, off = line.split()
                offsets[int(r[1:])] = None if off == "?" else float(off)
    except OSError:
        pass
    probes = {}
    for path in glob.glob(os.path.join(logdir, "probe_r[0-9].log")):
        r = int(os.path.basename(path)[7])
        off = offsets.get(r, 0.0 if r == 0 else None)
        if off is None:
            continue
        rows = {}
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith("#") or len(line) < 9:
                    continue
                sec = int(round(ts_ms(line[:8]) / 1000.0 - off))
                rows[sec] = line[9:].rstrip()
        probes[r] = rows
    return probes


def probe_brief(row):
    """The witness columns only: gpu clock/power/throttle + cpu group."""
    parts = row.split(" | ")
    gpu = parts[2] if len(parts) > 2 else ""
    cpu = parts[3] if len(parts) > 3 else ""
    gpu = re.sub(r"throttle=0x0+([0-9a-f]*)", lambda m: "throttle=0x" + (m.group(1) or "0"), gpu)
    return f"{gpu} | {cpu}"


def laggard_table(ranks, skip, logdir):
    r0 = ranks[0]
    peers, pts, steps = r0["peers"], r0.get("peers_ts", []), r0["step"]
    n = min(len(peers), len(pts), len(steps))
    probes = load_probes(logdir)
    by_sec = defaultdict(list)
    for w in range(skip, n):
        by_sec[int(pts[w] // 1000)].append(w)
    print(f"  laggard per second (rank 0's clock; probes {sorted(probes) or 'none'}):")
    print("    time     win  step_med  laggard  share  lag_us  probe(laggard node)")
    for sec in sorted(by_sec):
        ws = by_sec[sec]
        med = sorted(steps[w] for w in ws)[len(ws) // 2]
        last = defaultdict(int)
        lag = defaultdict(list)
        for w in ws:
            for r, (l, c) in peers[w].items():
                last[r] += c
                lag[r].append(l)
        total = sum(last.values()) or 1
        r_max = max(last, key=last.get) if last else None
        share = last[r_max] / total if r_max is not None else 0
        who = f"r{r_max}" if r_max is not None and share >= 2 / 3 else "-"
        lag_us = sum(lag[r_max]) / len(lag[r_max]) if r_max is not None and lag[r_max] else 0
        hms = f"{sec // 3600 % 24:02d}:{sec // 60 % 60:02d}:{sec % 60:02d}"
        row = probes.get(r_max, {}).get(sec) if who != "-" else None
        print(f"    {hms} {len(ws):4d} {med:8.0f}  {who:7} {share:5.2f} {lag_us:7.0f}  "
              f"{probe_brief(row) if row else ''}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    logdir, thresh, skip, show, summary, laggard = args[0], 3000.0, 100, 0, False, False
    i = 1
    while i < len(args):
        if args[i] == "--thresh-ms":
            thresh = float(args[i + 1]) * 1000; i += 2
        elif args[i] == "--skip":
            skip = int(args[i + 1]); i += 2
        elif args[i] == "--show":
            show = int(args[i + 1]); i += 2
        elif args[i] == "--summary":
            summary = True; i += 1
        elif args[i] == "--laggard":
            laggard = True; i += 1
        else:
            i += 1
    ranks = {int(os.path.basename(p)[1]): parse(p)
             for p in sorted(glob.glob(os.path.join(logdir, "r[0-9].log")))}
    if not ranks:
        print("no r[0-9].log in", logdir)
        sys.exit(1)
    n_win = min(len(v["tl"]) for v in ranks.values())
    print(f"xrank: ranks {sorted(ranks)} windows {n_win} (skip {skip})")

    for r in sorted(ranks):
        rec = ranks[r]
        steps = sorted(rec["step"][skip:])
        gaps = host_gaps(rec)
        gs = sorted(gaps[skip:])
        big = sum(1 for g in gaps[skip:] if g > 3.0)
        print(f"  r{r} steps n={len(steps)} median {pct(steps, .5):.0f} p90 {pct(steps, .9):.0f} "
              f"p99 {pct(steps, .99):.0f} max {steps[-1] if steps else 0:.0f} "
              f">=48ms {sum(1 for s in steps if s >= 48)} | host gap p50 {pct(gs, .5):.2f} "
              f"p99 {pct(gs, .99):.2f} max {gs[-1] if gs else 0:.2f} >3ms {big}")
        gi = [p["gapinfo"] for p in rec["ph"][skip:] if p["gapinfo"] and float(p["gapinfo"][0]) > 3.0]
        if gi:
            run = sum(float(g[3]) for g in gi) / len(gi)
            wait = sum(float(g[4]) for g in gi) / len(gi)
            maj = sum(int(g[5]) for g in gi)
            logp = sum(float(g[1]) for g in gi) / len(gi)
            top = sum(float(g[2]) for g in gi) / len(gi)
            moved = sum(1 for g in gi if g[9] != g[10])
            print(f"      big gaps (app-measured) {len(gi)}: avg log {logp:.2f} top {top:.2f} | "
                  f"sched run {run:.2f} wait {wait:.2f} ms | majflt total {maj} | cpu moved {moved}")

    # gen-0 stall classification, per rank.
    # Window w's first generation: the picks are pairs (s, s+1); the first
    # pair is the prefill pick, so window 0 opens at pair0 + 2 and every
    # step consumes GRAPH_GENS + 2 generations. Deriving from the modulus
    # (not from counting pairs) survives a missing pick line.
    pick_seqs = sorted(ranks[0]["pick"])
    pairs = [s for s in pick_seqs if s + 1 in ranks[0]["pick"] and (s - 1) not in ranks[0]["pick"]]
    gens_per_step = GRAPH_GENS + 2
    base = pairs[0] + 2 if pairs else 1

    def win_first(w):
        return base + w * gens_per_step

    def pair_before(w):
        return base + w * gens_per_step - 2

    gen0 = []
    for w in range(skip, n_win):
        f = win_first(w)
        views = {r: ranks[r]["tl"][w] for r in ranks}
        stalled = [r for r, t in views.items() if t["max_total"] >= thresh]
        if stalled:
            gen0.append((w, f, stalled, views))
    by_kind = defaultdict(int)
    for w, f, stalled, views in gen0:
        for r in stalled:
            t = views[r]
            where = "gen0" if t["max_gen"] == f else "mid"
            kind = "handshake" if t["max_hs"] > t["max_skew"] else "skew"
            by_kind[(r, where, kind)] += 1
    print(f"  stall windows (any rank max total >= {thresh/1000:.0f} ms): {len(gen0)}")
    for (r, where, kind), c in sorted(by_kind.items()):
        print(f"    r{r} {where:4} {kind:9} {c}")

    if laggard:
        laggard_table(ranks, skip, logdir)
    if summary and show == 0:
        return
    for w, f, stalled, views in gen0[:show]:
        print(f"\n=== window {w} gen0 {f} (step {w + 1}) stalled at ranks {stalled} ===")
        for r in sorted(ranks):
            t = views[r]
            lags = ranks[r]["peers"][w] if w < len(ranks[r]["peers"]) else {}
            lag_txt = " ".join(f"r{p}:{l}/{c}x" for p, (l, c) in sorted(lags.items()))
            print(f" r{r} win max {t['max_total']/1000:6.2f}ms @gen{t['max_gen']:<7} "
                  f"({'GEN0' if t['max_gen'] == f else 'mid '}) hs {t['max_hs']/1000:.2f} skew {t['max_skew']/1000:.2f} lags[{lag_txt}]")
            if w < len(ranks[r]["ph"]):
                ph = ranks[r]["ph"][w]
                print(f"    phases launch {ph['launch']:.2f} sync {ph['sync']:.2f} finish {ph['finish']:.2f} "
                      f"collect {ph['collect']:.2f} pick {ph['pick']:.2f}"
                      + (f"  gapinfo {' '.join(ph['gapinfo'])}" if ph['gapinfo'] else ""))
            if True:
                for q in (pair_before(w), pair_before(w) + 1):
                    pk = ranks[r]["pick"].get(q)
                    if pk:
                        print(f"    prev-pick seq {q}: total {pk['total']:7.1f} queue {pk['queue']:.0f} "
                              f"launch_call {pk['launch_call']:.0f} hs {pk['hs']:.0f} skew {pk['skew']:.0f} lags {pk['lags']}")


if __name__ == "__main__":
    main()
