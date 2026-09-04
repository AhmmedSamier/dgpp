#!/usr/bin/env python3
"""The width sweep's analysis (scripts/serve_width_sweep.sh -> sweep.tsv).

Per (mode, setting, width): the fallback rate over the setting's requests
and the mean ms per replay; then, per mode, the cost of one fallback fitted
by least squares over every request-row as
    ms_per_replay = base + cost * (fallbacks / replays)
(a fallback lands on the replay it serves, so the per-replay fallback
fraction is the right regressor), with the identical-output check: the
same request must return the same text hash at every width (the pick is
width-independent by construction).

Usage: python3 scripts/width_sweep_report.py RUN_DIR/sweep.tsv
"""
import csv
import math
import sys
from collections import defaultdict


def main(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        for k in ("width", "completion_tokens", "sampled", "fallbacks", "replays"):
            r[k] = int(r[k])
        for k in ("tok_per_replay", "ms_per_replay", "ms_per_token"):
            r[k] = float(r[k])
    modes = sorted({r["mode"] for r in rows})
    settings = ["default", "hot"]
    widths = sorted({r["width"] for r in rows}, reverse=True)

    # 1. Width independence: the same request, the same output at every width.
    print("== output identity across widths (sha of reasoning+content)")
    bad = 0
    for mode in modes:
        for req in sorted({r["request"] for r in rows}):
            shas = {r["width"]: r["sha"] for r in rows if r["mode"] == mode and r["request"] == req}
            if len(set(shas.values())) != 1:
                bad += 1
                print(f"  DIFFERS {mode} {req}: {shas}")
    print(f"  {'all identical' if bad == 0 else str(bad) + ' request(s) differ'}")

    # 2. The table.
    print("\n== fallback rate and pace per (mode, setting, width)")
    print(f"{'mode':<6} {'setting':<8} {'width':>5} {'sampled':>8} {'fallbacks':>9} {'rate':>7} {'ms/replay':>10} {'tok/replay':>10} {'ms/token':>9}")
    for mode in modes:
        for setting in settings:
            for w in widths:
                sel = [r for r in rows if r["mode"] == mode and r["setting"] == setting and r["width"] == w]
                if not sel:
                    continue
                sampled = sum(r["sampled"] for r in sel)
                fb = sum(r["fallbacks"] for r in sel)
                rate = fb / sampled if sampled > 0 else float("nan")
                msr = sum(r["ms_per_replay"] for r in sel) / len(sel)
                tpr = sum(r["tok_per_replay"] for r in sel) / len(sel)
                mst = sum(r["ms_per_token"] for r in sel) / len(sel)
                print(f"{mode:<6} {setting:<8} {w:>5} {sampled:>8} {fb:>9} {rate:>7.1%} {msr:>10.2f} {tpr:>10.2f} {mst:>9.2f}")

    # 3. The fit: ms_per_replay = base + cost * fallback fraction per replay.
    print("\n== fitted cost of one fallback (least squares over every request-row)")
    for mode in modes:
        sel = [r for r in rows if r["mode"] == mode and r["replays"] > 1 and r["sampled"] > 0]
        xs = [r["fallbacks"] / r["replays"] for r in sel]
        ys = [r["ms_per_replay"] for r in sel]
        n = len(xs)
        if n < 3 or max(xs) == min(xs):
            print(f"  {mode}: not enough spread in the fallback fraction to fit")
            continue
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        cost = sxy / sxx
        base = my - cost * mx
        resid = [y - (base + cost * x) for x, y in zip(xs, ys)]
        se = math.sqrt(sum(e * e for e in resid) / max(1, n - 2) / sxx)
        print(f"  {mode}: base {base:.2f} ms/replay, one fallback costs {cost:.2f} ms (+/- {se:.2f}), "
              f"over {n} rows with fallback fractions {min(xs):.2f}..{max(xs):.2f}")
        # The expected cost per width under the measured rates of the hot
        # setting and the recorded teacher-forced rates.
        for setting in settings:
            for w in widths:
                s = [r for r in rows if r["mode"] == mode and r["setting"] == setting and r["width"] == w]
                if not s:
                    continue
                fb = sum(r["fallbacks"] for r in s)
                rep = sum(r["replays"] for r in s)
                print(f"    {setting:<8} width {w:>3}: {fb / rep if rep else float('nan'):.3f} fallbacks per replay -> +{cost * (fb / rep if rep else 0):.2f} ms/replay expected")


if __name__ == "__main__":
    main(sys.argv[1])
