#!/usr/bin/env python3
"""The A/B table from run_ab.sh's legs: engine ms/step (step_ms / decode_steps)
per (mode, concurrency), the median over repeats per class, the mean over
the five classes; plus C1 tok/s and whether the greedy transcripts agree
across legs."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
legs = [p for p in sorted(out.iterdir(), key=lambda p: p.stat().st_mtime) if p.is_dir()]
CL = ["prose", "code", "json", "math", "chat"]


def cells(path):
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    per = {}
    for ph in d["phases"]:
        e = ph.get("engine") or {}
        if not e.get("decode_steps"):
            continue
        per.setdefault((ph["concurrency"], ph["class"]), []).append(
            (e["step_ms"] / e["decode_steps"], ph["metrics"]["completion_tokens"] / ph["metrics"]["wall_s"],
             hashlib.md5("".join(r.get("text", "") for r in ph.get("requests", [])).encode()).hexdigest()[:8]))
    res = {}
    for (c, cls), v in per.items():
        res[(c, cls)] = (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v), sorted({x[2] for x in v}))
    return res


rows = {}
for leg in legs:
    rows[leg.name] = {"mtp": cells(leg / "mtp.json"), "plain": cells(leg / "plain.json")}

print(f"{'metric':<22}" + "".join(f"{l:>14}" for l in rows))
for mode, concs in (("mtp", (1, 2, 4)), ("plain", (1,))):
    for c in concs:
        vals = []
        for leg, r in rows.items():
            xs = [r[mode][(c, cls)][0] for cls in CL if (c, cls) in r[mode]]
            vals.append(statistics.mean(xs) if len(xs) == len(CL) else float("nan"))
        print(f"{mode} C{c} ms/step".ljust(22) + "".join(f"{v:14.2f}" for v in vals))
        if c == 1:
            vals = []
            for leg, r in rows.items():
                xs = [r[mode][(c, cls)][1] for cls in CL if (c, cls) in r[mode]]
                vals.append(statistics.mean(xs) if len(xs) == len(CL) else float("nan"))
            print(f"{mode} C1 tok/s (wall)".ljust(22) + "".join(f"{v:14.1f}" for v in vals))
    for c in concs:
        print(f"  {mode} C{c} per class: " + "; ".join(
            f"{cls} " + "/".join(f"{r[mode][(c, cls)][0]:.1f}" if (c, cls) in r[mode] else "-" for r in rows.values())
            for cls in CL))
# Transcript agreement at C1 (greedy) against the first leg: the md5 of the
# concatenated texts per class and repeat.
first = next(iter(rows))
for mode in ("mtp", "plain"):
    verdicts = []
    for leg, r in rows.items():
        if leg == first:
            continue
        same = [tuple(r[mode].get((1, cls), (0, 0, []))[2]) == tuple(rows[first][mode].get((1, cls), (0, 0, []))[2])
                for cls in CL if (1, cls) in r[mode]]
        verdicts.append(f"{leg} {sum(same)}/{len(same)}" if same else f"{leg} -")
    print(f"{mode} C1 transcripts identical to {first} (classes): " + "; ".join(verdicts))
