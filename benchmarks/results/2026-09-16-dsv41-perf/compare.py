#!/usr/bin/env python3
"""Compare archived matched service runs; retain output and timing evidence."""
import argparse
import json
from pathlib import Path
from statistics import geometric_mean, median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("control", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--out", type=Path)
parser.add_argument("--plain", type=Path)
args = parser.parse_args()
a = json.loads((args.control / "load.json").read_text())
b = json.loads((args.candidate / "load.json").read_text())
for key in ("model", "max_tokens", "temperature", "thinking"):
    assert a[key] == b[key], (key, a[key], b[key])
assert len(a["phases"]) == len(b["phases"])
checks = 0
for x, y in zip(a["phases"], b["phases"]):
    assert (x["class"], x["concurrency"]) == (y["class"], y["concurrency"])
    assert len(x["requests"]) == len(y["requests"])
    for i, (u, v) in enumerate(zip(x["requests"], y["requests"])):
        assert u["status"] == v["status"] == 200
        for key in ("text", "usage", "finish"):
            assert u[key] == v[key], (x["class"], x["concurrency"], i, key)
        checks += 1

rows = []
for cls, c in dict.fromkeys((x["class"], x["concurrency"]) for x in a["phases"]):
    row = {"class": cls, "concurrency": c}
    for metric in ("wall", "engine"):
        for name, data in (("control", a), ("candidate", b)):
            phases = [x for x in data["phases"] if x["class"] == cls and x["concurrency"] == c]
            rates = [x["engine"]["tokens_per_s"] if metric == "engine" else
                     x["metrics"]["wall_tokens_per_s"] for x in phases]
            row[f"{metric}_{name}"] = median(rates)
            row[f"{metric}_{name}_samples"] = rates
        row[f"{metric}_gain_pct"] = 100 * (row[f"{metric}_candidate"] / row[f"{metric}_control"] - 1)
    rows.append(row)

summary = {"paired_requests_identical": checks, "rows": rows, "geomean_gain_pct": {}}
for c in sorted({x["concurrency"] for x in rows}):
    summary["geomean_gain_pct"][c] = {
        metric: 100 * (geometric_mean(x[f"{metric}_candidate"] / x[f"{metric}_control"]
                                     for x in rows if x["concurrency"] == c) - 1)
        for metric in ("wall", "engine")}
if args.plain:
    def transcripts(left, right):
        u = json.loads((left / "transcripts.json").read_text())["results"]
        v = json.loads((right / "transcripts.json").read_text())["results"]
        assert u.keys() == v.keys()
        for name in u:
            for key in ("reasoning", "content", "finish", "completion_tokens", "prompt_tokens"):
                assert u[name][key] == v[name][key], (name, key, left, right)
        return len(u)
    summary["transcript_anchors_identical"] = transcripts(args.control, args.candidate)
    summary["mtp_plain_anchors_identical"] = transcripts(args.candidate, args.plain)
    u = json.loads((args.control / "long_transcripts.json").read_text())
    v = json.loads((args.candidate / "long_transcripts.json").read_text())
    assert len(u) == len(v)
    for x, y in zip(u, v):
        assert x["prompt_sha256"] == y["prompt_sha256"]
        for key in ("choices", "usage"):
            assert x["response"][key] == y["response"][key], (x["words"], key)
    summary["long_prompts_identical"] = len(u)
    u = json.loads((args.control / "prefill.json").read_text())["samples"]
    v = json.loads((args.candidate / "prefill.json").read_text())["samples"]
    assert len(u) == len(v)
    for x, y in zip(u, v):
        assert x["cached_tokens"] == y["cached_tokens"] == 0
        for key in ("prompt_sha256", "text", "usage"):
            assert x[key] == y[key], (x["prompt_tokens"], key)
    summary["cold_prefill_samples_identical"] = len(u)

print("| class | C | wall baseline | wall final | change | decode baseline | decode final | change |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['class']} | {r['concurrency']} | {r['wall_control']:.2f} | {r['wall_candidate']:.2f} | "
          f"{r['wall_gain_pct']:+.1f}% | {r['engine_control']:.2f} | {r['engine_candidate']:.2f} | "
          f"{r['engine_gain_pct']:+.1f}% |")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
if args.out:
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
