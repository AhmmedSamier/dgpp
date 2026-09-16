"""Validate matched campaign records and summarize median service timings."""
import json
from pathlib import Path
import statistics
import sys

directory = Path(sys.argv[1])
read = lambda name: json.loads((directory / name).read_text())
baseline, final = read("baseline-load.json"), read("final-load.json")
for field in ("model", "max_tokens", "temperature", "thinking"):
    assert baseline[field] == final[field], field

def groups(report):
    result = {}
    for phase in report["phases"]:
        key = phase["class"], phase["concurrency"]
        result.setdefault(key, []).append(phase)
        assert len(phase["requests"]) == key[1]
        for request in phase["requests"]:
            assert request["status"] == 200 and not request.get("error")
    return result

before, after = groups(baseline), groups(final)
assert before.keys() == after.keys()
summary = {"decode": [], "prefill": [], "long_outputs_identical": True}
for key in sorted(before):
    a, b = before[key], after[key]
    assert len(a) >= 3 and len(b) >= 3
    fields = ["prompt_sha256", "tokens", "finish"]
    if key[1] == 1:
        fields.append("text")
    def signature(phase):
        return [[r[f] for f in fields] for r in phase["requests"]]
    assert all(signature(p) == signature(a[0]) for p in a + b), key
    row = {"class": key[0], "concurrency": key[1]}
    for label, phases in (("baseline", a), ("final", b)):
        row[label] = {
            "engine_tokens_per_s": statistics.median(p["engine"]["tokens_per_s"] for p in phases),
            "wall_tokens_per_s": statistics.median(p["metrics"]["wall_tokens_per_s"] for p in phases),
        }
    row["engine_change_percent"] = 100 * (row["final"]["engine_tokens_per_s"] /
                                           row["baseline"]["engine_tokens_per_s"] - 1)
    summary["decode"].append(row)

bp, fp = read("baseline-prefill.json"), read("final-prefill.json")
assert bp["model"] == fp["model"] and bp["data_sha256"] == fp["data_sha256"]
assert len(bp["samples"]) == len(fp["samples"])
for a, b in zip(bp["samples"], fp["samples"]):
    for field in ("prompt_sha256", "prompt_tokens", "computed_tokens", "cached_tokens",
                  "requested_tokens", "repeat", "usage", "text", "finish"):
        assert a[field] == b[field], (field, a["requested_tokens"])
    assert a["cached_tokens"] == 0
for target in sorted({s["requested_tokens"] for s in bp["samples"]}):
    row = {"target": target}
    for label, report in (("baseline", bp), ("final", fp)):
        samples = [s for s in report["samples"] if s["requested_tokens"] == target]
        assert len(samples) >= 3
        row[label] = {
            "prefill_ms": statistics.median(s["prefill_ms"] for s in samples),
            "ttft_ms": statistics.median(s["ttft_ms"] for s in samples),
            "ms_per_token": statistics.median(s["prefill_ms_per_token"] for s in samples),
            "actual_tokens": [s["prompt_tokens"] for s in samples],
        }
    row["time_reduction_percent"] = 100 * (1 - row["final"]["prefill_ms"] /
                                              row["baseline"]["prefill_ms"])
    row["speedup"] = row["baseline"]["prefill_ms"] / row["final"]["prefill_ms"]
    summary["prefill"].append(row)

bl, fl = read("baseline-long.json"), read("final-long.json")
assert len(bl) == len(fl) == 3
for a, b in zip(bl, fl):
    assert a["prompt_sha256"] == b["prompt_sha256"]
    for field in ("choices", "usage"):
        assert a["response"][field] == b["response"][field], (a["words"], field)
summary["long_prompt_tokens"] = [r["response"]["usage"]["prompt_tokens"] for r in fl]
print(json.dumps(summary, indent=2, allow_nan=False))
