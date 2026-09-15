#!/usr/bin/env python3
"""Compare matched serve_load JSON runs; fail on C1 transcript/rate regressions.

The default gate allows no median slowdown. A nonzero tolerance must be
explicitly chosen, and is printed with the report. Run repeated A/B trials
to distinguish small changes from measurement noise. This compares client
measurements; it does not establish kernel timings or model quality.
"""
import argparse
import json
import math
import statistics


METRICS = ("wall_tokens_per_s", "legacy_output_span_tokens_per_s")
CLASSES = {"prose", "code", "json", "math", "chat"}


def groups(report):
    out = {}
    for phase in report["phases"]:
        key = (phase["class"], phase["concurrency"])
        if len(phase["requests"]) != key[1]:
            raise ValueError(f"{key}: request count differs from concurrency")
        for request in phase["requests"]:
            if request.get("error") or request["status"] != 200 or not request["finish"]:
                raise ValueError(f"{key}: failed/incomplete request")
        for metric in METRICS:
            rate = phase["metrics"][metric]
            if rate is None or not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"{key}: invalid {metric}")
        out.setdefault(key, []).append(phase)
    return out


def signature(phase, transcript=False):
    fields = ("prompt_sha256", "tokens", "finish") + (("text",) if transcript else ())
    return tuple(tuple(request[field] for field in fields) for request in phase["requests"])


def compare(baseline, candidate, min_repeats=3, tolerance_percent=0.0):
    if min_repeats < 1 or not math.isfinite(tolerance_percent) or not 0 <= tolerance_percent < 100:
        raise ValueError("invalid repeat count or tolerance")
    for field in ("schema_version", "model", "max_tokens", "temperature", "thinking"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"unmatched {field}: {baseline[field]!r} vs {candidate[field]!r}")
    if baseline["schema_version"] != 1 or baseline["temperature"] != 0:
        raise ValueError("C1 transcript gate requires schema 1 greedy runs")
    before, after = groups(baseline), groups(candidate)
    required = {(name, 1) for name in CLASSES}
    if not required <= before.keys() or not required <= after.keys():
        raise ValueError("both reports must include all five C1 classes")
    rows, failures = [], []
    for key in sorted(before.keys() & after.keys()):
        a, b = before[key], after[key]
        if min(len(a), len(b)) < min_repeats:
            raise ValueError(f"{key}: need at least {min_repeats} repeats in both reports")
        # Different output lengths or prompts confound throughput comparisons.
        # C1 also requires byte-identical visible greedy transcripts.
        expected = signature(a[0], transcript=key[1] == 1)
        if any(signature(phase, transcript=key[1] == 1) != expected for phase in a + b):
            raise ValueError(f"{key}: unmatched prompts, lengths, finish reasons or C1 transcripts")
        for metric in METRICS:
            rates_a = [phase["metrics"][metric] for phase in a]
            rates_b = [phase["metrics"][metric] for phase in b]
            median_a, median_b = statistics.median(rates_a), statistics.median(rates_b)
            delta = 100 * (median_b / median_a - 1)
            row = {"class": key[0], "concurrency": key[1], "metric": metric,
                   "baseline": median_a, "candidate": median_b, "change_percent": delta,
                   "baseline_range": [min(rates_a), max(rates_a)],
                   "candidate_range": [min(rates_b), max(rates_b)]}
            rows.append(row)
            if key[1] == 1 and delta < -tolerance_percent:
                failures.append(row)
    return {"tolerance_percent": tolerance_percent, "rows": rows, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--tolerance-percent", type=float, default=0.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    try:
        with open(args.baseline) as stream:
            baseline = json.load(stream)
        with open(args.candidate) as stream:
            candidate = json.load(stream)
        report = compare(baseline, candidate, args.min_repeats, args.tolerance_percent)
    except (KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"uncomparable runs: {error}\n")
    print(f"C1 gate: {args.min_repeats}+ repeats, allowed slowdown {args.tolerance_percent:g}%")
    print("| class | C | metric | baseline median [min, max] | candidate median [min, max] | change |")
    print("|---|---:|---|---:|---:|---:|")
    for row in report["rows"]:
        def spread(name):
            lo, hi = row[name + "_range"]
            return f"{row[name]:.2f} [{lo:.2f}, {hi:.2f}]"
        print(f"| {row['class']} | {row['concurrency']} | {row['metric']} | "
              f"{spread('baseline')} | {spread('candidate')} | {row['change_percent']:+.2f}% |")
    if args.json_out:
        with open(args.json_out, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    if report["failures"]:
        parser.exit(1, f"C1 gate failed: {len(report['failures'])} median rate regression(s)\n")
    print("C1 gate passed: all five transcripts match and both rate scopes meet the threshold")


if __name__ == "__main__":
    main()
