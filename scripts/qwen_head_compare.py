#!/usr/bin/env python3
"""Compare complete qwen_head_check runs, including exact repeat controls.

The vocabulary shards are joined before NLL and top-1 comparisons. Every
declared corpus/shape/repeat must complete on every rank. The gate uses the
repository's 0.02 nat mean-NLL budget and 1% rate of >1 nat token changes,
plus unchanged hidden states and no top-1 flip beyond two BF16 ulps.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics

from fabric_logs import rank_logs


def require(ok, message):
    if not ok:
        raise ValueError(message)


def case_key(record):
    return tuple(record[k] for k in ("repeat", "corpus", "lane", "width"))


def read_run(directory):
    headers, rows, cases = {}, {}, {}
    paths = rank_logs(directory)
    require(paths, f"{directory}: no rank logs")
    for rank, path in paths.items():
        scores, completed, begins, ends = {}, {}, [], []
        for line in Path(path).read_text().splitlines():
            match = re.search(r"\[head_(begin|score|case|end)\] (.*)", line)
            if not match:
                continue
            kind, payload = match.groups()
            record = json.loads(payload)
            require(record["rank"] == rank, "rank mismatch")
            if kind == "begin":
                begins.append(record)
            elif kind == "end":
                ends.append(record)
            elif kind == "case":
                key = case_key(record)
                require(key not in completed, "duplicate case completion")
                completed[key] = record
            else:
                key = (*case_key(record), record["index"])
                require(key not in scores, "duplicate score")
                require(math.isfinite(record["lse"]) and
                        (record["target_logit"] is None or math.isfinite(record["target_logit"])) and
                        all(math.isfinite(p[1]) for p in record["top"]), "nonfinite score")
                scores[key] = record
        require(len(begins) == len(ends) == 1, f"rank {rank}: missing/duplicate begin or end")
        h = begins[0]
        require(h["repeats"] >= 2, "at least two unchanged-mode repeats required")
        require(h["capacity"] in (8, 16), "unsupported capacity")
        require(ends[0]["count"] == len(scores) > 0, "incomplete score count")
        names = {key[1] for key in completed}
        require(len(names) == h["corpora"] > 0, "missing corpus")
        widths = [("verify", w) for w in (4, 6, 8, 12, 16) if w <= h["capacity"]]
        widths += [("prefill", w) for w in (1, 4, 5, 8, 9, 16, 17)]
        expected = {(rep, name, lane, w) for rep in range(h["repeats"])
                    for name in names for lane, w in widths}
        require(set(completed) == expected, "missing or unexpected case")
        expected_rows = set()
        for key, c in completed.items():
            _, _, lane, width = key
            if lane == "prefill":
                count = h["prefill_trials"] * width
            else:
                requests, per_request = {4: (2, 2), 6: (3, 2), 8: (4, 2),
                                         12: (4, 3), 16: (4, 4)}[width]
                n = c["tokens"] if width == h["capacity"] else min(c["tokens"], h["boundary_tokens"])
                count = ((n // requests - 17) // per_request) * width
            require(c["count"] == count > 0, "incorrect declared case count")
            expected_rows.update((*key, i) for i in range(count))
        require(set(scores) == expected_rows, "missing or noncontiguous scores")
        for key, record in scores.items():
            repeat, *rest = key
            if repeat:
                baseline = scores[(0, *rest)]
                require({k: v for k, v in record.items() if k != "repeat"} ==
                        {k: v for k, v in baseline.items() if k != "repeat"},
                        f"unchanged-mode repeat differs: rank {rank}, {key}")
        headers[rank], rows[rank], cases[rank] = h, scores, completed
    require(set(headers) == set(range(headers[0]["world"])), "missing rank")
    next_begin = 0
    metadata = {k: v for k, v in headers[0].items() if k not in ("rank", "vocab_begin", "vocab_count")}
    for rank in sorted(headers):
        h = headers[rank]
        require({k: v for k, v in h.items() if k not in ("rank", "vocab_begin", "vocab_count")} == metadata,
                "rank configuration mismatch")
        require(h["vocab_begin"] == next_begin and h["vocab_count"] > 0, "vocabulary gap or overlap")
        next_begin += h["vocab_count"]
        require(rows[rank].keys() == rows[0].keys(), "rank score set mismatch")
        require({k: {f: v for f, v in c.items() if f != "rank"} for k, c in cases[rank].items()} ==
                {k: {f: v for f, v in c.items() if f != "rank"} for k, c in cases[0].items()},
                "rank case mismatch")
    require(next_begin == headers[0]["vocab"], "incomplete vocabulary")
    merged = {}
    for key in rows[0]:
        if key[0] != 0:
            continue
        slices = [records[key] for records in rows.values()]
        require(len({s["target"] for s in slices}) == 1, "rank target mismatch")
        require(len({s["hidden"] for s in slices}) == 1, "rank hidden-state mismatch")
        owners = [s["target_logit"] for s in slices if s["target_logit"] is not None]
        require(len(owners) == 1, "target must have exactly one owner")
        for rank, s in zip(rows, slices):
            h = headers[rank]
            owns = h["vocab_begin"] <= s["target"] < h["vocab_begin"] + h["vocab_count"]
            require(owns == (s["target_logit"] is not None), "wrong target owner")
            require(len(s["top"]) == 2 and len({p[0] for p in s["top"]}) == 2 and
                    all(h["vocab_begin"] <= p[0] < h["vocab_begin"] + h["vocab_count"] for p in s["top"]),
                    "invalid local top-2")
        mx = max(s["lse"] for s in slices)
        lse = mx + math.log(sum(math.exp(s["lse"] - mx) for s in slices))
        merged[key[1:]] = {"nll": lse - owners[0], "target": slices[0]["target"],
                           "top": sorted([p for s in slices for p in s["top"]], key=lambda p: (-p[1], p[0]))[:2],
                           "hidden": slices[0]["hidden"],
                           "hashes": [s["logits"] for s in slices]}
    return metadata, cases[0], merged


def compare(reference, candidate):
    rh, rc, ref = read_run(reference)
    ch, cc, new = read_run(candidate)
    require(rh["mode"] == "gemv" and ch["mode"] == "mma", "expected GEMV reference and MMA candidate")
    require({k: v for k, v in rh.items() if k != "mode"} ==
            {k: v for k, v in ch.items() if k != "mode"}, "run configuration mismatch")
    require(rc == cc and ref.keys() == new.keys(), "run corpus or case mismatch")
    groups = collections.defaultdict(list)
    flips = collections.Counter()
    for key, a in ref.items():
        b = new[key]
        require(a["target"] == b["target"] and a["hidden"] == b["hidden"],
                f"head comparisons require identical targets and hidden states: {key}")
        group = key[:3]
        _, _, width, _ = key
        if width <= 4 or width > rh["capacity"]:
            require(a["hashes"] == b["hashes"], f"dispatch control changed: {key}")
        if a["top"][0][0] != b["top"][0][0]:
            # Both winners must be the other's runner-up; otherwise the
            # retained top-2 cannot establish a near-tie for the actual flip.
            require(a["top"][0][0] == b["top"][1][0] and b["top"][0][0] == a["top"][1][0],
                    f"top-1 flip outside retained top-2: {key}")
            for value in (a, b):
                top, second = value["top"]
                magnitude = max(abs(top[1]), abs(second[1]))
                ulp = 2.0 ** (math.floor(math.log2(magnitude)) - 7) if magnitude else 2.0 ** -133
                require(top[1] - second[1] <= 2 * ulp, f"wide-margin top-1 flip: {key}")
            flips[group] += 1
        groups[group].append((a["nll"], b["nll"]))
    summaries = []
    passed = True
    for group, pairs in sorted(groups.items()):
        deltas = [b - a for a, b in pairs]
        mean = statistics.mean(deltas)
        big_rate = sum(abs(d) > 1 for d in deltas) / len(deltas)
        ok = abs(mean) <= 0.02 and big_rate <= 0.01
        passed &= ok
        summaries.append({"corpus": group[0], "lane": group[1], "width": group[2], "positions": len(pairs),
                          "gemv_mean_nll": statistics.mean(a for a, _ in pairs),
                          "mma_mean_nll": statistics.mean(b for _, b in pairs),
                          "mean_nll_delta": mean,
                          "standard_error": statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0,
                          "max_abs_token_delta": max(map(abs, deltas)), "big_delta_rate": big_rate,
                          "top1_changes": flips[group], "passed": ok})
    return {"passed": passed, "configuration": rh, "repeatability": "bitwise on every rank in both modes",
            "hidden_states": "identical across modes and ranks", "cases": summaries}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--output")
    args = ap.parse_args()
    result = compare(args.reference, args.candidate)
    report = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(report)
    print(report, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
