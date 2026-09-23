"""Summarize saved panels against an explicit control without contacting a server."""

import argparse
import json
import statistics
from benchmark import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, type=__import__("pathlib").Path)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous evidence"
    ref = {
        x["name"]: x
        for x in json.loads(
            (ROOT / "raw" / args.reference / "results.json").read_text()
        )
    }
    summaries = {}
    for tag in args.tags:
        dest = ROOT / "raw" / tag
        rows = json.loads((dest / "results.json").read_text())
        groups = {}
        for row in rows:
            groups.setdefault(row["name"].rsplit("-r", 1)[0], []).append(row)
        accfile = dest / "acceptance.json"
        acceptance = json.loads(accfile.read_text()) if accfile.exists() else []
        summaries[tag] = {
            "groups": {
                k: {
                    "ttft": statistics.median(x["ttft"] for x in v),
                    "decode_tps": statistics.median(
                        (x["usage"]["completion_tokens"] - 1) / x["decode_seconds"]
                        for x in v
                    ),
                    "n": len(v),
                }
                for k, v in groups.items()
            },
            "outputs_differing": [
                x["name"]
                for x in rows
                if (x["content"], x["reasoning"])
                != (ref[x["name"]]["content"], ref[x["name"]]["reasoning"])
            ],
            "acceptance": {
                x["name"]: {
                    "rounds": x["rounds"],
                    "accepted_per_position": x["accepted_per_position"],
                    "accepted_per_all_drafts": (
                        [v / x["rounds"] for v in x["accepted_per_position"]]
                        if x["rounds"]
                        else None
                    ),
                    "attempts": x["attempts_per_position"],
                }
                for x in acceptance
            },
            "requests": len(rows),
            "complete": (dest / "metrics.json").exists(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
