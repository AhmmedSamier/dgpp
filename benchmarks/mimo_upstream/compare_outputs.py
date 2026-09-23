"""Group exact content/reasoning variants across timing and arrival-order probes."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous evidence"
    groups = {}
    for path in args.files:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data["records"]
        for row in rows:
            request = row["request"]
            key = json.dumps(request, sort_keys=True)
            group = groups.setdefault(key, {"request": request, "variants": {}})
            text = json.dumps([row["reasoning"], row["content"]], ensure_ascii=False)
            digest = hashlib.sha256(text.encode()).hexdigest()
            variant = group["variants"].setdefault(
                digest,
                {
                    "reasoning": row["reasoning"],
                    "content": row["content"],
                    "occurrences": [],
                },
            )
            variant["occurrences"].append({"file": str(path), "name": row["name"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(list(groups.values()), indent=2))
    for group in groups.values():
        print(
            group["request"]["messages"][-1]["content"][:70],
            "variants:",
            len(group["variants"]),
            "requests:",
            sum(len(v["occurrences"]) for v in group["variants"].values()),
        )


if __name__ == "__main__":
    main()
