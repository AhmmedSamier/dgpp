"""Report request outcomes and pool metrics for an admission-policy run."""
import argparse
import json
import glob
import os
from pathlib import Path


def report(directory, start, end, name):
    for f in sorted(glob.glob(os.path.join(directory, "req_*.json"))):
        r = json.loads(Path(f).read_text())
        if "error" in r:
            print(f"  {os.path.basename(f)}: ERROR {r['error']}")
            continue
        ch = r["choices"][0]
        print(f"  {os.path.basename(f)}: finish {ch['finish_reason']}, completion_tokens {r['usage']['completion_tokens']}: {ch['message']['content'][:80]!r}")
    m = json.loads((Path(directory) / "metrics.json").read_text())
    sv, sc = m.get("service", {}), m.get("scheduler", {})
    print(f"  metrics: admission {sv.get('admission')}, reservations_grown {sv.get('reservations_grown')}, requests_shed_pool {sv.get('requests_shed_pool')}, pool {sc.get('pool_blocks_in_use')}/{sc.get('pool_blocks_total')} blocks, tokens {sc.get('tokens_generated')}")
    print(f"  batch wall time under {name}: {end - start:.1f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("start", type=float)
    parser.add_argument("end", type=float)
    parser.add_argument("name")
    args = parser.parse_args()
    return report(args.directory, args.start, args.end, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
