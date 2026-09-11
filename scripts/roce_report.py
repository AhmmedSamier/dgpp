"""Compare two RoCE counter snapshots produced by roce_counters.sh."""
import argparse
from pathlib import Path


def load_snapshot(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        result[parts[0], parts[1]] = {
            key: int(value) for key, value in (item.split("=") for item in parts[2:])
        }
    return result


def report(before, after):
    initial, final = load_snapshot(before), load_snapshot(after)
    for key in sorted(final):
        if key not in initial:
            continue
        deltas = {name: value - initial[key].get(name, 0) for name, value in final[key].items()}
        nonzero = {name: value for name, value in deltas.items() if value}
        # port_xmit_data counts four-byte words.
        transmitted = nonzero.pop("xmit_data", 0) * 4 / 1048576.0
        counters = "  ".join(f"{name}={value}" for name, value in nonzero.items())
        print(f"{key[0]} {key[1]}: xmit {transmitted:8.1f} MiB  "
              + (counters or "no retransmits, no CNPs, no discards"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    report(args.before, args.after)


if __name__ == "__main__":
    main()
