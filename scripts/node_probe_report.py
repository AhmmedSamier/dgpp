"""Summarize memory-pressure counters from a node-probe log."""
import argparse
from pathlib import Path
import re


def report(path):
    alloc = major = swap = scan = compact = 0
    masks = set()
    n = 0
    with open(path, errors="replace") as source:
        for line in source:
            m = re.match(r"\d\d:\d\d:\d\d (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) \|", line)
            if not m:
                continue
            a, mj, si, so, sc, cs = (int(x) for x in m.groups())
            alloc += a
            major += mj
            swap += si + so
            scan += sc
            compact += cs
            n += 1
            t = re.search(r"throttle=(\S+)", line)
            if t:
                masks.add(t.group(1))
    print(f"  {Path(path).name}: {n} samples, allocstall {alloc}, pgmajfault {major}, swap {swap}, pgscan_direct {scan}, compact_stall {compact}, throttle masks {sorted(masks)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    return report(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
