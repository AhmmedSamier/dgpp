"""Match rank-zero throughput statistics to load-test phases."""
import argparse
import re
from pathlib import Path


def seconds(stamp):
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def report(directory):
    directory = Path(directory)
    load = (directory / "load.log").read_text().splitlines()
    phases = []
    for line in load:
        m = re.match(r"== concurrency (\d+): (\S+) \.\. (\S+)", line)
        if m:
            phases.append((int(m.group(1)), m.group(2), m.group(3)))
    stats = []
    with open(directory / "world" / "serve_r0.log") as source:
        for line in source:
            m = re.match(r"\S+ (\S+) INFO  stats: rank 0 \| (\S+) s \| (.*)", line)
            if m:
                stats.append((m.group(1), float(m.group(2)), m.group(3)))
    for c, a, b in phases:
        ta, tb = seconds(a), seconds(b)
        print(f"-- concurrency {c} ({a} .. {b})")
        for stamp, period, rest in stats:
            te = seconds(stamp)
            ts = te - period
            if ts >= ta - 0.5 and te <= tb + 0.5:
                print("   " + re.sub(r" \| prefill.*", "", rest)[:150])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    return report(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
