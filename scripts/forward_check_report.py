"""Check cross-rank digests and merge tensor-parallel argmax results."""
import argparse
from pathlib import Path
import re


def check_forward(directory, world):
    digests, ids, logits = [], [], []
    for r in range(world):
        txt = (Path(directory) / f"w{world}_r{r}.log").read_text()
        digests.append(re.findall(r"digest ([0-9a-f]{16})", txt))
        m_ids = re.search(r"argmax ids: (.*)", txt)
        m_lg = re.search(r"argmax logits: (.*)", txt)
        if not m_ids or not m_lg:
            print("rank", r, "has no argmax lines")
            return 1
        ids.append([int(v) for v in m_ids.group(1).split(",")])
        logits.append([float(v) for v in m_lg.group(1).split(",")])
    same = all(d == digests[0] for d in digests) and len(digests[0]) > 0
    print("cross-rank digests identical:", same, f"({len(digests[0])} digests)")
    merged = []
    for t in range(len(ids[0])):
        best = max(range(world), key=lambda r: (logits[r][t], -ids[r][t]))
        merged.append(ids[best][t])
    print("merged argmax ids:", ",".join(str(v) for v in merged))
    return 0 if same else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("world", type=int)
    args = parser.parse_args()
    return check_forward(args.directory, args.world)


if __name__ == "__main__":
    raise SystemExit(main())
