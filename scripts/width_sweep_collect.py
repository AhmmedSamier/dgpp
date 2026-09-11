"""Collect per-request results from one sampling-width sweep run."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re

from serve_pace import request_paces


def collect(run, width, mode, tsv):
    log = os.path.join(run, "serve", "serve_r0.log")
    # Per-request fallback counts: the engine logs "slot S closed: N sampled
    # decode steps, F fallbacks" at close; the scheduler logs "request 'id'
    # admitted to slot S" at admission. Slots are reused in order, so the k-th
    # close on a slot belongs to the k-th admission to it.
    admitted = {}   # slot -> [request ids in order]
    closed = {}     # request id -> (sampled, fallbacks)
    with open(log, errors="replace") as source:
        for line in source:
            m = re.search(r"request '([^']+)' admitted to slot (\d+)", line)
            if m:
                admitted.setdefault(int(m.group(2)), []).append(m.group(1))
                continue
            m = re.search(r"slot (\d+) closed: (\d+) sampled decode steps, (\d+) fallbacks", line)
            if m:
                slot = int(m.group(1))
                pending = [r for r in admitted.get(slot, []) if r not in closed]
                if pending:
                    closed[pending[0]] = (int(m.group(2)), int(m.group(3)))
    # The TSV stores pace measurements rounded to two decimal places.
    pace = {
        row["request"]: (row["tokens"], row["replays"],
                         round(row["tokens_per_replay"], 2),
                         round(row["ms_per_token"], 2),
                         round(row["ms_per_replay"], 2))
        for row in request_paces(log)
        if not row["short"] and row["request"].startswith("chatcmpl-")
    }
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "req_*.json"))):
        name = os.path.basename(f)[4:-5]
        setting = name.split("_")[0]
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError) as e:
            print(f"  {name}: unreadable ({e})")
            continue
        if "error" in d:
            print(f"  {name}: ERROR {d['error']}")
            continue
        rid = d["id"]
        ch = d["choices"][0]
        text = ch["message"].get("content") or ""
        sha = hashlib.sha256(((ch["message"].get("reasoning_content") or "") + "\x00" + text).encode()).hexdigest()[:12]
        sampled, fallbacks = closed.get(rid, (-1, -1))
        tok, replays, tpr, mspt, mspr = pace.get(rid, (-1, -1, float("nan"), float("nan"), float("nan")))
        rows.append((width, mode, name, setting, d["usage"]["completion_tokens"], ch["finish_reason"], sampled, fallbacks, replays, tpr, mspr, mspt, sha))
    with open(tsv, "a") as out:
        for r in rows:
            out.write("\t".join(str(x) for x in r) + "\n")
            print("  " + "  ".join(str(x) for x in r))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("width")
    parser.add_argument("mode")
    parser.add_argument("tsv")
    args = parser.parse_args()
    collect(args.run, args.width, args.mode, args.tsv)


if __name__ == "__main__":
    main()
