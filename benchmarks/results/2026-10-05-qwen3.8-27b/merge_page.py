#!/usr/bin/env python3
"""Merge this campaign's deployment rows into docs/benchmarks.md.

summarize.py writes overview.md (this campaign's page fragment) beside the
record; the published page keeps the earlier campaigns' rows. This helper
copies every row whose first cell names this campaign's deployments from
overview.md into the matching table of docs/benchmarks.md (after the table's
last existing row), replacing rows already present for the same deployment,
and adds the campaign to the page's measurement note. Tables are matched by
their header line; the long-context and microbenchmark tables carry nothing
for this campaign and are left alone.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAGE = ROOT / "docs/benchmarks.md"
OVERVIEW = HERE / "overview.md"
MODEL_CELL = "Qwen3.8-27B FP8"
NOTE = ("The Qwen3.8-27B FP8 rows are the 2026-10-05 campaign's "
        "([record](../benchmarks/results/2026-10-05-qwen3.8-27b/README.md): the same runner, "
        "evaluator and workload on the tree whose every Qwen3.8-27B template is the DFlash2 drafter "
        "with drawn proposals; their \"Deeper MTP\" cell is the MTP world each template replaced — "
        "depth two on one node, depth three on two and four).")
OLD_NOTE_PREFIX = "The Qwen3.8-27B FP8 rows are the 2026-10-04 campaign's"
# The record's decode-modes table has the page's three value columns under its own
# header for the third ("Other MTP depth" / "Deeper MTP").
MODES_PAGE = "| Model / weights | Nodes | Options | Plain | Template default | Deeper MTP |"
MODES_RECORD = "| Model / weights | Nodes | Options | Plain | Template default | Other MTP depth |"


def tables(lines):
    """Yield (header_index, first_row_index, end_index) for every markdown table."""
    i = 0
    while i < len(lines):
        if lines[i].startswith("| ") and i + 1 < len(lines) and re.fullmatch(r"\|(?:---\|)+", lines[i + 1].strip()):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            yield i, i + 2, j
            i = j
        else:
            i += 1


def main():
    page = PAGE.read_text().split("\n")
    over = OVERVIEW.read_text().split("\n")
    new_rows = {}
    for h, r0, end in tables(over):
        rows = [l for l in over[r0:end] if l.startswith(f"| {MODEL_CELL} |")]
        if not rows:
            continue
        header = over[h]
        if header == MODES_RECORD:
            header = MODES_PAGE
        new_rows[header] = rows
    if not new_rows:
        sys.exit("overview.md has no rows for " + MODEL_CELL)
    merged = 0
    for h, r0, end in reversed(list(tables(page))):
        if page[h] not in new_rows:
            continue
        body = [l for l in page[r0:end] if not l.startswith(f"| {MODEL_CELL} |")]
        page[r0:end] = body + new_rows[page[h]]
        merged += len(new_rows[page[h]])
    if merged == 0:
        sys.exit("no matching tables in docs/benchmarks.md")
    if NOTE not in "\n".join(page):
        for k, line in enumerate(page):
            if line.startswith("Measured "):
                # The earlier Qwen campaign's note gives way to this one's.
                line = re.sub(re.escape(OLD_NOTE_PREFIX) + r".*?\)\.", NOTE, line) if OLD_NOTE_PREFIX in line else line.rstrip() + " " + NOTE
                page[k] = line
                break
    PAGE.write_text("\n".join(page))
    print(f"merged {merged} rows into {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
