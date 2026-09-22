#!/usr/bin/env python3
"""Capture solo/batched greedy responses and counters for the isolation prompt."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from serve_load import PROMPTS, served_model, stream_one
from serve_prefill_probe import get
from timed_load import TimedLoad

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--json-out", type=Path, required=True)
args = parser.parse_args()
host, port = "127.0.0.1", 18080
model = served_model(host, port)
load = TimedLoad()
load.settle(host, port, [])
warm = {}
stream_one(host, port, model, PROMPTS[0], 64, True, warm)
if warm.get("error"):
    raise RuntimeError(warm["error"])
load.settle(host, port, [warm])
report = {"model": model, "temperature": 0, "max_tokens": 256,
          "thinking": "server_default", "warmup": warm, "phases": []}
for c in (1, 1, 2, 2, 3, 3, 4, 4, 1, 1):
    before = get(host, port, "/v1/metrics")
    load.phase(host, port, model, c, 256, True, 0, PROMPTS, 0,
               f"isolation-c{c}", report["phases"])
    phase = report["phases"][-1]
    phase["metrics_before"] = before
    phase["metrics_after"] = get(host, port, "/v1/metrics")
    answer = phase["requests"][0]
    baseline = report["phases"][0]["requests"][0]
    a, b = baseline["text"], answer["text"]
    first = next((i for i, (left, right) in enumerate(zip(a, b)) if left != right), min(len(a), len(b)))
    phase["probe"] = {"text_sha256": hashlib.sha256(b.encode()).hexdigest(),
                      "identical_to_first_solo": (a, baseline["tokens"]) == (b, answer["tokens"]),
                      "first_difference": None if a == b else first,
                      "solo_excerpt": a[first:first + 120], "batch_excerpt": b[first:first + 120]}
    args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PROBE", c, json.dumps(phase["probe"]), flush=True)
