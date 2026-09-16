#!/usr/bin/env python3
"""Serial, matched service trials on idle hardware; preserve each run's raw evidence."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("name")
parser.add_argument("--threads", type=int)
parser.add_argument("--objective", action="store_true")
parser.add_argument("--global-stage", action="store_true")
parser.add_argument("--full-tiers", action="store_true")
parser.add_argument("--classes", default="code,prose")
parser.add_argument("--concurrency", default="1,6")
parser.add_argument("--repeat", type=int, default=2)
parser.add_argument("--tokens", type=int, default=192)
parser.add_argument("--prefill", action="store_true")
parser.add_argument("--chunk", type=int)
parser.add_argument("--gates", action="store_true")
parser.add_argument("--knobs", default="")
parser.add_argument("--bin")
parser.add_argument("--plain", action="store_true")
args = parser.parse_args()
root = Path.cwd()
out = root / "build-ci/dsv41-perf" / args.name
out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ)
for key, value in (("DGPP_BUS_GRAPH_WIDE_THREADS", args.threads),
                   ("DGPP_DSV41_PREFILL_CHUNK", args.chunk)):
    if value is None:
        env.pop(key, None)
    else:
        env[key] = str(value)
if args.full_tiers:
    env["DGPP_VERIFY_FULL_TIERS"] = "1"
else:
    env.pop("DGPP_VERIFY_FULL_TIERS", None)
if args.global_stage:
    env["DGPP_BUS_GRAPH_GLOBAL_STAGE"] = "1"
else:
    env.pop("DGPP_BUS_GRAPH_GLOBAL_STAGE", None)
if args.objective:
    env.pop("DGPP_VERIFY_ROUND_UP", None)
else:
    env["DGPP_VERIFY_ROUND_UP"] = "1"
cfg = "deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.json"
if args.plain:
    config = json.loads(Path(cfg).read_text())
    config["engine"].update(mtp=False, mtp_depth=1, mtp_schedule=False)
    cfg = str(out / "plain.json")
    Path(cfg).write_text(json.dumps(config, indent=2) + "\n")
common = ["--config", cfg, "--log-dir", str(out / "world")]
def run(log, command):
    with (out / log).open("w") as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"{args.name}: {log} complete", flush=True)

def long_transcripts():
    data = Path("/home/stephen/workspace/dgpp/data/gsm8k_test.jsonl")
    words = " ".join(json.loads(line)["question"] for line in data.read_text().splitlines()).split()
    results = []
    for length in (1600, 5200):
        prompt = f"Long-input check {length}. Summarize the mathematical topics below in ten paragraphs.\n\n" + " ".join(words[:length])
        body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        conn = http.client.HTTPConnection("127.0.0.1", 18080, timeout=300)
        start = time.monotonic()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            response = conn.getresponse()
            raw = response.read()
            if response.status != 200:
                raise RuntimeError(f"long-input gate: HTTP {response.status}: {raw[:300]}")
            result = json.loads(raw)
        finally:
            conn.close()
        results.append({"words": length, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "wall_s": time.monotonic() - start, "response": result})
    (out / "long_transcripts.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"{args.name}: long-input generation complete", flush=True)

binary = Path(args.bin) if args.bin else root / "build-ci/dgpp-serve"
metadata = {**vars(args), "binary": str(binary),
            "binary_sha256": hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
            "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
(out / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
try:
    command = ["python3", "scripts/dgpp-cluster", "up", *common]
    if args.knobs:
        command += ["--knobs=" + args.knobs]
    if args.bin:
        command += ["--bin", args.bin]
    run("up.log", command)
    rank_log = (out / "world/serve_r0.log").read_text()
    metadata["startup"] = [line for line in rank_log.splitlines() if any(
        token in line for token in ("model constructed", "memory plan total",
                                     "scheduled verify depth on", "graph variants warm"))]
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run("load.log", ["python3", str(Path(__file__).with_name("timed_load.py")), "127.0.0.1", "18080",
                     "--classes", args.classes, "--concurrency", args.concurrency,
                     "--repeat", str(args.repeat), "--max-tokens", str(args.tokens),
                     "--json-out", str(out / "load.json")])
    if args.prefill:
        run("prefill.log", ["python3", "scripts/serve_prefill_probe.py", "127.0.0.1", "18080",
                            "512", "2048", "8192", "--repeat", "3", "--no-think",
                            "--data", "/home/stephen/workspace/dgpp/data/gsm8k_test.jsonl",
                            "--tag", "dsv41-perf", "--json-out", str(out / "prefill.json")])
        long_transcripts()
    if args.gates:
        run("transcripts.log", ["python3", "scripts/serve_greedy_transcript.py", "127.0.0.1", "18080",
                                "--max-tokens", "256", "--out", str(out / "transcripts.json")])
        if not args.plain:
            run("isolation.log", ["python3", "scripts/serve_load.py", "127.0.0.1", "18080",
                                  "--isolation", "6", "--max-tokens", "192", "--concurrency", "1"])
finally:
    run("down.log", ["python3", "scripts/dgpp-cluster", "down", *common])
    if "op streams: identical across 4 ranks" not in (out / "down.log").read_text():
        raise RuntimeError(f"{args.name}: missing cross-rank operation identity")
