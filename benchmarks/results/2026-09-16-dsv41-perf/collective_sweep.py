#!/usr/bin/env python3
"""Run from the experiment worktree on the head; all test ranks are time bounded."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import site_env

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--threads", default="256,512,1024,256")
parser.add_argument("--bytes", default="10240,61440,153600,307200")
parser.add_argument("--global-stage", action="store_true")
args = parser.parse_args()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
cfg = site_env.resolve_config("deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.json")
nodes, user = cfg["nodes"], cfg["ssh_user"]
stage = "/tmp/dsv41-collective-20260916"
binary = ROOT / "build-ci/bus_check"
ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
for host in nodes[1:]:
    subprocess.run([*ssh, f"{user}@{host}", shlex.join(["mkdir", "-p", stage])], check=True)
    subprocess.run(["scp", "-q", str(binary), f"{user}@{host}:{stage}/bus_check"], check=True)
reports = []
for trial, width in enumerate(args.threads.split(",")):
    for size in map(int, args.bytes.split(",")):
        case = out / f"r{trial}_t{width}_b{size}"
        case.mkdir()
        procs, files = [], []
        try:
            for rank, host in enumerate(nodes):
                env = {**cfg["node_env"][rank], "DGPP_BUS_GRAPH_WIDE_THREADS": width,
                       "DGPP_WARM_SPINS": "0"}
                if args.global_stage:
                    env["DGPP_BUS_GRAPH_GLOBAL_STAGE"] = "1"
                command = ["timeout", "120", "env", *[f"{k}={v}" for k, v in env.items()],
                           str(binary) if rank == 0 else f"{stage}/bus_check", "allreduce",
                           "--world", str(len(nodes)), "--rank", str(rank),
                           "--port", "29985", "--iters", "80", "--graph-gens", "32",
                           "--lat-bytes", str(size), "--hold-ms", "0"]
                if rank:
                    command += ["--peer", nodes[0]]
                    command = [*ssh, f"{user}@{host}", shlex.join(command)]
                log = (case / f"r{rank}.log").open("w")
                files.append(log)
                procs.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
                if rank == 0:
                    time.sleep(0.5)
            codes = [p.wait(timeout=140) for p in procs]
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for f in files:
                f.close()
        if any(codes):
            raise RuntimeError(f"{case}: rank exits {codes}")
        samples = []
        for rank in range(len(nodes)):
            text = (case / f"r{rank}.log").read_text()
            match = re.search(r"GRAPH-PROBE.*per-collective min=([\d.]+)us p50=([\d.]+)us", text)
            if not match or "mismatches" in text:
                raise RuntimeError(f"{case}: missing or failed rank {rank} evidence")
            samples.append({"rank": rank, "min_us": float(match[1]), "p50_us": float(match[2])})
        report = {"trial": trial, "threads": int(width), "bytes": size,
                  "global_stage": args.global_stage, "ranks": samples}
        reports.append(report)
        (out / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(report), flush=True)
