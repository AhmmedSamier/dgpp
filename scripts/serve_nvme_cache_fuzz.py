#!/usr/bin/env python3
"""The NVMe cold tier under a random workload with restarts (issue #26): a
handful of document families of random size, random questions, random
prompt_cache_key pins and prefix_cache opt-outs, every (document, question)
pair repeated at random so its greedy transcript is checked against the first
time it was answered — whether that answer came from a cold prefill, a memory
hit, a restore from disk or an adopted slab after a restart. Every
`--restart-every` requests the world is taken down and brought back under the
same configuration (dgpp-cluster), the launcher's op-stream identity line is
checked, and the adopted entries are counted.

Boot the world first (a deployment JSON with the tier on and a small
`engine.prefix_cache_gib`), then:

  scripts/serve_nvme_cache_fuzz.py --config that.json --bin build-ci/dgpp-serve \\
      --port 18080 --requests 60 --restart-every 20 --seed 1 --out report.json

Exit 0 when every repeat matched, no tier op failed and every restart's op
streams agreed; the report names each mismatch.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_nvme_cache_check as chk  # noqa: E402
from serve_client import served_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def restart(config, bin_path, log_lines):
    down = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dgpp-cluster"), "down", "--config", config],
                          capture_output=True, text=True, timeout=600)
    identical = "identical across" in down.stdout or "identical across" in down.stderr
    single = "on 1 node" in down.stdout or "op streams" not in down.stdout + down.stderr
    log_lines.append(f"down: rc {down.returncode}, op streams {'identical' if identical else ('n/a' if single else 'DIFFER')}")
    up = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dgpp-cluster"), "up", "--config", config]
                        + (["--bin", bin_path] if bin_path else []), capture_output=True, text=True, timeout=1500)
    ready = "READY" in up.stdout
    log_lines.append(f"up: rc {up.returncode}, {'ready' if ready else 'NOT READY'}")
    return (identical or single) and ready


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--config", help="deployment JSON, for the restarts")
    ap.add_argument("--bin", default=None, help="the binary dgpp-cluster up stages (default: the release build)")
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--restart-every", type=int, default=0, help="0: no restarts")
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    model = served_model(args.host, args.port)
    docs = ["".join(chk.filler(args.seed * 100 + i, rng.randint(250, 700))) for i in range(args.docs)]
    questions = [f"\nQuestion {i}: write a numbered list describing the first {30 + i} parcels and their cities."
                 for i in range(4)]
    expected = {}  # (doc, question) -> the first transcript
    findings, log_lines, steps = [], [], []
    restarts = 0
    for n in range(args.requests):
        if args.restart_every and n > 0 and n % args.restart_every == 0:
            before = chk.metrics(args.host, args.port)["nvme_cache"]
            ok = restart(args.config, args.bin, log_lines)
            after = chk.metrics(args.host, args.port)["nvme_cache"]
            restarts += 1
            print(f"restart {restarts}: {log_lines[-2]}; {log_lines[-1]}; entries {before.get('entries')} -> "
                  f"{after.get('entries')}, adopted {after.get('adopted')}", flush=True)
            if not ok:
                findings.append({"restart": restarts, "what": "down/up did not complete cleanly"})
        d, q = rng.randrange(args.docs), rng.randrange(len(questions))
        body_extra = {}
        if rng.random() < 0.5:
            body_extra["prompt_cache_key"] = f"fuzz-doc-{d}"
        if rng.random() < 0.1:
            body_extra["prefix_cache"] = False
        prompt = docs[d] + questions[q]
        t0 = time.perf_counter()
        r = chk.ask(args.host, args.port, model, prompt, args.max_tokens, extra=body_extra)
        cached = r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        key = (d, q)
        step = {"n": n, "doc": d, "question": q, "ttft_ms": r["ttft_ms"], "cached": cached,
                "prompt_tokens": r["usage"].get("prompt_tokens"), "extra": body_extra,
                "wall_ms": (time.perf_counter() - t0) * 1000}
        if key in expected:
            step["repeat"] = True
            if r["text"] != expected[key]:
                step["mismatch"] = True
                findings.append({"n": n, "doc": d, "question": q, "what": "transcript differs from the first answer",
                                 "first": expected[key][:200], "now": r["text"][:200]})
        else:
            expected[key] = r["text"]
        steps.append(step)
        print(f"{n:3d} doc {d} q {q} ttft {r['ttft_ms']:7.0f} ms cached {str(cached):>6} "
              f"{'repeat' if step.get('repeat') else 'first '} {'MISMATCH' if step.get('mismatch') else ''} {body_extra}",
              flush=True)
    m = chk.metrics(args.host, args.port)
    nv = m["nvme_cache"]
    tier_ok = nv.get("spill_failed", 0) == 0 and nv.get("restore_failed", 0) == 0 and nv.get("failures", 0) == 0 \
        and nv.get("pages_used", 0) <= nv.get("pages_total", 0)
    if not tier_ok:
        findings.append({"what": "tier failures or capacity breach", "nvme_cache": nv})
    repeats = sum(1 for s in steps if s.get("repeat"))
    print(f"\n{args.requests} requests, {repeats} repeats checked, {restarts} restarts, {len(findings)} finding(s)")
    print("nvme_cache:", {k: nv.get(k) for k in ("entries", "spills", "spilled", "restores", "restored", "spill_failed",
                                                 "restore_failed", "evictions", "pinned", "adopted", "pages_used",
                                                 "pages_total")})
    print("prefix_cache:", {k: m["prefix_cache"].get(k) for k in ("hits", "misses", "evictions", "pinned")})
    for f in findings:
        print("FINDING:", json.dumps(f)[:400])
    print("PASS" if not findings else "FAIL")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "steps": steps, "findings": findings, "restarts": log_lines,
                       "metrics": m, "ok": not findings}, fh, indent=1)
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
