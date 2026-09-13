#!/usr/bin/env python3
"""MTP acceptance per prompt class through the endpoint (the full GLM-5.3
ritual, 2026-09-12; the same five prompts as fabric_mtp_classes.sh).

Against a running dgpp-serve: one greedy request per class (chat, code,
prose, json, math), max_tokens 300, then each request's line from the
scheduler's log — `retired ...: decode N tok / P passes ...: X tok/pass,
accept p1 Y %` — so the server-side pass time, tokens per pass and draft
acceptance are the engine's own numbers, not a client-side estimate.

Usage: serve_mtp_classes.py HOST PORT --out FILE [--log RANK0_LOG] [--max-tokens N]
"""
import argparse
import json
import re
import sys
import time
import urllib.request

PROMPTS = {
    "chat": "Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs.",
    "code": "Write a Python module that parses an OpenAI-style SSE stream of chat completion chunks into a single message object, with type hints, docstrings, and a small set of unit tests using pytest.",
    "prose": "Write a long, detailed history of the Roman Republic from its founding to the rise of Augustus, one era per paragraph.",
    "json": "Return a JSON array of 25 objects, each with the fields country, capital, population_millions and currency, for 25 different countries. Output only the JSON.",
    "math": "A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward it 30 minutes later. Work out step by step when and where they meet, then generalize the formula and check it with two other examples.",
}
RETIRED = re.compile(r"request '([^']+)' retired \((\w+)\): (\d+) tok in ([\d.]+) s — prefill (\d+) tok \((\d+) cached\) in (\d+) ms; "
                     r"decode (\d+) tok / (\d+) passes in ([\d.]+) s: ([\d.]+) tok/s, ([\d.]+) ms/tok, (\d+) ms/pass, ([\d.]+) tok/pass(?:, accept p1 (\d+) %)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None, help="rank 0's serve log; the retired lines are read from it")
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))["data"][0]["id"]
    results = {}
    for cls, prompt in PROMPTS.items():
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": args.max_tokens, "temperature": 0}).encode()
        req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        wall = time.time() - t0
        usage, choice = r["usage"], r["choices"][0]
        text = (choice["message"].get("reasoning_content") or "") + (choice["message"].get("content") or "")
        results[cls] = {"id": r["id"], "wall_s": round(wall, 3), "completion_tokens": usage["completion_tokens"],
                        "finish": choice["finish_reason"], "head": text[:200]}
    if args.log:
        by_id = {}
        for line in open(args.log, encoding="utf-8", errors="replace"):
            m = RETIRED.search(line)
            if m:
                by_id[m.group(1)] = m
        for cls, res in results.items():
            m = by_id.get(res["id"])
            if m is None:
                res["server"] = None
                continue
            res["server"] = {"prefill_tok": int(m.group(5)), "prefill_ms": int(m.group(7)), "decode_tok": int(m.group(8)),
                             "passes": int(m.group(9)), "tok_per_s": float(m.group(11)), "ms_per_tok": float(m.group(12)),
                             "ms_per_pass": int(m.group(13)), "tok_per_pass": float(m.group(14)),
                             "accept_p1": int(m.group(15)) if m.group(15) is not None else None}
    print(f"{'class':6s} {'tokens':>6s} {'wall ms/tok':>11s} {'ms/pass':>8s} {'tok/pass':>8s} {'accept':>6s}  finish")
    for cls, res in results.items():
        s = res.get("server")
        acc = f"{s['accept_p1']:5d} %" if s and s["accept_p1"] is not None else f"{'T=1':>7s}"
        srv = (f"{s['ms_per_pass']:8d} {s['tok_per_pass']:8.2f} {acc}" if s else f"{'-':>8s} {'-':>8s} {'-':>7s}")
        print(f"{cls:6s} {res['completion_tokens']:6d} {1000 * res['wall_s'] / max(1, res['completion_tokens']):11.1f} {srv}  {res['finish']}")
    json.dump({"model": model, "max_tokens": args.max_tokens, "classes": results}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
