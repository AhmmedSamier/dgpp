#!/usr/bin/env python3
"""Greedy transcripts from a running dgpp-serve world (Q6, 2026-09-09): the
same prompts at temperature 0, non-streamed, written as JSON so two worlds
(the MTP one and the plain T=1 one, or two ranks' configs) can be compared
line by line — the "MTP transcripts identical to plain decode" gate at the
API, and the pace of each request from the usage and the wall clock.
    serve_greedy_transcript.py HOST PORT --out FILE [--max-tokens N] [--model ID]
Prints one line per prompt: label, completion tokens, wall seconds, ms per
token, and the reply's first 60 characters. `--compare REF` reads a
previous --out file and reports whether every reply (reasoning + content)
is identical.
"""
import argparse
from serve_client import served_model
import http.client
import json
import sys
import time

PROMPTS = {
    "chat": "Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs.",
    "code": "Write a Python function that parses an ISO-8601 timestamp without external libraries, with docstring and three doctests.",
    "math": "A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward it 30 minutes later. Work out step by step when and where they meet.",
    "json": "Return a JSON array of 8 objects, each with the fields country, capital and population_millions. Output only the JSON.",
}


def one(host, port, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    })
    conn = http.client.HTTPConnection(host, port, timeout=3600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    wall = time.perf_counter() - t0
    if resp.status != 200:
        raise RuntimeError("HTTP %d: %s" % (resp.status, raw[:300]))
    data = json.loads(raw)
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return {
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "content": msg.get("content") or "",
        "finish": data["choices"][0].get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    model = args.model or served_model(args.host, args.port)
    results = {}
    for label, prompt in PROMPTS.items():
        r = one(args.host, args.port, model, prompt, args.max_tokens)
        results[label] = r
        n = r["completion_tokens"] or 0
        pace = (r["wall_s"] / n * 1000.0) if n else float("nan")
        head = (r["reasoning"] + r["content"]).replace("\n", " ")[:60]
        print("%-5s %4s tokens %7.2f s %7.1f ms/token (wall, prefill included) %s | %s" %
              (label, n, r["wall_s"], pace, r["finish"], head), flush=True)
    with open(args.out, "w") as f:
        json.dump({"model": model, "results": results}, f, indent=1)
    if args.compare:
        # The reply as the tokens spelled it: reasoning + content with the
        # split's whitespace and a leaked "</think>" literal (a frontend
        # that missed the opened block) normalized away — the tokens are
        # what the identity gate is about, and the API returns text.
        import re
        def spelled(r):
            t = (r["reasoning"] or "") + " " + (r["content"] or "")
            return re.sub(r"\s+", " ", t.replace("</think>", " ")).strip()
        ref = json.load(open(args.compare))["results"]
        same = 0
        for label, r in results.items():
            pa, pb = spelled(r), spelled(ref[label])
            if pa == pb and r["completion_tokens"] == ref[label]["completion_tokens"]:
                same += 1
            else:
                k = 0
                while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
                    k += 1
                print("%-5s DIFFERS at character %d of %d/%d (tokens %s vs %s): %r | %r" %
                      (label, k, len(pa), len(pb), r["completion_tokens"], ref[label]["completion_tokens"],
                       pa[max(0, k - 30):k + 30], pb[max(0, k - 30):k + 30]))
        print("identical transcripts: %d of %d" % (same, len(results)))
        return 0 if same == len(results) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
