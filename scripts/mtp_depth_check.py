#!/usr/bin/env python3
"""Greedy transcript identity across MTP depths.

Sends the same greedy (temperature 0) chat requests to a running dgpp-serve
and writes each completion's text to a file, so two launches — plain, MTP
depth 1, depth 2 — can be diffed: the speculative decode must produce the
plain greedy transcript token for token. Also prints the retire line's
acceptance for the run.

  scripts/mtp_depth_check.py --out /tmp/depth2 [--host URL] [--max-tokens N]
  diff -r /tmp/depth1 /tmp/depth2
"""
import argparse, json, os, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="http://127.0.0.1:18080")
ap.add_argument("--log", default=os.path.expanduser("~/dgpp/log/serve_r0.log"))
ap.add_argument("--model", default="unsloth/GLM-5.3-Flash-FP8")
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--out", required=True, help="directory for the completions")
args = ap.parse_args()

PROMPTS = {
    "prose": "Explain in two paragraphs how a Roman aqueduct kept its gradient.",
    "code": "Write a Python function that parses an ISO-8601 date string and returns the weekday name, with a docstring and three test cases.",
    "json": "Return a JSON object describing three planets with keys name, radius_km, moons (an array of moon names), and a one-sentence note. Only the JSON.",
}


def complete(prompt):
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(args.host + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\n<<<content>>>\n" + (msg.get("content") or "")
    return text, d.get("usage", {}), time.time() - t0


def last_retire():
    try:
        lines = [l for l in open(args.log, errors="replace") if "retired (" in l]
    except OSError:
        return ""
    return lines[-1].strip() if lines else ""


os.makedirs(args.out, exist_ok=True)
for name, prompt in PROMPTS.items():
    text, usage, s = complete(prompt)
    with open(os.path.join(args.out, name + ".txt"), "w") as f:
        f.write(text)
    retire = last_retire()
    m = re.search(r"decode .*?: ([\d.]+) tok/s.*?([\d.]+) tok/pass(.*)$", retire)
    tail = (m.group(1) + " tok/s, " + m.group(2) + " tok/pass" + m.group(3)) if m else retire[-160:]
    print(f"{name}: {usage.get('completion_tokens', '?')} tokens in {s:.1f} s; {tail}")
