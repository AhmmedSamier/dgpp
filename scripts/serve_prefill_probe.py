#!/usr/bin/env python3
"""The prefill's server-side cost at prompt lengths, through the endpoint.

Against a running dgpp-serve: for each LEN, REPEAT prompts of about LEN
tokens (natural prose from the prepared GSM8K dataset, calibrated
to the served tokenizer by one warmup request, each prompt behind a unique
nonce so the prefix cache never attaches), max_tokens 1; the cost read from
/v1/metrics deltas (the engine's cumulative prefill_ms and
prompt_tokens_computed — every chunk of a chunked prefill counted) and the
client's time to first token. Prints one line per request and a summary
line per LEN (the median of the repeats).

Usage: serve_prefill_probe.py HOST PORT LEN... [--repeat N] [--model M]
"""
import http.client
import json
import os
import random
import sys
import time
import argparse
from data_paths import data_dir, require_file
from bench_stream import read_completion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("host")
parser.add_argument("port", type=int)
parser.add_argument("lengths", type=int, nargs="*", default=[512, 2048, 8192])
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--model")
parser.add_argument("--data", help="GSM8K JSONL file used as prompt text")
args = parser.parse_args()
if args.repeat < 1 or any(n < 1 for n in args.lengths):
    parser.error("repeat count and prompt lengths must be positive")
HOST, PORT, lens, repeat, model = args.host, args.port, args.lengths, args.repeat, args.model
data_path = require_file(args.data or data_dir() / "gsm8k_test.jsonl")


def get(path):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    conn.request("GET", path)
    body = json.loads(conn.getresponse().read())
    conn.close()
    return body


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def metrics():
    m = get("/v1/metrics")
    return float(find_key(m, "prefill_ms")), int(find_key(m, "prompt_tokens_computed"))


if model is None:
    model = get("/v1/models")["data"][0]["id"]

words = []
with data_path.open() as f:
    for line in f:
        words.extend(json.loads(line)["question"].split())
random.seed(7)


def prompt_of(n_words, nonce):
    start = random.randrange(0, max(1, len(words) - n_words - 1))
    return f"Reference {nonce}. Summarize the following in one sentence.\n\n" + \
        " ".join(words[start:start + n_words])


def ask(prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    conn = http.client.HTTPConnection(HOST, PORT, timeout=1200)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    try:
        result = read_completion(resp)
    finally:
        conn.close()
    # A one-token completion can contain only an invisible special token.
    # Report missing client TTFT rather than counting a role/finish event.
    ttft = (result["first"] - t0) * 1e3 if result["first"] is not None else float("nan")
    return ttft, result["usage"]["prompt_tokens"]


# Calibration: tokens per word on the served tokenizer (also the warmup).
n_cal = 400
_, toks = ask(prompt_of(n_cal, "cal-0"))
per_word = (toks - 30) / n_cal
print(f"[calibrate] {n_cal} words -> {toks} prompt tokens ({per_word:.3f} tok/word)")

for L in lens:
    n_words = max(8, int((L - 30) / per_word))
    samples = []
    for r in range(repeat):
        p0, c0 = metrics()
        ttft, ptoks = ask(prompt_of(n_words, f"n{L}-r{r}-{random.randrange(1 << 30)}"))
        p1, c1 = metrics()
        ms, computed = p1 - p0, c1 - c0
        per = ms / computed if computed else float("nan")
        samples.append((ms, computed, per, ttft))
        print(f"[n{L} r{r}] prompt {ptoks} tok, computed {computed} in {ms:.0f} ms = "
              f"{per:.2f} ms/token; ttft {ttft:.0f} ms")
    samples.sort(key=lambda s: s[2])
    med = samples[len(samples) // 2]
    print(f"[n{L}] median: {med[1]} tokens in {med[0]:.0f} ms = {med[2]:.2f} ms/token "
          f"(min {samples[0][0]:.0f} ms, ttft {med[3]:.0f} ms)")
