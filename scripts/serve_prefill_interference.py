#!/usr/bin/env python3
"""Measure a decoding client's longest pause while another prompt prefills.

Run on an otherwise idle server. Compare the same prompts/output limit with
prefill_budget_tokens=0 and an enabled budget. Length is specified in words;
the JSON records the tokenizer's actual prompt count. This measures client
update gaps, not per-token kernel latency. Use --tag to make a cold prefix
and keep that tag identical in matched fresh-server runs.
"""
import argparse
import json
import math
import statistics
import threading
import time

from serve_load import CLASSES, served_model, stream_one


def gap_summary(gaps):
    if not gaps:
        raise ValueError("decode completion did not contain multiple visible updates")
    ordered = sorted(gaps)
    return {"p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
            "p99_ms": ordered[math.ceil(len(ordered) * 0.99) - 1],
            "max_ms": ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--prompt-words", type=int, default=24000)
    parser.add_argument("--decode-tokens", type=int, default=1024)
    parser.add_argument("--tag", default="prefill-interference-1")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.prompt_words < 1 or args.decode_tokens < 64:
        parser.error("prompt-words must be positive; decode-tokens must be at least 64")
    model = served_model(args.host, args.port)
    # Warm the existing decode path before measuring its interference.
    warm = {}
    stream_one(args.host, args.port, model, CLASSES["prose"][0], 64, False, warm)
    if warm.get("error"):
        raise RuntimeError(warm["error"])
    ready, decode = threading.Event(), {}
    thread = threading.Thread(target=stream_one, args=(
        args.host, args.port, model, CLASSES["prose"][0], args.decode_tokens, False, decode,
        0, lambda _: ready.set()))
    thread.start()
    if not ready.wait(120):
        thread.join()
        raise RuntimeError(decode.get("error", "decode produced no visible update within 120 seconds"))
    time.sleep(1)
    if not thread.is_alive():
        raise RuntimeError("decode ended before the long prompt arrived; increase decode-tokens")
    sentence = "The archive records the seasons and rainfall in a small mountain village over many years.".split()
    words = (sentence * ((args.prompt_words + len(sentence) - 1) // len(sentence)))[:args.prompt_words]
    prompt = f"Reference {args.tag}. Summarize the following records briefly.\n\n" + " ".join(words)
    prefill = {}
    stream_one(args.host, args.port, model, prompt, 1, False, prefill)
    thread.join()
    for result in (decode, prefill):
        if result.get("error"):
            raise RuntimeError(result["error"])
    if decode["last"] <= prefill["t0"]:
        raise RuntimeError("requests did not overlap")
    summary = gap_summary(decode["update_gaps_ms"])
    report = {"model": model, "tag": args.tag, "decode": decode, "prefill": prefill,
              "decode_update_gaps": summary}
    with open(args.json_out, "w") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(f"long prompt: {prefill['usage']['prompt_tokens']} tokens; "
          f"wall {(prefill['end'] - prefill['t0']):.2f}s; decode update gaps: {summary}")


if __name__ == "__main__":
    main()
