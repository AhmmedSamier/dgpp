#!/usr/bin/env python3
"""Measure C1 engine-call time separately from client-visible streaming time.

Run on an otherwise idle world. Scheduler snapshots bracket each complete
request, outside its measured client interval. The token count excludes the
first token produced by prefill. Engine time includes the step call's CPU,
GPU and communication work; it is not a GPU-only kernel measurement.

    serve_c1_probe.py HOST PORT --repeat 5 --json-out c1-engine.json
"""
import argparse
import json
import math
from pathlib import Path

from serve_load import CLASSES, stream_one
from serve_prefill_probe import get, idle_metrics


def measure(host, port, model, prompt, max_tokens):
    before = idle_metrics(host, port, timeout=5)
    response = {}
    stream_one(host, port, model, prompt, max_tokens, False, response)
    if response.get("error"):
        raise RuntimeError(response["error"])
    after = idle_metrics(host, port, timeout=5,
                         completed_after=before["prompts_prefilled"])
    counters = ("step_ms", "prefill_ms", "decode_steps", "decode_rows",
                "tokens_generated", "prompts_prefilled", "prompt_tokens",
                "prompt_tokens_computed")
    delta = {k: after[k] - before[k] for k in counters}
    usage = response["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if (delta["prompts_prefilled"] != 1 or
            delta["tokens_generated"] != usage["completion_tokens"] or
            delta["prompt_tokens"] != usage["prompt_tokens"] or
            delta["prompt_tokens_computed"] != usage["prompt_tokens"] - cached or
            delta["decode_rows"] != delta["decode_steps"]):
        raise RuntimeError("scheduler counters do not describe exactly this C1 request")
    decode_tokens = delta["tokens_generated"] - 1
    if decode_tokens < 1 or not math.isfinite(delta["step_ms"]) or delta["step_ms"] <= 0:
        raise RuntimeError("request did not produce measurable decode work")
    return {"engine": delta, "decode_tokens": decode_tokens,
            "engine_tokens_per_s": decode_tokens * 1000 / delta["step_ms"],
            "engine_ms_per_token": delta["step_ms"] / decode_tokens,
            "response": response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.max_tokens < 2:
        parser.error("repeat must be positive and max-tokens must be at least two")
    model = get(args.host, args.port, "/v1/models")["data"][0]["id"]
    measure(args.host, args.port, model, CLASSES["prose"][0], 64)
    report = {"model": model, "max_tokens": args.max_tokens,
              "requested_thinking": False, "samples": []}
    for name in ("prose", "code", "json", "math", "chat"):
        for trial in range(args.repeat):
            sample = measure(args.host, args.port, model, CLASSES[name][0], args.max_tokens)
            sample.update({"class": name, "trial": trial})
            report["samples"].append(sample)
            Path(args.json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{name} r{trial}: {sample['engine']['step_ms']:.1f} ms engine time, "
                  f"{sample['decode_tokens']} decode tokens, "
                  f"{sample['engine_tokens_per_s']:.3f} tok/s", flush=True)


if __name__ == "__main__":
    main()
