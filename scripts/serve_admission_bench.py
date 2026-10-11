#!/usr/bin/env python3
"""Measure admission fairness and throughput on an otherwise idle server.

Run identical arguments against fresh baseline and candidate processes, in
alternating order. Prompts are deterministic and cache-disabled; ignore_eos
fixes output lengths. This is a scheduling benchmark, not a quality test.
The native prefill monitor must observe the long request before arrivals are
dispatched. SSE gaps measure client updates, not individual token latency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlsplit


SCENARIOS = ("solo_decode", "solo_prefill", "mixed_short", "mixed_long",
             "sustained_idle", "sustained_busy", "cancel")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


class Bench:
    def __init__(self, args):
        self.args = args
        endpoint = urlsplit(args.url)
        if endpoint.scheme != "http" or not endpoint.hostname:
            raise ValueError("use an http:// test endpoint")
        self.host, self.port = endpoint.hostname, endpoint.port or 80
        self.model = self.get("/v1/models")["data"][0]["id"]

    def connection(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.args.timeout)

    def get(self, path):
        with closing(self.connection()) as conn:
            conn.request("GET", path)
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"GET {path}: {response.status} {response.read()!r}")
            return json.load(response)

    def idle(self):
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            metrics = self.get("/metrics")
            if metrics["service"]["engine_failed"]:
                raise RuntimeError("the engine failed")
            if not metrics["scheduler"]["active"] and not metrics["scheduler"]["queued"]:
                return metrics
            time.sleep(.05)
        raise TimeoutError("server did not drain")

    def stream(self, name, prompt, tokens, ready=None, cancel=False, dispatched=None):
        body = json.dumps({"model": self.model, "prompt": prompt,
                           "max_tokens": tokens, "temperature": 0,
                           "ignore_eos": True, "prefix_cache": False,
                           "stream": True, "stream_options": {"include_usage": True}},
                          separators=(",", ":")).encode()
        record = {"name": name, "body_sha256": hashlib.sha256(body).hexdigest(),
                  "requested_tokens": tokens, "start": time.monotonic(),
                  "updates": [], "cancelled": False}
        output = hashlib.sha256()
        done = False
        with closing(self.connection()) as conn:
            conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
            if dispatched:
                dispatched.set()
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"{name}: HTTP {response.status}: {response.read()!r}")
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if "error" in event:
                    raise RuntimeError(f"{name}: {event['error']}")
                record["id"] = event.get("id", record.get("id"))
                if event.get("usage"):
                    record["usage"] = event["usage"]
                if event.get("timings"):
                    record["timings"] = event["timings"]
                chunks = [c.get("text", "") for c in event.get("choices", [])]
                if any(chunks):
                    record["updates"].append(time.monotonic())
                    for chunk in chunks:
                        output.update(chunk.encode())
                    if ready:
                        ready.set()
                    if cancel and len(record["updates"]) >= 3:
                        record["cancelled"] = True
                        break
        record["end"] = time.monotonic()
        record["output_sha256"] = output.hexdigest()
        if not record["cancelled"]:
            if not done or "usage" not in record:
                raise RuntimeError(f"{name}: incomplete stream")
            if record["usage"]["completion_tokens"] != tokens:
                raise RuntimeError(f"{name}: output length changed: {record['usage']}")
            if record["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0):
                raise RuntimeError(f"{name}: unexpected cache hit")
        updates = record["updates"]
        if not updates:
            raise RuntimeError(f"{name}: no output updates")
        record["ttft_s"] = updates[0] - record["start"]
        record["wall_s"] = record["end"] - record["start"]
        gaps = [b - a for a, b in zip(updates, updates[1:])]
        record["update_gap_p95_s"] = percentile(gaps, .95)
        record["update_gap_max_s"] = max(gaps) if gaps else None
        if not record["cancelled"] and tokens > 1:
            record["output_tokens_per_second"] = (tokens - 1) / (record["end"] - updates[0])
        return record

    def run(self, scenario, trial):
        before = self.idle()
        rng = random.Random(self.args.seed + trial)
        vocabulary = "amber birch cobalt dune ember fjord granite harbor indigo juniper quartz willow".split()
        long_prompt = "Document:\n" + " ".join(rng.choice(vocabulary) for _ in range(self.args.prompt_words))
        long_prompt += "\nSummary:"
        short_prompt = "List the positive integers in order, starting with one:\n1,"
        records, observed = [], []
        warm = scenario in ("mixed_short", "mixed_long", "sustained_busy", "cancel")
        count = self.args.arrivals if scenario.startswith("sustained") else 1
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.args.arrivals + 2) as pool:
            futures = []
            if scenario == "solo_decode":
                futures.append(pool.submit(self.stream, "decoder", short_prompt, self.args.decode_tokens))
            elif scenario == "solo_prefill":
                futures.append(pool.submit(self.stream, "long", long_prompt, 1))
            else:
                if warm:
                    ready = threading.Event()
                    decode = pool.submit(self.stream, "decoder", short_prompt, self.args.decode_tokens, ready)
                    futures.append(decode)
                    if not ready.wait(self.args.timeout):
                        raise TimeoutError("decoder did not start")
                long = pool.submit(self.stream, "long", long_prompt, 1)
                futures.append(long)
                deadline = time.monotonic() + self.args.timeout
                while time.monotonic() < deadline:
                    metrics = self.get("/metrics")
                    readers = metrics["prefill"]["requests"]
                    if readers:
                        observed = [{"phase": "before_arrival", "observed_at": time.monotonic(), **r}
                                    for r in readers]
                        break
                    if long.done():
                        long.result()
                        raise RuntimeError("long request completed before native prefill overlap was observed")
                    time.sleep(.01)
                if not observed:
                    raise TimeoutError("long native prefill was not observed")
                if warm and decode.done():
                    raise RuntimeError("decoder ended before the long prefill; increase --decode-tokens")
                for i in range(count):
                    if i:
                        time.sleep(self.args.interval)
                    # Retain prompt variation and dispatch timing in the record.
                    tokens = 8 if scenario == "mixed_short" else self.args.arrival_tokens
                    dispatched = threading.Event()
                    arrival = pool.submit(self.stream, f"arrival-{i}",
                                          f"Task {i}. {short_prompt}", tokens,
                                          cancel=scenario == "cancel", dispatched=dispatched)
                    futures.append(arrival)
                    if i == 0:
                        if not dispatched.wait(self.args.timeout):
                            raise TimeoutError("first arrival was not dispatched")
                        readers = self.get("/metrics")["prefill"]["requests"]
                        if not ({r["id"] for r in readers} & {r["id"] for r in observed}):
                            raise RuntimeError("long native prefill ended before the first arrival was dispatched")
                        observed.extend({"phase": "after_first_dispatch", "observed_at": time.monotonic(), **r}
                                        for r in readers)
            records = [future.result() for future in futures]
        after = self.idle()
        ended = time.monotonic()
        if after["service"]["requests_total"] - before["service"]["requests_total"] != len(records):
            raise RuntimeError("unrelated traffic contaminated the benchmark")
        if observed:
            long_record = next(r for r in records if r["name"] == "long")
            if long_record["id"] not in {r["id"] for r in observed}:
                raise RuntimeError("native prefill observation did not match the long response ID")
        completed_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in records)
        generated = after["scheduler"]["tokens_generated"] - before["scheduler"]["tokens_generated"]
        computed = after["scheduler"]["prompt_tokens_computed"] - before["scheduler"]["prompt_tokens_computed"]
        return {"scenario": scenario, "trial": trial, "wall_s": ended - started,
                "native_prefill_observation": observed, "requests": records,
                "computed_prompt_tokens": computed,
                "completed_response_tokens": completed_tokens, "generated_tokens": generated,
                "aggregate_output_tokens_per_second": generated / (ended - started),
                "aggregate_prompt_tokens_per_second": computed / (ended - started),
                "before": before, "after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--label", required=True, help="build commit and configuration identifier")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompt-words", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=1024)
    parser.add_argument("--arrival-tokens", type=int, default=256)
    parser.add_argument("--arrivals", type=int, default=8)
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=104)
    args = parser.parse_args()
    if min(args.trials, args.prompt_words, args.decode_tokens, args.arrival_tokens, args.arrivals) < 1:
        parser.error("counts must be positive")
    if args.interval < 0 or args.timeout <= 0:
        parser.error("interval must be nonnegative and timeout positive")
    bench = Bench(args)
    bench.idle()
    bench.stream("warmup", "Continue counting: 1, 2, 3,", 32)
    report = {"label": args.label, "model": bench.model,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for trial in range(args.trials):
        for scenario in args.scenarios:
            run = bench.run(scenario, trial)
            report["runs"].append(run)
            temporary = args.out.with_suffix(args.out.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.out)
            print(json.dumps({"scenario": scenario, "trial": trial, "wall_s": run["wall_s"],
                              "ttft_s": {r["name"]: r["ttft_s"] for r in run["requests"]}}), flush=True)


if __name__ == "__main__":
    main()
