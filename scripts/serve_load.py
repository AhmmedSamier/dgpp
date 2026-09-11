#!/usr/bin/env python3
"""Measure decode throughput at fixed concurrency against dgpp-serve.

Each phase sends concurrent streaming requests with distinct long-answer
prompts. Requests are greedy with thinking disabled unless overridden.
The report includes per-request tokens, time to first token and decode
pace, plus aggregate throughput and timestamps for comparison with the
server's stats log.

Use --classes to run separate sweeps for prose, code, JSON, math and chat.
Use --temperature to measure sampled requests.

    serve_load.py HOST PORT [--concurrency 1,2,4] [--max-tokens 320] [--think]
                            [--classes prose,code,json,math,chat] [--temperature T]
"""


import argparse
import http.client
import json
import os
import sys
import threading
import time
from datetime import datetime

PROMPTS = [
    "Write a detailed technical essay of at least 800 words on how a paged key-value cache works in a "
    "transformer inference engine, covering block tables, eviction, prefix sharing and the trade-offs "
    "against a contiguous cache. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on speculative decoding with a draft model: "
    "the verify pass, acceptance rules, why throughput rises, when it falls, and the memory bandwidth "
    "argument. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on tensor parallelism across several GPUs "
    "connected by RDMA: sharding the weights, the all-reduce at each block boundary, the latency budget "
    "per step and the failure modes. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on 4-bit weight quantization for large "
    "mixture-of-experts models: block scales, the accuracy cost, the kernels needed and how to validate "
    "the result. Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on CUDA graphs for decode steps: capture, "
    "replay, what they save, the constraints on the kernels inside them and how to debug a bad capture. "
    "Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on the RoCE fabric of a small GPU cluster: "
    "queue pairs, incast drops, pacing, and how to measure whether the network or the GPU bounds a step. "
    "Use section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on the request scheduler of an inference "
    "server: admission, the KV budget, round-robin decode, cancellation and fairness under load. Use "
    "section headings and full paragraphs.",
    "Write a detailed technical essay of at least 800 words on tokenizers for large language models: "
    "byte-level BPE, pre-tokenization, special tokens, chat templates and the pitfalls of a mismatch "
    "between training and serving. Use section headings and full paragraphs.",
]

# The per-class corpus. Prompt 0 of every class is the same TEXT as that class in
# scripts/fabric_mtp_classes.sh, which has been the project's class corpus since
# 2026-09-05 and produced every published per-class table — so a concurrency-1
# phase here is comparable to those numbers rather than to a fresh set of words.
# Prompts 1..3 are of the same kind and exist only because a phase at
# concurrency c needs c distinct prompts: repeating one would share a decode
# path and attach to the prefix cache, and neither is what a class costs under
# load. check_corpus() below re-reads the shell script and fails if prompt 0 has
# drifted from it.
CLASSES = {
    "prose": [
        "Write a long, detailed history of the Roman Republic from its founding to the rise of "
        "Augustus, one era per paragraph.",
        "Write a long, detailed history of the Hanseatic League from its founding to its dissolution, "
        "one era per paragraph.",
        "Write a long, detailed history of the silk trade in Lyon from its arrival to its decline, "
        "one era per paragraph.",
        "Write a long, detailed history of the Dutch water boards from their origin to the present, "
        "one era per paragraph.",
    ],
    "code": [
        "Write a Python module that parses an OpenAI-style SSE stream of chat completion chunks into "
        "a single message object, with type hints, docstrings, and a small set of unit tests using "
        "pytest.",
        "Write a Python module implementing an LRU cache with a byte budget and per-entry sizes, with "
        "type hints, docstrings, and a small set of unit tests using pytest.",
        "Write a Python module that reads a safetensors header and reports each tensor's name, dtype "
        "and shape, with type hints, docstrings, and a small set of unit tests using pytest.",
        "Write a Python module implementing a token bucket rate limiter usable from several threads, "
        "with type hints, docstrings, and a small set of unit tests using pytest.",
    ],
    "json": [
        "Return a JSON array of 25 objects, each with the fields country, capital, "
        "population_millions and currency, for 25 different countries. Output only the JSON.",
        "Return a JSON array of 25 objects, each with the fields city, country, population_millions "
        "and founded_year, for 25 different cities. Output only the JSON.",
        "Return a JSON array of 25 objects, each with the fields element, symbol, atomic_number and "
        "group, for 25 different chemical elements. Output only the JSON.",
        "Return a JSON array of 25 objects, each with the fields river, continent, length_km and "
        "mouth, for 25 different rivers. Output only the JSON.",
    ],
    "math": [
        "A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward "
        "it 30 minutes later. Work out step by step when and where they meet, then generalize the "
        "formula and check it with two other examples.",
        "A cistern is filled by two pipes in 12 and 18 minutes and emptied by a third in 24. Work out "
        "step by step how long it takes to fill, then generalize the formula and check it with two "
        "other examples.",
        "Find the area between the curves y = x^2 and y = 2x + 3. Work it out step by step, then "
        "generalize the method and check it with two other pairs of curves.",
        "A bag holds 7 red and 5 blue balls and three are drawn without replacement. Work out step by "
        "step the probability of exactly two red, then generalize the formula and check it with two "
        "other bags.",
    ],
    "chat": [
        "Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same "
        "kernels eagerly, and what it costs.",
        "Explain, in a few paragraphs, why a paged key-value cache beats a contiguous one for a "
        "serving engine, and what it costs.",
        "Explain, in a few paragraphs, why tensor parallelism needs an all-reduce at each block "
        "boundary, and what bounds the step.",
        "Explain, in a few paragraphs, why 4-bit weight quantization helps a memory-bound decode "
        "step, and where the accuracy goes.",
    ],
}


def check_corpus():
    """Prompt 0 of each class must still be fabric_mtp_classes.sh's prompt."""
    import re as _re
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fabric_mtp_classes.sh")
    try:
        text = open(sh).read()
    except OSError:
        return  # run from outside the tree: nothing to check against
    fab = dict(_re.findall(r'^P\[(\w+)\]="(.*)"$', text, _re.M))
    for name, prompts in CLASSES.items():
        want = fab.get(name)
        if want is not None and prompts[0] != want:
            raise SystemExit(
                f"class {name!r} prompt 0 has drifted from fabric_mtp_classes.sh; "
                "the per-class numbers would not be comparable to the published tables")

def served_model(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read().decode())
    conn.close()
    ids = [m["id"] for m in data.get("data", [])]
    if not ids:
        raise SystemExit(f"no model served at {host}:{port}: {data}")
    return ids[0]


def stream_one(host, port, model, prompt, max_tokens, think, out, temperature=0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    conn = http.client.HTTPConnection(host, port, timeout=900)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status == 400 and not think:
        # The template has no thinking knob: plain.
        conn.close()
        body.pop("chat_template_kwargs", None)
        conn = http.client.HTTPConnection(host, port, timeout=900)
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    stamps = []
    text = []
    usage = None
    finish = None
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                obj = json.loads(payload)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        stamps.append(now)
                        text.append(delta.get("content") or delta.get("reasoning_content"))
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    conn.close()
    out["status"] = resp.status
    out["finish"] = finish
    out["usage"] = usage
    out["t0"] = t0
    out["first"] = stamps[0] if stamps else None
    out["last"] = stamps[-1] if stamps else None
    out["chunks"] = len(stamps)
    out["tokens"] = (usage or {}).get("completion_tokens", len(stamps))
    out["text"] = "".join(text)


def phase(host, port, model, c, max_tokens, think, prompt_base, prompts=None, temperature=0, label=""):
    prompts = prompts or PROMPTS
    outs = [dict() for _ in range(c)]
    threads = []
    t_start = time.perf_counter()
    wall_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for i in range(c):
        prompt = prompts[(prompt_base + i) % len(prompts)]
        th = threading.Thread(target=stream_one,
                              args=(host, port, model, prompt, max_tokens, think, outs[i], temperature))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    t_end = time.perf_counter()
    wall_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"== {label}concurrency {c}: {wall_start} .. {wall_end}  wall {t_end - t_start:.1f} s")
    total = 0
    span_first = min(o["first"] for o in outs if o["first"])
    span_last = max(o["last"] for o in outs if o["last"])
    for i, o in enumerate(outs):
        n = o["tokens"]
        total += n
        pace = ((o["last"] - o["first"]) / (o["chunks"] - 1) * 1e3) if o["chunks"] > 1 else float("nan")
        ttft = (o["first"] - o["t0"]) * 1e3 if o["first"] else float("nan")
        print(f"   req {i}: status {o['status']} finish={o['finish']} tokens {n} chunks {o['chunks']} "
              f"ttft {ttft:.0f} ms  decode pace {pace:.2f} ms/token")
    span = span_last - span_first
    print(f"   aggregate: {total} tokens in {span:.1f} s of decode = {total / span:.1f} tok/s "
          f"({total / span / c:.1f} per request)")
    return total / span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--think", action="store_true", help="leave the template's thinking on")
    ap.add_argument("--warm", type=int, default=1, help="warm requests before the phases")
    ap.add_argument("--isolation", type=int, default=0, metavar="C",
                    help="the transcript-isolation check: prompt 0 alone, then beside C-1 others; its text must match")
    ap.add_argument("--classes", default=None, metavar="LIST",
                    help="run the sweep once per class (prose,code,json,math,chat), each phase drawing "
                         "its prompts from that class alone; 'all' runs every class")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) is greedy; a positive value samples, for the pace at the card's defaults")
    args = ap.parse_args()
    if args.classes:
        check_corpus()
    model = served_model(args.host, args.port)
    print(f"model {model}")
    for w in range(args.warm):
        o = {}
        stream_one(args.host, args.port, model, PROMPTS[w % len(PROMPTS)], 64, args.think, o)
        print(f"warm {w}: status {o['status']} tokens {o['tokens']}")
    if args.isolation > 1:
        alone = {}
        stream_one(args.host, args.port, model, PROMPTS[0], args.max_tokens, args.think, alone)
        outs = [dict() for _ in range(args.isolation)]
        threads = [threading.Thread(target=stream_one,
                                    args=(args.host, args.port, model, PROMPTS[i % len(PROMPTS)], args.max_tokens,
                                          args.think, outs[i]))
                   for i in range(args.isolation)]
        for th in threads: th.start()
        for th in threads: th.join()
        same = alone["text"] == outs[0]["text"] and alone["tokens"] == outs[0]["tokens"]
        print(f"== isolation: prompt 0 alone ({alone['tokens']} tokens) vs beside {args.isolation - 1} others "
              f"({outs[0]['tokens']} tokens): {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            a, b = alone["text"], outs[0]["text"]
            k = 0
            while k < len(a) and k < len(b) and a[k] == b[k]: k += 1
            print(f"   first difference at char {k}: alone {a[k:k+60]!r} | batched {b[k:k+60]!r}")
    cs = [int(x) for x in args.concurrency.split(",")]
    draw = "greedy" if args.temperature == 0 else f"sampled T={args.temperature}"
    if args.classes:
        names = list(CLASSES) if args.classes == "all" else args.classes.split(",")
        for n in names:
            if n not in CLASSES:
                raise SystemExit(f"unknown class {n!r}; known: {', '.join(CLASSES)}")
        table = {}
        for n in names:
            base = 0
            for c in cs:
                rate = phase(args.host, args.port, model, c, args.max_tokens, args.think, base,
                             CLASSES[n], args.temperature, f"{n} ")
                table[(n, c)] = rate
                base += c
        print(f"== per-class aggregate tokens/s, {draw}, max_tokens {args.max_tokens}")
        print("| class | " + " | ".join(f"c={c}" for c in cs) + " |")
        print("|---|" + "---|" * len(cs))
        for n in names:
            print(f"| {n} | " + " | ".join(f"{table[(n, c)]:.1f}" for c in cs) + " |")
        return
    base = 0
    summary = []
    for c in cs:
        rate = phase(args.host, args.port, model, c, args.max_tokens, args.think, base,
                     None, args.temperature)
        summary.append((c, rate))
        base += c
    print(f"== summary ({draw}): " + "  ".join(f"c={c}: {r:.1f} tok/s" for c, r in summary))


if __name__ == "__main__":
    main()
