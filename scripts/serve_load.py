#!/usr/bin/env python3
"""Fixed-concurrency decode load against one dgpp-serve endpoint (2026-09-10,
the batched depth-2 study): for each concurrency c, c streaming chat
requests at once — distinct long-answer prompts, greedy, the template's
thinking off, max_tokens N — and the reading per phase:
  * per request: content tokens, time to first token, the decode pace
    between the first and the last content chunk (ms/token, client side);
  * the phase: wall, aggregate decode tokens/s (every request's tokens over
    the phase's decode span), and the phase's start/end stamps so the
    server's stats lines (rank 0's log: ms/step, tok/step/req, acceptance)
    can be laid beside it.

    serve_load.py HOST PORT [--concurrency 1,2,4] [--max-tokens 320] [--think]
"""
import argparse
import http.client
import json
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


def served_model(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read().decode())
    conn.close()
    ids = [m["id"] for m in data.get("data", [])]
    if not ids:
        raise SystemExit(f"no model served at {host}:{port}: {data}")
    return ids[0]


def stream_one(host, port, model, prompt, max_tokens, think, out):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
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


def phase(host, port, model, c, max_tokens, think, prompt_base):
    outs = [dict() for _ in range(c)]
    threads = []
    t_start = time.perf_counter()
    wall_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for i in range(c):
        prompt = PROMPTS[(prompt_base + i) % len(PROMPTS)]
        th = threading.Thread(target=stream_one, args=(host, port, model, prompt, max_tokens, think, outs[i]))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    t_end = time.perf_counter()
    wall_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"== concurrency {c}: {wall_start} .. {wall_end}  wall {t_end - t_start:.1f} s")
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
    args = ap.parse_args()
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
    base = 0
    summary = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        rate = phase(args.host, args.port, model, c, args.max_tokens, args.think, base)
        summary.append((c, rate))
        base += c
    print("== summary: " + "  ".join(f"c={c}: {r:.1f} tok/s" for c, r in summary))


if __name__ == "__main__":
    main()
