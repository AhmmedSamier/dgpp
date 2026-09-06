#!/usr/bin/env python3
"""Measure the serving pace of one dgpp-serve endpoint from the client side.

For each request: POST /v1/chat/completions with stream=true, stamp every
SSE data chunk's arrival, and report
  * time to first content token (prefill + the first pick + HTTP),
  * decode pace = (t_last_token - t_first_token) / (tokens - 1),
  * client wall time and the server's usage line.
The server log's per-token timestamps are the second witness (see the
awk one-liner in the record); this script is the one a user would feel.
"""
import http.client
import json
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.0.2.11"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
LABEL = sys.argv[4] if len(sys.argv) > 4 else "run"
PROMPT = (sys.argv[5] if len(sys.argv) > 5 else
          "Explain, in a few paragraphs, why a CUDA graph replay can be faster "
          "than launching the same kernels eagerly, and what it costs.")


def one(prompt, max_tokens, label):
    body = json.dumps({
        "model": "unsloth/GLM-5.3-Flash-FP8",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body,
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
                    if delta.get("content"):
                        stamps.append(now)
                        text.append(delta["content"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    t_end = time.perf_counter()
    conn.close()
    n = len(stamps)
    ttft = (stamps[0] - t0) * 1e3 if stamps else float("nan")
    pace = ((stamps[-1] - stamps[0]) / (n - 1) * 1e3) if n > 1 else float("nan")
    print(f"[{label}] status {resp.status} finish={finish} usage={usage}")
    print(f"[{label}] content chunks {n}, ttft {ttft:.0f} ms, decode pace "
          f"{pace:.2f} ms/chunk (client), wall {(t_end - t0) * 1e3:.0f} ms")
    print(f"[{label}] text: {''.join(text)[:400]!r}")
    return pace


if __name__ == "__main__":
    one(PROMPT, MAX_TOKENS, LABEL)
