"""Streaming measurements shared by the serving benchmarks.

SSE updates can contain several tokens. Only server usage counts are token
counts; update timestamps measure what a client observes, not device commits.
"""
import json
import re
import time


def read_completion(response, clock=time.perf_counter, on_update=None):
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read(1024)!r}")
    buf = b""
    stamps, text = [], []
    usage, finish, done = None, None, False
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        now = clock()
        buf += chunk
        while (end := re.search(br"\r?\n\r?\n", buf)) is not None:
            frame, buf = buf[:end.start()], buf[end.end():]
            data = [line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:")]
            if not data:
                continue
            payload = b"\n".join(data)
            if payload == b"[DONE]":
                done = True
                continue
            if done:
                raise RuntimeError("SSE data after [DONE]")
            obj = json.loads(payload)
            if obj.get("error"):
                raise RuntimeError(f"stream error: {obj['error']}")
            if obj.get("usage") is not None:
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                content = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if content:
                    stamps.append(now)
                    text.append(content)
                    if on_update is not None:
                        on_update(now)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not done or finish is None:
        raise RuntimeError("incomplete completion stream (missing finish or [DONE])")
    tokens = (usage or {}).get("completion_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise RuntimeError("benchmark requires a positive usage.completion_tokens count")
    return {"status": response.status, "finish": finish, "usage": usage,
            "first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None,
            "chunks": len(stamps), "tokens": tokens, "text": "".join(text),
            "update_gaps_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]}


def phase_metrics(requests, start, end):
    if not requests or end <= start:
        raise ValueError("benchmark phase must contain completed requests and positive elapsed time")
    for request in requests:
        if request.get("error"):
            raise RuntimeError(request["error"])
        if request.get("first") is None or request.get("last") is None:
            raise RuntimeError("completion has no visible output timestamps")
    total = sum(r["tokens"] for r in requests)
    span = max(r["last"] for r in requests) - min(r["first"] for r in requests)
    return {"completion_tokens": total, "wall_s": end - start,
            "wall_tokens_per_s": total / (end - start),
            # Historical rate: includes first tokens, excludes initial TTFT,
            # and includes any later admissions. It is not steady-state decode.
            "legacy_output_span_s": span,
            "legacy_output_span_tokens_per_s": total / span if span > 0 else None}
