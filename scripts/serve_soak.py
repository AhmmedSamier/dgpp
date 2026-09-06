#!/usr/bin/env python3
"""The serving soak (M9, 2026-09-05): a mixed workload against one dgpp-serve
endpoint for a fixed duration, with the evidence the pass criteria need.

Workers (threads), each looping until the deadline:
  * `short` x3 — multi-turn chat: a conversation of three short questions
    (max_tokens 64, temperature 0, the template keeping the reasoning), so
    the prefix cache sees a second and a third turn per conversation;
  * `long`  x1 — a 512-token generation (an essay), sampled at the card's
    defaults;
  * `cancel` x1 — a streaming request the client abandons after 5–40
    tokens (the disconnect → cancel path), then a short pause;
  * `burst` — every BURST_EVERY seconds, BURST_N one-shot requests fired at
    once, above the admission queue bound: some are shed with 503 at the
    door, the rest are served; both counted.
Every request records its kind, HTTP status, time to first token, token
count, wall time and decode pace. Every 60 s a /v1/metrics snapshot is
appended (JSONL). At the end: per-window (10 min) p50/p95/p99 of TTFT and
pace per kind, status counts, the prefix cache's hit line, and the
"flat p99" ratio (max window p99 / min window p99) per kind.

Usage: serve_soak.py HOST PORT MINUTES OUT_DIR
Artifacts: OUT_DIR/requests.jsonl, metrics.jsonl, summary.txt.
"""
import http.client
import json
import os
import random
import socket
import sys
import threading
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.0.2.11"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
MINUTES = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
OUT = sys.argv[4] if len(sys.argv) > 4 else "soak_out"
MODEL = "unsloth/GLM-5.3-Flash-FP8"
BURST_EVERY = 300.0
BURST_N = 14
os.makedirs(OUT, exist_ok=True)

SHORT_TOPICS = [
    "the capital of {c}", "the largest river of {c}", "the currency of {c}",
    "the highest mountain of {c}", "the official language of {c}",
]
COUNTRIES = ["Portugal", "Chile", "Kenya", "Norway", "Vietnam", "Peru", "Egypt",
             "Poland", "Nepal", "Ghana", "Iceland", "Uruguay", "Latvia", "Laos",
             "Mali", "Oman", "Fiji", "Cuba", "Malta", "Togo"]
LONG_PROMPTS = [
    "Write a long, detailed history of the Roman Republic, one era per paragraph.",
    "Explain at length how a paged key-value cache works in a transformer server and what a prefix cache adds.",
    "Tell a long story about a lighthouse keeper who discovers the sea can speak.",
    "Describe, step by step and in depth, how TCP congestion control reacts to packet loss on a data-center fabric.",
]

lock = threading.Lock()
requests_f = open(os.path.join(OUT, "requests.jsonl"), "a", encoding="utf-8")
metrics_f = open(os.path.join(OUT, "metrics.jsonl"), "a", encoding="utf-8")
deadline = time.time() + MINUTES * 60.0
t_start = time.time()
stop = threading.Event()


def record(rec):
    rec["t"] = round(time.time() - t_start, 3)
    with lock:
        requests_f.write(json.dumps(rec) + "\n")
        requests_f.flush()


def stream(messages, max_tokens, kind, abandon_after=None, temperature=0.0,
           clear_thinking=False):
    """One streaming chat request. Returns (status, ttft_ms, tokens, wall_ms,
    pace_ms, (reasoning, content), code) — code is the error code when the
    stream errored; reasoning and content are the two delta fields kept
    apart, as the next turn must hand them back."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "stream": True, "temperature": temperature,
            "chat_template_kwargs": {"clear_thinking": clear_thinking}}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.perf_counter()
    status, ttft, tokens, code = 0, None, 0, None
    reasoning, content = [], []
    t_first = t_last = None
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            raw = resp.read().decode("utf-8", "replace")
            try:
                code = json.loads(raw).get("error", {}).get("code")
            except json.JSONDecodeError:
                code = "unparsable"
            return status, None, 0, (time.perf_counter() - t0) * 1000, None, ("", ""), code
        buf = b""
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        code = obj["error"].get("code")
                        continue
                    for ch in obj.get("choices", []):
                        d = ch.get("delta", {})
                        for field, sink in (("reasoning_content", reasoning), ("content", content)):
                            piece = d.get(field)
                            if not piece:
                                continue
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                                ttft = (now - t0) * 1000
                            t_last = now
                            tokens += 1
                            sink.append(piece)
            if abandon_after is not None and tokens >= abandon_after:
                conn.sock.close()  # the client walks away mid-stream
                break
    except (OSError, http.client.HTTPException) as e:
        code = code or ("client_error:" + type(e).__name__)
    finally:
        try:
            conn.close()
        except OSError:
            pass
    wall = (time.perf_counter() - t0) * 1000
    pace = ((t_last - t_first) * 1000 / (tokens - 1)) if tokens > 1 else None
    return status, ttft, tokens, wall, pace, ("".join(reasoning), "".join(content)), code


def short_worker(idx):
    rng = random.Random(1000 + idx)
    while time.time() < deadline and not stop.is_set():
        c = rng.choice(COUNTRIES)
        messages = []
        for turn in range(3):
            q = rng.choice(SHORT_TOPICS).format(c=c)
            messages.append({"role": "user", "content": f"What is {q}? Answer in one short sentence."})
            status, ttft, tokens, wall, pace, (reasoning, content), code = stream(messages, 64, "short")
            record({"kind": "short", "turn": turn + 1, "status": status, "ttft_ms": ttft,
                    "tokens": tokens, "wall_ms": wall, "pace_ms": pace, "code": code})
            if status != 200 or code:
                break
            # The next turn carries the answer back as the template renders
            # it (the reasoning kept), so the turn re-renders to the same ids
            # and the prefix cache's close entry can hit.
            messages.append({"role": "assistant", "content": content.strip(),
                             "reasoning_content": reasoning.strip()})
        time.sleep(rng.uniform(0.2, 1.0))


def long_worker(idx):
    rng = random.Random(2000 + idx)
    while time.time() < deadline and not stop.is_set():
        p = rng.choice(LONG_PROMPTS)
        status, ttft, tokens, wall, pace, _, code = stream(
            [{"role": "user", "content": p}], 512, "long", temperature=1.0)
        record({"kind": "long", "status": status, "ttft_ms": ttft, "tokens": tokens,
                "wall_ms": wall, "pace_ms": pace, "code": code})
        time.sleep(rng.uniform(0.5, 2.0))


def cancel_worker(idx):
    rng = random.Random(3000 + idx)
    while time.time() < deadline and not stop.is_set():
        p = rng.choice(LONG_PROMPTS)
        n = rng.randint(5, 40)
        status, ttft, tokens, wall, pace, _, code = stream(
            [{"role": "user", "content": p}], 400, "cancel", abandon_after=n)
        record({"kind": "cancel", "status": status, "ttft_ms": ttft, "tokens": tokens,
                "wall_ms": wall, "pace_ms": pace, "code": code, "abandon_after": n})
        time.sleep(rng.uniform(2.0, 6.0))


def one_shot(messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"clear_thinking": False}}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.perf_counter()
    code = None
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        status = resp.status
        if status != 200:
            try:
                code = json.loads(raw).get("error", {}).get("code")
            except json.JSONDecodeError:
                code = "unparsable"
        tokens = 0
        try:
            tokens = json.loads(raw).get("usage", {}).get("completion_tokens", 0)
        except json.JSONDecodeError:
            pass
    except (OSError, http.client.HTTPException) as e:
        status, tokens, code = 0, 0, "client_error:" + type(e).__name__
    finally:
        conn.close()
    return status, tokens, (time.perf_counter() - t0) * 1000, code


def burst_worker():
    rng = random.Random(4000)
    next_burst = time.time() + 60.0
    while time.time() < deadline and not stop.is_set():
        if time.time() < next_burst:
            time.sleep(1.0)
            continue
        next_burst = time.time() + BURST_EVERY
        results = [None] * BURST_N

        def fire(i):
            c = rng.choice(COUNTRIES)
            results[i] = one_shot([{"role": "user", "content": f"Name one city in {c}. One word."}], 16)
        threads = [threading.Thread(target=fire, args=(i,)) for i in range(BURST_N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        shed = sum(1 for r in results if r and r[0] == 503)
        served = sum(1 for r in results if r and r[0] == 200)
        for r in results:
            record({"kind": "burst", "status": r[0], "tokens": r[1], "wall_ms": r[2], "code": r[3]})
        record({"kind": "burst_summary", "fired": BURST_N, "served": served, "shed": shed})


def metrics_worker():
    while time.time() < deadline and not stop.is_set():
        try:
            conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
            conn.request("GET", "/v1/metrics")
            raw = conn.getresponse().read().decode("utf-8", "replace")
            conn.close()
            obj = json.loads(raw)
            obj["t"] = round(time.time() - t_start, 1)
            with lock:
                metrics_f.write(json.dumps(obj) + "\n")
                metrics_f.flush()
        except Exception as e:  # noqa: BLE001 — the soak keeps going
            with lock:
                metrics_f.write(json.dumps({"t": round(time.time() - t_start, 1), "error": str(e)}) + "\n")
                metrics_f.flush()
        for _ in range(60):
            if time.time() >= deadline or stop.is_set():
                break
            time.sleep(1.0)


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize():
    recs = [json.loads(l) for l in open(os.path.join(OUT, "requests.jsonl"), encoding="utf-8")]
    window = 600.0
    n_windows = int(MINUTES * 60 / window + 0.999)
    lines = []
    lines.append(f"soak: {MINUTES:g} min against {HOST}:{PORT}, {len(recs)} records")
    by_status = {}
    for r in recs:
        if r["kind"] == "burst_summary":
            continue
        key = f"{r['kind']}:{r.get('status')}:{r.get('code') or ''}"
        by_status[key] = by_status.get(key, 0) + 1
    lines.append("status counts (kind:http:code): " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    bursts = [r for r in recs if r["kind"] == "burst_summary"]
    if bursts:
        lines.append("bursts: " + "; ".join(f"fired {b['fired']} served {b['served']} shed {b['shed']}" for b in bursts))
    for kind in ("short", "long", "cancel"):
        lines.append(f"-- {kind}: per {int(window/60)}-min window: n, TTFT p50/p95/p99 ms, pace p50/p95/p99 ms/token")
        p99s = []
        for w in range(n_windows):
            rs = [r for r in recs if r["kind"] == kind and r.get("status") == 200 and w * window <= r["t"] < (w + 1) * window]
            tt = [r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None]
            pc = [r["pace_ms"] for r in rs if r.get("pace_ms") is not None]
            if not rs:
                continue
            p99 = percentile(tt, 0.99)
            if p99 is not None:
                p99s.append(p99)
            fmt = lambda v: "-" if v is None else f"{v:.0f}"
            lines.append(f"   w{w}: n={len(rs)} ttft {fmt(percentile(tt,0.5))}/{fmt(percentile(tt,0.95))}/{fmt(p99)}"
                         f"  pace {fmt(percentile(pc,0.5))}/{fmt(percentile(pc,0.95))}/{fmt(percentile(pc,0.99))}")
        if len(p99s) >= 2 and min(p99s) > 0:
            lines.append(f"   flat p99 (TTFT): max/min window ratio {max(p99s)/min(p99s):.2f}")
    mets = [json.loads(l) for l in open(os.path.join(OUT, "metrics.jsonl"), encoding="utf-8")]
    mets = [m for m in mets if "prefix_cache" in m]
    if mets:
        last = mets[-1]
        sv, sc, pc = last.get("service", {}), last.get("scheduler", {}), last["prefix_cache"]
        lines.append(f"metrics at the end: requests_total {sv.get('requests_total')} shed {sv.get('requests_shed')} "
                     f"cancelled {sv.get('requests_cancelled')} failed {sv.get('requests_failed')} "
                     f"engine_failed {sv.get('engine_failed')} tokens_out {sv.get('tokens_out')} "
                     f"pool {sc.get('pool_blocks_in_use')}/{sc.get('pool_blocks_total')} blocks; "
                     f"prefix cache entries {pc.get('entries')} hits {pc.get('hits')} misses {pc.get('misses')} "
                     f"tokens_saved {pc.get('tokens_saved')} evictions {pc.get('evictions')} hops {pc.get('hop_snapshots')} "
                     f"skipped_no_block {pc.get('skipped_no_block')} attach {pc.get('attach_ms_avg', 0):.2f} ms "
                     f"snapshot {pc.get('snapshot_ms_avg', 0):.2f} ms ttft hit {pc.get('ttft_hit_ms_avg', 0):.0f} ms / "
                     f"miss {pc.get('ttft_miss_ms_avg', 0):.0f} ms")
        # Memory drift across the hour: the pool's blocks in use and the
        # cache's entries at each sample (a leak would climb monotonically).
        series = [(m["t"], m.get("scheduler", {}).get("pool_blocks_in_use"), m["prefix_cache"].get("entries")) for m in mets]
        pick = series[:: max(1, len(series) // 6)] + [series[-1]]
        lines.append("pool blocks in use / cache entries over time: " +
                     ", ".join(f"{int(t)//60}m {b}/{e}" for t, b, e in pick))
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def main():
    workers = [threading.Thread(target=short_worker, args=(i,)) for i in range(3)]
    workers.append(threading.Thread(target=long_worker, args=(0,)))
    workers.append(threading.Thread(target=cancel_worker, args=(0,)))
    workers.append(threading.Thread(target=burst_worker))
    workers.append(threading.Thread(target=metrics_worker))
    for w in workers:
        w.daemon = True
        w.start()
    last_report = time.time()
    while time.time() < deadline:
        time.sleep(5.0)
        if time.time() - last_report >= 300.0:
            last_report = time.time()
            n = sum(1 for _ in open(os.path.join(OUT, "requests.jsonl"), encoding="utf-8"))
            print(f"soak: {(time.time() - t_start)/60:.0f} min, {n} records", flush=True)
    stop.set()
    for w in workers:
        w.join(timeout=120)
    summarize()


if __name__ == "__main__":
    main()
