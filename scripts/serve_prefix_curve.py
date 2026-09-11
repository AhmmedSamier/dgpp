#!/usr/bin/env python3
"""The prefix cache's capacity/hit curve (M9's sign-off item for M7).

Against a running dgpp-serve: C conversations of three short turns each,
interleaved round-robin (turn 1 of every conversation, then turn 2 of every
conversation, then turn 3), at temperature 0 with the template keeping the
reasoning so a turn re-renders to the previous turn's ids. With S arena
slots, each conversation holds up to two entries (its prompt-cut entry and
its close entry); once C exceeds what S can hold, the LRU eviction turns
turns 2 and 3 into misses. Reports, from /v1/metrics deltas: hits, misses,
the hit fraction over turns 2 and 3, tokens saved, evictions, and the TTFT
split; and from the client, the mean TTFT per turn.

Usage: serve_prefix_curve.py HOST PORT C [LABEL]
Prints one summary line; the caller sweeps C and the service's
--prefix-cache-gib.
"""
import http.client
import json
import sys
import time
from serve_client import served_model

HOST, PORT, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LABEL = sys.argv[4] if len(sys.argv) > 4 else f"C{C}"
MODEL = served_model(HOST, PORT)
TOPICS = ["capital", "largest river", "currency", "highest mountain", "official language",
          "largest city", "national animal", "main export"]
PLACES = ["Portugal", "Chile", "Kenya", "Norway", "Vietnam", "Peru", "Egypt", "Poland",
          "Nepal", "Ghana", "Iceland", "Uruguay", "Latvia", "Laos", "Mali", "Oman",
          "Fiji", "Cuba", "Malta", "Togo", "Jordan", "Bolivia", "Estonia", "Bhutan",
          "Senegal", "Panama", "Qatar", "Samoa", "Tunisia", "Belize", "Guyana", "Brunei"]


def metrics():
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("GET", "/v1/metrics")
    m = json.loads(conn.getresponse().read())
    conn.close()
    return m


def ask(messages):
    body = {"model": MODEL, "messages": messages, "max_tokens": 96, "temperature": 0,
            "stream": True, "chat_template_kwargs": {"clear_thinking": False}}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    ttft, reasoning, content = None, [], []
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    # The reasoning and the content arrive as separate delta
                    # fields; the next turn hands both back, as the template
                    # renders them, so the turn re-renders to the same ids.
                    if d.get("reasoning_content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        reasoning.append(d["reasoning_content"])
                    if d.get("content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        content.append(d["content"])
    conn.close()
    return ttft, "".join(reasoning).strip(), "".join(content).strip()


def main():
    before = metrics()["prefix_cache"]
    convs = [[] for _ in range(C)]
    ttft_by_turn = {1: [], 2: [], 3: []}
    for turn in range(1, 4):
        for i in range(C):
            place = PLACES[i % len(PLACES)]
            topic = TOPICS[(i + turn) % len(TOPICS)]
            suffix = "" if i < len(PLACES) else f" (conversation {i})"
            convs[i].append({"role": "user", "content": f"What is the {topic} of {place}{suffix}? One short sentence."})
            ttft, reasoning, content = ask(convs[i])
            ttft_by_turn[turn].append(ttft or 0.0)
            convs[i].append({"role": "assistant", "content": content, "reasoning_content": reasoning})
    after = metrics()["prefix_cache"]
    d = {k: after[k] - before[k] for k in ("hits", "misses", "tokens_saved", "evictions", "hop_snapshots",
                                            "close_entries", "snapshots", "ttft_hit_count", "ttft_miss_count")}
    later = 2 * C  # turns 2 and 3 are the ones that can hit
    hit_frac = d["hits"] / later if later else 0.0
    hit_ms = ((after["ttft_hit_ms_avg"] * after["ttft_hit_count"] - before["ttft_hit_ms_avg"] * before["ttft_hit_count"])
              / d["ttft_hit_count"]) if d["ttft_hit_count"] else 0.0
    miss_ms = ((after["ttft_miss_ms_avg"] * after["ttft_miss_count"] - before["ttft_miss_ms_avg"] * before["ttft_miss_count"])
               / d["ttft_miss_count"]) if d["ttft_miss_count"] else 0.0
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print(f"{LABEL}: slots {after['slots']} conversations {C}: hits {d['hits']}/{later} later turns "
          f"({100*hit_frac:.0f} %), misses {d['misses']}, tokens saved {d['tokens_saved']}, "
          f"evictions {d['evictions']}, entries at the end {after['entries']}, "
          f"TTFT hit {hit_ms:.0f} ms / miss {miss_ms:.0f} ms; client TTFT per turn "
          f"{mean(ttft_by_turn[1]):.0f} / {mean(ttft_by_turn[2]):.0f} / {mean(ttft_by_turn[3]):.0f} ms")


if __name__ == "__main__":
    main()
