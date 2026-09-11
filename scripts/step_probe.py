#!/usr/bin/env python3
"""The decode-step probe (2026-09-06): one streaming request at a time against
a booted world, only when it is idle, and the engine's own step count from
rank 0's log ('slot N closed: K sampled decode steps') — ms/step, tok/step
and ms/token per context size. Modes: short (a 26-token prompt), short+tools
(the same with an agent client's tool schemas: the grammar's cost), long
(~16K tokens of prose), ctx:N (~N tokens of prose, a nonce defeats the
prefix cache). Usage:
  scripts/step_probe.py [--host URL] [--log PATH] [--model ID] [--max-tokens N] MODE...
The 2026-09-06 record: 43.4 ms/step at 26 tokens, 52–56 at 4K–32K before the
select-kernel rewrite; 41.8 / 44.4 / 44.5 / 45.2 at 26 / 8K / 18K / 32K after."""
import argparse, json, os, re, sys, time, urllib.request
from serve_client import default_url, default_log, model_at_url

_ap = argparse.ArgumentParser()
_ap.add_argument("--host")
_ap.add_argument("--log")
_ap.add_argument("--model")
_ap.add_argument("--max-tokens", type=int, default=160)
_args, _modes = _ap.parse_known_args()
HOST = _args.host or default_url()
LOG = _args.log or default_log()
MODEL = _args.model or model_at_url(HOST)

TOOLS = [
  {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {
      "command": {"type": "string", "description": "The command"},
      "timeout": {"type": "number", "minimum": 0, "description": "ms"},
      "description": {"type": "string"}}, "required": ["command"]}}},
  {"type": "function", "function": {"name": "read", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {
      "file_path": {"type": "string"},
      "offset": {"type": "integer", "minimum": 0},
      "limit": {"type": "integer", "minimum": 1}}, "required": ["file_path"]}}},
  {"type": "function", "function": {"name": "grep", "description": "Search.",
    "parameters": {"type": "object", "properties": {
      "pattern": {"type": "string"}, "path": {"type": "string"},
      "output_mode": {"type": "string", "enum": ["content", "files", "count"]}},
      "required": ["pattern"]}}},
]

PARA = ("The aqueduct carried water from the hills to the city through a series of "
        "arches whose stones were cut so precisely that no mortar was needed; the "
        "engineers measured the gradient with a chorobates and checked it every "
        "hundred paces, since a fall too steep would scour the channel and one too "
        "gentle would let silt settle and choke it within a season. ")

def log_lines():
    with open(LOG, "rb") as f:
        return f.read().decode("utf-8", "replace").splitlines()

def idle(lines):
    stats = [l for l in lines if "stats: rank" in l]
    if not stats: return True
    m = re.search(r"(?:running|live) (\d+), queued (\d+)", stats[-1])
    return m and m.group(1) == "0" and m.group(2) == "0"

def wait_idle(max_s):
    t0 = time.time()
    while time.time() - t0 < max_s:
        lines = log_lines()
        if idle(lines):
            # no admission in the last 12 s
            adm = [l for l in lines[-40:] if "admitted to slot" in l]
            last = adm[-1][:23] if adm else None
            if last is None or (time.time() - time.mktime(time.strptime(last[:19], "%Y-%m-%d %H:%M:%S"))) > 12:
                return True
        time.sleep(2)
    return False

def run(name, messages, tools=None, max_tokens=None):
    max_tokens = max_tokens or _args.max_tokens
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "temperature": 1.0, "top_p": 0.95}
    if tools: body["tools"] = tools
    before = len(log_lines())
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; n = 0; usage = None; arrivals = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            ev = json.loads(payload)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                    now = time.time()
                    if first is None: first = now
                    last = now; n += 1; arrivals.append(now)
    t_end = time.time()
    time.sleep(0.5)
    tail = log_lines()[before:]
    closed = [l for l in tail if "closed:" in l and "sampled decode steps" in l]
    steps = int(re.search(r"(\d+) sampled decode steps", closed[-1]).group(1)) if closed else None
    retired = [l for l in tail if "retired (" in l]
    gen = int((re.search(r"(\d+) tokens generated", retired[-1]) or re.search(r"retired \(\w+\): (\d+) tok", retired[-1])).group(1)) if retired else None
    others = [l for l in tail if "admitted to slot" in l]
    ttft = (first - t0) * 1000 if first else None
    decode_s = (last - first) if (first and last) else None
    print(f"--- {name}")
    print(f"  ttft {ttft:.0f} ms; chunks {n}; generated {gen}; decode wall {decode_s:.2f} s"
          if ttft is not None and decode_s is not None else f"  ttft {ttft} chunks {n} gen {gen}")
    if decode_s and gen and gen > 1:
        print(f"  ms/token (client) {1000*decode_s/(gen-1):.1f}")
    if decode_s and steps and steps > 1:
        print(f"  engine decode steps {steps}; ms/step {1000*decode_s/(steps-1):.1f}; tok/step {gen/steps:.2f}")
    print(f"  admissions in window: {len(others)} (1 = clean)")
    for l in tail:
        if "stats: rank 0" in l and "(0 steps" not in l and "decode 0 steps" not in l:
            print("  " + (l.split("| decode ")[1].split("| prefill")[0] if "| decode " in l else l[24:120]))
    sys.stdout.flush()

if __name__ == "__main__":
    which = _modes or ["short", "short+tools", "long"]
    short_msgs = [{"role": "user", "content": "Explain in two paragraphs how a Roman aqueduct kept its gradient."}]
    long_text = PARA * 260   # ~16k tokens of prose
    long_msgs = [{"role": "user", "content": long_text + "\n\nSummarize the passage above in two paragraphs."}]
    for w in which:
        if not wait_idle(540):
            print(f"--- {w}: world never went idle; skipped"); continue
        if w == "short": run("short prompt, no tools", short_msgs)
        elif w == "short+tools": run("short prompt, tools (grammar auto)", short_msgs, TOOLS)
        elif w == "long": run("long prompt (~16k tok), no tools", long_msgs)
        elif w.startswith("ctx:"):
            n = int(w[4:]); reps = max(1, n // 72)
            nonce = f"Nonce{n}{int(time.time())%100000}. "
            msgs = [{"role": "user", "content": nonce + PARA * reps + "\n\nSummarize the passage above in two paragraphs."}]
            run(f"ctx ~{reps*72} tok, no tools", msgs)
        time.sleep(11)  # let a clean stats window close
