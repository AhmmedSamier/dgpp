"""Matched cold-prefix generation checks. Run from the repository root."""
import hashlib
import http.client
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd() / "scripts"))
from data_paths import data_dir
from serve_prefill_probe import get, idle_metrics

host, port, destination = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
model = get(host, port, "/v1/models")["data"][0]["id"]
raw = (data_dir() / "gsm8k_test.jsonl").read_bytes()
words = " ".join(json.loads(line)["question"] for line in raw.splitlines()).split()
results = []
for length in (1600, 6000, 24000):
    prompt = f"GLM prefill validation {length}. Summarize the mathematical topics below in ten paragraphs.\n\n" + " ".join(words[:length])
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 128}
    before = idle_metrics(host, port, timeout=5)
    conn = http.client.HTTPConnection(host, port, timeout=1200)
    start = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {payload[:300]!r}")
        result = json.loads(payload)
    finally:
        conn.close()
    after = idle_metrics(host, port, timeout=5, completed_after=before["prompts_prefilled"])
    results.append({"words": length, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "data_sha256": hashlib.sha256(raw).hexdigest(), "wall_s": time.monotonic()-start,
                    "prefill_ms": after["prefill_ms"]-before["prefill_ms"], "response": result})
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{length} words: {result['usage']}, {results[-1]['wall_s']:.3f}s", flush=True)
