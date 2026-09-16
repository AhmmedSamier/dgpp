#!/usr/bin/env python3
"""The repository's serve_load client, with idle scheduler counters around each phase.

Engine decode time excludes admission/prefill and includes the engine call's
GPU, CPU and communication work. Client wall rates retain admission/prefill.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "scripts"))
import serve_load
from serve_prefill_probe import idle_metrics

original_phase = serve_load.phase

def phase(*args, **kwargs):
    host, port = args[:2]
    before = idle_metrics(host, port, timeout=5)
    rate = original_phase(*args, **kwargs)
    after = idle_metrics(host, port, timeout=5,
                         completed_after=before["prompts_prefilled"])
    reports = kwargs.get("reports") if "reports" in kwargs else args[10]
    result = reports[-1]
    names = ("step_ms", "prefill_ms", "decode_steps", "decode_rows", "tokens_generated",
             "prompts_prefilled", "prompt_tokens", "prompt_tokens_computed")
    delta = {name: after[name] - before[name] for name in names}
    if (delta["prompts_prefilled"] != result["concurrency"] or
            delta["tokens_generated"] != sum(r["tokens"] for r in result["requests"])):
        raise RuntimeError("scheduler counters include unexpected requests")
    tokens = delta["tokens_generated"] - delta["prompts_prefilled"]
    if tokens <= 0 or delta["step_ms"] <= 0:
        raise RuntimeError("no measurable decode work")
    result["engine"] = {**delta, "decode_tokens": tokens,
                         "tokens_per_s": tokens * 1000 / delta["step_ms"]}
    print(f"   engine decode: {result['engine']['tokens_per_s']:.3f} tok/s", flush=True)
    return rate

serve_load.phase = phase
serve_load.main()
