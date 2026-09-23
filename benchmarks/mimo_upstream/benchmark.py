"""Matched MiMo recipe candidate probes; parent alone executes on Sparks."""

import concurrent.futures, json, pathlib, time, urllib.request, sys, os

ROOT = pathlib.Path(
    os.environ.get("DGPP_MIMO_BENCH_ROOT", pathlib.Path(__file__).resolve().parent)
)
URL = (
    os.environ.get("DGPP_MIMO_BENCH_URL", "http://127.0.0.1:30001/v1/").rstrip("/")
    + "/"
)


def metrics():
    with urllib.request.urlopen(URL + "metrics", timeout=15) as r:
        return json.load(r)


def call(name, prompt, limit=256, cache=False, logprobs=False):
    request = {
        "model": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": limit,
        "stream": True,
        "stream_options": {"include_usage": True},
        "prefix_cache": cache,
    }
    if logprobs:
        request.update(logprobs=True, top_logprobs=2)
    start = time.monotonic()
    first = None
    events = []
    reasoning = []
    content = []
    with urllib.request.urlopen(
        urllib.request.Request(
            URL + "chat/completions",
            data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=1200,
    ) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            events.append(event)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if first is None:
                        first = time.monotonic()
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
    end = time.monotonic()
    usage = next((e["usage"] for e in reversed(events) if e.get("usage")), None)
    assert usage and first is not None, (name, events)
    result = dict(
        name=name,
        request=request,
        seconds=end - start,
        ttft=first - start,
        decode_seconds=end - first,
        content="".join(content),
        reasoning="".join(reasoning),
        usage=usage,
        events=events,
    )
    print(
        name,
        "prompt",
        usage["prompt_tokens"],
        "out",
        usage["completion_tokens"],
        "cached",
        usage.get("prompt_tokens_details"),
        "ttft",
        round(first - start, 3),
        "decode",
        round(end - first, 3),
        flush=True,
    )
    return result


PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals. Explain its time complexity and give three test cases.",
    "json": "Return a JSON array of 20 objects. Each object has id (integer 1 through 20), squared (id squared), and parity (even or odd). Be precise.",
    "prose": "Describe a quiet walk through a coastal town in winter in three paragraphs. Use specific sensory details.",
    "math": "Solve 17x + 23 = 210, then explain a general method for solving a linear equation and verify the result.",
}


def long_prompt(n):
    return (
        "Recipe experiment fixed prompt v1.\n"
        + "".join(f"Record {i}: value {i%17}.\n" for i in range(n))
        + "\nExplain what this dataset contains, describe a Python function to summarize the values, and provide an example implementation."
    )


def main(tag):
    output = ROOT / "raw" / tag
    output.mkdir(parents=True, exist_ok=True)
    assert not (
        output / "results.json"
    ).exists(), "Use new tag, preserve earlier evidence"
    before = metrics()
    assert before["scheduler"]["active"] == before["scheduler"]["queued"] == 0
    records = []

    def save(r):
        records.append(r)
        (output / "results.json").write_text(json.dumps(records, indent=2))

    for rep in range(2):
        for name, prompt in PROMPTS.items():
            save(call(f"{name}-c1-r{rep}", prompt))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            tasks = [(f"{name}-c2-r{rep}", PROMPTS[name]) for name in ["code", "json"]]
            for r in pool.map(lambda args: call(*args), tasks):
                save(r)
        for n in [700, 2800]:
            save(call(f"cold-{n}-r{rep}", long_prompt(n), 64))
    save(call("long-seed", long_prompt(6000), 128, True))
    for rep in range(2):
        save(call(f"long-warm-r{rep}", long_prompt(6000), 128, True))
    after = metrics()
    assert after["service"]["requests_total"] - before["service"][
        "requests_total"
    ] == len(records), "external traffic"
    assert (
        after["service"]["requests_failed"] == before["service"]["requests_failed"]
        and not after["service"]["engine_failed"]
    )
    assert after["scheduler"]["active"] == after["scheduler"]["queued"] == 0
    (output / "metrics.json").write_text(
        json.dumps(dict(before=before, after=after), indent=2)
    )
    print("PASS", tag, len(records), "uncontended requests", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
