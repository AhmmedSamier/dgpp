"""Read and summarize saved Chat Completions SSE streams."""
import argparse
import json
from pathlib import Path
import sys


def iter_events(path, *, skip_invalid=False, errors="strict"):
    """Yield JSON data events, using None for the [DONE] marker.

    Interrupted-stream recovery can opt into skipping truncated JSON. Normal
    reports remain strict so a malformed response is not silently accepted.
    """
    with open(path, encoding="utf-8", errors=errors) as stream:
        for line in stream:
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                yield None
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                if not skip_invalid:
                    raise


def committed_text(path):
    parts = []
    for event in iter_events(path, skip_invalid=True, errors="replace"):
        if event is None:
            continue
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            for field in ("reasoning_content", "content"):
                if delta.get(field):
                    parts.append(delta[field])
    return "".join(parts)


def no_tokens_after_error(path):
    # The failure drill checks compact wire-format fragments, including
    # partial JSON tails that cannot be read as complete events.
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    at = raw.find('"error"')
    tail = raw[at:] if at >= 0 else ""
    return '"content":"' not in tail and '"reasoning_content":"' not in tail


def summary(path):
    count, error, done, finish = 0, None, False, None
    for event in iter_events(path):
        if event is None:
            done = True
            continue
        if "error" in event:
            error = event["error"]
            continue
        count += 1
        for choice in event.get("choices", []):
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    print(f"stream: {count} chunks, error={error}, done={done}, finish_reason={finish}")


def run_lengths(kinds):
    runs = []
    for kind in kinds:
        if runs and runs[-1][0] == kind:
            runs[-1][1] += 1
        else:
            runs.append([kind, 1])
    return " ".join(f"{kind}x{count}" if count > 1 else kind for kind, count in runs)


def tools_summary(path):
    reasoning = content = ""
    calls, kinds = {}, []
    finish = usage = None
    for event in iter_events(path):
        if event is None:
            kinds.append("DONE")
            continue
        if event.get("usage") and not event["choices"]:
            usage = event["usage"]
            kinds.append("usage")
            continue
        choice = event["choices"][0]
        delta = choice["delta"]
        if "reasoning_content" in delta:
            reasoning += delta["reasoning_content"]
            kind = "R"
        elif "tool_calls" in delta:
            for tool in delta["tool_calls"]:
                call = calls.setdefault(tool["index"], {"id": None, "name": None, "args": ""})
                if "id" in tool:
                    call["id"] = tool["id"]
                    call["name"] = tool["function"]["name"]
                    kind = "Tstart"
                else:
                    kind = "Targs"
                call["args"] += tool["function"].get("arguments", "")
        elif "content" in delta:
            content += delta["content"]
            kind = "C" if delta["content"] else "role"
        else:
            kind = "final"
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        kinds.append(kind)
    print("chunks:", run_lengths(kinds))
    print("finish:", finish, "usage:", usage)
    print("reasoning:", repr(reasoning[:300]))
    print("content:", repr(content))
    for index, call in sorted(calls.items()):
        print("call", index, call["id"], call["name"], call["args"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("text", "summary", "tools", "no-tokens-after-error"))
    parser.add_argument("path")
    args = parser.parse_args()
    if args.command == "text":
        sys.stdout.write(committed_text(args.path))
    elif args.command == "summary":
        summary(args.path)
    elif args.command == "tools":
        tools_summary(args.path)
    else:
        return 0 if no_tokens_after_error(args.path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
