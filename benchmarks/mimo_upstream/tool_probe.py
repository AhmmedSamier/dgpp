"""Dependency-free streaming tool regression. Generated calls are never executed."""

import json
import sys
import urllib.request
from benchmark import ROOT, URL, metrics


def main(tag):
    source = json.loads(
        __import__("pathlib").Path(__file__).with_name("tool_request.json").read_text()
    )
    dest = ROOT / "raw" / tag / "tools.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists(), "Preserve previous evidence"
    schemas = {
        t["function"]["name"]: t["function"]["parameters"] for t in source["tools"]
    }
    prompts = [
        source["messages"],
        [
            {
                "role": "user",
                "content": 'Call run_shell_command with command "pwd", directory ".", timeout 10000 and is_background false. Emit the tool call now.',
            }
        ],
        [
            {
                "role": "user",
                "content": 'Call list_directory with path "." and glob with pattern "**/*.py" and path ".". Emit both calls now.',
            }
        ],
    ]
    before = metrics()
    assert before["scheduler"]["active"] == before["scheduler"]["queued"] == 0
    records = []
    try:
        for index, messages in enumerate(prompts):
            request = dict(
                source, messages=messages, stream=True, temperature=0, max_tokens=8192
            )
            events = []
            calls = {}
            record = {"request": request, "events": events, "valid": False}
            records.append(record)
            with urllib.request.urlopen(
                urllib.request.Request(
                    URL + "chat/completions",
                    data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=300,
            ) as response:
                for line in response:
                    if (
                        not line.startswith(b"data: ")
                        or line.strip() == b"data: [DONE]"
                    ):
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    for choice in event.get("choices", []):
                        for item in choice.get("delta", {}).get("tool_calls", []):
                            call = calls.setdefault(
                                item["index"], {"name": "", "arguments": ""}
                            )
                            for key in call:
                                call[key] += item.get("function", {}).get(key, "") or ""
            assert calls, "No structured tool calls"
            parsed = {}
            for call in calls.values():
                schema = schemas[call["name"]]
                args = json.loads(call["arguments"])
                assert isinstance(args, dict)
                assert (
                    set(schema.get("required", [])) <= args.keys()
                ), "Missing required arguments"
                props = schema["properties"]
                if schema.get("additionalProperties") is False:
                    assert args.keys() <= props.keys()
                for key, value in args.items():
                    spec = props[key]
                    assert (
                        type(value)
                        is {"string": str, "integer": int, "boolean": bool}[
                            spec["type"]
                        ]
                    )
                    if "enum" in spec:
                        assert value in spec["enum"]
                    if "minimum" in spec:
                        assert value >= spec["minimum"]
                    if "maximum" in spec:
                        assert value <= spec["maximum"]
                    if "minLength" in spec:
                        assert len(value) >= spec["minLength"]
                    if key in schema.get("required", []) and key in {
                        "path",
                        "file_path",
                        "command",
                        "pattern",
                        "old_string",
                    }:
                        assert value.strip(), "Empty required argument"
                parsed[call["name"]] = args
            if index == 1:
                expected = {
                    "command": "pwd",
                    "directory": ".",
                    "timeout": 10000,
                    "is_background": False,
                }
                assert all(
                    parsed["run_shell_command"].get(k) == v for k, v in expected.items()
                )
            if index == 2:
                assert parsed["list_directory"]["path"] == "."
                assert (
                    parsed["glob"]["pattern"] == "**/*.py"
                    and parsed["glob"]["path"] == "."
                )
            record["valid"] = True
            print("PASS tool probe", index, parsed, flush=True)
    finally:
        after = metrics()
        dest.write_text(
            json.dumps(
                {
                    "records": records,
                    "before": before,
                    "after": after,
                    "tools_executed": False,
                },
                indent=2,
            )
        )
    assert after["service"]["requests_total"] - before["service"]["requests_total"] == 3
    assert after["service"]["requests_failed"] == before["service"]["requests_failed"]
    assert not after["service"]["engine_failed"]


if __name__ == "__main__":
    main(sys.argv[1])
