"""Parent-only API check through the user's actual client/recording transport.

Generated tools are validated and recorded, never executed.
Run with artifact-agent-orchestration/.venv/bin/python.
"""

import asyncio
import json
import pathlib
import sys
import time

from benchmark import ROOT, URL
import os

HARNESS = pathlib.Path(os.environ["DGPP_HARNESS_ROOT"])
sys.path.insert(0, str(HARNESS / "smoke-client"))
from inference_runtime.conversations.conversation import (
    Conversation,
    Message,
    ToolCall,
    ToolDefinition,
)
from inference_runtime.inference_client.chat_completions import ChatCompletionsClient
from smoke_client.recording_transport import RecordingTransport
from benchmark import metrics

SOURCE = pathlib.Path(__file__).with_name("tool_request.json")


def validate(arguments, schema):
    assert isinstance(arguments, dict)
    assert all(
        k in arguments for k in schema.get("required", [])
    ), "missing required argument"
    properties = schema["properties"]
    if schema.get("additionalProperties") is False:
        assert not set(arguments) - set(properties), "unknown arguments"
    for key, value in arguments.items():
        spec = properties[key]
        expected = {"string": str, "integer": int, "boolean": bool}[spec["type"]]
        assert type(value) is expected, "wrong type for " + key
        if "enum" in spec:
            assert value in spec["enum"]
        if "minimum" in spec:
            assert value >= spec["minimum"]
        if "maximum" in spec:
            assert value <= spec["maximum"]
        if "minLength" in spec:
            assert len(value) >= spec["minLength"]
        if (
            key in schema.get("required", [])
            and key in {"path", "file_path", "command", "pattern", "old_string"}
            and expected is str
        ):
            assert value.strip(), "empty required string: " + key


async def main(tag):
    destination = ROOT / "raw" / tag / "actual-harness"
    destination.mkdir(parents=True, exist_ok=False)
    original = json.loads(SOURCE.read_text())
    schemas = {
        t["function"]["name"]: t["function"]["parameters"] for t in original["tools"]
    }
    tools = [
        ToolDefinition(
            name=t["function"]["name"],
            description=t["function"]["description"],
            input_schema=t["function"]["parameters"],
        )
        for t in original["tools"]
    ]
    metadata = {
        "calls": [],
        "local_model_reasoning_recorded": True,
        "tools_executed": False,
    }
    transport = RecordingTransport(destination, metadata, time.monotonic())
    client = ChatCompletionsClient(
        endpoint=URL + "chat/completions",
        model=original["model"],
        temperature=0,
        max_tokens=8192,
        timeout_seconds=300,
        transport=transport,
    )
    before = metrics()
    assert before["scheduler"]["active"] == before["scheduler"]["queued"] == 0
    results = []
    prompts = [
        original["messages"],
        [
            {
                "role": "user",
                "content": 'Call run_shell_command with command "pwd", directory ".", timeout 10000 and is_background false. Emit the tool call now.',
            }
        ],
        [
            {
                "role": "user",
                "content": 'Call list_directory with path "." and glob with pattern "**/*.py" and path "." to inspect the workspace. Emit both tool calls now.',
            }
        ],
    ]
    try:
        for number, messages in enumerate(prompts):
            answer = await client.complete(
                Conversation(
                    model=original["model"],
                    items=[Message(**m) for m in messages],
                    tools=tools,
                )
            )
            calls = [item for item in answer.items if isinstance(item, ToolCall)]
            record = {
                "number": number,
                "answer": answer.model_dump(mode="json"),
                "valid": False,
            }
            results.append(record)
            assert calls, "no tool calls returned"
            for call in calls:
                assert call.name in schemas
                validate(call.arguments, schemas[call.name])
            if number == 1:
                assert any(
                    c.name == "run_shell_command"
                    and c.arguments.get("command") == "pwd"
                    and c.arguments.get("directory") == "."
                    and c.arguments.get("timeout") == 10000
                    and c.arguments.get("is_background") is False
                    for c in calls
                )
            if number == 2:
                assert any(
                    c.name == "list_directory" and c.arguments.get("path") == "."
                    for c in calls
                )
                assert any(
                    c.name == "glob"
                    and c.arguments.get("pattern") == "**/*.py"
                    and c.arguments.get("path") == "."
                    for c in calls
                )
            record["valid"] = True
            print(
                "PASS actual harness request",
                number,
                [(c.name, c.arguments) for c in calls],
                flush=True,
            )
    finally:
        await transport.drain()
        after = metrics()
        (destination / "validation.json").write_text(
            json.dumps({"results": results, "before": before, "after": after}, indent=2)
        )
    assert after["service"]["requests_total"] - before["service"]["requests_total"] == 3
    assert after["service"]["requests_failed"] == before["service"]["requests_failed"]
    assert not after["service"]["engine_failed"]


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
