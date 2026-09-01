#!/usr/bin/env python3
"""Generates tests/data/glm_chat_template_goldens.jsonl — the differential
goldens for the GLM chat template (M6 Stage 3b, DESIGN §10).

The GOLDEN SOURCE is jinja2 3.1.2 configured EXACTLY like transformers'
chat-template environment (src/transformers/utils/chat_template_utils.py,
verified 2026-08-31):
  * trim_blocks=True, lstrip_blocks=True
  * extensions=[jinja2.ext.loopcontrols]  (the template uses {% break %};
    plain jinja2 rejects it)
  * the tojson OVERRIDE: json.dumps(ensure_ascii=..., indent=None,
    separators=None, sort_keys=False) — jinja's built-in tojson both
    escapes HTML (<, >, &, ') and rejects the ensure_ascii kwarg, so the
    override is REQUIRED to render this template at all.

The corpus is written with ensure_ascii=False (raw UTF-8) so minijson's
reader never meets surrogate pairs (it decodes single \\uXXXX only; the
tokenizer gate had to learn that lesson the hard way).

Each case carries the render globals as JSON ("kwargs"), the expected
render ("render"), and the token ids HF tokenizers 0.23.1 produces for
the render ("ids" — the template->tokenizer integration differential;
the C++ gate re-encodes with glm_tokenizer and compares).

Case selection (per DESIGN §10: English, Chinese, code, reasoning
blocks, tool calls — plus the structural edges the template branches
on): the plain/system/multimodal user paths; the reasoning-keep path
(trailing assistant) vs the reasoning-strip path (mid-conversation
assistant, and clear_thinking=false); reasoning_content as a field vs
think-markers inside content (the content.split path) vs both; the
close-without-open split edge; tool headers flat vs OpenAI-wrapped vs
defer_loading; aligned/sorted/unaligned tool-response blocks; string
vs JSON-serialized arguments; tool_reference entries direct and nested
in outputs; empty conversation and empty content.

Keyed by chat_template.jinja's FNV-1a-64 hash (header "template_hash"),
cross-referenced with the tokenizer revision ("tokenizer_revision").

Regenerating: /tmp/opencode/tokref/bin/python
    tools/gen_chat_template_goldens.py
(the venv pins jinja2 3.1.2 + tokenizers 0.23.1).
"""
import glob
import json
import pathlib
import sys

MODEL = "unsloth/GLM-5.3-Flash-FP8"


def fnv1a64(data: bytes) -> int:
    # The house hash (glm_loader.cpp / glm_tokenizer.cpp), not the FNV
    # standard basis.
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


# The template's reasoning markers, built by concatenation (heredocs and
# raw literals mangle them in tool transcripts; see the record).
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


def env():
    import jinja2
    e = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    # transformers' tojson override, character for character.
    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)
    e.filters["tojson"] = tojson
    return e


def weather_tool():
    return {
        "name": "get_weather",
        "description": "Get the current weather for a city (城市天气查询).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "minimum": 0},
            },
            "required": ["city"],
        },
    }


def call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


CASES = [
    ("simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("system_user_gen", {
        "messages": [
            {"role": "system",
             "content": "You are a concise geography assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    }),
    ("chinese_user_gen", {
        "messages": [{"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("code_user_gen", {
        "messages": [{"role": "user", "content":
                      "def add(a, b):\n    return a + b\nExplain this code."}],
        "add_generation_prompt": True,
    }),
    ("empty_conversation_gen", {
        "messages": [],
        "add_generation_prompt": True,
    }),
    ("empty_user_content_gen", {
        "messages": [{"role": "user", "content": ""}],
        "add_generation_prompt": True,
    }),
    ("effort_low", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "low",
    }),
    ("effort_high", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "high",
    }),
    ("multimodal_parts_gen", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe these in order."},
            {"type": "image_url", "image_url": {"url": "https://x/cat.png"}},
            {"type": "video_url"},
            {"type": "input_audio"},
        ]}],
        "add_generation_prompt": True,
    }),
    ("multi_turn_mid_reasoning_stripped", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_reasoning_content", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_think_markers", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + " Two plus two is four." + THINK_CLOSE +
                        " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("think_close_without_open", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": " partial thinking" + THINK_CLOSE + " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_both_reasoning_forms", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "reasoning_content": "Field form wins over markers.",
             "content": THINK_OPEN + " marker form" + THINK_CLOSE +
                        " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("clear_thinking_false_mid_turn", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
        "clear_thinking": False,
    }),
    ("no_generation_prompt", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "content": "4"},
        ],
    }),
    ("tools_flat", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [weather_tool(), {
            "name": "search_web", "description": "Search the web."}],
        "add_generation_prompt": True,
    }),
    ("tools_function_wrap", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Weather <queries> & conditions 天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "add_generation_prompt": True,
    }),
    ("tools_deferred", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [weather_tool(), {
            "name": "lazy_tool", "description": "Loaded on demand.",
            "defer_loading": True}],
        "add_generation_prompt": True,
    }),
    ("tool_call_aligned", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"city": "Paris", "days": 3, "factor": 0.5}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_unsorted", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_str_args", {
        # NOTE: arguments MUST be a dict (a bare JSON string crashes the
        # template at `_args.items()`); the `v is not string` branch is
        # for string VALUES inside it, rendered verbatim.
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"city": "Paris", "note": "asked twice"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_float_args", {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"scale": 1.0, "ratio": 0.5, "count": 3})]},
            {"role": "tool", "content": "18C",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_outputs_and_references", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": [
                {"tool_call_id": "call_1", "output": "18C, partly cloudy"},
                {"tool_call_id": "call_2", "output": [
                    {"type": "tool_reference", "name": "get_weather",
                     "tool_call_id": "call_9"}]},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_reference_direct", {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"})]},
            {"role": "tool", "content": [
                {"type": "tool_reference", "name": "get_weather",
                 "tool_call_id": "call_1"},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_no_prior_call", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_0"},
            {"role": "user", "content": "Thanks!"},
        ],
        "add_generation_prompt": True,
    }),
]


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1 else
                "tests/data/glm_chat_template_goldens.jsonl")
    snap = glob.glob(f"/home/user/.cache/huggingface/hub/"
                     f"models--{MODEL.replace('/', '--')}/snapshots/*/")
    if not snap:
        sys.exit(f"model snapshot for {MODEL} not found in the HF cache")
    tpl_raw = (pathlib.Path(snap[0]) / "chat_template.jinja").read_bytes()
    tok_raw = (pathlib.Path(snap[0]) / "tokenizer.json").read_bytes()
    template_hash = f"{fnv1a64(tpl_raw):016x}"
    tok_hash = f"{fnv1a64(tok_raw):016x}"
    if tok_hash != "700b4469fc43f23b":
        sys.exit(f"unexpected tokenizer revision {tok_hash} — the tokenizer "
                 "goldens are keyed to 700b4469fc43f23b")

    jinja_env = env()
    template = jinja_env.from_string(tpl_raw.decode("utf-8"))
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(snap[0] + "tokenizer.json")

    with open(out_path, "w", encoding="utf-8") as f:
        header = {
            "model": MODEL,
            "jinja2": jinja_env.jinja2.__version__ if hasattr(
                jinja_env, "jinja2") else "3.1.2",
            "template_hash": template_hash,
            "tokenizer_revision": tok_hash,
            "cases": len(CASES),
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for name, kwargs in CASES:
            rendered = template.render(**kwargs)
            if not rendered:
                sys.exit(f"case {name} rendered empty — generator bug")
            ids = tok.encode(rendered).ids
            f.write(json.dumps({"name": name, "kwargs": kwargs,
                                "render": rendered, "ids": ids},
                               ensure_ascii=False) + "\n")
            print(f"{name:42s} {len(rendered):5d} chars  {len(ids):4d} ids")
    print(f"wrote {out_path}: {len(CASES)} cases, template {template_hash}")


if __name__ == "__main__":
    main()
