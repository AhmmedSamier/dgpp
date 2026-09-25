# Hermes agent: "thinking-only / empty response" against Qwen3.8-Flash-Next (2026-09-25)

A forum user running Qwen3.8-Flash-Next on one Spark reported the Hermes agent
(NousResearch/hermes-agent) logging "Thinking-only response — prefilling to
continue", "Model returned empty after tool calls — nudging to continue",
"Empty response from model" and "Model produced reasoning but no visible
response after all retries". Reproduced here with the real agent; the cause is
the templates' `engine.default_max_tokens` of 256.

## Mechanism

- Hermes's custom / OpenAI-compatible provider profile sends **no
  `max_tokens`** ("the endpoint owns its generation default"; its docs state
  that custom endpoints receive no catalog-sized output cap) and no
  `reasoning_effort` unless configured.
- Every dgpp template except the two RadixArk Qwen ones set
  `default_max_tokens: 256`, and reasoning tokens count against that budget.
- The Qwen3.8-Flash-Next chat template thinks at `reasoning_effort` xhigh by
  default. On the first call of a Hermes session (a 12 967-token prompt with 24
  tools) the model spends 271–611 reasoning tokens before its first tool call,
  so the request retires `(steps)` at 256: `finish_reason: "length"`, empty
  `content`, 256 `reasoning_tokens`. That is the shape the agent's recovery
  ladder reacts to. Later calls in a session think 7–90 tokens and usually fit,
  so the failure is per-call and probabilistic at the default temperature.

## Setup

One Spark, `deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.json` (NVIDIA NVFP4,
MTP depth 1, decode graph, `kv_capacity` 65536, four slots), `build-ci` at
master ed0eba8 plus the NVMe-tier branch. A logging reverse proxy in front of
the API recorded every request body and response. Hermes 0.21.3 (checkout of
2026-09-14) ran from a separate machine with an isolated `HERMES_HOME`, a
`custom_providers` entry pointing at the proxy, `hermes chat --oneshot -v
--yolo` and a three-file work directory; the task asked it to list the
directory with the terminal tool, read the README and report.

## Results

Main-loop calls (system prompt + tools; the title-generation call is omitted):

| Server default | Call | Wire `max_tokens` | finish_reason | completion / reasoning tokens |
|---|---|---|---|---|
| 256 | first | none | `length` | 256 / 256 — no content |
| 256 | Hermes's recovery (reasoning off) | 8192 | `tool_calls` | 51 / 0 |
| 256 | after the tool result | none | `tool_calls` | 72 / 21 |
| 256 | final | none | `stop` | 129 / 7 |
| 256 (second task) | four calls | none | `tool_calls` ×3, `stop` | 179/66, 134/25, 89/41, 135/7 — the model happened to think briefly |
| 8192 | first (run 1) | none | `tool_calls` | 385 / 300 |
| 8192 | final (run 1) | none | `stop` | 192 / 87 |
| 8192 | first (run 2) | none | `tool_calls` | 336 / 295 |
| 8192 | two more (run 2) | none | `tool_calls`, `stop` | 96/45, 136/25 |

The first-turn request replayed with `max_tokens` 8192, four samples at the
server's sampling defaults: 271, 303, 450 and 611 reasoning tokens, every one a
tool call — all four would have been cut at 256. Four more samples with an
empty assistant turn and a "please continue" nudge inserted (the transcript an
older Hermes leaves behind after such a cut): 288–612 completion tokens, all
tool calls. No reasoning-only response that ended with `stop` was observed in
twelve samples; the budget cut is the only server-side trigger found.

Hermes 0.21.3 printed "⚠️ Response truncated (finish_reason='length') - model
hit max output tokens" and recovered by re-sending with `reasoning_effort:
"none"` and `max_tokens: 8192` (its fix of 2026-08-31); versions before that
appended the empty turn and re-thought against the same budget up to four
times, which is the longer ladder in the report.

## Change

All twelve templates that carried 256 now set `default_max_tokens: 32768`. The
server never clamps the default to the remaining capacity — a prompt plus the
default over `kv_capacity` is a 400 `context_length_exceeded` — so the YaRN
512K template's pool grows from 532 480 to 565 248 tokens (ceiling + 32 768 +
8 192): its memory plan moves from 52.97 to 53.53 GiB per rank (pool 7.01 →
7.45 GiB), and the two-full-stream variant becomes `--kv-capacity 1114112`
(62.86 GiB). Under full admission the budget is reserved at admission, so on the
64K single-Spark template a client that omits `max_tokens` can run one long
request at a time. The RadixArk templates keep their 65500.

Gates: `tests/python/portability_test.py` 19/19, `bootstrap_test.py` 22/22,
`cluster_config_test` (the YaRN pool clears the ceiling plus the default).
