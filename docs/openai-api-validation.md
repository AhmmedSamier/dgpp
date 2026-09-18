# Chat API extension validation — 2026-09-18

This review covers file inputs, custom tools, model-native reasoning effort,
and the expanded JSON Schema constraints. The contract was checked against
the official [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[file input guide](https://developers.openai.com/api/docs/guides/file-inputs),
[custom tool guide](https://developers.openai.com/api/docs/guides/function-calling),
and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
DGPP's supported profile and explicit limitations are in
[OpenAI compatibility](openai-compatibility.md).

## Added regression coverage

Sixteen new test groups and an expanded DeepSeek prompt test cover the
following cases. Parameterized groups exercise many inputs per test.

| Area | Cases and assertions |
| --- | --- |
| File decoding | Invalid base64 alphabet, length, padding placement and padding bits; valid one/two/three-byte tails; data URIs; Japanese and emoji content; Unicode-escaped JSON field/type names still trigger preprocessing. |
| Multipart uploads | Boundary prefixes inside file contents remain literal bytes, including false closing delimiters; MIME parameter names are case-insensitive; malformed media types, truncated delimiters, invalid dispositions and duplicate parameters produce client errors. |
| File limits and storage | Exact byte limits are accepted; one byte over is rejected; per-request decoded/text totals are enforced; deletion reclaims quota; repeated deletion fails; twelve concurrent uploads cannot exceed a five-file quota. |
| Invalid file references | Wrong JSON shapes/types, missing data, simultaneous ID/data, path-like IDs and invalid filenames are rejected; malformed persisted metadata fails cleanly instead of dereferencing missing fields or converting out-of-range numbers. |
| Custom tool validation | Missing/invalid names and descriptions, bad format/grammar/syntax, named selection errors, duplicate allowed tools, required history IDs and string inputs; an empty history input string remains valid. |
| Custom tool responses | Mixed function/custom calls and `n=2`; IDs are unique across choices; Unicode, quotes, slashes, tabs and newlines survive exact SSE reconstruction; start/finish/usage events occur the expected number of times. Malformed generated custom arguments return an error in both one-shot and streaming responses. |
| Reasoning settings | Invalid effort types/names, conflicting aliases, duplicate template keys, agreeing aliases producing exactly one native switch, switch-only models, and models without controls. All effort levels are checked against GLM, Qwen and DeepSeek checkpoint-native mappings; `none` disables thinking only where supported. |
| HTTP lifecycle | File preprocessing preserves pipelined response order; clients can disconnect during preprocessing without preventing subsequent health requests. Existing PDF worker responsiveness, limits and shutdown checks remain in the full service suite. |
| Numeric/schema constraints | 3,507 decimal/boundary combinations checked against an independent integer-arithmetic oracle; oversized/fractional array bounds; empty enum intersections; local pointers through arrays; invalid indices/escapes, external references and reference cycles. |
| Strings and grammar | Unicode and surrogate handling; valid/invalid boundaries for all nine formats, including leap years and timezone-adjusted leap seconds; the official arithmetic Lark example; escaped literals; nullable groups, right recursion and zero repetitions; malformed/unproductive grammars and unsupported priorities/directives. |

The existing mask tests compare optimized token masks against exhaustive
per-token simulation, including random walks. The numeric oracle independently
checks decimal divisibility rather than copying the implementation's decimal
algorithm. HTTP tests use real sockets and SSE framing with a deterministic
fake model engine. Checkpoint tests use actual tokenizer/template metadata.

## Defects found and corrected

- File detection skipped valid requests whose `file` fields/types used JSON
  Unicode escapes. It now falls back to parsing escaped requests.
- Multipart parsing accepted invalid media-type prefixes and duplicate
  parameters, confused boundary-like file contents with delimiters, and
  incorrectly treated MIME parameter names as case-sensitive.
- Persisted file metadata could dereference missing fields or convert invalid
  byte counts. Startup now validates metadata against its stored file.
- Array bounds could truncate or overflow; local JSON Pointers did not
  traverse array indices. Both now validate before conversion/traversal.
- Scalar enum members were not intersected with string constraints and
  `multipleOf` at compilation. Impossible intersections now fail before
  generation is admitted.
- Lark character escapes could change the accepted language; nonproductive
  and unsupported left-recursive cycles could compile without a usable
  language. Escapes and recursion are validated explicitly.
- Time formats accepted leap seconds at invalid UTC times. Validation now
  accounts for the supplied timezone offset.
- Custom selection errors named the function field; custom history did not
  require its call ID. Errors now identify `tool_choice.custom.name`, and
  history validates the ID.
- Agreeing thinking aliases emitted duplicate native globals. They now
  produce one setting, while conflicting aliases/duplicate keys fail.

Sanitizers also identified two test-harness defects: an existing safetensors
fixture read unaligned mapped bytes through typed pointers, and an HTTP test
kept JSON string views into a destroyed temporary response substring. The
fixture now copies bytes into aligned locals; the HTTP parser borrows from
the still-live response string.

## Verification

On this ARM64 workspace, the warnings-as-errors CI build passed all ten
selected CTest suites: unit, service, HTTP transport, fabric service,
scheduler, four checkpoint template variants and the DeepSeek prompt suite.
The unit binary contains 204 cases and the service binary contains 55 cases.

The complete 204-case unit suite and 55-case service suite passed with
AddressSanitizer and with UndefinedBehaviorSanitizer, with no remaining
reported errors. The final edge-case unit groups also passed AddressSanitizer
after the last test refinements. The final MIME parameter adjustment passed
all seven file-input tests in CI, ASan and UBSan builds; the focused HTTP
extension and edge-case groups also passed in all three builds after that
adjustment (nine service cases per build).

The release server rebuilt successfully and its ELF is stripped, with no
debug sections. These changes have not been deployed by this review.

Reproduce the main regression run after configuring the CI preset:

```bash
cmake --build build-ci -j2 --target unit_tests serve_test http_server_test fabric_serve_test scheduler_test chat_template_test dsv41_prompt_test
ctest --test-dir build-ci --output-on-failure -j2 -R '^(unit_tests|serve_test|fabric_serve_test|http_server_test|scheduler_test|chat_template_test|qwen_chat_template_test|glm4_chat_template_test|glm_dsa_chat_template_test|dsv41_prompt_test)$'
```

For quick focused checks:

```bash
DGPP_TEST_FILTER=edge build-ci/unit_tests
DGPP_TEST_FILTER=serve_edge build-ci/serve_test
```

For sanitizer checks, configure/build each sanitizer preset, then run the
service binaries sequentially because the HTTP fixtures share fixed ports:

```bash
cmake --preset asan
cmake --build build-asan -j2 --target unit_tests serve_test
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 build-asan/unit_tests
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 build-asan/serve_test
cmake --preset ubsan
cmake --build build-ubsan -j2 --target unit_tests serve_test
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 build-ubsan/unit_tests
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 build-ubsan/serve_test
```

## Confidence boundary

These checks validate request acceptance/rejection, error fields, native
prompt controls, wire output, constrained decoding and host lifecycle behavior.
They do not establish model quality, large-context GPU performance, or
multi-rank deployment behavior. No live OpenCode request or production
deployment was exercised in this review.

The subsequent [combined integration check](../benchmarks/results/2026-09-18-api-vision-integration.md)
passed the complete 107-test suite and live two-node release checks for
file inputs, custom tools, structured output and image inputs.

This is conformance evidence for DGPP's documented supported profile, not
complete OpenAI platform parity. In particular, PDF inputs provide extracted
text without page vision/OCR; Lark and JSON Schema retain the documented
unsupported constructs. The format tests cover common and boundary inputs,
not every possible RFC string. A deployed multi-rank smoke test remains the
next validation layer for actual generated responses.
