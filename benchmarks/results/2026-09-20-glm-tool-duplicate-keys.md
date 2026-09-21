# GLM tool parser duplicate argument keys

Base: `0e4f240f607a8152799afc57d5f3cb41820bed6b`.
Issue: [#22](https://github.com/HawkBearPig/dgpp/issues/22).

The GLM marker parser appended every completed `(key, value)` pair,
allowing one call to emit `{"city": "Rome", "city": "Oslo"}`. It now checks
the completed arguments before appending another. A repeated name uses
the existing malformed-block fallback: preserve the literal text as
content, emit no call for that block, and permit a subsequent valid call.
The check compares decoded names and allocates no additional key ledger.

This changes host-side response parsing. It leaves the grammar, sampling,
model arithmetic, scheduling and rank protocol unchanged. The Qwen XML
duplicate-key fix remains in [PR #12](https://github.com/HawkBearPig/dgpp/pull/12);
this change covers the GLM marker format independently.

## Reproduction and coverage

The new tests were built and run before changing the parser:

```text
[FAIL] tool_parser_glm_duplicateKeysFallBackToContent
[FAIL] tool_parser_glm_duplicateKeysDoNotLeakAcrossCalls
[FAIL] serve_toolCalls_duplicateGlmKeysBecomeContent
```

The HTTP failure returned a structured `get_weather` call containing both
`city` entries and `finish_reason: tool_calls`. Unit coverage includes
adjacent and separated repeats, identical and different values, declared,
undeclared and Unicode keys, exact content/reasoning preservation, valid
calls before and after rejection, and key reuse across separate calls.
The API regression exercises streaming and non-streaming responses with
an unconstrained fake engine: the literal block stays in content, no
structured call is emitted, and the response finishes as text.

## Validation

Native aarch64 GB10 `ci` build, with warnings treated as errors:

```bash
cmake --preset ci
cmake --build --preset ci -j 4
ctest --test-dir build-ci -L host --output-on-failure -j 1
build-ci/chat_template_test tests/data/glm_dsa_chat_template_goldens.jsonl
```

The full build passed. All 27 host CTest entries passed with no skips in
165.91 seconds, including the nine checkpoint-backed gates, 224 unit
cases and 61 service cases. Both new unit regressions and the streaming /
non-streaming API regression passed. The additional GLM-5.3 corpus
passed all ten template/parser/grammar checks. Changed C++ ranges were
formatted with clang-format 19.1.7; the formatting check and
`git diff --check` passed.

## Performance and accuracy scope

A host parser microbenchmark compared the unchanged base's `dgpp_text`
library with the fixed library, using the same `-O2 -DNDEBUG` harness and
the unit tests' fake marker decoder. Each sample warmed up for 100 turns
and timed 3000 turns, including parser construction and output checksums.
Three ABBA rounds produced six samples per build, pinned to the same CPU
after the build and tests finished. Median time per turn:

| Input | Base | Fixed | Change |
| --- | ---: | ---: | ---: |
| 64 content bytes, no call | 17.716 us | 17.792 us | +0.43% |
| One call, `city` and `days` | 2.568 us | 2.559 us | -0.37% |
| One call, 16 distinct keys | 153.331 us | 153.177 us | -0.10% |

Output checksums matched across all samples. The comparison shows no
material overhead for these valid inputs; it is a parser microbenchmark,
not a model-throughput measurement. Valid tool serialization and prompt
rendering are also covered by the real-tokenizer goldens. Model, sampler,
scheduler and GPU code are unchanged. GPU/RDMA suites,
live deployment benchmarks and model-quality evaluations were not rerun
for this response-parser change.
