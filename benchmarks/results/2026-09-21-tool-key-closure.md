# Tool key closure: PR #12 follow-up and merge validation

Base master: `c4b435750ae833c8fff340e40713af89de5a6e58`.
Original PR head: `1a9a3fe7e171aef45a2eb4eb03dc2f193d5f3c78`.
Issue [#11](https://github.com/HawkBearPig/dgpp/issues/11),
PR [#12](https://github.com/HawkBearPig/dgpp/pull/12).

The PR closes top-level tool argument names to declared `properties` unless
`additionalProperties` is explicitly `true`, and rejects repeated Qwen XML
parameter names as malformed tool content. Master was merged into the PR
branch first, retaining the independent GLM duplicate-key fix from #22. The
only conflict was overlapping parser documentation; the merged wording covers
distinct keys in every format.

## Reproduction

The PR's two unit regressions were applied to unchanged master first:

```text
[FAIL] tool_grammar_open_schema_closes_the_parameter_names
[FAIL] tool_parser_rejects_aRepeatedParameterName
```

Both pass with the PR. Before the follow-up, the grammar gate failed on all
four cached tokenizer/template corpora, exactly as reported in the
[review](https://github.com/HawkBearPig/dgpp/pull/12#pullrequestreview-5261790058):
GLM, GLM4 and GLM DSA refused `<arg_key>` in `tool_call_aligned`, and Qwen
refused the `parameter` token in `tool_call_multiline_and_nested_args`.

## Follow-up changes

Ten golden turns intentionally contain undeclared keys: three each in the
GLM, GLM4 and GLM DSA corpora (`factor`, `note`, and `scale/ratio/count`),
and one Qwen turn (`opts` and `flag`). The original PR record's assertion
that none did was corrected.

The reference JSONL corpora, rendered prompts and token IDs remain unchanged.
The grammar gates now copy a function schema and set
`additionalProperties: true` only when its golden call uses undeclared keys.
They derive that grammar through the same production helper and retain the
existing typed-value, required/named, single-call and EOS checks. A
complementary replay of the original schema must reject the first undeclared
parameter, verified against both `allows()` and the sampling mask. All ten
turns exercise both policies over the real tokenizers.

Additional unit coverage pins omitted/false/true/schema-valued
`additionalProperties`, empty `properties`, absent `properties`, and the
unchanged open default inside nested JSON objects. Qwen parser coverage checks
separated repeats, identical/different values, declared/undeclared/Unicode
names, literal-content preservation, reasoning, recovery after rejection,
and reusing the same key in separate valid calls. An existing PR assertion
was also split so a failed call-count check cannot index an empty vector.

Documentation now states that an empty declared property set is closed;
schema-valued extras close that set in non-strict mode; nested objects and
`response_format: json_schema` keep their existing JSON semantics; and with
free keys a duplicate block becomes content, making required/named calls
best-effort. The follow-up changes no production behavior beyond the PR's
two fixes.

## Validation

```bash
cmake --build --preset ci -j 4
CUDA_VISIBLE_DEVICES="" ctest --test-dir build-ci -L host -V -j 1
CUDA_VISIBLE_DEVICES="" build-ci/chat_template_test tests/data/glm_dsa_chat_template_goldens.jsonl
```

The full native build passed with warnings treated as errors. All 29 host
CTest entries passed in 166.97 seconds, including all nine checkpoint-backed
gates and 61 service cases. The unit binary reports 229 cases: 226 executed
and the same three GPU arena cases skipped in each of its two CTest entries.
The extra GLM DSA corpus passed all ten template/parser/grammar cases. No
checkpoint case was skipped. Changed C++ ranges pass clang-format 19.1.7,
and `git diff --check` passes. The reference JSONL corpora have no diff.

## Host performance and output comparison

Two standalone harnesses used the unit tests' fake Qwen vocabulary/decoder.
Baseline translation units came from the base master above and used the
candidate library's `-O2 -DNDEBUG -fPIC` settings; both versions linked the
same remaining libraries. The grammar harness constructs a state and computes
every mask while consuming a valid two-argument call. The parser harness
feeds a complete valid call with two or sixteen distinct parameters.

After the build and tests completed, three ABBA rounds ran pinned to CPU 0,
giving six samples per build. Each sample warmed up for 100 turns, then
timed 2000 grammar turns or 10000 parser turns. Median microseconds per turn:

| Case | Base | Fixed | Change |
| --- | ---: | ---: | ---: |
| Grammar, explicit closed keys | 26.789 | 26.370 | -1.57% |
| Grammar, explicit open opt-out | 24.549 | 24.093 | -1.85% |
| Parser, two parameters | 22.884 | 23.025 | +0.62% |
| Parser, sixteen parameters | 164.215 | 164.291 | +0.05% |

Every mask-word checksum and parser-output checksum matched between builds.
These host microbenchmarks show no material overhead for the tested valid
calls. The omitted-flag policy intentionally changes the allowed key set;
the unchanged explicit policies are the appropriate performance/output
controls. These are not model-throughput measurements.

## Fabric and accuracy scope

The original PR head already has
[recorded hardware validation](https://github.com/HawkBearPig/dgpp/pull/12#pullrequestreview-5259831093):
two-node Qwen NVFP4 and four-node GLM-5.3-Flash NVFP4. Each produced six clean
calls with unspecified or false `additionalProperties`; the explicit opt-out
and empty-property shapes were also exercised. API checks passed and op-stream
digests agreed across all participating ranks. That review's approval was
later withdrawn because the checkpoint gates failed, not because the wire
results failed.

Those are prior results on the original PR head, not a new fabric run of the
combined branch. Production owns the four GPUs; it was not restarted for
this follow-up. The changed policy is derived on rank 0 and its existing
`constrain_keys`/`keys` fields are journaled; no rank protocol or kernel changes
were introduced. Real-tokenizer gates validate the final branch's allowed
tokens and parsing. New GPU-throughput or model-quality measurements are not
claimed, and no published performance numbers were changed.
