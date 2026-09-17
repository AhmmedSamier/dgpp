# tool-eval-bench baseline, 2026-09-17

The existing DGPP deployment scored **90/100** (122/136 points) on
[SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
at commit `d84fce442aee49ffe433108901fd4216a784eefb`. All 69 standard
scenarios ran: 56 passed, 10 were partial, and 3 failed. The benchmark
excluded TC-64 from scoring after its pre-inference HTTP 400, leaving
68 scored scenarios and a 98.6% completion rate. No safety warning was recorded.

**This is a pre-fix baseline**, using the already-running service binary.
It does not measure the subsequent numeric-bound, combined tools/JSON, or
Qwen reasoning-effort changes. Those changes have separate host regression
coverage; the running deployment was not replaced for this measurement.

## Configuration

- Model: `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` on four DGX Sparks.
- Captured decode graphs, native MTP, four request slots, BF16 KV,
  786,432-token pool, 8 GiB prefix cache; full admission.
- Temperature 0, seed 42, `reasoning_effort: low`, one scenario at a time,
  eight turns maximum, 180-second request timeout, 16,384 completion tokens.
- One trial, standard fixtures, no hard mode. Run duration about 6m 41s.
- The benchmark's `openai` adapter label selects the compatible wire format.
  Its generated report calls the engine “OpenAI API”; the server was DGPP.
- Benchmark tracked source files were unchanged. Its `-dirty` metadata comes
  from the untracked output JSON and progress log in the benchmark checkout.

The executable hash, base checkout commit, engine settings and exact command
are retained in [provenance.json](provenance.json).

## Findings

| Scenario | Observed behavior |
|---|---|
| TC-64, simple schema compliance | HTTP 400 on a `number` property's `minimum`; excluded from the score. |
| TC-65/66/67/69, tools followed by structured output | Tool calls succeeded, then the final `response_format` request received HTTP 400 because tools were still present. Each received partial credit. |
| TC-68, schema violation resistance | The model respected the requested fields but wrapped the JSON in prose and a code fence. This case did not request API schema enforcement. |
| TC-43, missing required parameter | The model supplied an empty search query. |

The other six partial results concern unnecessary calculator use (TC-11/39),
error acknowledgement (TC-14), incomplete validation (TC-21), incomplete
market research (TC-52), and omissions in a long research chain (TC-62).
These are useful quality observations, not evidence that the API fixes have
been evaluated on the model. The original report's infrastructure-exclusion
heading mentions timeout/connection/5xx generically; TC-64 actually failed
with the HTTP 400 shown in its trace.

## Artifacts and reproduction

- [Full report and conversation traces](baseline-report.md)
- [Machine-readable results, including traces](baseline.json)
- [Progress log and request errors](baseline-progress.log)

Install the pinned benchmark into an isolated environment, then run against
the intended DGPP deployment:

```bash
uv venv /tmp/dgpp-tool-eval-venv
uv pip install --python /tmp/dgpp-tool-eval-venv/bin/python \
  'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@d84fce442aee49ffe433108901fd4216a784eefb'
/tmp/dgpp-tool-eval-venv/bin/tool-eval-bench run \
  --base-url http://127.0.0.1:18080 --backend openai --format openai \
  --seed 42 --temperature 0 --backend-kwargs '{"reasoning_effort":"low"}' \
  --timeout 180 --parallel 1 --label dgpp-glm-flash \
  --json-file results.json
```

For a focused post-deployment check, add
`--scenarios TC-64 TC-65 TC-66 TC-67 TC-68 TC-69`. A full repeat should keep
the same model, deployment settings, benchmark commit, seed and reasoning
level. Compare completion rates as well as scores; an excluded scenario
changes the denominator.

The accompanying source fixes build as `dgpp_serve_app` with warnings treated
as errors. Validation passed all 188 unit cases, the HTTP service and journal
suites, and the four GLM/Qwen checkpoint-template suites. Coverage includes
fractional/exponent bounds, exact boundary comparisons, optimized masks
against byte-by-byte simulation, both combined-output branches, tool-choice
controls, final turns with tool results, journal serialization, Qwen effort
rendering, and preserving numeric tool arguments through response parsing.
These are host and tokenizer checks, not a post-fix model benchmark.
