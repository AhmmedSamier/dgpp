# Combined API and vision validation — 2026-09-18

This integration check covers the complete workspace change over
`a5fcb189c28f`: native GLM image inputs and numerical parity, file inputs,
custom tools, JSON Schema constraints, request/usage/metrics compatibility,
and separate release/testing builds with PCRE2 packaging.

## Build and complete suite

The warnings-as-errors CI build completed for every target. All **107 CTest
tests passed**, with no failed or skipped CTest entries, in **550.52 seconds**.
This includes host/Python tests, CPU reference generators, CUDA kernels,
model-forward checks, graph/eager engines and loopback RDMA collectives.
The optional `glm_tp_forward_parity_real` subcase was not enabled; its log
explicitly skips it unless `DGPP_TP_REAL_MODEL` is set. The separate real-model
vision/fabric evidence is linked below.

```bash
cmake --build --preset ci -j 4
ctest --test-dir build-ci --output-on-failure -j 1
```

The [full CTest summary](2026-09-18-api-vision-integration/ctest.log) is retained.
The [vision numerical record](2026-09-18-glm-vision-numerics.md) documents
30 bitwise encoder comparisons, 39 identical serving responses and matching
rank operation streams. The [API extension review](../../docs/openai-api-validation.md)
records its host regression and sanitizer coverage.

The release server also built and installed successfully:

```bash
cmake --preset release
cmake --build --preset release -j 4
cmake --install build-release --prefix /tmp/dgpp-pre-push-install
/tmp/dgpp-pre-push-install/bin/dgpp-serve --version
```

The installed executable runs, its ELF is stripped and contains no debug
sections, and all three dependency/adaptation licenses are installed.
The complete suite also passed the bundled-PCRE2 install regression.

## Live release API checks

The newly built release server ran on two GB10 nodes with
`deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2.json` (GLM-5.3-Flash hybrid,
graph decode, MTP and four slots). The standard request-field smoke test
passed stop strings, streaming, multiple choices, forced/banned token bias,
validation errors and cold/hot prefix-cache usage.

That smoke test initially failed because it expected token log probabilities
inside a forced, unclosed reasoning block. Chat log probabilities now correctly
cover visible content only. The probe now uses plain completions for that
assertion and requires exactly six identical forced tokens, with matching
text whether log probabilities are requested or not. The cold-cache probe
also uses a fresh leading marker so rerunning it does not inherit an earlier
run's cached prompt. No serving behavior changed during this final check.

Four additional real-model requests passed: inline text-file extraction,
a required custom regex tool, a required custom Lark tool, and strict JSON
Schema with a string pattern and exact decimal multiple. The tools are
validated as returned data; the harness never executes them.

```bash
python3 scripts/serve_api_check.py 127.0.0.1 18080
python3 benchmarks/results/2026-09-18-api-vision-integration/api_extensions.py
python3 scripts/vision_api_check.py
```

[API smoke output](2026-09-18-api-vision-integration/api.log),
[extension check output](2026-09-18-api-vision-integration/extensions.log) and
[extension requests/responses](2026-09-18-api-vision-integration/extension-responses.json)
are retained.

The [release image checks](2026-09-18-api-vision-integration/vision.log) passed
streaming, multiple choices/images, resizing, image spans crossing prefill
chunks, and repeated/concurrent images without incorrect cache reuse.
Both ranks then stopped cleanly, with operation-stream MD5
`26545957fd226300b3f3c8edbc07fefe` on each rank. No serving world remains running.
The [metadata](2026-09-18-api-vision-integration/metadata.json) records the
release binary SHA-256 and deployment path.
