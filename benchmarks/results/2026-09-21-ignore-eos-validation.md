# `ignore_eos` request validation

PR #27 carries `ignore_eos` through the completion endpoints, scheduler and
fabric journal. Review found that its HTTP parser used `optional_field`,
which treats explicit JSON null as an omitted field. Both endpoints therefore
accepted `ignore_eos: null` with HTTP 200 and defaulted it to false, despite
the boolean-only contract.

The follow-up uses `body.find` so every supplied value reaches the boolean
check. It also removes a duplicate request assignment and documents the
extension in the API compatibility profile. Regression coverage checks null,
strings, numbers, arrays and objects on both completion endpoints, plus
omitted and explicit-false behavior on Chat Completions.

The extended `serve_ignoreEos` regression failed before the parser fix with
HTTP 200 for null. After rebuilding with the CI preset, both focused tests
passed. The full `serve_test`, `fabric_serve_test` and `scheduler_test` suites
also passed (3/3 suites, 118.83 seconds). Validation commands:

```bash
cmake --build build-ci -j 4 --target serve_test fabric_serve_test scheduler_test
DGPP_TEST_FILTER=serve_ignoreEos build-ci/serve_test
ctest --test-dir build-ci -R '^(serve_test|fabric_serve_test|scheduler_test)$' --output-on-failure -j 1
```

These are host tests with fake engines; they do not exercise GPU execution
or a deployed multi-node fabric.
