# PR #42 tool rendering review — 2026-09-23

The review of `a0183a3` found two regressions in prompt filtering. Accepted
flat tool definitions became empty objects, causing the DeepSeek renderer
to reject requests with HTTP 400. DeepSeek namespaces were removed from
prompts while decoding constraints still required qualified names such as
`search::lookup`.

The follow-up preserves the supported flat shape and member order, and
retains DeepSeek namespace metadata on both tool and function objects.
Unrelated metadata is still filtered, including function-level `response`
schemas. Nested parameter schemas and the raw compatibility switch retain
their behavior. See [API compatibility](../../../docs/openai-compatibility.md#function-tool-rendering).

## Validation

The two original HTTP/renderer reproductions failed on `a0183a3` and passed
on its parent. Both also passed with `DGPP_TOOLS_RAW=1` on `a0183a3`.
The permanent regression tests reproduced both failures before the fix.
After the fix, all four `serve_tools_` cases passed with filtering enabled.
They cover flat definitions with and without `type`, field order, nested
properties, and namespace placement on wrappers, functions and flat
definitions under required, named and allowed-tool selection.

The host build uses Release with `DGPP_WERROR=ON`. These checks use fake
engines and the production DeepSeek renderer; no GPU inference or BFCL
benchmark was run. The author's prompt-token measurements were not
remeasured during this follow-up.
