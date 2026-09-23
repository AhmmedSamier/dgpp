# Integrated MiMo upstream preparation — 2026-09-23

The compact tool grammar, native MTP3 and both prefill work reductions were reviewed and validated as one branch. Shared-engine regression gates passed, and the three serving configurations completed the matched request panels. All performance switches remain opt-in.

## Source and shared-path review

- Upstream base `c6ca191`; preparation source `810144c19fdc`. The serving binary SHA256 is `111bd8768bb000ecf4e309b2056137fdb4e6c47ddb5912e0f8c49cdd93dad945`. Later edits are reporting, benchmark tooling and a comment-formatting correction, with no serving behavior changes.
- The seven optional `mtp_select_block` calls are guarded by a C++20 `requires` expression. Only MiMo implements the hook. Prefill/eager/scalar/batched draft entry points select head 0; chain entry points select index+1. The diagnostic forward also resets selection. Qwen and GLM-4.7 retain their previous instantiated paths.
- Native head selection is host dispatch/graph-construction state derived from the common draft schedule. Captured operations bind the selected weights and independent KV layers. Native heads gather backbone p-d and ignore the shared recursive hidden staging. Eager/graph rollback and prefix cuts preserve the full history ring; chain proposals do not modify it.
- New copies remain device kernels. History-only heads append K/V without extra attention/MLP/TP reductions. Compact XML transitions and non-strict required-key closure are limited to `xml_compact`; other grammar formats retain their existing paths. No blocking shared-hook issue was found in this review.
- Touched C++ ranges and new files were formatted with clang-format 18; benchmark Python was formatted with Black. The protocol signature comment is protected from prose reflow. [Operations](../../docs/operations.md#mimo-native-mtp-and-prefill-options) and the README describe normal use, memory implications and tradeoffs.

## Regression gates

All cross-build targets built with the `spark-cross` Release preset. The added native configuration unit case was rebuilt before host execution. These are cross-built ARM64 binaries run on Spark hardware, not a claim of a native Werror CI build.

- **22/22 host CTest entries passed**, covering unit, HTTP/serving, fabric-service/scheduler fakes and Python operational suites.
- **41/41 selected CTest entries passed**: four fixture writers and 37 GPU entries, 18 also labeled RDMA. Selection: `engine|decode|tp_test|mimo_port_test|mimo_native_mtp_test|forward_fixture`. Covers Qwen, GLM-4.7, GLM DSA, DeepSeek and MiMo, including Qwen BF12/FP8, wide MTP3 and compaction cases.
- The MiMo port matrix now tests one/three heads, checkpoint/BF12 weights, BF16/FP8 KV, and each prefill switch separately/together: 24 comparisons of 28 outputs. Cache-only results are bitwise equal; head-shape comparisons retain top-1 under the existing relative-L2 gate.
- All six MiMo engine cases passed again with both prefill switches enabled, including native depth-3 scalar/batched execution. Native history/state under Compute Sanitizer with both switches passed with **zero errors**.
- The new config case checks native three-head loading, rejection of incomplete one/two-head configurations with the field named, valid zero-head models, and return to the default one-head path.
- The broad gates ran on Spark-1, using synthetic and two-rank loopback worlds. Actual serving panels below ran across both physical Sparks. The earlier native-only state/engine/sanitizer gates ran on both machines. This is not four-node physical-fabric validation or the entire checkpoint-dependent CTest suite; the optional internal `glm_tp_forward_parity_real` case explicitly skipped because `DGPP_TP_REAL_MODEL` was unset.
- The first driver raced test-archive staging: four commands failed before tests started. Recovery restored serving. The second attempt verified staged binaries first and completed all gates; failed logs are retained.

## Matched real-model panels

All panels use checkpoint snapshot `3b38d063180c3e4aed9691fdc735f3d10b266ee4` on both ranks and the same restamped binary, C2, BF16 KV, BF12 weight residency, shared 131072-token pool, 1.5 GiB prefix budget, temperature zero and default thinking. Resident image caching is disabled with `DGPP_RESIDENT_CACHE=off`. Native-only leaves both prior prefill switches off; the integrated panel enables all three MiMo flags. Startup/warm graph capture is outside timing.

Client decode rate is `(completion_tokens - 1) / seconds_after_first_visible_content_or_reasoning_delta`. Short cases are medians of two requests; cold 67K is one request. These are bounded workload measurements, not universal speed claims.

| Configuration | Code | JSON | Prose | Maths | 67K decode | 31K cold TTFT | 67K cold TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| MTP1 control | 47.69 | 47.92 | 42.79 | 49.28 | 40.19 | 50.663s | 127.415s |
| Native MTP3 | 54.28 | 58.43 | 37.47 | 60.15 | 44.97 | 50.116s | 125.939s |
| Native MTP3 + both prefill switches | 54.33 | 58.50 | 37.31 | 60.31 | 45.39 | 49.841s | 125.429s |

Against native MTP3 alone, enabling both prefill switches reduced measured cold TTFT by about 0.4–0.6% on this panel. Decode differences were below 1%. These small deltas are integration evidence, not a robust standalone speedup claim; acceptance counters were identical between native and integrated configurations.

| Configuration | Exact content/reasoning matches to MTP1 | Differing request names |
|---|---:|---|
| MTP1 control | 19/19 | None |
| Native MTP3 | 19/19 | None |
| Native MTP3 + both prefill switches | 19/19 | None |

Each configuration passed **19 timing requests, four acceptance probes, eight prefix-reuse checks, three dependency-free streaming tool probes, 18 staggered arrival-order requests, 18 zero-delay arrival-order requests and six separate probability-trace requests**. Native-only acceptance reproduced the earlier positional counters. See the JSON for all acceptance denominators and per-position rates.

The fully integrated configuration additionally passed **3/3 actual custom-client probes**, with populated/typed arguments and no generated tool execution, and **2/2 exact JSON retrieval probes** at 30921, 67437 prompt tokens. This is not a broad SWE-bench/HumanEval evaluation or full-pool capacity stress.

After each of the three panels, both stopped ranks had matching nonempty operation-stream MD5s. The portable capture retains logs, binary hashes, configurations, selected flags and op streams.

## Concurrent-output investigation

The historical native C2 difference starts at generated token 81 (zero-based index 80), after identical reasoning and an identical content prefix. The native response exactly matches the solo response already produced by both original and final MTP1 controls. The newly rebuilt MTP1 timing panel reproduces all 19 historical MTP1 responses exactly.

Changing arrival timing on the **same MTP1 control** reproduces both code versions: staggered arrivals match solo, while back-to-back arrivals reproduce the alternative C2 result. At content token index 55, the probability traces show a target-model ranking reversal:

| Arrival | First token / log probability | Second token / log probability |
|---|---|---|
| Solo | ` pairs` / -0.952116 | ` lists` / -1.1250916 |
| Back-to-back | ` lists` / -0.9850826 | ` pairs` / -1.1295033 |

All six matched probability-trace requests have exactly identical emitted content-token logprob records across MTP1, native MTP3, and the fully integrated configuration: 1,419 records per configuration, including their top-two candidates. All three also reproduce the same variants in all 36 matched arrival-order requests. This establishes arrival-dependent target output without native MTP. Grouped prefills change GEMM shapes, and the dense lowering documents row-dependent arithmetic across its GEMV/GEMM boundary. Those are mechanisms consistent with the observation, not proof of a particular kernel cause. The ranking reversal should not be described as a proven near-zero tie. Temperature zero and speculative verification do not establish universal cross-batch greedy equivalence. The timing responses have a 256-token cap and are not complete coding solutions.

A separate native-MTP3 diagnostic raises `DGPP_DENSE_GEMV_ROWS` to 256 with both prefill switches off; all 18 responses and all six probability traces (1,419 content-token records) matched the default-threshold native run exactly, including the solo/burst distinction. Raising this threshold did not remove the arrival dependence. Details are in `diagnostic-summary.json`. It is not a timing panel and is not the deployed setting. The option remains capped by the model decode-row capacity and does not make all prefill shapes identical.

## Reproduction and final deployment

The optional custom-client replay used a clean `artifact-agent-orchestration` checkout at `d7142aa532c365e9665f118942bf468b9401b1de`. All request/analysis tools are now in [benchmarks/mimo_upstream](../mimo_upstream/README.md): timing, acceptance, prefix reuse, retrieval, tool fixture/SSE checks, optional actual-client replay, arrival ordering, probability traces, output grouping, summary generation, process provenance and stopped-rank log/MD5 capture. They accept endpoint/output/host settings rather than depending on private experiment directories. The portable summarizer reproduced all four historical summaries exactly.

The tested integrated build was left deployed on both Sparks at `http://192.168.0.171:30001/v1`, with native MTP3 and both prefill switches enabled, resident cache off, C2, BF16 KV, shared 128K capacity and 94 prefix snapshots within 1.5 GiB. Final identical binary/config/flag checks and eight prefix-reuse requests passed; the service ended idle with no engine or request failures. The previous 256K-per-request capacity remains unported.

Raw attempts, logs, configuration captures and SSE traces: `/mnt/benchmarks/dgpp-mimo-integrated-20260923/`. The dirty main checkout was not staged or reset.

## Maintainer tool-grammar follow-up

Review of PR #37 at `4dc7a2207b8747cda2aca2421f1455a0dc7c6745`, following
[issue #35](https://github.com/HawkBearPig/dgpp/issues/35), found two further
compact XML cases. A free field's mask admitted tokens whose decoded text
was empty, but consuming one marked the grammar dead and left subsequent
generation unconstrained. Free argument names also admitted duplicates,
which the parser would reject as literal content.

The follow-up keeps empty decoded tokens as no-ops and checks the used-key
ledger before completing any free argument name. It covers ordinary and
merged delimiters, undeclared names, names sharing a prefix, and ledger
reset between calls. Every allowed token in representative free-field
states must preserve an active grammar. Both new grammar tests failed on
the original PR code; all 22 grammar tests passed with the fix.

A separate MiMo parser regression preserves literal fallback for the
reported mixed-dialect prefix, complete functions missing the outer close,
partial arguments and duplicate parameters. This does not salvage malformed
calls or change capped requests' `finish_reason: length`.

Validation on native ARM64 with the `ci` preset (`DGPP_WERROR=ON`):

- Built `dgpp_serve_app`, all host test executables, and the MiMo port,
  native MTP and engine test executables successfully.
- All 22 host CTest entries passed with checkpoint-labeled tests excluded.
  The final unit executable includes both new grammar cases and the MiMo
  malformed-output regression.
- `git diff --check` passed.

The follow-up changes only host grammar behavior, regression tests and
documentation. GPU/fabric tests and real-checkpoint request panels were not
rerun while the local production service was active; the original PR's
hardware evidence is recorded above. The reported live failure frequency
has not been remeasured.
