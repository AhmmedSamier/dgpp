# GLM Flash prefill memory investigation

The initial matrix attempt is [invalid](../invalid-swapping/INVALID.md). OS swap remains enabled; benchmark serving processes must report zero `VmSwap` throughout monitoring.

| Finding | Evidence |
|---|---|
| Serving processes used swap | At the interruption, rank 0 had about 6.85 GiB in `VmSwap`; peers had about 5.22 GiB each. The OS page-out counters were still increasing. [Process snapshot](../invalid-swapping/raw/glm-flash-hybrid-w4/refresh/default/context-256k-process-swap.json) |
| Cause | `GlmDiagnosticModel::session_run_rows` copied and retained prefill routing diagnostics even after `set_decode_route_traces(false)`. The prefill cursor accumulated these outputs across chunks. |
| Size | 42 MoE layers × (288 expert scores × 4 bytes + 8 selected IDs/weights × 8 bytes) = 51,072 bytes per input token per rank, or 12.47 GiB at 262,144 tokens. This host allocation was additional to the KV pool. |
| Fix | Prefill honors the existing trace setting, skipping diagnostic copies and retained vectors. The serving app disables traces for eager and graph execution. [Source patch](glm-prefill-trace-fix.patch), [committed fix](https://github.com/HawkBearPig/dgpp/commit/8f1c4cbcc1e98bb6e506c41a27e71126c2e6bed8) |
| Introduction | `b8174b800029` (2026-08-31) added unconditional accumulation across prefill chunks. HTTP serving inherited it in `75962284a61f` (2026-09-01). `af5bd1f6cc72` (2026-09-02) added the trace-disable switch only for decode, leaving prefill unaffected. |
| Configuration | KV capacity, prefix cache and request-slot settings are unchanged. |
| Regression | The new test fails on the original implementation and passes with the fix. It checks empty serving traces, retained diagnostic traces, and bitwise-equivalent chunked/grouped prefill, following decode and MTP outputs. [Before](trace-regression-before.log), [after](trace-regression-after.log) |
| Existing checks | Two resumable-prefill tests, two grouped-prefill tests and the four-rank graph/device-pick test passed. [Resumable](glm_tp_resumable_prefill.log), [grouped](glm_tp_group_prefill.log), [graph](glm_tp_device_pick_graph_loopback_matches_host_pick.log) |
| Real-model validation | Four-node GLM Flash hybrid, unchanged 768K BF16 KV pool and 22 GiB prefix cache: all three 261,614-token requests completed. Every observed serving-process `VmSwap` was zero (1,556–1,560 samples per rank); anonymous CPU memory remained about 1.11 GiB on rank 0 and 0.96 GiB on peers. The four operation streams matched. [Raw validation run](../raw/glm-flash-hybrid-w4/memory-validation/default/) |
| Validation-probe shutdown error | After the client finished at 03:29:56 UTC, the old rank-0 probe recorded zero swap through 03:29:58, then tried to read an exiting process's missing `VmRSS` field. The strict `memory-check.json` correctly rejected that run. Its timings are excluded from benchmark tables. The revised probe skips processes whose memory fields disappear; a live-process/zombie regression passes. [Check](memory-probe-exit-check.json) |

The benchmark probes record serving-process memory every second on every active rank. A nonzero `VmSwap` sample sends SIGINT; failed or missing per-rank memory evidence prevents publishing the mode and stops the campaign. System-wide swap allocation is recorded separately and does not invalidate a run by itself.

The old `STALLED` messages are diagnostics emitted after a collective has not completed within 500 ms. They do not establish a network fault. A 57.5-second counter sample recorded small packet-sequence and congestion-counter increases, but no retransmission, timeout, CRC or ingress-discard increases. [Counter record](../invalid-swapping/raw/glm-flash-hybrid-w4/refresh/default/context-256k-roce.json).
