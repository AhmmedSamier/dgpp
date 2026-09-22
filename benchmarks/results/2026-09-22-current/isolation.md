# Solo/batched greedy text diagnostic

Source: `a3ed8994a34cf05f79e63dc1ad51ffd1f65c9f82`.
Deployment: GLM-5.3-Flash NVFP4/FP8, four nodes, four request slots,
default MTP. The benchmark's exact-text assertion failed with the default
dense-kernel setting. These controlled runs investigate that failure.
Follow-up: [GitHub issue #32](https://github.com/HawkBearPig/dgpp/issues/32).

## Procedure

[isolation_probe.py](isolation_probe.py) uses the first prompt from
`scripts/serve_load.py` as its probe. Other concurrent requests use distinct
prompts from that script. Requests use temperature zero, a 256-token output
cap and the server's default thinking mode. The prefix cache remains enabled.

After a 64-token warmup, the probe runs at concurrency
`1, 1, 2, 2, 3, 3, 4, 4, 1, 1`. Every result is compared with the first
solo response from the same server configuration. Returning to C1 after C4
checks whether the solo response remains repeatable after the batched work.
The JSON files contain full responses, usage, scheduler counters and hashes.

Each configuration starts a fresh server using the same release binary:

| `DGPP_DENSE_GEMV_ROWS` | C1 | C2 | C3 | C4 | Final C1 |
|---|---|---|---|---|---|
| 4 (default) | Identical | Identical | Different | Different | Identical |
| 8 | Identical | Identical | Identical | Identical | Identical |
| 256 | Identical | Identical | Identical | Identical | Identical |

Each cell covers two repetitions. In the default configuration, all C3/C4
responses first differ from the solo response at character offset 495.
All probe requests report 60 cached prompt tokens out of 62 prompt tokens.
Comparisons are within each configuration; changing the kernel setting can
also change the solo transcript.

## Interpretation

The [dense-kernel dispatch contract](../../../src/kernels/gemm.hpp) permits
different floating-point reduction orders across batch shapes. The engine
tests also allow a greedy decision to change when numerical differences
affect a near tie; see [glm_dsa_engine_test.cpp](../../../tests/cuda/glm_dsa_engine_test.cpp).
Increasing the GEMV row bound changes that dispatch, and restores exact
text agreement in this probe.

The results support numerical batch sensitivity as the explanation for this
failure. The original check asserts exact text equality; its failure alone
does not establish request-state contamination. These runs do not capture
logits, certify a numerical error bound, or establish isolation for every
workload. Operation-stream hashes check agreement in rank execution order
and are a separate property.

The performance campaign retains the default kernel setting. Diagnostic
timings are not substituted into the performance tables.

## Raw evidence

- [Default setting](raw/glm-flash-hybrid-w4/isolation-diagnostic/default/probe.json)
- [Row bound 8](raw/glm-flash-hybrid-w4/isolation-diagnostic/rows8/probe.json)
- [Row bound 256](raw/glm-flash-hybrid-w4/isolation-diagnostic/row-independent/probe.json)

Each adjacent `server/cluster.resolved.json` records the effective per-rank
environment. The adjacent command receipts and logs record startup, probe
execution and shutdown.
