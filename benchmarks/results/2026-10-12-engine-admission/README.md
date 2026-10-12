# Engine admission and mixed session protocol

Structural follow-up to issue #104 and PR #108. The launcher no longer selects
admission policy by model architecture or drafter name. Settings retain
automatic/on/off until the engine exists; the warm journal carries a resolved
boolean. Older records with no admission field still decode as disabled.

The common `SessionModel` now owns mixed-row planning, budgets, cursor positions,
snapshots, output slicing, MTP catch-up and suspension around the graph tail.
The graph engine owns global versus per-reader budgeting and selects mixed
execution through the explicit `MixedPrefillModel` contract. Layer executors
retain their state preparation, layer mathematics and input validation.

This is a structural refactor, **not universal backend enablement**. Qwen4Exp is
currently the only layer executor implementing mixed rows. Other executors,
including Qwen35 with DFlash, still require the corresponding layer execution
and correctness/performance verification. There is no blanket drafter exclusion
in the graph adapter. The contract requires preserving the features and state
needed by whatever drafter is configured.

## Verification

- All serving model adapters compile in `dgpp_serve_app`.
- Four host suites passed: scheduler, service, fabric and unit tests (172.91 s).
  New tests cover capability-based defaults, explicit overrides, monolithic mode,
  automatic/on/off settings round trips, legacy records and malformed modes.
- Ten GPU gates passed on idle hardware (36.22 s): mixed prefill with/without MTP,
  scheduled verification, compact sampling/fallback, 64 verification rows, images,
  and decode/snapshot restoration. The mixed graph tests resolve automatic
  admission from the engine. Invalid graph geometry is rejected before state
  advances.
- The first GPU attempt ran with production resident and hit the fixture
  loader's required 4 GiB headroom. Those results are retained in
  `gpu-memory-pressure.log`; the successful run stopped production first and
  restored it afterward. They are not counted as correctness passes.

## Matched serving comparison

`source.json` identifies both immutable executables and their runtime sources.
The reference is the runtime merged by PR #108 (`e17cf1eb`, equivalent runtime
at master `5503c189`). **Early admission is automatically enabled in both
executables.** This checks the refactor against the accepted implementation;
it is not the still-outstanding comparison against the pre-PR105 release.

One and four nodes; C4, MTP3, compact batches disabled, busy/idle budgets
256/2048. Two trials per workload, reversed workload order in the second trial;
control runs first on one node and candidate first on four. Timings and exact
response/accounting audits are recorded under `timing/`.

| Nodes | Workload | Refactor, s | Merged runtime, s | Throughput change |
|---|---|---:|---:|---:|
| 1 | mixed_short | 21.271406 | 21.367300 | +0.45% |
| 1 | solo_decode | 7.829827 | 7.813088 | -0.21% |
| 1 | solo_prefill | 7.632010 | 7.624073 | -0.10% |
| 1 | sustained_idle | 14.246485 | 14.257116 | +0.07% |
| 4 | mixed_short | 12.402807 | 12.419717 | +0.14% |
| 4 | solo_decode | 4.120513 | 4.104724 | -0.38% |
| 4 | solo_prefill | 5.998736 | 5.970746 | -0.47% |
| 4 | sustained_idle | 9.750335 | 9.733927 | -0.17% |

32 completed scenarios, 112 requests, 56 exact paired responses; zero output
differences or accounting failures. All four ranks' operation-stream hashes
match. Before/after probes verify executable hashes and zero process/cgroup
swap. Mean throughput changes are -0.47% through +0.45%, within the scale of
run-to-run variation seen here. Two trials do not establish a tight statistical
equivalence bound, and these checks do not resolve the earlier workload-dependent
regressions against the pre-admission release.

Production was restored and verified: original configuration, original binary
hash `5b457ffed0f6290344c68609370e4b58ae893f52b4713ae39362babefea15542`
on all four ranks, healthy HTTP. This refactor was not deployed to production.

Reproduce the serving campaign with `campaign.py` in the neighboring
`2026-10-12-shared-prefill-decode` directory, `--control-early`, the candidate and
control paths from `source.json`, `--worlds 1 4 --idle-budgets 2048 --trials 2
--scenarios solo_decode solo_prefill mixed_short sustained_idle`, and the
production configuration/binary arguments recorded in the command receipts.
