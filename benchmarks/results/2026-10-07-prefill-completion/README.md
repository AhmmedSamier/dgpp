# Prefill benchmark completion

Targeted reruns after the [prefill efficiency investigation](../2026-10-07-prefill-followup/README.md). All 46 groups across ten launches passed the [measurement audit](measurement-audit.json). The [plan](plan.json) covers seven affected configurations; unchanged paths retain their earlier results.

| Configuration | Measurements |
|---|---|
| GLM Flash NVFP4/FP8, 2 nodes, 160K FP8 KV | C1/C2/C4/C8 serving; context buckets 0–128K |
| GLM Flash NVFP4/FP8, 2 nodes, 256K FP8 KV | C1/C2/C4/C8 serving; ~32K cold prefill; context buckets 0–256K |
| Full GLM int4/int8, 4 nodes, 256K FP4 KV | C1/C2/C4/C8 serving; context buckets 0–256K; plain/MTP2/MTP3 decode |
| MiMo MXFP4/FP8, 2 nodes, 128K BF16 KV | C1/C2/C4/C8 serving; context buckets 0–128K |
| MiMo MXFP4/FP8, 2 nodes, 256K FP8 KV | C1/C2/C4/C8 serving; context buckets 0–256K |
| MiMo MXFP4/FP8, 4 nodes, 128K BF16 KV | C1/C2/C4/C8 serving; ~2K/~8K/~32K cold prefill; context buckets 0–128K |
| MiMo MXFP4/FP8, 4 nodes, 1M BF16 KV | C1/C2/C4/C8 serving; ~2K/~8K/~32K cold prefill; context buckets 0–512K |

The [scope review](scope.json) records why other documented families and KV formats retain their existing measurements.

Every default configuration uses MTP1. GLM Flash has a 2,048-token idle prefill budget and a 256-token busy budget. KV formats, capacities, request slots and prefix-cache budgets match the recorded deployments. Previously validated diagnostic cold-prefill results remain for the other cells; these reruns fill the gaps and replace affected throughput/context measurements.

## Source and method

The [manifest](manifest.json) records the tested candidate binary (`616d35a9…`), source hashes, harness hashes and clients. It includes both the [FP4 conversion patch](../2026-10-07-prefill-followup/fp4-conversion-experiment/source.patch) and the [FP8 projection-dispatch patch](../2026-10-07-prefill-followup/mimo-dispatch-fix/source.patch) on `90ebbf1`. The original release executable is preserved; its hash is used only by the launcher's integrity guard. Each launch records the actual candidate hash as its `binary_sha256`.

- Serving: five prompt classes, three repetitions at C1/C2/C4/C8, 256 output tokens, greedy sampling.
- Decode modes: the same classes and repetitions at C1.
- Cold prefill: three distinct uncached GSM8K probes per requested length. Earlier diagnostic cells use two probes.
- Context: deterministic parcel documents, one cold request followed by two repetitions, 256 output tokens. The 0 bucket is a minimal prompt. No profiler is active.
- Validation: request usage and timing counters must reconcile; operation-stream hashes must match across all ranks after a clean shutdown. Every serving process and its cgroup must report zero swap throughout monitoring; OS swap stays enabled.

A launch contributes results only after all of its checks pass. Each `raw/<deployment>/refresh/<mode>/record.json` bundles the commands, configuration, resolved deployment, binary provenance, memory audit and rank identity. The original request JSON files remain separate and are bound by SHA-256. Full runtime logs and telemetry remain local; the compact records retain their audit summaries and telemetry hashes.

## Measurement variation

All repetitions contribute to the reported medians; slower repetitions were not discarded. For example, GLM Flash's two-node 160K configuration measured 61.01 / 66.52 / 60.93 engine tok/s in its three C4 JSON repetitions, with identical outputs and decode counters. The cause of that timing variation was not established. [Request results](raw/glm-flash-hybrid-w2/refresh/default/greedy.json).

The [telemetry summary](telemetry-summary.json) records one software thermal-slowdown sample during that configuration's later 128K context group, outside its serving sweep. No other sampled thermal flags appeared in the ten launches. Timings are unadjusted. GPU status was sampled approximately every five seconds; full captures remain local.

## Reproduce and update

The runner uses the original clients and validation routines from the recorded matrix. It launches one configuration at a time, checks free disk space before each launch, and preserves failed attempts separately. MiMo is pinned to revision `5711b268169967567844e1e560e8a3966da959b1`. It was absent on every node at the start; its downloaded checkpoints and dedicated resident cache are removed after the MiMo tests finish. The original GLM Flash service is restored afterward.

```bash
python3 benchmarks/results/2026-10-07-prefill-completion/run.py --plan
python3 benchmarks/results/2026-10-07-prefill-completion/run.py
python3 benchmarks/results/2026-10-07-prefill-completion/results.py --require-complete
python3 benchmarks/results/2026-10-07-prefill-completion/update_document.py
python3 benchmarks/results/2026-10-07-prefill-completion/update_document.py --check
```

The table updater first incorporates the earlier diagnostic cold-prefill results, then overlays these validated reruns. Run it after the original matrix's `summarize.py` to reproduce the current document. It preserves unrelated deployment rows and historical quality results.

## Completion and service state

All ten launches passed request-counter, rank-identity and memory checks. DGPP process and cgroup swap remained zero; OS swap stayed enabled. The downloaded MiMo checkpoint and dedicated resident caches were removed on all four nodes after its tests finished. The original four-node GLM Flash service was restored with its original configuration and binary; API, inference, rank and logout-persistence checks passed.

- [Storage cleanup](cleanup.json) and [preservation audit](storage-preservation-audit.json)
- [Service restoration record](restoration/record.json)
- [Telemetry summary](telemetry-summary.json)
- [Evidence retention](retention.json) and [retained-package validation](retained-evidence-audit.json)
