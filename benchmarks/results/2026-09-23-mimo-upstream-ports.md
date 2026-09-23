# MiMo ports onto upstream master — 2026-09-23

Upstream provides a substantially faster long-context prefill foundation on our two Sparks. Porting the local compact-tool grammar repaired a failure in the actual custom harness. The two isolated prefill work reductions provide a further small, measured improvement when enabled together. They remain opt-in; this branch is the first migration increment, not a replacement for the existing native-MTP3/256K-per-request service.

## Source and configuration

- Upstream baseline: `c6ca191863d6`; saved custom implementation: `808e919` (serving code `03a7f5f`).
- Performance candidates: `1fb11dc`; compact grammar fix: `fb92284`. Every candidate and the final control used the same frozen `fb92284` binary. Later commits add tests, scripts and records.
- Baseline binary SHA256: `490b8bf2e128a4a28cb2cffa66375244fdeb60917cffc6b46981a1200e14ca47`.
- Candidate/control SHA256: `1f114a095df47ae4074caa0b31b411010e7bd8aa9ba154f5e204ccb1bedf82d2`.
- Two Sparks, C2, native MTP **depth 1**, BF16 KV, lossless BF12 storage for eligible BF16 matrices, shared **131072-token pool**, 4096 small prefix metadata slots. Native FP8 dense and MXFP4 expert formats remain upstream's.
- Upstream MiMo rejects prefill-budget controls. An initial startup with the old 256/2048 budgets was refused; no timing samples came from that attempt. The measured runs use upstream scheduling defaults.
- Our saved deployment uses MTP3, two **262144-token per-request** caches, sliding-window rings and four prefix slots. Capacity and scheduling are not equivalent. It was restored after the experiment.

## Prefill measurements

TTFT is client time to the first streamed content/reasoning delta. Identical fixed prompts, temperature 0, no competing requests, 19 timing requests per panel. The 7K/31K entries are medians of two cold requests; each 67K cold entry is one request, followed by two warm repeats. Startup and the explicit warmup are excluded. This is a bounded panel, not a broad workload average.

| Configuration | 7,213 cold | 30,876 cold | 67,396 cold | 67K warm |
|---|---:|---:|---:|---:|
| Unmodified upstream | 11.042s | 50.810s | 127.032s | 297.2ms |
| Same-source control (tool fix only) | 11.018s | 50.682s | 127.356s | 306.2ms |
| Last-row head only | 10.880s | 50.179s | 126.236s | 292.3ms |
| Cache-only history | 10.620s | 49.058s | 123.484s | 302.8ms |
| Both performance ports | 10.534s | 48.712s | 122.839s | 310.7ms |

Compared with the final same-source control, the combined TTFT reductions are 4.39% at 7K, 3.89% at 31K, 3.55% at 67K. Cache-only history supplies most of the gain; the last-row head contribution is smaller. Warm-prefix timings do not establish a win.

## Decode measurements

Client decode rate is `(completion_tokens - 1) / time_after_first_streamed_delta`, with the model's default thinking behavior. It is not upstream's published engine-counter rate. These ports target prefill; no material decode improvement is claimed. C2 timings are retained in JSON but are not used to claim a gain because request arrival order varied.

| Configuration | Code C1 | JSON C1 | Prose C1 | Math C1 |
|---|---:|---:|---:|---:|
| Unmodified upstream | 47.47 | 47.87 | 42.65 | 49.22 |
| Same-source control (tool fix only) | 47.59 | 47.78 | 42.68 | 49.27 |
| Last-row head only | 47.50 | 47.87 | 42.59 | 49.15 |
| Cache-only history | 47.57 | 47.82 | 42.77 | 49.28 |
| Both performance ports | 47.65 | 47.95 | 42.78 | 49.22 |

For context, our saved implementation recorded 7.024/56.734/222.210s at these same cold lengths and C1 code/JSON/prose/math rates of 44.50/48.33/28.09/48.40 tok/s. Upstream improves the long case considerably but remains slower at 7K. This is an implementation comparison with different MTP/cache configurations, not an isolated attention-kernel A/B.

## Correctness and limits

- Both Sparks passed the synthetic port regression, upstream decode tests, two-rank loopback engine tests and Compute Sanitizer memcheck. The expanded port test also exercised BF12-only storage with actual matrices packed and their BF16 storage released, for both BF16/FP8 KV. Cache-only logits were bitwise identical. Changed head shapes had worst relative logit L2 about `1.3e-7` (gate `1e-5`) with unchanged top-1 and hidden states.
- All 256 unit tests and 63 serving tests passed for the compact grammar port. Upstream's newer schema/reference support and model-opened thinking were retained.
- Unmodified upstream returned malformed XML as text on the first actual harness probe: **0/1 attempted passed, two unattempted**. Its timing panel and four acceptance probes had completed; prefix reuse was not run after the harness failure.
- Each repaired candidate and the final control passed **3/3 actual harness probes** and **8/8 prefix branch/reuse checks**. Required path/pattern/command fields and typed shell arguments were validated; generated tools were never executed.
- The combined version passed exact JSON retrieval of all three seeded values at **30,921 and 67,437 tokens** (2/2, no skips). No full-pool stress, 256K C2 stress or broad SWE-bench/HumanEval evaluation was performed.
- The head optimization deliberately changes GEMM shape. The original full-forward bitwise contract remains for the default path; the opt-in head uses a separate numerical gate. These spot checks are not proof of universal greedy equivalence.

| Panel | Text/reasoning matches final control | MTP acceptance vectors match | Harness | Prefix checks |
|---|---:|---:|---:|---:|
| Unmodified upstream | 19/19 | 4/4 | 0/1 attempted | not run |
| Same-source control (tool fix only) | 19/19 | 4/4 | 3/3 attempted | 8/8 |
| Last-row head only | 19/19 | 4/4 | 3/3 attempted | 8/8 |
| Cache-only history | 19/19 | 4/4 | 3/3 attempted | 8/8 |
| Both performance ports | 18/19 | 4/4 | 3/3 attempted | 8/8 |

Against the initial upstream panel, head-only and cache-only matched all 19 timing outputs; the combination matched 18/19. The changed concurrent code response changed its code body, not just wording. The deliberately staggered final-control probe ran solo code/JSON plus both arrival orders twice, with both performance switches off.
The unchanged control reproduced the combined variant exactly in: `solo-code`, `pair-0-code-first`, `pair-0-code-second`, `pair-1-code-first`, `pair-1-code-second`. This demonstrates that the variant can arise from arrival/batch scheduling without either optimization enabled; it does not prove universal numerical equivalence.

## Selected opt-in configuration

Use the checked-in [configuration](2026-09-23-mimo-upstream-ports/config.json) and enable both flags on both ranks:

```sh
DGPP_MIMO_PREFILL_LAST_HEAD=1
DGPP_MIMO_MTP_CACHE_ONLY=1
```

The compact XML/required-key repair is automatic for MiMo. [The port contract](../../docs/experiments/mimo-upstream-ports.md) describes the code and request-panel commands. Native heads 1/2, sliding-window rings, and the old interleaved-prefill controls have not yet been migrated; those are separate future changes.

## Disk cache and restoration

Upstream attempted to create ~80GB resident images per rank. Spark-1 ran out of space while capturing layer 42, logged a warning and continued from checkpoint; all weights were resident before timing. Later candidate starts restored the available image layers and loaded the rest from checkpoint. These are startup differences, not timing samples.

After the combined panel, both servers were stopped and only the two images created by this experiment were removed (`c4355edc78cf9f13.img` on rank 0, `0bd64f46b75071c2.img` on rank 1), reclaiming approximately 69/78 GiB. The final control uses `DGPP_RESIDENT_CACHE=0` on both ranks to prevent rebuilding those files. Its inference configuration and frozen binary otherwise match the candidates; its outputs and timings are recorded explicitly. No checkpoint was removed.

The original service was restored and both ranks attested to binary SHA256 `68845fee37a20a99b4107bec90321e57ab363277cbc2f71d71205d290115f6e6` with all six original MiMo flags. Final verification passed 8/8 prefix branch/reuse requests, MTP depth 3, four prefix slots, C2/262144 configuration, idle scheduler and zero failed requests. Both rank logs were archived; evidence is under `raw/final-restored/`. Raw SSE, requests, outputs, deployment attestations, both-rank logs, startup failures, build logs and cleanup manifests remain in `/mnt/benchmarks/dgpp-mimo-upstream-20260923/`. The main dirty checkout and its index were left intact.
