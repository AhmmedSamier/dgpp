# Notes: the 2026-10-05 Qwen3.8-27B-FP8 campaign

The runner, evaluator, summarizer and workload are the 2026-10-04 campaign's
([record](../2026-10-04-qwen3.8-27b/README.md)); the templates differ: every Qwen3.8-27B
template is now the DFlash2 drafter on the decode graph (two and four nodes switched from
MTP depth 3 on 2026-10-05), a sampled request's drafts drawn from the drafter's distribution at
0.7 of the request's temperature and verified by the ratio rule (`engine.mtp_draft_temperature`),
the drafter's matrices packed lossless 12-bit (`bf16_weights: bf12`). The decode-modes stage
measures each template's plain world and the MTP world it replaced (depth 2 on one node, depth 3
on two and four); the record's decode-modes table has the page's three value columns.

The isolation check (solo against eight co-admitted) compares transcripts across the
family's row-count dispatch classes (one row against up to 32), which are not bitwise
each other's for Qwen3.8-27B (the 2026-10-04 notes); a DIFFERENT reading there is the
dispatch class, not a fault. The reference harness, HumanEval fences and the reference
cross-check are as described in the 2026-10-04 notes.

## The afternoon: a request's logits the same alone, at every verify depth and in a batch

The isolation check's DIFFERENT reading above was taken apart after the campaign (the
[changelog](../../../CHANGELOG.md) entry of 2026-10-05 names the six causes and the fixes). The
legs, in order, each a `*.log` beside its `raw/` directory: `isolation4` (the head, the a / b
row-group GEMV, the fp8 projections and the one-chain split-K: co-tenants joining a running
request read IDENTICAL on one, two and four nodes; the four-node eight-slot step 106 → 91.7 ms),
`diag4` (`--dflash-depth` is the eager engine's knob: vacuous on the graph; a fixed lambda made
the first request after boot read like the second; without the prefix cache the same), `bisect5`
and `bisect6` (the schedule's minimum depth: 6- and 8-row steps match whole blocks 4/4, 4-row
steps do not), `verify8` (the recorded commit's rows: a real defect, not the operative one),
`final9` (a forced-shallow schedule 0 of 4 against whole blocks; the ctest subset 74/74 with the
node free), `verify10` (the in-projections and the attention q|k|v on the streaming form at every
row count: every probe IDENTICAL from the first request after boot, the template, min-depth 3 and
the forced-shallow schedule 4/4 against whole blocks on one node, the four-node template 4/4
against `w4-fp8-2`, the drafter templates level, plain T=1 8.2 tok/s on one node and 28.0 on four).
The tiny-fixture test `qwen35_decode_rows_invariance` found the last cause in minutes after the
transcript legs could not: a 2-, 3- or 4-row verify's second row differed from the eight-row
verify's by one bf16 ulp in one column of the first GDN layer's in-projection
(`dense_gemv_rows()` defaults to 4). Prompts started together still differ (the group prefill).
