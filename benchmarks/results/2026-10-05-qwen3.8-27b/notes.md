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

## Round three: the open items

`lam11`, `ab12`, `verify13`, `final15`, `fn18`: see the previous section. `trace16`: the one-node plain T=1
step under nsys — 91 % one kernel, the width-8 streaming form at one row (368 launches, 301.7 us each,
219 GB/s against the head launch's 231). `ab17` / `ab19`: the campaign-era release binary (the GEMV chunks at
one row, 112.7 ms a step) against the current one (121.1), each with the L2 weight prefetcher on and off —
no difference either way; `pf21`: the prefetcher's forms through the new `engine.l2_prefetch*` keys (off /
load / lines / touch / touch with a 48 MiB window) — no difference at one row. The streaming form's 8 % at
one row is wave quantization of its 64-row blocks; a finer unit at one to four rows is the next kernel item.
Qwen3.8-Flash-Next (`fn18`): with the fp8 head and the dense sites on the streaming form from one row its
one-node recipe reads 40.9 / 47.6 tok/s (prose / code) at 39.4 ms a step and plain 31.5 at 31.3 (final15,
before: 44.5 / 48.7 at 38.5 and 32.3 at 30.5; the prose cell is one prompt whose continuation moved);
`qwen_decode_rows_invariance` holds every solo row count and the two-by-eight and four-by-four batched
verifies bitwise. `gates22`: the head tests under the new rule, the tiny gates and the ctest subset.

The head rule, closed (`gates24`, `gates25`): under `engine.fp8_head: "mma"` the Flash-Next head takes the
streaming form at every width — the compact prefill row and prompts above the decode capacity included (the
GEMV chunks below five rows and above the capacity were two more chains). `qwen_decode_test --fp8-head`
now asks for the streaming kernel at every batched row count and holds the minimum-capacity control bitwise
the wide model at every prompt length; `qwen_head_check` counts the streaming kernel at every captured
shape and names every captured kernel when the count is off; `qwen_head_compare.py` drops its width
controls (no width shares the GEMV reference's chain any more — the compaction gate keeps its dense
controls). `gates24`'s head-check failure was a stale object: the full build had not recompiled
`qwen_head_check.cpp.o` after the predicate edit, so the binary still expected the GEMV chunks at four rows;
a relink with the diagnostic rebuilt it and the gate passed on the free node three of three (`gates25`).
`gates24`'s Flash-Next recipe leg (36.3 / 41.4 tok/s at 43–46 ms a step) is void: my own head-check runs
and a relink overlapped it — nothing else of one's own on the node during a timing leg; its plain leg,
unperturbed, read 31.5 / 31.4 at 31.3 ms (= `fn18`). `gates25` re-measured the recipe leg alone: 41.1 / 47.8
tok/s at 39.4 ms a step, 1.78 tokens a step (= `fn18`); the head check on the free node three of three, the
head gates five of five, the ctest subset 85 of 86 with the compare script's own unit test the one failure
(it asserted the retired width controls — updated to the one-chain rule).
