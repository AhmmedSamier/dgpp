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
