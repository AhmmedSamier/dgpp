# Judging a numerics change

Kernel work is allowed to change floating-point reduction order when it buys
latency, so two builds can legitimately produce different bits. These evidence
tools judge numerical changes and size sampling; all read run directories through
`scripts/fabric_logs.py`, the shared parser for the per-rank
`[gen]`/`[tf]`/`[sample_mass]` step lines — start there when writing the next
one:

- `scripts/fabric_xcript.py REF_DIR NEW_DIR` — for ordinary generation runs:
  finds the first token where the transcripts diverge and reports the global
  top-2 logit margin there in bf16 ulps. A flip at ≤ 1-2 ulp is a near-tie;
  a flip at a wide margin is a defect.
- `scripts/fabric_logprob.py NEW_DIR REF_DIR` — the quantitative gate. Run
  both builds with `--teacher-file benchmarks/teacher_text.txt` (the app
  scores the text's own tokens instead of generating; `--stage-file` puts the
  text on every rank), and the tool joins the ranks' vocabulary slices into
  log p(token) per position and reports the text's perplexity, the per-token
  deltas with a standard error, and a PASS/FAIL against a mean-NLL bound
  (default 0.02 nat ≈ 2% of perplexity):

  ```
  scripts/fabric_run.sh --stage-file benchmarks/teacher_text.txt -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --decode-graph
  scripts/fabric_logprob.py /path/to/new_logs /path/to/ref_logs
  ```

  A run of the same binary twice must show a delta of exactly 0 (the decode
  path is deterministic). The logits are fp32 (the head's accumulators,
  unrounded), but the residual stream feeding the head is bf16, so
  single-token deltas of ~0.1 nat are normal between builds that differ in
  reduction order; the mean over the text is the signal. Three
  texts ship: `teacher_text.txt` (556 tokens, a quick look),
  `teacher_text_hard.txt` (7,331 tokens of unmemorized technical prose,
  perplexity ~10.8 — the sensitive one; use it for the verdict) and
  `teacher_text_memorized.txt` (6,549 tokens of Conan Doyle, perplexity
  1.04 — the confident regime). Expect a handful of positions per thousand
  to move by more than 1 nat between any two rounding-level builds: a
  router's top-k boundary flipped an expert there. The tool bounds the
  *rate* of those, not the worst one.

- `scripts/fabric_sampling_profile.py DIR...` — the M6 sampling-width gate.
  Make one teacher-forced run per shipped text with `--sampling-profile` and
  fetched rank logs, then pass all three directories together:

  ```
  scripts/fabric_run.sh --fetch-logs \
      --stage-file benchmarks/teacher_text.txt --log-dir /tmp/mass-quick -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --sampling-profile \
      --decode-graph
  # Repeat as /tmp/mass-hard and /tmp/mass-memorized with the other texts.
  python3 scripts/fabric_sampling_profile.py \
      /tmp/mass-quick /tmp/mass-hard /tmp/mass-memorized
  ```

  The tool requires identical contiguous measurements on every rank, reports
  top-{32,64,128,256} mass and exact-gather fallback rate at T=1/top_p=0.95,
  and chooses the smallest k at or below 1%. The serving-side sweep is
  `scripts/serve_width_sweep.sh [OUT_DIR] [WIDTHS] [MODES]` (boots the world per
  width and graph mode over `--sampling-candidates`, the same seeded requests
  every time) with `scripts/width_sweep_report.py OUT_DIR/sweep.tsv` fitting
  the cost of one fallback. The extra diagnostic gather is
  intentionally outside the production path, so these runs are not throughput
  measurements.
