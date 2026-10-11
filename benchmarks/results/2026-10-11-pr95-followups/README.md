# PR 95 follow-ups: correctness and dense NVFP4 prefill

This continues the independent review of PR head
`46715829a5de6ffa9934f17b585c0aaf9dbed010` on four GB10 nodes.
The preceding local optimization, `d92ee25d`, widened the fused FP4 tile
from 128 to 256 rows. The follow-ups preserve native checkpoint formats
and the FP32 dot product followed by one global-scale division.

## Review fixes

- Context lookup now runs only on greedy MTP slots. The former sampled
  path repriced only changed tokens, which biases the accepted output;
  agreement-conditioned lookup also needs a separate distributional proof.
  Sampled slots retain their original MTP proposals. The existing sampled
  lookup metric remains present and zero.
- Lookup engine tests use a controlled zero-head fixture. They require
  lookup to fire and exercise rejection, accepted prefixes, batching and
  slot reuse. Seeded sampled requests verify that enabling lookup leaves
  the sampled MTP path unchanged at tail depths 0 and 3.
- Mixed resident initialization only harvests an MTP layer when MTP is
  enabled. The DFlash templates no longer materialize an unused 710 MiB
  draft layer at world 1. A test compares actual resident allocations with
  the memory plan for both MTP settings.

## Kernel and numerical checks

The 256x128x64 fused tile now uses eight warps, each covering 64x64 output
values. Each thread fetches 32 FP4 codes and applies the two 16-code scale
groups. Two activation stages and two decoded weight tiles use 96 KiB of
shared memory. The ascending-K FP32 accumulation and final division are
unchanged; the production decode kernel remains selected below 1024 rows.

The mixed fixture reference chain now also covers 3072-token prefills.
Tensor-parallel tests cover the FP8 fixture and both short and 2304-token
mixed forwards at worlds 2 and 4 against world 1. Their latency transport
slot holds a complete 1.125 MiB fold, so this numerical test does not
accidentally become a bulk-transfer timeout test.

A real-checkpoint probe scored 19,456 positions: overlapping first/last
4096-token windows of the hard and memorized teacher texts, plus a
3072-token stress input made by repeating the short teacher text. All
full-vocabulary per-row logit hashes matched the preceding 16-warp kernel;
mean/max NLL deltas and top-1 changes were zero. A separate process repeated
4096 positions bitwise. See [comparison](numerical-t256-comparison.json)
and [input manifest](numerical-manifest.json). This is a measured sample,
not a proof over every input.

A BF16-output cuBLAS bridge was also evaluated and rejected. It improved
speed but changed real-checkpoint numerics: the repeated-text case moved
mean NLL by +0.0769 nat/token, with 23.99% of positions moving by more than
one nat and 550 top-1 changes outside the two-ULP criterion. Synthetic
fixtures alone did not expose this. The [rejected comparison](numerical-comparison.json)
is retained; none of that bridge is in the implementation.

CUDA memcheck reported zero errors and racecheck reported zero hazards.
The full Release build passed before the suite. Of 215 non-checkpoint
CTest entries, 211 passed, two MiMo tokenizer/template cases skipped for
an absent checkpoint, and two bus cases failed because the serving site's
single-lane setting could not satisfy their explicit dual-lane assertions.
Both bus cases passed when rerun with the head node's two active NICs,
`rocep1s0f0 roceP2p1s0f0`, GID indices `3 3`: 213 passing entries and two
skips in total. No source change was required. The initial failure remains
in [validation status](validation-status.json); the corrected run is in
[bus retest](bus-retest.json).

The first four-node serving check completed 512/2K/8K/32K cold-prefill
buckets, three repetitions each, with matching rank operation streams,
identical deployed binary hashes and zero process swap on all four nodes.
The 8K median is 4594 ms, still 0.3% above the preceding FP8 measurement;
this preliminary run does not establish the final performance gate.
See [request measurements](eight-warp-w4/prefill.json).
This was followed by the paired-write candidate and base integration below;
the final matched serving matrix remains the merge gate.

## Paired output writes

The next candidate packs adjacent BF16 columns into one conversion and
one aligned 32-bit store. Unaligned views, odd row strides and a final odd
column keep scalar stores. A sentinel test exercises those cases across
3073 rows, including the last partial M tile. For walks of at least 3072
rows, three M tiles per cache group replace four; shorter walks retain
four. This changes only block order and output writes, not a dot's
arithmetic.

All 18 targeted mixed-reference, tensor-parallel and FP4 kernel entries
passed after integrating the current base branch. The same 19,456 real
checkpoint positions and the 4096-position process repeat still match the
original kernel's full-vocabulary hashes exactly; see the
[paired-write comparison](numerical-paired-comparison.json).

## Operational evidence

Raw logs, executable hashes, per-request JSON, numerical TSVs and rejected
experiments are retained locally in
`/home/stephen/dgpp/pr95-followups-20261011/`. The earlier review and cache
cleanup audit are in `/home/stephen/dgpp/pr95-review-20261010/`.

## Base integration and full validation

The branch includes master at `38d3af69485158a926eefb7eb46fbac475295df4`,
including the shared GLM kernel changes and startup diagnostics. The full
Release build completed before the serial 241-entry CTest run. It recorded
234 passes, six absent-checkpoint tokenizer/template skips and one stale
HTTP-default assertion inherited from master. That assertion now checks
`0.0.0.0` by default and an explicit localhost override. All three affected
Python suites passed on retest: 235 passing entries, six skips and no
unresolved failures. See [suite log](merged-suite.log),
[retest](binding-retest.log) and [resolution](full-suite-resolution.json).
The FP4 memcheck and racecheck runs also reported zero errors/hazards.

The paired-write four-node smoke measured median cold-prefill latencies
of 367.6 / 1261.7 / 4545.5 / 18955.6 ms at approximately 512 / 2K / 8K /
32K tokens. The preceding same-prompt FP8 medians were 415.8 / 1292.8 /
4566.0 / 19046.4 ms. These are preliminary comparisons: the small long-
prompt margin needs confirmation by the final matrix. All four paired-
write rank operation streams matched, deployed executable hashes agreed,
and process/cgroup swap stayed zero at the before/after checks.
See [NVFP4 requests](paired-w4/prefill.json),
[FP8 requests](eight-warp-fp8-w4/prefill.json) and
[rank streams](paired-w4/opstreams.json).
