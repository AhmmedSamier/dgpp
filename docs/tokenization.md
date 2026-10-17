# Tokenization

Prompt encoding preserves the checkpoint's normalization, added-token and
pretokenization rules. BPE merges begin with the lowest-ranked adjacent pair
and merge all its occurrences from left to right without overlap.

Words below 256 initial byte-level symbols use in-place passes over spans
into the mapped input. Longer words can scan at most four times their initial
symbol count in pair probes before switching to a linked list and min-heap.
Runs that halve on each pass fit within that budget. The heap orders candidates
by rank and original position, rejects consumed or changed endpoints, and
defers new neighboring pairs until the current phase ends. Deferral matters
for checkpoint rules whose rank precedes the rules that create their operands.
The algorithm is selected internally and requires no deployment setting.

## Optional word cache

`tokenizer.word_cache_entries` accepts an integer from 0 to 10000 in deployment
JSON; its default is 0, which allocates no cache. The native tokenizer's
`load` method also accepts this capacity as its second argument. The cache is
owned by one loaded tokenizer and lives until that tokenizer is destroyed.
Only rank 0's serving tokenizer enables it; peer grammar construction does not.

Keys are the complete mapped pretokens after normalization and added-token
splitting. GLM whole-word vocabulary hits retain their direct lookup. Each
eligible word hashes to one of the configured slots. Exact key comparison
prevents a collision from returning another word's IDs; a fill replaces the
slot so the cache can adapt to later traffic. Eight shards use try-locks for
lookup and insertion. A busy shard computes normally, and all merge work
happens outside the locks. Cache state affects performance only.

Keys above 256 mapped bytes and results above 256 IDs are never retained.
At 10000 slots the key/ID payload is bounded by about 22 MiB, plus slot and
allocator overhead; empty slots take about 0.54 MiB on the supported 64-bit
build. Actual usage depends on word lengths and occupied slots. Keys and IDs
are owned copies, so prompt buffers can be released after encoding. Cache
contents are process-local and are neither persisted nor journaled.

Caching can improve repeated prose, code and template tokenization, but unique
words pay lookup and fill costs. Compare warm traffic, first use and traffic
that replaces the working set before enabling it. It does not accelerate
model prefill or decoding.

## Verification

Build `bpe_test` and `tokenizer_test` before running them. `bpe_test` requires
no checkpoint. It checks phase ordering, overlaps, stale candidates and
generated rule tables against an independent pass implementation. Work
counters assert a bounded number of pair probes and heap entries on difficult
inputs, and that uniform runs retain the pass path. These assertions do not
depend on machine speed.

Run `tokenizer_test tests/data/<family>_tokenizer_goldens.jsonl` for each of
the seven corpora. The test checks the tokenizer revision before the golden
comparison, and compares generated mapped words with the original merge loop.
`DGPP_TP_REAL_MODEL` can select an alternative cached checkpoint only when its
tokenizer revision matches the corpus. Record missing fixtures explicitly.

Measure optimized builds with tokenizer loading outside the timed region.
Include mixed letters, CJK, single-character repeats, whitespace, punctuation,
ordinary prose, code and chat templates. Use fixed input seeds and several
sizes, repeat measurements, compare exact token IDs, and report peak process
memory separately from elapsed time. Include both cold and repeated inputs.
Keep generated inputs and raw results outside tracked documentation.

For serving, use the same single-node configuration and model for both builds.
Measure normal requests and the effect of a long incoming pretoken on an
already-running stream and `/health`. Prompt tokenization runs on the HTTP
thread for text requests, before the token-budget check; a request rejected
for exceeding context can still exercise this path without a long GPU prefill.
Distinguish tokenizer latency and HTTP responsiveness from model throughput.
Record the tested revisions, configuration, results and limitations in the PR.

For the cache, build and run `word_cache_test`, then run the seven tokenizer
corpora with both `DGPP_TEST_WORD_CACHE_ENTRIES=0` and `10000`. The cached runs
also exercise concurrent encodes. The host cache tests check skipped work on
hits, exact collision handling, replacement, key/value bounds and progress
while a shard lock is held. Include sanitizer checks and concurrent replacement
stress. Config and launcher tests must reject wrong types, out-of-range values
and unknown tokenizer fields.
