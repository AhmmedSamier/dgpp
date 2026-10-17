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
