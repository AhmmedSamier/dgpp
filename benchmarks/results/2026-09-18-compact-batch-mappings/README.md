# Compact Qwen batch mappings

Fixed-depth Qwen serving now selects a decode graph by active-request count,
rather than the highest occupied physical slot. Two live requests in slots
6 and 7 use the two-slot graph. Five live requests in slots 3 through 7 use
the six-slot graph. Rounding to existing bucket capacities remains; this
change does not add buckets or modify MoE/dense/collective kernels to skip
padding inside a selected graph.

## Implementation

Each graph family/parity owns a pinned group-to-physical-request map, at
most 16 int32 IDs. At replay entry, a kernel uploads that map and expands
row metadata; another gathers the small persistent token feeds into compact
rows. KV, recurrent/conv state, PLE context, draft windows, prefix arenas,
and sampling counts/RNG/bias/proposals retain their physical-slot addresses.

Commit descriptors use physical destination strides and compact snapshot
row offsets. Draft position/next-token kernels and hidden-window writes
translate group IDs back to physical IDs; feeds scatter back at the tail.
Sampler outputs/masks/logits remain compact while persistent sampling state
uses the mapping. Host verdict collection and prefix-hop/fallback snapshot
lookup use each replay's saved request order, not a subsequently staged map.

Separate pinned maps per parity avoid overwriting a map still in flight.
The host stages the next map without draining the previous replay. The GPU
upload/gather nodes are ordered on the existing stream and included in the
measured decode step. No new GPU synchronization or KV/recurrent-state move
was added. Draft-ring backup/restore still copies small rings for all physical
slots in compact captures; those copies are included in measured overhead.

Padding uses map entry -1 and negative positions. PLE hashing/context/conv
kernels explicitly handle negative request IDs before dereferencing state.
Other stateful kernels already test negative positions before state access.

Compaction is enabled for Qwen at fixed depth. Confidence-scheduled depth
and other families retain the original physical-prefix policy. Set
DGPP_COMPACT_BATCH=0 consistently on all ranks before startup for the old
layout. No scheduler admission or model capacity changes were made.

## Validation

- ARM64 server, Qwen engine test and picker test rebuilt successfully.
- Six Qwen engine tests passed in reference lowering, including MTP depths
  1/2/3, mmap PLE, sparse slots, cancellation/reuse and cached/yielded prefill.
- Focused C16/MTP3 tests passed with production lowering and sampled proposals.
- New sampling integration compares seeded transcripts between low slots,
  high slots and reused slots, including reversal of batch order between
  pipelined replays. Candidate caps 1 and 32 exercise exact-gather fallback
  and ordinary sampling. Penalties and distinct per-request seeds are used.
- A direct sampler test maps groups to physical slots 15 and 3 and padding,
  checks identity-layout decisions/digests/counts, and verifies that unmapped
  sampling state remains untouched. CUDA memcheck passes this isolated test.
- The full two-rank engine under memcheck hit the 5-second RDMA inactivity
  watchdog before completing. It reported zero memory errors, but that is
  not a completed full-engine memcheck result.
- Full-model TP2 startup succeeded. Sixteen concurrent sampled HTTP requests
  with distinct strict JSON schemas returned their own required values.
- git diff --check passed. Four-node hardware and the full repository suite
  were not run. clang-format is unavailable in this environment.

An initial candidate allocated pinned maps inside graph capture, which CUDA
rejected; allocation moved before capture. The new sampling test initially
omitted configure_sampling for its greedy setup; that failed with compaction
both enabled and disabled. The test now follows the serving API's per-request
configuration sequence. Transcript assertions remain intact.

## Performance

Same probe as the preceding padding-cost record: identical 8481-token prompts,
temperature zero, two 15-second trials per layout, compact/sparse/sparse/compact
order. Eight streams are admitted; unwanted streams are disconnected to keep
two or five requests in low or high physical slots. Samples reject competing
requests or changing occupancy. All eight candidate trials passed.

| Active requests / physical layout | Old bucket | New bucket | Old ms/step | New ms/step | Old decode tok/s | New decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 2 / low slots | 2 | 2 | 54.95 | 55.95 | 93.26 | 91.34 |
| 2 / high slots | 8 | 2 | 89.49 | 55.75 | 59.83 | 95.14 |
| 5 / low slots | 6 | 6 | 96.64 | 98.27 | 137.44 | 135.63 |
| 5 / high slots | 8 | 6 | 105.56 | 98.51 | 126.89 | 136.46 |

Sparse two-request steps improved by 37.7%; sparse five-request steps improved
by 6.7%. Output throughput rose about 59% and 7.5%, respectively. The already
compact layouts show about 1.7-1.8% longer steps than the previous epoch. These
short, non-interleaved before/after measurements cannot distinguish staging
cost from thermal/clock or other run-to-run variation. They establish a net
benefit for the tested sparse cases, not an overall agent-workload speedup.
Sampling acceptance and generated/context trajectories can differ between
shapes. Step time includes the engine call, not only GPU kernels. Repeated
identical prompts have less route diversity than typical agent workloads.

Raw build/test logs, HTTP responses and probe samples remain local under
artifacts/compact-batch-20260918. The CSV here contains aggregated measurements.
The previous batchmetrics release and its stable launcher configuration backup
are retained for rollback; the candidate is left running at C16/MTP3, port
30001, 512-token prefill chunks. Diagnostic stream cancellations are expected
in the service counters and should be excluded from workload comparisons.
