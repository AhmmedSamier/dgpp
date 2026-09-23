# Native MiMo MTP on the upstream target

The upstream depth setting recurses through `model.mtp.layers.0`. The opt-in
`DGPP_MIMO_NATIVE_MTP=1` loads blocks 0, 1 and 2 instead. Set it on every
rank, together with `engine.mtp=true` and `engine.mtp_depth=3`.
The option is off by default. The config must declare three native heads.

For head d, the input at absolute row p consists of the normalized token
embedding for p+1 and the backbone's post-final-norm hidden at p-d. Its
attention position is p-d. Head d does not consume head d-1's output.
Base draft catch-up updates all heads; chain proposals select heads 1/2.
Unselected heads append only K/V, without attention, MLP or TP reductions.

Each head has its own layer in upstream's paged BF16 or scaled FP8 pool.
The backbone history ring holds 2083 rows per slot: the 2048-token prefill
chunk, the 32-row decode ceiling and three extra rows. Native mode rejects
walks larger than 2048 tokens. Prefix snapshots and draft rollback preserve
the ring; prefix cuts before a speculative draft use the pre-draft backup.
The memory plan includes the extra weights, BF12 companions, two extra KV
layers, history/rollback rings and scratch. Prefix snapshots are larger,
so the same prefix byte budget retains fewer entries.

The ordinary one-head path remains available with the option unset. Both
prior prefill flags are independent and are kept off for the MTP comparison.
The optional session-core selection hook is inert for other model families.

Validation surfaces:

- `mimo_native_mtp_test`: CPU oracle for history indexing, shifted positions,
  request slots, wrap and padding; distinct heads; eager rollback; prefix
  attachment into another slot; reopen; 2/23/97/2111-token prompts; both KV
  formats and checkpoint/BF12 weight residency.
- `mimo_engine_test`: native depth-3 scalar and row-batched graphs on a
  two-rank loopback world, checked against plain greedy decode, alongside
  the existing default one-head engine gates.
- `DGPP_MIMO_NATIVE_MTP=1 ./mimo_decode_test`: native config through the
  existing prefill, session, speculative and snapshot tests.

Real-checkpoint measurements and all-attempt hardware test outcomes live
in `/mnt/benchmarks/dgpp-mimo-mtp3-20260923/`; the final dated benchmark
record distinguishes recursive and native depth 3 and lists quality limits.
