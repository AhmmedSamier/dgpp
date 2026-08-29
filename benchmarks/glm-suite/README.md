# GLM-5.3-Flash curated reference suite (M4, DESIGN §7.5)

Three parity cases (real prompts, tokenized from the checkpoint tokenizer)
plus one engine-only trace case. Reference dumps are NOT stored in the repo
(30-100 MB each, and they carry real weights' outputs); they are generated
where the checkpoint lives and cached locally:

```
CKPT=<checkpoint snapshot dir>
# Full-depth dumps (~3.3 s/layer measured; 4-10 min/case at 45 layers):
for c in smoke factual code; do
  python tools/glm_reference_dump.py gen-torch \
      --checkpoint-dir $CKPT --out $OUT/$c.glmdump \
      --token-ids "$(cat benchmarks/glm-suite/ids/$c.ids)" --tokens 512
done
./build-release/glm_forward_check --config $CKPT/config.json \
    --checkpoint-dir $CKPT --suite <suite file with dump paths>

# Reduced-budget variant (M4 close-out): --layers truncates BOTH the dump
# generation and the suite run (the budgets must match; truncation drops
# the MTP binding with it). At --layers 12: ~60-120 s/case generation and
# a few seconds per suite case, with ~10 routed layers of real flips:
for c in smoke factual code; do
  python tools/glm_reference_dump.py gen-torch \
      --checkpoint-dir $CKPT --out $OUT/$c.glmdump \
      --token-ids "$(cat benchmarks/glm-suite/ids/$c.ids)" --tokens 512 \
      --layers 12
done
./build-release/glm_forward_check --config $CKPT/config.json \
    --checkpoint-dir $CKPT --suite <suite file> --layers 12

# engine-only real trace (no reference needed):
./build-release/glm_forward_check --config $CKPT/config.json \
    --checkpoint-dir $CKPT \
    --trace-ids-file benchmarks/glm-suite/ids/trace.ids \
    --trace-out real.trace [--layers N]
```

`smoke` (the first case) uses the hand-picked id list in ids/smoke.ids — a
mid-conversation chat prefix. Dumps generated before the M4 close-out
lack the `router_biased` tensor (the per-flip certification inputs) and are
rejected by the runner — regenerate them with a current
`tools/glm_reference_dump.py`.

Comparison discipline: ISOLATED per-layer parity — every layer starts from
the reference trajectory, so cross-implementation noise (the measured
module floors: KDA 3.6e-3, DSA ~3e-3, MoE ~2.3e-3 vs their oracles) does
not compound. Free-run end-to-end drift IS chaos-limited at 45 layers
(measured: the floor amplifies ~1.18x/layer and decorrelates by layer ~30),
which is why free-run parity is reported, not asserted. Kept-row l2
budgets: < 0.02 per layer; route near-tie flips are excluded from the l2
and CERTIFIED individually (engine-vs-reference biased score rows, the
noise measured per token over the unswapped experts —
`src/models/glm_route_audit.hpp`); a 10% flip-rate net remains as a
wholesale-breakage detector. The head runs on the isolated final streams
and must agree on top-1, with disagreements certified as boundary near
ties (reference top-2 logit margin vs measured logit noise). At reduced
layer budgets the truncated stack crowds both the router and head
boundaries — expect certified flips/top-1s where the full 45-layer stack
showed zero.
