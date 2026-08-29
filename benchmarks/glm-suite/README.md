# GLM-5.3-Flash curated reference suite (M4, DESIGN §7.5)

Three parity cases (real prompts, tokenized from the checkpoint tokenizer)
plus one engine-only trace case. Reference dumps are NOT stored in the repo
(30-100 MB each, and they carry real weights' outputs); they are generated
where the checkpoint lives and cached locally:

```
CKPT=<checkpoint snapshot dir>
for c in factual code; do
  python tools/glm_reference_dump.py gen-torch \
      --checkpoint-dir $CKPT --out $OUT/$c.glmdump \
      --token-ids "$(cat benchmarks/glm-suite/ids/$c.ids)" --tokens 512
done
./build-release/glm_forward_check --config $CKPT/config.json \
    --checkpoint-dir $CKPT --suite <suite file with dump paths>
# engine-only real trace (no reference needed):
./build-release/glm_forward_check --config $CKPT/config.json \
    --checkpoint-dir $CKPT \
    --trace-ids-file benchmarks/glm-suite/ids/trace.ids \
    --trace-out real.trace
```

`smoke` (the first case) uses the hand-picked id list in ids/smoke.ids — a
mid-conversation chat prefix.

Comparison discipline: ISOLATED per-layer parity — every layer starts from
the reference trajectory, so cross-implementation noise (the measured
module floors: KDA 3.6e-3, DSA ~3e-3, MoE ~2.3e-3 vs their oracles) does
not compound. Free-run end-to-end drift IS chaos-limited at 45 layers
(measured: the floor amplifies ~1.18x/layer and decorrelates by layer ~30),
which is why free-run parity is reported, not asserted. Kept-row l2
budgets: < 0.02 per layer; route near-tie flips are counted (bounded < 1%)
and excluded from the l2; the head runs on the isolated final streams and
must agree on top-1.
