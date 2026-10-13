# GLM-5.3-Flash curated reference suite (M4, DESIGN §7.5)

The original suite used three reference comparisons with tokenized prompts
and one engine-only trace case. Its input corpus is not bundled with the
repository. Supply your own `smoke.ids`, `factual.ids`, `code.ids` and
`trace.ids` files, containing comma-separated token IDs from the checkpoint's
tokenizer. To tokenize a local text, use
`python3 scripts/prepare_data.py tokens --model MODEL --text FILE --out FILE`
with the matching model cached locally. Retain the inputs and tokenizer
revision alongside the results.

Generate reference dumps where the checkpoint is available. Original dumps
were 30–100 MB each and were cached locally.

Run the following commands from the repository root. Keep dumps and traces
in ignored local storage or external artifacts. The suite file has one
`name|/absolute/path/to/name.glmdump` entry per line, naming each generated
dump; blank lines and `#` comments are allowed.

```
CKPT=/path/to/checkpoint/snapshot
IDS=/path/to/reference/ids
OUT=/path/to/reference/dumps
SUITE=/path/to/suite-file
mkdir -p "$OUT"
# Full-depth dumps (~3.3 s/layer measured; 4-10 min/case at 45 layers):
for c in smoke factual code; do
  python tools/glm_reference_dump.py gen-torch \
      --checkpoint-dir "$CKPT" --out "$OUT/$c.glmdump" \
      --token-ids "$(cat "$IDS/$c.ids")" --tokens 512
done
./build-release/glm_forward_check --config "$CKPT/config.json" \
    --checkpoint-dir "$CKPT" --suite "$SUITE"

# Reduced-budget variant (M4 close-out): --layers truncates BOTH the dump
# generation and the suite run (the budgets must match; truncation drops
# the MTP binding with it). At --layers 12: ~60-120 s/case generation and
# a few seconds per suite case, with ~10 routed layers of real flips:
for c in smoke factual code; do
  python tools/glm_reference_dump.py gen-torch \
      --checkpoint-dir "$CKPT" --out "$OUT/$c.glmdump" \
      --token-ids "$(cat "$IDS/$c.ids")" --tokens 512 \
      --layers 12
done
./build-release/glm_forward_check --config "$CKPT/config.json" \
    --checkpoint-dir "$CKPT" --suite "$SUITE" --layers 12

# engine-only real trace (no reference needed):
./build-release/glm_forward_check --config "$CKPT/config.json" \
    --checkpoint-dir "$CKPT" \
    --trace-ids-file "$IDS/trace.ids" \
    --trace-out "$OUT/real.trace"
```

The original `smoke` case used a mid-conversation chat prefix. Dumps generated
before the M4 close-out lack the `router_biased` tensor (the per-flip
certification inputs) and are
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
`src/models/glm/route_audit.hpp`); a 10% flip-rate net remains as a
wholesale-breakage detector. The head runs on the isolated final streams
and must agree on top-1, with disagreements certified as boundary near
ties (reference top-2 logit margin vs measured logit noise). At reduced
layer budgets the truncated stack crowds both the router and head
boundaries — expect certified flips/top-1s where the full 45-layer stack
showed zero.
