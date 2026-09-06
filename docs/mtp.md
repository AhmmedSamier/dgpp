# Speculative decode with the MTP layer (`--mtp`)

The checkpoint ships a multi-token-prediction layer (`num_nextn_predict_layers
= 1`): one extra DSA + MoE block that, given the main stack's hidden at
position q and the token chosen for q+1, guesses the token at q+2.
`glm_gen_check --mtp` uses it for greedy speculative decoding on the fabric:

```
scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
    --chat "Write a history of the Roman Republic." --steps 300 \
    --decode-graph --mtp
```

Every step is **one CUDA graph replay** that carries its own control flow:
the two-row verify of `[next, draft]`, the pick behind the head (each
rank's argmax gathered through one bus collective and judged on the
device), the commit (a rejected second row is rolled back on the device —
the KDA/DSA kernels snapshot their state after every speculative row, so a
retraction is a predicated copy per state family — and the position
advances), the MTP block over the accepted rows (a fixed two-row batch; the
second row is padding after a miss), its own pick, and the next step's
tokens written on the device. The host launches, waits, reads two small
pinned verdicts and logs. **The transcript is exactly the plain loop's** —
the verify rows are bitwise the single-token rows, and
`scripts/fabric_xcript.py PLAIN_DIR MTP_DIR` must print `IDENTICAL`; MTP
only changes what a token costs. Measured (2026-09-03, TP=4): 88.7% of
drafts accepted on coherent text, 1.89 tokens per step, **22.45 ms/token
effective vs 31.3 plain** (−28%). The step is 42.4 ms: the second verify
row costs ~7 ms because its MoE experts are extra DRAM bytes (only the
experts both rows share are read once — the slots run in expert order so
the second read is an L2 hit), the draft block ~2 ms. Acceptance drops on
incoherent text (63% on the post-EOS rambling `--no-eos` produces), and
below ~45% speculation stops paying; the summary line reports the rate.
Every rank computes the verdict itself from an identical gathered table; a
rank whose table was corrupt is caught at the next pick by a digest every
rank carries (`GlmDevicePicker`). Without `--decode-graph` the same step
runs eagerly (26 ms/token). The draft layer adds ~7.3 GiB per rank to the
resident footprint. Serving uses adaptive scalar and row-batched variants
with up to four MTP requests per replay. At concurrency 1 on the four-node service
(2026-09-03), a replay is 43.6–44.1 ms and carries 1.69–1.88 tokens by the
text's acceptance, 21.8–26.0 ms/token. The original always-eight-row graph
delivered 19.3/35.9/65.8 tok/s at 1/2/4 live requests; adaptive selection
raises that curve to 38.48/38.97/60.27, using the batch only at four. The
corresponding adaptive T=1 curve is 30.97/31.67/41.76/76.18 tok/s at
1/2/4/8 live requests. Every measured world's four op streams were
identical, and the final 256-token transcript was identical across modes,
slots, occupancy, and scalar↔batch transitions. Per user, those curves are
32/63/96/105 ms per token at 1/2/4/8 live T=1 requests and 26/51/66 at 1/2/4
live MTP requests: below the crossover each user pays the live count times
the scalar replay, at the crossover the batch is faster for the user too
(four scalar replays would be ~129 and ~104 ms per token).
