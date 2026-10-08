# Prefill configuration metadata

The renderer now adds resolved busy/idle prefill token budgets to the configuration table. Values come from the rank-zero startup log; source paths and hashes are recorded in `../resolved-prefill-budgets.json`. This does not change measurements, deployment settings, or the serving binary. Saved per-launch manifests remain unchanged.

The previous renderer and master manifest are preserved here. The updated master manifest records the new renderer hash and update time.
