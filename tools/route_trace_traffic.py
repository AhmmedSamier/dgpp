#!/usr/bin/env python3
"""Route-trace traffic model (M4 deliverable 5): replaces the uniform
expert-routing assumption in DESIGN §3's per-rank decode traffic model with
measured routing decisions.

Consumes a DGPPTC1 trace file (per-layer top-k ids/weights recorded from the
engine's router kernels) and reports, for contiguous whole-expert partitions
(72 experts per rank at TP=4):

  * per-token busiest-rank expert occupancy — the §3 "3.515 experts/layer"
    metric, measured;
  * per-layer batch critical rank (max over ranks of total selections across
    all trace tokens) — honest when routes are correlated across tokens;
  * the corrected weight-bandwidth critical path, using the same fixed
    non-routed per-rank component as §3's mean-rank row.

The uniform model's anchors are built in: a uniformly random trace must
reproduce busiest ~3.515 experts/layer and critical path ~7.457 GB/token
(32.42 ms at 230 GB/s); a single-rank trace must reproduce the 12.198 GB
worst-placement row. tools' tests pin both.

Stdlib only (house style); the trace format is pinned by golden bytes in
both tests/unit/glm_trace_test.cpp and tests/python/route_trace_test.py.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass

MAGIC = b"DGPPTC1\x01"

# DESIGN §3 / docs/checkpoint_budget.md constants for GLM-5.3-Flash:
UNIFORM_BUSIEST = 3.515          # expected busiest-rank experts/layer/token
CRITICAL_GB_UNIFORM = 7.457      # expected synchronized critical path GB/token
CRITICAL_MS_UNIFORM = 32.42      # ... at 230 GB/s
FIXED_PER_RANK_GB = 3.7405       # mean-rank non-routed bytes (5.855 - 8.458/4)
BUS_BANDWIDTH = 230e9            # GB/s floor
N_MOE_LAYERS = 42
N_EXPERTS = 288
TP = 4


@dataclass
class TraceLayer:
    layer_idx: int
    top_k: int
    tokens: int
    ids: list[int]
    weights: list[float]


def read_trace(path: str) -> list[TraceLayer]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic {data[:8]!r}")
    (n_layers,) = struct.unpack_from("<I", data, 8)
    off = 12
    layers = []
    for _ in range(n_layers):
        layer_idx, top_k, tokens = struct.unpack_from("<IIQ", data, off)
        off += 16
        n = tokens * top_k
        ids = list(struct.unpack_from(f"<{n}i", data, off))
        off += 4 * n
        weights = list(struct.unpack_from(f"<{n}f", data, off))
        off += 4 * n
        layers.append(TraceLayer(layer_idx, top_k, tokens, ids, weights))
    if off != len(data):
        raise ValueError(f"{path}: {len(data) - off} trailing bytes")
    return layers


def expert_bytes(inter: int = 2048, hidden: int = 4096) -> int:
    """One routed expert's resident bytes: 3 E4M3 matrices + block scales."""
    payload = 3 * inter * hidden
    scales = 3 * 4 * ((inter + 127) // 128) * ((hidden + 127) // 128)
    return payload + scales


def rank_of(expert: int, tp: int = TP, n_experts: int = N_EXPERTS) -> int:
    per_rank = n_experts // tp
    return expert // per_rank


def analyze(layers: list[TraceLayer], tp: int = TP,
            n_experts: int = N_EXPERTS,
            e_bytes: int | None = None) -> dict:
    """Returns per-layer busiest-rank stats and the corrected critical path.

    per_token_busiest: mean over (token, layer) of the max per-rank selection
    count for that token — directly comparable to UNIFORM_BUSIEST.
    per_layer_critical: per layer, max over ranks of total selections across
    all tokens (batch critical rank), normalized per token.
    """
    if e_bytes is None:
        e_bytes = expert_bytes()
    per_token_busiest = []
    per_layer = []
    for l in layers:
        if l.tokens == 0:
            continue
        per_rank_total = [0] * tp
        token_busiest = []
        for t in range(l.tokens):
            counts = [0] * tp
            for i in range(l.top_k):
                counts[rank_of(l.ids[t * l.top_k + i], tp, n_experts)] += 1
            token_busiest.append(max(counts))
            for r in range(tp):
                per_rank_total[r] += counts[r]
        per_token_busiest.extend(token_busiest)
        per_layer.append((l.layer_idx, max(per_rank_total) / l.tokens,
                          token_busiest.count(max(token_busiest))))
    mean_busiest = (
        sum(per_token_busiest) / len(per_token_busiest)
        if per_token_busiest
        else 0.0
    )
    # Critical path per token: sum over layers of that layer's per-token
    # critical rank occupancy (per-token busiest is the batch-1 model; for
    # correlated multi-token batches use the per-layer batch row below).
    critical_bytes = mean_busiest * N_MOE_LAYERS * e_bytes + FIXED_PER_RANK_GB * 1e9
    return {
        "mean_busiest": mean_busiest,
        "critical_gb": critical_bytes / 1e9,
        "critical_ms": critical_bytes / BUS_BANDWIDTH * 1e3,
        "per_layer": per_layer,
        "n_tokens": len(per_token_busiest) // max(1, len(per_layer)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="DGPPTC1 route trace file")
    ap.add_argument("--tp", type=int, default=TP)
    args = ap.parse_args()

    layers = read_trace(args.trace)
    r = analyze(layers, tp=args.tp)
    print(f"trace: {len(layers)} layers, ~{r['n_tokens']} tokens/layer")
    print(f"per-token busiest-rank experts/layer: {r['mean_busiest']:.3f}"
          f"  (uniform model: {UNIFORM_BUSIEST})")
    print(f"per-layer batch critical rank (experts/token):")
    for layer_idx, per_tok, _ in r["per_layer"][:8]:
        print(f"  layer {layer_idx:2d}: {per_tok:.2f}")
    if len(r["per_layer"]) > 8:
        print(f"  ... {len(r['per_layer']) - 8} more layers")
    print(f"corrected critical path: {r['critical_gb']:.3f} GB/token"
          f" = {r['critical_ms']:.2f} ms @ {BUS_BANDWIDTH / 1e9:.0f} GB/s"
          f"  (uniform: {CRITICAL_GB_UNIFORM} GB, {CRITICAL_MS_UNIFORM} ms)")
    delta = (r["critical_gb"] / CRITICAL_GB_UNIFORM - 1) * 100
    print(f"vs uniform: {delta:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
