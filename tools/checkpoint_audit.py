#!/usr/bin/env python3
"""Checkpoint audit: inventory safetensors tensors and derive per-token byte budget.

Reads only JSON headers (no tensor data), so it is fast and memory-safe.

Usage:
  python3 tools/checkpoint_audit.py [model_dir]

Writes:
  artifacts/checkpoint_inventory.json  (raw per-tensor listing)
  docs/checkpoint_budget.md            (human-readable budget report)
"""

import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = Path.home() / ".cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-FP8"

DT_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F16": 2,
    "F8_E4M3": 1,
    "I32": 4,
    "U32": 4,
    "U8": 1,
}

CLASS_PATTERNS = [
    ("embed", r"(embed_tokens)\."),
    ("mtp", r"layers\.45\."),
    ("vision", r"^(model\.vision|visual|vision_tower)"),
    ("proj_layer", r"(eh_proj|enorm|hnorm)"),
    ("router", r"\.mlp\.gate(\.|$)|e_score_correction_bias"),
    ("shared_expert", r"shared_expert"),
    ("moe_expert", r"\.experts\.\d+\."),
    # linear-attn path (LA layers keep these bf16 per audit)
    ("attn_lin", r"\.self_attn\.(k_proj|v_proj)\.(weight)(?!_)"),
    ("attn_dsa_fp8", r"\.self_attn\..*(weight_scale_inv)$"),
    ("attn_dsa", r"\.self_attn\.(q_a_layernorm|kv_a_layernorm|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|indexer)\.|o_norm"),
    ("mhc", r"(hc_attn|hc_ffn)_(base|fn|scale)|mapping_proj"),
    ("dense_mlp", r"\.(up_mlp|down_mlp|gate_proj|up_proj|down_proj)(\.weight)?(\.weight_scale_inv)?$"),
    ("lm_head", r"^lm_head\.weight$"),
    ("norm", r"\.(input_layernorm|post_attention_layernorm|final_layernorm)\."),
]


def classify(name: str) -> str:
    for cls, pat in CLASS_PATTERNS:
        if re.search(pat, name):
            return cls
    return "other"


def numel(shape):
    p = 1
    for d in shape:
        p *= int(d)
    return p


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    snaps = list((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else [root]
    snap = snaps[0]
    index_path = snap / "model.safetensors.index.json"
    idx = json.loads(index_path.read_text())
    weight_map = idx["weight_map"]
    files = sorted(set(weight_map.values()),
                   key=lambda s: int(re.findall(r"\d+", s)[0]))

    tensors = {}
    for fname in files:
        hdr = read_header(snap / fname)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            tensors[name] = (fname, meta.get("dtype"), tuple(meta.get("shape") or []))

    cls_bytes, cls_count, dt_bytes, unmatched = defaultdict(int), defaultdict(int), defaultdict(int), []
    total = 0
    for name, (_f, dt, shape) in sorted(tensors.items()):
        b = numel(shape) * DT_BYTES.get(dt, 0)
        dt_bytes[dt] += b
        cls = classify(name)
        cls_bytes[cls] += b
        cls_count[cls] += 1
        total += b
        if cls == "other":
            unmatched.append((name, dt, shape))

    L_LIN, L_DSA, L_MOE, TOPK = 34, 11, 42, 8  # from config.json (45 layers, 3 dense)

    def cb(cls):
        return cls_bytes.get(cls, 0)

    # Every MoE layer stores 288 experts; a token activates 8 -> proportional
    # read share of the expert pool (assuming uniform spread of selections).
    routed_tok = cb("moe_expert") * TOPK / (288 * L_MOE) * L_MOE
    shared_tok = cb("shared_expert")          # every-layer always-on expert
    dense_tok = cb("dense_mlp")               # 3 dense layers, always on
    dsa_tok = cb("attn_dsa") + cb("attn_dsa_fp8")
    lin_tok = cb("attn_lin")
    # layerwise leftover auxiliaries (A_log, dt_bias, gates, convs, b_proj ...)
    aux_tok = cb("other")
    lm_tok = cb("lm_head")
    misc_tok = cb("router") + cb("mhc")       # replicated-but-read modules
    est_active = routed_tok + shared_tok + dense_tok + dsa_tok + lin_tok + aux_tok + lm_tok + misc_tok

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/checkpoint_inventory.json").write_text(json.dumps(
        {n: {"file": v[0], "dtype": v[1], "shape": list(v[2])} for n, v in tensors.items()},
        indent=0))

    lines = ["# GLM-5.3-Flash Checkpoint Budget Report", "",
             f"- Source: `{snap}`",
             f"- Files: {len(files)} shards; tensors: {len(tensors)}",
             f"- **Total on-disk weights: {total/1e9:.1f} GB**",
             "", "## Bytes by class", "",
             "| class | count | GB | share |", "|---|---:|---:|---:|"]
    for cls, b in sorted(cls_bytes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls} | {cls_count[cls]} | {b/1e9:.2f} | {100*b/max(total,1):.1f}% |")
    lines += ["", "## By dtype", "", "| dtype | GB |", "|---|---:|"]
    for dt, b in sorted(dt_bytes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {dt} | {b/1e9:.2f} |")

    lines += [
        "",
        "## Per-token active-traffic estimate (FP8 + BF16 mix, bs=1 decode)",
        "",
        f"- routed experts (42 MoE layers, top-{TOPK} of 288): "
        f"**{routed_tok/1e9:.2f} GB/token**",
        f"- shared expert (always-on): {shared_tok/1e9:.2f} GB/token",
        f"- dense MLPs (3 layers): {dense_tok/1e9:.2f} GB/token",
        f"- DSA attention projections: {dsa_tok/1e9:.2f} GB/token ({L_DSA} layers)",
        f"- linear-attn projections (BF16 in checkpoint): {lin_tok/1e9:.2f} GB/token ({L_LIN} layers)",
        f"- linear-attn auxiliaries (A_log/dt/gates/convs): {aux_tok/1e9:.2f} GB/token",
        f"- lm_head matmul (bf16, full matrix): {lm_tok/1e9:.2f} GB/token",
        f"- router+mHC misc: {misc_tok/1e6:.0f} MB/token",
        "",
        f"- **Total ≈ {est_active/1e9:.1f} GB/token cluster-wide**",
        f"- TP=4 share ≈ **{est_active/4/1e9:.2f} GB/node** → decode floor ≈ "
        f"{est_active/4/230e9*1000:.0f} ms/token @230 GB/s ⇒ ~{1000/(est_active/4/230e9*1000):.0f} tok/s pre-speculation",
        "- ⚠ Optimization headroom: BF16 linear-attn projections could be "
        "requantized to FP8 (~saves most of the bf16 traffic) — quality-gated "
        "track for later.",
    ]
    if unmatched:
        lines += ["", f"## Unmatched names ({len(unmatched)}):"] + \
                 [f"- `{n}` [{d}] {s}" for n, d, s in unmatched[:80]]

    Path("docs").mkdir(exist_ok=True)
    Path("docs/checkpoint_budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
