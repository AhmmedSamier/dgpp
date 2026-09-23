#!/usr/bin/env python3
"""Inventory a GLM-5 or Qwen3.8-Flash-Next safetensors checkpoint and model decode traffic.

Only safetensors JSON headers are read. The report separates vision, MTP, and
base text weights and includes a placement-aware TP critical-rank estimate.

The architecture is read from config.json's `architectures`: the GLM path
writes docs/checkpoint_budget.md (unchanged since M0); the Qwen4Exp path
(2026-09-09, the Qwen plan's Q0) writes docs/qwen38_checkpoint_budget.md with
the placement formulas of docs/qwen38_flash_next_plan.md §2.1 evaluated at
TP=2 and TP=4; the GlmMoeDsa path (2026-09-12, the full GLM-5.3 plan's G0)
writes docs/checkpoint_budget_glm53.md: the compressed-tensors pack-quantized
triples (`weight_packed` I32, `weight_scale` BF16, `weight_shape` I64) are
checked against the config's quantization groups and the placement formulas
of docs/glm53_plan.md §2.1 / §2.2 are evaluated per rank; the DeepseekV41 path
(2026-09-13, docs/deepseek_v41_flash_plan.md G0) writes
docs/checkpoint_budget_dsv41.md: every fp8 pair (`.weight` F8_E4M3 + `.scale`
F8_E8M0 on the 32x32 grid), every MXFP4 pair (`.weight` I8 [N, K/2] +
`.scale` F8_E8M0 [N, K/32]) and the two Engram tables are checked, and the
placement of plan §2 is evaluated at world 4 (and 2, for the record).

Usage:
  python3 tools/checkpoint_audit.py [model_dir] [--world N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT = Path.home() / ".cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-FP8"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = 4
DEFAULT_BANDWIDTH_GBPS = 230.0

# ---------------------------------------------------------------------------
# MiMo-V2.6-Flash (2026-09-22, docs/mimo_v26_flash_plan.md): fp8 dense on the
# 128x128 grid (`.weight` F8_E4M3 + `.weight_scale_inv` F32), MXFP4 experts
# (`.weight` U8 [N, K/2] + `.weight_scale` U8 e8m0 [N, K/32]), BF16 o_proj /
# router / norms / embedding / head; the vision and audio encoders and the
# draft layers past the first present and never loaded.
MIMO_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
MIMO_DRAFT_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.(.+)$")
MIMO_FP8_BLOCK = 128
MIMO_FP4_BLOCK = 32


def classify_mimo(name: str, num_layers: int) -> str:
    """Mutually exclusive storage class of a MiMo-V2.6-Flash tensor."""
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.embed_tokens.weight":
        return "embed"
    if name == "model.norm.weight":
        return "norm"
    if name.startswith(("visual.", "audio_encoder.", "speech_embeddings.", "audio_projector.")):
        return "encoder"
    match = MIMO_LAYER_RE.match(name)
    if match is None:
        draft = MIMO_DRAFT_RE.match(name)
        if draft is None:
            return "other"
        return "draft" if int(draft.group(1)) == 0 else "draft_unserved"
    layer = int(match.group(1))
    if layer >= num_layers:
        return "other"
    tail = match.group(2)
    if tail in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return "norm"
    if tail.startswith("self_attn.qkv_proj."):
        return "attention_qkv"
    if tail.startswith("self_attn.o_proj."):
        return "attention_o"
    if tail == "self_attn.attention_sink_bias":
        return "attention_sink"
    if tail.startswith("mlp.experts."):
        return "routed_expert"
    if tail.startswith("mlp.gate."):
        return "router"
    if tail.startswith(("mlp.gate_proj.", "mlp.up_proj.", "mlp.down_proj.")):
        return "dense_mlp"
    return "other"


def validate_mimo_pairs(tensors: dict[str, tuple[str, str, tuple[int, ...]]],
                        qkv_chunks: int) -> dict[str, dict]:
    """Check every quantized pair and return {base: {fmt, n, k, payload, scales}}.

    fp8 pairs: `.weight` F8_E4M3 [N, K] with `.weight_scale_inv` F32 on the
    128x128 grid — the fused qkv_proj's grid tiled per chunk (ceil(N / chunks
    / 128) rows per chunk); MXFP4 pairs: `.weight` U8 [N, K/2] with
    `.weight_scale` U8 [N, K/32]. Every scale belongs to exactly one pair.
    """
    pairs: dict[str, dict] = {}
    scales_seen = set()
    for name, (_file, dtype, shape) in tensors.items():
        if not name.endswith(".weight") or dtype not in ("F8_E4M3", "U8"):
            continue
        base = name[: -len(".weight")]
        if dtype == "U8":
            scale = tensors.get(base + ".weight_scale")
            if scale is None:
                raise ValueError(f"{name}: a U8 payload without its .weight_scale")
            _sfile, sdtype, sshape = scale
            n, k = shape[0], shape[1] * 2
            if sdtype != "U8" or tuple(sshape) != (n, k // MIMO_FP4_BLOCK) or k % MIMO_FP4_BLOCK:
                raise ValueError(f"{base}: MXFP4 scale {sdtype} {sshape} does not tile [{n}, {k}] by {MIMO_FP4_BLOCK}")
            scales_seen.add(base + ".weight_scale")
            pairs[base] = {"fmt": "mxfp4", "n": n, "k": k, "payload": numel(shape), "scales": numel(sshape)}
            continue
        scale = tensors.get(base + ".weight_scale_inv")
        if scale is None:
            raise ValueError(f"{name}: an F8_E4M3 payload without its .weight_scale_inv")
        _sfile, sdtype, sshape = scale
        n, k = shape[0], shape[1]
        if sdtype != "F32" or len(sshape) != 2 or k % MIMO_FP8_BLOCK:
            raise ValueError(f"{base}: fp8 scale is {sdtype} {sshape}, not F32 [rows, {k // MIMO_FP8_BLOCK}]")
        if base.endswith("qkv_proj"):
            rows_per_chunk = n // qkv_chunks
            expected = (qkv_chunks * (-(-rows_per_chunk // MIMO_FP8_BLOCK)), k // MIMO_FP8_BLOCK)
        else:
            expected = (-(-n // MIMO_FP8_BLOCK), k // MIMO_FP8_BLOCK)
        if tuple(sshape) != expected:
            raise ValueError(f"{base}: fp8 scale {sshape} is not the grid {expected}")
        scales_seen.add(base + ".weight_scale_inv")
        pairs[base] = {"fmt": "fp8", "n": n, "k": k, "payload": numel(shape), "scales": numel(sshape)}
    for name, (_file, _dtype, _shape) in tensors.items():
        if (name.endswith(".weight_scale") or name.endswith(".weight_scale_inv")) and name not in scales_seen:
            raise ValueError(f"{name}: a scale without its payload")
    return pairs


def audit_mimo(root: Path, snapshot: Path, config: dict, world: int,
               bandwidth_gbps: float) -> tuple[dict, str]:
    num_layers = int(config["num_hidden_layers"])
    drafts = int(config.get("num_nextn_predict_layers", 0))
    hidden = int(config["hidden_size"])
    experts = int(config["n_routed_experts"])
    topk = int(config["num_experts_per_tok"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    swa_kv_heads = int(config.get("swa_num_key_value_heads", kv_heads))
    head_dim = int(config["head_dim"])
    v_head_dim = int(config.get("v_head_dim", head_dim))
    pattern = [int(x) for x in config["hybrid_layer_pattern"]]
    moe_pattern = [int(x) for x in config["moe_layer_freq"]]
    window = int(config.get("sliding_window", 0))
    if len(pattern) != num_layers or len(moe_pattern) != num_layers:
        raise ValueError("hybrid_layer_pattern / moe_layer_freq do not match num_hidden_layers")
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "fp8" or quant.get("store_dtype") != "mxfp4" \
            or list(quant.get("weight_block_size", [])) != [128, 128] or int(quant.get("mxfp4_block_size", 32)) != 32:
        raise ValueError("quantization_config is not the release's fp8 [128, 128] + MXFP4 expert format")

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    tensors: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    file_bytes: dict[str, int] = defaultdict(int)
    for filename in files:
        for name, meta in read_header(snapshot / filename).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor in safetensors headers: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(f"index maps {name} to {weight_map.get(name)!r}, not {filename!r}")
            tensors[name] = (filename, meta["dtype"], tuple(int(v) for v in meta.get("shape", [])))
    if set(tensors) != set(weight_map):
        missing = set(weight_map) - set(tensors)
        extra = set(tensors) - set(weight_map)
        raise ValueError(f"index/header mismatch: missing={len(missing)} extra={len(extra)}")

    pairs = validate_mimo_pairs(tensors, kv_heads)
    class_bytes: dict[str, int] = defaultdict(int)
    class_count: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    unmatched: list[tuple[str, str, tuple[int, ...]]] = []
    inventory: dict[str, dict] = {}
    total = 0
    for name, (filename, dtype, shape) in sorted(tensors.items()):
        if dtype not in DT_BYTES:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        size = numel(shape) * DT_BYTES[dtype]
        cls = classify_mimo(name, num_layers)
        class_bytes[cls] += size
        class_count[cls] += 1
        dtype_bytes[dtype] += size
        file_bytes[filename] += size
        total += size
        base = None
        for suffix in (".weight", ".weight_scale", ".weight_scale_inv"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        inventory[name] = {"file": filename, "dtype": dtype, "shape": list(shape), "class": cls, "nbytes": size,
                           "fmt": pairs[base]["fmt"] if base in pairs else "plain"}
        if cls == "other":
            unmatched.append((name, dtype, shape))

    fmt_count: dict[str, int] = defaultdict(int)
    for info in pairs.values():
        fmt_count[info["fmt"]] += 1
    moe_layers = sum(moe_pattern)
    if fmt_count.get("mxfp4", 0) != moe_layers * experts * 3:
        raise ValueError(f"{fmt_count.get('mxfp4', 0)} MXFP4 matrices; expected {moe_layers * experts * 3}")
    swa_layers = sum(pattern)
    sinks = class_count.get("attention_sink", 0)
    if sinks != swa_layers:
        raise ValueError(f"{sinks} attention_sink_bias tensors; expected one per sliding-window layer ({swa_layers})")

    # ---- placement (docs/mimo_v26_flash_plan.md §2) --------------------------
    # Sharded by W: the fused qkv_proj (its pre-sharded chunks), o_proj
    # (columns), the sink (heads), the dense MLPs and every expert
    # (intermediate slices), the head (vocab). Replicated: the router, the
    # norms, the embedding, the draft's enorm / hnorm / eh_proj / norms.
    # The encoders and the draft layers past the first are never loaded.
    def rank_terms(w: int) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for name, meta in inventory.items():
            cls = meta["class"]
            size = meta["nbytes"]
            if cls in ("routed_expert", "dense_mlp", "attention_qkv", "attention_o", "attention_sink", "lm_head"):
                out[cls] += size / w
            elif cls == "draft":
                tail = MIMO_DRAFT_RE.match(name).group(2)
                if tail.startswith(("self_attn.qkv_proj.", "self_attn.o_proj.", "self_attn.attention_sink_bias", "mlp.")):
                    out["draft"] += size / w
                else:
                    out["draft"] += size
            elif cls in ("encoder", "draft_unserved"):
                continue
            else:  # router, norm, embed: replicated
                out[cls] += size
        out["total"] = sum(out.values())
        return dict(out)

    def traffic_terms(w: int) -> dict[str, float]:
        """Per token per rank at batch 1, T=1: the resident set minus the
        experts a token does not route to and minus the embedding (one row);
        the draft pass is listed apart."""
        terms = dict(rank_terms(w))
        terms["routed_expert"] *= topk / experts
        terms["embed"] = 0.0
        draft_pass = terms.get("draft", 0.0)
        terms["draft"] = 0.0
        terms["total"] = sum(v for k, v in terms.items() if k != "total")
        terms["draft_pass"] = draft_pass
        return terms

    worlds = sorted({2, 4, world})
    resident = {w: rank_terms(w) for w in worlds}
    traffic = {w: traffic_terms(w) for w in worlds}

    # The paged K/V per context token per rank (every layer's rows in bf16,
    # the sliding-window layers' too — read through the window; the draft
    # layer's rows with MTP): kv heads / W x (192 + 128) x 2 bytes.
    def kv_per_token(w: int) -> float:
        per_head = (head_dim + v_head_dim) * 2
        ga = (num_layers - swa_layers) * (kv_heads / w) * per_head
        swa = swa_layers * (swa_kv_heads / w) * per_head
        draft = (swa_kv_heads / w) * per_head if drafts else 0
        return ga + swa + draft

    summary = {
        "arch": "mimo_v2",
        "model": model_label(root, snapshot),
        "files": len(files),
        "tensors": len(tensors),
        "total_bytes": total,
        "class_bytes": dict(class_bytes),
        "class_count": dict(class_count),
        "dtype_bytes": dict(dtype_bytes),
        "pairs": dict(fmt_count),
        "kv_bytes_per_token": {str(w): kv_per_token(w) for w in worlds},
        "resident_rank_bytes": {str(w): resident[w] for w in worlds},
        "traffic_rank_bytes": {str(w): traffic[w] for w in worlds},
        "world": world,
        "bandwidth_gbps": bandwidth_gbps,
        "unmatched": len(unmatched),
        "inventory": inventory,
    }

    def ms(nbytes: float) -> float:
        return nbytes / (bandwidth_gbps * 1e9) * 1000

    shard_sizes = sorted(file_bytes.items())
    lines = [
        "# MiMo-V2.6-Flash Checkpoint Budget Report",
        "",
        f"- Model revision: `{summary['model']}`",
        f"- Files: {len(files)} shards; tensors: {len(tensors):,}",
        f"- Total weights: **{total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)**",
        "- Generated by `tools/checkpoint_audit.py`; no tensor payloads were read.",
        f"- Layers: {num_layers} ({swa_layers} sliding-window of {window} tokens with a per-head sink, "
        f"{num_layers - swa_layers} global; {moe_layers} MoE, {num_layers - moe_layers} dense) + {drafts} draft layers "
        f"(the first served); experts {experts} top-{topk}, no shared expert; hidden {hidden}; "
        f"{heads} query heads over {kv_heads} global / {swa_kv_heads} sliding-window kv heads, qk {head_dim} / v {v_head_dim}.",
        f"- Quantized pairs: {fmt_count.get('mxfp4', 0):,} MXFP4 matrices (U8 e2m1 codes + e8m0 per 32), "
        f"{fmt_count.get('fp8', 0):,} fp8 matrices (e4m3 + F32 on the 128x128 grid, the fused qkv_proj's tiled per "
        f"chunk); every scale checked against its payload.",
        "",
        "## Storage inventory",
        "",
        "| class | tensors | GB | share |",
        "|---|---:|---:|---:|",
    ]
    for name, size in sorted(class_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {class_count[name]:,} | {size / 1e9:.3f} | {100 * size / total:.2f}% |")
    lines.extend(["", "## Storage by dtype", "", "| dtype | GB |", "|---|---:|"])
    for dtype, size in sorted(dtype_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {dtype} | {size / 1e9:.3f} |")
    lines.extend([
        "",
        "## Shards",
        "",
        f"- {len(shard_sizes)} shards from {min(s for _, s in shard_sizes) / 1e9:.2f} to "
        f"{max(s for _, s in shard_sizes) / 1e9:.2f} GB (the expert shards hold four experts of every "
        "layer each; the first also carries the embedding, the head, the dense layer and the encoders).",
        "",
        "## Resident bytes per rank",
        "",
        "Placement follows docs/mimo_v26_flash_plan.md §2: the fused qkv_proj (its "
        "pre-sharded chunks), o_proj (columns), the sink (heads), the dense MLPs and "
        "every routed expert (intermediate slices) and the head (vocabulary) divide by "
        "W; the router, the norms, the embedding and the draft's fusion tensors are "
        "replicated; the vision and audio encoders and the draft layers past the first "
        "are never loaded.",
        "",
        "| class | " + " | ".join(f"TP={w} GiB" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    keys = ("routed_expert", "attention_qkv", "attention_o", "attention_sink", "dense_mlp", "router", "norm",
            "embed", "lm_head", "draft", "other")
    for key in keys:
        if any(resident[w].get(key, 0) for w in worlds):
            lines.append(f"| {key} | " + " | ".join(f"{resident[w].get(key, 0) / 2**30:.2f}" for w in worlds) + " |")
    lines.append("| **weights** | " + " | ".join(f"**{resident[w]['total'] / 2**30:.2f}**" for w in worlds) + " |")
    lines.extend([
        "",
        "The paged K/V cache costs " + ", ".join(f"**{kv_per_token(w) / 1024:.1f} KiB per context token per rank at TP={w}**" for w in worlds) +
        " (every layer's bf16 K 192 / V 128 rows for the rank's kv heads, the draft's included). The CUDA "
        "context, the prefix cache and the bus staging come on top; `dgpp-serve --memory-plan` is the authority.",
        "",
        "## Decode traffic model (batch size 1)",
        "",
        f"Per token per rank at T=1: the resident set minus the experts a token does not route to "
        f"(top-{topk} of {experts}) and minus the embedding (one row). The draft pass (the first draft "
        "layer, then the shared head once more) is listed apart. The bf16 o_proj, head and eh_proj stream "
        "their 12-bit companions under `engine.bf16_weights = bf12` (0.75 of the bytes listed).",
        "",
        "| class | " + " | ".join(f"TP={w} MB/token" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    for key in keys:
        if any(traffic[w].get(key, 0) for w in worlds):
            lines.append(f"| {key} | " + " | ".join(f"{traffic[w].get(key, 0) / 1e6:,.1f}" for w in worlds) + " |")
    lines.append("| **total** | " + " | ".join(f"**{traffic[w]['total'] / 1e6:,.1f}**" for w in worlds) + " |")
    lines.append(f"| floor at {bandwidth_gbps:.0f} GB/s | " + " | ".join(f"**{ms(traffic[w]['total']):.1f} ms**" for w in worlds) + " |")
    lines.append("| replicated share | " + " | ".join(
        f"{100 * sum(traffic[w].get(k, 0) for k in ('router', 'norm')) / traffic[w]['total']:.0f}%"
        for w in worlds) + " |")
    lines.append("| draft pass (apart) | " + " | ".join(f"{traffic[w]['draft_pass'] / 1e6:,.1f} MB, +{ms(traffic[w]['draft_pass']):.1f} ms" for w in worlds) + " |")
    lines.extend([
        "",
        "This is a weight-bandwidth floor: the collectives (two folds per layer, the "
        "draft and the head), the cache reads and the kernels add to it.",
        "",
        "## Reconciliation and exclusions",
        "",
        f"- Unmatched tensors: **{len(unmatched)}**.",
        f"- Quantized pairs: **{len(pairs):,}** — every F8_E4M3 `.weight` has its F32 `.weight_scale_inv` of the "
        "contract's grid, every U8 `.weight` its U8 `.weight_scale`, every scale its payload.",
        f"- Encoder tensors ({class_count.get('encoder', 0)}, {class_bytes.get('encoder', 0) / 1e9:.2f} GB) and the "
        f"draft layers past the first ({class_count.get('draft_unserved', 0)}, "
        f"{class_bytes.get('draft_unserved', 0) / 1e9:.2f} GB) are present in the files and never loaded.",
    ])
    if unmatched:
        lines.extend(["", f"## First {min(80, len(unmatched))} unmatched names", ""])
        lines.extend(f"- `{name}` [{dtype}] {shape}" for name, dtype, shape in unmatched[:80])
    return summary, "\n".join(lines) + "\n"


DT_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F16": 2,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I32": 4,
    "U32": 4,
    "U8": 1,
    "I8": 1,
}

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
)
MHC_RE = re.compile(r"(^|\.)(hc_attn|hc_ffn)_(base|fn|scale)|mapping_proj")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT)
    parser.add_argument("--world", type=int, default=DEFAULT_WORLD)
    parser.add_argument("--bandwidth-gbps", type=float, default=DEFAULT_BANDWIDTH_GBPS)
    parser.add_argument(
        "--no-write", action="store_true", help="print only; do not update artifacts/docs"
    )
    return parser.parse_args(argv)


def resolve_snapshot(root: Path) -> Path:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return root
    main_ref = root / "refs" / "main"
    if main_ref.is_file():
        selected = snapshots / main_ref.read_text(encoding="utf-8").strip()
        if selected.is_dir():
            return selected
    candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"cannot select snapshot under {root}: expected refs/main or one directory"
        )
    return candidates[0]


def model_label(root: Path, snapshot: Path) -> str:
    name = root.name
    if name.startswith("models--"):
        name = name.removeprefix("models--").replace("--", "/")
    return f"{name}@{snapshot.name}"


def numel(shape: Iterable[int]) -> int:
    return math.prod(int(d) for d in shape)


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors header length: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"short safetensors header: {path}")
        return json.loads(raw_header.decode("utf-8"))


def classify(name: str, layer_types: list[str]) -> str:
    """Return a mutually exclusive storage/traffic class for a tensor."""
    if name.startswith("model.visual."):
        return "vision"
    if name == "lm_head.weight":
        return "lm_head"
    if name.startswith("model.language_model.embed_tokens."):
        return "embed"
    if name.startswith("model.language_model.norm."):
        return "norm"

    match = LAYER_RE.match(name)
    if not match:
        return "other"
    layer = int(match.group(1))
    tail = match.group(2)
    if layer >= len(layer_types):
        return "mtp"
    if ".experts." in tail:
        return "routed_expert"
    if ".shared_experts." in tail:
        return "shared_expert"
    if tail.startswith("mlp.gate.") or "e_score_correction_bias" in tail:
        return "router"
    if tail.startswith("self_attn.indexer."):
        return "dsa_indexer"
    if MHC_RE.search(tail):
        return "mhc"
    if tail.startswith("self_attn."):
        return "kda" if layer_types[layer] == "linear_attention" else "dsa"
    if tail.startswith("mlp."):
        return "dense_mlp"
    if "layernorm" in tail:
        return "norm"
    return "other"


def occupancy_compositions(total: int, capacities: tuple[int, ...]):
    if len(capacities) == 1:
        if total <= capacities[0]:
            yield (total,)
        return
    for count in range(min(total, capacities[0]) + 1):
        for rest in occupancy_compositions(total - count, capacities[1:]):
            yield (count,) + rest


def expected_busiest_rank(experts: int, topk: int, world: int) -> float:
    """Expected max rank occupancy for top-k draws without replacement."""
    if experts <= 0 or topk <= 0 or world <= 0 or topk > experts:
        raise ValueError("invalid expert placement geometry")
    base, remainder = divmod(experts, world)
    capacities = tuple(base + (rank < remainder) for rank in range(world))
    denominator = math.comb(experts, topk)
    weighted = 0
    counted = 0
    for occupancy in occupancy_compositions(topk, capacities):
        ways = math.prod(
            math.comb(capacity, count)
            for capacity, count in zip(capacities, occupancy)
        )
        counted += ways
        weighted += max(occupancy) * ways
    if counted != denominator:
        raise AssertionError("expert occupancy probabilities did not reconcile")
    return weighted / denominator


def validate_expert_geometry(inventory: dict[str, dict], experts: int) -> int:
    """Validate equal-size whole experts within every routed base layer."""
    by_layer: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for name, meta in inventory.items():
        if meta["class"] != "routed_expert":
            continue
        match = EXPERT_RE.match(name)
        if not match:
            raise ValueError(f"routed-expert tensor has unexpected name: {name}")
        layer, expert = (int(value) for value in match.groups())
        by_layer[layer][expert] += int(meta["nbytes"])
    if not by_layer:
        raise ValueError("no routed-expert layers found")
    expected_ids = set(range(experts))
    for layer, sizes in by_layer.items():
        if set(sizes) != expected_ids:
            raise ValueError(
                f"layer {layer} has {len(sizes)} routed experts; expected {experts}"
            )
        if len(set(sizes.values())) != 1:
            raise ValueError(f"layer {layer} routed experts have unequal byte sizes")
    return len(by_layer)


def audit(root: Path, world: int, bandwidth_gbps: float) -> tuple[dict, str]:
    if world <= 0 or bandwidth_gbps <= 0:
        raise ValueError("world and bandwidth must be positive")
    snapshot = resolve_snapshot(root)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    arch = (config.get("architectures") or [""])[0]
    if arch.startswith("Qwen4Exp"):
        return audit_qwen(root, snapshot, config, world, bandwidth_gbps)
    if arch.startswith("GlmMoeDsa"):
        return audit_glm53(root, snapshot, config, world, bandwidth_gbps)
    if arch.startswith("DeepseekV41"):
        return audit_dsv41(root, snapshot, config, world, bandwidth_gbps)
    if arch.startswith("MiMoV2"):
        return audit_mimo(root, snapshot, config, world, bandwidth_gbps)
    text_config = config["text_config"]
    layer_types = text_config["layer_types"]
    if len(layer_types) != int(text_config["num_hidden_layers"]):
        raise ValueError("layer_types does not match num_hidden_layers")

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    tensors: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for filename in files:
        for name, meta in read_header(snapshot / filename).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor in safetensors headers: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(
                    f"index maps {name} to {weight_map.get(name)!r}, not {filename!r}"
                )
            tensors[name] = (
                filename,
                meta["dtype"],
                tuple(int(value) for value in meta.get("shape", [])),
            )
    if set(tensors) != set(weight_map):
        missing = set(weight_map) - set(tensors)
        extra = set(tensors) - set(weight_map)
        raise ValueError(
            f"index/header mismatch: missing={len(missing)} extra={len(extra)}"
        )

    class_bytes: dict[str, int] = defaultdict(int)
    class_count: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    unmatched: list[tuple[str, str, tuple[int, ...]]] = []
    inventory: dict[str, dict] = {}
    total = 0
    for name, (filename, dtype, shape) in sorted(tensors.items()):
        if dtype not in DT_BYTES:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        size = numel(shape) * DT_BYTES[dtype]
        tensor_class = classify(name, layer_types)
        class_bytes[tensor_class] += size
        class_count[tensor_class] += 1
        dtype_bytes[dtype] += size
        total += size
        inventory[name] = {
            "file": filename,
            "dtype": dtype,
            "shape": list(shape),
            "class": tensor_class,
            "nbytes": size,
        }
        if tensor_class == "other":
            unmatched.append((name, dtype, shape))

    fp8_weights = [
        (name, shape)
        for name, (_filename, dtype, shape) in tensors.items()
        if dtype == "F8_E4M3" and name.endswith(".weight")
    ]
    invalid_scales: list[str] = []
    for name, shape in fp8_weights:
        scale_name = f"{name}_scale_inv"
        scale = tensors.get(scale_name)
        if len(shape) != 2 or scale is None:
            invalid_scales.append(name)
            continue
        _scale_file, scale_dtype, scale_shape = scale
        expected_shape = tuple(math.ceil(dimension / 128) for dimension in shape)
        if scale_dtype != "F32" or scale_shape != expected_shape:
            invalid_scales.append(name)
    if invalid_scales:
        raise ValueError(
            f"{len(invalid_scales)} FP8 weights have missing/invalid 128x128 F32 scales; "
            f"first={invalid_scales[0]}"
        )
    scale_names = {name for name in tensors if name.endswith(".weight_scale_inv")}
    expected_scale_names = {f"{name}_scale_inv" for name, _shape in fp8_weights}
    orphan_scales = scale_names - expected_scale_names
    if orphan_scales:
        raise ValueError(f"orphan FP8 inverse scale: {min(orphan_scales)}")

    experts = int(text_config["n_routed_experts"])
    topk = int(text_config["num_experts_per_tok"])
    routed_layers = validate_expert_geometry(inventory, experts)
    routed_total = class_bytes["routed_expert"] * topk / experts
    sharded_classes = ("shared_expert", "dense_mlp", "kda", "dsa", "lm_head")
    sharded_total = sum(class_bytes[name] for name in sharded_classes)
    replicated_classes = ("dsa_indexer", "router", "mhc", "norm")
    replicated_total = sum(class_bytes[name] for name in replicated_classes)
    unique_active_total = routed_total + sharded_total + replicated_total
    aggregate_cluster_total = routed_total + sharded_total + world * replicated_total

    busiest = expected_busiest_rank(experts, topk, world)
    routed_mean_rank = routed_total / world
    routed_critical_rank = routed_total * busiest / topk
    mean_rank = routed_mean_rank + sharded_total / world + replicated_total
    critical_rank = routed_critical_rank + sharded_total / world + replicated_total
    worst_rank = routed_total + sharded_total / world + replicated_total
    floor_ms = critical_rank / (bandwidth_gbps * 1e9) * 1000
    floor_tps = 1000 / floor_ms

    summary = {
        "model": model_label(root, snapshot),
        "files": len(files),
        "tensors": len(tensors),
        "total_bytes": total,
        "class_bytes": dict(class_bytes),
        "class_count": dict(class_count),
        "dtype_bytes": dict(dtype_bytes),
        "active_total_bytes": unique_active_total,
        "aggregate_cluster_bytes": aggregate_cluster_total,
        "mean_rank_bytes": mean_rank,
        "critical_rank_bytes": critical_rank,
        "worst_rank_bytes": worst_rank,
        "expected_busiest_experts": busiest,
        "routed_layers": routed_layers,
        "floor_ms": floor_ms,
        "floor_tps": floor_tps,
        "world": world,
        "bandwidth_gbps": bandwidth_gbps,
        "unmatched": len(unmatched),
        "fp8_scaled_weights": len(fp8_weights),
        "inventory": inventory,
    }

    lines = [
        "# GLM-5.3-Flash Checkpoint Budget Report",
        "",
        f"- Model revision: `{summary['model']}`",
        f"- Files: {len(files)} shards; tensors: {len(tensors):,}",
        f"- Total weights: **{total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)**",
        "- Generated by `tools/checkpoint_audit.py`; no tensor payloads were read.",
        "",
        "## Storage inventory",
        "",
        "| class | tensors | GB | share |",
        "|---|---:|---:|---:|",
    ]
    for name, size in sorted(class_bytes.items(), key=lambda item: -item[1]):
        lines.append(
            f"| {name} | {class_count[name]:,} | {size / 1e9:.3f} | "
            f"{100 * size / total:.2f}% |"
        )
    lines.extend(
        ["", "## Storage by dtype", "", "| dtype | GB |", "|---|---:|"]
    )
    for dtype, size in sorted(dtype_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {dtype} | {size / 1e9:.3f} |")

    lines.extend(
        [
            "",
            "## Base-text decode traffic model (batch size 1)",
            "",
            "Vision, input embedding, and MTP weights are excluded from the base decode "
            "step. The estimate assumes each active matrix is streamed once.",
            "",
            "| component | unique active bytes/token | placement |",
            "|---|---:|---|",
            f"| routed experts (top-{topk} of {experts}) | {routed_total / 1e9:.3f} GB | whole experts |",
            f"| shared experts | {class_bytes['shared_expert'] / 1e9:.3f} GB | TP-sharded |",
            f"| dense MLP | {class_bytes['dense_mlp'] / 1e9:.3f} GB | TP-sharded |",
            f"| KDA | {class_bytes['kda'] / 1e9:.3f} GB | TP-sharded |",
            f"| DSA/MLA core | {class_bytes['dsa'] / 1e9:.3f} GB | TP-sharded |",
            f"| DSA indexer | {class_bytes['dsa_indexer'] / 1e9:.3f} GB | replicated |",
            f"| lm_head | {class_bytes['lm_head'] / 1e9:.3f} GB | vocab-sharded |",
            f"| router + mHC + norms | {(replicated_total - class_bytes['dsa_indexer']) / 1e9:.3f} GB | replicated |",
            f"| **unique active total** | **{unique_active_total / 1e9:.3f} GB** | mixed |",
            "",
            f"Replicated modules are read on every rank. Their duplicate reads raise actual "
            f"aggregate cluster traffic to **{aggregate_cluster_total / 1e9:.3f} GB/token**; "
            "the unique-active total is an inventory value, not physical cluster traffic.",
            "",
            f"For TP={world}, uniformly distributed top-{topk} selections over contiguous "
            f"expert partitions give an exact expected busiest-rank occupancy of "
            f"**{busiest:.3f} experts/layer** (multivariate hypergeometric), versus "
            f"{topk / world:.3f} on the average rank.",
            f"The headers contain {routed_layers} routed base layers with all {experts} "
            "whole experts present and equal in byte size within each layer.",
            "",
            f"| per-rank view | GB/token | floor at {bandwidth_gbps:.0f} GB/s |",
            "|---|---:|---:|",
            f"| mean rank (includes replicated modules) | {mean_rank / 1e9:.3f} | "
            f"{mean_rank / (bandwidth_gbps * 1e9) * 1000:.2f} ms |",
            f"| **expected synchronized critical path** | **{critical_rank / 1e9:.3f}** | "
            f"**{floor_ms:.2f} ms / {floor_tps:.1f} token/s** |",
            f"| worst placement (all selected experts on one rank) | {worst_rank / 1e9:.3f} | "
            f"{worst_rank / (bandwidth_gbps * 1e9) * 1000:.2f} ms |",
            "",
            "The critical path sums the busiest rank independently at every MoE layer; "
            "it does not imply that one physical rank is busiest for the whole token.",
            "",
            "This is a weight-bandwidth floor, not a throughput prediction: collectives, "
            "cache misses, routing skew/correlation, kernels, state traffic, and MTP add work. "
            "Real router traces must replace the uniform model before a performance target is "
            "frozen.",
            "",
            "## Reconciliation and exclusions",
            "",
            f"- Unique base-text weights active per token: {unique_active_total / 1e9:.3f} GB.",
            f"- Aggregate physical cluster traffic after replication: "
            f"{aggregate_cluster_total / 1e9:.3f} GB/token.",
            f"- Vision storage excluded from text decode: {class_bytes['vision'] / 1e9:.3f} GB.",
            f"- MTP storage excluded until speculation is enabled: {class_bytes['mtp'] / 1e9:.3f} GB.",
            "- Input embedding storage excluded because decode reads selected rows, not the "
            f"full matrix: {class_bytes['embed'] / 1e9:.3f} GB.",
            f"- Unmatched tensors: **{len(unmatched)}**.",
            f"- FP8 scale contract: **{len(fp8_weights):,}/{len(fp8_weights):,}** "
            "E4M3 matrices have an F32 `weight_scale_inv` tensor with exact "
            "128×128-block geometry.",
        ]
    )
    if unmatched:
        limit = 80
        lines.extend(
            [
                "",
                f"## First {min(limit, len(unmatched))} unmatched names "
                f"(of {len(unmatched)})",
                "",
            ]
        )
        lines.extend(
            f"- `{name}` [{dtype}] {shape}"
            for name, dtype, shape in unmatched[:limit]
        )
    return summary, "\n".join(lines) + "\n"



# ---------------------------------------------------------------------------
# Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration), 2026-09-09.
# ---------------------------------------------------------------------------

QWEN_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
QWEN_MTP_RE = re.compile(r"^mtp\.(.+)$")
QWEN_HC_RE = re.compile(r"(^|\.)(attn_hyper_connection|mlp_hyper_connection|hyper_connection_mixer)\.")


def classify_qwen(name: str) -> str:
    """Mutually exclusive storage/traffic class for a Qwen4Exp tensor.

    The MTP head keeps its own class so the base decode step and the MTP
    replay are budgeted separately; inside it the same module names apply.
    """
    if name.startswith("model.visual."):
        return "vision"
    if name == "lm_head.weight":
        return "lm_head"
    if name.startswith("model.language_model.embed_tokens."):
        return "embed"
    if name.startswith("model.language_model.hyper_connection_mixer."):
        return "gr"
    if QWEN_MTP_RE.match(name):
        return "mtp"
    match = QWEN_LAYER_RE.match(name)
    if not match:
        return "other"
    tail = match.group(2)
    if ".experts." in tail:
        return "routed_expert"
    if tail.startswith("mlp.shared_expert."):
        return "shared_expert"
    if tail.startswith("mlp.gate.") or tail.startswith("mlp.shared_expert_gate."):
        return "router"
    if QWEN_HC_RE.search(tail):
        return "gr"
    if tail.startswith("ple.ple_embedding.ngram_embedding."):
        return "ple_table"
    if tail.startswith("ple."):
        return "ple"
    if tail.startswith("self_attn.indexer."):
        return "qsa_indexer"
    if tail.startswith("self_attn.q_proj.") or tail.startswith("self_attn.o_proj."):
        return "qsa_qo"
    if tail.startswith("self_attn.k_proj.") or tail.startswith("self_attn.v_proj."):
        return "qsa_kv"
    if tail.startswith("self_attn."):
        return "norm"
    if tail.startswith("linear_attn.norm."):
        return "norm"
    if tail.startswith("linear_attn."):
        return "gdn"
    return "other"


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def qwen_ngram_geometry(text_config: dict) -> dict:
    """The n-gram table's head sizes from config alone (the reference's
    `_find_nth_prime_after`): the h-th head's vocabulary is the (h+1)-th prime
    after ngram_vocab_size_base - 1; the total is padded to the divisor."""
    heads = (int(text_config["ngram_size"]) - 1) * int(text_config["heads_per_ngram"])
    base = int(text_config["ngram_vocab_size_base"])
    divisor = int(text_config["make_ngram_vocab_size_divisible_by"])
    sizes: list[int] = []
    prime = base - 1
    for _head in range(heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
    total = sum(sizes)
    padded = math.ceil(total / divisor) * divisor
    return {"heads": heads, "sizes": sizes, "total": total, "padded": padded}


def qwen_rank_bytes(class_bytes: dict[str, int], per_token: dict[str, float],
                    world: int, gr_sliced: bool) -> dict[str, float]:
    """Per-rank decode bytes per token at world `world`, the §2.1 placement:
    sharded classes /W, one kv head per rank (k/v halves for W >= 2), the
    replicated set (router, indexer, GR unless sliced, PLE norms/conv)."""
    kv_div = 2 if world >= 2 else 1
    out = {
        "routed_expert": per_token["routed_expert"] / world,
        "gdn": per_token["gdn"] / world,
        "gr": per_token["gr"] / (world if gr_sliced else 1),
        "lm_head": per_token["lm_head"] / world,
        "qsa_qo": per_token["qsa_qo"] / world,
        "qsa_kv": per_token["qsa_kv"] / kv_div,
        "qsa_indexer": per_token["qsa_indexer"],
        "shared_expert": per_token["shared_expert"] / world,
        "router": per_token["router"],
        "ple": per_token["ple"] / world,
        "ple_table_rows": per_token["ple_table_rows"] / world,
    }
    out["total"] = sum(out.values())
    return out


def audit_qwen(root: Path, snapshot: Path, config: dict, world: int,
               bandwidth_gbps: float) -> tuple[dict, str]:
    text_config = config["text_config"]
    layer_types = text_config["layer_types"]
    num_layers = int(text_config["num_hidden_layers"])
    if len(layer_types) != num_layers:
        raise ValueError("layer_types does not match num_hidden_layers")
    experts = int(text_config["num_experts"])
    topk = int(text_config["num_experts_per_tok"])
    hidden = int(text_config["hidden_size"])
    hc_count = int(text_config["hc_count"])
    ngram = qwen_ngram_geometry(text_config)
    ple_embed_dim = int(text_config.get("ple_embed_dim") or hidden)
    head_dim_per_ngram = ple_embed_dim // ngram["heads"]

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    tensors: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for filename in files:
        for name, meta in read_header(snapshot / filename).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor in safetensors headers: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(
                    f"index maps {name} to {weight_map.get(name)!r}, not {filename!r}"
                )
            tensors[name] = (
                filename,
                meta["dtype"],
                tuple(int(value) for value in meta.get("shape", [])),
            )
    if set(tensors) != set(weight_map):
        missing = set(weight_map) - set(tensors)
        extra = set(tensors) - set(weight_map)
        raise ValueError(
            f"index/header mismatch: missing={len(missing)} extra={len(extra)}"
        )

    dt_bytes = dict(DT_BYTES)
    dt_bytes["I64"] = 8
    class_bytes: dict[str, int] = defaultdict(int)
    class_count: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    unmatched: list[tuple[str, str, tuple[int, ...]]] = []
    inventory: dict[str, dict] = {}
    total = 0
    for name, (filename, dtype, shape) in sorted(tensors.items()):
        if dtype not in dt_bytes:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        size = numel(shape) * dt_bytes[dtype]
        tensor_class = classify_qwen(name)
        class_bytes[tensor_class] += size
        class_count[tensor_class] += 1
        dtype_bytes[dtype] += size
        total += size
        inventory[name] = {
            "file": filename,
            "dtype": dtype,
            "shape": list(shape),
            "class": tensor_class,
            "nbytes": size,
        }
        if tensor_class == "other":
            unmatched.append((name, dtype, shape))

    # FP8 contract: every E4M3 expert matrix carries a BF16 weight_scale_inv on
    # the ceil(dim/128) grid (the release's block 128x128); the n-gram shards
    # share one BF16 per-tensor weight_scale and are the only other E4M3.
    fp8_weights = [
        (name, shape)
        for name, (_filename, dtype, shape) in tensors.items()
        if dtype == "F8_E4M3" and name.endswith(".weight")
    ]
    invalid_scales: list[str] = []
    table_shards: list[tuple[str, tuple[int, ...]]] = []
    for name, shape in fp8_weights:
        if ".ngram_embedding.shard_" in name:
            table_shards.append((name, shape))
            continue
        scale = tensors.get(f"{name}_scale_inv")
        if len(shape) != 2 or scale is None:
            invalid_scales.append(name)
            continue
        _scale_file, scale_dtype, scale_shape = scale
        expected_shape = tuple(math.ceil(dimension / 128) for dimension in shape)
        if scale_dtype != "BF16" or scale_shape != expected_shape:
            invalid_scales.append(name)
    if invalid_scales:
        raise ValueError(
            f"{len(invalid_scales)} FP8 weights have missing/invalid 128x128 BF16 "
            f"scales; first={invalid_scales[0]}"
        )
    table_rows = sum(shape[0] for _name, shape in table_shards)
    table_widths = {shape[1] for _name, shape in table_shards}
    if table_widths != {head_dim_per_ngram}:
        raise ValueError(f"n-gram shard width {table_widths} != {head_dim_per_ngram}")
    if table_rows != ngram["padded"]:
        raise ValueError(
            f"n-gram shards hold {table_rows} rows; config derives {ngram['padded']}"
        )
    split_parts = int(text_config.get("split_ngram_parts", 512))
    if len(table_shards) != split_parts:
        raise ValueError(f"{len(table_shards)} n-gram shards; config says {split_parts}")
    table_scale_names = [
        name for name in tensors
        if name.endswith(".ngram_embedding.weight_scale")
    ]
    if len(table_scale_names) != 1 or tensors[table_scale_names[0]][1] != "BF16":
        raise ValueError("n-gram table: expected exactly one BF16 weight_scale")

    routed_layers = validate_expert_geometry(inventory, experts)
    if routed_layers != num_layers:
        raise ValueError(f"{routed_layers} routed layers; config has {num_layers}")

    # Per-token bytes of the full model (batch 1), before placement.
    per_token = {
        "routed_expert": class_bytes["routed_expert"] * topk / experts,
        "gdn": float(class_bytes["gdn"]),
        "gr": float(class_bytes["gr"]),
        "lm_head": float(class_bytes["lm_head"]),
        "qsa_qo": float(class_bytes["qsa_qo"]),
        "qsa_kv": float(class_bytes["qsa_kv"]),
        "qsa_indexer": float(class_bytes["qsa_indexer"]),
        "shared_expert": float(class_bytes["shared_expert"]),
        "router": float(class_bytes["router"]),
        "ple": float(class_bytes["ple"]),
        "ple_table_rows": float(ngram["heads"] * head_dim_per_ngram),
    }
    per_token["total"] = sum(per_token.values())
    worlds = sorted({2, 4, world})
    ranks = {w: qwen_rank_bytes(class_bytes, per_token, w, False) for w in worlds}
    ranks_sliced = {w: qwen_rank_bytes(class_bytes, per_token, w, True) for w in worlds}

    # MTP: the replay adds the head's own layer (its experts at top-k, its
    # attention, GR sites, fc projections) on top of the verify rows.
    mtp_bytes: dict[str, int] = defaultdict(int)
    for name, meta in inventory.items():
        if meta["class"] != "mtp":
            continue
        tail = QWEN_MTP_RE.match(name).group(1)
        if ".experts." in tail:
            mtp_bytes["routed_expert"] += meta["nbytes"]
        elif "hyper_connection" in tail:
            mtp_bytes["gr"] += meta["nbytes"]
        elif ".indexer." in tail:
            mtp_bytes["qsa_indexer"] += meta["nbytes"]
        elif ".k_proj." in tail or ".v_proj." in tail:
            mtp_bytes["qsa_kv"] += meta["nbytes"]
        elif ".q_proj." in tail or ".o_proj." in tail:
            mtp_bytes["qsa_qo"] += meta["nbytes"]
        elif tail.startswith("fc_"):
            mtp_bytes["fc"] += meta["nbytes"]
        elif "shared_expert." in tail:
            mtp_bytes["shared_expert"] += meta["nbytes"]
        elif "mlp.gate." in tail or "shared_expert_gate" in tail:
            mtp_bytes["router"] += meta["nbytes"]
        else:
            mtp_bytes["norm"] += meta["nbytes"]

    def mtp_rank(w: int) -> float:
        return (
            mtp_bytes["routed_expert"] * topk / experts / w
            + mtp_bytes["qsa_qo"] / w + mtp_bytes["qsa_kv"] / 2
            + mtp_bytes["qsa_indexer"] + mtp_bytes["gr"] + mtp_bytes["fc"]
            + mtp_bytes["shared_expert"] / w + mtp_bytes["router"]
        )

    # Resident bytes per rank: the sharded classes /W, the replicated set
    # once, the n-gram table /W (head-sharded), embeddings vocab-sharded,
    # the MTP head by the same rules, vision excluded.
    def resident_rank(w: int) -> dict[str, float]:
        out = {
            "routed_expert": class_bytes["routed_expert"] / w,
            "gdn": class_bytes["gdn"] / w,
            "qsa": (class_bytes["qsa_qo"]) / w + class_bytes["qsa_kv"] / 2
                   + class_bytes["qsa_indexer"],
            "shared_expert": class_bytes["shared_expert"] / w,
            "router": float(class_bytes["router"]),
            "gr": float(class_bytes["gr"]),
            "ple": float(class_bytes["ple"]),
            "ple_table": class_bytes["ple_table"] / w,
            "embed": class_bytes["embed"] / w,
            "lm_head": class_bytes["lm_head"] / w,
            "norm": float(class_bytes["norm"]),
            "mtp": (mtp_bytes["routed_expert"] + mtp_bytes["qsa_qo"]
                    + mtp_bytes["shared_expert"]) / w
                   + mtp_bytes["qsa_kv"] / 2 + mtp_bytes["qsa_indexer"]
                   + mtp_bytes["gr"] + mtp_bytes["fc"] + mtp_bytes["router"]
                   + mtp_bytes["norm"],
        }
        out["total"] = sum(out.values())
        return out

    resident = {w: resident_rank(w) for w in worlds}

    summary = {
        "arch": "qwen4_exp",
        "model": model_label(root, snapshot),
        "files": len(files),
        "tensors": len(tensors),
        "total_bytes": total,
        "class_bytes": dict(class_bytes),
        "class_count": dict(class_count),
        "dtype_bytes": dict(dtype_bytes),
        "per_token_bytes": per_token,
        "rank_bytes": {str(w): ranks[w] for w in worlds},
        "rank_bytes_gr_sliced": {str(w): ranks_sliced[w] for w in worlds},
        "resident_rank_bytes": {str(w): resident[w] for w in worlds},
        "mtp_rank_bytes": {str(w): mtp_rank(w) for w in worlds},
        "ngram": ngram,
        "world": world,
        "bandwidth_gbps": bandwidth_gbps,
        "unmatched": len(unmatched),
        "fp8_scaled_weights": len(fp8_weights) - len(table_shards),
        "inventory": inventory,
    }

    def ms(nbytes: float) -> float:
        return nbytes / (bandwidth_gbps * 1e9) * 1000

    lines = [
        "# Qwen3.8-Flash-Next Checkpoint Budget Report",
        "",
        f"- Model revision: `{summary['model']}`",
        f"- Files: {len(files)} shards; tensors: {len(tensors):,}",
        f"- Total weights: **{total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)**",
        "- Generated by `tools/checkpoint_audit.py`; no tensor payloads were read.",
        f"- Layers: {num_layers} ({layer_types.count('linear_attention')} GDN, "
        f"{num_layers - layer_types.count('linear_attention')} QSA); experts {experts} "
        f"top-{topk}; hidden {hidden}; {hc_count} residual branches.",
        "",
        "## Storage inventory",
        "",
        "| class | tensors | GB | share |",
        "|---|---:|---:|---:|",
    ]
    for name, size in sorted(class_bytes.items(), key=lambda item: -item[1]):
        lines.append(
            f"| {name} | {class_count[name]:,} | {size / 1e9:.3f} | "
            f"{100 * size / total:.2f}% |"
        )
    lines.extend(["", "## Storage by dtype", "", "| dtype | GB |", "|---|---:|"])
    for dtype, size in sorted(dtype_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {dtype} | {size / 1e9:.3f} |")

    lines.extend([
        "",
        "## N-gram table geometry (derived from config, checked against the headers)",
        "",
        f"- {ngram['heads']} hash heads, per-head vocabularies the primes after "
        f"{int(text_config['ngram_vocab_size_base']) - 1}: {ngram['sizes'][0]} … {ngram['sizes'][-1]}.",
        f"- Total rows {ngram['total']:,}, padded to {ngram['padded']:,}; the "
        f"{len(table_shards)} shards hold exactly that many rows of {head_dim_per_ngram} "
        f"E4M3 with one BF16 `weight_scale`.",
        "",
        "## Decode traffic model (batch size 1)",
        "",
        "Vision, the input embedding rows and the MTP head are excluded from the base "
        "step. Placement follows docs/qwen38_flash_next_plan.md §2.1: sharded classes "
        "divide by W, one kv head per rank, the router / indexer / GR / PLE norms "
        "replicated. `gr sliced` is the `gr_placement = sliced` variant of D3.",
        "",
        "| class | full model MB/token | " + " | ".join(f"TP={w} MB/rank" for w in worlds) + " |",
        "|---|---:|" + "---:|" * len(worlds),
    ])
    for key in ("routed_expert", "gdn", "gr", "lm_head", "qsa_qo", "qsa_kv",
                "qsa_indexer", "shared_expert", "router", "ple", "ple_table_rows"):
        lines.append(
            f"| {key} | {per_token[key] / 1e6:,.1f} | "
            + " | ".join(f"{ranks[w][key] / 1e6:,.1f}" for w in worlds) + " |"
        )
    lines.append(
        f"| **total** | **{per_token['total'] / 1e6:,.1f}** | "
        + " | ".join(f"**{ranks[w]['total'] / 1e6:,.1f}**" for w in worlds) + " |"
    )
    lines.append(
        f"| floor at {bandwidth_gbps:.0f} GB/s | | "
        + " | ".join(f"**{ms(ranks[w]['total']):.1f} ms**" for w in worlds) + " |"
    )
    lines.append(
        "| gr sliced: total / floor | | "
        + " | ".join(f"{ranks_sliced[w]['total'] / 1e6:,.1f} / {ms(ranks_sliced[w]['total']):.1f} ms" for w in worlds) + " |"
    )
    lines.append(
        "| replicated share (gr replicated) | | "
        + " | ".join(
            f"{100 * (ranks[w]['gr'] + ranks[w]['router'] + ranks[w]['qsa_indexer']) / ranks[w]['total']:.0f}%"
            for w in worlds) + " |"
    )
    lines.append(
        "| MTP head per replay | | "
        + " | ".join(f"{mtp_rank(w) / 1e6:,.1f} MB, +{ms(mtp_rank(w)):.2f} ms" for w in worlds) + " |"
    )
    lines.extend([
        "",
        "This is a weight-bandwidth floor: collectives (96 boundary folds per token, "
        "plus 97 more when GR is sliced), the QSA index scan at long context, state "
        "traffic and kernels add to it.",
        "",
        "## Resident bytes per rank",
        "",
        "| class | " + " | ".join(f"TP={w} GiB" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    for key in ("routed_expert", "ple_table", "gdn", "qsa", "gr", "embed", "lm_head",
                "mtp", "shared_expert", "router", "ple", "norm"):
        lines.append(f"| {key} | " + " | ".join(f"{resident[w][key] / 2**30:.2f}" for w in worlds) + " |")
    lines.append("| **total** | " + " | ".join(f"**{resident[w]['total'] / 2**30:.1f}**" for w in worlds) + " |")
    lines.extend([
        "",
        "The vision tower "
        f"({class_bytes['vision'] / 2**30:.2f} GiB) is not loaded. The CUDA context "
        "(~14 GiB), the KV pool (~13 KB per token per rank), the prefix cache and the "
        "bus staging come on top; the memory plan is the authority.",
        "",
        "## Reconciliation and exclusions",
        "",
        f"- Unmatched tensors: **{len(unmatched)}**.",
        f"- FP8 scale contract: **{summary['fp8_scaled_weights']:,}** E4M3 matrices carry a "
        "BF16 `weight_scale_inv` on the exact 128×128 grid; the n-gram shards carry the "
        "per-tensor scale.",
        f"- Routed layers with all {experts} equal-size experts: {routed_layers}.",
    ])
    if unmatched:
        lines.extend(["", f"## First {min(80, len(unmatched))} unmatched names", ""])
        lines.extend(f"- `{name}` [{dtype}] {shape}" for name, dtype, shape in unmatched[:80])
    return summary, "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Full GLM-5.3 (GlmMoeDsaForCausalLM), the int4/int8 g64 pack-quantized
# release, 2026-09-12 (docs/glm53_plan.md G0).

GLM53_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
GLM53_ATTENTION = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
GLM53_SHARDED_ATTENTION = ("q_b_proj", "kv_b_proj", "o_proj")   # by heads; q_a / kv_a replicated
GLM53_PACKED_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_shape")
GLM53_GROUP = 64


def classify_glm53(name: str, num_layers: int) -> str:
    """Mutually exclusive storage class of a GlmMoeDsa tensor (main and draft
    layers share the `model.layers.N.` prefix; N >= num_layers is the draft)."""
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.embed_tokens.weight":
        return "embed"
    if name == "model.norm.weight":
        return "norm"
    match = GLM53_LAYER_RE.match(name)
    if not match:
        return "other"
    layer = int(match.group(1))
    tail = match.group(2)
    if layer >= num_layers:
        return "mtp"
    if tail.startswith("self_attn.indexer."):
        return "dsa_indexer"
    if "layernorm" in tail:
        return "norm"
    if tail.startswith("self_attn."):
        return "dsa_attention"
    if ".experts." in tail:
        return "routed_expert"
    if ".shared_experts." in tail:
        return "shared_expert"
    if tail.startswith("mlp.gate."):
        return "router"
    if tail.startswith("mlp."):
        return "dense_mlp"
    return "other"


def glm53_packed_base(name: str) -> str | None:
    for suffix in GLM53_PACKED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def glm53_quant_groups(config: dict) -> list[tuple[re.Pattern, int]]:
    """(target regex, num_bits) per pack-quantized group of the config."""
    quant = config.get("quantization_config") or {}
    if quant.get("format") != "pack-quantized":
        raise ValueError(f"quantization_config.format is {quant.get('format')!r}, not pack-quantized")
    groups = []
    for key, group in sorted((quant.get("config_groups") or {}).items()):
        weights = group.get("weights") or {}
        bits = int(weights.get("num_bits", 0))
        if (weights.get("strategy") != "group" or int(weights.get("group_size", 0)) != GLM53_GROUP
                or not weights.get("symmetric", False) or weights.get("type") != "int" or bits not in (4, 8)):
            raise ValueError(f"quantization group {key} is not symmetric int4/int8 group-{GLM53_GROUP}")
        for target in group.get("targets") or []:
            if not target.startswith("re:"):
                raise ValueError(f"quantization group {key}: non-regex target {target!r}")
            groups.append((re.compile(target[3:]), bits))
    if not groups:
        raise ValueError("quantization_config has no pack-quantized groups")
    return groups


def validate_glm53_packed(tensors: dict[str, tuple[str, str, tuple[int, ...]]],
                          groups: list[tuple[re.Pattern, int]]) -> dict[str, dict]:
    """Check every packed triple and return {base: {bits, n, k, words, scales}}.

    The contract (compressed-tensors 0.18 pack-quantized): `weight_packed`
    I32 [N, K*bits/32], `weight_scale` BF16 [N, K/64], `weight_shape` I64
    [2]; the width follows from the scale's K/64 and the packed row, and it
    must be the bits of the config group whose target regex names the
    module. No member without its two siblings; nothing packed outside the
    groups' targets; every module a group targets is packed.
    """
    triples: dict[str, dict] = {}
    seen_bases = set()
    for name in tensors:
        base = glm53_packed_base(name)
        if base is not None:
            seen_bases.add(base)
    for base in sorted(seen_bases):
        members = {suffix: tensors.get(base + suffix) for suffix in GLM53_PACKED_SUFFIXES}
        if any(member is None for member in members.values()):
            raise ValueError(f"packed triple incomplete: {base}")
        _file, packed_dtype, packed_shape = members[".weight_packed"]
        _file, scale_dtype, scale_shape = members[".weight_scale"]
        _file, shape_dtype, shape_shape = members[".weight_shape"]
        if packed_dtype != "I32" or len(packed_shape) != 2:
            raise ValueError(f"{base}.weight_packed is {packed_dtype} {packed_shape}, not I32 [N, words]")
        if scale_dtype != "BF16" or len(scale_shape) != 2 or scale_shape[0] != packed_shape[0]:
            raise ValueError(f"{base}.weight_scale is {scale_dtype} {scale_shape}, not BF16 [N, K/{GLM53_GROUP}]")
        if shape_dtype != "I64" or tuple(shape_shape) != (2,):
            raise ValueError(f"{base}.weight_shape is {shape_dtype} {shape_shape}, not I64 [2]")
        n = packed_shape[0]
        k = scale_shape[1] * GLM53_GROUP
        if (packed_shape[1] * 32) % k:
            raise ValueError(f"{base}: {packed_shape[1]} words do not tile K={k}")
        bits = packed_shape[1] * 32 // k
        matched = [group_bits for pattern, group_bits in groups if pattern.search(base)]
        if len(matched) != 1:
            raise ValueError(f"{base}: {len(matched)} quantization groups target it")
        if bits != matched[0]:
            raise ValueError(f"{base}: packed at {bits} bits, the group says {matched[0]}")
        if (base + ".weight") in tensors:
            raise ValueError(f"{base}: both a packed triple and a plain .weight")
        triples[base] = {"bits": bits, "n": n, "k": k,
                         "words": numel(packed_shape) * 4, "scales": numel(scale_shape) * 2}
    for name in tensors:
        if name.endswith(".weight") and any(pattern.search(name[: -len(".weight")]) for pattern, _bits in groups):
            raise ValueError(f"{name}: a quantization group targets it but it is stored plain")
    return triples


def audit_glm53(root: Path, snapshot: Path, config: dict, world: int,
                bandwidth_gbps: float) -> tuple[dict, str]:
    text_config = config.get("text_config") or config
    num_layers = int(text_config["num_hidden_layers"])
    hidden = int(text_config["hidden_size"])
    experts = int(text_config["n_routed_experts"])
    topk = int(text_config["num_experts_per_tok"])
    dense_layers = int(text_config.get("first_k_dense_replace", 0))
    draft_layers = int(text_config.get("num_nextn_predict_layers", 0))
    indexer_types = list(text_config.get("indexer_types") or [])
    if len(indexer_types) != num_layers:
        raise ValueError("indexer_types does not match num_hidden_layers")
    index_layers = [i for i, kind in enumerate(indexer_types) if kind == "full"]
    groups = glm53_quant_groups(config)

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    tensors: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    file_bytes: dict[str, int] = defaultdict(int)
    for filename in files:
        for name, meta in read_header(snapshot / filename).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor in safetensors headers: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(
                    f"index maps {name} to {weight_map.get(name)!r}, not {filename!r}"
                )
            tensors[name] = (
                filename,
                meta["dtype"],
                tuple(int(value) for value in meta.get("shape", [])),
            )
    if set(tensors) != set(weight_map):
        missing = set(weight_map) - set(tensors)
        extra = set(tensors) - set(weight_map)
        raise ValueError(
            f"index/header mismatch: missing={len(missing)} extra={len(extra)}"
        )

    dt_bytes = dict(DT_BYTES)
    dt_bytes["I64"] = 8
    dt_bytes["I32"] = 4
    triples = validate_glm53_packed(tensors, groups)
    class_bytes: dict[str, int] = defaultdict(int)
    class_count: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    unmatched: list[tuple[str, str, tuple[int, ...]]] = []
    inventory: dict[str, dict] = {}
    total = 0
    for name, (filename, dtype, shape) in sorted(tensors.items()):
        if dtype not in dt_bytes:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        size = numel(shape) * dt_bytes[dtype]
        tensor_class = classify_glm53(name, num_layers)
        class_bytes[tensor_class] += size
        class_count[tensor_class] += 1
        dtype_bytes[dtype] += size
        file_bytes[filename] += size
        total += size
        base = glm53_packed_base(name)
        inventory[name] = {
            "file": filename,
            "dtype": dtype,
            "shape": list(shape),
            "class": tensor_class,
            "nbytes": size,
            "bits": triples[base]["bits"] if base in triples else 0,
        }
        if tensor_class == "other":
            unmatched.append((name, dtype, shape))

    # The census the plan's §1.5 states: 3 x 256 int4 experts and 5 + 3
    # int8 modules per packed layer, nothing packed on the dense layers, the
    # indexers, the routers or the draft.
    packed_layers = sorted({int(GLM53_LAYER_RE.match(base).group(1)) for base in triples})
    bits_count: dict[int, int] = defaultdict(int)
    packed_by_class: dict[str, int] = defaultdict(int)
    for base, info in triples.items():
        bits_count[info["bits"]] += 1
        packed_by_class[classify_glm53(base + ".weight_packed", num_layers)] += 1
    expected_packed = list(range(dense_layers, num_layers))
    if packed_layers != expected_packed:
        raise ValueError(f"packed layers {packed_layers[:3]}..{packed_layers[-3:]} != [{dense_layers}, {num_layers})")
    moe_layers = num_layers - dense_layers
    if packed_by_class.get("routed_expert", 0) != moe_layers * experts * 3:
        raise ValueError(f"{packed_by_class.get('routed_expert', 0)} packed expert matrices; expected {moe_layers * experts * 3}")
    if packed_by_class.get("dsa_attention", 0) != moe_layers * len(GLM53_ATTENTION):
        raise ValueError(f"{packed_by_class.get('dsa_attention', 0)} packed attention matrices; expected {moe_layers * 5}")
    if packed_by_class.get("shared_expert", 0) != moe_layers * 3:
        raise ValueError(f"{packed_by_class.get('shared_expert', 0)} packed shared-expert matrices; expected {moe_layers * 3}")
    if set(packed_by_class) - {"routed_expert", "dsa_attention", "shared_expert"}:
        raise ValueError(f"packed tensors outside the expected classes: {sorted(packed_by_class)}")
    expert_bits = {info["bits"] for base, info in triples.items() if ".mlp.experts." in base}
    other_bits = {info["bits"] for base, info in triples.items() if ".mlp.experts." not in base}
    if expert_bits != {4} or other_bits != {8}:
        raise ValueError(f"expert widths {sorted(expert_bits)}, attention/shared widths {sorted(other_bits)}")

    # ---- placement (docs/glm53_plan.md §2.1, the loader's rules) ---------
    # Sharded by W: routed experts (intermediate slices), shared experts,
    # q_b / kv_b / o_proj (heads), the dense MLP, the lm head (vocab).
    # Replicated: q_a, kv_a, the indexers, routers, norms, the embedding, the
    # draft's eh_proj. kv_b is held BF16 (the absorbed-attention bridge);
    # the draft's BF16 experts are requantized at load to the packed
    # layers' widths (D5: int4 g64 routed, int8 g64 shared).
    def packed_weight_bytes(numel_: int, bits: int) -> float:
        return numel_ * (bits / 8 + 2 / GLM53_GROUP)

    def rank_terms(w: int) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for name, meta in inventory.items():
            cls = meta["class"]
            size = meta["nbytes"]
            match = GLM53_LAYER_RE.match(name)
            tail = match.group(2) if match else ""
            base = glm53_packed_base(name)
            if base is not None and name.endswith(".weight_shape"):
                continue
            if cls == "routed_expert":
                out["routed_expert"] += size / w
            elif cls == "shared_expert":
                out["shared_expert"] += size / w
            elif cls == "dsa_attention":
                module = tail.split(".")[1]
                if module == "kv_b_proj":
                    if base is not None:
                        if name.endswith(".weight_packed"):
                            out["attention_sharded"] += triples[base]["n"] * triples[base]["k"] * 2 / w
                    else:
                        out["attention_sharded"] += size / w
                elif module in GLM53_SHARDED_ATTENTION:
                    out["attention_sharded"] += size / w
                else:
                    out["attention_replicated"] += size
            elif cls == "dense_mlp":
                out["dense_mlp"] += size / w
            elif cls == "lm_head":
                out["lm_head"] += size / w
            elif cls == "mtp":
                if tail.startswith("mlp.experts."):
                    out["draft"] += packed_weight_bytes(numel(meta["shape"]), 4) / w
                elif tail.startswith("mlp.shared_experts."):
                    out["draft"] += packed_weight_bytes(numel(meta["shape"]), 8) / w
                elif tail.startswith("self_attn.") and "indexer" not in tail and "layernorm" not in tail \
                        and tail.split(".")[1] in GLM53_SHARDED_ATTENTION:
                    out["draft"] += size / w
                else:
                    out["draft"] += size
            else:  # dsa_indexer, router, norm, embed, other: replicated
                out[cls] += size
        out["total"] = sum(out.values())
        return dict(out)

    def draft_expert_bytes(w: int) -> float:
        return sum(packed_weight_bytes(numel(meta["shape"]), 4) / w
                   for name, meta in inventory.items()
                   if meta["class"] == "mtp" and GLM53_LAYER_RE.match(name).group(2).startswith("mlp.experts."))

    def traffic_terms(w: int) -> dict[str, float]:
        """Per token per rank at batch 1: the resident set minus the experts not
        routed to (top-k of E) and minus the embedding (one row)."""
        terms = dict(rank_terms(w))
        terms["routed_expert"] *= topk / experts
        terms["draft"] -= draft_expert_bytes(w) * (1 - topk / experts)
        terms["embed"] = 0.0
        terms["total"] = sum(v for k, v in terms.items() if k != "total")
        return terms

    worlds = sorted({2, 4, world})
    resident = {w: rank_terms(w) for w in worlds}
    traffic = {w: traffic_terms(w) for w in worlds}

    summary = {
        "arch": "glm_moe_dsa",
        "model": model_label(root, snapshot),
        "files": len(files),
        "tensors": len(tensors),
        "total_bytes": total,
        "class_bytes": dict(class_bytes),
        "class_count": dict(class_count),
        "dtype_bytes": dict(dtype_bytes),
        "packed_triples": {str(bits): count for bits, count in sorted(bits_count.items())},
        "packed_layers": [dense_layers, num_layers],
        "index_layers": index_layers,
        "resident_rank_bytes": {str(w): resident[w] for w in worlds},
        "traffic_rank_bytes": {str(w): traffic[w] for w in worlds},
        "world": world,
        "bandwidth_gbps": bandwidth_gbps,
        "unmatched": len(unmatched),
        "inventory": inventory,
    }

    def ms(nbytes: float) -> float:
        return nbytes / (bandwidth_gbps * 1e9) * 1000

    layer_files = sorted(size for name, size in file_bytes.items() if name.startswith("layer-"))
    other_files = [(name, size) for name, size in sorted(file_bytes.items()) if not name.startswith("layer-")]
    lines = [
        "# GLM-5.3 (int4/int8 g64) Checkpoint Budget Report",
        "",
        f"- Model revision: `{summary['model']}`",
        f"- Files: {len(files)} shards; tensors: {len(tensors):,}",
        f"- Total weights: **{total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)**",
        "- Generated by `tools/checkpoint_audit.py`; no tensor payloads were read.",
        f"- Layers: {num_layers} ({dense_layers} dense, {moe_layers} MoE) + {draft_layers} draft; "
        f"experts {experts} top-{topk}; hidden {hidden}; {len(index_layers)} indexers "
        f"(layers {', '.join(str(i) for i in index_layers[:4])}, …, {index_layers[-1]}).",
        f"- Packed triples: {bits_count.get(4, 0):,} int4 (the routed experts) and {bits_count.get(8, 0):,} int8 "
        f"(attention and shared experts) on layers [{dense_layers}, {num_layers}); every triple checked against "
        "the config's quantization groups (I32 words, BF16 group-64 scales, I64 shapes).",
        "",
        "## Storage inventory",
        "",
        "| class | tensors | GB | share |",
        "|---|---:|---:|---:|",
    ]
    for name, size in sorted(class_bytes.items(), key=lambda item: -item[1]):
        lines.append(
            f"| {name} | {class_count[name]:,} | {size / 1e9:.3f} | "
            f"{100 * size / total:.2f}% |"
        )
    lines.extend(["", "## Storage by dtype", "", "| dtype | GB |", "|---|---:|"])
    for dtype, size in sorted(dtype_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {dtype} | {size / 1e9:.3f} |")
    lines.extend([
        "",
        "## Shards",
        "",
        f"- {len(layer_files)} per-layer shards from {layer_files[0] / 1e9:.2f} to {layer_files[-1] / 1e9:.2f} GB "
        f"(median {layer_files[len(layer_files) // 2] / 1e9:.2f} GB)." if layer_files else "- no per-layer shards",
    ])
    for name, size in other_files:
        lines.append(f"- `{name}`: {size / 1e9:.2f} GB.")
    lines.extend([
        "",
        "## Resident bytes per rank",
        "",
        "Placement follows docs/glm53_plan.md §2.1 and the loader: the routed and "
        "shared experts, `q_b` / `kv_b` / `o_proj`, the dense MLP and the lm head "
        "divide by W; `q_a` / `kv_a`, the indexers, routers, norms, the embedding "
        "and the draft's `eh_proj` are replicated; `kv_b` is held BF16 (the "
        "absorbed-attention bridge); the draft's BF16 experts are requantized at "
        "load to int4 g64 (routed) and int8 g64 (shared). The `weight_shape` "
        "tensors are read and dropped.",
        "",
        "| class | " + " | ".join(f"TP={w} GiB" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    keys = ("routed_expert", "attention_sharded", "attention_replicated", "shared_expert",
            "dense_mlp", "dsa_indexer", "router", "norm", "embed", "lm_head", "draft", "other")
    for key in keys:
        if any(resident[w].get(key, 0) for w in worlds):
            lines.append(f"| {key} | " + " | ".join(f"{resident[w].get(key, 0) / 2**30:.2f}" for w in worlds) + " |")
    lines.append("| **weights** | " + " | ".join(f"**{resident[w]['total'] / 2**30:.2f}**" for w in worlds) + " |")
    lines.append("| without the draft | " + " | ".join(f"{(resident[w]['total'] - resident[w].get('draft', 0)) / 2**30:.2f}" for w in worlds) + " |")
    lines.extend([
        "",
        "The CUDA context, the DSA pool (the memory plan measured 91.8 KiB per "
        "context token at bf16 and 52.6 KiB at fp8 on 2026-09-12), the prefix "
        "cache and the bus staging come on top; `dgpp-serve --memory-plan` is the "
        "authority.",
        "",
        "## Decode traffic model (batch size 1)",
        "",
        "Per token per rank: the resident set minus the experts a token does not "
        f"route to (top-{topk} of {experts}, in the main and the draft layer) and "
        "minus the embedding (one row).",
        "",
        "| class | " + " | ".join(f"TP={w} MB/token" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    for key in keys:
        if any(traffic[w].get(key, 0) for w in worlds):
            lines.append(f"| {key} | " + " | ".join(f"{traffic[w].get(key, 0) / 1e6:,.1f}" for w in worlds) + " |")
    lines.append("| **total** | " + " | ".join(f"**{traffic[w]['total'] / 1e6:,.1f}**" for w in worlds) + " |")
    lines.append(
        f"| floor at {bandwidth_gbps:.0f} GB/s | "
        + " | ".join(f"**{ms(traffic[w]['total']):.1f} ms**" for w in worlds) + " |"
    )
    lines.append(
        "| replicated share | "
        + " | ".join(
            f"{100 * (traffic[w].get('attention_replicated', 0) + traffic[w].get('dsa_indexer', 0) + traffic[w].get('router', 0)) / traffic[w]['total']:.0f}%"
            for w in worlds) + " |"
    )
    lines.extend([
        "",
        "This is a weight-bandwidth floor: the collectives (two folds per layer, "
        "the draft and the head), the DSA index scan at long context, the cache "
        "reads and the kernels add to it.",
        "",
        "## Reconciliation and exclusions",
        "",
        f"- Unmatched tensors: **{len(unmatched)}**.",
        f"- Packed contract: **{len(triples):,}** triples, each `weight_packed` I32 [N, K·bits/32] with a "
        f"BF16 `weight_scale` [N, K/{GLM53_GROUP}] and an I64 `weight_shape` [2]; the width of every triple "
        "is the width of the one quantization group that targets it; nothing a group targets is stored plain.",
        f"- Packed layers: [{dense_layers}, {num_layers}); the dense layers, the indexers, the routers, the norms, "
        "the embedding, the lm head and the draft layer are BF16 (the router bias F32).",
    ])
    if unmatched:
        lines.extend(["", f"## First {min(80, len(unmatched))} unmatched names", ""])
        lines.extend(f"- `{name}` [{dtype}] {shape}" for name, dtype, shape in unmatched[:80])
    return summary, "\n".join(lines) + "\n"


# ---- DeepSeek-V4.1-Flash (2026-09-13, docs/deepseek_v41_flash_plan.md G0) ----

DSV41_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
DSV41_DRAFT_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
DSV41_BLOCK = 32          # the fp8 32x32 grid and the MXFP4 block along K
DSV41_ENGRAM_DIM = 256


def classify_dsv41(name: str, num_layers: int) -> str:
    """Mutually exclusive storage class of a DeepSeek-V4.1-Flash tensor
    (backbone layers under `layers.N.`, the DSpark draft under `mtp.S.`)."""
    if name == "head.weight":
        return "lm_head"
    if name == "embed.weight":
        return "embed"
    if name == "norm.weight":
        return "norm"
    if name.startswith(("vision.", "aligner.", "image_")):
        return "vision"
    match = DSV41_LAYER_RE.match(name)
    if match is None:
        draft = DSV41_DRAFT_RE.match(name)
        if draft is None:
            return "other"
        tail = draft.group(2)
        if tail.startswith("ffn.experts."):
            return "draft_expert"
        return "draft"
    layer = int(match.group(1))
    if layer >= num_layers:
        return "other"
    tail = match.group(2)
    if tail.startswith("engram.embed."):
        return "engram_table"
    if tail.startswith("engram."):
        return "engram"
    if tail.startswith("attn.indexer."):
        return "indexer"
    if tail.startswith("attn.compressor."):
        return "compressor"
    if tail in ("attn.q_norm.weight", "attn.kv_norm.weight", "attn_norm.weight", "ffn_norm.weight"):
        return "norm"
    if tail.startswith("attn."):
        return "attention"
    if tail.startswith("ffn.experts."):
        return "routed_expert"
    if tail.startswith("ffn.shared_experts."):
        return "shared_expert"
    if tail.startswith("ffn.gate."):
        return "router"
    if tail.startswith("hc_"):
        return "mhc"
    return "other"


def validate_dsv41_pairs(tensors: dict[str, tuple[str, str, tuple[int, ...]]]) -> dict[str, dict]:
    """Check every quantized pair and return {base: {fmt, n, k, payload, scales}}.

    fp8 pairs: `.weight` F8_E4M3 [N, K] with `.scale` F8_E8M0 [ceil(N/32),
    ceil(K/32)]; MXFP4 pairs: `.weight` I8 [N, K/2] with `.scale` F8_E8M0
    [N, K/32]; the Engram tables: `.weight` F8_E4M3 [rows, 256] with `.scale`
    F8_E8M0 [rows, 8]. Every `.scale` must belong to exactly one such pair;
    every F8_E4M3 or I8 `.weight` must have its scale.
    """
    pairs: dict[str, dict] = {}
    scales_seen = set()
    for name, (_file, dtype, shape) in tensors.items():
        if not name.endswith(".weight") or dtype not in ("F8_E4M3", "I8"):
            continue
        base = name[: -len(".weight")]
        scale = tensors.get(base + ".scale")
        if scale is None:
            raise ValueError(f"{name}: a {dtype} payload without its .scale")
        _sfile, sdtype, sshape = scale
        if sdtype != "F8_E8M0" or len(sshape) != 2 or len(shape) != 2:
            raise ValueError(f"{base}: scale is {sdtype} {sshape}, not F8_E8M0 [rows, cols]")
        n = shape[0]
        if dtype == "I8":
            k = shape[1] * 2
            if tuple(sshape) != (n, k // DSV41_BLOCK) or k % DSV41_BLOCK:
                raise ValueError(f"{base}: MXFP4 scale {sshape} does not tile [{n}, {k}] by {DSV41_BLOCK}")
            fmt = "mxfp4"
        elif ".engram.embed" in base:
            k = shape[1]
            if k != DSV41_ENGRAM_DIM or tuple(sshape) != (n, k // DSV41_BLOCK):
                raise ValueError(f"{base}: Engram scale {sshape} does not tile [{n}, {k}] by {DSV41_BLOCK}")
            fmt = "engram"
        else:
            k = shape[1]
            expected = (-(-n // DSV41_BLOCK), -(-k // DSV41_BLOCK))
            if tuple(sshape) != expected:
                raise ValueError(f"{base}: fp8 scale {sshape} is not the {DSV41_BLOCK}x{DSV41_BLOCK} grid {expected}")
            fmt = "fp8"
        scales_seen.add(base + ".scale")
        pairs[base] = {"fmt": fmt, "n": n, "k": k, "payload": numel(shape), "scales": numel(sshape)}
    for name, (_file, dtype, _shape) in tensors.items():
        if name.endswith(".scale") and name not in scales_seen:
            raise ValueError(f"{name}: a scale without its payload")
        if dtype == "F8_E8M0" and not name.endswith(".scale"):
            raise ValueError(f"{name}: an e8m0 tensor that is not a .scale")
    return pairs


def audit_dsv41(root: Path, snapshot: Path, config: dict, world: int,
                bandwidth_gbps: float) -> tuple[dict, str]:
    text = config.get("text_config") or config
    num_layers = int(text["num_hidden_layers"])
    drafts = int(text.get("num_nextn_predict_layers", 0))
    hidden = int(text["hidden_size"])
    experts = int(text["n_routed_experts"])
    topk = int(text["num_experts_per_tok"])
    draft_experts = int(text.get("dspark_n_routed_experts", experts))
    draft_topk = int(text.get("dspark_num_experts_per_tok", topk))
    o_groups = int(text["o_groups"])
    engram_layers = [int(x) for x in text.get("engram_layer_ids", [])]
    engram_rows = [int(x) for x in text.get("engram_num_embeddings", [])]
    engram_heads = int(text.get("engram_n_heads", 0))
    ngram = int(text.get("engram_max_ngram_size", 1))
    kv_sources = [int(x) for x in text["kv_source_layer_ids"]]
    index_sources = [int(x) for x in text["index_source_layer_ids"]]
    ratios = [int(x) for x in text["compress_ratios"]]
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "fp8" or quant.get("expert_dtype") != "fp4" \
            or quant.get("scale_fmt") != "ue8m0" or list(quant.get("weight_block_size", [])) != [32, 32]:
        raise ValueError("quantization_config is not the release's fp8 [32, 32] ue8m0 + fp4 expert format")

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    tensors: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    file_bytes: dict[str, int] = defaultdict(int)
    for filename in files:
        for name, meta in read_header(snapshot / filename).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor in safetensors headers: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(f"index maps {name} to {weight_map.get(name)!r}, not {filename!r}")
            tensors[name] = (filename, meta["dtype"], tuple(int(v) for v in meta.get("shape", [])))
    if set(tensors) != set(weight_map):
        missing = set(weight_map) - set(tensors)
        extra = set(tensors) - set(weight_map)
        raise ValueError(f"index/header mismatch: missing={len(missing)} extra={len(extra)}")

    pairs = validate_dsv41_pairs(tensors)
    class_bytes: dict[str, int] = defaultdict(int)
    class_count: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    unmatched: list[tuple[str, str, tuple[int, ...]]] = []
    inventory: dict[str, dict] = {}
    total = 0
    for name, (filename, dtype, shape) in sorted(tensors.items()):
        if dtype not in DT_BYTES:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        size = numel(shape) * DT_BYTES[dtype]
        cls = classify_dsv41(name, num_layers)
        class_bytes[cls] += size
        class_count[cls] += 1
        dtype_bytes[dtype] += size
        file_bytes[filename] += size
        total += size
        base = name[: -len(".weight")] if name.endswith(".weight") else (name[: -len(".scale")] if name.endswith(".scale") else None)
        inventory[name] = {"file": filename, "dtype": dtype, "shape": list(shape), "class": cls, "nbytes": size,
                           "fmt": pairs[base]["fmt"] if base in pairs else "plain"}
        if cls == "other":
            unmatched.append((name, dtype, shape))

    # The census the plan's §1.10 states.
    fmt_count: dict[str, int] = defaultdict(int)
    for info in pairs.values():
        fmt_count[info["fmt"]] += 1
    if fmt_count.get("mxfp4", 0) != (num_layers * experts + drafts * draft_experts) * 3:
        raise ValueError(f"{fmt_count.get('mxfp4', 0)} MXFP4 matrices; expected {(num_layers * experts + drafts * draft_experts) * 3}")
    if fmt_count.get("engram", 0) != len(engram_layers):
        raise ValueError(f"{fmt_count.get('engram', 0)} Engram tables; expected {len(engram_layers)}")
    for layer_index, layer in enumerate(engram_layers):
        table = tensors.get(f"layers.{layer}.engram.embed.weight")
        if table is None or table[2][0] != engram_rows[layer_index]:
            raise ValueError(f"layers.{layer}.engram.embed.weight rows do not match engram_num_embeddings")

    # ---- placement (docs/deepseek_v41_flash_plan.md §2) ------------------
    # Sharded by W: routed / draft experts and the shared expert (intermediate
    # slices), wq_b / wo_a / wo_b (heads, output groups), the head (vocab),
    # the Engram wkv (its K columns follow the hash heads). Replicated: wq_a,
    # wkv, the indexers, compressors, routers, mHC coefficients, norms, the
    # embedding, main_proj, the Markov embedding, the confidence head. The
    # Engram tables are never resident (mmap'ed, sharded by hash head).
    SHARDED_ATTENTION = ("wq_b", "wo_a", "wo_b")

    def module_of(tail: str) -> str:
        parts = tail.split(".")
        return parts[1] if len(parts) > 1 else parts[0]

    def rank_terms(w: int) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for name, meta in inventory.items():
            cls = meta["class"]
            size = meta["nbytes"]
            if cls in ("routed_expert", "draft_expert", "shared_expert"):
                out[cls] += size / w
            elif cls == "attention":
                tail = DSV41_LAYER_RE.match(name).group(2)
                if module_of(tail) in SHARDED_ATTENTION:
                    out["attention_sharded"] += size / w
                else:
                    out["attention_replicated"] += size
            elif cls == "draft":
                tail = DSV41_DRAFT_RE.match(name).group(2)
                if tail.startswith("attn.") and module_of(tail) in SHARDED_ATTENTION:
                    out["draft"] += size / w
                elif tail.startswith("ffn.shared_experts."):
                    out["draft"] += size / w
                elif tail.startswith("markov_head.head."):
                    out["draft"] += size / w
                else:
                    out["draft"] += size
            elif cls == "lm_head":
                out["lm_head"] += size / w
            elif cls == "engram":
                tail = DSV41_LAYER_RE.match(name).group(2)
                out["engram"] += size / w if tail.startswith("engram.wkv.") else size
            elif cls == "engram_table":
                out["engram_table_mmap"] += size / w   # mapped, not resident: reported apart
            elif cls == "vision":
                continue  # not loaded (plan D9)
            else:  # indexer, compressor, router, mhc, norm, embed: replicated
                out[cls] += size
        out["total"] = sum(v for k, v in out.items() if k != "engram_table_mmap")
        return dict(out)

    def expert_slice_bytes(w: int, cls: str) -> float:
        return sum(meta["nbytes"] / w for meta in inventory.values() if meta["class"] == cls)

    def traffic_terms(w: int) -> dict[str, float]:
        """Per token per rank at batch 1, T=1: the resident set minus the
        experts a token does not route to and minus the embedding (one row),
        plus the token's Engram rows; the draft pass is listed apart."""
        terms = dict(rank_terms(w))
        terms["routed_expert"] *= topk / experts
        terms["draft_expert"] = expert_slice_bytes(w, "draft_expert") * draft_topk / draft_experts
        terms["embed"] = 0.0
        terms["engram_table_mmap"] = 0.0
        rows_per_rank = (ngram - 1) * engram_heads * len(engram_layers) / w if engram_heads else 0
        terms["engram_rows"] = rows_per_rank * (DSV41_ENGRAM_DIM + DSV41_ENGRAM_DIM // DSV41_BLOCK)
        draft_pass = terms["draft"] + terms["draft_expert"]
        terms["draft"] = 0.0
        terms["draft_expert"] = 0.0
        terms["total"] = sum(v for k, v in terms.items() if k not in ("total",))
        terms["draft_pass"] = draft_pass
        return terms

    worlds = sorted({2, 4, world})
    resident = {w: rank_terms(w) for w in worlds}
    traffic = {w: traffic_terms(w) for w in worlds}

    # The global KV cache per token (plan §1.3): per kv source a main-KV
    # entry (512 e2m1 + 32 e4m3 block scales) and an index key (128 e2m1 +
    # 4 e8m0 scales) per `ratio` tokens.
    head_dim = int(text["head_dim"])
    index_dim = int(text["index_head_dim"])
    kv_per_token = sum((head_dim / 2 + head_dim / 16 + index_dim / 2 + index_dim / 32) / ratios[l] for l in kv_sources)

    summary = {
        "arch": "deepseek_v41",
        "model": model_label(root, snapshot),
        "files": len(files),
        "tensors": len(tensors),
        "total_bytes": total,
        "class_bytes": dict(class_bytes),
        "class_count": dict(class_count),
        "dtype_bytes": dict(dtype_bytes),
        "pairs": dict(fmt_count),
        "kv_sources": kv_sources,
        "index_sources": index_sources,
        "kv_bytes_per_token": kv_per_token,
        "resident_rank_bytes": {str(w): resident[w] for w in worlds},
        "traffic_rank_bytes": {str(w): traffic[w] for w in worlds},
        "world": world,
        "bandwidth_gbps": bandwidth_gbps,
        "unmatched": len(unmatched),
        "inventory": inventory,
    }

    def ms(nbytes: float) -> float:
        return nbytes / (bandwidth_gbps * 1e9) * 1000

    shard_sizes = sorted(file_bytes.items())
    lines = [
        "# DeepSeek-V4.1-Flash Checkpoint Budget Report",
        "",
        f"- Model revision: `{summary['model']}`",
        f"- Files: {len(files)} shards; tensors: {len(tensors):,}",
        f"- Total weights: **{total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)**",
        "- Generated by `tools/checkpoint_audit.py`; no tensor payloads were read.",
        f"- Layers: {num_layers} backbone + {drafts} DSpark draft stages; experts {experts} top-{topk} "
        f"(draft {draft_experts} top-{draft_topk}); hidden {hidden}; {o_groups} output groups; "
        f"kv sources {kv_sources}, index sources {index_sources}; Engram at layers {engram_layers}.",
        f"- Quantized pairs: {fmt_count.get('mxfp4', 0):,} MXFP4 matrices (I8 codes + e8m0 per 32), "
        f"{fmt_count.get('fp8', 0):,} fp8 matrices (e4m3 + e8m0 on the 32x32 grid), "
        f"{fmt_count.get('engram', 0)} Engram tables (e4m3 rows + e8m0 per 32); every scale checked against its payload.",
        "",
        "## Storage inventory",
        "",
        "| class | tensors | GB | share |",
        "|---|---:|---:|---:|",
    ]
    for name, size in sorted(class_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {class_count[name]:,} | {size / 1e9:.3f} | {100 * size / total:.2f}% |")
    lines.extend(["", "## Storage by dtype", "", "| dtype | GB |", "|---|---:|"])
    for dtype, size in sorted(dtype_bytes.items(), key=lambda item: -item[1]):
        lines.append(f"| {dtype} | {size / 1e9:.3f} |")
    lines.extend([
        "",
        "## Shards",
        "",
        f"- {len(shard_sizes)} shards from {min(s for _, s in shard_sizes) / 1e9:.2f} to "
        f"{max(s for _, s in shard_sizes) / 1e9:.2f} GB; the two Engram tables are "
        + ", ".join(f"`{n}` ({s / 2**30:.2f} GiB)" for n, s in shard_sizes if s > 50e9) + ".",
        "",
        "## Resident bytes per rank",
        "",
        "Placement follows docs/deepseek_v41_flash_plan.md §2: the routed, draft and "
        "shared experts (intermediate slices), `wq_b` / `wo_a` / `wo_b` (heads and "
        "output groups), the head and the Markov head (vocabulary) and the Engram "
        "`wkv` (its hash heads' columns) divide by W; `wq_a`, `wkv`, the indexers, "
        "compressors, routers, mHC coefficients, norms, the embedding, `main_proj`, "
        "the Markov embedding and the confidence head are replicated; the Engram "
        "tables are mmap'ed from the NVMe (sharded by hash head) and never resident; "
        "the vision tower is not loaded.",
        "",
        "| class | " + " | ".join(f"TP={w} GiB" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    keys = ("routed_expert", "draft_expert", "attention_sharded", "attention_replicated", "shared_expert",
            "indexer", "compressor", "router", "mhc", "engram", "norm", "embed", "lm_head", "draft", "other")
    for key in keys:
        if any(resident[w].get(key, 0) for w in worlds):
            lines.append(f"| {key} | " + " | ".join(f"{resident[w].get(key, 0) / 2**30:.2f}" for w in worlds) + " |")
    lines.append("| **weights** | " + " | ".join(f"**{resident[w]['total'] / 2**30:.2f}**" for w in worlds) + " |")
    lines.append("| Engram tables, mmap'ed (not resident) | " + " | ".join(f"{resident[w].get('engram_table_mmap', 0) / 2**30:.2f}" for w in worlds) + " |")
    lines.extend([
        "",
        f"The global KV cache costs **{kv_per_token:.0f} bytes per context token per rank** "
        "(the four kv sources' main-KV entries and index keys at their ratios); the window "
        "rings, the compressor tails and the Engram context are per request slot. The CUDA "
        "context, the prefix cache and the bus staging come on top; `dgpp-serve --memory-plan` "
        "is the authority.",
        "",
        "## Decode traffic model (batch size 1)",
        "",
        f"Per token per rank at T=1: the resident set minus the experts a token does not route "
        f"to (top-{topk} of {experts}) and minus the embedding (one row), plus the Engram rows "
        f"gathered from the NVMe ({(ngram - 1) * engram_heads * len(engram_layers)} rows per token, "
        "head-sharded). The DSpark draft pass (three stages at top-"
        f"{draft_topk} of {draft_experts}, `main_proj`, the Markov and confidence heads) is listed apart; "
        "the shared head is read once more over the block's rows.",
        "",
        "| class | " + " | ".join(f"TP={w} MB/token" for w in worlds) + " |",
        "|---|" + "---:|" * len(worlds),
    ])
    for key in keys + ("engram_rows",):
        if any(traffic[w].get(key, 0) for w in worlds):
            lines.append(f"| {key} | " + " | ".join(f"{traffic[w].get(key, 0) / 1e6:,.1f}" for w in worlds) + " |")
    lines.append("| **total** | " + " | ".join(f"**{traffic[w]['total'] / 1e6:,.1f}**" for w in worlds) + " |")
    lines.append(f"| floor at {bandwidth_gbps:.0f} GB/s | " + " | ".join(f"**{ms(traffic[w]['total']):.1f} ms**" for w in worlds) + " |")
    lines.append("| replicated share | " + " | ".join(
        f"{100 * sum(traffic[w].get(k, 0) for k in ('attention_replicated', 'indexer', 'compressor', 'router', 'mhc', 'norm')) / traffic[w]['total']:.0f}%"
        for w in worlds) + " |")
    lines.append("| draft pass (apart) | " + " | ".join(f"{traffic[w]['draft_pass'] / 1e6:,.1f} MB, +{ms(traffic[w]['draft_pass']):.1f} ms" for w in worlds) + " |")
    lines.extend([
        "",
        "This is a weight-bandwidth floor: the collectives (two folds per layer, the "
        "draft and the head), the Engram gather latency, the selects, the cache reads "
        "and the kernels add to it.",
        "",
        "## Reconciliation and exclusions",
        "",
        f"- Unmatched tensors: **{len(unmatched)}**.",
        f"- Quantized pairs: **{len(pairs):,}** — every F8_E4M3 or I8 `.weight` has its F8_E8M0 `.scale` of the "
        "contract's shape, every `.scale` its payload, no e8m0 tensor outside a pair.",
        f"- Vision tensors ({class_count.get('vision', 0)}, {class_bytes.get('vision', 0) / 1e9:.2f} GB) are present in the file and never loaded.",
    ])
    if unmatched:
        lines.extend(["", f"## First {min(80, len(unmatched))} unmatched names", ""])
        lines.extend(f"- `{name}` [{dtype}] {shape}" for name, dtype, shape in unmatched[:80])
    return summary, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary, report = audit(args.model_dir, args.world, args.bandwidth_gbps)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"checkpoint audit failed: {error}", file=sys.stderr)
        return 1

    if not args.no_write:
        artifacts_dir = REPO_ROOT / "artifacts"
        docs_dir = REPO_ROOT / "docs"
        artifacts_dir.mkdir(exist_ok=True)
        docs_dir.mkdir(exist_ok=True)
        arch = summary.get("arch")
        if arch == "qwen4_exp":
            inventory_name, report_name = "qwen38_checkpoint_inventory.json", "qwen38_checkpoint_budget.md"
        elif arch == "glm_moe_dsa":
            inventory_name, report_name = "glm53_checkpoint_inventory.json", "checkpoint_budget_glm53.md"
        elif arch == "deepseek_v41":
            inventory_name, report_name = "dsv41_checkpoint_inventory.json", "checkpoint_budget_dsv41.md"
        elif arch == "mimo_v2":
            inventory_name, report_name = "mimo_checkpoint_inventory.json", "checkpoint_budget_mimo.md"
        else:
            inventory_name, report_name = "checkpoint_inventory.json", "checkpoint_budget.md"
        (artifacts_dir / inventory_name).write_text(
            json.dumps(summary["inventory"], separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (docs_dir / report_name).write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
