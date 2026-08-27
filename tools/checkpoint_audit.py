#!/usr/bin/env python3
"""Inventory a GLM-5 safetensors checkpoint and model decode traffic.

Only safetensors JSON headers are read. The report separates vision, MTP, and
base text weights and includes a placement-aware TP critical-rank estimate.

Usage:
  python3 tools/checkpoint_audit.py [model_dir]
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

DT_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F16": 2,
    "F8_E4M3": 1,
    "I32": 4,
    "U32": 4,
    "U8": 1,
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
        (artifacts_dir / "checkpoint_inventory.json").write_text(
            json.dumps(summary["inventory"], separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (docs_dir / "checkpoint_budget.md").write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
