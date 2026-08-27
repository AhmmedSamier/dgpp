#!/usr/bin/env python3
"""Generate artifacts/shardspec.json from a sharded safetensors checkpoint.

Walks the checkpoint directory, parses every safetensors header, classifies
tensor roles with model-family regex rules, and emits a single JSON plan:

  { "model": <id>, "shards": [{file,size_bytes,tensor_count}], "tensors":
    { "<name>": {shard, dtype, shape, role, nbytes} } }

Roles are opaque strings for C++ consumers; the classification table below is
the single place to extend when new model families land.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

# (role, compiled-regex) — first match wins; order matters.
ROLE_RULES = [
    ("embed", re.compile(r"\.embed_tokens\.")),
    ("lm_head", re.compile(r"^lm_head\.")),
    ("mtp_enorm", re.compile(r"\.enorm\.")),
    ("mtp_eh_proj", re.compile(r"\.eh_proj\.")),
    ("mtp_hnorm", re.compile(r"\.hnorm\.")),
    ("norm_final", re.compile(r"\.final_layernorm\.|\.norm\.$")),
    ("norm_attn", re.compile(r"\.input_layernorm\.")),
    ("norm_post", re.compile(r"\.post_attention_layernorm\.")),
    ("router_gate", re.compile(r"\.(mlp\.gate|moe_gate|gate_weight)\b")),
    ("router_bias", re.compile(r"\.(e_score_correction_bias|gate_bias)\b")),
    ("shared_expert_w13", re.compile(r"\.shared_expert\.(gate_proj|up_proj)\.")),
    ("shared_expert_w2", re.compile(r"\.shared_expert\.down_proj\.")),
    ("la_conv", re.compile(r"\.(conv1d|in_proj_qkvz|in_proj_a)\b")),
    ("la_dtba", re.compile(r"\.(A_log|dt_bias|a)\b")),
    ("la_norm", re.compile(r"\.(q_norm|k_norm)\.")),
    ("attn_qkvz", re.compile(r"\.in_proj_qkvz\.|\.qkvz\.")),
    ("attn_ba", re.compile(r"\.in_proj_ba\b")),
    ("attn_qkv", re.compile(r"\.(self_attn\.)?(q_a_proj|kv_a_proj|qkv_proj)\b")),
    ("attn_qb_kvbo", re.compile(r"\.(q_b_proj|kv_b_proj)\b")),
    ("attn_o", re.compile(r"\.o_proj\.")),
    ("indexer_wq_b", re.compile(r"\.indexer\.wq_b\.")),
    ("indexer_wk", re.compile(r"\.indexer\.wk\.")),
    ("indexer_weightsproj", re.compile(r"\.indexer\.weights_proj\.")),
    ("indexer_kpool", re.compile(r"\.indexer\.(k_pool|fp8_block_scales)\b")),
    ("experts_w13", re.compile(r"experts\.\d+\.(gate_up_proj|gate_proj|up_proj)")),
    ("experts_w2", re.compile(r"experts\.\d+\.down_proj")),
    ("mlp_w13", re.compile(r"\.(gate_up_proj|gate_proj|up_proj)\b")),
    ("mlp_w2", re.compile(r"\.down_proj\b")),
]

try:
    from safetensors import safe_open  # optional nicety

    def _read_header(path: Path):
        with safe_open(str(path), framework="pt") as f:
            return dict((k, f.get_slice(k).get_dtype()) for k in f.keys())
except ImportError:
    def _read_header(path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))


def classify(name: str) -> str:
    low = name.lower()
    for role, rx in ROLE_RULES:
        if rx.search(low):
            return role
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir")
    ap.add_argument("-o", "--out", default="artifacts/shardspec.json")
    ap.add_argument("--model-id", default=None,
                    help="override 'model' field (defaults to dir name)")
    args = ap.parse_args()

    root = Path(args.checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    files = sorted(root.glob("*.safetensors"))
    if not files:
        print(f"no safetensors shards under {root}", file=sys.stderr)
        return 2

    shards = []
    tensors = {}
    total_bytes = 0
    role_bytes = {}
    dtype_bytes = {}

    for path in files:
        hdr = _read_header(path)
        shard_idx = len(shards)
        # safetensors header does not store byte sizes per dtype directly;
        # recover nbytes via dtype width map.
        widths = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1,
                  "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}
        with open(path, "rb") as f:
            raw_len = struct.unpack("<Q", f.read(8))[0]
        size_bytes = path.stat().st_size
        shards.append({"file": path.name, "size_bytes": size_bytes,
                       "tensor_count": len(hdr)})
        for name, desc in sorted(hdr.items()):
            shape = desc["shape"]
            dtype = desc["dtype"]
            w = widths.get(dtype)
            if w is None:
                raise SystemExit(f"{path.name}:{name}: unknown dtype {dtype}")
            nbytes = w * eval_shape(shape)
            role = classify(name)
            tensors[name] = {"shard": shard_idx, "dtype": dtype,
                             "shape": shape, "role": role, "nbytes": nbytes}
            total_bytes += nbytes
            role_bytes[role] = role_bytes.get(role, 0) + nbytes
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + nbytes

    out = {
        "model": args.model_id or root.name,
        "checkpoint_dir": str(root),
        "num_shards": len(files),
        "total_nbytes": total_bytes,
        "bytes_by_role": dict(sorted(role_bytes.items(),
                                     key=lambda kv: -kv[1])),
        "bytes_by_dtype": dict(sorted(dtype_bytes.items(),
                                      key=lambda kv: -kv[1])),
        "shards": [{"file": s["file"], "size_bytes": s["size_bytes"]}
                   for s in shards],
        "tensors": tensors,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {out_path}: {len(tensors)} tensors, "
          f"{len(files)} shards, {total_bytes / 2**30:.1f} GiB")
    print("by-role:", {k: f"{v/2**30:.1f}GiB"
                       for k, v in sorted(role_bytes.items(),
                                          key=lambda kv: -kv[1])[:10]})
    return 0


def eval_shape(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


if __name__ == "__main__":
    sys.exit(main())
