#!/usr/bin/env python3
"""Compose the hybrid GLM-5.3-Flash checkpoint: NVFP4 routed experts beside
the FP8 release's bytes for everything else.

The default composition keeps the FP8 checkpoint's non-expert tensors
and MTP layer while replacing the main stack's routed experts with
NVFP4 tensors. This reduces expert storage without expanding the other
weight classes to BF16. Both source checkpoints must be available in the
local Hugging Face cache.

Composition copies bytes without quantization or other arithmetic and
verifies output tensors against their sources. A comparison with the FP8
checkpoint therefore isolates the effect of the replaced tensor classes.
See docs/nvfp4_plan.md for the checkpoint survey and quality measurements.

For every tensor of the FP8 release, by name:
  * a routed expert of a main-stack MoE layer (mlp_layer_types == "sparse",
    layers 3..44 here): the release's `weight` + `weight_scale_inv` are
    replaced by the NVFP4 source's `weight_packed` (U8, e2m1 pairs, low
    nibble = even element), `weight_scale` (F8_E4M3, one per 16 elements
    along K) and `weight_global_scale` (F32, one per tensor);
      dequant: w = e2m1(code) * (float32(weight_scale) / weight_global_scale)
  * the MTP draft layer (index num_hidden_layers) and the DSA attention
    projections come from the release by default (--mtp-from / --dsa-from
    can take them from the NVFP4 source instead, where they are the BF16
    repo's bytes);
  * everything else (KDA, indexer, router, mHC, norms, embeddings, lm_head,
    vision, the shared experts, the dense MLPs) comes from the release.
Tokenizer, chat template, generation and processor configs come from the
release (its chat template is the one the engine's goldens pin; the NVFP4
source ships the BF16 repo's, which differs).

HOW. Shard headers are read (a few MB), the output shards are laid out
deterministically (tensors in name order, ~5 GB per shard), the bytes move
with sendfile(2), and a verification pass re-reads every output tensor
against its source byte for byte while hashing each output file. The result
lands in the Hub cache layout — blobs named by sha256, a snapshot of
symlinks, refs/main — under a model id of your choosing, so `--model
<id>` resolves it exactly like a downloaded repo (loaders/hf_cache); or in
a plain directory with --out-dir for --checkpoint-dir. The snapshot
revision is a hash of the composition plan (tool version, source
revisions, options), so four nodes composing from the same sources produce
the same revision and byte-identical shards — the MANIFEST's shard hashes
are the cross-node check.

Usage (defaults are the production sources):
  tools/compose_nvfp4_hybrid.py --dry-run
  tools/compose_nvfp4_hybrid.py [--out-model-id dgpp/GLM-5.3-Flash-NVFP4-FP8]
  tools/compose_nvfp4_hybrid.py --out-dir /path/to/dir
Python 3 standard library only (no numpy, no torch): the tool must run on a
bare node.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

TOOL_VERSION = "1"
DT_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "U8": 1, "I64": 8, "I32": 4, "BOOL": 1}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)$")
COPY_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja", "processor_config.json", "LICENSE"]


def say(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- the Hub cache (mirrors loaders/hf_cache.cpp) -----------------------
def cache_root():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def model_dir_name(model_id):
    return "models--" + model_id.replace("/", "--")


def resolve_snapshot(root, model_id):
    mroot = os.path.join(root, model_dir_name(model_id))
    if not os.path.isdir(mroot):
        sys.exit(f"no cached model '{model_id}' under {root}")
    ref = os.path.join(mroot, "refs", "main")
    if os.path.isfile(ref):
        rev = open(ref).read().strip()
        snap = os.path.join(mroot, "snapshots", rev)
        if not os.path.isdir(snap):
            sys.exit(f"{model_id}: refs/main -> {rev} but the snapshot directory is missing")
        return snap, rev
    snaps = [d for d in os.listdir(os.path.join(mroot, "snapshots"))]
    if len(snaps) != 1:
        sys.exit(f"{model_id}: no refs/main and {len(snaps)} snapshots — ambiguous")
    return os.path.join(mroot, "snapshots", snaps[0]), snaps[0]


# ---- safetensors headers -------------------------------------------------
def read_headers(snapshot):
    """name -> {dtype, shape, nbytes, path, begin (absolute file offset)}"""
    tensors, shards = {}, []
    for fn in sorted(os.listdir(snapshot)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(snapshot, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        size = os.path.getsize(path)
        shards.append({"file": fn, "bytes": size, "header_bytes": n})
        for name, v in hdr.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                sys.exit(f"duplicate tensor {name} in {snapshot}")
            a, b = v["data_offsets"]
            numel = 1
            for d in v["shape"]:
                numel *= d
            if b - a != numel * DT_SIZE[v["dtype"]]:
                sys.exit(f"{fn}: {name} byte count disagrees with dtype x shape")
            if 8 + n + b > size:
                sys.exit(f"{fn}: {name} overruns the file")
            tensors[name] = {"dtype": v["dtype"], "shape": list(v["shape"]), "nbytes": b - a,
                             "path": path, "begin": 8 + n + a}
    return tensors, shards


# ---- classification ------------------------------------------------------
class Geometry:
    def __init__(self, cfg):
        tc = cfg["text_config"]
        self.num_layers = tc["num_hidden_layers"]
        self.mtp_layer = self.num_layers if tc.get("num_nextn_predict_layers", 0) == 1 else -1
        self.moe_layers = {i for i, t in enumerate(tc["mlp_layer_types"]) if t == "sparse"}
        self.dsa_layers = {i for i, t in enumerate(tc["layer_types"]) if t == "deepseek_sparse_attention"}
        self.n_experts = tc["n_routed_experts"]
        self.hidden = tc["hidden_size"]
        self.inter = tc["moe_intermediate_size"]

    def classify(self, name):
        """(class, layer, expert, proj, suffix). class in routed/shared/dense/
        dsa_attn/kda_attn/indexer/router/mhc/norm/mtp_head/vision/global/other."""
        if name.startswith("model.visual."):
            return ("vision", -1, -1, None, None)
        m = LAYER_RE.match(name)
        if not m:
            return ("global", -1, -1, None, None)
        layer, rest = int(m.group(1)), m.group(2)
        e = EXPERT_RE.match(rest)
        if e:
            return ("routed", layer, int(e.group(1)), e.group(2), e.group(3))
        if rest.startswith("mlp.shared_experts."):
            return ("shared", layer, -1, None, rest.split(".")[-1])
        if rest.startswith("mlp.gate."):
            return ("router", layer, -1, None, None)
        if rest.startswith("mlp."):
            return ("dense", layer, -1, None, rest.split(".")[-1])
        if rest.startswith("self_attn.indexer."):
            return ("indexer", layer, -1, None, None)
        if rest.startswith("self_attn."):
            return ("dsa_attn" if layer in self.dsa_layers or layer == self.mtp_layer else "kda_attn",
                    layer, -1, None, rest.split(".")[-1])
        if rest.startswith("hc_"):
            return ("mhc", layer, -1, None, None)
        if rest in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"):
            return ("mtp_head", layer, -1, None, None)
        if rest.endswith("layernorm.weight"):
            return ("norm", layer, -1, None, None)
        return ("other", layer, -1, None, None)


# ---- the plan ------------------------------------------------------------
def build_plan(geo, base, experts, mtp_from, dsa_from):
    """name -> (source_key, tensor) for the output; fail-closed census."""
    out = {}
    replaced_pairs = 0
    for name, t in base.items():
        cls, layer, _, _, _ = geo.classify(name)
        if cls == "routed" and layer in geo.moe_layers:
            replaced_pairs += 1
            continue
        if layer == geo.mtp_layer and mtp_from == "experts":
            continue
        if cls == "dsa_attn" and layer in geo.dsa_layers and dsa_from == "experts":
            continue
        out[name] = ("base", t)
    taken = 0
    for name, t in experts.items():
        cls, layer, _, _, suffix = geo.classify(name)
        if cls == "routed" and layer in geo.moe_layers:
            if suffix not in ("weight_packed", "weight_scale", "weight_global_scale"):
                sys.exit(f"experts source: unexpected routed-expert tensor {name}")
            out[name] = ("experts", t)
            taken += 1
        elif layer == geo.mtp_layer and mtp_from == "experts":
            out[name] = ("experts", t)
        elif cls == "dsa_attn" and layer in geo.dsa_layers and dsa_from == "experts":
            out[name] = ("experts", t)
    # Every routed expert of every MoE layer: exactly the NVFP4 triple, right shapes.
    H, I = geo.hidden, geo.inter
    want = {"gate_proj": ([I, H // 2], [I, H // 16]), "up_proj": ([I, H // 2], [I, H // 16]),
            "down_proj": ([H, I // 2], [H, I // 16])}
    for L in sorted(geo.moe_layers):
        for e in range(geo.n_experts):
            for proj, (pk_shape, sc_shape) in want.items():
                p = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}."
                for suffix, dtype, shape in (("weight_packed", "U8", pk_shape), ("weight_scale", "F8_E4M3", sc_shape),
                                             ("weight_global_scale", "F32", [1])):
                    t = out.get(p + suffix)
                    if t is None or t[0] != "experts":
                        sys.exit(f"missing NVFP4 tensor {p + suffix}")
                    if t[1]["dtype"] != dtype or t[1]["shape"] != shape:
                        sys.exit(f"{p + suffix}: {t[1]['dtype']} {t[1]['shape']} != {dtype} {shape}")
    expected_triples = len(geo.moe_layers) * geo.n_experts * 3
    if taken != expected_triples * 3 or replaced_pairs != expected_triples * 2:
        sys.exit(f"routed-expert census: took {taken} NVFP4 tensors, replaced {replaced_pairs} FP8 tensors, "
                 f"expected {expected_triples * 3} and {expected_triples * 2}")
    # Every FP8 payload in the output has its scale partner (and vice versa).
    for name, (src, t) in out.items():
        if t["dtype"] == "F8_E4M3" and name.endswith(".weight"):
            if name + "_scale_inv" not in out:
                sys.exit(f"{name}: FP8 payload without weight_scale_inv")
        if name.endswith(".weight_scale_inv") and name[:-len("_scale_inv")] not in out:
            sys.exit(f"{name}: orphan scale")
    return out


def census(geo, plan):
    counts = {}
    for name, (src, t) in plan.items():
        cls = geo.classify(name)[0]
        if geo.classify(name)[1] == geo.mtp_layer:
            cls = "mtp:" + cls
        key = (cls, src, t["dtype"])
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---- writing ---------------------------------------------------------------
def pack_shards(plan, shard_bytes):
    names = sorted(plan)
    shards, cur, cur_bytes = [], [], 0
    for n in names:
        nb = plan[n][1]["nbytes"]
        if cur and cur_bytes + nb > shard_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(n)
        cur_bytes += nb
    if cur:
        shards.append(cur)
    return shards


def header_bytes(names, plan):
    hdr = {"__metadata__": {"format": "pt"}}
    off = 0
    for n in names:
        t = plan[n][1]
        hdr[n] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + t["nbytes"]]}
        off += t["nbytes"]
    js = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(js)) % 8
    return struct.pack("<Q", len(js) + pad) + js + b" " * pad, off


class FdCache:
    def __init__(self):
        self.fds = {}

    def get(self, path):
        fd = self.fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self.fds[path] = fd
        return fd

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def copy_range(out_fd, in_fd, offset, count):
    while count > 0:
        n = os.sendfile(out_fd, in_fd, offset, min(count, 1 << 30))
        if n <= 0:
            raise OSError("sendfile returned %d" % n)
        offset += n
        count -= n


def write_shard(path, names, plan, fds):
    hdr, data_bytes = header_bytes(names, plan)
    out_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(out_fd, hdr)
        for n in names:
            t = plan[n][1]
            copy_range(out_fd, fds.get(t["path"]), t["begin"], t["nbytes"])
    finally:
        os.close(out_fd)
    return len(hdr) + data_bytes


def verify_shard(path, plan, fds, chunk=64 << 20):
    """Re-read the shard: every tensor byte-equal to its source; sha256 of the file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        n8 = fh.read(8)
        n = struct.unpack("<Q", n8)[0]
        hj = fh.read(n)
        h.update(n8)
        h.update(hj)
        hdr = json.loads(hj)
        entries = sorted(((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in hdr.items() if k != "__metadata__"))
        pos = 0
        for a, b, name in entries:
            if a != pos:
                raise RuntimeError(f"{path}: gap before {name}")
            t = plan[name][1]
            if hdr[name]["dtype"] != t["dtype"] or hdr[name]["shape"] != t["shape"] or b - a != t["nbytes"]:
                raise RuntimeError(f"{path}: {name} header disagrees with the plan")
            src_fd, src_off, left = fds.get(t["path"]), t["begin"], t["nbytes"]
            while left > 0:
                want = min(chunk, left)
                got = fh.read(want)
                ref = os.pread(src_fd, want, src_off)
                if got != ref:
                    raise RuntimeError(f"{path}: {name} differs from its source at byte {t['nbytes'] - left}")
                h.update(got)
                src_off += want
                left -= want
            pos = b
        if fh.read(1):
            raise RuntimeError(f"{path}: trailing bytes after the last tensor")
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ---- config / README -------------------------------------------------------
def compose_config(base_cfg, experts_cfg, geo, args, src):
    cfg = json.loads(json.dumps(base_cfg))
    if base_cfg.get("text_config") != experts_cfg.get("text_config"):
        sys.exit("text_config differs between the two sources — refusing to compose")
    fp8 = json.loads(json.dumps(base_cfg.get("quantization_config", {})))
    cfg["quantization_config"] = {
        "quant_method": "dgpp_mixed",
        "description": "NVFP4 routed experts (main-stack MoE layers) beside the FP8 release's "
                       "block-scaled e4m3 tensors; composed by tools/compose_nvfp4_hybrid.py, no arithmetic",
        "routed_experts": {
            "layers": sorted(geo.moe_layers),
            "format": "nvfp4-pack-quantized",
            "num_bits": 4, "group_size": 16, "symmetric": True,
            "scale_dtype": "torch.float8_e4m3fn", "global_scale_dtype": "torch.float32",
            "tensor_suffixes": ["weight_packed", "weight_scale", "weight_global_scale"],
            "packing": "two e2m1 codes per byte, low nibble = even element",
            "dequantize": "e2m1(code) * (float32(weight_scale) / weight_global_scale)",
            "activations": "bf16 (weight-only, NVFP4A16)",
            "source": src["experts"],
        },
        "fp8": dict(fp8, source=src["base"]),
        "mtp_layer": {"index": geo.mtp_layer, "source": args.mtp_from},
        "dsa_attention": {"layers": sorted(geo.dsa_layers), "source": args.dsa_from},
    }
    return cfg


def render_readme(args, src, counts, shards, total_bytes, verified, rev, out_id):
    lines = [f"# {out_id}", "",
             "GLM-5.3-Flash with **NVFP4 routed experts** beside the **FP8 release's own bytes** for every other tensor.",
             "Composed from two cached checkpoints by `tools/compose_nvfp4_hybrid.py` (dgpp) with no arithmetic:",
             "every output tensor is a byte-for-byte copy of its source, verified on write.", "",
             "| role | source | revision |", "|---|---|---|",
             f"| routed experts of the main-stack MoE layers (layers {min(args.moe_layers)}–{max(args.moe_layers)}) | `{src['experts']['repo']}` | `{src['experts']['revision']}` |",
             f"| everything else (attention, shared experts, dense MLPs, router, mHC, norms, embeddings, lm_head, vision, MTP layer{' from the NVFP4 source' if args.mtp_from == 'experts' else ''}) | `{src['base']['repo']}` | `{src['base']['revision']}` |",
             "", "Routed expert tensors: `weight_packed` (U8, two e2m1 codes per byte, low nibble = even element),",
             "`weight_scale` (F8_E4M3, one per 16 elements along K), `weight_global_scale` (F32, one per tensor);",
             "dequantize as `e2m1(code) * (float32(weight_scale) / weight_global_scale)`. Activations stay bf16 (NVFP4A16).",
             "FP8 tensors keep the release's `weight` + `weight_scale_inv` (128×128 block scales, multiply on dequant).",
             f"MTP draft layer: from the {'NVFP4 source (BF16 repo bytes)' if args.mtp_from == 'experts' else 'FP8 release'}.",
             f"DSA attention projections: from the {'NVFP4 source (BF16 repo bytes)' if args.dsa_from == 'experts' else 'FP8 release'}.",
             "Tokenizer, chat template, generation and processor configs: the FP8 release's.", "",
             f"Size: {total_bytes / 1e9:.1f} GB in {len(shards)} shards. Snapshot revision `{rev}` (a hash of the composition plan).",
             "", "## Tensor census (class, source, dtype → count)", "", "| class | source | dtype | tensors |", "|---|---|---|---:|"]
    for (cls, s, dt), n in sorted(counts.items()):
        lines.append(f"| {cls} | {s} | {dt} | {n} |")
    lines += ["", "## Verification", "",
              f"- every output tensor re-read and compared byte-for-byte with its source: **{'PASS' if verified else 'not run'}**",
              "- per-shard sha256 in `MANIFEST.json`; four nodes composing from the same sources must agree on every hash",
              "", "## Serving", "", "This checkpoint's mixed format is consumed by dgpp (`--model " + out_id + "`).",
              "Other engines would need a loader for the `dgpp_mixed` quantization_config.", "",
              "License: MIT, as the base model."]
    return "\n".join(lines) + "\n"


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", default="dabsLabs/GLM-5.3-Flash-NVFP4", help="NVFP4 source model id (Hub cache)")
    ap.add_argument("--experts-dir", help="NVFP4 source snapshot directory (overrides --experts)")
    ap.add_argument("--base", default="unsloth/GLM-5.3-Flash-FP8", help="FP8 release model id (Hub cache)")
    ap.add_argument("--base-dir", help="FP8 release snapshot directory (overrides --base)")
    ap.add_argument("--out-model-id", default="dgpp/GLM-5.3-Flash-NVFP4-FP8", help="model id of the composed checkpoint in the Hub cache")
    ap.add_argument("--out-dir", help="write a plain directory instead of a Hub cache entry")
    ap.add_argument("--mtp-from", choices=["base", "experts"], default="base", help="MTP draft layer source")
    ap.add_argument("--dsa-from", choices=["base", "experts"], default="base", help="DSA attention projections source")
    ap.add_argument("--shard-bytes", type=float, default=5e9)
    ap.add_argument("--expect-experts-rev", help="refuse unless the NVFP4 source snapshot has this revision")
    ap.add_argument("--expect-base-rev", help="refuse unless the FP8 source snapshot has this revision")
    ap.add_argument("--no-verify", action="store_true", help="skip the byte-for-byte re-read (hashes still computed)")
    ap.add_argument("--dry-run", action="store_true", help="plan and census only")
    ap.add_argument("--force", action="store_true", help="rebuild even if the output snapshot is already published")
    args = ap.parse_args()

    root = cache_root()
    if args.experts_dir:
        exp_snap, exp_rev = os.path.abspath(args.experts_dir), "local"
    else:
        exp_snap, exp_rev = resolve_snapshot(root, args.experts)
    if args.base_dir:
        base_snap, base_rev = os.path.abspath(args.base_dir), "local"
    else:
        base_snap, base_rev = resolve_snapshot(root, args.base)
    if args.expect_experts_rev and exp_rev != args.expect_experts_rev:
        sys.exit(f"NVFP4 source revision {exp_rev} != expected {args.expect_experts_rev}")
    if args.expect_base_rev and base_rev != args.expect_base_rev:
        sys.exit(f"FP8 source revision {base_rev} != expected {args.expect_base_rev}")
    say(f"NVFP4 source: {exp_snap} ({exp_rev})")
    say(f"FP8 source:   {base_snap} ({base_rev})")

    base_cfg = json.load(open(os.path.join(base_snap, "config.json")))
    exp_cfg = json.load(open(os.path.join(exp_snap, "config.json")))
    geo = Geometry(base_cfg)
    args.moe_layers = geo.moe_layers
    say("reading shard headers")
    base, base_shards = read_headers(base_snap)
    experts, exp_shards = read_headers(exp_snap)
    say(f"base: {len(base)} tensors in {len(base_shards)} shards; NVFP4 source: {len(experts)} tensors in {len(exp_shards)} shards")
    plan = build_plan(geo, base, experts, args.mtp_from, args.dsa_from)
    counts = census(geo, plan)
    total_bytes = sum(t["nbytes"] for _, t in plan.values())
    shards = pack_shards(plan, args.shard_bytes)
    say(f"plan: {len(plan)} tensors, {total_bytes / 1e9:.1f} GB in {len(shards)} shards")
    for (cls, s, dt), n in sorted(counts.items()):
        print(f"   {cls:14s} {s:8s} {dt:8s} {n:6d}")

    src = {"experts": {"repo": args.experts if not args.experts_dir else args.experts_dir, "revision": exp_rev,
                       "shards": exp_shards},
           "base": {"repo": args.base if not args.base_dir else args.base_dir, "revision": base_rev,
                    "shards": base_shards}}
    plan_id = {"tool_version": TOOL_VERSION, "experts": [src["experts"]["repo"], exp_rev],
               "base": [src["base"]["repo"], base_rev], "mtp_from": args.mtp_from, "dsa_from": args.dsa_from,
               "shard_bytes": int(args.shard_bytes), "tensors": len(plan), "bytes": total_bytes}
    rev = hashlib.sha256(json.dumps(plan_id, sort_keys=True).encode()).hexdigest()[:40]
    say(f"composition revision {rev}")
    if args.dry_run:
        return

    # Output layout.
    cache_mode = args.out_dir is None
    if cache_mode:
        mroot = os.path.join(root, model_dir_name(args.out_model_id))
        snap = os.path.join(mroot, "snapshots", rev)
        blobs = os.path.join(mroot, "blobs")
        tmp = os.path.join(blobs, ".tmp-" + rev)
        ref = os.path.join(mroot, "refs", "main")
        if os.path.isfile(ref) and open(ref).read().strip() == rev and not args.force:
            say(f"already published: {snap} (use --force to rebuild)")
            return
        for d in (snap, tmp):
            if os.path.isdir(d):
                shutil.rmtree(d)
        os.makedirs(snap)
        os.makedirs(tmp)
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        out_id = args.out_model_id
    else:
        snap = os.path.abspath(args.out_dir)
        if os.path.isdir(snap) and os.listdir(snap) and not args.force:
            sys.exit(f"{snap} exists and is not empty (use --force)")
        os.makedirs(snap, exist_ok=True)
        tmp = snap
        blobs = None
        out_id = os.path.basename(snap)

    def publish(tmp_path, name, digest):
        """Move a finished file into place: blob + symlink (cache) or in place (dir)."""
        if cache_mode:
            blob = os.path.join(blobs, digest)
            if os.path.exists(blob):
                os.remove(tmp_path)
            else:
                os.rename(tmp_path, blob)
            link = os.path.join(snap, name)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.join("..", "..", "blobs", digest), link)
        elif os.path.abspath(tmp_path) != os.path.join(snap, name):
            os.rename(tmp_path, os.path.join(snap, name))

    fds = FdCache()
    manifest_shards = []
    t_all = time.time()
    written = 0
    try:
        for i, names in enumerate(shards):
            fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
            path = os.path.join(tmp, fn)
            t0 = time.time()
            nbytes = write_shard(path, names, plan, fds)
            dt = time.time() - t0
            written += nbytes
            say(f"wrote {fn}: {len(names)} tensors, {nbytes / 1e9:.2f} GB in {dt:.1f} s ({nbytes / 1e9 / max(dt, 1e-9):.2f} GB/s); "
                f"{written / 1e9:.1f} of {total_bytes / 1e9:.1f} GB")
            t0 = time.time()
            if args.no_verify:
                digest = sha256_file(path)
            else:
                digest = verify_shard(path, plan, fds)
            say(f"   {'verified' if not args.no_verify else 'hashed'} {fn} in {time.time() - t0:.1f} s: sha256 {digest[:16]}…")
            publish(path, fn, digest)
            manifest_shards.append({"file": fn, "bytes": nbytes, "tensors": len(names), "sha256": digest,
                                    "verified": not args.no_verify})
    finally:
        fds.close()

    # The index, the config, the small files, the manifest, the README.
    weight_map = {}
    for i, names in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        for n in names:
            weight_map[n] = fn
    small = {}
    small["model.safetensors.index.json"] = json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                                                       indent=2, sort_keys=True).encode()
    small["config.json"] = json.dumps(compose_config(base_cfg, exp_cfg, geo, args, src), indent=2).encode()
    for fn in COPY_FILES:
        if fn == "config.json":
            continue
        p = os.path.join(base_snap, fn)
        if os.path.isfile(p):
            small[fn] = open(p, "rb").read()
    verified = not args.no_verify
    manifest = {"schema": 1, "tool": "tools/compose_nvfp4_hybrid.py", "tool_version": TOOL_VERSION,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": os.uname().nodename,
                "out_model_id": out_id, "revision": rev, "plan": plan_id, "sources": src,
                "options": {"mtp_from": args.mtp_from, "dsa_from": args.dsa_from},
                "census": [{"class": c, "source": s, "dtype": d, "tensors": n} for (c, s, d), n in sorted(counts.items())],
                "shards": manifest_shards, "total_tensor_bytes": total_bytes, "verified": verified,
                "notes": ["chat_template.jinja is the FP8 release's (the NVFP4 source ships the BF16 repo's, which differs)"]}
    small["MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    small["README.md"] = render_readme(args, src, counts, shards, total_bytes, verified, rev, out_id).encode()
    for fn, data in small.items():
        p = os.path.join(tmp, fn + ".part")
        with open(p, "wb") as fh:
            fh.write(data)
        publish(p, fn, hashlib.sha256(data).hexdigest())
    if cache_mode:
        os.rmdir(tmp)
        with open(ref + ".part", "w") as fh:
            fh.write(rev + "\n")
        os.rename(ref + ".part", ref)  # the atomic publish: refs/main appears last
    say(f"done in {(time.time() - t_all) / 60:.1f} min: {snap}")
    say(f"load with: --model {out_id}" if cache_mode else f"load with: --checkpoint-dir {snap}")


if __name__ == "__main__":
    main()
