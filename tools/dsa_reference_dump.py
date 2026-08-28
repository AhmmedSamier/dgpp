#!/usr/bin/env python3
"""DSA reference-dump harness (M3 deliverable, DESIGN §12 tier 2).

Generates reference dumps for one DSA layer of GLM-5.3-Flash so the C++
parity test can compare our CUDA layer against an independently computed
result.

Two backends:

  pure   (stdlib only)
      Synthetic seeded weights and activations at a small geometry,
      reference math in python floats (IEEE double) with bf16 rounding at
      the same boundaries as the engine. This is the local CI oracle: it
      runs anywhere python runs.

  torch  (requires torch; run where the checkpoint lives)
      Reads the real weights straight out of the safetensors shards
      (stdlib header parsing), generates random activations, and runs the
      same reference equations in torch. This is the "real checkpoint
      slice" parity input for the DSA layer.

The reference implements the DGPP-pinned DSA layer contract
(src/models/dsa_reference.cpp is the C++ twin):

  qkv_a   = x @ W_qkva.T                       (bf16 out; fused [q_a|kv_a])
  q_c     = rmsnorm(q_a, q_aln, eps)            (fp32 math, bf16 out)
  latent  = rmsnorm(kv_a, kv_aln, eps)          (cached rows)
  q_mla   = q_c @ W_qb.T                        (bf16 out)
  q_idx   = q_c @ W_wqb.T                       (bf16 out)
  k_raw   = x @ W_wk.T ; k = layernorm(k_raw, k_norm_w, k_norm_b, eps 1e-6)
  gate    = x @ W_gate.T                        (bf16 out)
  weights[t,h] = fp32 dot(x_t, wp_h)            (NO bf16 rounding — pinned)
  q_fp8, q_scale = fp8_quant(bf16_round(hadamard128(q_idx_row)))
  w_folded = (weights * q_scale) * (128^-0.5 * heads^-0.5)
  per complete pool j: index_k[j], index_scale[j] = compress_pool(
      k, gate, ape)   (per-dim softmax(gate+ape) weighted sum -> bf16 ->
      hadamard128 -> bf16 -> absmax fp8 with power-of-two scale)
  tail ring: last kpool tokens' k/gate rows (stash semantics)
  per query at pos p, visible = (p+1)//kpool pools:
    visible <= select_k: dense causal over [0, p]
    else: logits[j] = sum_h w_folded[h] * index_scale[j] *
              dot(q_fp8[h], index_k[j])   (fp32)
          select top select_k (exact ties -> lower index, ascending),
          expand pools to tokens + append incomplete tail
  out = W_uv-absorbed attention over the selected latent rows
        (q~ = W_uk^T q, bf16 GEMM rounding; probs round to bf16 for the
        c accumulation; v-absorbed output), then @ W_o.T

File format ("DGPPDSAD"): 8-byte magic, u32 version=1, u32 header length,
JSON header, payload — same container as the KDA dump.

Usage:
  dsa_reference_dump.py selftest
  dsa_reference_dump.py gen-pure --out FILE [--heads H] [--q-lora Q]
        [--kv-lora K] [--nope N] [--v-dim V] [--idx-heads IH] [--topk TK]
        [--kpool KP] [--hidden X] [--tokens T] [--seed S] [--layer L]
  dsa_reference_dump.py gen-torch --model-dir DIR --layer N --out FILE
        [--tokens T] [--seed S]

gen-pure is wired into CTest; gen-torch is a manual deployment test on the
box holding the checkpoint (record its invocation with the result).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

MAGIC = b"DGPPDSAD"
VERSION = 1


# ---------------------------------------------------------------------------
# bf16 / fp8 helpers (round-to-nearest-even, matching dtypes.hpp; the fp8
# encoder is the saturating OCP convention the engine pins)
# ---------------------------------------------------------------------------

def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16_round(x: float) -> float:
    b = f32_bits(x)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFF0000))[0]


def bf16_bits(x: float) -> int:
    b = f32_bits(x)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return (b >> 16) & 0xFFFF


def bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _rne(v: float) -> int:
    """Round to nearest, ties to even (lrintf semantics)."""
    n = math.floor(v)
    rem = v - n
    if rem > 0.5 or (rem == 0.5 and (n & 1)):
        n += 1
    return n


def fp8_bits(x: float) -> int:
    """Saturating round-to-nearest-even e4m3 encode (max finite 448).
    Mirrors dtypes.hpp exactly: subnormal zone (av < 2^-6) quantizes av/2^-9
    with ties-to-even, n in [1,7] as the mantissa field and n >= 8 rounding
    up to the minimum normal; the normal zone is a LINEAR RNE over
    av/space with n in [8,15] (man3 = n-8), saturating at 448."""
    if math.isnan(x):
        return 0x7F
    sign = 0x80 if math.copysign(1.0, x) < 0 else 0
    a = abs(x)
    if a < 2.0 ** -6:  # subnormal zone: quantum 2^-9
        n = _rne(a * 512.0)
        if n <= 0:
            return sign
        if n >= 8:  # rounds up to the minimum normal
            return sign | (1 << 3)
        return sign | n
    # Normal zone: linear RNE over av / space, n in [8, 15].
    e = 0
    q = a
    while q < 1.0:
        q *= 2.0
        e -= 1
    while q >= 2.0:
        q *= 0.5
        e += 1
    space = 2.0 ** (e - 3)
    n = _rne(a / space)
    if n >= 16:
        e += 1
        n >>= 1
    if e > 8 or (e == 8 and n >= 15):  # into the NaN code or beyond
        return sign | 0x7E
    return sign | ((e + 7) << 3) | (n - 8)


def fp8_from_bits(bits: int) -> float:
    sign = -1.0 if bits & 0x80 else 1.0
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    if exp == 0:
        return sign * man * 0.125 * 0.015625
    return sign * (1.0 + man * 0.125) * (2.0 ** (exp - 7))


def next_pow2_at_or_above(v: float) -> float:
    """Smallest power of two >= v; exact powers map to themselves (the
    engine's bit-manipulation form, not exp2(ceil(log2(v))))."""
    b = f32_bits(v)
    e = ((b >> 23) & 0xFF) - 127
    exact = (b & 0x7FFFFF) == 0
    return math.ldexp(1.0, e if exact else e + 1)


def quant_row_fp8(x):
    """absmax (floor 1e-4) + power-of-two scale + saturating encode."""
    absmax = max((abs(v) for v in x), default=0.0)
    absmax = max(absmax, 1e-4)
    scale = next_pow2_at_or_above(absmax * (1.0 / 448.0))
    return [fp8_bits(v / scale) for v in x], scale


# ---------------------------------------------------------------------------
# Deterministic random (same scheme as the KDA tool)
# ---------------------------------------------------------------------------

def make_rng(seed: int):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def next_uniform() -> float:
        nonlocal state
        state ^= state >> 12
        state = (state * 12605985483714908141) & ((1 << 64) - 1)
        state ^= state >> 26
        return ((state >> 11) + 0.5) / (1 << 53)

    return next_uniform


def randn(rng, n: int, stddev: float = 1.0):
    out = []
    while len(out) < n:
        u1 = max(rng(), 1e-12)
        u2 = rng()
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2) * stddev)
        if len(out) < n:
            out.append(r * math.sin(2 * math.pi * u2) * stddev)
    return out[:n]


# ---------------------------------------------------------------------------
# Reference layer math (pure python, double interior, pinned boundaries)
# ---------------------------------------------------------------------------

INV_SQRT128 = 0.08838834764831845  # 1/sqrt(128), the pinned constant


def fwht128(x):
    """In-place Hadamard-128 (butterfly strides 1..64) then * 1/sqrt(128)."""
    stride = 1
    while stride < 128:
        for p in range(0, 128, 2 * stride):
            for i in range(stride):
                a = x[p + i]
                b = x[p + i + stride]
                x[p + i] = a + b
                x[p + i + stride] = a - b
        stride <<= 1
    for d in range(128):
        x[d] *= INV_SQRT128


def gemm_bf16(rows, weight, n_out, n_in):
    """out[m,n] = bf16(sum_k rows[m,k]*weight[n,k]) — reference GEMM."""
    out = []
    for m in range(len(rows) // n_in):
        row = rows[m * n_in:(m + 1) * n_in]
        for n in range(n_out):
            acc = 0.0
            for k in range(n_in):
                acc += row[k] * weight[n * n_in + k]
            out.append(bf16_round(acc))
    return out


def rmsnorm_row(x, w, eps):
    ss = sum(v * v for v in x)
    inv = 1.0 / math.sqrt(ss / len(x) + eps)
    return [bf16_round(v * inv * w[i]) for i, v in enumerate(x)]


def layernorm_row(x, w, b, eps):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    inv = 1.0 / math.sqrt(var + eps)
    return [bf16_round((v - mean) * inv * w[i] + b[i])
            for i, v in enumerate(x)]


def compress_pool(k_rows, gate_rows, ape, kpool):
    """Per-dim softmax(gate+ape) weighted sum -> bf16 -> fwht128 -> bf16 ->
    absmax fp8 with power-of-two scale. Returns (fp8 row, scale)."""
    dim = len(k_rows[0])
    x = []
    for d in range(dim):
        scores = [gate_rows[s][d] + ape[s * dim + d] for s in range(kpool)]
        mx = max(scores)
        acc = 0.0
        denom = 0.0
        for s in range(kpool):
            prob = math.exp(scores[s] - mx)
            denom += prob
            acc += k_rows[s][d] * prob
        x.append(bf16_round(acc / denom))
    fwht128(x)
    x = [bf16_round(v) for v in x]
    return quant_row_fp8(x)


def sortable_f32(f: float) -> int:
    u = f32_bits(f)
    return (~u) & 0xFFFFFFFF if u >> 31 else (u | 0x80000000)


def select_pools(logits, select_k):
    """Top select_k by composite key (~sortable << 21) | idx — highest
    logit, exact ties to the lower index, output ascending."""
    n = min(select_k, len(logits))
    keys = [((~sortable_f32(l)) & 0xFFFFFFFFFFFFFFFF) << 21 | j
            for j, l in enumerate(logits)]
    keys.sort()
    pools = sorted(int(k & 0x1FFFFF) for k in keys[:n])
    return pools


def reference_layer(cfg, weights, hidden_in, tokens):
    """The full pinned forward. `weights` is a dict of flat bf16/f32 lists
    (python floats for bf16-rounded values, fp32 for ape/weights); returns
    (layer_out, cache_state, topk) where cache_state is the logical HostState
    after the chunk."""
    hidden = cfg["hidden"]
    heads = cfg["idx_heads"]
    dim = cfg["idx_dim"]
    kpool = cfg["kpool"]
    select_k = cfg["topk"] // kpool
    max_selected = cfg["topk"] + kpool - 1
    q_lora = cfg["q_lora"]
    kv_lora = cfg["kv_lora"]
    nope = cfg["nope"]
    v_dim = cfg["v_dim"]
    local_heads = cfg["heads"]  # tp=1
    eps = cfg.get("rms_eps", 1e-5)
    logit_scale = float(dim) ** -0.5 * float(heads) ** -0.5
    attn_scale = float(nope) ** -0.5

    W = weights
    # Projections (hidden rows are bf16-rounded floats).
    qkv_cols = q_lora + kv_lora
    qkv_a = gemm_bf16(hidden_in, W["qkv_a"], qkv_cols, hidden)
    q_c, latent_rows = [], []
    for t in range(tokens):
        row = qkv_a[t * qkv_cols:(t + 1) * qkv_cols]
        q_c.extend(rmsnorm_row(row[:q_lora], W["q_aln"], eps))
        latent_rows.append(
            rmsnorm_row(row[q_lora:], W["kv_aln"], eps))
    q_mla = gemm_bf16(q_c, W["q_b"], local_heads * nope, q_lora)
    q_idx = gemm_bf16(q_c, W["wq_b"], heads * dim, q_lora)

    k_rows, gate_rows, weights_rows = [], [], []
    for t in range(tokens):
        row = hidden_in[t * hidden:(t + 1) * hidden]
        k_raw = gemm_bf16([v for v in row], W["wk"], dim, hidden)
        k_rows.append(layernorm_row(k_raw, W["k_norm_w"], W["k_norm_b"], 1e-6))
        gate_rows.append(gemm_bf16([v for v in row], W["gate"], dim, hidden))
        wr = []
        for h in range(heads):
            acc = 0.0  # fp32 weights: NO bf16 rounding (pinned)
            for i in range(hidden):
                acc += row[i] * W["wp"][h * hidden + i]
            wr.append(acc)
        weights_rows.append(wr)

    # Indexer q path: hadamard + fp8, then fold scales into weights.
    q_fp8, q_scale = [], []
    for t in range(tokens):
        for h in range(heads):
            x = [q_idx[(t * heads + h) * dim + d] for d in range(dim)]
            fwht128(x)
            x = [bf16_round(v) for v in x]
            codes, scale = quant_row_fp8(x)
            q_fp8.extend(codes)
            q_scale.append(scale)
    w_folded = []
    for t in range(tokens):
        for h in range(heads):
            base = weights_rows[t][h] * q_scale[t * heads + h]
            w_folded.append(base * logit_scale)

    # Cache state: pools, tail (logical layout, HostState semantics).
    n_pools = tokens // kpool
    index_k, index_scale = [], []
    for j in range(n_pools):
        codes, scale = compress_pool(k_rows[j * kpool:(j + 1) * kpool],
                                     gate_rows[j * kpool:(j + 1) * kpool],
                                     W["ape"], kpool)
        index_k.extend(codes)
        index_scale.append(scale)
    tail = []
    for s in range(kpool):  # raw K half then gate half
        t = tokens - kpool + s
        if t >= 0:
            tail.extend(k_rows[t])
    for s in range(kpool):
        t = tokens - kpool + s
        if t >= 0:
            tail.extend(gate_rows[t])

    # Per-query selection + attention.
    topk = []
    attn_out = []
    head_rows = nope + v_dim
    for t in range(tokens):
        pos = t
        visible = (pos + 1) // kpool
        row_toks = [-1] * max_selected
        if visible <= select_k:
            for i in range(pos + 1):
                row_toks[i] = i
            n_tok = pos + 1
        else:
            logits = []
            for j in range(visible):
                total = 0.0
                for h in range(heads):
                    dot = 0.0
                    for d in range(dim):
                        dot += (fp8_from_bits(
                            q_fp8[(t * heads + h) * dim + d]) *
                                fp8_from_bits(index_k[j * dim + d]))
                    total += (w_folded[t * heads + h] * index_scale[j]) * dot
                logits.append(total)
            pools = select_pools(logits, select_k)
            w = 0
            for p in pools:
                for s in range(kpool):
                    row_toks[w] = p * kpool + s
                    w += 1
            tail_start = ((pos + 1) // kpool) * kpool
            for tt in range(tail_start, pos + 1):
                row_toks[w] = tt
                w += 1
            n_tok = w
        topk.extend(row_toks)

        # Absorbed attention for this row.
        for h in range(local_heads):
            w_uk = W["kv_b"][h * head_rows * kv_lora:
                             h * head_rows * kv_lora + nope * kv_lora]
            q_tilde = []
            for c in range(kv_lora):
                acc = 0.0
                for d in range(nope):
                    acc += (q_mla[t * local_heads * nope + h * nope + d] *
                            w_uk[d * kv_lora + c])
                q_tilde.append(bf16_round(acc))
            scores = []
            for i in range(n_tok):
                tok = row_toks[i]
                acc = 0.0
                for c in range(kv_lora):
                    acc += q_tilde[c] * latent_rows[tok][c]
                scores.append(acc * attn_scale)
            mx = max(scores) if scores else 0.0
            denom = sum(math.exp(s - mx) for s in scores)
            c_vec = [0.0] * kv_lora
            for i, s in enumerate(scores):
                p = bf16_round(math.exp(s - mx) / denom)
                for c in range(kv_lora):
                    c_vec[c] += p * latent_rows[row_toks[i]][c]
            w_uv = W["kv_b"][(h * head_rows + nope) * kv_lora:
                             (h + 1) * head_rows * kv_lora]
            for d in range(v_dim):
                acc = 0.0
                for c in range(kv_lora):
                    acc += w_uv[d * kv_lora + c] * c_vec[c]
                attn_out.append(bf16_round(acc))

    # Output projection.
    layer_out = gemm_bf16(attn_out, W["o_proj"], hidden, local_heads * v_dim)
    return layer_out, (index_k, index_scale, latent_rows, tail), topk


# ---------------------------------------------------------------------------
# Dump container (same as the KDA tool)
# ---------------------------------------------------------------------------

def write_dump(path, meta, cfg, tensors):
    header = {
        "format": "dgpp-dsa-reference-dump",
        "version": VERSION,
        "model": meta["model"],
        "revision": meta["revision"],
        "backend": meta["backend"],
        "layer_idx": meta["layer_idx"],
        "config": cfg,
        "tensors": {},
    }
    payload = bytearray()
    for name, (dtype, shape, blob) in tensors.items():
        header["tensors"][name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": len(payload),
            "nbytes": len(blob),
        }
        payload.extend(blob)
    header_bytes = json.dumps(header, indent=1, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(payload))


def read_dump(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError("bad magic")
    version, header_len = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise ValueError("unsupported version %d" % version)
    header = json.loads(data[16:16 + header_len])
    payload = memoryview(data)[16 + header_len:]
    tensors = {}
    for name, desc in header["tensors"].items():
        tensors[name] = (desc["dtype"], desc["shape"],
                         payload[desc["offset"]:desc["offset"] + desc["nbytes"]])
    return header, tensors


def blob_bf16(values):
    return struct.pack("<%dH" % len(values), *(bf16_bits(v) for v in values))


def blob_f32(values):
    return struct.pack("<%df" % len(values), *(float(v) for v in values))


def blob_u8(values):
    return bytes(values)


def blob_i32(values):
    return struct.pack("<%di" % len(values), *(int(v) for v in values))


# ---------------------------------------------------------------------------
# Backend: pure (synthetic weights, stdlib math)
# ---------------------------------------------------------------------------

def gen_pure(args):
    cfg = {
        "hidden": args.hidden,
        "heads": args.heads,
        "q_lora": args.q_lora,
        "kv_lora": args.kv_lora,
        "nope": args.nope,
        "v_dim": args.v_dim,
        "idx_heads": args.idx_heads,
        "idx_dim": 128,
        "topk": args.topk,
        "kpool": args.kpool,
        "tokens": args.tokens,
        "rms_eps": 1e-5,
        "seed": args.seed,
    }
    rng = make_rng(args.seed)
    W = {}
    # Weights: small-normal bf16-rounded floats (magnitudes that keep the
    # softmax/fp8 paths well-conditioned).
    def rn(n, s=0.3):
        return [bf16_round(v) for v in randn(rng, n, s)]
    W["qkv_a"] = rn((cfg["q_lora"] + cfg["kv_lora"]) * cfg["hidden"])
    W["q_aln"] = rn(cfg["q_lora"], 0.5)
    W["kv_aln"] = rn(cfg["kv_lora"], 0.5)
    W["q_b"] = rn(cfg["heads"] * cfg["nope"] * cfg["q_lora"])
    W["wq_b"] = rn(cfg["idx_heads"] * 128 * cfg["q_lora"])
    W["wk"] = rn(128 * cfg["hidden"])
    W["gate"] = rn(128 * cfg["hidden"])
    W["wp"] = rn(cfg["idx_heads"] * cfg["hidden"])
    W["k_norm_w"] = rn(128, 0.5)
    W["k_norm_b"] = rn(128, 0.05)
    W["kv_b"] = rn(cfg["heads"] * (cfg["nope"] + cfg["v_dim"]) * cfg["kv_lora"])
    W["o_proj"] = rn(cfg["hidden"] * cfg["heads"] * cfg["v_dim"])
    W["ape"] = [v * 0.1 for v in randn(rng, cfg["kpool"] * 128, 1.0)]
    hidden_in = [bf16_round(v) for v in randn(rng, cfg["tokens"] * cfg["hidden"], 1.0)]

    layer_out, (index_k, index_scale, latent_rows, tail), topk = \
        reference_layer(cfg, W, hidden_in, cfg["tokens"])

    tensors = {
        "qkv_a": ("BF16", [cfg["q_lora"] + cfg["kv_lora"], cfg["hidden"]],
                  blob_bf16(W["qkv_a"])),
        "q_aln": ("BF16", [cfg["q_lora"]], blob_bf16(W["q_aln"])),
        "kv_aln": ("BF16", [cfg["kv_lora"]], blob_bf16(W["kv_aln"])),
        "q_b": ("BF16", [cfg["heads"] * cfg["nope"], cfg["q_lora"]],
                blob_bf16(W["q_b"])),
        "wq_b": ("BF16", [cfg["idx_heads"] * 128, cfg["q_lora"]],
                 blob_bf16(W["wq_b"])),
        "wk": ("BF16", [128, cfg["hidden"]], blob_bf16(W["wk"])),
        "gate": ("BF16", [128, cfg["hidden"]], blob_bf16(W["gate"])),
        "wp": ("BF16", [cfg["idx_heads"], cfg["hidden"]], blob_bf16(W["wp"])),
        "k_norm_w": ("BF16", [128], blob_bf16(W["k_norm_w"])),
        "k_norm_b": ("BF16", [128], blob_bf16(W["k_norm_b"])),
        "kv_b": ("BF16",
                 [cfg["heads"] * (cfg["nope"] + cfg["v_dim"]), cfg["kv_lora"]],
                 blob_bf16(W["kv_b"])),
        "o_proj": ("BF16", [cfg["hidden"], cfg["heads"] * cfg["v_dim"]],
                   blob_bf16(W["o_proj"])),
        "ape": ("F32", [cfg["kpool"], 128], blob_f32(W["ape"])),
        "hidden_in": ("BF16", [cfg["tokens"], cfg["hidden"]],
                      blob_bf16(hidden_in)),
        "layer_out": ("BF16", [cfg["tokens"], cfg["hidden"]],
                      blob_bf16(layer_out)),
        "index_k": ("U8", [cfg["tokens"] // cfg["kpool"], 128],
                    blob_u8(index_k)),
        "index_scale": ("F32", [cfg["tokens"] // cfg["kpool"]],
                        blob_f32(index_scale)),
        "latent": ("BF16", [cfg["tokens"], cfg["kv_lora"]],
                   blob_bf16([v for row in latent_rows for v in row])),
        "tail": ("BF16", [2, cfg["kpool"], 128], blob_bf16(tail)),
        "topk": ("I32", [cfg["tokens"], cfg["topk"] + cfg["kpool"] - 1],
                 blob_i32(topk)),
    }
    meta = {
        "model": "synthetic-dsa",
        "revision": "pure-%d" % args.seed,
        "backend": "pure",
        "layer_idx": args.layer,
    }
    write_dump(args.out, meta, cfg, tensors)
    print("wrote %s (%d tokens, heads=%d, idx_heads=%d, kpool=%d)" %
          (args.out, cfg["tokens"], cfg["heads"], cfg["idx_heads"],
           cfg["kpool"]))
    return 0


# ---------------------------------------------------------------------------
# Backend: torch (real checkpoint slices)
# ---------------------------------------------------------------------------

def read_safetensors_index(model_dir):
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        shard_map = index["weight_map"]
        entries = {}
        for shard in sorted(set(shard_map.values())):
            _parse_shard_header(os.path.join(model_dir, shard), shard, entries)
        return entries
    if os.path.exists(single):
        entries = {}
        _parse_shard_header(single, "model.safetensors", entries)
        return entries
    raise FileNotFoundError("no safetensors index or file in %s" % model_dir)


def _parse_shard_header(path, shard, entries):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_base = 8 + header_len
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        entries[name] = (path, data_base + begin, end - begin, info["dtype"],
                         info["shape"])


def load_tensor(entries, name, torch):
    path, offset, nbytes, dtype_str, shape = entries[name]
    dtype_map = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
    }
    itemsize = {"BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype_str]
    numel = 1
    for d in shape:
        numel *= d
    if numel * itemsize != nbytes:
        raise ValueError("shape/dtype disagree with nbytes for %s" % name)
    with open(path, "rb") as f:
        f.seek(offset)
        buf = bytearray(f.read(nbytes))
    t = torch.frombuffer(buf, dtype=dtype_map[dtype_str])
    return t.reshape(shape).clone()


def gen_torch(args):
    import torch  # noqa: delayed import: backend requires torch

    with open(os.path.join(args.model_dir, "config.json")) as f:
        full = json.load(f)
    text = full.get("text_config", full)
    sa = text.get("sparse_attention_config", {})
    hidden = text["hidden_size"]
    heads = text.get("num_attention_heads", 64)
    q_lora = sa.get("q_lora_rank", text.get("q_lora_rank", 1536))
    kv_lora = sa.get("kv_lora_rank", text.get("kv_lora_rank", 512))
    nope = sa.get("qk_nope_head_dim", text.get("qk_nope_head_dim", 256))
    rope = sa.get("qk_rope_head_dim", text.get("qk_rope_head_dim", 0))
    v_dim = sa.get("v_head_dim", text.get("v_head_dim", 256))
    idx_heads = sa.get("num_index_heads", 32)
    idx_dim = sa.get("index_head_dim", 128)
    topk = sa.get("index_topk", text.get("index_topk", 2048))
    kpool = sa.get("index_kpool", 4)
    if rope not in (0, None):
        raise SystemExit("nonzero rope dim: engine implements rope-free only")

    prefix = "model.language_model.layers.%d.self_attn" % args.layer
    entries = read_safetensors_index(args.model_dir)

    def w(name):
        return load_tensor(entries, "%s.%s" % (prefix, name), torch)

    # Checkpoint names -> engine layout. Indexer weights are BF16; core
    # q_a/kv_a/q_b/o_proj are FP8 with 128x128 block scales — dequantize
    # to BF16 for the reference (M3 consumes BF16). Checkpoint names:
    # indexer projections are `.<name>.weight`, but the gate/ape/k_norm
    # parameter tensors have no `.weight` suffix (they are plain tensors,
    # not nn.Linear modules).
    wq_b = w("indexer.wq_b.weight")
    wk = w("indexer.wk.weight")
    wp = w("indexer.weights_proj.weight")
    gate = w("indexer.index_kpool_compress_gate")
    k_norm_w = w("indexer.k_norm.weight")
    k_norm_b = w("indexer.k_norm.bias")
    ape = w("indexer.index_kpool_compress_ape")

    def dequant_with(t, scale_inv):
        """E4M3 [N,K] * F32 scale_inv [ceil(N/128), ceil(K/128)] -> bf16.

        `weight_scale_inv` is the inverse of the quantization scale, so
        dequantization MULTIPLIES (verified against the checkpoint: mul
        gives std 0.015, div gives 5.6e5 on q_a_proj).
        """
        if t.dtype == torch.bfloat16:
            return t
        n, k = t.shape
        if tuple(scale_inv.shape) != (-(-n // 128), -(-k // 128)):
            raise ValueError(
                "scale_inv shape %s does not tile %s in 128x128 blocks"
                % (tuple(scale_inv.shape), (n, k)))
        t = t.float()
        out = torch.empty(n, k, dtype=torch.bfloat16)
        for r in range(0, n, 128):
            for c in range(0, k, 128):
                blk = t[r:r + 128, c:c + 128] * scale_inv[r // 128, c // 128].float()
                out[r:r + 128, c:c + 128] = blk.bfloat16()
        return out

    # Block scales live next to the weights as `<name>_scale_inv`.
    def dq(name):
        t = w(name)
        if t.dtype == torch.bfloat16:
            return t
        return dequant_with(t, w("%s_scale_inv" % name))

    qkv_a = torch.cat([dq("q_a_proj.weight"), dq("kv_a_proj_with_mqa.weight")],
                      dim=0).contiguous()
    q_b = dq("q_b_proj.weight")
    kv_b = dq("kv_b_proj.weight")
    o_proj = dq("o_proj.weight")

    tokens = args.tokens
    torch.manual_seed(args.seed)
    hidden_in = (torch.randn(tokens, hidden, dtype=torch.float32)).bfloat16()

    # Reference forward in torch (fp32 interior, pinned bf16 boundaries).
    def gemm_bf16_t(x, weight):
        return (x.float() @ weight.float().T).bfloat16()

    qkv_cols = q_lora + kv_lora
    proj = gemm_bf16_t(hidden_in, qkv_a)
    q_c = proj[:, :q_lora]
    kv_c = proj[:, q_lora:]

    def rmsnorm_t(x, wgt, eps):
        xf = x.float()
        var = (xf * xf).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * wgt.float()).bfloat16()

    q_c = rmsnorm_t(q_c, w("q_a_layernorm.weight"), 1e-5)
    latent = rmsnorm_t(kv_c, w("kv_a_layernorm.weight"), 1e-5)

    q_mla = gemm_bf16_t(q_c, q_b)
    q_idx = gemm_bf16_t(q_c, wq_b)
    k_raw = gemm_bf16_t(hidden_in, wk)
    gate_rows = gemm_bf16_t(hidden_in, gate)
    # weights: fp32, no bf16 rounding.
    weights_f = hidden_in.float() @ wp.float().T

    # LayerNorm (full, with bias; the indexer's own eps 1e-6).
    def layernorm_t(x, wgt, bias, eps):
        xf = x.float()
        mean = xf.mean(-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(-1, keepdim=True)
        return ((xf - mean) * torch.rsqrt(var + eps) * wgt.float() +
                bias.float()).bfloat16()

    k_rows = layernorm_t(k_raw, k_norm_w, k_norm_b, 1e-6)

    # Hadamard-128 + fp8 quant (per row; python loop over rows for exact
    # boundary semantics — tokens*idx_heads rows).
    import torch as _t

    def fwht_quant_rows(x_bf16):
        n, d = x_bf16.shape
        xf = x_bf16.float()
        # Hadamard via explicit butterflies to match the pinned stage order.
        h = xf.clone()
        stride = 1
        while stride < d:
            h = h.view(n, -1, 2 * stride)
            a = h[:, :, :stride].clone()
            b = h[:, :, stride:].clone()
            h = _t.cat([a + b, a - b], dim=2).view(n, d)
            stride <<= 1
        h = h * (1.0 / math.sqrt(128.0))
        h = h.bfloat16().float()  # bf16 round after rotation
        absmax = h.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
        # Power-of-two scale via the shared exact-bit helper: torch.log2
        # rounds across power-of-two boundaries (the DESIGN 7.2 hazard), so
        # scales are computed per row on python floats, exactly.
        scales = [next_pow2_at_or_above(a.item() * (1.0 / 448.0))
                  for a in absmax.flatten()]
        scale = torch.tensor(scales, dtype=torch.float32).unsqueeze(1)
        q = (h / scale)
        # Saturating e4m3 encode via torch's cast — cross-checked bit-exact
        # against the hand codec (fp8_bits) over a 1.1M-value sweep
        # including subnormals and the +-448 boundary.
        q = q.clamp(-448, 448).to(torch.float8_e4m3fn)
        return q, scale.squeeze(1).float()

    q_fp8, q_scale = fwht_quant_rows(q_idx.view(tokens * idx_heads, 128))
    q_fp8 = q_fp8.view(tokens, idx_heads, 128)

    logit_scale = float(idx_dim) ** -0.5 * float(idx_heads) ** -0.5
    w_folded = (weights_f * q_scale.view(tokens, idx_heads)) * logit_scale

    # Pool compression.
    n_pools = tokens // kpool
    index_k_t = torch.zeros(n_pools, 128, dtype=torch.uint8)
    index_scale_t = torch.zeros(n_pools)
    ape_f = ape.float()  # [kpool, 128]
    for j in range(n_pools):
        k_blk = k_rows[j * kpool:(j + 1) * kpool].float()   # [kpool,128]
        g_blk = gate_rows[j * kpool:(j + 1) * kpool].float()
        scores = g_blk + ape_f
        mx = scores.max(dim=0, keepdim=True).values
        prob = torch.exp(scores - mx)
        denom = prob.sum(dim=0, keepdim=True)
        x = (k_blk * prob).sum(dim=0) / denom
        x = x.bfloat16().float()
        # hadamard
        h = x
        stride = 1
        while stride < 128:
            h = h.view(-1)
            hh = h.clone()
            for p in range(0, 128, 2 * stride):
                for i in range(stride):
                    a = h[p + i]
                    b = h[p + i + stride]
                    hh[p + i] = a + b
                    hh[p + i + stride] = a - b
            h = hh
            stride <<= 1
        h = h * (1.0 / math.sqrt(128.0))
        h = h.bfloat16().float()
        absmax = h.abs().max().clamp_min(1e-4)
        sc = next_pow2_at_or_above(absmax.item() * (1.0 / 448.0))
        codes = [int(c) for c in
                 torch.clamp(h / sc, -448, 448).to(torch.float8_e4m3fn
                                                   ).view(torch.uint8).tolist()]
        index_k_t[j] = torch.tensor(codes, dtype=torch.uint8)
        index_scale_t[j] = sc

    # Selection + attention per query (python loops, exact semantics).
    select_k = topk // kpool
    max_selected = topk + kpool - 1
    attn_scale = float(nope) ** -0.5
    head_rows = nope + v_dim
    topk_rows = []
    attn_out = torch.zeros(tokens, heads * v_dim)
    q_fp8_f = q_fp8.view(tokens, idx_heads, 128).float()
    for t in range(tokens):
        pos = t
        visible = (pos + 1) // kpool
        row = [-1] * max_selected
        if visible <= select_k:
            for i in range(pos + 1):
                row[i] = i
        else:
            # Decode the fp8 codes before the dot: `.float()` on the raw
            # uint8 tensor would pair decoded q with raw byte values —
            # rankings stay nearly correct (e4m3 is monotone in the byte
            # within each sign class) and the error only surfaces as
            # single boundary swaps, which tolerance absorbs silently.
            ik_f = index_k_t[:visible].view(torch.float8_e4m3fn).float()
            is_f = index_scale_t[:visible]
            dots = torch.einsum("hd,pd->hp", q_fp8_f[t], ik_f)
            logits = (w_folded[t].unsqueeze(1) * is_f.unsqueeze(0) *
                      dots).sum(dim=0)
            order = torch.argsort(logits, descending=True, stable=True)
            pools = sorted(order[:select_k].tolist())
            wcount = 0
            for p in pools:
                for s in range(kpool):
                    row[wcount] = int(p) * kpool + s
                    wcount += 1
            tail_start = ((pos + 1) // kpool) * kpool
            for tt in range(tail_start, pos + 1):
                row[wcount] = tt
                wcount += 1
        topk_rows.extend(row)
        toks = [i for i in row if i >= 0]
        lat = latent[toks].float()
        for h in range(heads):
            w_uk = kv_b[h * head_rows:h * head_rows + nope].float()
            q_tilde = (q_mla[t, h * nope:(h + 1) * nope].float() @ w_uk
                       ).bfloat16().float()
            scores = (lat @ q_tilde) * attn_scale
            mx = scores.max()
            p = torch.exp(scores - mx)
            p = p / p.sum()
            p = p.bfloat16().float()
            c_vec = p @ lat
            w_uv = kv_b[h * head_rows + nope:(h + 1) * head_rows].float()
            attn_out[t, h * v_dim:(h + 1) * v_dim] = \
                (w_uv @ c_vec).bfloat16().float()

    layer_out = gemm_bf16_t(attn_out.bfloat16(), o_proj)

    cfg = {
        "hidden": hidden, "heads": heads, "q_lora": int(q_lora),
        "kv_lora": int(kv_lora), "nope": int(nope), "v_dim": int(v_dim),
        "idx_heads": int(idx_heads), "idx_dim": int(idx_dim),
        "topk": int(topk), "kpool": int(kpool), "tokens": tokens,
        "rms_eps": 1e-5, "seed": args.seed,
    }

    def tb(t):
        return t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()

    tensors = {
        "qkv_a": ("BF16", list(qkv_a.shape), tb(qkv_a)),
        "q_aln": ("BF16", [q_lora], tb(w("q_a_layernorm.weight"))),
        "kv_aln": ("BF16", [kv_lora], tb(w("kv_a_layernorm.weight"))),
        "q_b": ("BF16", list(q_b.shape), tb(q_b)),
        "wq_b": ("BF16", list(wq_b.shape), tb(wq_b)),
        "wk": ("BF16", list(wk.shape), tb(wk)),
        "gate": ("BF16", list(gate.shape), tb(gate)),
        "wp": ("BF16", list(wp.shape), tb(wp)),
        "k_norm_w": ("BF16", [128], tb(k_norm_w)),
        "k_norm_b": ("BF16", [128], tb(k_norm_b)),
        "kv_b": ("BF16", list(kv_b.shape), tb(kv_b)),
        "o_proj": ("BF16", list(o_proj.shape), tb(o_proj)),
        "ape": ("F32", [kpool, 128],
                ape.float().contiguous().numpy().tobytes()),
        "hidden_in": ("BF16", [tokens, hidden], tb(hidden_in)),
        "layer_out": ("BF16", [tokens, hidden], tb(layer_out)),
        "index_k": ("U8", [n_pools, 128],
                    index_k_t.contiguous().numpy().tobytes()),
        "index_scale": ("F32", [n_pools],
                        index_scale_t.contiguous().numpy().tobytes()),
        "latent": ("BF16", [tokens, kv_lora], tb(latent)),
        "topk": ("I32", [tokens, max_selected],
                 struct.pack("<%di" % len(topk_rows), *topk_rows)),
    }
    meta = {
        "model": os.path.basename(os.path.normpath(args.model_dir)),
        "revision": "checkpoint",
        "backend": "torch",
        "layer_idx": args.layer,
    }
    write_dump(args.out, meta, cfg, tensors)
    print("wrote %s (layer %d, %d tokens, heads=%d)" %
          (args.out, args.layer, tokens, heads))
    return 0


# ---------------------------------------------------------------------------
# Selftest: format round-trip + reference sanity, no torch required
# ---------------------------------------------------------------------------

def cmd_selftest(args):
    import tempfile

    # fp8 codec sanity against known encodings.
    known = {0.0: 0x00, 1.0: 0x38, -1.0: 0xB8, 0.5: 0x30, 2.0: 0x40,
             448.0: 0x7E, -448.0: 0xFE, 1e9: 0x7E}
    for v, code in known.items():
        got = fp8_bits(v)
        assert got == code, "fp8_bits(%g) = %02x, want %02x" % (v, got, code)
        if abs(v) <= 448:
            back = fp8_from_bits(code)
            assert abs(back - v) <= max(abs(v) * 0.125, 1e-9), \
                "fp8 round trip %g -> %g" % (v, back)

    # Power-of-two scales: exact powers map to themselves.
    for v in (1.0, 2.0, 0.5, 4.0, 2 ** -20):
        assert next_pow2_at_or_above(v) == v
    for v, want in ((1.5, 2.0), (3.0, 4.0), (0.6, 1.0)):
        assert next_pow2_at_or_above(v) == want

    # Selection: exact ties break to the lower pool index; ascending output.
    logits = [1.0, 3.0, 3.0, 2.0, 0.0]
    assert select_pools(logits, 3) == [1, 2, 3]
    assert select_pools([5.0, 5.0, 5.0], 2) == [0, 1]
    assert select_pools([1.0, 2.0], 8) == [0, 1]  # k > n -> all

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "selftest.dsadump")
        ns = argparse.Namespace(
            out=path, heads=4, q_lora=32, kv_lora=64, nope=32, v_dim=32,
            idx_heads=32, topk=16, kpool=4, hidden=64, tokens=24, seed=7,
            layer=0)
        gen_pure(ns)
        header, tensors = read_dump(path)
        assert header["backend"] == "pure"
        cfg = header["config"]
        assert cfg["tokens"] == 24 and cfg["kpool"] == 4

        # Different seeds -> different outputs (guards constant output).
        ns2 = argparse.Namespace(
            out=path + "2", heads=4, q_lora=32, kv_lora=64, nope=32, v_dim=32,
            idx_heads=32, topk=16, kpool=4, hidden=64, tokens=24, seed=8,
            layer=0)
        gen_pure(ns2)
        _, tensors2 = read_dump(path + "2")
        assert tensors["layer_out"][2] != tensors2["layer_out"][2]

        # Determinism: same seed, same bytes.
        ns3 = argparse.Namespace(
            out=path + "3", heads=4, q_lora=32, kv_lora=64, nope=32, v_dim=32,
            idx_heads=32, topk=16, kpool=4, hidden=64, tokens=24, seed=7,
            layer=0)
        gen_pure(ns3)
        _, tensors3 = read_dump(path + "3")
        assert tensors3["layer_out"][2] == tensors["layer_out"][2]

    print("dsa_reference_dump selftest OK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="format + reference sanity (no torch)")
    st.add_argument("--out", default=None, help="unused; kept for symmetry")
    st.set_defaults(fn=cmd_selftest)

    gp = sub.add_parser("gen-pure", help="synthetic pure-python reference dump")
    gp.add_argument("--out", required=True)
    # Defaults satisfy every kernel constraint (kv_lora >= per-head groups,
    # nope <= 256, kv_lora %% 8 == 0, 512 %% kv_lora == 0) while staying
    # small enough for the pure-python oracle's O(tokens^2) loops.
    gp.add_argument("--heads", type=int, default=4)
    gp.add_argument("--q-lora", type=int, default=32)
    gp.add_argument("--kv-lora", type=int, default=64)
    gp.add_argument("--nope", type=int, default=32)
    gp.add_argument("--v-dim", type=int, default=32)
    gp.add_argument("--idx-heads", type=int, default=32)
    gp.add_argument("--topk", type=int, default=16)
    gp.add_argument("--kpool", type=int, default=4)
    gp.add_argument("--hidden", type=int, default=64)
    gp.add_argument("--tokens", type=int, default=24)
    gp.add_argument("--seed", type=int, default=1234)
    gp.add_argument("--layer", type=int, default=0)
    gp.set_defaults(fn=gen_pure)

    gt = sub.add_parser("gen-torch", help="real-checkpoint reference dump (torch)")
    gt.add_argument("--model-dir", required=True)
    gt.add_argument("--layer", type=int, required=True)
    gt.add_argument("--out", required=True)
    gt.add_argument("--tokens", type=int, default=32)
    gt.add_argument("--seed", type=int, default=1234)
    gt.set_defaults(fn=gen_torch)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
