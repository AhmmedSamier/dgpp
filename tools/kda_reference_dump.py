#!/usr/bin/env python3
"""KDA reference-dump harness (M2 deliverable 5, DESIGN §12 tier 2).

Generates reference dumps for one KDA layer of GLM-5.3-Flash so the C++
parity test can compare our CUDA layer against an independently computed
result.

Two backends:

  pure   (stdlib only)
      Synthetic seeded weights and activations at a (by default small)
      geometry, reference math in python floats (IEEE double) with bf16
      rounding at the same boundaries as the engine. This is the local CI
      oracle: it runs anywhere python runs.

  torch  (requires torch; run where the checkpoint lives)
      Reads the real BF16 weights straight out of the safetensors shards
      (stdlib header parsing — no safetensors dependency), generates random
      activations, and runs the same reference equations in torch. This is
      the "real checkpoint slice" parity input.

The reference implements the vLLM GLM-5 KDA layer contract:

  qkvbfg_a = x @ in_proj.T                     (bf16 out, fp32 accumulate)
  g1 = f_a @ f_b.T ; g2 = g_a @ g_b.T          (bf16 out)
  q,k,v = silu(causal_dw_conv(q|k|v, w, state)) (fp32 math, bf16 io)
  per token (fp32 state S in R^{VxK}):
    q,k  <- l2norm(q,k)  (eps 1e-6 inside sqrt); q <- q * K**-0.5
    g[k] <- lower_bound / (1 + exp(-exp(A_log[h]) * (g1[h,k] + dt_bias[h,k])))
    S    <- diag(exp(g)) S
    u     = sigmoid(b_raw) * (v - S k)
    S    <- S + u k^T
    o     = S q                              (post-update read)
  core_out -> o_norm: rmsnorm * w * sigmoid(g2) (bf16 out)
  layer_out = core_normed @ o_proj.T           (bf16 out)

File format ("DGPPKDAD"): 8-byte magic, u32 version=1, u32 header length,
JSON header, payload. See src/models/kda_dump.cpp for the reader side.

Usage:
  kda_reference_dump.py selftest [--out DIR]
  kda_reference_dump.py gen-pure --out FILE [--heads H] [--head-dim D]
        [--hidden N] [--tokens T] [--conv-width W] [--lower-bound LB]
        [--seed S] [--layer L]
  kda_reference_dump.py gen-torch --model-dir DIR --layer N --out FILE
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

MAGIC = b"DGPPKDAD"
VERSION = 1


# ---------------------------------------------------------------------------
# bf16 helpers (round-to-nearest-even, matching src/common/dtypes.hpp)
# ---------------------------------------------------------------------------

def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16_round(x: float) -> float:
    """Round a python float (via fp32) to bf16 and back to python float."""
    b = f32_bits(x)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFF0000))[0]


def bf16_bits(x: float) -> int:
    b = f32_bits(x)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return (b >> 16) & 0xFFFF


def bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def bf16_to_bytes(values) -> bytes:
    return struct.pack("<%dH" % len(values), *(bf16_bits(v) for v in values))


def f32_to_bytes(values) -> bytes:
    return struct.pack("<%df" % len(values), *(float(v) for v in values))


def read_bf16(buf: memoryview, count: int, offset: int = 0):
    words = struct.unpack_from("<%dH" % count, buf, offset)
    return [bf16_from_bits(w) for w in words]


def read_f32(buf: memoryview, count: int, offset: int = 0):
    return list(struct.unpack_from("<%df" % count, buf, offset))


# ---------------------------------------------------------------------------
# Deterministic random (xorshift-ish hash seeding; no external deps)
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
    """Box-Muller normal samples."""
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
# Reference layer math (pure python, double interior)
# ---------------------------------------------------------------------------

def reference_conv(src, src_stride, weights, state, state_width, channels,
                   conv_width, tokens, start_col=0):
    """Causal depthwise conv + silu over `channels` starting at `start_col`
    of each `src_stride`-wide row. Returns (dst rows, rolled state)."""
    hist = [state[c * state_width:(c + 1) * state_width] for c in range(channels)]
    dst = []
    for t in range(tokens):
        row = []
        for c in range(channels):
            x = src[t * src_stride + start_col + c]
            acc = weights[c * conv_width + conv_width - 1] * x
            for j in range(conv_width - 1):
                acc += weights[c * conv_width + j] * hist[c][j]
            y = acc * (1.0 / (1.0 + math.exp(-acc)))
            row.append(bf16_round(y))
            hist[c] = hist[c][1:] + [x]
        dst.extend(row)
    rolled = []
    for c in range(channels):
        rolled.extend(bf16_round(v) for v in hist[c])
    return dst, rolled


def reference_recurrent(qkv, g1, beta, beta_stride, beta_off, a_log, dt_bias,
                        state, heads, head_dim, tokens, lower_bound):
    """KDA recurrence. `state` is a flat [heads, V, K] list of doubles,
    mutated in place. Returns bf16 core outputs [tokens, heads, V]."""
    k_dim = head_dim
    v_dim = head_dim
    scale = head_dim ** -0.5
    out = []
    qkv_stride = 2 * heads * k_dim + heads * v_dim
    for t in range(tokens):
        for h in range(heads):
            q_base = t * qkv_stride + h * k_dim
            k_base = q_base + heads * k_dim
            v_base = t * qkv_stride + 2 * heads * k_dim + h * v_dim
            q = [float(qkv[q_base + i]) for i in range(k_dim)]
            k = [float(qkv[k_base + i]) for i in range(k_dim)]
            v = [float(qkv[v_base + i]) for i in range(v_dim)]
            qs = sum(x * x for x in q)
            ks = sum(x * x for x in k)
            qn = 1.0 / math.sqrt(qs + 1e-6)
            kn = 1.0 / math.sqrt(ks + 1e-6)
            q = [x * qn * scale for x in q]
            k = [x * kn for x in k]

            a = math.exp(a_log[h])
            g = [g1[(t * heads + h) * k_dim + i] + dt_bias[h * k_dim + i]
                 for i in range(k_dim)]
            gate = [lower_bound / (1.0 + math.exp(-(a * x))) for x in g]

            s_off = h * v_dim * k_dim
            for i in range(k_dim):
                decay = math.exp(gate[i])
                if decay != 1.0:
                    base = s_off + i
                    for vv in range(v_dim):
                        state[base + vv * k_dim] *= decay

            beta_v = 1.0 / (1.0 + math.exp(-float(
                beta[t * beta_stride + beta_off + h])))
            for vv in range(v_dim):
                row = s_off + vv * k_dim
                dot = sum(state[row + i] * k[i] for i in range(k_dim))
                u = (v[vv] - dot) * beta_v
                o = 0.0
                for i in range(k_dim):
                    state[row + i] += u * k[i]  # must write through to state
                    o += state[row + i] * q[i]
                out.append(bf16_round(o))
    return out


def reference_layer(weights, hidden_in, state, conv_state, cfg):
    """Full layer forward on pure-python values. `weights` maps names to
    lists of python floats (already in bf16-rounded form for bf16 tensors).
    Mutates state (doubles) and conv_state (bf16-rounded floats); returns
    (layer_out, core_out, rolled_conv_state)."""
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    head_dim = cfg["head_dim"]
    conv_width = cfg["conv_width"]
    lower_bound = cfg["lower_bound"]
    tokens = cfg["tokens"]
    lp = heads * head_dim
    channels = 3 * lp
    n_in = 3 * lp + heads + 2 * head_dim
    # Must mirror the engine's fused layout: [f_a | g_a | q | k | v | b].
    off_q = 2 * head_dim
    off_k = off_q + lp
    off_v = off_k + lp
    off_b = off_v + lp

    def gemm(act, act_stride, w, m, n, kk):
        rows = []
        for r in range(m):
            for c in range(n):
                acc = 0.0
                for i in range(kk):
                    acc += act[r * act_stride + i] * w[c * kk + i]
                rows.append(bf16_round(acc))
        return rows

    proj = gemm(hidden_in, hidden, weights["in_proj"], tokens, n_in, hidden)
    # The two strided-input GEMMs read K-column slices of each fused row.
    f_a = [proj[t * n_in + i] for t in range(tokens) for i in range(head_dim)]
    g_a = [proj[t * n_in + head_dim + i] for t in range(tokens)
           for i in range(head_dim)]
    g1 = gemm(f_a, head_dim, weights["f_b"], tokens, lp, head_dim)
    g2 = gemm(g_a, head_dim, weights["g_b"], tokens, lp, head_dim)

    qkv_conv, rolled_conv = reference_conv(
        proj, n_in, weights["conv"], conv_state, cfg["conv_state_width"],
        channels, conv_width, tokens, start_col=off_q)

    core = reference_recurrent(qkv_conv, g1, proj, n_in, off_b,
                               weights["a_log"], weights["dt_bias"], state,
                               heads, head_dim, tokens, lower_bound)

    # o_norm: rmsnorm(core) * w * sigmoid(g2), bf16 out.
    normed = []
    for r in range(tokens * heads):
        row = core[r * head_dim:(r + 1) * head_dim]
        grw = g2[r * head_dim:(r + 1) * head_dim]
        var = sum(x * x for x in row) / head_dim
        rstd = 1.0 / math.sqrt(var + 1e-5)
        for i in range(head_dim):
            normed.append(bf16_round(row[i] * rstd * weights["o_norm"][i] *
                                     (1.0 / (1.0 + math.exp(-grw[i])))))

    layer_out = gemm(normed, lp, weights["o_proj"], tokens, hidden, lp)
    return layer_out, core, rolled_conv


# ---------------------------------------------------------------------------
# Dump file writer / reader (stdlib)
# ---------------------------------------------------------------------------

def write_dump(path, meta, cfg, tensors):
    header = {
        "format": "dgpp-kda-reference-dump",
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


# ---------------------------------------------------------------------------
# Backend: pure (synthetic weights, stdlib math)
# ---------------------------------------------------------------------------

def gen_pure(args):
    cfg = {
        "hidden": args.hidden,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "conv_width": args.conv_width,
        "lower_bound": args.lower_bound,
        "tokens": args.tokens,
        "conv_state_width": args.conv_width - 1,
        "seed": args.seed,
    }
    lp = cfg["heads"] * cfg["head_dim"]
    channels = 3 * lp
    n_in = 3 * lp + cfg["heads"] + 2 * cfg["head_dim"]
    rng = make_rng(args.seed)

    # Synthetic bf16 weights at modest scale. Input hidden states use a
    # stddev that keeps post-projection activations in a sane range.
    weights = {
        "in_proj": [bf16_round(v) for v in randn(rng, n_in * cfg["hidden"], 0.02)],
        "f_b": [bf16_round(v) for v in randn(rng, lp * cfg["head_dim"], 0.08)],
        "g_b": [bf16_round(v) for v in randn(rng, lp * cfg["head_dim"], 0.08)],
        "conv": [bf16_round(v) for v in randn(rng, channels * cfg["conv_width"], 0.15)],
        "a_log": [v for v in randn(rng, cfg["heads"], 0.4)],
        "dt_bias": [v for v in randn(rng, lp, 0.3)],
        "o_norm": [bf16_round(abs(v)) for v in randn(rng, cfg["head_dim"], 0.3)],
        "o_proj": [bf16_round(v) for v in randn(rng, cfg["hidden"] * lp, 0.02)],
    }
    hidden_in = [bf16_round(v) for v in randn(rng, cfg["tokens"] * cfg["hidden"], 1.0)]
    state = [0.0] * (cfg["heads"] * cfg["head_dim"] * cfg["head_dim"])
    conv_state = [0.0] * (channels * cfg["conv_state_width"])

    layer_out, core, rolled_conv = reference_layer(
        weights, hidden_in, state, conv_state, cfg)

    tensors = {
        "in_proj": ("BF16", [n_in, cfg["hidden"]], bf16_to_bytes(weights["in_proj"])),
        "f_b": ("BF16", [lp, cfg["head_dim"]], bf16_to_bytes(weights["f_b"])),
        "g_b": ("BF16", [lp, cfg["head_dim"]], bf16_to_bytes(weights["g_b"])),
        "conv": ("BF16", [channels, cfg["conv_width"]], bf16_to_bytes(weights["conv"])),
        "a_log": ("F32", [cfg["heads"]], f32_to_bytes(weights["a_log"])),
        "dt_bias": ("F32", [lp], f32_to_bytes(weights["dt_bias"])),
        "o_norm": ("BF16", [cfg["head_dim"]], bf16_to_bytes(weights["o_norm"])),
        "o_proj": ("BF16", [cfg["hidden"], lp], bf16_to_bytes(weights["o_proj"])),
        "hidden_in": ("BF16", [cfg["tokens"], cfg["hidden"]], bf16_to_bytes(hidden_in)),
        "layer_out": ("BF16", [cfg["tokens"], cfg["hidden"]], bf16_to_bytes(layer_out)),
        "core_out": ("BF16", [cfg["tokens"], lp], bf16_to_bytes(core)),
        "recurrent_state_out": ("F32", [cfg["heads"], cfg["head_dim"], cfg["head_dim"]],
                                f32_to_bytes(state)),
        "conv_state_out": ("BF16", [channels, cfg["conv_state_width"]],
                           bf16_to_bytes(rolled_conv)),
    }
    meta = {
        "model": "synthetic-kda",
        "revision": "pure-%d" % args.seed,
        "backend": "pure",
        "layer_idx": args.layer,
    }
    write_dump(args.out, meta, cfg, tensors)
    print("wrote %s (%d tokens, heads=%d, head_dim=%d)" %
          (args.out, cfg["tokens"], cfg["heads"], cfg["head_dim"]))
    return 0


# ---------------------------------------------------------------------------
# Backend: torch (real checkpoint slices)
# ---------------------------------------------------------------------------

def read_safetensors_index(model_dir):
    """Returns {tensor_name: (shard_file, byte_offset, nbytes, dtype_str,
    shape)} using only the stdlib."""
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
    if dtype_str not in dtype_map:
        raise ValueError("unsupported dtype %s for %s" % (dtype_str, name))
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

    if args.heads or args.head_dim or args.hidden:
        raise SystemExit("gen-torch reads geometry from the checkpoint config")

    with open(os.path.join(args.model_dir, "config.json")) as f:
        full = json.load(f)
    text = full.get("text_config", full)
    lin = text.get("linear_attn_config", {})
    hidden = text["hidden_size"]
    heads = lin.get("num_heads", text.get("linear_num_heads"))
    head_dim = lin.get("head_dim", text.get("linear_head_dim"))
    conv_width = lin.get("short_conv_kernel_size",
                         text.get("linear_conv_kernel_dim", 4))
    lower_bound = lin.get("gate_lower_bound",
                          text.get("linear_lower_bound", -5.0))
    if not (heads and head_dim):
        raise SystemExit("config.json lacks linear attention head geometry")

    prefix = "model.language_model.layers.%d.self_attn" % args.layer
    entries = read_safetensors_index(args.model_dir)

    def w(name):
        return load_tensor(entries, "%s.%s" % (prefix, name), torch)

    q_proj, k_proj, v_proj = w("q_proj.weight"), w("k_proj.weight"), w("v_proj.weight")
    b_proj = w("b_proj.weight")
    f_a, g_a = w("f_a_proj.weight"), w("g_a_proj.weight")
    f_b, g_b = w("f_b_proj.weight"), w("g_b_proj.weight")
    q_conv, k_conv, v_conv = w("q_conv1d.weight"), w("k_conv1d.weight"), w("v_conv1d.weight")
    a_log, dt_bias = w("A_log"), w("dt_bias")
    o_norm, o_proj = w("o_norm.weight"), w("o_proj.weight")

    # Engine fused layout: [f_a | g_a | q | k | v | b].
    in_proj = torch.cat([f_a, g_a, q_proj, k_proj, v_proj, b_proj], dim=0).contiguous()
    conv_w = torch.cat([q_conv.reshape(q_conv.shape[0], q_conv.shape[2]),
                        k_conv.reshape(k_conv.shape[0], k_conv.shape[2]),
                        v_conv.reshape(v_conv.shape[0], v_conv.shape[2])],
                       dim=0).contiguous()

    lp = heads * head_dim
    channels = 3 * lp
    tokens = args.tokens
    torch.manual_seed(args.seed)
    hidden_in = (torch.randn(tokens, hidden, dtype=torch.float32) * 1.0).bfloat16()

    # ---- reference forward in torch (fp32 interior, bf16 boundaries) ----
    def gemm_bf16(x, weight):
        return (x.float() @ weight.float().T).bfloat16()

    proj = gemm_bf16(hidden_in, in_proj)                       # [T, n_in]
    f_a_act = proj[:, :head_dim]
    g_a_act = proj[:, head_dim:2 * head_dim]
    qkv = proj[:, 2 * head_dim:2 * head_dim + 3 * lp]
    beta_raw = proj[:, 2 * head_dim + 3 * lp:]
    g1 = gemm_bf16(f_a_act, f_b)                               # [T, lp]
    g2 = gemm_bf16(g_a_act, g_b)

    # causal depthwise conv + silu, per channel, width conv_width
    conv_in = qkv.float()                                      # [T, channels]
    w_conv = conv_w.float()                                    # [channels, W]
    y = torch.zeros(tokens, channels)
    for t in range(tokens):
        acc = w_conv[:, conv_width - 1] * conv_in[t]
        for j in range(conv_width - 1):
            src_t = t - (conv_width - 1) + j
            xj = conv_in[src_t] if src_t >= 0 else torch.zeros(channels)
            acc = acc + w_conv[:, j] * xj
        y[t] = acc * torch.sigmoid(acc)
    qkv_conv = y.bfloat16()

    # recurrence in fp32, sequential
    state = torch.zeros(heads, head_dim, head_dim, dtype=torch.float32)
    core = torch.zeros(tokens, heads, head_dim)
    scale = head_dim ** -0.5
    qkv_f = qkv_conv.float()
    for t in range(tokens):
        q = qkv_f[t, 0 * lp:1 * lp].view(heads, head_dim)
        k = qkv_f[t, 1 * lp:2 * lp].view(heads, head_dim)
        v = qkv_f[t, 2 * lp:3 * lp].view(heads, head_dim)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * scale
        g = g1[t].float().view(heads, head_dim) + dt_bias.view(heads, head_dim)
        gate = lower_bound / (1.0 + torch.exp(-(torch.exp(a_log).view(heads, 1) * g)))
        state = state * torch.exp(gate).unsqueeze(1)           # decay K columns
        beta = torch.sigmoid(beta_raw[t].float())              # [heads]
        # delta: u = beta * (v - (S @ k)); S is [H, V, K]
        sk = torch.bmm(state, k.unsqueeze(2)).squeeze(2)       # [H, V]
        u = (v - sk) * beta.unsqueeze(1)
        state = state + u.unsqueeze(2) * k.unsqueeze(1)
        core[t] = torch.bmm(state, q.unsqueeze(2)).squeeze(2)
    core = core.bfloat16()

    # o_norm
    core3 = core.view(tokens * heads, head_dim).float()
    var = (core3 * core3).mean(-1, keepdim=True)
    normed = core3 * torch.rsqrt(var + 1e-5) * o_norm.float() * torch.sigmoid(
        g2.view(tokens * heads, head_dim).float())
    normed = normed.bfloat16().view(tokens, lp)

    layer_out = gemm_bf16(normed, o_proj)

    cfg = {
        "hidden": hidden, "heads": heads, "head_dim": head_dim,
        "conv_width": conv_width, "lower_bound": lower_bound,
        "tokens": tokens, "conv_state_width": conv_width - 1, "seed": args.seed,
    }
    meta = {
        "model": os.path.basename(os.path.normpath(args.model_dir)),
        "revision": "checkpoint",
        "backend": "torch",
        "layer_idx": args.layer,
    }

    def blob_bf16(t):
        t = t.detach().cpu().contiguous()
        return t.view(torch.uint16).numpy().tobytes()

    def blob_f32(t):
        return t.detach().cpu().contiguous().numpy().tobytes()

    tensors = {
        "in_proj": ("BF16", list(in_proj.shape), blob_bf16(in_proj)),
        "f_b": ("BF16", list(f_b.shape), blob_bf16(f_b)),
        "g_b": ("BF16", list(g_b.shape), blob_bf16(g_b)),
        "conv": ("BF16", list(conv_w.shape), blob_bf16(conv_w.bfloat16())),
        "a_log": ("F32", [heads], blob_f32(a_log.float())),
        "dt_bias": ("F32", [lp], blob_f32(dt_bias.float())),
        "o_norm": ("BF16", [head_dim], blob_bf16(o_norm)),
        "o_proj": ("BF16", list(o_proj.shape), blob_bf16(o_proj)),
        "hidden_in": ("BF16", [tokens, hidden], blob_bf16(hidden_in)),
        "layer_out": ("BF16", [tokens, hidden], blob_bf16(layer_out)),
        "core_out": ("BF16", [tokens, lp], blob_bf16(core)),
        "recurrent_state_out": ("F32", [heads, head_dim, head_dim],
                                blob_f32(state)),
    }
    write_dump(args.out, meta, cfg, tensors)
    print("wrote %s (layer %d, %d tokens, heads=%d, head_dim=%d)" %
          (args.out, args.layer, tokens, heads, head_dim))
    return 0


# ---------------------------------------------------------------------------
# Selftest: format round-trip + reference sanity, no torch required
# ---------------------------------------------------------------------------

def cmd_selftest(args):
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "selftest.kdadump")
        ns = argparse.Namespace(
            out=path, heads=2, head_dim=8, hidden=16, tokens=3,
            conv_width=4, lower_bound=-5.0, seed=7, layer=0)
        gen_pure(ns)

        header, tensors = read_dump(path)
        assert header["backend"] == "pure"
        cfg = header["config"]
        assert cfg["heads"] == 2 and cfg["head_dim"] == 8

        # Recompute one reference output element independently: layer_out
        # must change when the input changes (guards a constant-output bug).
        ns2 = argparse.Namespace(
            out=path + "2", heads=2, head_dim=8, hidden=16, tokens=3,
            conv_width=4, lower_bound=-5.0, seed=8, layer=0)
        gen_pure(ns2)
        _, tensors2 = read_dump(path + "2")
        a = tensors["layer_out"][2]
        b = tensors2["layer_out"][2]
        assert a != b, "different seeds produced identical outputs"

        # Determinism: same seed, same bytes.
        ns3 = argparse.Namespace(
            out=path + "3", heads=2, head_dim=8, hidden=16, tokens=3,
            conv_width=4, lower_bound=-5.0, seed=7, layer=0)
        gen_pure(ns3)
        _, tensors3 = read_dump(path + "3")
        assert tensors3["layer_out"][2] == a, "same seed did not reproduce"

        # The decay gate must stay bounded in (lower_bound, 0) — evaluate
        # directly at a few points.
        for x in (-10.0, -1.0, 0.0, 1.0, 10.0):
            g = -5.0 / (1.0 + math.exp(-(math.exp(0.4) * x)))
            assert -5.0 < g < 0.0

        # bf16 rounding sanity: known values round-trip exactly; a large but
        # finite fp32 stays finite (values past bf16 max-finite legitimately
        # round to inf, same as the C++ helper).
        for v in (0.0, 1.0, -1.0, 0.5, 1e-8, 3.0e38):
            assert bf16_round(v) == v or abs(bf16_round(v) - v) <= abs(v) * 2 ** -8
        assert bf16_round(3.4e38) == math.inf

    print("kda_reference_dump selftest OK")
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
    gp.add_argument("--heads", type=int, default=2)
    gp.add_argument("--head-dim", type=int, default=32)
    gp.add_argument("--hidden", type=int, default=64)
    gp.add_argument("--tokens", type=int, default=8)
    gp.add_argument("--conv-width", type=int, default=4)
    gp.add_argument("--lower-bound", type=float, default=-5.0)
    gp.add_argument("--seed", type=int, default=1234)
    gp.add_argument("--layer", type=int, default=0)
    gp.set_defaults(fn=gen_pure)

    gt = sub.add_parser("gen-torch", help="real-checkpoint reference dump (torch)")
    gt.add_argument("--model-dir", required=True)
    gt.add_argument("--layer", type=int, required=True)
    gt.add_argument("--out", required=True)
    gt.add_argument("--tokens", type=int, default=32)
    gt.add_argument("--seed", type=int, default=1234)
    gt.add_argument("--heads", type=int, default=0)
    gt.add_argument("--head-dim", type=int, default=0)
    gt.add_argument("--hidden", type=int, default=0)
    gt.set_defaults(fn=gen_torch)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
