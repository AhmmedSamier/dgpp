#!/usr/bin/env python3
"""GLM-4.7 full-model reference dump (docs/glm47_plan.md G4, 2026-09-09).

Computes the Glm4MoeForCausalLM forward — the embedding, the pre-norm
decoder layers (biased GQA attention with per-head q/k norms and the
half-split partial RoPE; the NVFP4 dense MLP on the first layers, the
sigmoid-routed MoE with its NVFP4 shared expert after), the final norm,
the lm head — and the MTP draft block's rows over the same prompt, in pure
python doubles with bf16 rounding at the engine's boundaries, from the SAME
checkpoint the engine runs, and writes the comparable outputs (tokens,
every layer's residual, the final read, per-token top-k logits, the
routing decisions, the draft's rows) as a DGPPG4ND dump read by
tests/cuda/glm4_forward_test.cpp.

The wiring is pinned to transformers' modeling_glm4_moe.py (4.57):
Glm4MoeDecoderLayer, Glm4MoeAttention (the standard half-split rotate_half
over the first rotary_dim dims), Glm4MoeMoE (noaux_tc sigmoid router,
e_score_correction_bias, routed_scaling_factor, norm_topk_prob) and vLLM's
glm4_moe_mtp.py for the draft; the rounding points are the engine's
(src/models/glm4/attn_reference.cpp, src/models/glm/moe_reference.cpp).

NVFP4 weights (modelopt): w = e2m1(code) * e4m3(block scale) * weight_scale_2;
the engine keeps e2m1 * e4m3 exact (bf16) and applies the tensor scale in
the GEMV's epilogue, so the reference carries the exact products and
scales the dot. The draft layer's BF16 experts are requantized with the
engine's recipe (src/loaders/nvfp4_quant.hpp) before use — the dump
compares against what the engine actually multiplies.

Usage:
  glm4_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from operator import mul

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_reference_dump as krd  # noqa: E402
from dsa_reference_dump import _rne, fp8_from_bits  # noqa: E402
from glm_reference_dump import read_safetensors_index  # noqa: E402

MAGIC = b"DGPPG4ND"
VERSION = 1

bf16_round = krd.bf16_round
bf16_bits = krd.bf16_bits
bf16_from_bits = krd.bf16_from_bits
make_rng = krd.make_rng


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def dot(a, b) -> float:
    return sum(map(mul, a, b))


def gemv(x, rows):
    """[bf16(x . row) for row in rows] — a bf16 Linear (one rounding)."""
    return [bf16_round(dot(x, r)) for r in rows]


def as_rows(flat, n, k):
    return [flat[i * k:(i + 1) * k] for i in range(n)]


# ---------------------------------------------------------------------------
# the NVFP4 codecs (src/kernels/latent_format.hpp, src/common/dtypes.hpp)
# ---------------------------------------------------------------------------

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_value(code: int) -> float:
    v = E2M1[code & 7]
    return -v if code & 8 else v


def e2m1_bits(f: float) -> int:
    """Round-to-nearest-even on the e2m1 grid, saturating (the engine's
    float_to_fp4_e2m1_bits)."""
    f = f32(f)
    sign = 8 if math.copysign(1.0, f) < 0 else 0
    a = abs(f)
    if a != a:
        return sign | 7
    if a <= 0.25:
        code = 0
    elif a < 0.75:
        code = 1
    elif a <= 1.25:
        code = 2
    elif a < 1.75:
        code = 3
    elif a <= 2.5:
        code = 4
    elif a < 3.5:
        code = 5
    elif a <= 5.0:
        code = 6
    else:
        code = 7
    return sign | code


def e4m3_bits(f: float) -> int:
    """The engine's float_to_fp8_e4m3_bits (saturating, RNE)."""
    f = f32(f)
    sign = 0x80 if math.copysign(1.0, f) < 0 else 0
    av = abs(f)
    if av != av:
        return sign | 0x7F
    if av < 0.015625:
        n = _rne(f32(av * 512.0))
        if n <= 0:
            return sign
        if n >= 8:
            fld, man = 1, 0
        else:
            fld, man = 0, n
    else:
        e = 0
        q = av
        while q < 1.0:
            q *= 2.0
            e -= 1
        while q >= 2.0:
            q *= 0.5
            e += 1
        space = 2.0 ** (e - 3)
        n = _rne(f32(av / space))
        if n >= 16:
            e += 1
            n >>= 1
        if e > 8 or (e == 8 and n >= 15):
            return sign | 0x7E
        fld = e + 7
        man = n - 8
    return sign | (fld << 3) | man


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

_ITEMSIZE = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}


def load_raw(entries, name):
    """(dtype, shape, values): BF16/F32 as floats, F8_E4M3/U8 as integer codes."""
    path, offset, nbytes, dtype_str, shape = entries[name]
    if dtype_str not in _ITEMSIZE:
        raise ValueError("unsupported dtype %s for %s" % (dtype_str, name))
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    if dtype_str == "BF16":
        words = struct.unpack("<%dH" % (nbytes // 2), buf)
        return dtype_str, shape, [bf16_from_bits(w) for w in words]
    if dtype_str == "F32":
        return dtype_str, shape, list(struct.unpack("<%df" % (nbytes // 4), buf))
    return dtype_str, shape, list(buf)


def load_bf16(entries, name):
    dtype, shape, values = load_raw(entries, name)
    if dtype not in ("BF16", "F32"):
        raise ValueError("expected BF16/F32 for %s" % name)
    return shape, values


class Fp4:
    """A dequantized NVFP4 matrix: rows of the exact e2m1 x e4m3 products
    and the tensor scale the GEMV applies to the dot."""

    def __init__(self, rows, ws2):
        self.rows = rows
        self.ws2 = ws2


def load_nvfp4(entries, name):
    dtype, (n, half), payload = load_raw(entries, name + ".weight")
    if dtype != "U8":
        raise ValueError("expected U8 payload for %s" % name)
    _, (n2, blocks), scales = load_raw(entries, name + ".weight_scale")
    _, _, ws2 = load_raw(entries, name + ".weight_scale_2")
    k = half * 2
    if n2 != n or blocks * 16 != k:
        raise ValueError("NVFP4 shapes disagree for %s" % name)
    rows = []
    for r in range(n):
        row = [0.0] * k
        pbase, sbase = r * half, r * blocks
        for c in range(k):
            byte = payload[pbase + c // 2]
            code = (byte >> 4) if (c & 1) else (byte & 0xF)
            row[c] = e2m1_value(code) * fp8_from_bits(scales[sbase + c // 16])
        rows.append(row)
    return Fp4(rows, f32(ws2[0]))


def requant_bf16(shape, values):
    """The engine's load-time requant of a BF16 [n, k] matrix
    (nvfp4_tensor_scale + nvfp4_encode_row), decoded back to the products
    the kernels multiply."""
    n, k = shape
    amax = max((abs(v) for v in values), default=0.0)
    ws2 = f32(amax / (6.0 * 448.0)) if amax > 0 else 1.0
    rows = []
    for r in range(n):
        src = values[r * k:(r + 1) * k]
        row = [0.0] * k
        for b in range(k // 16):
            blk = src[b * 16:(b + 1) * 16]
            bmax = max(abs(v) for v in blk)
            sc = e4m3_bits(f32(f32(bmax / 6.0) / ws2))
            S = f32(fp8_from_bits(sc) * ws2)
            inv = f32(1.0 / S) if S > 0 else 0.0
            svalue = fp8_from_bits(sc)
            for j in range(16):
                code = e2m1_bits(f32(blk[j] * inv))
                row[b * 16 + j] = e2m1_value(code) * svalue
        rows.append(row)
    return Fp4(rows, ws2)


def fp4_gemv(x, m: Fp4):
    """bf16 rows of x against an NVFP4 matrix: the dot, the tensor scale,
    one rounding (the fp4 GEMV core's epilogue)."""
    return [bf16_round(dot(x, r) * m.ws2) for r in m.rows]


def fp4_dots(x, m: Fp4):
    """The same, unrounded (the MoE chain's fp32 partial dots)."""
    return [dot(x, r) * m.ws2 for r in m.rows]


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        tc = json.load(f)
    eos = tc["eos_token_id"]
    hd = tc["head_dim"]
    return {
        "hidden": tc["hidden_size"], "vocab": tc["vocab_size"],
        "num_layers": tc["num_hidden_layers"], "first_dense": tc["first_k_dense_replace"],
        "eps": tc.get("rms_norm_eps", 1e-5), "eos": eos[0] if isinstance(eos, list) else eos,
        "heads": tc["num_attention_heads"], "kv_heads": tc["num_key_value_heads"], "head_dim": hd,
        "rotary": int(hd * tc.get("partial_rotary_factor", 1.0)), "theta": float(tc.get("rope_theta", 1e6)),
        "attention_bias": tc.get("attention_bias", True), "qk_norm": tc.get("use_qk_norm", True),
        "dense_inter": tc["intermediate_size"], "moe_inter": tc["moe_intermediate_size"],
        "experts": tc["n_routed_experts"], "top_k": tc["num_experts_per_tok"],
        "n_shared": tc.get("n_shared_experts", 1), "norm_topk": tc.get("norm_topk_prob", True),
        "scaling": float(tc.get("routed_scaling_factor", 1.0)),
        "mtp": tc.get("num_nextn_predict_layers", 0),
    }


LAYER = "model.layers.%d."


def layer_weights(cfg, entries, layer):
    """A main-stack layer, or the draft layer (layer == num_layers): the
    same set plus enorm/hnorm/eh_proj/shared_head.norm, its experts
    requantized from BF16."""
    p = LAYER % layer
    H, D = cfg["hidden"], cfg["head_dim"]
    draft = layer == cfg["num_layers"]
    w = {"moe": layer >= cfg["first_dense"], "draft": draft}
    w["input_norm"] = load_bf16(entries, p + "input_layernorm.weight")[1]
    w["post_norm"] = load_bf16(entries, p + "post_attention_layernorm.weight")[1]
    a = p + "self_attn."
    w["q"] = as_rows(load_bf16(entries, a + "q_proj.weight")[1], cfg["heads"] * D, H)
    w["k"] = as_rows(load_bf16(entries, a + "k_proj.weight")[1], cfg["kv_heads"] * D, H)
    w["v"] = as_rows(load_bf16(entries, a + "v_proj.weight")[1], cfg["kv_heads"] * D, H)
    w["o"] = as_rows(load_bf16(entries, a + "o_proj.weight")[1], H, cfg["heads"] * D)
    if cfg["attention_bias"]:
        w["q_bias"] = load_bf16(entries, a + "q_proj.bias")[1]
        w["k_bias"] = load_bf16(entries, a + "k_proj.bias")[1]
        w["v_bias"] = load_bf16(entries, a + "v_proj.bias")[1]
    if cfg["qk_norm"]:
        w["q_norm"] = load_bf16(entries, a + "q_norm.weight")[1]
        w["k_norm"] = load_bf16(entries, a + "k_norm.weight")[1]

    def fp4_or_requant(name):
        if draft:
            return requant_bf16(*load_bf16(entries, name + ".weight"))
        return load_nvfp4(entries, name)

    m = p + "mlp."
    if w["moe"]:
        E = cfg["experts"]
        w["router"] = as_rows(load_bf16(entries, m + "gate.weight")[1], E, H)
        w["router_bias"] = load_bf16(entries, m + "gate.e_score_correction_bias")[1]
        w["experts"] = [(fp4_or_requant(m + "experts.%d.gate_proj" % e), fp4_or_requant(m + "experts.%d.up_proj" % e),
                         fp4_or_requant(m + "experts.%d.down_proj" % e)) for e in range(E)]
        w["shared"] = (fp4_or_requant(m + "shared_experts.gate_proj"), fp4_or_requant(m + "shared_experts.up_proj"),
                       fp4_or_requant(m + "shared_experts.down_proj"))
    else:
        w["dense"] = (fp4_or_requant(m + "gate_proj"), fp4_or_requant(m + "up_proj"), fp4_or_requant(m + "down_proj"))
    if draft:
        w["enorm"] = load_bf16(entries, p + "enorm.weight")[1]
        w["hnorm"] = load_bf16(entries, p + "hnorm.weight")[1]
        w["eh_proj"] = as_rows(load_bf16(entries, p + "eh_proj.weight")[1], H, 2 * H)
        w["shared_head_norm"] = load_bf16(entries, p + "shared_head.norm.weight")[1]
    return w


# ---------------------------------------------------------------------------
# modules (double interior, bf16 at the engine's boundaries)
# ---------------------------------------------------------------------------

def rmsnorm2(x, w, eps):
    """The two-rounding GLM RMSNorm: u = bf16(x * rstd), y = bf16(w * u)."""
    var = 0.0
    for v in x:
        var += v * v
    var /= len(x)
    rstd = 1.0 / math.sqrt(var + eps)
    return [bf16_round(w[i] * bf16_round(x[i] * rstd)) for i in range(len(x))]


def rope_inv(cfg):
    R = cfg["rotary"]
    return [f32(1.0 / (f32(cfg["theta"]) ** f32(2 * i / R))) for i in range(R // 2)]


def head_finish(dots, bias, norm, pos, inv, R, eps):
    """bf16(dot + bias), the two-rounding head norm, the half-split partial
    RoPE with the reference's bf16 ops (attn_reference.cpp's qkv_finish)."""
    x = [bf16_round(d + (bias[i] if bias is not None else 0.0)) for i, d in enumerate(dots)]
    if norm is None:
        return x
    x = rmsnorm2(x, norm, eps)
    half = R // 2  # transformers' rotate_half: pairs (i, i + half), not interleaved
    for i in range(half):
        ang = f32(f32(float(pos)) * inv[i])
        c = bf16_round(f32(math.cos(ang)))
        s = bf16_round(f32(math.sin(ang)))
        x1, x2 = x[i], x[i + half]
        x[i] = bf16_round(bf16_round(x1 * c) + bf16_round(-x2 * s))
        x[i + half] = bf16_round(bf16_round(x2 * c) + bf16_round(x1 * s))
    return x


ATTN_TILE = 32  # kernels/glm4_attn.hpp's kGlm4AttnTile: the prefill's chain runs one split of 32-token tiles


def attention_forward(x_rows, w, cfg, T):
    """Rows t = 0 .. T-1 at positions t; row t attends [0, t] (the paged
    cache holds exactly these rows). The kernel's online chain over
    32-token tiles: each tile's probabilities rounded to bf16 against the
    running max, the sums rescaled (attn_reference.cpp's tile form), the
    denominator unrounded, out = bf16(c / l)."""
    D, R = cfg["head_dim"], cfg["rotary"]
    lh, lkv = cfg["heads"], cfg["kv_heads"]
    hpk = lh // lkv
    inv = rope_inv(cfg)
    eps = cfg["eps"]
    scale = f32(D ** -0.5)
    qb, kb, vb = w.get("q_bias"), w.get("k_bias"), w.get("v_bias")
    qn, kn = w.get("q_norm"), w.get("k_norm")
    q, k, v = [], [], []
    for t, x in enumerate(x_rows):
        qd = [dot(x, r) for r in w["q"]]
        kd = [dot(x, r) for r in w["k"]]
        vd = [dot(x, r) for r in w["v"]]
        q.append([head_finish(qd[h * D:(h + 1) * D], qb[h * D:(h + 1) * D] if qb else None, qn, t, inv, R, eps)
                  for h in range(lh)])
        k.append([head_finish(kd[h * D:(h + 1) * D], kb[h * D:(h + 1) * D] if kb else None, kn, t, inv, R, eps)
                  for h in range(lkv)])
        v.append([head_finish(vd[h * D:(h + 1) * D], vb[h * D:(h + 1) * D] if vb else None, None, t, inv, 0, eps)
                  for h in range(lkv)])
    out = []
    for t in range(T):
        o = []
        for h in range(lh):
            kvh = h // hpk
            qh = q[t][h]
            s = [dot(qh, k[j][kvh]) * scale for j in range(t + 1)]
            m, l = -math.inf, 0.0
            c = [0.0] * D
            for t0 in range(0, t + 1, ATTN_TILE):
                t1 = min(t + 1, t0 + ATTN_TILE)
                m_new = max(m, max(s[t0:t1]))
                rescale = math.exp(m - m_new)
                l = l * rescale + sum(math.exp(v_ - m_new) for v_ in s[t0:t1])
                for d in range(D):
                    c[d] *= rescale
                for j in range(t0, t1):
                    p = bf16_round(math.exp(s[j] - m_new))
                    vrow = v[j][kvh]
                    for d in range(D):
                        c[d] += p * vrow[d]
                m = m_new
            o.extend(bf16_round(c[d] / l) for d in range(D))
        out.append(gemv(o, w["o"]))
    return out


def swiglu(g, u):
    """act = bf16(bf16(silu(g)) * u) — no clamps (Glm4MoeMLP)."""
    return bf16_round(bf16_round(g * sigmoid(g)) * u)


def dense_forward(x, dense):
    wg, wu, wd = dense
    g = fp4_gemv(x, wg)
    u = fp4_gemv(x, wu)
    act = [swiglu(gi, ui) for gi, ui in zip(g, u)]
    return fp4_gemv(act, wd)


def moe_forward(x, w, cfg):
    """The GLM router (sigmoid scores, the correction bias on the selection
    key, ties to the lower id, the picked scores normalized and scaled) and
    the engine's chain: unrounded partial dots, the experts in ascending id
    order, the shared expert last, one rounding."""
    H, E, K = cfg["hidden"], cfg["experts"], cfg["top_k"]
    logits = [dot(x, r) for r in w["router"]]
    scores = [sigmoid(v) for v in logits]
    biased = [scores[e] + w["router_bias"][e] for e in range(E)]
    order = sorted(range(E), key=lambda e: (-biased[e], e))
    sel = sorted(order[:K])
    # The selection margin (the near-tie report: a flip against the engine
    # inside it is the fp32 router's rounding, not a wiring difference).
    NEAR_TIES.append(biased[order[K - 1]] - biased[order[K]])
    denom = (sum(scores[e] for e in sel) + 1e-20) if cfg["norm_topk"] else 1.0
    weights = [(scores[e] / denom) * cfg["scaling"] for e in sel]
    acc = [0.0] * H
    for e, we in zip(sel, weights):
        wg, wu, wd = w["experts"][e]
        g = fp4_gemv(x, wg)
        u = fp4_gemv(x, wu)
        act = [swiglu(gi, ui) for gi, ui in zip(g, u)]
        y = fp4_dots(act, wd)
        for j in range(H):
            acc[j] += we * y[j]
    wg, wu, wd = w["shared"]
    g = fp4_gemv(x, wg)
    u = fp4_gemv(x, wu)
    act = [swiglu(gi, ui) for gi, ui in zip(g, u)]
    y = fp4_dots(act, wd)
    for j in range(H):
        acc[j] += y[j]
    return [bf16_round(v) for v in acc], sel


NEAR_TIES = []  # every routed row's margin between the K-th and (K+1)-th biased score


def layer_forward(h_rows, w, cfg, T):
    """h += attn(input_norm(h)); h += mlp(post_norm(h)) — bf16 residual adds."""
    H, eps = cfg["hidden"], cfg["eps"]
    x_rows = [rmsnorm2(h, w["input_norm"], eps) for h in h_rows]
    y = attention_forward(x_rows, w, cfg, T)
    h_rows = [[bf16_round(a + b) for a, b in zip(h, yy)] for h, yy in zip(h_rows, y)]
    x_rows = [rmsnorm2(h, w["post_norm"], eps) for h in h_rows]
    routes = []
    if w["moe"]:
        y = []
        for x in x_rows:
            yy, sel = moe_forward(x, w, cfg)
            y.append(yy)
            routes.append(sel)
    else:
        y = [dense_forward(x, w["dense"]) for x in x_rows]
    h_rows = [[bf16_round(a + b) for a, b in zip(h, yy)] for h, yy in zip(h_rows, y)]
    return h_rows, routes


def reference_forward(cfg, entries, tokens, progress=False):
    H = cfg["hidden"]
    T = len(tokens)
    _, embed = load_bf16(entries, "model.embed_tokens.weight")
    h = [list(embed[tok * H:(tok + 1) * H]) for tok in tokens]
    layer_states, routes = [], []
    for layer in range(cfg["num_layers"]):
        w = layer_weights(cfg, entries, layer)
        if progress:
            print("layer %d (%s)" % (layer, "moe" if w["moe"] else "dense"), file=sys.stderr, flush=True)
        NEAR_TIES.clear()
        h, route = layer_forward(h, w, cfg, T)
        layer_states.append([v for row in h for v in row])
        if w["moe"]:
            routes.append(route)
            if progress:
                tight = sorted((m, t) for t, m in enumerate(NEAR_TIES))[:3]
                print("  routing margins (smallest):" + "".join(" t%d %.3g" % (t, m) for m, t in tight),
                      file=sys.stderr, flush=True)
    _, final_norm = load_bf16(entries, "model.norm.weight")
    hn = [rmsnorm2(row, final_norm, cfg["eps"]) for row in h]
    _, lm = load_bf16(entries, "lm_head.weight")
    lm_rows = as_rows(lm, cfg["vocab"], H)
    logits = [[f32(dot(x, r)) for r in lm_rows] for x in hn]
    return layer_states, hn, logits, routes, h


def mtp_forward(cfg, entries, tokens, h_last, progress=False):
    """The draft block over the prompt: row q embeds tokens[q + 1] and takes
    the main stack's OUTPUT hidden at q (after the final norm — vLLM's
    glm4_moe_mtp receives the model's hidden states and applies hnorm) —
    x_q = eh_proj([enorm(e) | hnorm(h_q)]) — then the draft layer (its own
    cache, positions q), its shared_head.norm and the shared lm head.
    Returns (h_rows, logits) for the T-1 rows."""
    H, eps = cfg["hidden"], cfg["eps"]
    T = len(tokens) - 1
    _, embed = load_bf16(entries, "model.embed_tokens.weight")
    w = layer_weights(cfg, entries, cfg["num_layers"])
    if progress:
        print("draft layer", file=sys.stderr, flush=True)
    x = []
    for q in range(T):
        tok = tokens[q + 1]
        en = rmsnorm2(embed[tok * H:(tok + 1) * H], w["enorm"], eps)
        hn = rmsnorm2(h_last[q], w["hnorm"], eps)
        x.append(gemv(en + hn, w["eh_proj"]))
    h, _ = layer_forward(x, w, cfg, T)
    hn = [rmsnorm2(row, w["shared_head_norm"], eps) for row in h]
    _, lm = load_bf16(entries, "lm_head.weight")
    lm_rows = as_rows(lm, cfg["vocab"], H)
    logits = [[f32(dot(r, lr)) for lr in lm_rows] for r in hn]
    return hn, logits


def topk_row(values, k):
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    return order, [values[i] for i in order]


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-glm4-reference-dump", "version": VERSION, "backend": "pure",
              "config": cfg_summary, "tensors": {}}
    payload = bytearray()
    for name, (dtype, shape, blob) in tensors.items():
        header["tensors"][name] = {"dtype": dtype, "shape": list(shape), "offset": len(payload), "nbytes": len(blob)}
        payload.extend(blob)
    hb = json.dumps(header, indent=1, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(hb)))
        f.write(hb)
        f.write(bytes(payload))


def gen_pure(args):
    cfg = text_config(args.checkpoint_dir)
    entries = read_safetensors_index(args.checkpoint_dir)
    rng = make_rng(args.seed)
    tokens = [int(rng() * cfg["vocab"]) % cfg["vocab"] for _ in range(args.tokens)]
    if args.tokens > 9:
        tokens[9] = cfg["eos"]
    layer_states, h_rows, logits, routes, _ = reference_forward(cfg, entries, tokens, progress=True)
    topk = 8
    top_ids, top_vals = [], []
    for row in logits:
        ids, vals = topk_row(row, topk)
        top_ids.extend(ids)
        top_vals.extend(vals)
    T, H = args.tokens, cfg["hidden"]
    L, K = cfg["num_layers"], cfg["top_k"]
    Lm = L - cfg["first_dense"]
    flat_routes = [e for layer in routes for row in layer for e in row]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], krd.bf16_to_bytes([v for row in h_rows for v in row])),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, H], krd.bf16_to_bytes([v for st in layer_states for v in st])),
        "route_ids": ("I32", [Lm, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "num_layers": L, "tokens": T, "top_k": topk, "moe_layers": Lm}
    if args.mtp and cfg["mtp"] == 1 and T >= 2:
        mh_rows, mlogits = mtp_forward(cfg, entries, tokens, h_rows, progress=True)
        m_ids, m_vals = [], []
        for row in mlogits:
            ids, vals = topk_row(row, topk)
            m_ids.extend(ids)
            m_vals.extend(vals)
        tensors["mtp_final_hidden"] = ("BF16", [T - 1, H], krd.bf16_to_bytes([v for row in mh_rows for v in row]))
        tensors["mtp_topk_ids"] = ("I32", [T - 1, topk], struct.pack("<%di" % ((T - 1) * topk), *m_ids))
        tensors["mtp_topk_logits"] = ("F32", [T - 1, topk], struct.pack("<%df" % ((T - 1) * topk), *m_vals))
        cfg_summary["mtp_rows"] = T - 1
    write_dump(args.out, cfg_summary, tensors)
    print("wrote %s (%d tokens, %d layers)" % (args.out, T, L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-pure")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=int, default=72)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True,
                   help="also dump the draft block's rows over the prompt (default on)")
    g.set_defaults(func=gen_pure)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
