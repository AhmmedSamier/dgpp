#!/usr/bin/env python3
"""MiMo-V2.6-Flash full-model reference dump (docs/mimo_v26_flash_plan.md
G3, 2026-09-22).

Computes the MiMoV2ForCausalLM forward — the embedding, the pre-norm
decoder layers (the fused fp8 qkv projection in the checkpoint's pre-sharded
chunk layout, the half-split partial RoPE over the 64-wide rotary slice, the
value scale, the sliding-window GQA attention with its per-head sink on the
SWA layers / global attention on the others; the fp8 dense MLP on layer 0,
the sigmoid-routed MXFP4 MoE without a shared expert after), the final norm,
the lm head — and the MTP draft block's rows over the same prompt, in numpy
doubles with bf16 rounding at the engine's boundaries, from the SAME
checkpoint the engine runs, and writes the comparable outputs (tokens, every
layer's residual, the final read, per-token top-k logits, the routing
decisions, the draft's rows) as a DGPPMIMO dump read by
tests/cuda/mimo_forward_test.cpp.

The wiring is pinned to the release's modeling_mimo_v2.py (transformers 5.3
remote code): MiMoV2DecoderLayer, MiMoV2Attention (rotate_half over the
first int(192 x 0.334) = 64 dims, `value_states * attention_value_scale`
before the cache, the sink as one more softmax column dropped after the
normalization, the window [pos - W + 1, pos]), MiMoV2MoEGate (noaux_tc
sigmoid, e_score_correction_bias on the selection key, the picked scores
normalized, routed_scaling_factor 1) and vLLM's mimo_v2_mtp.py for the
draft; the rounding points are the engine's (src/models/mimo/
attn_reference.cpp, src/models/glm/moe_reference.cpp, kernels/glm_norm.cu).

fp8 weights (128 x 128 blocks, weight_scale_inv): the engine's dequant
bridge is w = bf16(e4m3(code) x scale) — one fp32 product, one bf16
rounding — read with bf16 activations and fp32 accumulation, one rounding of
the output. MXFP4 experts: w = e2m1(code) x 2^(e8m0 - 127), exact.

Usage:
  mimo_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S] [--teacher STATES]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_reference_dump import E2M1_TABLE, bf16, e4m3_decode, e8m0_decode  # noqa: E402
from glm_reference_dump import read_safetensors_index  # noqa: E402
from kda_reference_dump import make_rng  # noqa: E402

MAGIC = b"DGPPMIMO"
VERSION = 1

DK, DV, ROTARY = 192, 128, 64
ATTN_TILE = 32  # kernels/mimo_attn.hpp's kMimoAttnTile: the prefill's chain runs one split of 32-token tiles


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def bf16_to_bytes(values):
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def load_np(entries, name):
    """BF16 / F32 as float64 (exact), F8_E4M3 / U8 as uint8 codes."""
    path, offset, nbytes, dtype, shape = entries[name]
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    if dtype == "BF16":
        w = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return w.view(np.float32).astype(np.float64).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).astype(np.float64).reshape(shape)
    if dtype in ("F8_E4M3", "U8", "I8"):
        return np.frombuffer(buf, dtype=np.uint8).reshape(shape)
    raise ValueError("unsupported dtype %s for %s" % (dtype, name))


def load_fp8(entries, base):
    """An fp8 pair on the 128 x 128 grid -> the engine's dequant bridge:
    bf16(e4m3 x scale) as an exact [N, K] float64 matrix."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".weight_scale_inv")
    n, k = p.shape
    sc = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[:n, :k]
    prod = e4m3_decode(p).astype(np.float32) * sc.astype(np.float32)
    return bf16(prod)


def load_fp8_qkv(entries, base, chunk_rows, chunks):
    """The fused projection: its scale grid is tiled per chunk (ceil(chunk_rows
    / 128) scale rows each), so chunk c's rows [c * chunk_rows, +chunk_rows)
    read scale rows [c * sr, +sr) — local row x's block is x // 128."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".weight_scale_inv")
    n, k = p.shape
    sr = (chunk_rows + 127) // 128
    if n != chunk_rows * chunks or s.shape[0] != sr * chunks:
        raise ValueError("fused qkv geometry disagrees for %s" % base)
    out = np.empty((n, k), dtype=np.float64)
    for c in range(chunks):
        sc = np.repeat(np.repeat(s[c * sr:(c + 1) * sr], 128, axis=0), 128, axis=1)[:chunk_rows, :k]
        codes = p[c * chunk_rows:(c + 1) * chunk_rows]
        out[c * chunk_rows:(c + 1) * chunk_rows] = bf16(e4m3_decode(codes).astype(np.float32) * sc.astype(np.float32))
    return out


def load_mxfp4(entries, base):
    """An MXFP4 pair (`.weight` U8 pairs, `.weight_scale` e8m0 per 32) -> the exact [N, K] floats."""
    p = load_np(entries, base + ".weight")
    s = load_np(entries, base + ".weight_scale")
    n, half = p.shape
    codes = np.empty((n, half * 2), dtype=np.int64)
    codes[:, 0::2] = p & 0xF
    codes[:, 1::2] = p >> 4
    vals = E2M1_TABLE[codes]
    sc = np.repeat(e8m0_decode(s), 32, axis=1)[:, : half * 2]
    return vals * sc


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        tc = json.load(f)
    eos = tc["eos_token_id"]
    hd = tc["head_dim"]
    layers = tc["num_hidden_layers"]
    return {
        "hidden": tc["hidden_size"], "vocab": tc["vocab_size"], "num_layers": layers,
        "swa": [int(x) for x in tc["hybrid_layer_pattern"]], "moe": [int(x) for x in tc["moe_layer_freq"]],
        "eps": tc.get("layernorm_epsilon", tc.get("rms_norm_eps", 1e-6)),
        "eos": eos[0] if isinstance(eos, list) else eos,
        "heads": tc["num_attention_heads"], "kv_heads": tc["num_key_value_heads"],
        "swa_kv_heads": tc.get("swa_num_key_value_heads", tc["num_key_value_heads"]),
        "head_dim": hd, "v_head_dim": tc.get("v_head_dim", hd),
        "rotary": int(hd * tc.get("partial_rotary_factor", 1.0)),
        "theta": float(tc.get("rope_theta", 1e7)), "swa_theta": float(tc.get("swa_rope_theta", tc.get("rope_theta", 1e7))),
        "window": int(tc.get("sliding_window", 0)),
        "swa_sink": bool(tc.get("add_swa_attention_sink_bias", False)),
        "full_sink": bool(tc.get("add_full_attention_sink_bias", False)),
        "value_scale": float(tc.get("attention_value_scale", 1.0)),
        "dense_inter": tc["intermediate_size"], "moe_inter": tc["moe_intermediate_size"],
        "experts": tc["n_routed_experts"], "top_k": tc["num_experts_per_tok"],
        "norm_topk": tc.get("norm_topk_prob", True),
        "scaling": float(tc.get("routed_scaling_factor") or 1.0),
        "mtp": tc.get("num_nextn_predict_layers", 0),
    }


def load_engine_states(path):
    """The engine's layer residuals (DGPP_MIMO_DUMP_ENGINE_STATES: int32 L,
    T, H, then bf16 rows per layer, then the post-final-norm hidden): the
    teacher-forced mode feeds each reference layer the engine's own input, so
    a routing near-tie flip cannot cascade through the stack — the strict
    gate's dump."""
    with open(path, "rb") as f:
        L, T, H = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(L * T * H * 2), dtype=np.uint16).reshape(L, T, H)
        fin = np.frombuffer(f.read(T * H * 2), dtype=np.uint16).reshape(T, H)
    to_f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return [to_f(raw[l]) for l in range(L)], to_f(fin)


def layer_prefix(cfg, layer):
    return "model.mtp.layers.0." if layer == cfg["num_layers"] else "model.layers.%d." % layer


def layer_kind(cfg, layer):
    """(swa, moe) of a main layer or the draft (SWA, dense)."""
    if layer == cfg["num_layers"]:
        return True, False
    return cfg["swa"][layer] == 1, cfg["moe"][layer] == 1


def layer_weights(cfg, entries, layer):
    p = layer_prefix(cfg, layer)
    H = cfg["hidden"]
    draft = layer == cfg["num_layers"]
    swa, moe = layer_kind(cfg, layer)
    w = {"moe": moe, "swa": swa, "draft": draft}
    w["input_norm"] = load_np(entries, p + "input_layernorm.weight")
    w["post_norm"] = load_np(entries, p + ("pre_mlp_layernorm.weight" if draft else "post_attention_layernorm.weight"))
    a = p + "self_attn."
    chunks = cfg["kv_heads"]
    kv = cfg["swa_kv_heads"] if swa else cfg["kv_heads"]
    qpc, kvpc = cfg["heads"] // chunks, kv // chunks
    chunk_rows = qpc * DK + kvpc * (DK + DV)
    w["qkv"] = load_fp8_qkv(entries, a + "qkv_proj", chunk_rows, chunks)
    w["chunks"], w["qpc"], w["kvpc"], w["chunk_rows"], w["kv"] = chunks, qpc, kvpc, chunk_rows, kv
    w["o"] = load_np(entries, a + "o_proj.weight")  # [H, heads * DV]
    sink = (cfg["swa_sink"] if swa else cfg["full_sink"])
    w["sink"] = load_np(entries, a + "attention_sink_bias") if sink else None
    m = p + "mlp."
    if moe:
        E = cfg["experts"]
        w["router"] = load_np(entries, m + "gate.weight")
        w["router_bias"] = load_np(entries, m + "gate.e_score_correction_bias")
        w["experts"] = [(load_mxfp4(entries, m + "experts.%d.gate_proj" % e), load_mxfp4(entries, m + "experts.%d.up_proj" % e),
                         load_mxfp4(entries, m + "experts.%d.down_proj" % e)) for e in range(E)]
    else:
        w["dense"] = (load_fp8(entries, m + "gate_proj"), load_fp8(entries, m + "up_proj"), load_fp8(entries, m + "down_proj"))
    if draft:
        w["enorm"] = load_np(entries, p + "enorm.weight")
        w["hnorm"] = load_np(entries, p + "hnorm.weight")
        w["eh_proj"] = load_np(entries, p + "eh_proj.weight")  # [H, 2H]
        w["final_norm"] = load_np(entries, p + "final_layernorm.weight")
    return w


# ---------------------------------------------------------------------------
# modules (double interior, bf16 at the engine's boundaries)
# ---------------------------------------------------------------------------

def rmsnorm2(x, w, eps):
    """The two-rounding RMSNorm over the last axis: u = bf16(x * rstd), y = bf16(w * u)."""
    x = np.asarray(x, dtype=np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    return bf16(w * bf16(x * rstd))


def gemm_bf16(x, w):
    """bf16(x @ w^T): a bf16 Linear, one rounding (fp32 accumulate in the engine)."""
    return bf16(np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64).T)


def rope_inv(theta):
    R = ROTARY
    base = np.float32(theta)
    return np.array([np.float32(1.0) / np.float32(base ** np.float32(2 * i / R)) for i in range(R // 2)],
                    dtype=np.float32).astype(np.float64)


def rope_cos_sin(pos, inv):
    """cos / sin of fp32 pos x inv_freq, rounded to bf16 (the reference's cache dtype)."""
    ang = (np.float32(pos) * inv.astype(np.float32)).astype(np.float64)
    return bf16(np.cos(ang)), bf16(np.sin(ang))


def rope_heads(x, cos, sin):
    """x [heads, DK] (bf16 values): rotate_half over the first 64 dims with the
    reference's bf16 ops — pairs (i, i + 32)."""
    half = ROTARY // 2
    x1, x2 = x[:, :half], x[:, half:ROTARY]
    r1 = bf16(bf16(x1 * cos) + bf16(-x2 * sin))
    r2 = bf16(bf16(x2 * cos) + bf16(x1 * sin))
    return np.concatenate([r1, r2, x[:, ROTARY:]], axis=1)


def attention_forward(x_rows, w, cfg, T):
    """Rows t = 0 .. T-1 at positions t; row t attends [max(0, t - W + 1), t]
    on a sliding-window layer, [0, t] on a global one (the paged cache holds
    exactly these rows). The kernel's online chain over 32-token tiles from
    the first visible row: each tile's probabilities rounded to bf16 against
    the running max, the sums rescaled (attn_reference.cpp's tile form), the
    sink as split 0's opening (max, sum) = (sink, 1), the denominator
    unrounded, out = bf16(c / l)."""
    lh, kv = cfg["heads"], w["kv"]
    hpk = lh // kv
    chunks, qpc, kvpc, cr = w["chunks"], w["qpc"], w["kvpc"], w["chunk_rows"]
    inv = rope_inv(cfg["swa_theta"] if w["swa"] else cfg["theta"])
    scale = float(np.float32(DK ** -0.5))
    vscale = np.float32(cfg["value_scale"])
    window = cfg["window"] if w["swa"] else 0
    dots = np.asarray(x_rows, dtype=np.float64) @ w["qkv"].T  # [T, chunks * chunk_rows]

    q = np.empty((T, lh, DK))
    k = np.empty((T, kv, DK))
    v = np.empty((T, kv, DV))
    for c in range(chunks):
        base = c * cr
        q[:, c * qpc:(c + 1) * qpc, :] = bf16(dots[:, base:base + qpc * DK]).reshape(T, qpc, DK)
        kb = base + qpc * DK
        k[:, c * kvpc:(c + 1) * kvpc, :] = bf16(dots[:, kb:kb + kvpc * DK]).reshape(T, kvpc, DK)
        vb = kb + kvpc * DK
        vv = bf16(dots[:, vb:vb + kvpc * DV]).reshape(T, kvpc, DV)
        if vscale != np.float32(1.0):
            vv = bf16((vv.astype(np.float32) * vscale).astype(np.float64))
        v[:, c * kvpc:(c + 1) * kvpc, :] = vv
    for t in range(T):
        cos, sin = rope_cos_sin(t, inv)
        q[t] = rope_heads(q[t], cos, sin)
        k[t] = rope_heads(k[t], cos, sin)
    sink = w["sink"]
    out = np.empty((T, lh * DV))
    for t in range(T):
        lo = max(0, t - window + 1) if window > 0 else 0
        n = t - lo + 1
        for h in range(lh):
            kvh = h // hpk
            s = (k[lo:t + 1, kvh, :] @ q[t, h, :]) * scale  # [n]
            m, l = (float(sink[h]), 1.0) if sink is not None else (-np.inf, 0.0)
            c = np.zeros(DV)
            for t0 in range(0, n, ATTN_TILE):
                t1 = min(n, t0 + ATTN_TILE)
                m_new = max(m, float(np.max(s[t0:t1])))
                rescale = np.exp(m - m_new)
                l = l * rescale + float(np.sum(np.exp(s[t0:t1] - m_new)))
                c *= rescale
                p = bf16(np.exp(s[t0:t1] - m_new))
                c += p @ v[lo + t0:lo + t1, kvh, :]
                m = m_new
            out[t, h * DV:(h + 1) * DV] = bf16(c / l)
    return gemm_bf16(out, w["o"])


def swiglu(g, u):
    """act = bf16(bf16(silu(g)) * u) — no clamps (MiMoV2MLP)."""
    return bf16(bf16(g * sigmoid(g)) * u)


def dense_forward(x, dense):
    wg, wu, wd = dense
    g = gemm_bf16(x, wg)
    u = gemm_bf16(x, wu)
    return gemm_bf16(swiglu(g, u), wd)


NEAR_TIES = []  # every routed row's margin between the K-th and (K+1)-th biased score


def moe_forward(x, w, cfg):
    """The router (sigmoid scores, the correction bias on the selection key,
    ties to the lower id, the picked scores normalized and scaled) and the
    engine's chain: unrounded partial dots, the experts in ascending id
    order, one rounding. x: one row."""
    E, K = cfg["experts"], cfg["top_k"]
    logits = w["router"] @ x
    scores = sigmoid(logits)
    biased = scores + w["router_bias"]
    order = sorted(range(E), key=lambda e: (-biased[e], e))
    sel = sorted(order[:K])
    NEAR_TIES.append(biased[order[K - 1]] - biased[order[K]])
    denom = (float(np.sum(scores[sel])) + 1e-20) if cfg["norm_topk"] else 1.0
    weights = [(scores[e] / denom) * cfg["scaling"] for e in sel]
    acc = np.zeros(cfg["hidden"])
    for e, we in zip(sel, weights):
        wg, wu, wd = w["experts"][e]
        g = bf16(wg @ x)
        u = bf16(wu @ x)
        act = swiglu(g, u)
        y = wd @ act
        acc += we * y
    return bf16(acc), sel


def layer_forward(h_rows, w, cfg, T):
    """h += attn(input_norm(h)); h += mlp(post_norm(h)) — bf16 residual adds."""
    eps = cfg["eps"]
    x_rows = rmsnorm2(h_rows, w["input_norm"], eps)
    y = attention_forward(x_rows, w, cfg, T)
    h_rows = bf16(h_rows + y)
    x_rows = rmsnorm2(h_rows, w["post_norm"], eps)
    routes = []
    if w["moe"]:
        y = np.empty_like(h_rows)
        for t in range(T):
            yy, sel = moe_forward(x_rows[t], w, cfg)
            y[t] = yy
            routes.append(sel)
    else:
        y = dense_forward(x_rows, w["dense"])
    h_rows = bf16(h_rows + y)
    return h_rows, routes


def reference_forward(cfg, entries, tokens, progress=False, teacher=None):
    """teacher: (layer states, final hidden) of the engine — every layer past
    the first takes the engine's residual after the layer before it, the
    final norm the engine's last state, so each layer is judged alone."""
    T = len(tokens)
    embed = load_np(entries, "model.embed_tokens.weight")
    h = embed[np.asarray(tokens)]
    layer_states, routes = [], []
    for layer in range(cfg["num_layers"]):
        w = layer_weights(cfg, entries, layer)
        if progress:
            print("layer %d (%s, %s)" % (layer, "swa" if w["swa"] else "global", "moe" if w["moe"] else "dense"),
                  file=sys.stderr, flush=True)
        NEAR_TIES.clear()
        if teacher is not None and layer > 0:
            h = teacher[0][layer - 1]
        h, route = layer_forward(h, w, cfg, T)
        layer_states.append(h.copy())
        if w["moe"]:
            routes.append(route)
            if progress:
                tight = sorted((m, t) for t, m in enumerate(NEAR_TIES))[:3]
                print("  routing margins (smallest):" + "".join(" t%d %.3g" % (t, m) for m, t in tight),
                      file=sys.stderr, flush=True)
    if teacher is not None:
        h = teacher[0][cfg["num_layers"] - 1]
    final_norm = load_np(entries, "model.norm.weight")
    hn = rmsnorm2(h, final_norm, cfg["eps"])
    lm = load_np(entries, "lm_head.weight")
    logits = f32(hn @ lm.T)
    return layer_states, hn, logits, routes, h


def mtp_forward(cfg, entries, tokens, h_last, progress=False):
    """The draft block over the prompt: row q embeds tokens[q + 1] and takes
    the main stack's OUTPUT hidden at q (after the final norm — vLLM's
    mimo_v2_mtp receives the model's hidden states and applies hnorm) —
    x_q = eh_proj([enorm(e) | hnorm(h_q)]) — then the draft layer (SWA with
    the dense MLP, its own cache, positions q), its final_layernorm and the
    shared lm head. Returns (h_rows, logits) for the T-1 rows."""
    eps = cfg["eps"]
    T = len(tokens) - 1
    embed = load_np(entries, "model.embed_tokens.weight")
    w = layer_weights(cfg, entries, cfg["num_layers"])
    if progress:
        print("draft layer", file=sys.stderr, flush=True)
    en = rmsnorm2(embed[np.asarray(tokens[1:])], w["enorm"], eps)
    hn = rmsnorm2(h_last[:T], w["hnorm"], eps)
    x = gemm_bf16(np.concatenate([en, hn], axis=1), w["eh_proj"])
    h, _ = layer_forward(x, w, cfg, T)
    hn = rmsnorm2(h, w["final_norm"], eps)
    lm = load_np(entries, "lm_head.weight")
    logits = f32(hn @ lm.T)
    return hn, logits


def topk_rows(logits, k):
    ids, vals = [], []
    for row in logits:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:k]
        ids.extend(order)
        vals.extend(float(row[i]) for i in order)
    return ids, vals


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-mimo-reference-dump", "version": VERSION, "backend": "numpy",
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
    teacher = load_engine_states(args.teacher) if args.teacher else None
    if teacher is not None and (len(teacher[0]) != cfg["num_layers"] or teacher[1].shape != (args.tokens, cfg["hidden"])):
        raise ValueError("the engine states' shape is not this dump's")
    layer_states, h_rows, logits, routes, _ = reference_forward(cfg, entries, tokens, progress=True, teacher=teacher)
    topk = 8
    top_ids, top_vals = topk_rows(logits, topk)
    T, H = args.tokens, cfg["hidden"]
    L, K = cfg["num_layers"], cfg["top_k"]
    Lm = sum(cfg["moe"])
    flat_routes = [e for layer in routes for row in layer for e in row]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], bf16_to_bytes(h_rows.reshape(-1))),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, H], bf16_to_bytes(np.stack(layer_states).reshape(-1))),
        "route_ids": ("I32", [Lm, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "num_layers": L, "tokens": T, "top_k": topk, "moe_layers": Lm,
                   "teacher": teacher is not None}
    if args.mtp and cfg["mtp"] >= 1 and T >= 2:
        # The draft over the engine's own post-final-norm hidden under a teacher.
        mh_rows, mlogits = mtp_forward(cfg, entries, tokens, teacher[1] if teacher is not None else h_rows, progress=True)
        m_ids, m_vals = topk_rows(mlogits, topk)
        tensors["mtp_final_hidden"] = ("BF16", [T - 1, H], bf16_to_bytes(mh_rows.reshape(-1)))
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
    g.add_argument("--teacher", help="the engine's layer states (DGPP_MIMO_DUMP_ENGINE_STATES): every layer "
                   "past the first is fed the engine's residual after the layer before it, the draft the "
                   "engine's post-final-norm hidden — the strict gate's teacher-forced dump")
    g.set_defaults(func=gen_pure)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
