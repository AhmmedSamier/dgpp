#!/usr/bin/env python3
"""Full GLM-5.3 (GlmMoeDsaForCausalLM) reference dump (docs/glm53_plan.md G5,
2026-09-12).

Computes the forward — the embedding, the pre-norm decoder layers (MLA
with the decoupled interleaved RoPE, the DSA indexer's per-token top-k on
the "full" layers and the inherited selection on the "shared" ones, the
BF16 dense MLP on the first layers, the sigmoid-routed MoE with its
packed-int experts after), the final norm, the lm head — and the MTP draft
block's rows over the same prompt, in pure python doubles with bf16
rounding at the engine's boundaries, from the SAME checkpoint the engine
runs, and writes the comparable outputs (tokens, every layer's residual,
the final read, per-token top-k logits, the routing decisions, every
indexed layer's selection with its boundary margin, the draft's rows) as
a DGPPGDSD dump read by tests/cuda/glm_dsa_forward_test.cpp.

The wiring is pinned to transformers' modeling_glm_moe_dsa.py
(GlmMoeDsaAttention, GlmMoeDsaIndexer with apply_rotary_pos_emb_interleave,
GlmMoeDsaTopkRouter) with the engine's rounding points
(src/models/dsa_reference.cpp, src/models/glm/moe_reference.cpp): the
indexer's q and k go through the engine's Hadamard-128 + fp8 path, the
selection is the pinned composite-key top-k, the attention is the
absorbed MLA with bf16 probabilities, the rope table is the engine's
(dsa_rope_table_host). Packed-int weights (compressed-tensors
pack-quantized, plan D2) are code x bf16 scale exactly.

Usage:
  glm_dsa_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S]
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
from dsa_reference_dump import (fwht128, layernorm_row, quant_row_fp8, rmsnorm_row,  # noqa: E402
                                select_pools, fp8_from_bits)
from glm_reference_dump import (read_safetensors_index, rmsnorm2, moe_reference,  # noqa: E402
                                dense_mlp_rows, gemm_row)

MAGIC = b"DGPPGDSD"
VERSION = 1

bf16_round = krd.bf16_round
bf16_from_bits = krd.bf16_from_bits
make_rng = krd.make_rng


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def dot(a, b) -> float:
    return sum(map(mul, a, b))


def gemv(x, rows):
    """[bf16(x . row) for row in rows] — a bf16 Linear (one rounding)."""
    return [bf16_round(dot(x, r)) for r in rows]


def as_rows(flat, n, k):
    return [flat[i * k:(i + 1) * k] for i in range(n)]


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

_ITEMSIZE = {"BF16": 2, "F32": 4, "I32": 4, "I64": 8}


def load_raw(entries, name):
    """(dtype, shape, values): BF16/F32 as floats, I32/I64 as ints."""
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
    if dtype_str == "I32":
        return dtype_str, shape, list(struct.unpack("<%dI" % (nbytes // 4), buf))
    return dtype_str, shape, list(struct.unpack("<%dq" % (nbytes // 8), buf))


def load_bf16(entries, name):
    dtype, shape, values = load_raw(entries, name)
    if dtype not in ("BF16", "F32"):
        raise ValueError("expected BF16/F32 for %s" % name)
    return shape, values


def load_packed(entries, module):
    """A pack-quantized triple -> rows of the exact code x scale values
    (the engine's dequant, plan D2): weight_packed I32 [N, K*bits/32]
    (unsigned codes, low nibble/byte first), weight_scale BF16 [N, K/64],
    weight_shape I64 [2]."""
    _, (n, words_per_row), words = load_raw(entries, module + ".weight_packed")
    _, (n2, groups), scales = load_raw(entries, module + ".weight_scale")
    _, _, shape = load_raw(entries, module + ".weight_shape")
    k = shape[1]
    bits = words_per_row * 32 // k
    if bits not in (4, 8) or n != shape[0] or n2 != n or groups * 64 != k:
        raise ValueError("packed shapes disagree for %s" % module)
    per = 32 // bits
    mask = (1 << bits) - 1
    off = 1 << (bits - 1)
    rows = []
    for r in range(n):
        row = [0.0] * k
        wbase, sbase = r * words_per_row, r * groups
        for c in range(k):
            code = ((words[wbase + c // per] >> (bits * (c % per))) & mask) - off
            row[c] = code * scales[sbase + c // 64]
        rows.append(row)
    return rows


def load_linear(entries, module, n, k):
    """A Linear's rows: the packed triple where present, else the bf16 weight."""
    if module + ".weight_packed" in entries:
        rows = load_packed(entries, module)
        if len(rows) != n or len(rows[0]) != k:
            raise ValueError("%s: packed [%d, %d], expected [%d, %d]" % (module, len(rows), len(rows[0]), n, k))
        return rows
    shape, values = load_bf16(entries, module + ".weight")
    if list(shape) != [n, k]:
        raise ValueError("%s: shape %s, expected [%d, %d]" % (module, shape, n, k))
    return as_rows(values, n, k)


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        tc = json.load(f)
    eos = tc["eos_token_id"]
    L = tc["num_hidden_layers"]
    # The indexer schedule: an explicit list, else the freq/offset rule
    # (src/models/glm_dsa/config.cpp's parse_indexer_schedule).
    freq, offset = tc.get("index_topk_freq", 1), tc.get("index_skip_topk_offset", 0)
    if "indexer_types" in tc:
        full = [t == "full" for t in tc["indexer_types"]]
    else:
        full = [max(l - offset + 1, 0) % freq == 0 for l in range(L)]
    rp = tc.get("rope_parameters", {})
    return {
        "hidden": tc["hidden_size"], "vocab": tc["vocab_size"], "num_layers": L,
        "first_dense": tc["first_k_dense_replace"], "eps": tc.get("rms_norm_eps", 1e-5),
        "eos": eos[0] if isinstance(eos, list) else eos,
        "heads": tc["num_attention_heads"], "q_lora": tc["q_lora_rank"], "kv_lora": tc["kv_lora_rank"],
        "nope": tc["qk_nope_head_dim"], "rope": tc["qk_rope_head_dim"], "v": tc["v_head_dim"],
        "index_heads": tc["index_n_heads"], "index_dim": tc["index_head_dim"], "index_topk": tc["index_topk"],
        "indexer_full": full, "theta": float(rp.get("rope_theta", tc.get("rope_theta", 8e6))),
        "dense_inter": tc["intermediate_size"], "moe_inter": tc["moe_intermediate_size"],
        "experts": tc["n_routed_experts"], "top_k": tc["num_experts_per_tok"],
        "n_shared": tc.get("n_shared_experts", 1), "norm_topk": tc.get("norm_topk_prob", True),
        "scaling": float(tc.get("routed_scaling_factor", 1.0)),
        "mtp": tc.get("num_nextn_predict_layers", 0),
    }


LAYER = "model.layers.%d."


def layer_weights(cfg, entries, layer):
    """A main-stack layer, or the draft layer (layer == num_layers)."""
    p = LAYER % layer
    H = cfg["hidden"]
    draft = layer == cfg["num_layers"]
    lh, nope, rope, v = cfg["heads"], cfg["nope"], cfg["rope"], cfg["v"]
    ql, kvl = cfg["q_lora"], cfg["kv_lora"]
    w = {"moe": layer >= cfg["first_dense"], "draft": draft,
         "indexer": draft or cfg["indexer_full"][layer]}
    w["input_norm"] = load_bf16(entries, p + "input_layernorm.weight")[1]
    w["post_norm"] = load_bf16(entries, p + "post_attention_layernorm.weight")[1]
    a = p + "self_attn."
    w["q_a"] = load_linear(entries, a + "q_a_proj", ql, H)
    w["q_aln"] = load_bf16(entries, a + "q_a_layernorm.weight")[1]
    w["q_b"] = load_linear(entries, a + "q_b_proj", lh * (nope + rope), ql)
    w["kv_a"] = load_linear(entries, a + "kv_a_proj_with_mqa", kvl + rope, H)
    w["kv_aln"] = load_bf16(entries, a + "kv_a_layernorm.weight")[1]
    w["kv_b"] = load_linear(entries, a + "kv_b_proj", lh * (nope + v), kvl)
    w["o"] = load_linear(entries, a + "o_proj", H, lh * v)
    if w["indexer"]:
        i = a + "indexer."
        w["wq_b"] = as_rows(load_bf16(entries, i + "wq_b.weight")[1], cfg["index_heads"] * cfg["index_dim"], ql)
        w["wk"] = as_rows(load_bf16(entries, i + "wk.weight")[1], cfg["index_dim"], H)
        w["wp"] = as_rows(load_bf16(entries, i + "weights_proj.weight")[1], cfg["index_heads"], H)
        w["k_norm_w"] = load_bf16(entries, i + "k_norm.weight")[1]
        w["k_norm_b"] = load_bf16(entries, i + "k_norm.bias")[1]
    m = p + "mlp."
    if w["moe"]:
        E, I = cfg["experts"], cfg["moe_inter"]
        w["router"] = load_bf16(entries, m + "gate.weight")[1]
        w["router_bias"] = load_bf16(entries, m + "gate.e_score_correction_bias")[1]
        experts = []
        for e in range(E):
            for name, n, k in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                experts.append([x for row in load_linear(entries, m + "experts.%d.%s" % (e, name), n, k) for x in row])
        w["experts"] = experts
        S = I * cfg["n_shared"]
        w["shared"] = [[x for row in load_linear(entries, m + "shared_experts." + name, n, k) for x in row]
                       for name, n, k in (("gate_proj", S, H), ("up_proj", S, H), ("down_proj", H, S))]
    else:
        I = cfg["dense_inter"]
        w["dense"] = [[x for row in load_linear(entries, m + name, n, k) for x in row]
                      for name, n, k in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I))]
    if draft:
        w["enorm"] = load_bf16(entries, p + "enorm.weight")[1]
        w["hnorm"] = load_bf16(entries, p + "hnorm.weight")[1]
        w["eh_proj"] = as_rows(load_bf16(entries, p + "eh_proj.weight")[1], H, 2 * H)
        w["shared_head_norm"] = load_bf16(entries, p + "shared_head.norm.weight")[1]
    return w


# ---------------------------------------------------------------------------
# RoPE (kernels/dsa.hpp: dsa_rope_table_host / dsa_rope_interleave)
# ---------------------------------------------------------------------------

def rope_table(theta, rope, positions):
    """bf16 (cos, sin) per position and pair: inv_freq = f32(1 / theta^(2i/rope))
    (the power in double), angle = f32(p * inv_freq), trig in double -> f32
    -> bf16."""
    half = rope // 2
    base = f32(theta)
    inv = [f32(1.0 / math.pow(base, f32(f32(2 * i) / f32(rope)))) for i in range(half)]
    cos, sin = [], []
    for p in range(positions):
        c, s = [], []
        for i in range(half):
            ang = f32(f32(p) * inv[i])
            c.append(bf16_round(f32(math.cos(ang))))
            s.append(bf16_round(f32(math.sin(ang))))
        cos.append(c)
        sin.append(s)
    return cos, sin


def rope_rotate(x, table, pos):
    """In place on x[0:rope]: pair (x0, x1) -> (bf16(bf16(x0 c) - bf16(x1 s)),
    bf16(bf16(x1 c) + bf16(x0 s))) — three roundings, the pair kept in place."""
    cos, sin = table
    c, s = cos[pos], sin[pos]
    for i in range(len(c)):
        x0, x1 = x[2 * i], x[2 * i + 1]
        t1, t2 = bf16_round(x0 * c[i]), bf16_round(x1 * s[i])
        u1, u2 = bf16_round(x1 * c[i]), bf16_round(x0 * s[i])
        x[2 * i] = bf16_round(t1 - t2)
        x[2 * i + 1] = bf16_round(u1 + u2)


# ---------------------------------------------------------------------------
# The DSA attention layer (src/models/dsa_reference.cpp with rope 64,
# kpool 1, relu; the selection inherited on the shared layers)
# ---------------------------------------------------------------------------

class DsaState:
    """One layer's caches in logical token order."""

    def __init__(self):
        self.latent = []   # [t] -> [kv_lora + rope] bf16 values
        self.index_k = []  # [t] -> fp8 codes [128] (indexed layers)
        self.index_s = []  # [t] -> scale


def indexer_rows(cfg, w, h_rows, q_c_rows, table, pos0):
    """Per token: (q8 [heads][128], w_folded [heads]) and the key cache
    entry (codes, scale) — the engine's indexer_query_inputs and the kpool-1
    compress (Hadamard + fp8 of the roped, normed key)."""
    heads, dim, rope = cfg["index_heads"], cfg["index_dim"], cfg["rope"]
    logit_scale = f32(math.pow(dim, -0.5) * math.pow(heads, -0.5))
    q8s, wfs, keys = [], [], []
    for t, (h, q_c) in enumerate(zip(h_rows, q_c_rows)):
        pos = pos0 + t
        q_idx = gemv(q_c, w["wq_b"])
        q8, wf = [], []
        for hh in range(heads):
            x = q_idx[hh * dim:(hh + 1) * dim]
            rope_rotate(x, table, pos)
            fwht128(x)
            x = [bf16_round(v) for v in x]
            codes, scale = quant_row_fp8(x)
            q8.append(codes)
            wraw = f32(dot(h, w["wp"][hh]))          # fp32 weights, no bf16 rounding
            wf.append(f32(f32(wraw * scale) * logit_scale))
        k = layernorm_row(gemv(h, w["wk"]), w["k_norm_w"], w["k_norm_b"], 1e-6)
        rope_rotate(k, table, pos)
        kx = list(k)
        fwht128(kx)
        kx = [bf16_round(v) for v in kx]
        keys.append(quant_row_fp8(kx))
        q8s.append(q8)
        wfs.append(wf)
    return q8s, wfs, keys


def select_tokens(cfg, q8, wf, state, pos):
    """The pinned selection for a query at `pos` over entries [0, pos]:
    logits from the fp8 dots with the relu per head; returns (tokens,
    margin) with margin the boundary gap (the select_k-th vs the next
    logit) relative to the row's largest logit magnitude, or -1 in the
    dense regime."""
    select_k = cfg["index_topk"]
    visible = pos + 1
    if visible <= select_k:
        return list(range(visible)), -1.0
    heads, dim = cfg["index_heads"], cfg["index_dim"]
    q = [[fp8_from_bits(c) for c in q8[hh]] for hh in range(heads)]
    logits = []
    for j in range(visible):
        kj = [fp8_from_bits(c) for c in state.index_k[j]]
        ks = state.index_s[j]
        total = 0.0
        for hh in range(heads):
            d = dot(q[hh], kj)
            if d < 0.0:
                d = 0.0
            total += wf[hh] * ks * d
        logits.append(f32(total))
    order = sorted(range(visible), key=lambda j: (-logits[j], j))
    gap = logits[order[select_k - 1]] - logits[order[select_k]]
    margin = abs(gap) / (max(abs(v) for v in logits) + 1e-30)
    return select_pools(logits, select_k), margin


def attention_rows(cfg, w, q_rows, state, sels):
    """Absorbed MLA per token over its selected tokens: q~ = [W_uk^T q_nope |
    q_rot] (bf16), scores over [latent | rope key] scaled by (nope+rope)^-0.5,
    one-shot softmax with bf16 probabilities, c in double, out_h = bf16(W_uv c)."""
    lh, nope, rope, v, kvl = cfg["heads"], cfg["nope"], cfg["rope"], cfg["v"], cfg["kv_lora"]
    head_rows = nope + v
    scale = 1.0 / math.sqrt(float(nope + rope))
    out_rows = []
    for q, sel in zip(q_rows, sels):
        o = []
        for hh in range(lh):
            qh = q[hh * (nope + rope):(hh + 1) * (nope + rope)]
            wuk = w["kv_b"][hh * head_rows:hh * head_rows + nope]      # [nope][kvl]
            wuv = w["kv_b"][hh * head_rows + nope:(hh + 1) * head_rows]  # [v][kvl]
            qt = [bf16_round(sum(qh[d] * wuk[d][c] for d in range(nope))) for c in range(kvl)]
            qt.extend(qh[nope:])
            s = [dot(qt, state.latent[t]) * scale for t in sel]
            m = max(s)
            p = [math.exp(x - m) for x in s]
            denom = sum(p)
            c = [0.0] * kvl
            for pt, t in zip(p, sel):
                pb = bf16_round(pt / denom)
                lat = state.latent[t]
                for cc in range(kvl):
                    c[cc] += pb * lat[cc]
            o.extend(bf16_round(dot(wuv[d], c)) for d in range(v))
        out_rows.append(gemv(o, w["o"]))
    return out_rows


def attention_forward(cfg, w, x_rows, state, table, pos0, inherited):
    """One layer's attention over the chunk rows [pos0, pos0 + T) — the
    caches updated, the selections made (indexed layer) or `inherited`
    (shared layer). Returns (out_rows, selections, margins)."""
    lh, nope, rope, v = cfg["heads"], cfg["nope"], cfg["rope"], cfg["v"]
    ql, kvl, eps = cfg["q_lora"], cfg["kv_lora"], cfg["eps"]
    q_rows, q_c_rows = [], []
    for t, x in enumerate(x_rows):
        pos = pos0 + t
        qa = gemv(x, w["q_a"])
        kva = gemv(x, w["kv_a"])
        q_c = rmsnorm_row(qa, w["q_aln"], eps)
        lat = rmsnorm_row(kva[:kvl], w["kv_aln"], eps)
        k_rot = list(kva[kvl:])
        rope_rotate(k_rot, table, pos)
        assert len(state.latent) == pos
        state.latent.append(lat + k_rot)
        q = gemv(q_c, w["q_b"])
        for hh in range(lh):
            head = q[hh * (nope + rope):(hh + 1) * (nope + rope)]
            rot = head[nope:]
            rope_rotate(rot, table, pos)
            q[hh * (nope + rope) + nope:(hh + 1) * (nope + rope)] = rot
        q_rows.append(q)
        q_c_rows.append(q_c)
    if w["indexer"]:
        q8s, wfs, keys = indexer_rows(cfg, w, x_rows, q_c_rows, table, pos0)
        sels, margins = [], []
        for t in range(len(x_rows)):
            state.index_k.append(keys[t][0])
            state.index_s.append(keys[t][1])
            sel, margin = select_tokens(cfg, q8s[t], wfs[t], state, pos0 + t)
            sels.append(sel)
            margins.append(margin)
    else:
        sels, margins = inherited, [-1.0] * len(x_rows)
    return attention_rows(cfg, w, q_rows, state, sels), sels, margins


def layer_forward(cfg, w, h_rows, state, table, pos0, inherited):
    """h += attn(input_norm(h)); h += mlp(post_norm(h)) — bf16 residual adds."""
    eps = cfg["eps"]
    x_rows = [rmsnorm2(h, w["input_norm"], eps) for h in h_rows]
    y, sels, margins = attention_forward(cfg, w, x_rows, state, table, pos0, inherited)
    h_rows = [[bf16_round(a + b) for a, b in zip(h, yy)] for h, yy in zip(h_rows, y)]
    x_rows = [rmsnorm2(h, w["post_norm"], eps) for h in h_rows]
    H = cfg["hidden"]
    route_margins = []
    if w["moe"]:
        moe = (w["router"], w["router_bias"], w["experts"], w["shared"])
        E, K = cfg["experts"], cfg["top_k"]
        y, ids, _, biased = moe_reference(x_rows, moe, H, cfg["moe_inter"], K, E,
                                          math.inf, cfg["norm_topk"], cfg["scaling"])
        routes = [ids[t * K:(t + 1) * K] for t in range(len(x_rows))]
        # The router's boundary margin per token: the K-th biased score
        # minus the next (sigmoid units) — a flip inside it is noise.
        for t in range(len(x_rows)):
            b = sorted(biased[t * E:(t + 1) * E], reverse=True)
            # Selecting every expert has no excluded candidate at the boundary.
            route_margins.append(b[K - 1] - b[K] if K < E else math.inf)
    else:
        y = dense_mlp_rows(x_rows, w["dense"], H, cfg["dense_inter"], math.inf)
        routes = []
    h_rows = [[bf16_round(a + b) for a, b in zip(h, yy)] for h, yy in zip(h_rows, y)]
    return h_rows, routes, sels, margins, route_margins


def reference_forward(cfg, entries, tokens, table, progress=False):
    H = cfg["hidden"]
    _, embed = load_bf16(entries, "model.embed_tokens.weight")
    h = [list(embed[tok * H:(tok + 1) * H]) for tok in tokens]
    layer_states, routes, route_margins, selections, margins = [], [], [], [], []
    inherited = None
    for layer in range(cfg["num_layers"]):
        w = layer_weights(cfg, entries, layer)
        if progress:
            print("layer %d (%s%s)" % (layer, "moe" if w["moe"] else "dense",
                                       ", indexer" if w["indexer"] else ", shared selection"),
                  file=sys.stderr, flush=True)
        state = DsaState()
        h, route, sels, marg, rmarg = layer_forward(cfg, w, h, state, table, 0, inherited)
        layer_states.append([v for row in h for v in row])
        if w["moe"]:
            routes.append(route)
            route_margins.append(rmarg)
            if progress:
                tight = sorted((m, t) for t, m in enumerate(rmarg))[:3]
                print("  routing margins (smallest):" + "".join(" t%d %.3g" % (t, m) for m, t in tight),
                      file=sys.stderr, flush=True)
        if w["indexer"]:
            inherited = sels
            selections.append(sels)
            margins.append(marg)
            if progress:
                tight = sorted((m, t) for t, m in enumerate(marg) if m >= 0)[:3]
                print("  selection margins (smallest):" + "".join(" t%d %.3g" % (t, m) for m, t in tight),
                      file=sys.stderr, flush=True)
    _, final_norm = load_bf16(entries, "model.norm.weight")
    hn = [rmsnorm2(row, final_norm, cfg["eps"]) for row in h]
    _, lm = load_bf16(entries, "lm_head.weight")
    lm_rows = as_rows(lm, cfg["vocab"], H)
    logits = [[f32(dot(x, r)) for r in lm_rows] for x in hn]
    return layer_states, hn, logits, routes, route_margins, selections, margins


def mtp_forward(cfg, entries, tokens, h_last, table, progress=False):
    """The draft block over the prompt: row q embeds tokens[q + 1] and takes
    the main stack's OUTPUT hidden at q (after the final norm), x_q =
    eh_proj([enorm(e) | hnorm(h_q)]), then the draft layer (its own caches
    and indexer, positions q), shared_head.norm and the shared lm head."""
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
    h, _, _, _, _ = layer_forward(cfg, w, x, DsaState(), table, 0, None)
    hn = [rmsnorm2(row, w["shared_head_norm"], eps) for row in h]
    _, lm = load_bf16(entries, "lm_head.weight")
    lm_rows = as_rows(lm, cfg["vocab"], H)
    logits = [[f32(dot(r, lr)) for lr in lm_rows] for r in hn]
    return hn, logits


def topk_row(values, k):
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    return order, [values[i] for i in order]


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-glm-dsa-reference-dump", "version": VERSION, "backend": "pure",
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
    table = rope_table(cfg["theta"], cfg["rope"], args.tokens + 8)
    layer_states, h_rows, logits, routes, route_margins, selections, margins = reference_forward(
        cfg, entries, tokens, table, progress=True)
    topk = 8
    top_ids, top_vals = [], []
    for row in logits:
        ids, vals = topk_row(row, topk)
        top_ids.extend(ids)
        top_vals.extend(vals)
    T, H = args.tokens, cfg["hidden"]
    L, K = cfg["num_layers"], cfg["top_k"]
    Lm = L - cfg["first_dense"]
    Li = len(selections)
    S = cfg["index_topk"]  # max_selected at kpool 1
    flat_routes = [e for layer in routes for row in layer for e in row]
    flat_sel = []
    for layer in selections:
        for row in layer:
            flat_sel.extend(row + [-1] * (S - len(row)))
    flat_margins = [m for layer in margins for m in layer]
    flat_rmargins = [m for layer in route_margins for m in layer]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], krd.bf16_to_bytes([v for row in h_rows for v in row])),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, H], krd.bf16_to_bytes([v for st in layer_states for v in st])),
        "route_ids": ("I32", [Lm, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
        "route_margins": ("F32", [Lm, T], struct.pack("<%df" % len(flat_rmargins), *flat_rmargins)),
        "dsa_selections": ("I32", [Li, T, S], struct.pack("<%di" % len(flat_sel), *flat_sel)),
        "dsa_margins": ("F32", [Li, T], struct.pack("<%df" % len(flat_margins), *flat_margins)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "num_layers": L, "tokens": T, "top_k": topk,
                   "moe_layers": Lm, "index_layers": Li, "max_selected": S}
    if args.mtp and cfg["mtp"] == 1 and T >= 2:
        mh_rows, mlogits = mtp_forward(cfg, entries, tokens, h_rows, table, progress=True)
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
    print("wrote %s (%d tokens, %d layers, %d indexed)" % (args.out, T, L, Li))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-pure")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=int, default=32)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True,
                   help="also dump the draft block's rows over the prompt (default on)")
    g.set_defaults(fn=gen_pure)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
