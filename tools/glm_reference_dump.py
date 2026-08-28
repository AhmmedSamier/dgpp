#!/usr/bin/env python3
"""GLM full-model reference-dump harness (M4 chunk 6, DESIGN §7.5).

Computes the full text-model forward — embedding -> mHC-wired decoder
layers (KDA or DSA attention, dense MLP or MoE) -> stream mean -> final
norm -> lm head — in pure python doubles with bf16 rounding at the engine's
boundaries, from the SAME checkpoint the engine runs, and writes the
comparable outputs (input tokens, final hidden state, per-token top-k
logits, per-layer routing decisions) as a DGPPGLMD dump.

The wiring is pinned to the transformers Glm5NextTextDecoderLayer forward:
per site, attn_hc(streams) -> collapsed; ln1(collapsed); sublayer;
streams' = bf16(bf16(post*h) + bf16(sum_j comb[j,i]*streams[j])). The
sublayer math is the SAME reference the M2/M3 dump tools pin (imported
below), and the mHC/MoE/norm/head modules are reimplemented here from the
pinned module semantics (glm_mhc.hpp §7.3, glm_moe.hpp §7.4) in the double
oracle style of src/models/glm_*_reference.cpp.

Backends:
  pure   (stdlib only) reads the checkpoint with stdlib safetensors
         parsing. This is the CI oracle: it runs anywhere python runs and
         consumes the synthetic mini-checkpoint written by
         glm_forward_test --write-fixture.

  torch  (requires torch; run where the real checkpoint lives) is chunk 6b:
         the same wiring over real weights, streaming one layer at a time.

File format ("DGPPGLMD"): 8-byte magic, u32 version=1, u32 header length,
JSON header, payload — same container as the KDA/DSA dumps. Read by
src/models/glm_dump.cpp.

Usage:
  glm_reference_dump.py selftest
  glm_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE
        [--tokens T]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_reference_dump as krd  # noqa: E402
import dsa_reference_dump as drd  # noqa: E402
from dsa_reference_dump import fp8_from_bits  # noqa: E402

MAGIC = b"DGPPGLMD"
VERSION = 1

bf16_round = krd.bf16_round
bf16_bits = krd.bf16_bits
bf16_from_bits = krd.bf16_from_bits
make_rng = krd.make_rng


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# safetensors reading (stdlib; mirrors kda_reference_dump's index walker)
# ---------------------------------------------------------------------------

def read_safetensors_index(model_dir):
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        entries = {}
        for shard in sorted(set(index["weight_map"].values())):
            _parse_shard_header(os.path.join(model_dir, shard), entries)
        return entries
    if os.path.exists(single):
        entries = {}
        _parse_shard_header(single, entries)
        return entries
    raise FileNotFoundError("no safetensors index or file in %s" % model_dir)


def _parse_shard_header(path, entries):
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


_ITEMSIZE = {"BF16": 2, "F32": 4, "F8_E4M3": 1}


def load_tensor(entries, name):
    """Returns (dtype, shape, flat python values). BF16/F32 come back as
    python floats (bf16 already rounded to representable values); fp8
    payloads come back as integer codes."""
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


def dequant(entries, name):
    """E4M3 payload + block scales -> flat bf16-rounded weight values (the
    engine's decode x scale, one bf16 round — scale_gemm's weight policy)."""
    _, (rows, cols), payload = load_tensor(entries, name)
    _, _, scales = load_tensor(entries, name + "_scale_inv")
    sc = (cols + 127) // 128
    out = [0.0] * (rows * cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        out[i] = bf16_round(fp8_from_bits(payload[i]) *
                            scales[(r // 128) * sc + c // 128])
    return out


# ---------------------------------------------------------------------------
# Wiring math (double interior, bf16 at the engine's boundaries)
# ---------------------------------------------------------------------------

def mhc_reference(streams, fn, base, scale, hc_eps, norm_eps, iters, tokens,
                  n, hidden):
    """Mirror of glm_mhc_ref_compute: unweighted RMSNorm over the flattened
    streams (fp32 semantics, double interior), 24 fp32-projection logits,
    pre/post/comb, Sinkhorn, collapse. post/comb are rounded to bf16 (the
    kernel exports them as bf16); pre stays double (consumed by the
    collapse only). Returns (post, comb, collapsed) flat lists."""
    coeffs = (2 + n) * n
    K = n * hidden
    post = [0.0] * (tokens * n)
    comb = [0.0] * (tokens * n * n)
    collapsed = [0.0] * (tokens * hidden)
    for t in range(tokens):
        x = streams[t * K:(t + 1) * K]
        ssq = 0.0
        for v in x:
            ssq += v * v
        inv_rms = 1.0 / math.sqrt(ssq / K + norm_eps)

        logits = []
        for i in range(coeffs):
            row = fn[i * K:(i + 1) * K]
            acc = 0.0
            for d in range(K):
                acc += x[d] * inv_rms * row[d]
            logits.append(acc)

        pre = [sigmoid(logits[i] * scale[0] + base[i]) + hc_eps
               for i in range(n)]
        for i in range(n):
            post[t * n + i] = 2.0 * sigmoid(logits[n + i] * scale[1] +
                                            base[n + i])

        c = [0.0] * (n * n)
        for row in range(n):
            vals = [logits[2 * n + row * n + col] * scale[2] +
                    base[2 * n + row * n + col] for col in range(n)]
            m = max(vals)
            ex = [math.exp(v - m) for v in vals]
            s = sum(ex)
            for col in range(n):
                c[row * n + col] = ex[col] / s
        for i in range(n * n):
            c[i] += hc_eps

        def col_pass():
            for col in range(n):
                s = 0.0
                for row in range(n):
                    s += c[row * n + col]
                for row in range(n):
                    c[row * n + col] = c[row * n + col] / (s + hc_eps)

        def row_pass():
            for row in range(n):
                s = 0.0
                for col in range(n):
                    s += c[row * n + col]
                for col in range(n):
                    c[row * n + col] = c[row * n + col] / (s + hc_eps)

        col_pass()
        for _ in range(iters - 1):
            row_pass()
            col_pass()
        for i in range(n * n):
            comb[t * n * n + i] = c[i]

        for d in range(hidden):
            acc = 0.0
            for j in range(n):
                acc += pre[j] * x[j * hidden + d]
            collapsed[t * hidden + d] = bf16_round(acc)

    post = [bf16_round(v) for v in post]
    comb = [bf16_round(v) for v in comb]
    return post, comb, collapsed


def mhc_update(post, comb, sub_out, streams, tokens, n, hidden):
    """Mirror of glm_mhc_ref_stream_update:
    streams'[i] = bf16(bf16(post[i]*h) + bf16(sum_j comb[j,i]*streams[j]))."""
    K = n * hidden
    out = [0.0] * (tokens * K)
    for t in range(tokens):
        h = sub_out[t * hidden:(t + 1) * hidden]
        res = streams[t * K:(t + 1) * K]
        for i in range(n):
            pi = post[t * n + i]
            for d in range(hidden):
                t1 = bf16_round(pi * h[d])
                mix = 0.0
                for j in range(n):
                    mix += comb[t * n * n + j * n + i] * res[j * hidden + d]
                t2 = bf16_round(mix)
                out[t * K + i * hidden + d] = bf16_round(t1 + t2)
    return out


def rmsnorm2(x, w, eps):
    """Two-rounding RMSNorm (Glm5NextTextRMSNorm): u = bf16(x*rstd),
    y = bf16(w*u)."""
    var = 0.0
    for v in x:
        var += v * v
    var /= len(x)
    rstd = 1.0 / math.sqrt(var + eps)
    return [bf16_round(w[i] * bf16_round(x[i] * rstd)) for i in range(len(x))]


def swiglu(g, u, limit):
    """Asymmetric-clamp swiglu with the reference's two roundings:
    gate clamps max only, up clamps both sides."""
    if g > limit:
        g = limit
    u = min(max(u, -limit), limit)
    t = bf16_round(g * sigmoid(g))
    return bf16_round(t * u)


def gemm_row(x, w, n_out, n_in):
    """Strict GEMM (one row): double accumulate, one bf16 round."""
    return [bf16_round(sum(x[k] * w[o * n_in + k] for k in range(n_in)))
            for o in range(n_out)]


def expert_mlp(x, wg, wu, wd, hidden, inter, limit):
    g = gemm_row(x, wg, inter, hidden)
    u = gemm_row(x, wu, inter, hidden)
    act = [swiglu(g[i], u[i], limit) for i in range(inter)]
    return gemm_row(act, wd, hidden, inter)


def dense_mlp_rows(x_rows, dense, hidden, inter, limit):
    wg, wu, wd = dense
    return [expert_mlp(x, wg, wu, wd, hidden, inter, limit) for x in x_rows]


def moe_reference(x_rows, moe, hidden, inter, top_k, n_experts, limit,
                  norm_topk, scaling):
    """Mirror of glm_moe_ref_router + glm_moe_ref_forward: biased selection,
    uncorrected weights, per-element normalization, ascending-id
    accumulation, shared expert last."""
    gate, bias, experts, shared = moe
    out_rows, ids_all, weights_all = [], [], []
    for x in x_rows:
        scores = [sigmoid(sum(x[k] * gate[e * hidden + k]
                              for k in range(hidden)))
                  for e in range(n_experts)]
        biased = [scores[e] + bias[e] for e in range(n_experts)]
        taken = [False] * n_experts
        sel, wsel = [], []
        for _ in range(top_k):
            best, bv = -1, -math.inf
            for e in range(n_experts):
                if not taken[e] and biased[e] > bv:
                    bv, best = biased[e], e
            taken[best] = True
            sel.append(best)
            wsel.append(scores[best])
        pairs = sorted(zip(sel, wsel))  # ascending expert id
        if norm_topk:
            denom = sum(w for _, w in pairs) + 1e-20
            weights = [(w / denom) * scaling for _, w in pairs]
        else:
            weights = [w * scaling for _, w in pairs]
        ids = [e for e, _ in pairs]

        out = [0.0] * hidden
        for idx in range(top_k):
            e = ids[idx]
            y = expert_mlp(x, experts[e * 3], experts[e * 3 + 1],
                           experts[e * 3 + 2], hidden, inter, limit)
            we = weights[idx]
            for d in range(hidden):
                contrib = bf16_round(we * y[d])
                out[d] = bf16_round(out[d] + contrib)
        y = expert_mlp(x, shared[0], shared[1], shared[2], hidden, inter,
                       limit)
        for d in range(hidden):
            out[d] = bf16_round(out[d] + y[d])

        out_rows.append(out)
        ids_all.extend(ids)
        weights_all.extend(float(w) for w in weights)
    return out_rows, ids_all, weights_all


# ---------------------------------------------------------------------------
# Checkpoint -> wiring weights
# ---------------------------------------------------------------------------

def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        full = json.load(f)
    return full.get("text_config", full)


def layer_weights(cfg, entries, layer):
    p = "model.language_model.layers.%d." % layer
    lw = {}
    lw["ln1"] = load_tensor(entries, p + "input_layernorm.weight")[2]
    lw["ln2"] = load_tensor(entries, p + "post_attention_layernorm.weight")[2]

    for site in ("attn", "ffn"):
        lw["mhc_" + site] = (
            load_tensor(entries, p + "hc_%s_fn" % site)[2],
            load_tensor(entries, p + "hc_%s_base" % site)[2],
            load_tensor(entries, p + "hc_%s_scale" % site)[2],
        )

    if cfg["layer_types"][layer] == "linear_attention":
        sp = p + "self_attn."
        head_dim = cfg["linear_attn_config"]["head_dim"]
        proj = cfg["linear_attn_config"]["num_heads"] * head_dim
        heads = cfg["linear_attn_config"]["num_heads"]
        conv_w = 4
        in_proj = []
        for part in ("f_a_proj", "g_a_proj", "q_proj", "k_proj", "v_proj",
                     "b_proj"):
            in_proj += load_tensor(entries, sp + part + ".weight")[2]
        conv = []
        for part in ("q_conv1d", "k_conv1d", "v_conv1d"):
            conv += load_tensor(entries, sp + part + ".weight")[2]
        lw["kda"] = {
            "in_proj": in_proj, "f_b": load_tensor(
                entries, sp + "f_b_proj.weight")[2],
            "g_b": load_tensor(entries, sp + "g_b_proj.weight")[2],
            "conv": conv,
            "a_log": load_tensor(entries, sp + "A_log")[2],
            "dt_bias": load_tensor(entries, sp + "dt_bias")[2],
            "o_norm": load_tensor(entries, sp + "o_norm.weight")[2],
            "o_proj": load_tensor(entries, sp + "o_proj.weight")[2],
        }
        lw["kda_cfg"] = {
            "hidden": cfg["hidden_size"], "heads": heads,
            "head_dim": head_dim, "conv_width": conv_w,
            "lower_bound": cfg["linear_attn_config"]["gate_lower_bound"],
            "conv_state_width": conv_w - 1,
        }
    else:
        sp = p + "self_attn."
        ip = sp + "indexer."
        qkv_a = dequant(entries, sp + "q_a_proj.weight") + \
            dequant(entries, sp + "kv_a_proj_with_mqa.weight")
        ape = load_tensor(entries, ip + "index_kpool_compress_ape")[2]
        lw["dsa"] = {
            "qkv_a": qkv_a,
            "q_aln": load_tensor(entries, sp + "q_a_layernorm.weight")[2],
            "kv_aln": load_tensor(entries, sp + "kv_a_layernorm.weight")[2],
            "q_b": dequant(entries, sp + "q_b_proj.weight"),
            "kv_b": load_tensor(entries, sp + "kv_b_proj.weight")[2],
            "o_proj": dequant(entries, sp + "o_proj.weight"),
            "wq_b": load_tensor(entries, ip + "wq_b.weight")[2],
            "wk": load_tensor(entries, ip + "wk.weight")[2],
            "wp": load_tensor(entries, ip + "weights_proj.weight")[2],
            "gate": load_tensor(entries, ip + "index_kpool_compress_gate")[2],
            "k_norm_w": load_tensor(entries, ip + "k_norm.weight")[2],
            "k_norm_b": load_tensor(entries, ip + "k_norm.bias")[2],
            "ape": ape,
        }
        lw["dsa_cfg"] = {
            "hidden": cfg["hidden_size"],
            "heads": cfg["num_attention_heads"],
            "q_lora": cfg["q_lora_rank"], "kv_lora": cfg["kv_lora_rank"],
            "nope": cfg["qk_nope_head_dim"], "v_dim": cfg["v_head_dim"],
            "idx_heads": cfg["index_n_heads"], "idx_dim": 128,
            "topk": cfg["index_topk"], "kpool": cfg["index_kpool"],
            "rms_eps": cfg["rms_norm_eps"],
        }

    if cfg["mlp_layer_types"][layer] == "dense":
        m = p + "mlp."
        lw["dense"] = (
            dequant(entries, m + "gate_proj.weight"),
            dequant(entries, m + "up_proj.weight"),
            dequant(entries, m + "down_proj.weight"),
        )
    else:
        m = p + "mlp."
        n_experts = cfg["n_routed_experts"]
        experts = []
        for e in range(n_experts):
            ep = m + "experts.%d." % e
            experts.append(dequant(entries, ep + "gate_proj.weight"))
            experts.append(dequant(entries, ep + "up_proj.weight"))
            experts.append(dequant(entries, ep + "down_proj.weight"))
        sp_ = m + "shared_experts."
        lw["moe"] = (
            load_tensor(entries, m + "gate.weight")[2],
            load_tensor(entries, m + "gate.e_score_correction_bias")[2],
            experts,
            (dequant(entries, sp_ + "gate_proj.weight"),
             dequant(entries, sp_ + "up_proj.weight"),
             dequant(entries, sp_ + "down_proj.weight")),
        )
    return lw


# ---------------------------------------------------------------------------
# The full forward
# ---------------------------------------------------------------------------

def reference_forward(cfg, entries, tokens, progress=False):
    hidden = cfg["hidden_size"]
    n = cfg["hc_mult"]
    hc_eps = cfg["hc_eps"]
    iters = cfg["hc_sinkhorn_iters"]
    eps = cfg["rms_norm_eps"]
    limit = cfg["swiglu_limit"]
    T = len(tokens)
    K = n * hidden

    embed = load_tensor(entries, "model.language_model.embed_tokens.weight")[2]
    lm_head = load_tensor(entries, "lm_head.weight")[2]
    final_norm = load_tensor(entries, "model.language_model.norm.weight")[2]

    streams = []
    for t in tokens:
        row = embed[t * hidden:(t + 1) * hidden]
        streams.extend(row)
        streams.extend(row)
        streams.extend(row)
        streams.extend(row)

    routes = []
    for layer in range(cfg["num_hidden_layers"]):
        lw = layer_weights(cfg, entries, layer)
        for site, sublayer in (("attn", "attn"), ("ffn", "mlp")):
            fn, base, scale = lw["mhc_" + site]
            post, comb, collapsed = mhc_reference(
                streams, fn, base, scale, hc_eps, eps, iters, T, n, hidden)
            ln = lw["ln1"] if site == "attn" else lw["ln2"]
            x_rows = [rmsnorm2(collapsed[t * hidden:(t + 1) * hidden], ln,
                               eps) for t in range(T)]
            xflat = [v for row in x_rows for v in row]

            if site == "attn":
                if "kda" in lw:
                    kcfg = dict(lw["kda_cfg"])
                    kcfg["tokens"] = T
                    state = [0.0] * (kcfg["heads"] * kcfg["head_dim"] ** 2)
                    conv = [0.0] * (3 * kcfg["heads"] * kcfg["head_dim"] *
                                    kcfg["conv_state_width"])
                    out, _, _ = krd.reference_layer(
                        lw["kda"], xflat, state, conv, kcfg)
                else:
                    dcfg = dict(lw["dsa_cfg"])
                    dcfg["tokens"] = T
                    out, _, _ = drd.reference_layer(dcfg, lw["dsa"], xflat, T)
            else:
                if "dense" in lw:
                    rows = dense_mlp_rows(
                        x_rows, lw["dense"], hidden,
                        cfg["intermediate_size"], limit)
                    out = [v for row in rows for v in row]
                else:
                    rows, ids, weights = moe_reference(
                        x_rows, lw["moe"], hidden,
                        cfg["moe_intermediate_size"],
                        cfg["num_experts_per_tok"], cfg["n_routed_experts"],
                        limit, cfg["norm_topk_prob"],
                        cfg["routed_scaling_factor"])
                    out = [v for row in rows for v in row]
                    if site == "ffn":
                        routes.append((layer, ids, weights))

            streams = mhc_update(post, comb, out, streams, T, n, hidden)
        if progress:
            print("  layer %d/%d done" % (layer + 1,
                                          cfg["num_hidden_layers"]))

    # Head: unweighted mean over streams, final two-rounding norm, lm head.
    mean = [bf16_round(sum(streams[t * K + s * hidden + h]
                           for s in range(n)) / n)
            for t in range(T) for h in range(hidden)]
    hidden_rows = [rmsnorm2(mean[t * hidden:(t + 1) * hidden], final_norm,
                            eps) for t in range(T)]
    vocab = cfg["vocab_size"]
    logits = [gemm_row(row, lm_head, vocab, hidden) for row in hidden_rows]
    return hidden_rows, logits, routes


def topk_row(values, k):
    order = sorted(range(len(values)), key=lambda c: (-values[c], c))
    return order[:k], [float(values[c]) for c in order[:k]]


# ---------------------------------------------------------------------------
# Dump container (same layout family as the KDA/DSA tools)
# ---------------------------------------------------------------------------

def write_dump(path, meta, cfg_summary, tensors, route_layers):
    header = {
        "format": "dgpp-glm-reference-dump",
        "version": VERSION,
        "model": meta["model"],
        "revision": meta["revision"],
        "backend": meta["backend"],
        "config": cfg_summary,
        "route_layers": route_layers,
        "tensors": {},
    }
    payload = bytearray()
    for name, (dtype, shape, blob) in tensors.items():
        header["tensors"][name] = {
            "dtype": dtype, "shape": list(shape),
            "offset": len(payload), "nbytes": len(blob),
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
                         payload[desc["offset"]:desc["offset"] +
                                 desc["nbytes"]])
    return header, tensors


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def gen_pure(args):
    cfg = text_config(args.checkpoint_dir)
    entries = read_safetensors_index(args.checkpoint_dir)
    rng = make_rng(args.seed)
    tokens = [int(rng() * cfg["vocab_size"]) % cfg["vocab_size"]
              for _ in range(args.tokens)]

    hidden_rows, logits, routes = reference_forward(cfg, entries, tokens,
                                                    progress=True)

    topk = 8
    top_ids, top_vals = [], []
    for row in logits:
        ids, vals = topk_row(row, topk)
        top_ids.extend(ids)
        top_vals.extend(vals)

    route_layers = [{"layer_idx": layer, "top_k": cfg["num_experts_per_tok"],
                     "tokens": args.tokens} for layer, _, _ in routes]
    flat_ids, flat_w = [], []
    for _, ids, weights in routes:
        flat_ids.extend(ids)
        flat_w.extend(weights)

    tensors = {
        "tokens": ("I64", [args.tokens],
                   struct.pack("<%dq" % args.tokens, *tokens)),
        "final_hidden": ("BF16", [args.tokens, cfg["hidden_size"]],
                         krd.bf16_to_bytes([v for row in hidden_rows
                                            for v in row])),
        "topk_ids": ("I32", [args.tokens, topk],
                     struct.pack("<%di" % (args.tokens * topk), *top_ids)),
        "topk_logits": ("F32", [args.tokens, topk],
                        struct.pack("<%df" % (args.tokens * topk), *top_vals)),
        "route_ids": ("I32", [len(flat_ids)],
                      struct.pack("<%di" % len(flat_ids), *flat_ids)),
        "route_weights": ("F32", [len(flat_w)],
                          struct.pack("<%df" % len(flat_w), *flat_w)),
    }
    meta = {
        "model": os.path.basename(os.path.normpath(args.checkpoint_dir)),
        "revision": "pure-%d" % args.seed,
        "backend": "pure",
    }
    cfg_summary = {
        "hidden": cfg["hidden_size"], "vocab": cfg["vocab_size"],
        "num_layers": cfg["num_hidden_layers"], "tokens": args.tokens,
        "top_k": topk,
    }
    write_dump(args.out, meta, cfg_summary, tensors, route_layers)
    print("wrote %s (%d tokens, %d layers, %d routed layers)" %
          (args.out, args.tokens, cfg["num_hidden_layers"], len(routes)))
    return 0


# ---------------------------------------------------------------------------
# Selftest (no torch, no checkpoint)
# ---------------------------------------------------------------------------

def cmd_selftest(args):
    import tempfile

    # mHC on a hand-checkable 4x8 case: all-zero streams -> norm of zero ->
    # logits 0 -> pre = sigmoid(base)+eps etc. Verify against direct math.
    n, H = 4, 8
    streams = [0.0] * (2 * n * H)
    fn = [0.05] * ((2 + n) * n * n * H)
    base = [0.1] * ((2 + n) * n)
    scale = [1.0, 1.0, 1.0]
    post, comb, collapsed = mhc_reference(streams, fn, base, scale, 1e-6,
                                          1e-6, 20, 2, n, H)
    want_post = bf16_round(2.0 * sigmoid(0.1))
    assert all(abs(v - want_post) < 1e-12 for v in post)
    # comb: uniform softmax + eps -> Sinkhorn stays (near-)uniform.
    assert abs(comb[0] - bf16_round(0.25)) < 5e-3
    assert all(v == 0.0 for v in collapsed)

    # Two-rounding norm: pick values where the choreographies differ.
    x = [1.00390625, 0.99609375, 1.0]  # 1 + 1/256, 1 - 1/256
    w = [3.0] * 3
    got = rmsnorm2(x, w, 0.0)
    rstd = 1.0 / math.sqrt(sum(v * v for v in x) / 3.0)
    two = [bf16_round(w[i] * bf16_round(x[i] * rstd)) for i in range(3)]
    one = [bf16_round(w[i] * x[i] * rstd) for i in range(3)]
    assert got == two and two != one, "two-rounding choreography"

    # Router tie rule: equal biased scores -> lower ids.
    class FakeEntries(dict):
        pass
    x_rows = [[0.5, -0.5], [0.5, -0.5]]
    gate = [0.0] * (4 * 2)  # all-zero gate rows: scores all 0.5
    bias = [0.0, 0.0, 0.0, 0.0]
    experts = [[0.01] * (2 * 2)] * 12
    out_rows, ids, weights = moe_reference(
        x_rows, (gate, bias, experts, experts), 2, 2, 2, 4, 10.0, True, 1.0)
    assert ids[:2] == [0, 1] and ids[2:4] == [0, 1], "tie rule"

    # Dump round-trip.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.glmdump")
        write_dump(path, {"model": "m", "revision": "r", "backend": "pure"},
                   {"hidden": 8, "vocab": 16, "num_layers": 1, "tokens": 2,
                    "top_k": 8},
                   {"tokens": ("I64", [2], struct.pack("<2q", 1, 2))}, [])
        header, tensors = read_dump(path)
        assert header["config"]["tokens"] == 2
        assert struct.unpack("<2q", tensors["tokens"][2]) == (1, 2)

    print("glm_reference_dump selftest OK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="module sanity (no torch)")
    st.set_defaults(fn=cmd_selftest)

    gp = sub.add_parser("gen-pure",
                        help="full-stack pure-python reference dump")
    gp.add_argument("--checkpoint-dir", required=True)
    gp.add_argument("--out", required=True)
    gp.add_argument("--tokens", type=int, default=24)
    gp.add_argument("--seed", type=int, default=1234)
    gp.set_defaults(fn=gen_pure)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
