#!/usr/bin/env python3
"""Perception goldens for the Nemotron-Transcribe host oracle (milestone 5
of docs/nemotron_transcribe_plan.md).

NeMo itself is not installed on this box (and the disk cannot hold it right
now), so this is a faithful CPU-torch MIRROR of the NeMo sources the
release's semantics were resolved from — cited per stage below — exercised
on a tiny geometry with every structural choice live:
  nemo/collections/asr/parts/preprocessing/features.py
      AudioToMelSpectrogramPreprocessor.forward: dither is TRAINING-ONLY
      (eval stays deterministic with the config's 1e-5); preemph 0.97
      (the release's config omits it, so the module default applies);
      center-padded STFT (pads n_fft/2), magnitude sqrt(sum of squares)
      then pow(mag_power=2.0); fb matmul; log(x + 2^-24)
      (log_zero_guard 'add' default); normalize null, frame_splicing 1.
      normalize_batch(mel, length, 'per_feature') — the ASR branch input
      (asr_normalize_type); the diarization branch consumes the RAW mel
      (diar_normalize_type 'NA').
  nemo/collections/asr/parts/submodules/subsampling.py
      FeatureStacking: (B,C,T) -> zero-pad T to a multiple of 8 ->
      (B, T/8, 8*C) frame-major -> bias-free Linear(8*C, d_model).
  nemo/collections/asr/modules/transformer_encoder.py
      pre_block_norm embed_norm LayerNorm; block = x + attn(norm1(x)),
      x + ffn(norm2(x)); fused w_qkv view(B,T,3,H,d); per-head LayerNorm
      on q AND k BEFORE the rotary WHEN qk_norm (the ASR branch; the
      diar branch has qk_norm False — no q/k norm tensors);
      bidirectional softmax at 1/sqrt(d); BIASED out_proj; FFN
      Linear+bias, GELU (erf), Linear+bias; final_norm.
  nemo/collections/asr/parts/submodules/multi_head_attention.py
      RotaryPositionalEncoding: d_k_rot = int(d * rotary_fraction) lanes
      rotate in pairs (j, j + d_k_rot/2), cos/sin duplicated, the tail
      lanes pass through; inv_freq exponents over d_k_rot. ASR 0.5
      (half), diar 1.0 (full — the release's encoder block omits
      rotary_fraction, TransformerEncoder's default).
  nemo/collections/asr/models/sortformer_diar_models.py +
  nemo/collections/asr/modules/sortformer_modules.py
      the SINGLE-PASS route (eval, T <= chunk_feat_len): frontend_encoder
      = encoder(bypass_pre_encode=False) -> transpose -> encoder_proj
      Linear(tf, d)+bias (kept: fc_d_model 512 != tf_d_model 192; the
      tiny fixture uses 32 != 24); forward_infer: no tf-transformer (the
      release cfg has none), mask_padding (full length: no-op),
      upsample_hidden = subpixel_upsample Conv1d(tf, tf*8, k3, pad 1) on
      the (B, tf, T') plane then (B, T', 8*tf) -> (B, T'*8, tf);
      forward_speaker_logits: relu -> first_hidden_to_hidden+b -> relu ->
      single_hidden_to_spks+b; sigmoid (eval dropout off, output_mask all
      ones); _align_diarization_output_resolution: native_factor 1
      (high_resolution), downsample 8 with lengths = T' * 8 — every
      window fully valid, so the masked mean is a plain 8-frame mean.
  nemo/collections/asr/modules/parallel_expert_encoder.py
      _speaker_features (thresholded: (a > threshold)), diar_norm
      LayerNorm(n_spk), the BUILT-IN sinusoidal kernel (SIN on the even
      columns, cos on odd — _build_sinusoid_position_encoding),
      asr_norm LayerNorm(d_model) plus spk_kernel_scale * infusion (the
      scale hits the infusion only), then the connector Linear+bias.

Each case records the diar branch's preds on the ASR grid, the fusion
rows built from them (the full release route: perceive() with a null
activity), AND the same fusion fed with a fixed random external activity
(the milestone-5a entry point). Re-emit against the real NeMo classes
when the toolkit lands — the file shape stays, only the provenance
changes.

Usage:
  python3 tests/python/nemotron_transcribe_perception_oracle.py --emit \
      tests/data/nemotron_transcribe_perception_oracle.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 20261031
# Tiny geometry mirroring the release's structure (feat_in 128 and the x8
# stacking are the real contract; the widths shrink for a CPU fixture).
CFG = dict(
    mel_features=128, n_fft=512, win=400, hop=160, preemph=0.97, log_eps=2.0**-24,
    # ASR branch: half rope, qk norms.
    d_model=64, heads=4, head_dim=16, layers=2, ff_expansion=4,
    subsampling=8, rope_base=10000.0, rotary_fraction=0.5, qk_norm=True,
    asr_normalize_per_feature=True,
    # Diar branch: full rope, no qk norms, tf head.
    dd_model=32, dheads=2, dhead_dim=16, dlayers=2, dff_expansion=4,
    tf_d_model=24,
    # Fusion.
    n_spk=4, threshold=0.3, spk_kernel_scale=0.7,  # != 1.0 on purpose
    output_dim=96,
)
AUDIOS = [3568, 5120]  # frames (N+112)/160+1 = 24, 33 -> tokens 3, 5 (tail pad live)


def frontend(audio, window, fb, cfg):
    """features.py AudioToMelSpectrogramPreprocessor.forward, eval mode."""
    x = audio  # dither: training-only, skipped
    x = torch.cat((x[0:1], x[1:] - cfg["preemph"] * x[:-1]))  # preemph, unpadded
    x = F.pad(x, (cfg["n_fft"] // 2, cfg["n_fft"] // 2))      # center=True
    frames = x.unfold(0, cfg["win"], cfg["hop"]) * window     # (T, win)
    spec = torch.fft.rfft(frames, n=cfg["n_fft"])             # (T, 257)
    power = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 0.0)  # guard 0, eval
    power = power.pow(2.0)                                     # mag_power
    mel = torch.matmul(fb.to(power.dtype), power.transpose(0, 1))  # (128, T)
    return torch.log(mel + cfg["log_eps"])


def normalize_per_feature(mel):
    """features.normalize_batch(mel, length, 'per_feature'), batch 1, full
    length (reference frame + masking are no-ops there)."""
    x = mel.unsqueeze(0)                              # (1, C, T)
    reference = x[:, :, 0:1]
    centered = torch.where(torch.ones_like(x, dtype=torch.bool), x - reference, 0.0)
    n = torch.tensor(x.shape[-1], dtype=x.dtype)
    x_mean = reference + centered.sum(2, keepdim=True) / n.clamp(min=1)
    diff = x - x_mean
    x_var = torch.where(torch.ones_like(x, dtype=torch.bool), diff * diff, 0.0).sum(2, keepdim=True)
    x_std = torch.sqrt(x_var / (n - 1).clamp(min=1))
    x_std = torch.nan_to_num(x_std)
    x_std = x_std + 1e-5
    return ((x - x_mean) / x_std).squeeze(0)


def stack_pre_encode(mel, proj, cfg):
    """subsampling.py FeatureStacking.forward + bias-free proj (frame-major)."""
    c, t = mel.shape
    sf = cfg["subsampling"]
    x = mel.transpose(0, 1)
    t_new = (t + sf - 1) // sf
    x = F.pad(x, (0, 0, 0, t_new * sf - t))
    x = x.reshape(t_new, c * sf)
    return F.linear(x, proj)


class Block(torch.nn.Module):
    """transformer_encoder.py TransformerBlock: spec picks qk_norm and the
    rope fraction (the ASR and diar branches share everything else)."""

    def __init__(self, spec):
        super().__init__()
        d, h, dd, ff = spec["d_model"], spec["heads"], spec["head_dim"], spec["ff_expansion"] * spec["d_model"]
        self.spec = spec
        self.norm1 = torch.nn.LayerNorm(d)
        self.norm2 = torch.nn.LayerNorm(d)
        self.w_qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.qk_norm = spec["qk_norm"]
        if self.qk_norm:
            self.q_norm = torch.nn.LayerNorm(dd)
            self.k_norm = torch.nn.LayerNorm(dd)
        self.out_proj = torch.nn.Linear(d, d)          # biased
        self.ff1 = torch.nn.Linear(d, ff)
        self.ff2 = torch.nn.Linear(ff, d)

    def forward(self, x):
        b, t, d = x.shape
        h, dd = self.spec["heads"], self.spec["head_dim"]
        y = self.norm1(x)
        qkv = self.w_qkv(y).view(b, t, 3, h, dd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        q, k = rope(q, k, self.spec)
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x + self.out_proj(att.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


def rope(q, k, spec):
    """multi_head_attention.py RotaryPositionalEncoding (q,k: B,H,T,d)."""
    d = spec["head_dim"]
    d_rot = int(d * spec["rotary_fraction"])
    if d_rot == d:  # full rope is the half-rope formula on all lanes
        pass
    half = d_rot // 2
    inv = 1.0 / (spec["rope_base"] ** (torch.arange(0, d_rot, 2, dtype=torch.float32) / d_rot))
    pos = torch.arange(q.shape[2], dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)            # (T, d_rot)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    def rot(t):
        head, tail = t[..., :d_rot], t[..., d_rot:]
        h1, h2 = head[..., :half], head[..., half:]
        r = torch.cat((-h2, h1), dim=-1)
        return torch.cat((head * cos + r * sin, tail), dim=-1)
    return rot(q), rot(k)


def asr_spec(cfg):
    return dict(d_model=cfg["d_model"], heads=cfg["heads"], head_dim=cfg["head_dim"],
                ff_expansion=cfg["ff_expansion"], rope_base=cfg["rope_base"],
                rotary_fraction=cfg["rotary_fraction"], qk_norm=cfg["qk_norm"])


def diar_spec(cfg):
    return dict(d_model=cfg["dd_model"], heads=cfg["dheads"], head_dim=cfg["dhead_dim"],
                ff_expansion=cfg["dff_expansion"], rope_base=cfg["rope_base"],
                rotary_fraction=1.0, qk_norm=False)


def spk_kernel(n_spk, d_model):
    """parallel_expert_encoder.py _build_sinusoid_position_encoding."""
    position = torch.arange(n_spk, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * -(math.log(10000.0) / d_model))
    enc = torch.zeros(n_spk, d_model, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(position * div_term)
    enc[:, 1::2] = torch.cos(position * div_term)
    return enc


def init_norms(blk):
    with torch.no_grad():
        for nm, p in blk.named_parameters():
            if p.dim() >= 2 and "norm" not in nm:
                p.normal_(0.0, 0.25)
            elif nm.endswith("norm.weight"):
                p.normal_(1.0, 0.1)
            elif nm.endswith("norm.bias"):
                p.normal_(0.0, 0.1)
            elif p.dim() == 1:  # linear biases
                p.normal_(0.0, 0.1)


def build():
    cfg = CFG
    torch.manual_seed(SEED)
    d, C = cfg["d_model"], cfg["mel_features"]
    dd, tf, S, sf = cfg["dd_model"], cfg["tf_d_model"], cfg["n_spk"], cfg["subsampling"]
    ff = cfg["ff_expansion"] * d
    dff = cfg["dff_expansion"] * dd

    window = torch.rand(cfg["win"]) * 0.9 + 0.1
    fb = torch.rand(C, cfg["n_fft"] // 2 + 1) * 0.02
    proj_w = torch.randn(d, sf * C) * 0.15
    embed_w, embed_b = torch.randn(d) * 0.1 + 1, torch.randn(d) * 0.1
    layers = [Block(asr_spec(cfg)) for _ in range(cfg["layers"])]
    for blk in layers:
        init_norms(blk)
    final_w, final_b = torch.randn(d) * 0.1 + 1, torch.randn(d) * 0.1
    asrw, asrb = torch.randn(d) * 0.1 + 1, torch.randn(d) * 0.1
    diaw, diab = torch.randn(S) * 0.1 + 1, torch.randn(S) * 0.1
    conn_w = torch.randn(cfg["output_dim"], d) * 0.2
    conn_b = torch.randn(cfg["output_dim"]) * 0.1

    # The diar branch: encoder (full rope, no qk norms) + the mask head.
    dproj = torch.randn(dd, sf * C) * 0.15
    demb_w, demb_b = torch.randn(dd) * 0.1 + 1, torch.randn(dd) * 0.1
    dlayers = [Block(diar_spec(cfg)) for _ in range(cfg["dlayers"])]
    for blk in dlayers:
        init_norms(blk)
    dfinal_w, dfinal_b = torch.randn(dd) * 0.1 + 1, torch.randn(dd) * 0.1
    epw, epb = torch.randn(tf, dd) * 0.2, torch.randn(tf) * 0.1
    sup_w = torch.randn(tf * sf, tf, 3) * 0.2     # Conv1d(tf, tf*sf, k3)
    sup_b = torch.randn(tf * sf) * 0.1
    fhw, fhb = torch.randn(tf, tf) * 0.2, torch.randn(tf) * 0.1
    shw, shb = torch.randn(S, tf) * 0.2, torch.randn(S) * 0.1

    def t(x):
        return x.detach().cpu().to(torch.float32).flatten().tolist()

    def w(name, tensor):
        return {"dtype": "f32", "shape": list(tensor.shape), "data": t(tensor)}

    weights = {
        "perception.preprocessor.featurizer.window": w("win", window),
        "perception.preprocessor.featurizer.fb": w("fb", fb.reshape(1, C, -1)),
        "perception.encoder.asr_encoder.pre_encode.proj.weight": w("pe", proj_w),
        "perception.encoder.asr_encoder.embed_norm.weight": w("enw", embed_w),
        "perception.encoder.asr_encoder.embed_norm.bias": w("enb", embed_b),
        "perception.encoder.asr_encoder.final_norm.weight": w("fnw", final_w),
        "perception.encoder.asr_encoder.final_norm.bias": w("fnb", final_b),
        "perception.encoder.asr_norm.weight": w("arw", asrw),
        "perception.encoder.asr_norm.bias": w("arb", asrb),
        "perception.encoder.diar_norm.weight": w("diw", diaw),
        "perception.encoder.diar_norm.bias": w("dib", diab),
        "perception.proj.weight": w("cw", conn_w),
        "perception.proj.bias": w("cb", conn_b),
        "perception.encoder.diarization_model.encoder.pre_encode.proj.weight": w("dpe", dproj),
        "perception.encoder.diarization_model.encoder.embed_norm.weight": w("denw", demb_w),
        "perception.encoder.diarization_model.encoder.embed_norm.bias": w("denb", demb_b),
        "perception.encoder.diarization_model.encoder.final_norm.weight": w("dfnw", dfinal_w),
        "perception.encoder.diarization_model.encoder.final_norm.bias": w("dfnb", dfinal_b),
        "perception.encoder.diarization_model.sortformer_modules.encoder_proj.weight": w("epw", epw),
        "perception.encoder.diarization_model.sortformer_modules.encoder_proj.bias": w("epb", epb),
        "perception.encoder.diarization_model.sortformer_modules.subpixel_upsample.weight": w("supw", sup_w),
        "perception.encoder.diarization_model.sortformer_modules.subpixel_upsample.bias": w("supb", sup_b),
        "perception.encoder.diarization_model.sortformer_modules.first_hidden_to_hidden.weight": w("fhw", fhw),
        "perception.encoder.diarization_model.sortformer_modules.first_hidden_to_hidden.bias": w("fhb", fhb),
        "perception.encoder.diarization_model.sortformer_modules.single_hidden_to_spks.weight": w("shw", shw),
        "perception.encoder.diarization_model.sortformer_modules.single_hidden_to_spks.bias": w("shb", shb),
    }
    for li, blk in enumerate(layers):
        pre = f"perception.encoder.asr_encoder.layers.{li}."
        weights[pre + "norm1.weight"] = w("n1w", blk.norm1.weight)
        weights[pre + "norm1.bias"] = w("n1b", blk.norm1.bias)
        weights[pre + "norm2.weight"] = w("n2w", blk.norm2.weight)
        weights[pre + "norm2.bias"] = w("n2b", blk.norm2.bias)
        weights[pre + "attn.w_qkv.weight"] = w("qkv", blk.w_qkv.weight)
        weights[pre + "attn.q_norm.weight"] = w("qnw", blk.q_norm.weight)
        weights[pre + "attn.q_norm.bias"] = w("qnb", blk.q_norm.bias)
        weights[pre + "attn.k_norm.weight"] = w("knw", blk.k_norm.weight)
        weights[pre + "attn.k_norm.bias"] = w("knb", blk.k_norm.bias)
        weights[pre + "attn.out_proj.weight"] = w("opw", blk.out_proj.weight)
        weights[pre + "attn.out_proj.bias"] = w("opb", blk.out_proj.bias)
        weights[pre + "ffn.net.0.weight"] = w("f1w", blk.ff1.weight)
        weights[pre + "ffn.net.0.bias"] = w("f1b", blk.ff1.bias)
        weights[pre + "ffn.net.3.weight"] = w("f2w", blk.ff2.weight)
        weights[pre + "ffn.net.3.bias"] = w("f2b", blk.ff2.bias)
    for li, blk in enumerate(dlayers):
        pre = f"perception.encoder.diarization_model.encoder.layers.{li}."
        weights[pre + "norm1.weight"] = w("dn1w", blk.norm1.weight)
        weights[pre + "norm1.bias"] = w("dn1b", blk.norm1.bias)
        weights[pre + "norm2.weight"] = w("dn2w", blk.norm2.weight)
        weights[pre + "norm2.bias"] = w("dn2b", blk.norm2.bias)
        weights[pre + "attn.w_qkv.weight"] = w("dqkv", blk.w_qkv.weight)
        weights[pre + "attn.out_proj.weight"] = w("dopw", blk.out_proj.weight)
        weights[pre + "attn.out_proj.bias"] = w("dopb", blk.out_proj.bias)
        weights[pre + "ffn.net.0.weight"] = w("df1w", blk.ff1.weight)
        weights[pre + "ffn.net.0.bias"] = w("df1b", blk.ff1.bias)
        weights[pre + "ffn.net.3.weight"] = w("df2w", blk.ff2.weight)
        weights[pre + "ffn.net.3.bias"] = w("df2b", blk.ff2.bias)

    kernel = spk_kernel(S, d)

    def fuse(states, activity):
        thr = (activity > cfg["threshold"]).to(torch.float32)
        dn = F.layer_norm(thr, (S,), diaw, diab)
        fused = F.layer_norm(states, (d,), asrw, asrb) + cfg["spk_kernel_scale"] * (dn @ kernel)
        return F.linear(fused, conn_w, conn_b)

    cases = []
    with torch.no_grad():
        for n_samples in AUDIOS:
            audio = torch.randn(n_samples) * 0.1
            mel = frontend(audio, window, fb, cfg)                 # (128, T) RAW
            # --- diar branch (raw mel) ---
            dx = stack_pre_encode(mel, dproj, cfg)
            dx = F.layer_norm(dx, (dd,), demb_w, demb_b)
            for blk in dlayers:
                dx = blk(dx.unsqueeze(0)).squeeze(0)
            dx = F.layer_norm(dx, (dd,), dfinal_w, dfinal_b)
            Tt = dx.shape[0]
            h = F.linear(dx, epw, epb)                              # (Tt, tf)
            conv = F.conv1d(h.T.unsqueeze(0), sup_w, sup_b, padding=1)  # (1, tf*8, Tt)
            up = conv.squeeze(0).T.reshape(Tt, sf, tf).reshape(Tt * sf, tf)
            s = F.relu(up)
            s = F.relu(F.linear(s, fhw, fhb))
            logits = F.linear(s, shw, shb)
            preds = torch.sigmoid(logits).reshape(Tt, sf, S).mean(1)  # downsample, all valid
            # --- ASR branch (normalized mel) ---
            mel_n = normalize_per_feature(mel)
            x = stack_pre_encode(mel_n, proj_w, cfg)
            x = F.layer_norm(x, (d,), embed_w, embed_b)
            for blk in layers:
                x = blk(x.unsqueeze(0)).squeeze(0)
            states = F.layer_norm(x, (d,), final_w, final_b)
            # --- fusions ---
            rows_pred = fuse(states, preds)
            activity = torch.rand(Tt, S)
            rows_ext = fuse(states, activity)
            cases.append({
                "n_samples": n_samples,
                "audio": t(audio),
                "mel": t(mel), "mel_frames": list(mel.shape),
                "mel_norm": t(mel_n),
                "states": t(states), "states_shape": list(states.shape),
                "diar_preds": t(preds), "diar_shape": list(preds.shape),
                "activity": t(activity),
                "rows": t(rows_ext), "rows_shape": list(rows_ext.shape),
                "rows_pred": t(rows_pred), "rows_pred_shape": list(rows_pred.shape),
            })

    return {
        "generated_by": "tests/python/nemotron_transcribe_perception_oracle.py",
        "note": "CPU-torch mirror of the NeMo perception chain (milestone 5, "
                "docs/nemotron_transcribe_plan.md), transcribed from "
                "nemo features.py / transformer_encoder.py / subsampling.py / "
                "multi_head_attention.py / parallel_expert_encoder.py / "
                "sortformer_modules.py / sortformer_diar_models.py at main "
                "(fetched 2026-10-31): the single-pass diar route (no tf "
                "transformer, full-length masks), ASR per_feature "
                "normalization, diar on the raw mel.",
        "seed": SEED,
        "config": CFG,
        "weights": weights,
        "cases": cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    args = ap.parse_args()
    doc = build()
    out = Path(args.emit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(doc['weights'])} tensors, {len(doc['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
