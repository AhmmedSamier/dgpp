#!/usr/bin/env python3
"""Perception goldens for the Nemotron-Transcribe host oracle (milestone 5
of docs/nemotron_transcribe_plan.md).

NeMo itself is not installed on this box (and the disk cannot hold it right
now), so this is a faithful CPU-torch MIRROR of the four NeMo sources the
release's semantics were resolved from — cited per stage below — exercised
on a tiny geometry with every structural choice live:
  nemo/collections/asr/parts/preprocessing/features.py
      AudioToMelSpectrogramPreprocessor.forward: dither is TRAINING-ONLY
      (eval stays deterministic with the config's 1e-5); preemph 0.97
      (the release's config omits it, so the module default applies);
      center-padded STFT (pads n_fft/2), magnitude sqrt(sum of squares)
      then pow(mag_power=2.0); fb matmul; log(x + 2^-24)
      (log_zero_guard 'add' default); normalize null, frame_splicing 1.
  nemo/collections/asr/parts/submodules/subsampling.py
      FeatureStacking: (B,C,T) -> zero-pad T to a multiple of 8 ->
      (B, T/8, 8*C) frame-major -> bias-free Linear(8*C, d_model).
  nemo/collections/asr/modules/transformer_encoder.py
      pre_block_norm embed_norm LayerNorm; block = x + attn(norm1(x)),
      x + ffn(norm2(x)); fused w_qkv view(B,T,3,H,d) -> per-head
      LayerNorm on q AND k BEFORE the rotary; bidirectional softmax at
      1/sqrt(d); BIASED out_proj; FFN Linear+bias, GELU (erf),
      Linear+bias; final_norm.
  nemo/collections/asr/parts/submodules/multi_head_attention.py
      RotaryPositionalEncoding with rotary_fraction 0.5: d_k_rot = d/2
      lanes rotate in pairs (j, j + d_k_rot/2), cos/sin duplicated, the
      tail lanes pass through; inv_freq exponents over d_k_rot.
  nemo/collections/asr/modules/parallel_expert_encoder.py
      _speaker_features (thresholded: (a > threshold)), diar_norm
      LayerNorm(n_spk), the BUILT-IN sinusoidal kernel (SIN on the even
      columns, cos on odd — _build_sinusoid_position_encoding),
      asr_norm LayerNorm(d_model) plus spk_kernel_scale * infusion (the
      scale hits the infusion only), then the connector Linear+bias.

The Sortformer branch itself (its ~380 tensors) is milestone 5b; the
fusion consumes its activity as an input here, so every other stage is
pinned. Re-emit this fixture against the real NeMo classes when the
toolkit lands — the file is the same shape, only the provenance note
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
# Tiny geometry mirroring the release's structure (feat_in 128 and the ×8
# stacking are the real contract; the widths shrink for a CPU fixture).
CFG = dict(
    mel_features=128, n_fft=512, win=400, hop=160, preemph=0.97, log_eps=2.0**-24,
    d_model=64, heads=4, head_dim=16, layers=2, ff_expansion=4,
    subsampling=8, rope_base=10000.0, rotary_fraction=0.5,
    n_spk=4, threshold=0.3, spk_kernel_scale=0.7,  # != 1.0 on purpose
    output_dim=96,
)
AUDIOS = [3568, 5120]  # frames 23 -> 3 tokens; frames 33 -> 5 tokens (tail pad live)


def frontend(audio, window, fb, cfg):
    """features.py AudioToMelSpectrogramPreprocessor.forward, eval mode."""
    n = audio.shape[0]
    x = audio  # dither: training-only, skipped
    # preemph on the unpadded signal (NeMo's masked_fill keeps it to length)
    x = torch.cat((x[0:1], x[1:] - cfg["preemph"] * x[:-1]))
    # center=True: pad n_fft/2 zeros each side
    x = F.pad(x, (cfg["n_fft"] // 2, cfg["n_fft"] // 2))
    frames = x.unfold(0, cfg["win"], cfg["hop"]) * window  # (T, win)
    spec = torch.fft.rfft(frames, n=cfg["n_fft"])          # (T, 257)
    power = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 0.0)  # guard 0
    power = power.pow(2.0)                                           # mag_power
    mel = torch.matmul(fb.to(power.dtype), power.transpose(0, 1))  # (128, T)
    mel = torch.log(mel + cfg["log_eps"])
    return mel  # (128, T), NeMo's channel-major


def stack_pre_encode(mel, proj, cfg):
    """subsampling.py FeatureStacking.forward + bias-free proj."""
    c, t = mel.shape  # (128, T)
    sf = cfg["subsampling"]
    x = mel.transpose(0, 1)  # (T, C)
    t_new = (t + sf - 1) // sf
    x = F.pad(x, (0, 0, 0, t_new * sf - t))  # zero-pad frames
    x = x.reshape(t_new, c * sf)             # frame-major rows
    return F.linear(x, proj)                 # (T/8, d_model)


class Block(torch.nn.Module):
    """transformer_encoder.py TransformerBlock with the rope path, eval."""

    def __init__(self, cfg):
        super().__init__()
        d, h, dd = cfg["d_model"], cfg["heads"], cfg["head_dim"]
        ff = cfg["ff_expansion"] * d
        self.norm1 = torch.nn.LayerNorm(d)
        self.norm2 = torch.nn.LayerNorm(d)
        self.w_qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.q_norm = torch.nn.LayerNorm(dd)
        self.k_norm = torch.nn.LayerNorm(dd)
        self.out_proj = torch.nn.Linear(d, d)  # biased
        self.ff1 = torch.nn.Linear(d, ff)
        self.ff2 = torch.nn.Linear(ff, d)

    def forward(self, x):
        cfg = CFG
        b, t, d = x.shape
        h, dd = cfg["heads"], cfg["head_dim"]
        y = self.norm1(x)
        qkv = self.w_qkv(y).view(b, t, 3, h, dd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        q, k = rope(q, k, cfg)
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x + self.out_proj(att.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


def rope(q, k, cfg):
    """multi_head_attention.py RotaryPositionalEncoding (q,k: B,H,T,d)."""
    d = cfg["head_dim"]
    d_rot = int(d * cfg["rotary_fraction"])
    half = d_rot // 2
    inv = 1.0 / (cfg["rope_base"] ** (torch.arange(0, d_rot, 2, dtype=torch.float32) / d_rot))
    pos = torch.arange(q.shape[2], dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)             # (T, d_rot)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    def rot(t):
        head, tail = t[..., :d_rot], t[..., d_rot:]
        h1, h2 = head[..., :half], head[..., half:]
        r = torch.cat((-h2, h1), dim=-1)
        return torch.cat((head * cos + r * sin, tail), dim=-1)
    return rot(q), rot(k)


def spk_kernel(n_spk, d_model):
    """parallel_expert_encoder.py _build_sinusoid_position_encoding."""
    position = torch.arange(n_spk, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * -(math.log(10000.0) / d_model))
    enc = torch.zeros(n_spk, d_model, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(position * div_term)
    enc[:, 1::2] = torch.cos(position * div_term)
    return enc


def build():
    torch.manual_seed(SEED)
    cfg = CFG
    d, h, dd, ff = cfg["d_model"], cfg["heads"], cfg["head_dim"], cfg["ff_expansion"] * cfg["d_model"]
    window = torch.rand(cfg["win"]) * 0.9 + 0.1
    fb = torch.rand(cfg["mel_features"], cfg["n_fft"] // 2 + 1) * 0.02
    proj_w = torch.randn(d, cfg["subsampling"] * cfg["mel_features"]) * 0.15
    embed_w, embed_b = torch.randn(d) * 0.1 + 1, torch.randn(d) * 0.1
    layers = [Block(cfg) for _ in range(cfg["layers"])]
    for blk in layers:  # informative scales
        with torch.no_grad():
            for p in blk.parameters():
                if p.dim() >= 2:
                    p.normal_(0.0, 0.25)
            # LayerNorm weights: near 1; LN biases near 0.
            for nm, p in blk.named_parameters():
                if "norm" in nm and nm.endswith(".weight"):
                    p.normal_(1.0, 0.1)
                elif "norm" in nm and nm.endswith(".bias"):
                    p.normal_(0.0, 0.1)
    final_w, final_b = torch.randn(d) * 0.1 + 1, torch.randn(d) * 0.1
    asrw, asrb = torch.randn(d) * 0.1 + 1, torch.randn(d) * 0.1
    diaw, diab = torch.randn(cfg["n_spk"]) * 0.1 + 1, torch.randn(cfg["n_spk"]) * 0.1
    conn_w = torch.randn(cfg["output_dim"], d) * 0.2
    conn_b = torch.randn(cfg["output_dim"]) * 0.1

    def t(x):
        return x.detach().cpu().to(torch.float32).flatten().tolist()

    def w(name, tensor):
        return {"dtype": "f32", "shape": list(tensor.shape), "data": t(tensor)}

    weights = {
        "perception.preprocessor.featurizer.window": w("win", window),
        "perception.preprocessor.featurizer.fb": w("fb", fb.reshape(1, *fb.shape)),
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

    cases = []
    with torch.no_grad():
        for n_samples in AUDIOS:
            audio = torch.randn(n_samples) * 0.1
            mel = frontend(audio, window, fb, cfg)                 # (128, T)
            x = stack_pre_encode(mel, proj_w, cfg)                 # (T_tok, d)
            x = F.layer_norm(x, (d,), embed_w, embed_b)
            for blk in layers:
                x = blk(x.unsqueeze(0)).squeeze(0)
            states = F.layer_norm(x, (d,), final_w, final_b)       # (T_tok, d)
            activity = torch.rand(states.shape[0], cfg["n_spk"])   # [0,1]
            thr = (activity > cfg["threshold"]).to(torch.float32)
            dn = F.layer_norm(thr, (cfg["n_spk"],), diaw, diab)
            kernel = spk_kernel(cfg["n_spk"], d)
            fused = F.layer_norm(states, (d,), asrw, asrb) + cfg["spk_kernel_scale"] * (dn @ kernel)
            rows = F.linear(fused, conn_w, conn_b)
            cases.append({
                "n_samples": n_samples,
                "audio": t(audio),
                "mel": t(mel), "mel_frames": list(mel.shape),
                "states": t(states), "states_shape": list(states.shape),
                "activity": t(activity),
                "rows": t(rows), "rows_shape": list(rows.shape),
            })

    return {
        "generated_by": "tests/python/nemotron_transcribe_perception_oracle.py",
        "note": "CPU-torch mirror of the NeMo perception chain (milestone 5, "
                "docs/nemotron_transcribe_plan.md), transcribed from "
                "nemo features.py / transformer_encoder.py / subsampling.py / "
                "multi_head_attention.py / parallel_expert_encoder.py at main "
                "(fetched 2026-10-31). Sortformer's internals are 5b; its "
                "activity is an input here.",
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
