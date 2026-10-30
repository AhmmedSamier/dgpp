#!/usr/bin/env python3
"""Golden frontend features for dgpp's audio admission path (docs/audio.md).

Runs transformers' Qwen3ASRFeatureExtractor over deterministic synthetic
clips and dumps, per case, the exact int16 samples, the frame count, the
encoder token count and the fp32 log-mel features (base64, frame-major
after the transpose dgpp's tower consumes). The unit tests in
tests/unit/audio_*_test.cpp must match these within tight tolerances.

Usage:  python3 tools/qwen3_asr_feature_goldens.py \
            --out tests/data/qwen3_asr_feature_goldens.json
"""

import argparse
import base64
import json
import struct

import numpy as np
import torch

from transformers import Qwen3ASRFeatureExtractor
from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths

SR = 16000
MAGIC = 0.4


def clips():
    rng = np.random.default_rng(20261012)
    t = lambda s: np.arange(int(s * SR), dtype=np.float64) / SR

    chirp = 0.6 * np.sin(2 * np.pi * (120 * t(1.0) + 1600 * t(1.0) ** 2))
    voice = np.zeros(int(2.2 * SR))
    tt = t(2.2)
    for k, f0 in enumerate((110.0, 233.0)):
        env = (1 + np.sin(2 * np.pi * (2.1 + k) * tt)) / 2
        harm = sum(np.sin(2 * np.pi * f0 * h * tt + h) / h for h in (1, 2, 3, 4, 5))
        voice += 0.25 * env * harm
    voice += 1e-3 * rng.standard_normal(voice.size)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t(0.2))
    silence = 1e-5 * rng.standard_normal(int(0.5 * SR))
    longish = 0.5 * np.sin(2 * np.pi * (300 * t(10.2) + 900 * t(10.2) ** 2))
    longish += 1e-3 * rng.standard_normal(longish.size)
    return {
        "chirp_1s": chirp.astype(np.float32),
        "voice_2s2": voice.astype(np.float32),
        "tone_0s2": tone.astype(np.float32),
        "silence_0s5": silence.astype(np.float32),
        "chirp_10s2": longish.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fe = Qwen3ASRFeatureExtractor()
    print("feature extractor:", {k: getattr(fe, k) for k in
                                 ("n_fft", "hop_length", "n_mels", "n_samples", "min_length",
                                  "padding") if hasattr(fe, k)})
    out = {"format": "dgpp-asr-frontends-v1", "cases": {}}
    for name, x in clips().items():
        enc = np.clip(np.round(x.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)
        dec = enc.astype(np.float32) / 32768.0
        out_c = fe(dec, sampling_rate=SR, return_attention_mask=True, return_tensors="pt")
        feats, mask = out_c["input_features"], out_c["attention_mask"]
        valid = int(mask[0].sum().item())
        tokens = _get_feat_extract_output_lengths(valid, fe.n_window)
        n = valid
        # frame-major float32 bytes, cropped to the clip's own frames
        f = feats[0, :, :n].T.contiguous().to(torch.float32).numpy()
        out["cases"][name] = {
            "samples_b64": base64.b64encode(enc.tobytes()).decode(),
            "n_samples": int(dec.size),
            "valid_frames": valid,
            "tokens": int(tokens),
            "features_b64": base64.b64encode(f.tobytes()).decode(),
            "feature_frames": n,
        }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out, "cases:", len(out["cases"]))


if __name__ == "__main__":
    main()
