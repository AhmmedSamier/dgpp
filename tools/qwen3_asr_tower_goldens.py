#!/usr/bin/env python3
"""Golden tower pass for dgpp's Qwen3-ASR encoder (docs/audio.md).

Runs the checkpoint's audio tower (Qwen3ASREncoder + projector) in fp32 over
synthetic clips featurized by the reference processor, and dumps the exact
(input_features, input_features_mask, projected embedding rows) triples the
C++ encoder must reproduce. Rows come out packed - chunk column-major with
the valid rows of each 100-frame chunk in frame order - the packing order
dgpp's stage plan pins.

Weights are loaded BY KEY from the safetensors index (newer transformers
renamed the projector modules; a blind from_pretrained would initialize
them at random). Clips longer than the extractor's 30 s window are
segmented, matching the reference pipeline, and the segments' rows are
concatenated.

Usage: python3 tools/qwen3_asr_tower_goldens.py --out tests/data/
           qwen3_asr_tower_goldens.json [--checkpoint DIR]
"""

import argparse
import base64
import glob
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoFeatureExtractor, Qwen3ASRForConditionalGeneration

SR = 16000
WINDOW_SAMPLES = 480000  # the feature extractor's 30 s window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--checkpoint",
        default=glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/*"))[0])
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    index = json.load(open(os.path.join(args.checkpoint, "model.safetensors.index.json")))
    from transformers import AutoConfig

    raw = json.load(open(os.path.join(args.checkpoint, "config.json")))
    # The checkpoint nests its sub-configs under thinker_config; the installed
    # config class wants them flat, so build from the nested dict directly.
    cfg = AutoConfig.from_pretrained(args.checkpoint)
    cfg.audio_config.output_dim = raw["thinker_config"]["audio_config"]["output_dim"]
    model = Qwen3ASRForConditionalGeneration(cfg).to(torch.float32)
    model.eval()
    have = dict(model.named_parameters()) | dict(model.named_buffers())
    loaded = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(os.path.join(args.checkpoint, shard))
        for name, t in tensors.items():
            dst = name[len("thinker."):] if name.startswith("thinker.") else name
            if dst.startswith("model.") and not dst.startswith("model.model."):
                # thinker.model.* is the language model: model.<x> -> model.language_model.<x>
                dst = "model.language_model." + dst[len("model."):]
            elif dst.startswith("audio_tower."):
                dst = "model." + dst
            dst = dst.replace("audio_tower.proj1.", "multi_modal_projector.linear_1.")
            dst = dst.replace("audio_tower.proj2.", "multi_modal_projector.linear_2.")
            if dst not in have and name in have:
                dst = name
            if dst not in have and name == "thinker.lm_head.weight":
                continue  # tied to the (already loaded) embedding table
            if dst not in have:
                raise KeyError(f"checkpoint tensor {name} has no home in the module tree")
            have[dst].data.copy_(t.to(torch.float32))
            loaded += 1
        del tensors
    print("loaded", loaded, "tensors")
    tower, projector = model.model.audio_tower, model.model.multi_modal_projector
    fe = AutoFeatureExtractor.from_pretrained(args.checkpoint)

    rng = np.random.default_rng(4242)
    t = lambda s: np.arange(int(s * SR)) / SR
    clips = {
        # A multi-chunk clip inside one 30 s window, a sub-chunk clip, and a
        # >30 s clip that exercises the 104-token attention window (8 chunks
        # * 13) and the segmentation path.
        "chirp_10s2": (0.5 * np.sin(2 * np.pi * (220 * t(10.2) + 900 * t(10.2) ** 2))
                       + 1e-3 * rng.standard_normal(int(10.2 * SR))).astype(np.float32),
        "hush_3s": (0.02 * (np.sin(2 * np.pi * 180 * t(3.0)) ** 3)
                    + 3e-3 * rng.standard_normal(int(3.0 * SR))).astype(np.float32),
        "sweep_37s": (0.4 * np.sin(2 * np.pi * (80 * t(37.0) + 40 * t(37.0) ** 2))
                      + 1e-3 * rng.standard_normal(int(37.0 * SR))).astype(np.float32),
    }
    out = {"format": "dgpp-asr-tower-v1", "clips": {}}
    for name, x in clips.items():
        segs, rows_l = [], []
        for w in range(int(np.ceil(x.size / WINDOW_SAMPLES))):
            seg = x[w * WINDOW_SAMPLES:(w + 1) * WINDOW_SAMPLES]
            got = fe(seg, sampling_rate=SR, padding="longest", truncation=False,
                     return_attention_mask=True, return_tensors="pt")
            feats = got["input_features"]
            mask = got["attention_mask"]
            if mask.shape[-1] != feats.shape[-1]:  # raw sample-resolution mask
                mask = mask[:, ::160]
            mask = mask[:, :feats.shape[-1]]
            # The encoder chunks the frame axis at 100 frames; "longest"
            # padding stops at the clip's own frames, so close the last chunk
            # exactly as the reference processor's n_window padding does.
            if feats.shape[-1] % 100:
                pad = 100 - feats.shape[-1] % 100
                feats = torch.nn.functional.pad(feats, (0, pad))
                mask = torch.nn.functional.pad(mask, (0, pad))
            mask = mask.contiguous().to(torch.int64)
            with torch.no_grad():
                rows = projector(tower(input_features=feats, input_features_mask=mask)
                                 .last_hidden_state)
            segs.append({
                "features_b64": base64.b64encode(feats[0].numpy().tobytes()).decode(),
                "features_shape": list(feats[0].shape),
                "mask_b64": base64.b64encode(mask[0].to(torch.int32).numpy().tobytes()).decode(),
                "valid_frames": int(mask[0].sum()),
            })
            rows_l.append(rows)
        out["clips"][name] = {
            "segments": segs,
            "rows": int(sum(r.shape[0] for r in rows_l)),
            "embed_b64": base64.b64encode(torch.cat(rows_l).numpy().tobytes()).decode(),
        }
        print(name, "rows", sum(r.shape[0] for r in rows_l),
              "valid frames", [s["valid_frames"] for s in segs],
              "absmax", float(torch.cat(rows_l).abs().max()))
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
