#!/usr/bin/env python3
"""Qwen3-ASR language-half goldens: the plain Qwen3 decoder of the real
checkpoint replayed fp32 on CPU torch (docs/audio.md).

The checkpoint's text tensors are a stock Qwen3ForCausalLM under the
`thinker.` prefix; transformers 5.17's Qwen3ASR wrapper random-inits parts
of the tower (the projector key drift), so the halves are replayed
separately and loaded BY KEY here: thinker.model.* -> model.*, the tied
lm_head duplicated as the real head. One fixed token sequence; the dump is
the last row's fp32 logits (the engine's forward test compares argmax and
the last row within the bf16 walk's budget).

Writes tests/data/qwen_asr_text_goldens.json.
"""
import base64
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots")


def main() -> None:
    snap = sorted(glob.glob(os.path.join(CACHE, "*")))[-1]
    cfg = json.load(open(os.path.join(snap, "config.json")))
    text = cfg["thinker_config"]["text_config"]
    conf = Qwen3Config(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text.get("head_dim", 128),
        hidden_act=text["hidden_act"],
        max_position_embeddings=text["max_position_embeddings"],
        rms_norm_eps=text["rms_norm_eps"],
        rope_theta=text["rope_theta"],
        tie_word_embeddings=False,
        attention_bias=text.get("attention_bias", False),
    )
    torch.set_grad_enabled(False)
    model = Qwen3ForCausalLM(conf).to(torch.float32).eval()
    have = dict(model.named_parameters()) | dict(model.named_buffers())
    index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    remap = {"thinker.lm_head.weight": "lm_head.weight"}
    seen = set()
    for shard in sorted(set(index.values())):
        for name, t in load_file(os.path.join(snap, shard)).items():
            if not name.startswith("thinker.model."):
                continue
            dst = "model." + name[len("thinker.model."):]
            if dst in have:
                have[dst].data.copy_(t.to(torch.float32))
            seen.add(name)
    # The tied head, duplicated by key (the loader aliases it; this dump
    # proves the checkpoint's copy is the one we alias).
    dup = None
    for shard in sorted(set(index.values())):
        sd = load_file(os.path.join(snap, shard))
        if "thinker.lm_head.weight" in sd:
            dup = sd["thinker.lm_head.weight"]
    assert dup is not None, "thinker.lm_head.weight missing"
    assert torch.equal(dup, have["model.embed_tokens.weight"]), "lm_head is not a tied copy"
    have["lm_head.weight"].data.copy_(dup.to(torch.float32))

    # The fixed sequence: 96 deterministic ids, the audio delimiters and a
    # couple of high ids inside them.
    tokens = [(1500 + i * 7919) % (text["vocab_size"] - 1) + 1 for i in range(96)]
    for pos, tid in ((20, 151669), (21, 151676), (40, 151670), (7, 151644)):
        tokens[pos] = tid
    ids = torch.tensor([tokens], dtype=torch.long)
    logits = model(ids).logits[0, -1].contiguous()  # fp32 [vocab]
    top = torch.topk(logits, 8)
    out = {
        "note": "last-row fp32 logits of a fixed 96-token prompt, CPU torch fp32",
        "tokens": tokens,
        "logits_b64": base64.b64encode(logits.numpy().astype("<f4").tobytes()).decode(),
        "argmax": int(logits.argmax()),
        "top_ids": [int(v) for v in top.indices],
        "top_vals": [float(v) for v in top.values],
    }
    path = os.path.join(ROOT, "tests/data/qwen_asr_text_goldens.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"{path}: argmax {out['argmax']} top8 {out['top_ids'][:4]} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
