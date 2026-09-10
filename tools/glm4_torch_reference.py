#!/usr/bin/env python3
"""GLM-4.7 independent reference (2026-09-10): transformers' own Glm4Moe
layer code (modeling_glm4_moe.py) over the REAL checkpoint's first layers,
against the engine's per-layer residual dump (glm4_forward_check
--dump-states --layers N). The engine's other gates compare it to a
python reference written from the same reading of the architecture; this
one is the reading transformers ships.

The NVFP4 tensors (modelopt: `weight` U8 e2m1 pairs, low nibble = even
element; `weight_scale` e4m3 per 16; `weight_scale_2` f32) are dequantized
to bf16 with w = e2m1 * e4m3 * weight_scale_2 — the value the engine's
kernels multiply — and loaded into a bf16 model on the CPU (a 5090's 32 GB
is taken by other work; four layers run in seconds there).

  glm4_torch_reference.py --checkpoint-dir DIR --ids 1,2,3 --layers 4 \
      --engine-dump states.bin [--out ref.bin]
"""
import argparse
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_nvfp4(payload_u8, scale_e4m3, ws2):
    """[N, K/2] u8, [N, K/16] e4m3 (as uint8 bits), f32 -> [N, K] float32."""
    n, half = payload_u8.shape
    lo = (payload_u8 & 0x0F).to(torch.long)
    hi = ((payload_u8 >> 4) & 0x0F).to(torch.long)
    codes = torch.stack([lo, hi], dim=-1).reshape(n, half * 2)
    vals = E2M1[codes]
    scales = scale_e4m3.view(torch.float8_e4m3fn).to(torch.float32)  # [N, K/16]
    scales = scales.repeat_interleave(16, dim=1)
    return vals * scales * float(ws2)


class Shards:
    def __init__(self, d):
        self.d = d
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, name):
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.d, shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name)

    def has(self, name):
        return name in self.index

    def linear(self, prefix):
        """A Linear's weight as float32: BF16 verbatim or NVFP4 dequantized."""
        if self.has(prefix + ".weight_scale_2"):
            w = self.get(prefix + ".weight")
            s = self.get(prefix + ".weight_scale")
            if s.dtype != torch.uint8:
                s = s.view(torch.uint8)
            return dequant_nvfp4(w, s, self.get(prefix + ".weight_scale_2").item())
        return self.get(prefix + ".weight").to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--engine-dump", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    from transformers import AutoConfig
    from transformers.models.glm4_moe import modeling_glm4_moe as mg
    cfg = AutoConfig.from_pretrained(args.checkpoint_dir)
    cfg.num_hidden_layers = args.layers
    for k in ("quantization_config",):
        if hasattr(cfg, k):
            try:
                delattr(cfg, k)
            except Exception:
                pass
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    model = mg.Glm4MoeModel(cfg).to(dtype).eval()
    sh = Shards(args.checkpoint_dir)
    sd = model.state_dict()
    print("model params:", len(sd), "e.g.", [k for k in list(sd.keys())[:6]], file=sys.stderr)
    new = {}
    new["embed_tokens.weight"] = sh.get("model.embed_tokens.weight")
    new["norm.weight"] = sh.get("model.norm.weight")
    for l in range(args.layers):
        p = f"model.layers.{l}."
        q = f"layers.{l}."
        new[q + "input_layernorm.weight"] = sh.get(p + "input_layernorm.weight")
        new[q + "post_attention_layernorm.weight"] = sh.get(p + "post_attention_layernorm.weight")
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            new[q + f"self_attn.{m}.weight"] = sh.linear(p + f"self_attn.{m}")
            if sh.has(p + f"self_attn.{m}.bias"):
                new[q + f"self_attn.{m}.bias"] = sh.get(p + f"self_attn.{m}.bias")
        for m in ("q_norm", "k_norm"):
            new[q + f"self_attn.{m}.weight"] = sh.get(p + f"self_attn.{m}.weight")
        if l < cfg.first_k_dense_replace:
            for m in ("gate_proj", "up_proj", "down_proj"):
                new[q + f"mlp.{m}.weight"] = sh.linear(p + f"mlp.{m}")
        else:
            new[q + "mlp.gate.weight"] = sh.get(p + "mlp.gate.weight")
            new[q + "mlp.gate.e_score_correction_bias"] = sh.get(p + "mlp.gate.e_score_correction_bias")
            for m in ("gate_proj", "up_proj", "down_proj"):
                new[q + f"mlp.shared_experts.{m}.weight"] = sh.linear(p + f"mlp.shared_experts.{m}")
            E = cfg.n_routed_experts
            # transformers 5.x fuses the experts: gate_up_proj [E, 2I, H], down_proj [E, H, I];
            # older versions keep experts.{e}.{gate,up,down}_proj.
            if q + "mlp.experts.gate_up_proj" in sd:
                gu = torch.empty_like(sd[q + "mlp.experts.gate_up_proj"], dtype=torch.float32)
                dn = torch.empty_like(sd[q + "mlp.experts.down_proj"], dtype=torch.float32)
                for e in range(E):
                    g = sh.linear(p + f"mlp.experts.{e}.gate_proj")
                    u = sh.linear(p + f"mlp.experts.{e}.up_proj")
                    d = sh.linear(p + f"mlp.experts.{e}.down_proj")
                    if gu.shape[1] == g.shape[0] + u.shape[0]:      # [E, 2I, H]
                        gu[e, : g.shape[0]] = g
                        gu[e, g.shape[0]:] = u
                    else:                                             # [E, H, 2I] (transposed layout)
                        gu[e, :, : g.shape[0]] = g.T
                        gu[e, :, g.shape[0]:] = u.T
                    if dn.shape[1] == d.shape[0]:
                        dn[e] = d
                    else:
                        dn[e] = d.T
                new[q + "mlp.experts.gate_up_proj"] = gu
                new[q + "mlp.experts.down_proj"] = dn
            else:
                for e in range(E):
                    for m in ("gate_proj", "up_proj", "down_proj"):
                        new[q + f"mlp.experts.{e}.{m}.weight"] = sh.linear(p + f"mlp.experts.{e}.{m}")
        print(f"layer {l} loaded", file=sys.stderr)
    missing = [k for k in sd if k not in new]
    extra = [k for k in new if k not in sd]
    print("missing:", missing[:10], "extra:", extra[:10], file=sys.stderr)
    cast = {k: v.to(sd[k].dtype) for k, v in new.items() if k in sd}
    model.load_state_dict(cast, strict=False)
    ids = torch.tensor([[int(v) for v in args.ids.split(",")]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states  # [0] = embeddings, [l+1] = after layer l (the last one post-norm in HF)
    T, H = ids.shape[1], cfg.hidden_size
    print("hidden states:", len(hs), "shape", tuple(hs[1].shape), file=sys.stderr)
    if args.engine_dump:
        raw = open(args.engine_dump, "rb").read()
        per = T * H * 2
        n_layers = (len(raw) - per) // per
        print(f"engine dump: {n_layers} layers + final of [{T}, {H}] bf16", file=sys.stderr)
        for l in range(min(n_layers, args.layers)):
            eng = torch.frombuffer(bytearray(raw[l * per:(l + 1) * per]), dtype=torch.bfloat16).reshape(T, H).float()
            # HF's last hidden state carries the final norm; the layer outputs before it are the residuals.
            ref = hs[l + 1][0].float() if l + 1 < len(hs) - 1 else None
            if ref is None:
                print(f"layer {l}: HF's last hidden state is post-norm; skipped")
                continue
            d = eng - ref
            l2 = d.norm() / (ref.norm() + 1e-30)
            row = (d.norm(dim=1) / (ref.norm(dim=1) + 1e-30))
            print(f"layer {l}: rel l2 {l2:.4g}  max|d| {d.abs().max():.4g}  ref rms {ref.pow(2).mean().sqrt():.4g} max {ref.abs().max():.4g}"
                  f"  worst rows " + " ".join(f"t{int(i)}:{row[i]:.3g}" for i in row.argsort(descending=True)[:4]))


if __name__ == "__main__":
    main()
