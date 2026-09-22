#!/usr/bin/env python3
"""Compare qwen_vision_check's deterministic RGB fixture with a PyTorch oracle.

Requires torch and numpy, and reads only the checkpoint's model.visual.* BF16
weights: no text stack is instantiated. The equations are transformers'
Qwen3VL vision tower -- which is what Qwen4ExpForConditionalGeneration ships,
with an empty deepstack_visual_indexes -- restated here directly, so a shared
bug with the C++ side is unlikely and an indexing mistake shows up as a large
error rather than a silent match.

The CUDA oracle uses the native encoder's BF16 projections and probabilities
with FP32 QK scores and scaling. It does not assert bitwise parity with a
Transformers attention backend that rounds scores to BF16.

Usage: python tools/qwen_vision_reference.py CHECKPOINT WIDTH HEIGHT NATIVE.bf16

WIDTH and HEIGHT are the processed canvas (multiples of 32) and must match the
qwen_vision_check run that wrote NATIVE.bf16:

  build-ci/qwen_vision_check "$CKPT" 64 64 /tmp/qwen-vision.bf16
  python3 tools/qwen_vision_reference.py "$CKPT" 64 64 /tmp/qwen-vision.bf16

--trace-dir also writes each stage so a mismatch can be located layer by
layer. --diagnostic runs the oracle in fp32 on CPU, which separates a
precision contract from an arithmetic or indexing bug.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PATCH, MERGE, TEMPORAL, SIDE = 16, 2, 2, 48  # patch side, merge, frames, table side
GRID = PATCH * MERGE  # 32 pixels per visual token


def load(checkpoint, device):
    cfg = json.loads((checkpoint / "config.json").read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted({v for k, v in index.items() if k.startswith("model.visual.")}):
        path = checkpoint / shard
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        data = np.memmap(path, dtype=np.uint8, mode="c")
        for name, info in header.items():
            if not name.startswith("model.visual."):
                continue
            assert info["dtype"] == "BF16", (name, info["dtype"])
            lo, hi = info["data_offsets"]
            weights[name.removeprefix("model.visual.")] = torch.from_numpy(
                data[8 + size + lo:8 + size + hi].view(np.uint16).reshape(info["shape"])
            ).view(torch.bfloat16).to(device)
    return cfg, weights


def fixture(width, height, device):
    """The RGB bytes qwen_vision_check generates for the same canvas."""
    x = torch.arange(width, device=device, dtype=torch.int64).view(1, -1)
    y = torch.arange(height, device=device, dtype=torch.int64).view(-1, 1)
    r = (x * 255 // (width - 1)).to(torch.uint8).expand(height, -1)
    g = (y * 255 // (height - 1)).to(torch.uint8).expand(-1, width)
    b = ((x + y) * 255 // (width + height - 2)).to(torch.uint8).expand(height, width)
    return torch.stack([r, g, b], dim=-1).contiguous()


def token_coords(width, height):
    """(patch row, patch column) per token, in merge-block order.

    Token (bh*wb + bw)*4 + mh*2 + mw is the merge offset (mh, mw) of block
    (bh, bw), which is the order the merger concatenates patches in.
    """
    hb, wb = height // GRID, width // GRID
    bh = np.repeat(np.repeat(np.arange(hb), wb), 4)
    bw = np.repeat(np.tile(np.arange(wb), hb), 4)
    mm = np.tile(np.arange(4), hb * wb)
    return bh * MERGE + mm // MERGE, bw * MERGE + mm % MERGE


def patchify(rgb, width, height, device):
    """[H,W,3] uint8 -> [4*tokens, 3*2*16*16] with mean = std = 0.5."""
    py, px = token_coords(width, height)
    f = np.arange(3 * TEMPORAL * PATCH * PATCH)
    c = f // (TEMPORAL * PATCH * PATCH)
    r = f % (TEMPORAL * PATCH * PATCH)
    ph, pw = (r // PATCH) % PATCH, r % PATCH  # the frame index picks no pixel
    y = py[:, None] * PATCH + ph[None, :]
    x = px[:, None] * PATCH + pw[None, :]
    host = rgb.cpu().numpy()
    out = host[y, x, c[None, :]]  # (tokens*4, patch_in)
    return torch.from_numpy(out.astype(np.float32)).to(device) / 255.0 * 2.0 - 1.0


def position_embeddings(weights, width, height, device, dtype):
    """The learned 48x48 table bilinearly resampled, in token order."""
    table = weights["pos_embed.weight"].to(dtype)
    py, px = token_coords(width, height)
    hy = np.linspace(0, SIDE - 1, height // PATCH)[py]
    wx = np.linspace(0, SIDE - 1, width // PATCH)[px]
    hf, wf = np.floor(hy).astype(np.int64), np.floor(wx).astype(np.int64)
    hc, wc = np.minimum(hf + 1, SIDE - 1), np.minimum(wf + 1, SIDE - 1)
    dh, dw = (hy - hf)[:, None], (wx - wf)[:, None]
    idx = np.stack([hf * SIDE + wf, hf * SIDE + wc, hc * SIDE + wf, hc * SIDE + wc])
    wgt = np.stack([(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw])
    # Interpolate in FP32 and round the final sum to the encoder's dtype.
    gathered = table[torch.from_numpy(idx).to(device)] * torch.from_numpy(
        wgt.astype(np.float32)).to(device)
    return gathered.sum(dim=0).to(dtype)


def rope_tables(width, height, dim, theta, device, dtype):
    """cos/sin of width [tokens, dim]: transformers'
    Qwen2VLVisionRotaryEmbedding(dim // 2) -- dim/4 frequencies per axis, the
    angle vector [h grid, w grid] of length dim/2, stored twice because
    rotate_half reads a full-head-width angle vector."""
    py, px = token_coords(width, height)
    angles, n_freq = dim // 2, dim // 4
    inv = np.array([theta ** (-2.0 * i / angles) for i in range(n_freq)], dtype=np.float64)
    ang_h = torch.from_numpy(py.astype(np.float64)[:, None] * inv[None, :]).to(device)
    ang_w = torch.from_numpy(px.astype(np.float64)[:, None] * inv[None, :]).to(device)
    ang = torch.cat([ang_h, ang_w], dim=1)  # (tokens, dim/2)
    ang = torch.cat([ang, ang], dim=1)  # (tokens, dim), rotate_half's view
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def layernorm(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight.to(x.dtype), bias.to(x.dtype), eps)


def encode(cfg, weights, width, height, device, dtype, trace_dir=None):
    v = cfg["vision_config"]
    assert v["patch_size"] == PATCH and v["temporal_patch_size"] == TEMPORAL
    assert v["spatial_merge_size"] == MERGE and not v["deepstack_visual_indexes"]
    hidden, heads = v["hidden_size"], v["num_heads"]
    dim, eps = hidden // heads, v.get("layer_norm_eps", 1e-6)
    tokens = (width // GRID) * (height // GRID)

    def dump(name, value):
        if trace_dir is None:
            return
        (trace_dir / (name + ".f32")).write_bytes(value.detach().float().cpu().numpy().tobytes())

    rgb = fixture(width, height, device)
    dump("fixture_rgb", rgb)
    x = patchify(rgb, width, height, device).to(dtype)
    # nn.Linear/Conv3d include bias before the BF16 output rounding. A
    # separate BF16 matmul followed by addition introduces another rounding.
    def lin(name, inp):
        weight = weights[name + ".weight"].to(dtype)
        return F.linear(inp, weight.reshape(weight.shape[0], -1),
                        weights[name + ".bias"].to(dtype))

    x = lin("patch_embed.proj", x)
    dump("patch_projection", x)
    positions = position_embeddings(weights, width, height, device, dtype)
    dump("position_embeddings", positions)
    x = x + positions
    dump("patch_embed", x)
    cos, sin = rope_tables(width, height, dim, v.get("rope_theta", 10000.0), device, torch.float32)
    scale = dim ** -0.5
    for i in range(v["depth"]):
        p = f"blocks.{i}."
        h = layernorm(x, weights[p + "norm1.weight"], weights[p + "norm1.bias"], eps)
        dump(f"layer{i}.norm1", h)
        qkv = lin(p + "attn.qkv", h)
        dump(f"layer{i}.qkv", qkv)
        qkv = qkv.view(-1, 3, heads, dim)
        q, k, val = qkv.unbind(dim=1)
        # Upstream apply_rotary_pos_emb_vision promotes the entire rotation
        # to FP32, then casts the final q/k back to the input dtype.
        q = (q.float() * cos.view(-1, 1, dim) +
             rotate_half(q.float()) * sin.view(-1, 1, dim)).to(dtype)
        k = (k.float() * cos.view(-1, 1, dim) +
             rotate_half(k.float()) * sin.view(-1, 1, dim)).to(dtype)
        dump(f"layer{i}.q", q.transpose(0, 1).contiguous())
        dump(f"layer{i}.k", k.transpose(0, 1).contiguous())
        qh, kh = q.transpose(0, 1), k.transpose(0, 1)
        # Keep BF16 GEMM inputs with FP32 scores, matching the encoder's
        # attention contract without switching to a different FP32 GEMM.
        if q.is_cuda and dtype == torch.bfloat16:
            scores = torch.bmm(qh, kh.transpose(1, 2), out_dtype=torch.float32)
        else:
            scores = torch.bmm(qh.float(), kh.float().transpose(1, 2))
        att = torch.softmax(scores * scale, dim=-1).to(dtype)
        o = torch.einsum("hqk,khd->qhd", att, val)
        projected = lin(p + "attn.proj", o.reshape(-1, hidden))
        dump(f"layer{i}.attn_proj", projected)
        x = x + projected
        h = layernorm(x, weights[p + "norm2.weight"], weights[p + "norm2.bias"], eps)
        dump(f"layer{i}.norm2", h)
        h = lin(p + "mlp.linear_fc1", h)
        dump(f"layer{i}.mlp_hidden", h)
        h = F.gelu(h, approximate="tanh")
        dump(f"layer{i}.mlp_act", h)
        projected = lin(p + "mlp.linear_fc2", h)
        dump(f"layer{i}.mlp_proj", projected)
        x = x + projected
        dump(f"layer{i}", x)
    x = layernorm(x, weights["merger.norm.weight"], weights["merger.norm.bias"], eps)
    dump("merger_norm", x)
    x = x.view(tokens, hidden * MERGE * MERGE)
    x = lin("merger.linear_fc1", x)
    x = F.gelu(x)
    return lin("merger.linear_fc2", x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("native", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diagnostic", action="store_true", help="fp32 on CPU")
    ap.add_argument("--trace-dir", type=Path)
    ap.add_argument("--max-rel", type=float, default=0.005)
    ap.add_argument("--min-cosine", type=float, default=0.99998)
    args = ap.parse_args(argv)
    for side in (args.width, args.height):
        if side < GRID or side % GRID:
            ap.error(f"each side must be a multiple of {GRID}")
    device = "cpu" if args.diagnostic else args.device
    dtype = torch.float32 if args.diagnostic else torch.bfloat16
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    cfg, weights = load(args.checkpoint, device)
    want = (args.width // GRID) * (args.height // GRID) * cfg["vision_config"]["out_hidden_size"]
    native = torch.from_numpy(np.fromfile(args.native, dtype=np.uint16)).view(torch.bfloat16)
    if native.numel() != want:
        ap.error(f"{args.native} holds {native.numel()} values, expected {want}")
    mine = encode(cfg, weights, args.width, args.height, device, dtype, args.trace_dir)
    native = native.view(want // cfg["vision_config"]["out_hidden_size"],
                         cfg["vision_config"]["out_hidden_size"]).to(device=device, dtype=dtype)
    delta = (mine.float() - native.float()).abs()
    rel = (delta.pow(2).mean().sqrt() / native.float().pow(2).mean().sqrt()).item()
    cos = F.cosine_similarity(mine.float().flatten(), native.float().flatten(), dim=0).item()
    print(f"{args.width}x{args.height}, {mine.shape[0]} rows: max {delta.max().item():.6g}, "
          f"rel {rel:.6g}, cosine {cos:.8f}")
    if not np.isfinite(rel) or rel > args.max_rel or cos < args.min_cosine:
        print("FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
