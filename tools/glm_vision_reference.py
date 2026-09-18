#!/usr/bin/env python3
"""Compare glm_vision_check's deterministic RGB fixture with a PyTorch oracle.

Requires torch and numpy, reads only model.visual.* BF16 weights. Equations
follow transformers/models/glm5_next/modeling_glm5_next.py. No model text stack
is instantiated. Usage: python tools/glm_vision_reference.py CHECKPOINT WIDTH
HEIGHT NATIVE.bf16. The native utility must use the same width and height.

The default target is CUDA BF16 eager attention, with a 0.5% relative-RMS
and 0.99998 cosine full-depth gate. --isolate-layers and --isolate-ops feed
native inputs into each block/operation to locate a mismatch. Use
--diagnostic when investigating another precision contract or CPU backend.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import struct

import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def load_weights(checkpoint, device):
    cfg = json.loads((checkpoint / 'config.json').read_text())['vision_config']
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    weights = {}
    maps = []
    for shard in sorted({v for k, v in index.items() if k.startswith('model.visual.')}):
        p = checkpoint / shard
        with p.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        data = np.memmap(p, dtype=np.uint8, mode='c')
        maps.append(data)
        for name, info in header.items():
            if not name.startswith('model.visual.'):
                continue
            assert info['dtype'] == 'BF16', (name, info['dtype'])
            lo, hi = info['data_offsets']
            weights[name.removeprefix('model.visual.')] = torch.from_numpy(
                data[8 + size + lo:8 + size + hi].view(np.uint16).reshape(info['shape'])
            ).view(torch.bfloat16).to(device)

    return cfg, weights


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('width', type=int)
    ap.add_argument('height', type=int)
    ap.add_argument('native', type=Path, nargs='?')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--attention', choices=('eager', 'fp32'), default='eager',
                    help='fp32 scores (initial native contract) or upstream BF16 eager attention')
    ap.add_argument('--trace-dir', type=Path)
    ap.add_argument('--isolate-layers', action='store_true')
    ap.add_argument('--isolate-ops', action='store_true', help='feed native inputs to each traced operation')
    ap.add_argument('--reference-output', type=Path)
    ap.add_argument('--rgb', type=Path, help='optional preprocessed HWC uint8 RGB fixture')
    ap.add_argument('--max-relative-rms', type=float, default=.005, help='full-depth comparison bound (default: 0.5%)')
    ap.add_argument('--min-cosine', type=float, default=.99998)
    ap.add_argument('--diagnostic', action='store_true', help='report full-depth distances without a parity verdict')
    args = ap.parse_args(argv)
    if args.diagnostic and (args.isolate_layers or args.isolate_ops):
        ap.error('--diagnostic cannot disable isolated arithmetic checks')
    if (args.isolate_layers or args.isolate_ops) and not args.trace_dir:
        ap.error('isolated comparisons require --trace-dir')
    if not args.native and not args.reference_output:
        ap.error('provide a native output to compare or --reference-output to export')
    if (min(args.width, args.height) < 28 or args.width % 28 or args.height % 28
            or args.width * args.height > 1024 * 28 * 28):
        ap.error('dimensions must be positive multiples of 28 within 1024 visual tokens')
    if args.max_relative_rms is not None and not 0 <= args.max_relative_rms < float('inf'):
        ap.error('--max-relative-rms must be finite and nonnegative')
    if args.min_cosine is not None and not -1 <= args.min_cosine <= 1:
        ap.error('--min-cosine must be between -1 and 1')
    torch.set_num_threads(8)
    torch.set_default_device(args.device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg, weights = load_weights(args.checkpoint, args.device)

    def linear(x, name):
        return F.linear(x, weights[name + '.weight'].flatten(1), weights.get(name + '.bias'))

    def rms(x, name):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + cfg['rms_norm_eps'])
        return value.bfloat16() * weights[name + '.weight']

    def mlp(x, name):
        gate = linear(x, name + '.gate_proj').clamp(max=cfg['swiglu_limit'])
        up = linear(x, name + '.up_proj').clamp(-cfg['swiglu_limit'], cfg['swiglu_limit'])
        act = F.silu(gate) * up
        if name == 'merger':
            act = trace('merger_swiglu', act)
        return linear(act, name + '.down_proj')

    w, h = args.width, args.height
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    rgb = torch.stack((xs * 255 // (w - 1), ys * 255 // (h - 1), (xs + ys) * 255 // (w + h - 2))).float()
    if args.rgb:
        rgb = torch.from_numpy(np.fromfile(args.rgb, dtype=np.uint8).reshape(h, w, 3)).to(args.device).permute(2, 0, 1).float()
    mean = torch.tensor([.48145466, .4578275, .40821073])[:, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711])[:, None, None]
    pixels = (rgb * (1 / 255) - mean) / std
    gh, gw = h // 14, w // 14
    patches = pixels.reshape(3, gh // 2, 2, 14, gw // 2, 2, 14).permute(1, 4, 2, 5, 0, 3, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1).reshape(-1, 1176).bfloat16()
    def trace(name, expected, operation=False, round_bf16=False):
        if not args.trace_dir:
            return expected
        f32 = expected.dtype == torch.float32
        path = args.trace_dir / (name + ('.f32' if f32 else '.bf16'))
        tiles = sorted(args.trace_dir.glob(name + '.tile*' + path.suffix),
                       key=lambda p: int(p.stem.rsplit('tile', 1)[1]))
        if operation and not path.exists() and not tiles:
            if args.isolate_ops:
                raise SystemExit(f'missing operation trace: {name}')
            return expected
        data = [np.fromfile(p, dtype=np.float32 if f32 else np.uint16) for p in (tiles or [path])]
        actual = torch.from_numpy(np.concatenate(data))
        if not f32:
            actual = actual.view(torch.bfloat16)
        actual = actual.reshape(expected.shape).float().to(args.device)
        if round_bf16:
            actual = actual.bfloat16().float()
        e = expected.float()
        relative_rms = ((actual-e).square().mean().sqrt() / e.square().mean().sqrt()).item()
        cosine = F.cosine_similarity(actual.flatten(), e.flatten(), dim=0).item()
        print(name, 'max', (actual-e).abs().max().item(), 'rel', relative_rms, 'cosine', cosine,
              'mismatches', (actual != e).count_nonzero().item(), '/', e.numel())
        if name == 'patches' and not torch.equal(actual, e):
            raise SystemExit('patch layout/normalization mismatch')
        isolated = args.isolate_ops or (args.isolate_layers and not operation)
        if isolated and (not torch.isfinite(actual).all() or relative_rms > .005 or cosine < .99998):
            raise SystemExit(f'{name}: isolated block parity failed')
        return actual.to(expected.dtype) if isolated else expected
    patches = trace('patches', patches)
    x = linear(patches, 'patch_embed.proj')
    x = trace('patch_embed', x)
    heads, hidden = cfg['num_heads'], cfg['hidden_size']
    d = hidden // heads
    ys = torch.arange(gh)[:, None].expand(gh, gw).reshape(gh // 2, 2, gw // 2, 2).permute(0, 2, 1, 3).reshape(-1)
    xs = torch.arange(gw)[None, :].expand(gh, gw).reshape(gh // 2, 2, gw // 2, 2).permute(0, 2, 1, 3).reshape(-1)
    inv = 1 / (10000 ** (torch.arange(0, d // 2, 2).float() / (d // 2)))
    angles = torch.cat((ys[:, None] * inv, xs[:, None] * inv), -1).repeat(1, 2)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]

    def rope(x):
        x = x.float()
        return (x * cos + torch.cat((-x[..., d // 2:], x[..., :d // 2]), -1) * sin).bfloat16()

    for layer in range(cfg['depth']):
        p = f'blocks.{layer}'
        t = f'layer{layer}.'
        norm = trace(t + 'norm1', rms(x, p + '.norm1'), True)
        qkv = trace(t + 'qkv', linear(norm, p + '.attn.qkv'), True)
        q, k, v = qkv.reshape(-1, 3, heads, d).unbind(1)
        q, k = rope(rms(q, p + '.attn.q_norm')), rope(rms(k, p + '.attn.k_norm'))
        q = trace(t + 'q', q.transpose(0, 1), True).transpose(0, 1)
        k = trace(t + 'k', k.transpose(0, 1), True).transpose(0, 1)
        qh, kh = q.transpose(0, 1), k.transpose(0, 1)
        if args.attention == 'eager':
            scores = qh @ kh.transpose(-1, -2)
            # Native captures the FP32 accumulator before the BF16 boundary.
            scores[0] = trace(t + 'scores0', scores[0].float(), True, True).bfloat16()
            probs = (scores * (d ** -.5)).float().softmax(-1).bfloat16()
        else:
            scores = qh.float() @ kh.float().transpose(-1, -2)
            scores[0] = trace(t + 'scores0', scores[0], True)
            probs = (scores / d ** .5).softmax(-1).bfloat16()
        for head in range(heads):
            probs[head] = trace(t + f'probs{head}', probs[head], True)
        attn = (probs @ v.transpose(0, 1)).transpose(0, 1).reshape(-1, hidden)
        attn = trace(t + 'attn', attn, True)
        proj = trace(t + 'attn_proj', linear(attn, p + '.attn.proj'), True)
        x = trace(t + 'residual', x + proj, True)
        norm = trace(t + 'norm2', rms(x, p + '.norm2'), True)
        gate = trace(t + 'gate', linear(norm, p + '.mlp.gate_proj'), True)
        up = trace(t + 'up', linear(norm, p + '.mlp.up_proj'), True)
        act = F.silu(gate.clamp(max=cfg['swiglu_limit'])) * up.clamp(-cfg['swiglu_limit'], cfg['swiglu_limit'])
        act = trace(t + 'swiglu', act, True)
        down = trace(t + 'down', linear(act, p + '.mlp.down_proj'), True)
        x = x + down
        x = trace('layer' + str(layer), x)
    x = trace('post_norm', rms(x, 'post_layernorm'))
    x = trace('merge', x.reshape(-1, 4, hidden).transpose(1, 2).reshape(-1, 4 * hidden))
    x = linear(x, 'downsample')
    x = trace('downsample', x)
    x = linear(x, 'merger.proj')
    x = trace('projection', x)
    x = F.layer_norm(x, (cfg['out_hidden_size'],), weights['merger.post_projection_norm.weight'],
                     weights['merger.post_projection_norm.bias'], 1e-5)
    x = trace('projection_norm', x, True)
    x = F.gelu(x)
    x = trace('norm_gelu', x)
    expected = mlp(x, 'merger').float()
    if not torch.isfinite(expected).all():
        raise SystemExit('non-finite reference embeddings')
    if args.reference_output:
        expected.bfloat16().view(torch.uint16).cpu().numpy().tofile(args.reference_output)
    if not args.native:
        return
    actual = torch.from_numpy(np.fromfile(args.native, dtype=np.uint16)).view(torch.bfloat16).reshape(expected.shape).float().to(args.device)
    diff = (actual - expected).abs()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    relative_rms = (diff.square().mean().sqrt() / expected.square().mean().sqrt()).item()
    print(json.dumps({'tokens': len(expected), 'max_abs': diff.max().item(), 'relative_rms': relative_rms, 'cosine': cosine}, indent=2))
    if not torch.isfinite(actual).all():
        raise SystemExit('non-finite native embeddings')
    if args.diagnostic:
        print('Diagnostic comparison: no full-depth parity verdict.')
        return
    max_rms, min_cosine = (.005, .99998) if args.isolate_layers or args.isolate_ops else (args.max_relative_rms, args.min_cosine)
    if relative_rms > max_rms or cosine < min_cosine:
        raise SystemExit('vision encoder parity failed')
    print('PASS: vision encoder parity')



if __name__ == '__main__':
    main()
