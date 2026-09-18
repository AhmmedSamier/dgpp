#!/usr/bin/env python3
"""Regenerate glm_vision_test's CUDA PyTorch LayerNorm/GELU golden hashes.

The fixed integer generator is independent of torch's RNG. No checkpoint is
needed. Recorded with PyTorch 2.14.0+cu130 on GB10; inspect numerical changes
before updating a golden value for another backend/version.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F


def main():
    state = 337

    def values(count):
        nonlocal state
        result = []
        for _ in range(count):
            state = (state * 1664525 + 1013904223) & 0xffffffff
            mantissa = ((state >> 8) % 251 - 125) / 128
            result.append(np.ldexp(np.float32(mantissa), int(state % 9) - 4))
        return torch.tensor(np.array(result), device='cuda').bfloat16()

    def digest(value):
        result = 14695981039346656037
        for b in value.view(torch.uint16).cpu().numpy().astype('<u2').tobytes():
            result = ((result ^ b) * 1099511628211) & ((1 << 64) - 1)
        return hex(result)

    x = values(16 * 4096).reshape(16, 4096)
    weight, bias = values(4096), values(4096)
    norm = F.layer_norm(x, (4096,), weight, bias, 1e-5)
    print(json.dumps({'torch': torch.__version__, 'rows': 16, 'dim': 4096,
                      'normalized_hash': digest(norm), 'gelu_hash': digest(F.gelu(norm))}, indent=2))


if __name__ == '__main__':
    main()
