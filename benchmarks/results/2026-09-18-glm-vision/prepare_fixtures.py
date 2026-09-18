#!/usr/bin/env python3
"""Recreate the vision comparison inputs; optionally export PyTorch embeddings.

Requires the prepare.cpp utility built as described in the adjacent record.
Only the optional reference subprocess needs NumPy/PyTorch. Downloaded data
stays outside the repository. DGPP_VISION_FIXTURES overrides the work directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from urllib.request import urlopen
import zlib


def png(rgb, width, height):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    pixels = b''.join(b'\0' + rgb[y * width * 3:(y + 1) * width * 3]
                      for y in range(height))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-bin', required=True)
    ap.add_argument('--checkpoint', type=Path, help='export CUDA PyTorch reference rows when set')
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()
    root = Path(os.environ.get('DGPP_VISION_FIXTURES', '/tmp/dgpp-vision-quality'))
    root.mkdir(parents=True, exist_ok=True)
    synthetic = []
    for name, width, height in [('gradient', 168, 112), ('quadrants', 224, 224), ('bars', 224, 224)]:
        rgb = bytearray()
        for y in range(height):
            for x in range(width):
                if name == 'gradient':
                    color = (x * 255 // (width - 1), y * 255 // (height - 1),
                             (x + y) * 255 // (width + height - 2))
                elif name == 'quadrants':
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][
                        (y >= height // 2) * 2 + (x >= width // 2)]
                else:
                    color = (255, 255, 255)
                    for i, (top, c) in enumerate([(112, (255, 0, 0)), (28, (0, 0, 255)),
                                                   (70, (0, 180, 0))]):
                        if 14 + i * 70 <= x < 56 + i * 70 and top <= y < 210:
                            color = c
                rgb.extend(color)
        digest = 14695981039346656037
        for byte in rgb:
            digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
        (root / (name + '.rgb')).write_bytes(rgb)
        (root / (name + '.png')).write_bytes(png(rgb, width, height))
        synthetic.append(dict(name=name, width=width, height=height, digest=digest))
    (root / 'fixtures.json').write_text(json.dumps(synthetic, indent=2) + '\n')

    with urlopen('https://datasets-server.huggingface.co/first-rows?'
                 'dataset=lmms-lab-encoder/ai2d&config=default&split=test', timeout=30) as response:
        rows = json.load(response)['rows'][::4]
    expected = json.loads((Path(__file__).parent / 'case_manifest.json').read_text())
    cases = []
    images = root / 'ai2d'
    images.mkdir(exist_ok=True)
    for item, saved in zip(rows, expected, strict=True):
        row, index = item['row'], item['row_idx']
        with urlopen(row['image']['src'], timeout=30) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        question_digest = hashlib.sha256(json.dumps(
            {k: row[k] for k in ('question', 'options', 'answer')}, sort_keys=True).encode()).hexdigest()
        if (index != saved['index'] or digest != saved['sha256']
                or question_digest != saved['question_sha256']):
            raise SystemExit('dataset/image bytes changed since the recorded comparison')
        image = images / f'{index}.jpg'
        image.write_bytes(data)
        rgb = images / f'{index}.rgb'
        w, h, tokens, image_hash = map(int, subprocess.check_output(
            [args.prepare_bin, str(image), str(rgb)], text=True).split())
        cases.append(dict(index=index, question=row['question'], options=row['options'],
                          answer=int(row['answer']), file=image.name, sha256=digest,
                          width=w, height=h, tokens=tokens, digest=image_hash))
    (images / 'prepared.json').write_text(json.dumps(cases, indent=2) + '\n')
    if args.checkpoint:
        seen = set()
        for case in synthetic + cases:
            digest = case['digest']
            if digest in seen:
                continue
            seen.add(digest)
            rgb = (root / (case['name'] + '.rgb') if 'name' in case
                   else images / f"{case['index']}.rgb")
            subprocess.run([args.python, 'tools/glm_vision_reference.py', str(args.checkpoint),
                            str(case['width']), str(case['height']), '--device', 'cuda',
                            '--rgb', str(rgb), '--reference-output', str(root / f'{digest}.bf16')],
                           check=True)


if __name__ == '__main__':
    main()
