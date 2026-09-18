#!/usr/bin/env python3
"""Require bitwise CUDA eager parity on synthetic and pinned AI2D RGBs.

First run prepare_fixtures.py. This runner reuses the oracle's loaded weights,
exports embeddings for controlled serving comparisons, and records each result.
The GPU must be idle; do not run this alongside a serving world.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--fixtures', type=Path, default=Path('/tmp/dgpp-vision-quality'))
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--check-bin', type=Path, default=Path('build-ci/glm_vision_check'))
    ap.add_argument('--only', action='append', help='run only a named case (repeatable)')
    args = ap.parse_args()
    repository = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('vision_reference', repository / 'tools/glm_vision_reference.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    args.output.mkdir(parents=True, exist_ok=True)
    refs = args.output / 'references'
    refs.mkdir(exist_ok=True)
    cases = [{'name': 'gradient112', 'width': 112, 'height': 112},
             {'name': 'gradient896', 'width': 896, 'height': 896}]
    for case in json.loads((args.fixtures / 'fixtures.json').read_text()):
        cases.append(dict(case, rgb=str(args.fixtures / (case['name'] + '.rgb'))))
    for case in json.loads((args.fixtures / 'ai2d/prepared.json').read_text()):
        cases.append(dict(case, name=f"ai2d-{case['index']}",
                          rgb=str(args.fixtures / 'ai2d' / f"{case['index']}.rgb")))
    results = []
    if args.only:
        missing = set(args.only) - {case['name'] for case in cases}
        if missing:
            ap.error(f'unknown cases: {sorted(missing)}')
        cases = [case for case in cases if case['name'] in args.only]
    for case in cases:
        name = case['name']
        native = args.output / (name + '.bf16')
        reference = refs / (str(case.get('digest', name)) + '.bf16')
        command = [str(args.check_bin.resolve()), str(args.checkpoint), str(case['width']),
                   str(case['height']), str(native)]
        if case.get('rgb'):
            command += ['-', case['rgb']]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        argv = [str(args.checkpoint), str(case['width']), str(case['height']), str(native),
                '--device', 'cuda', '--reference-output', str(reference),
                '--max-relative-rms', '.005', '--min-cosine', '.99998']
        if case.get('rgb'):
            argv += ['--rgb', case['rgb']]
        log = args.output / (name + '.log')
        with log.open('w') as stream, contextlib.redirect_stdout(stream):
            oracle.main(argv)
        metrics = json.loads(re.search(r'^\{.*?^\}', log.read_text(), re.M | re.S)[0])
        metrics['bitwise_equal'] = native.read_bytes() == reference.read_bytes()
        results.append(dict(name=name, width=case['width'], height=case['height'], **metrics))
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name, json.dumps(metrics), flush=True)
        if not metrics['bitwise_equal']:
            raise SystemExit(f'{name}: bitwise regression; inspect operation traces and backend versions')
    print(f'PASS: {len(results)} bitwise full-depth CUDA eager comparisons', flush=True)


if __name__ == '__main__':
    main()
