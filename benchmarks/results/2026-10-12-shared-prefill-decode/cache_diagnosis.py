"""Compare a cache-under-load observation across fresh candidate/control servers."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from campaign import ROOT, run, capture

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--candidate', type=Path, required=True)
ap.add_argument('--control', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--logprobs', action='store_true')
ap.add_argument('--variants', nargs='+', choices=['thinking', 'no-thinking'], default=['thinking'])
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
here = Path(__file__).parent
production = ROOT / 'deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json'
with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
    initial = json.load(response)
assert initial['scheduler']['active'] == initial['scheduler']['queued'] == initial['service']['pending_admissions'] == 0
try:
    run(['python3', 'scripts/dgpp-cluster', 'down', '--config', production], args.out / 'production-down.log')
    for label, binary, early in [('candidate', args.candidate, True),
                                 ('candidate-deferred', args.candidate, False),
                                 ('control', args.control, False)]:
        directory = args.out / label
        directory.mkdir()
        config = json.loads((here / 'qwen-w1.json').read_text())
        config['world_size'] = 4
        config['model'] = 'Qwen/Qwen3.8-Flash-Next-FP8'
        config['engine']['queue_limit'] = 8
        config['engine']['prefill_early_admission'] = early
        if args.logprobs:
            config['engine']['reasoning_in_content'] = True
        path = directory / 'deployment.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        (directory / 'binary.sha256').write_text(hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest() + '\n')
        try:
            run(['python3', 'scripts/dgpp-cluster', 'up', '--config', path,
                 '--bin', binary, '--log-dir', directory / 'world'], directory / 'up.log')
            capture(directory, 'before')
            for variant in args.variants:
                options = ['--no-thinking'] if variant == 'no-thinking' else []
                if args.logprobs:
                    options.append('--logprobs')
                run(['python3', here / 'mixed_cache_check.py', '--diagnostic', *options,
                     '--out', directory / f'{variant}.json'], directory / f'{variant}.log', check=False)
            capture(directory, 'after')
        finally:
            run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
                 '--log-dir', directory / 'world'], directory / 'down.log', check=False)
            hashes = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                      for p in sorted((directory / 'world').glob('serve_rank*.ops'))}
            match = len(hashes) == 4 and len(set(hashes.values())) == 1
            (directory / 'opstreams.json').write_text(json.dumps({'hashes': hashes, 'match': match}, indent=2) + '\n')
        assert match
finally:
    run(['python3', 'scripts/dgpp-cluster', 'up', '--config', production,
         '--bin', 'build-release/dgpp-serve'], args.out / 'production-restore.log')
