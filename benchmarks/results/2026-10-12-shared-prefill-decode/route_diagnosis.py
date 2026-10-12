"""Record final-layer decoder routing with the separate diagnostic patch.

The patch replaces DGPP_QWEN_HEAD_DUMP payloads with compact int32 expert IDs.
These instrumented runs are not performance measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

from campaign import ROOT, ENV, run, capture, Bench

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--binary', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
production = ROOT / 'deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json'
with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
    initial = json.load(response)
assert initial['scheduler']['active'] == initial['scheduler']['queued'] == initial['service']['pending_admissions'] == 0
try:
    run(['python3', 'scripts/dgpp-cluster', 'down', '--config', production], args.out / 'production-down.log')
    for label, early in [('candidate-deferred', False), ('candidate', True)]:
        directory = args.out / label
        directory.mkdir()
        config = json.loads((Path(__file__).parent / 'qwen-w1.json').read_text())
        config['engine'].update(prefill_early_admission=early, mtp=False, mtp_depth=1, compact_batches=True)
        path = directory / 'deployment.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        (directory / 'binary.sha256').write_text(hashlib.file_digest(args.binary.open('rb'), 'sha256').hexdigest() + '\n')
        raw = (directory / 'routes.bin').resolve()
        ENV['DGPP_QWEN_HEAD_DUMP'] = str(raw)
        try:
            run(['python3', 'scripts/dgpp-cluster', 'up', '--config', path,
                 '--bin', args.binary, '--log-dir', directory / 'world'], directory / 'up.log')
            bench = Bench(argparse.Namespace(url='http://127.0.0.1:18081', timeout=600,
                seed=104, prompt_words=8192, decode_tokens=128, arrival_tokens=128, arrivals=8, interval=.25))
            bench.stream('warmup', 'Continue counting: 1, 2, 3,', 32)
            bench.idle()
            capture(directory, 'before')
            raw.unlink(missing_ok=True)
            result = bench.run('sustained_idle', 0)
            (directory / 'diagnostic-work.json').write_text(json.dumps(result, indent=2) + '\n')
            capture(directory, 'after')
        finally:
            ENV.pop('DGPP_QWEN_HEAD_DUMP', None)
            run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
                 '--log-dir', directory / 'world'], directory / 'down.log', check=False)
        data = raw.read_bytes()
        records, offset = [], 0
        while offset < len(data):
            rows, hidden, site = struct.unpack_from('<iii', data, offset)
            assert 0 < rows <= 4 and hidden == 2560 and site == 0
            offset += 12
            ids = list(struct.unpack_from('<' + 'i' * (rows * 10), data, offset))
            assert all(0 <= x < 512 for x in ids)
            records.append({'rows': rows, 'ids': ids, 'unique_experts': len(set(ids))})
            offset += rows * hidden * 2
        assert offset == len(data)
        (directory / 'routes.json').write_text(json.dumps(records, indent=2) + '\n')
        four = [r for r in records if r['rows'] == 4]
        print(label, len(four), 'four-row batches; mean unique experts',
              sum(r['unique_experts'] for r in four) / len(four), flush=True)
finally:
    ENV.pop('DGPP_QWEN_HEAD_DUMP', None)
    run(['python3', 'scripts/dgpp-cluster', 'up', '--config', production,
         '--bin', 'build-release/dgpp-serve'], args.out / 'production-restore.log')
