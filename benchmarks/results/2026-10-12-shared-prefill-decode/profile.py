"""Profile the MTP-off idle burst separately from the uninstrumented timings."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import urllib.request

from campaign import ROOT, run, capture
sys.path.insert(0, str(ROOT / 'scripts'))
from serve_admission_bench import Bench

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--candidate', type=Path, required=True)
ap.add_argument('--control', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
production = ROOT / 'deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json'
with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
    initial = json.load(response)
assert initial['scheduler']['active'] == initial['scheduler']['queued'] == initial['service']['pending_admissions'] == 0
try:
    run(['python3', 'scripts/dgpp-cluster', 'down', '--config', production], args.out / 'production-down.log')
    for label, binary, early in [('candidate-deferred', args.candidate, False),
                                 ('candidate', args.candidate, True), ('control', args.control, False)]:
        directory = args.out / label
        directory.mkdir()
        config = json.loads((Path(__file__).parent / 'qwen-w1.json').read_text())
        config['engine'].update(prefill_early_admission=early, mtp=False, mtp_depth=1, compact_batches=True)
        path = directory / 'deployment.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        (directory / 'binary.sha256').write_text(hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest() + '\n')
        session = 'dgpp-final-mtp-off-' + label
        try:
            run(['python3', 'scripts/dgpp-cluster', 'up', '--config', path,
                 '--bin', binary, '--log-dir', directory / 'world', '--head-wrap',
                 f'nsys launch --session-new={session} --trace=cuda --cuda-graph-trace=node'], directory / 'up.log')
            bench = Bench(argparse.Namespace(url='http://127.0.0.1:18081', timeout=600,
                seed=104, prompt_words=8192, decode_tokens=128, arrival_tokens=128, arrivals=8, interval=.25))
            warmup = bench.run('sustained_idle', 0)
            (directory / 'warmup.json').write_text(json.dumps(warmup, indent=2) + '\n')
            capture(directory, 'before', profile=True)
            run(['nsys', 'start', '--session', session, '--sample=none', '--cpuctxsw=none',
                 '--output', directory / 'r0'], directory / 'profile-start.log')
            result = bench.run('sustained_idle', 0)
            (directory / 'profiled-work.json').write_text(json.dumps(result, indent=2) + '\n')
            run(['nsys', 'stop', '--session', session], directory / 'profile-stop.log')
            capture(directory, 'after', profile=True)
        finally:
            run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
                 '--log-dir', directory / 'world'], directory / 'down.log', check=False)
            run(['nsys', 'shutdown', '--session', session, '--kill=none'], directory / 'profile-shutdown.log', check=False)
        run(['nsys', 'export', '--type=sqlite', '--output', directory / 'r0.sqlite',
             directory / 'r0.nsys-rep'], directory / 'export.log')
        with sqlite3.connect(directory / 'r0.sqlite') as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('''SELECT s.value AS kernel, k.graphNodeId != 0 AS graph,
              k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
              COUNT(*) AS calls, SUM(k.end-k.start)/1e6 AS total_ms,
              AVG(k.end-k.start)/1e3 AS mean_us
              FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
              GROUP BY s.value, graph, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
              ORDER BY total_ms DESC''').fetchall()
            (directory / 'kernels.json').write_text(json.dumps([dict(row) for row in rows], indent=2) + '\n')
        print(label, 'profile complete', flush=True)
finally:
    run(['python3', 'scripts/dgpp-cluster', 'up', '--config', production,
         '--bin', 'build-release/dgpp-serve'], args.out / 'production-restore.log')
