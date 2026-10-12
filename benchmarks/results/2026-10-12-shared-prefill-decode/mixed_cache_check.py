"""Reuse a cached conversation while an incumbent and a cold prefill run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from serve_admission_bench import Bench

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--url', default='http://127.0.0.1:18081')
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.timeout = 600
bench = Bench(args)


def cached(prompt):
    body = {'model': bench.model, 'prompt': prompt, 'max_tokens': 16,
            'temperature': 0, 'ignore_eos': True, 'prefix_cache': True, 'stream': False}
    started = time.monotonic()
    with closing(bench.connection()) as connection:
        connection.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 200, (response.status, response.read())
        data = json.load(response)
    return {'start': started, 'end': time.monotonic(), 'response': data}


initial = bench.idle()
prompt = 'Reference notes: ' + 'Alpha beta gamma delta. ' * 400
prompt += '\nList the positive integers in order, starting with one:\n1,'
warm = cached(prompt)
reference = cached(prompt)
assert reference['response']['usage']['prompt_tokens_details']['cached_tokens'] > 0
before = bench.idle(min_terminal=initial['scheduler']['terminal'] + 2)
rng = random.Random(104)
long_prompt = 'Document: ' + ' '.join(rng.choice('amber birch cobalt dune ember fjord granite harbor indigo juniper quartz willow'.split()) for _ in range(8192)) + '\nSummary:'
with ThreadPoolExecutor(max_workers=3) as pool:
    ready = threading.Event()
    decoder = pool.submit(bench.stream, 'decoder', 'List the positive integers in order, starting with one:\n1,', 512, ready)
    assert ready.wait(600)
    long = pool.submit(bench.stream, 'long', long_prompt, 1)
    deadline = time.monotonic() + 600
    observed = []
    while time.monotonic() < deadline:
        observed = bench.get('/metrics')['prefill']['requests']
        if observed:
            break
        if long.done():
            raise RuntimeError('long prefill ended before the cache probe')
        time.sleep(.01)
    assert observed
    actual = cached(prompt)
    long_record, decoder_record = long.result(), decoder.result()
after = bench.idle(min_terminal=before['scheduler']['terminal'] + 3)
assert actual['response']['choices'][0]['text'] == reference['response']['choices'][0]['text']
assert actual['response']['usage']['completion_tokens'] == 16
assert actual['response']['usage']['prompt_tokens_details']['cached_tokens'] > 0
assert actual['end'] < long_record['updates'][0], 'cached request waited for the entire cold prefill'
assert long_record['id'] in {r['id'] for r in observed}
assert after['service']['requests_total'] - before['service']['requests_total'] == 3
assert after['scheduler']['pool_blocks_in_use'] == after['prefix_cache']['blocks_pinned']
args.out.write_text(json.dumps({'warm': warm, 'reference': reference, 'mixed': actual,
    'long': long_record, 'decoder': decoder_record, 'observed': observed,
    'before': before, 'after': after}, indent=2) + '\n')
print('Cached request finished during cold prefill; transcript matches; idle pool equals pinned prefix blocks')
