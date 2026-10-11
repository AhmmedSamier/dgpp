"""MTP3 physical-reservation boundary through HTTP on an idle test server."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from serve_admission_bench import Bench

bench = Bench(SimpleNamespace(url='http://127.0.0.1:18081', timeout=120))
prompt = 'Count the integers in order, starting with one: 1,'
initial = bench.idle()
calibration = bench.stream('calibration', prompt, 1)
prompt_tokens = calibration['usage']['prompt_tokens']
before = bench.idle(min_terminal=initial['scheduler']['terminal'] + 1)
assert before['scheduler']['pool_blocks_in_use'] == 0, 'use a fresh cache-empty server'
assert before['scheduler']['spec_decode']['depth'] == 3, 'this probe requires native MTP3'
assert before['service']['admission']['mode'] == 'full', 'this probe requires full reservation'
total = before['scheduler']['pool_blocks_total']
assert total == 2048, 'this probe requires the recorded 131072-token Qwen pool'
held_blocks = total // 4
tail, block_tokens = 2, 64  # native MTP3: depth - 1 trailing rows
holder_limit = held_blocks * block_tokens - tail - prompt_tokens
contender_limit = held_blocks * block_tokens - prompt_tokens
ready = [threading.Event() for _ in range(4)]
cancel = [threading.Event() for _ in range(4)]
records = [{} for _ in range(4)]

def stream(index, limit):
    body = {'model': bench.model, 'prompt': prompt, 'temperature': 0,
            'ignore_eos': True, 'prefix_cache': False, 'stream': True,
            'max_tokens': limit}
    with closing(bench.connection()) as conn:
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        assert response.status == 200, (index, response.status, response.read())
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            payload = raw[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            assert 'error' not in event, event
            records[index]['id'] = event.get('id')
            if any(c.get('text') for c in event.get('choices', [])):
                ready[index].set()
            if cancel[index].is_set():
                break
    records[index]['cancelled'] = cancel[index].is_set()

def wait_for(predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        metrics = bench.get('/metrics')
        assert not metrics['service']['engine_failed'], metrics
        if predicate(metrics):
            return metrics
        time.sleep(.02)
    raise TimeoutError('reservation transition not observed')

futures = []
with ThreadPoolExecutor(max_workers=4) as pool:
    try:
        for index in range(3):
            futures.append(pool.submit(stream, index, holder_limit))
            assert ready[index].wait(30), 'holder did not start'
        held = wait_for(lambda m: m['scheduler']['active'] == 3 and
                        m['scheduler']['records'] == 3 and m['scheduler']['pool_blocks_in_use'] == 3 * held_blocks)
        assert held['scheduler']['pool_blocks_in_use'] == 3 * held_blocks, held
        futures.append(pool.submit(stream, 3, contender_limit))
        queued = wait_for(lambda m: m['scheduler']['queued'] == 1 and m['scheduler']['records'] == 4)
        assert queued['scheduler']['active'] == 3 and not ready[3].is_set(), queued
        assert queued['scheduler']['pool_blocks_in_use'] == 3 * held_blocks, queued
        cancel[0].set()
        futures[0].result(timeout=30)
        assert ready[3].wait(30), 'contender did not start after a holder released its reservation'
        admitted = wait_for(lambda m: m['scheduler']['queued'] == 0 and
                            m['scheduler']['pool_blocks_in_use'] == 3 * held_blocks + 1)
        assert admitted['scheduler']['pool_blocks_in_use'] == 3 * held_blocks + 1, admitted
    finally:
        for flag in cancel:
            flag.set()
        for future in futures:
            future.result(timeout=30)
after = bench.idle(min_terminal=before['scheduler']['terminal'] + 4)
assert after['scheduler']['pool_blocks_in_use'] == 0, after
assert after['service']['requests_total'] - before['service']['requests_total'] == 4, after
recovery = bench.stream('recovery', prompt, 4)
recovered = bench.idle(min_terminal=after['scheduler']['terminal'] + 1)
Path(sys.argv[1]).write_text(json.dumps({'prompt_tokens': prompt_tokens, 'holder_limit': holder_limit,
    'contender_limit': contender_limit, 'physical_tail': tail, 'block_tokens': block_tokens,
    'records': records, 'before': before, 'held': held, 'queued': queued, 'admitted': admitted,
    'after': after, 'recovery': recovery, 'recovered': recovered}, indent=2) + '\n')
print('MTP3 physical boundary: contender needs 513 blocks; defers with 512 free, admits after cancellation; zero blocks after drain')
