"""Reuse a cached conversation while an incumbent and a cold prefill run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
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
ap.add_argument('--diagnostic', action='store_true', help='record transcript/overlap differences without accepting the gate')
ap.add_argument('--no-thinking', action='store_true')
ap.add_argument('--logprobs', action='store_true')
ap.add_argument('--decoder-only', action='store_true', help='diagnose batching without a cold reader')
args = ap.parse_args()
if args.decoder_only and not args.diagnostic:
    ap.error('--decoder-only requires --diagnostic; it is not an early-admission gate')
args.timeout = 600
bench = Bench(args)
probe_arguments = {k: getattr(args, k) for k in ('diagnostic', 'no_thinking', 'logprobs', 'decoder_only')}


def cached(prompt):
    body = {'model': bench.model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 128,
            'temperature': 0, 'ignore_eos': True, 'prefix_cache': True, 'stream': True, 'stream_options': {'include_usage': True}}
    if args.no_thinking:
        body['chat_template_kwargs'] = {'enable_thinking': False}
    if args.logprobs:
        body.update(logprobs=True, top_logprobs=5)
    started = time.monotonic()
    with closing(bench.connection()) as connection:
        connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 200, (response.status, response.read())
        data = {'choices': [{'message': {'role': 'assistant', 'content': '', 'reasoning_content': ''}}]}
        message = data['choices'][0]['message']
        first_token = None
        done = False
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            payload = raw[6:].strip()
            if payload == b'[DONE]':
                done = True
                break
            event = json.loads(payload)
            assert 'error' not in event, event
            if event.get('id'):
                data['id'] = event['id']
            for field in ('usage', 'timings'):
                if event.get(field):
                    data[field] = event[field]
            for choice in event.get('choices', []):
                if choice.get('logprobs'):
                    data['choices'][0].setdefault('logprobs', []).append(choice['logprobs'])
                delta = choice.get('delta', {})
                for field in ('content', 'reasoning_content'):
                    text = delta.get(field) or ''
                    if text:
                        first_token = first_token or time.monotonic()
                        message[field] += text
                if choice.get('finish_reason'):
                    data['choices'][0]['finish_reason'] = choice['finish_reason']
        assert done and first_token is not None and 'usage' in data
    return {'start': started, 'first_token': first_token, 'end': time.monotonic(), 'response': data}


initial = bench.idle()
prompt = 'Reference notes: ' + 'Alpha beta gamma delta. ' * 400
prompt += '\nList the positive integers in order, starting with one:\n1,'
warm = cached(prompt)
reference = cached(prompt)
args.out.write_text(json.dumps({'arguments': probe_arguments, 'warm': warm, 'reference': reference}, indent=2) + '\n')
assert reference['response']['usage']['prompt_tokens_details']['cached_tokens'] > 0
before = bench.idle(min_terminal=initial['scheduler']['terminal'] + 2)
rng = random.Random(104)
long_prompt = 'Document: ' + ' '.join(rng.choice('amber birch cobalt dune ember fjord granite harbor indigo juniper quartz willow'.split()) for _ in range(8192)) + '\nSummary:'
with ThreadPoolExecutor(max_workers=3) as pool:
    ready = threading.Event()
    decoder = pool.submit(bench.stream, 'decoder', 'List the positive integers in order, starting with one:\n1,', 512, ready)
    assert ready.wait(600)
    long = None
    observed = []
    if not args.decoder_only:
        long = pool.submit(bench.stream, 'long', long_prompt, 1)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            observed = bench.get('/metrics')['prefill']['requests']
            if observed:
                break
            if long.done():
                raise RuntimeError('long prefill ended before the cache probe')
            time.sleep(.01)
        assert observed
    actual = cached(prompt)
    long_record, decoder_record = long.result() if long else None, decoder.result()
requests = 2 if args.decoder_only else 3
after = bench.idle(min_terminal=before['scheduler']['terminal'] + requests)
same = actual['response']['choices'][0]['message'] == reference['response']['choices'][0]['message']
# A one-token cold probe may return only a special token, with no visible
# update. Its completed stream still bounds when prefill finished.
cold_output = (long_record['updates'][0] if long_record['updates'] else long_record['end']) if long_record else None
overlap = actual['end'] < cold_output if cold_output is not None else None
# Admission bounds time to first output. A 128-token reply can legitimately
# outlast a cold prefill that now completes faster; retain that measurement
# separately instead of treating completion order as evidence of deferral.
first_token_during_prefill = actual['first_token'] < cold_output if cold_output is not None else None
args.out.write_text(json.dumps({'arguments': probe_arguments, 'warm': warm, 'reference': reference, 'mixed': actual,
    'long': long_record, 'decoder': decoder_record, 'observed': observed,
    'before': before, 'after': after, 'transcript_matches': same, 'finishes_during_prefill': overlap, 'first_token_during_prefill': first_token_during_prefill}, indent=2) + '\n')
assert actual['response']['usage']['completion_tokens'] == 128
assert actual['response']['usage']['prompt_tokens_details']['cached_tokens'] > 0
if long_record:
    assert long_record['id'] in {r['id'] for r in observed}
assert after['service']['requests_total'] - before['service']['requests_total'] == requests
# The entry meter sums references: two snapshots may pin the same physical
# blocks. This exact repeat must return to its warmed physical-block count.
assert after['scheduler']['pool_blocks_in_use'] <= after['prefix_cache']['blocks_pinned']
if args.diagnostic:
    # A baseline transcript change can create a distinct cached continuation.
    # Record its additional retained blocks rather than calling them a leak
    # or preventing the candidate comparison from running.
    print('DIAGNOSTIC ONLY: transcript_matches =', same, '; finishes_during_prefill =', overlap,
          '; retained blocks =', before['scheduler']['pool_blocks_in_use'], '->',
          after['scheduler']['pool_blocks_in_use'])
    sys.exit(0)
assert after['scheduler']['pool_blocks_in_use'] == before['scheduler']['pool_blocks_in_use']
assert same, 'cached transcript differs under load'
assert first_token_during_prefill, 'cached request waited for the entire cold prefill'
print('Cached request produced output during cold prefill; transcript matches; warmed physical KV usage unchanged; full response finished during prefill:', overlap)
