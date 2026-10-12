"""Record top-five token probabilities for selected campaign workloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from campaign import Checks

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--url', default='http://127.0.0.1:18081')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--words', type=int, default=4096)
ap.add_argument('--seed', type=int, default=104)
ap.add_argument('--trials', type=int, default=2)
ap.add_argument('--scenarios', nargs='+', default=['solo_prefill', 'batch_prefill'])
args = ap.parse_args()
args.timeout = 600
args.prompt_words = args.words
args.decode_tokens = 128
args.arrival_tokens = 64
args.arrivals = 3
args.interval = .25
args.record_text = True
args.logprobs = True
bench = Checks(args)


def completion(prompt):
    body = dict(model=bench.model, prompt=prompt, max_tokens=1,
                temperature=0, ignore_eos=True, prefix_cache=False, logprobs=5)
    encoded = json.dumps(body).encode()
    with closing(bench.connection()) as connection:
        connection.request('POST', '/v1/completions', encoded,
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 200, (response.status, response.read())
        data = json.load(response)
    assert data['usage']['completion_tokens'] == 1
    assert not data['usage']['prompt_tokens_details']['cached_tokens']
    assert data['choices'][0]['logprobs']
    return dict(input_sha256=hashlib.sha256(encoded).hexdigest(), response=data)


results = []
if 'batch_decode' in args.scenarios:
    before = bench.idle()
    responses = [bench.stream(f'batch-{n}',
        f'Task {n}. List the positive integers in order:\n1,', args.decode_tokens)
        for n in range(4)]
    after = bench.idle(before['scheduler']['terminal'] + len(responses))
    results.append(dict(scenario='isolated_batch_decode', trial=0,
                        requests=responses, before=before, after=after))
    args.out.write_text(json.dumps(results, indent=2) + '\n')
    print('isolated_batch_decode', [r['output_text'] for r in responses], flush=True)
for trial in range(args.trials):
    rng = random.Random(args.seed + trial)
    vocabulary = 'amber birch cobalt dune ember fjord granite harbor indigo juniper quartz willow'.split()
    prompt = 'Document:\n' + ' '.join(rng.choice(vocabulary) for _ in range(args.words)) + '\nSummary:'
    for name in args.scenarios:
        if name == 'batch_decode':
            results.append(bench.batch(name, trial))
            args.out.write_text(json.dumps(results, indent=2) + '\n')
            print(name, trial, [r['output_text'] for r in results[-1]['requests']], flush=True)
            continue
        if name == 'decoder_pair':
            before = bench.idle()
            short = 'List the positive integers in order, starting with one:\n1,'
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda n: bench.stream(f'decoder-{n}', short, args.decode_tokens), range(2)))
            after = bench.idle(before['scheduler']['terminal'] + 2)
            assert after['service']['requests_total'] - before['service']['requests_total'] == 2
            assert after['scheduler']['pool_blocks_in_use'] <= before['scheduler']['pool_blocks_in_use']
            results.append(dict(scenario=name, trial=trial, requests=responses, before=before, after=after))
            args.out.write_text(json.dumps(results, indent=2) + '\n')
            print(name, trial, [r['output_text'] for r in responses], flush=True)
            continue
        if name not in ('solo_prefill', 'batch_prefill'):
            results.append(bench.run(name, trial))
            args.out.write_text(json.dumps(results, indent=2) + '\n')
            print(name, trial, [r['output_text'] for r in results[-1]['requests']], flush=True)
            continue
        prompts = [prompt] if name == 'solo_prefill' else [f'Task {n}. {prompt}' for n in range(4)]
        before = bench.idle()
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            responses = list(pool.map(completion, prompts))
        after = bench.idle(before['scheduler']['terminal'] + len(prompts))
        assert after['service']['requests_total'] - before['service']['requests_total'] == len(prompts)
        assert after['scheduler']['pool_blocks_in_use'] <= before['scheduler']['pool_blocks_in_use']
        results.append(dict(scenario=name, trial=trial, responses=responses,
                            before=before, after=after))
        args.out.write_text(json.dumps(results, indent=2) + '\n')
        print(name, trial, [r['response']['choices'][0]['text'] for r in responses], flush=True)
