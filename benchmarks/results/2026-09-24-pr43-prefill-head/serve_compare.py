import http.client
import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
conn = http.client.HTTPConnection('127.0.0.1', 18081, timeout=600)
conn.request('GET', '/v1/models')
response = conn.getresponse()
assert response.status == 200
model = json.loads(response.read())['data'][0]['id']
conn.close()
results = []
for name, prompt in [
    ('short', 'The capital of France is'),
    ('over_128', 'red green blue ' * 64 + 'The next color is'),
    ('long', 'Below is a list of colors:\n' + 'red green blue ' * 1400 + '\nQuestion: What are the three colors in the list?\nAnswer:'),
    ('chunked', 'Below is a list of colors:\n' + 'red green blue ' * 2800 + '\nQuestion: What are the three colors in the list?\nAnswer:'),
]:
    body = {'model': model, 'prompt': prompt, 'max_tokens': 32,
            'temperature': 0, 'seed': 43, 'ignore_eos': True, 'logprobs': 5, 'stream': True,
            'stream_options': {'include_usage': True}}
    conn = http.client.HTTPConnection('127.0.0.1', 18081, timeout=600)
    start = time.perf_counter()
    conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError((response.status, response.read().decode()))
    result = {'case': name, 'text': '', 'logprobs': [], 'usage': None, 'finish_reason': None}
    first = None
    for raw in response:
        line = raw.decode('utf-8').strip()
        if not line.startswith('data: '):
            continue
        data = line[6:]
        if data == '[DONE]':
            break
        event = json.loads(data)
        if event.get('usage'):
            result['usage'] = event['usage']
        for choice in event.get('choices', []):
            text = choice.get('text', '')
            if text and first is None:
                first = time.perf_counter()
            result['text'] += text
            if choice.get('logprobs'):
                lp = choice['logprobs']
                result['logprobs'].extend(zip(lp['tokens'], lp['token_logprobs'], lp['top_logprobs']))
            if choice.get('finish_reason'):
                result['finish_reason'] = choice['finish_reason']
    conn.close()
    result['ttft_ms'] = (first - start) * 1000 if first is not None else None
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000
    assert result['usage'] and result['finish_reason'] and result['logprobs'], result
    results.append(result)
    out.write_text(json.dumps(results, indent=2) + '\n')
    print(name, result['usage']['prompt_tokens'], result['usage']['completion_tokens'],
          f"TTFT {result['ttft_ms']:.2f} ms", flush=True)
