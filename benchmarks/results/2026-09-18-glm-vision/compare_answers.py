#!/usr/bin/env python3
"""Compare preserved native/reference responses from the vision harness.

Reports answer equality and correctness separately from token logprob changes.
Only aligns visible-token logprobs when token/byte sequences and reasoning match.
"""
import argparse
import json
from pathlib import Path
import re
import statistics


def compare(native_path, reference_path, diagrams):
    def load(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        keyed = {}
        for row in rows:
            key = str(row['index']) if diagrams else row['fixture'] + ': ' + row['question']
            if key in keyed:
                raise ValueError('duplicate case ' + key)
            keyed[key] = row
        return keyed
    native, reference = load(native_path), load(reference_path)
    if native.keys() != reference.keys():
        raise ValueError('case sets differ')
    result = {'cases': len(native), 'content_equal': 0, 'reasoning_equal': 0,
              'changed_answers': [], 'identical_token_sequences': 0}
    if diagrams:
        result.update(native_correct=0, reference_correct=0, normalized_equal=0)
    deltas = []
    for key, row in native.items():
        other = reference[key]
        a, b = (entry['response']['choices'][0] for entry in (row, other))
        ca, cb = (entry['message']['content'].strip() for entry in (a, b))
        result['content_equal'] += ca == cb
        result['reasoning_equal'] += a['message'].get('reasoning_content') == b['message'].get('reasoning_content')
        if ca != cb:
            result['changed_answers'].append({'case': key, 'native': ca, 'reference': cb})
        if diagrams:
            def option(content):
                found = re.match(r'^\s*(?:\*\*)?([1-4])(?:\*\*)?(?=[\s.]|$)', content)
                return found[1] if found else None
            if row['gold'] != other['gold']:
                raise ValueError('gold labels differ')
            aa, bb = option(ca), option(cb)
            result['native_correct'] += aa == row['gold']
            result['reference_correct'] += bb == row['gold']
            result['normalized_equal'] += aa is not None and aa == bb
        la, lb = (entry.get('logprobs', {}).get('content', []) for entry in (a, b))
        # The API supplies visible token strings and bytes. Require matching
        # reasoning too, so the scored answer has the same generated context.
        identity = lambda rows: [(x['token'], x.get('bytes')) for x in rows]
        if (la and identity(la) == identity(lb)
                and a['message'].get('reasoning_content') == b['message'].get('reasoning_content')):
            result['identical_token_sequences'] += 1
            deltas.extend(x['logprob'] - y['logprob'] for x, y in zip(la, lb))
    result['aligned_tokens'] = len(deltas)
    if deltas:
        result.update(mean_logprob_delta=statistics.mean(deltas),
                      max_abs_logprob_delta=max(abs(x) for x in deltas))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('native', type=Path)
    ap.add_argument('reference', type=Path)
    ap.add_argument('--diagrams', action='store_true')
    args = ap.parse_args()
    print(json.dumps(compare(args.native, args.reference, args.diagrams), indent=2))


if __name__ == '__main__':
    main()
