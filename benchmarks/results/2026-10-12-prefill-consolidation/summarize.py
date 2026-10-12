"""Summarize a matched campaign without dropping failed or changed outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = {label: read(args.campaign / label / 'admission.json') for label in ('control', 'candidate')}
    runs = {label: {(r['scenario'], r['trial']): r for r in value['runs']} for label, value in data.items()}
    assert runs['control'].keys() == runs['candidate'].keys(), 'unmatched scenarios/trials'
    records = []
    for key, baseline in runs['control'].items():
        candidate = runs['candidate'][key]
        requests = []
        left = {r['name']: r for r in baseline['requests']}
        right = {r['name']: r for r in candidate['requests']}
        assert left.keys() == right.keys(), 'unmatched requests'
        for name, before in left.items():
            after = right[name]
            assert before['body_sha256'] == after['body_sha256'], 'request inputs differ'
            requests.append(dict(name=name, input_sha256=before['body_sha256'],
                control_output_sha256=before['output_sha256'], candidate_output_sha256=after['output_sha256'],
                output_matches=before['output_sha256'] == after['output_sha256'],
                control_ttft_s=before['ttft_s'], candidate_ttft_s=after['ttft_s'],
                control_usage=before.get('usage'), candidate_usage=after.get('usage')))
        assert baseline['computed_prompt_tokens'] == candidate['computed_prompt_tokens'], 'prompt work differs'
        records.append(dict(scenario=key[0], trial=key[1], control_wall_s=baseline['wall_s'],
            candidate_wall_s=candidate['wall_s'], wall_change_percent=100 * (candidate['wall_s'] / baseline['wall_s'] - 1),
            computed_prompt_tokens=baseline['computed_prompt_tokens'],
            control_generated_tokens=baseline['generated_tokens'], candidate_generated_tokens=candidate['generated_tokens'],
            requests=requests))
    means = []
    for scenario in sorted({r['scenario'] for r in records}):
        selected = [r for r in records if r['scenario'] == scenario]
        before = statistics.mean(r['control_wall_s'] for r in selected)
        after = statistics.mean(r['candidate_wall_s'] for r in selected)
        entry = dict(scenario=scenario, trials=len(selected), control_wall_s=before, candidate_wall_s=after,
                     wall_change_percent=100 * (after / before - 1))
        means.append(entry)
        print(f'{scenario:16s} {before:9.3f} -> {after:9.3f} s  {entry["wall_change_percent"]:+7.2f}%')
    metadata = {}
    for label in data:
        directory = args.campaign / label
        metadata[label] = dict(binary_sha256=(directory / 'binary.sha256').read_text().strip(),
            config=read(directory / 'deployment.json'), arguments=data[label]['arguments'],
            opstreams=read(directory / 'opstreams.json'),
            nodes_before=read(directory / 'nodes-before.json'), nodes_after=read(directory / 'nodes-after.json'),
            transcripts=read(directory / 'transcripts.json'),
            evidence_sha256=hashlib.sha256((directory / 'admission.json').read_bytes()).hexdigest())
        assert metadata[label]['opstreams']['match'], 'rank operation streams differ'
        cache_path = directory / 'mixed-cache.json'
        if cache_path.exists():
            cache = read(cache_path)
            actual = cache['mixed']
            metadata[label]['cache_check'] = dict(
                transcript_matches=cache['transcript_matches'],
                first_token_during_prefill=cache.get('first_token_during_prefill'),
                finishes_during_prefill=cache['finishes_during_prefill'],
                cached_ttft_s=(actual['first_token'] - actual['start']) if 'first_token' in actual else None,
                cached_wall_s=actual['end'] - actual['start'], cold_ttft_s=cache['long']['ttft_s'],
                decoder_wall_s=cache['decoder']['wall_s'],
                pool_blocks_before=cache['before']['scheduler']['pool_blocks_in_use'],
                pool_blocks_after=cache['after']['scheduler']['pool_blocks_in_use'])
    chat = {}
    for label, before in metadata['control']['transcripts']['results'].items():
        after = metadata['candidate']['transcripts']['results'][label]
        fields = ('reasoning', 'content', 'finish', 'completion_tokens', 'prompt_tokens')
        chat[label] = dict(matches=all(before[k] == after[k] for k in fields),
                           control=before, candidate=after)
    args.out.write_text(json.dumps(dict(campaign=str(args.campaign), deployments=metadata,
                                        means=means, pairs=records, chat_comparisons=chat), indent=2) + '\n')
    requests = [r for pair in records for r in pair['requests']]
    print(f'Exact response hashes: {sum(r["output_matches"] for r in requests)}/{len(requests)}')
    print(f'Exact chat transcripts: {sum(r["matches"] for r in chat.values())}/{len(chat)}')


if __name__ == '__main__':
    main()
