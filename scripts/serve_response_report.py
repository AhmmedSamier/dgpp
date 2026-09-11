"""Summarize saved non-streaming tool and structured-output responses."""
import argparse
import json
from pathlib import Path


def report(path):
    path = Path(path)
    raw = path.read_text()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        print(raw[:800])
        return
    if 'error' in d:
        print('ERROR', d['error'])
        return
    ch = d['choices'][0]
    m = ch['message']
    print('finish:', ch['finish_reason'], 'usage:', d['usage'])
    print('reasoning_content:', repr((m.get('reasoning_content') or '')[:400]))
    print('content:', repr(m.get('content')))
    for tc in m.get('tool_calls') or []:
        print('tool_call:', tc['id'], tc['function']['name'], tc['function']['arguments'])
    if path.name.startswith('json_'):
        try:
            obj = json.loads(m.get('content') or '')
            print('json: parses', type(obj).__name__, sorted(obj) if isinstance(obj, dict) else '')
        except json.JSONDecodeError as e:
            print('json: DOES NOT PARSE:', e)
            obj = None
        if path.name.endswith('json_schema.json') and isinstance(obj, dict):
            ok = (set(obj) <= {'city', 'country', 'population', 'landmarks'} and
                  isinstance(obj.get('city'), str) and isinstance(obj.get('country'), str) and
                  isinstance(obj.get('population'), int) and not isinstance(obj.get('population'), bool) and
                  isinstance(obj.get('landmarks'), list) and 1 <= len(obj['landmarks']) <= 3 and
                  all(isinstance(x, str) for x in obj['landmarks']))
            print('schema:', 'CONFORMS' if ok else 'VIOLATES')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    report(parser.parse_args().path)


if __name__ == "__main__":
    main()
