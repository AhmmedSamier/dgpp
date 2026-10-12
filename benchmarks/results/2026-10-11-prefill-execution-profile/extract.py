#!/usr/bin/env python3
"""Extract admission timings and kernel activity from the four Nsight captures.

Usage: python3 extract.py /path/to/build-pr104-shared > measurements.json
Input reports must cover only the measured workload, after warmup and graph
capture. Eager/graph activity is an execution classification, not NVTX ranges.
Kernel-duration sums include overlapping streams and are not elapsed time.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def union_ns(intervals):
    total, end = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def trace(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        captures = con.execute('''
            SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId
            WHERE s.value LIKE '%StreamBeginCapture%'
        ''').fetchone()[0]
        if captures:
            raise ValueError(f'{path}: capture-time work contaminated the measured range')
        rows = con.execute('''
            SELECT k.start,k.end,COALESCE(k.graphNodeId,0),s.value,k.deviceId
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start
        ''').fetchall()
    if not rows or len({r[4] for r in rows}) != 1:
        raise ValueError(f'{path}: expected a nonempty single-device trace')
    phases = {}
    for phase in ('eager', 'decode_graph'):
        selected = [r for r in rows if bool(r[2]) == (phase == 'decode_graph')]
        kernels = {}
        for start, end, _, name, _ in selected:
            item = kernels.setdefault(name, {'name': name, 'instances': 0, 'duration_ns': 0})
            item['instances'] += 1
            item['duration_ns'] += end - start
        phases[phase] = {
            'kernel_instances': len(selected),
            'kernel_duration_sum_ms': sum(r[1] - r[0] for r in selected) / 1e6,
            'kernel_union_ms': union_ns((r[0], r[1]) for r in selected) / 1e6,
            'kernels': sorted(kernels.values(), key=lambda k: (-k['duration_ns'], k['name'])),
        }
    first, last = min(r[0] for r in rows), max(r[1] for r in rows)
    busy = union_ns((r[0], r[1]) for r in rows)
    return {'first_to_last_kernel_ms': (last-first)/1e6,
            'kernel_union_ms': busy/1e6,
            'no_kernel_running_ms': (last-first-busy)/1e6,
            'phases': phases}


def extract(directory):
    records = []
    for idle, early in ((256, False), (256, True), (2048, True), (2048, False)):
        case = directory / f'profile-w1-{idle}-early{int(early)}'
        admission = json.loads((case/'admission.json').read_text())
        run = admission['runs'][0]
        before, after = (run[k]['scheduler'] for k in ('before', 'after'))
        fields = ('prefill_ms', 'step_ms', 'decode_steps', 'prompt_tokens_computed', 'tokens_generated')
        records.append({
            'label': case.name,
            'binary_sha256': (case/'binary.sha256').read_text().strip(),
            'configuration': json.loads((case/'deployment.json').read_text()),
            'arguments': admission['arguments'],
            'native_delta': {k: after[k]-before[k] for k in fields},
            'wall_s': run['wall_s'],
            'requests': run['requests'],
            'trace': trace(case/'r0.sqlite'),
            'input_sha256': {name: digest(case/name) for name in
                             ('admission.json', 'r0.nsys-rep', 'r0.sqlite')},
        })
    by_name = {r['label']: r for r in records}
    for idle in (256, 2048):
        off, on = (by_name[f'profile-w1-{idle}-early{i}'] for i in (0, 1))
        for a, b in zip(off['requests'], on['requests'], strict=True):
            for field in ('name', 'body_sha256', 'output_sha256', 'usage'):
                if a[field] != b[field]:
                    raise ValueError(f'{idle}: {a["name"]} differs in {field}')
    return {'matched_request_bodies_outputs_usage': True, 'runs': records}


if __name__ == '__main__':
    print(json.dumps(extract(Path(sys.argv[1])), indent=2, allow_nan=False))
