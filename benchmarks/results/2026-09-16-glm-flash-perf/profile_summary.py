import bisect
import collections
import json
import sqlite3
import statistics
import sys

db, slots = sys.argv[1], int(sys.argv[2])
con = sqlite3.connect(db)
rows = con.execute('SELECT k.start,k.end,s.value,k.gridY FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
marks = [r[0] for r in rows if 'publish_seq_kernel' in r[2]]
steps = [[] for _ in marks]
for row in rows:
    i = bisect.bisect_right(marks, row[0]) - 1
    if 0 <= i < len(marks) - 1:
        steps[i].append(row)
selected = []
for i, step in enumerate(steps[:-1]):
    gates = [r for r in step if 'moe_slot_gate_up_swiglu_fp4_kernel' in r[2]]
    if len(gates) == 42 and all(r[3] == slots for r in gates):
        selected.append(i)
median = statistics.median(marks[i+1]-marks[i] for i in selected)
selected = [i for i in selected if marks[i+1]-marks[i] < median*1.5]
by = collections.defaultdict(lambda: [0, 0])
busy = wall = 0
for i in selected:
    a, b = marks[i], marks[i+1]
    wall += b-a
    end = a
    for s, e, name, _ in steps[i]:
        busy += max(0, min(e, b)-max(s, end))
        end = max(end, min(e, b))
        for prefix in ('void ', 'dgpp::', '<unnamed>::', '(anonymous namespace)::', 'net::'):
            name = name.replace(prefix, '')
        depth = 0
        for j, ch in enumerate(name):
            if ch == '<': depth += 1
            elif ch == '>': depth -= 1
            elif ch == '(' and depth == 0:
                name = name[:j]
                break
        by[name][0] += min(e,b)-s
        by[name][1] += 1
n = len(selected)
print(json.dumps({'steps': n, 'slots': slots, 'wall_ms_per_step': wall/n/1e6,
                  'gpu_union_ms_per_step': busy/n/1e6,
                  'kernels': [{'name': k,'ms_per_step': v[0]/n/1e6,'us_per_call': v[0]/v[1]/1e3,'calls_per_step':v[1]/n}
                              for k,v in sorted(by.items(),key=lambda kv:-kv[1][0])]}, indent=2))
