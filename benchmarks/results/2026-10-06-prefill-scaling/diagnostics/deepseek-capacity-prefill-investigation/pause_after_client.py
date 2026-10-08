"""Finish the active context client, then unwind the stopped controller cleanly."""
import datetime
import json
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
target = json.loads((HERE / 'pause-target.json').read_text())
deadline = time.monotonic() + 1800

def identity(pid, ticks):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[19] == ticks, 'PID identity changed'
    return fields

while time.monotonic() < deadline:
    parent = identity(target['controller_pid'], target['controller_start_ticks'])
    assert parent[0] in ['T', 't'], parent[0]
    child = identity(target['client_pid'], target['client_start_ticks'])
    if child[0] == 'Z':
        report = HERE.parents[1] / 'raw/deepseek-1m-w4/refresh/default/context-65536.json'
        data = json.loads(report.read_text())
        assert len(data['samples']) == 3
        record = {**target, 'client_finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'client_wait_status': int(child[-1]), 'samples': len(data['samples']),
                  'action': 'Send SIGINT to controller, then SIGCONT; runner finally block stops ranks and checks memory/rank identity.'}
        (HERE / 'client-boundary.json').write_text(json.dumps(record, indent=2) + '\n')
        os.kill(target['controller_pid'], signal.SIGINT)
        os.kill(target['controller_pid'], signal.SIGCONT)
        print(json.dumps(record), flush=True)
        break
    time.sleep(0.1)
else:
    raise RuntimeError('Timed out waiting for context client; controller remains stopped')
