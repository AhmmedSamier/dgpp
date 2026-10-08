"""Stop the exact campaign controller after the active launch validates."""
import datetime
import json
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
target = json.loads((HERE / 'pause-target.json').read_text())
directory = ROOT / 'raw/glm53-w4/refresh/default'
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    try:
        ready = all(json.loads((directory / name).read_text()).get(key) == value
                    for name, key, value in [('memory-check.json', 'passed', True),
                        ('rank-identity.json', 'passed', True),
                        ('down.log.command.json', 'returncode', 0)])
    except (FileNotFoundError, ValueError):
        ready = False
    if ready:
        process = Path('/proc') / str(target['pid'])
        assert process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == target['start_ticks']
        os.kill(target['pid'], signal.SIGSTOP)
        record = {**target, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'reason': 'User requested an accuracy-based assessment of single-query fusion; '
                            'pause between cleanly completed model launches for GPU diagnostics.',
                  'validated_launch': str(directory)}
        (HERE / 'paused-after-clean-launch.json').write_text(json.dumps(record, indent=2) + '\n')
        os.kill(target['pid'], signal.SIGTERM)
        os.kill(target['pid'], signal.SIGCONT)
        print(json.dumps(record), flush=True)
        break
    time.sleep(0.05)
else:
    raise RuntimeError('Timed out waiting for a clean benchmark boundary')
