#!/usr/bin/env python3
"""Repeat the final serving matrix on reserved, idle cluster hardware.

Set PR95_BIN to the executable and PR95_OUT to a new, empty result directory.
Uses the repository's site configuration and pinned deployment templates.
"""
import os
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
root = here.parents[2]
binary = Path(os.environ['PR95_BIN']).resolve()
out = Path(os.environ['PR95_OUT']).resolve()
assert binary.is_file() and os.access(binary, os.X_OK), binary
out.mkdir(parents=True, exist_ok=True)
assert not any(out.iterdir()), 'Use a new, empty result directory'
env = {**os.environ, 'PR95_BIN': str(binary), 'PR95_OUT': str(out)}
for world in (4, 1, 2):
    for suffix, quant in (('fp8a', 'fp8'), ('nvfp4', 'nvfp4'), ('fp8b', 'fp8')):
        label = f'decode-wide-final-w{world}-{suffix}'
        cmd = [sys.executable, str(here / 'campaign.py'), quant, str(world), label, '--final']
        if world == 4:
            cmd.append('--decode-context')
        print(label, flush=True)
        with (out / f'{label}-controller.log').open('w') as log:
            subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=7200, check=True)
codes = []
for script in ('audit_decode_wide_final.py', 'audit_decode_context.py'):
    with (out / f'{Path(script).stem}.log').open('w') as log:
        result = subprocess.run([sys.executable, str(here / script)], cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    codes.append(result.returncode)
raise SystemExit(max(codes))
