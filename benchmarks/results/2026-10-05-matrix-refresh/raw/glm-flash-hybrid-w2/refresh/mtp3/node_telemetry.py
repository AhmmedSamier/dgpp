#!/usr/bin/env python3
"""Monitor benchmark ranks; stop a serving process immediately if it uses swap."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(value):
    print(json.dumps(value), flush=True)


def serving_processes():
    rows = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid() or (proc / 'comm').read_text().strip() != 'dgpp-serve':
                continue
            fields = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines()
                          if ':' in line)
            rows.append({'pid': int(proc.name), **{key + '_KiB': int(fields[key].split()[0])
                         for key in ('VmRSS', 'VmData', 'VmSwap', 'RssAnon', 'RssFile', 'RssShmem')}})
        except (FileNotFoundError, ProcessLookupError, KeyError):
            continue
    return rows


tick = 0
started = time.monotonic()
signalled = set()
try:
    while True:
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemFree', 'MemAvailable', 'Cached', 'SwapTotal',
                       'SwapFree', 'SwapCached', 'Mlocked', 'Unevictable', 'Slab'):
                memory[key + '_KiB'] = int(value.split()[0])
        vm = {}
        for line in Path('/proc/vmstat').read_text().splitlines():
            key, value = line.split()
            if key.startswith(('pswp', 'pgscan', 'pgsteal', 'allocstall')) or key == 'oom_kill':
                vm[key] = int(value)
        processes = serving_processes()
        emit({'at': stamp(), 'elapsed_s': time.monotonic() - started, 'kind': 'memory', **memory, 'vmstat': vm,
              'serving_processes': processes,
              'pressure': Path('/proc/pressure/memory').read_text().strip()})
        for process in processes:
            if process['VmSwap_KiB'] and process['pid'] not in signalled:
                emit({'at': stamp(), 'kind': 'violation', 'reason': 'dgpp used swap', **process})
                signalled.add(process['pid'])
                try:
                    os.kill(process['pid'], signal.SIGINT)
                except ProcessLookupError:
                    pass
        if tick % 5 == 0:
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown',
                                    '--format=csv,noheader'], capture_output=True, text=True, timeout=2)
                emit({'at': stamp(), 'kind': 'gpu', 'returncode': p.returncode,
                      'data': p.stdout.strip(), 'error': p.stderr.strip()})
            except (OSError, subprocess.TimeoutExpired) as error:
                emit({'at': stamp(), 'kind': 'gpu', 'error': str(error)})
        tick += 1
        time.sleep(1)
except BrokenPipeError:
    pass
