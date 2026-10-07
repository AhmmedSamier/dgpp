#!/usr/bin/env python3
"""Resumable, sequential refresh of the documented deployment matrix."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("campaign", HERE.parent / "2026-09-22-current/run.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
campaign.HERE = HERE
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
os.environ["DGPP_NO_SWAP"] = "1"


class MemoryViolation(RuntimeError):
    pass


def start_telemetry(directory, config):
    cfg = campaign.resolve_config(str(config))
    source = (HERE / 'node_telemetry.py').read_bytes()
    saved_probe = directory / 'node_telemetry.py'
    saved_probe.write_bytes(source)
    shutil.copy2(HERE / 'manifest.json', directory / 'manifest.json')
    running = []
    campaign.TELEMETRY[str(directory)] = running
    for rank, host in enumerate(cfg['nodes']):
        output = (directory / f'node{rank}-telemetry.jsonl').open('w')
        command = ([sys.executable, str(saved_probe)] if rank == 0 else
                   ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    f"{cfg['ssh_user']}@{host}", 'python3', '-'])
        proc = subprocess.Popen(command, stdin=subprocess.PIPE if rank else subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        running.append((proc, output))
        if rank:
            proc.stdin.write(source)
            proc.stdin.close()
    if len(cfg['nodes']) == 4:
        output = (directory / 'node2-live-kernel.log').open('w')
        proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                 f"{cfg['ssh_user']}@{cfg['nodes'][2]}", 'sudo', '-n',
                                 'journalctl', '-k', '-f', '-n', '0', '-o', 'short-iso'],
                                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        running.append((proc, output))


campaign.start_telemetry = start_telemetry


def check_memory(directory, config):
    manifest = json.loads((directory / 'manifest.json').read_text())
    require_policy = manifest.get('benchmark_environment', {}).get('DGPP_NO_SWAP') == '1'
    ranks = []
    for rank in range(len(campaign.resolve_config(str(config))['nodes'])):
        samples, violations, errors, times = [], [], [], []
        path = directory / f'node{rank}-telemetry.jsonl'
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                record = json.loads(line)
            except ValueError:
                errors.append(line)
                continue
            if record.get('kind') == 'memory':
                samples.extend(record.get('serving_processes', []))
                times.append(record.get('elapsed_s', datetime.datetime.fromisoformat(record['at']).timestamp()))
            elif record.get('kind') == 'violation':
                violations.append(record)
        gaps = [b - a for a, b in zip(times, times[1:])]
        ranks.append({'rank': rank, 'samples': len(samples),
                      'max_swap_KiB': max((s['VmSwap_KiB'] for s in samples), default=None),
                      'swap_policy_required': require_policy,
                      'swap_policy_passed': all(s.get('swap_limit_bytes') == 0 and
                                                s.get('cgroup_swap_bytes') == 0 for s in samples)
                                            if require_policy else None,
                      'max_sample_gap_s': max(gaps, default=None),
                      'violations': violations, 'errors': errors})
    monitor_failure = directory / 'memory-monitor-failure.json'
    passed = not monitor_failure.exists() and all(r['samples'] and r['max_swap_KiB'] == 0 and not r['violations'] and not r['errors']
                 and (not r['swap_policy_required'] or r['swap_policy_passed'])
                 and r['max_sample_gap_s'] is not None and r['max_sample_gap_s'] <= 10
                 for r in ranks)
    campaign.save(directory / 'memory-check.json', {'passed': passed, 'ranks': ranks})
    if not passed:
        raise MemoryViolation(f"serving-process swap check failed: {directory}")


def mode_overrides(mode):
    if mode == "default":
        return {}
    if mode == "dflash2-fixed":
        return {"mtp_schedule": False}
    return {"dflash_model": "", "mtp": mode != "plain", "mtp_schedule": False,
            "mtp_depth": 1 if mode == "plain" else int(mode[-1])}


def measure(entry, mode, stages):
    directory = HERE / "raw" / entry["id"] / "refresh" / mode
    jobs = (["greedy", "prefill"] + [f"context-{n}" for n in entry["context_targets"]]
            if mode == "default" else ["greedy"])
    jobs = [j for j in jobs if j.split("-")[0] in stages]
    if not jobs:
        return
    def finished(job):
        receipt = directory / f"{job}.log.command.json"
        memory = directory / 'memory-check.json'
        identity = directory / 'rank-identity.json'
        shutdown = directory / 'down.log.command.json'
        return (memory.exists() and json.loads(memory.read_text()).get('passed') and
                identity.exists() and json.loads(identity.read_text()).get('passed') and
                shutdown.exists() and json.loads(shutdown.read_text()).get('returncode') == 0 and
                receipt.exists() and json.loads(receipt.read_text()).get("returncode") == 0)
    if all(finished(j) for j in jobs):
        print(f"SKIP {entry['id']} {mode}", flush=True)
        return
    # Keep the process telemetry and results from each launch together. A
    # partially completed mode is rerun as a unit, never combined with a new
    # launch's memory evidence.
    if directory.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = HERE / 'attempts' / entry['id'] / (mode + '-' + stamp)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), str(archive))
    config = server = None
    try:
        try:
            directory, config, server = campaign.launch(entry, "refresh", mode, mode_overrides(mode))
        except Exception:
            check_memory(directory, directory / 'config.json')
            raise
        def check_monitors(job):
            monitors = campaign.TELEMETRY[str(directory)][:len(campaign.resolve_config(str(config))['nodes'])]
            stopped = [rank for rank, (proc, _) in enumerate(monitors) if proc.poll() is not None]
            if stopped:
                campaign.save(directory / 'memory-monitor-failure.json',
                              {'at': campaign.now(), 'job': job, 'ranks': stopped})
                raise MemoryViolation(f'memory monitor stopped during {job}: {directory}')
        for job in jobs:
            check_monitors(job)
            if job == "greedy":
                command = campaign.client("timed_load.py", "--concurrency",
                    "1,2,4,8" if mode == "default" else "1", "--classes", "all",
                    "--repeat", 3, "--max-tokens", 256, "--json-out", directory / "greedy.json")
                if "GLM-5.3-Int4" in entry["model"]:
                    command.append("--think")
            elif job == "prefill":
                command = campaign.client("serve_prefill_probe.py", 2048, 8192, 32768,
                    "--repeat", 3, *campaign.thinking_args(entry), "--seed", 7,
                    "--tag", "matrix-" + uuid.uuid4().hex,
                    "--json-out", directory / "prefill.json")
            else:
                command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/serve_context_matrix.py"),
                    "127.0.0.1", "18080", "--model", entry["model"], "--tag", uuid.uuid4().hex,
                    "--lengths", job.split("-")[1],
                    "--json-out", str(directory / f"{job}.json")]
            if not campaign.run(command, directory / f"{job}.log", timeout=43200, resume=False):
                raise RuntimeError(f"{job} failed: {directory}")
            check_monitors(job)
    finally:
        if config is not None:
            monitors = campaign.TELEMETRY[str(directory)][:len(campaign.resolve_config(str(config))['nodes'])]
            stopped = [rank for rank, (proc, _) in enumerate(monitors) if proc.poll() is not None]
            if stopped:
                campaign.save(directory / 'memory-monitor-failure.json',
                              {'at': campaign.now(), 'job': 'before shutdown', 'ranks': stopped})
            campaign.stop(directory, config, server)
            check_memory(directory, config)
            if not json.loads((directory / "rank-identity.json").read_text())["passed"]:
                raise RuntimeError(f"rank operation streams did not match: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--stages", nargs="+", choices=["greedy", "prefill", "context", "modes"],
                        default=["greedy", "prefill", "context", "modes"])
    args = parser.parse_args()
    binary = ROOT / "build-release/dgpp-serve"
    manifest_path = HERE / "manifest.json"
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["binary_sha256"] != digest:
            raise RuntimeError("binary differs from this campaign's manifest")
    else:
        campaign.save(manifest_path, {
            "started_at": campaign.now(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "version": subprocess.check_output([str(binary), "--version"], text=True).strip(),
            "binary_sha256": digest,
            "benchmark_environment": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
            "scope": "rank 0; peer processes retain their login environment",
            "source_changes": "GLM prefill route-trace retention fix; see diagnostics/glm-prefill-trace-fix.patch.",
            "concurrency": [1, 2, 4, 8], "repeat": 3, "max_tokens": 256,
            "clients": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [ROOT / "scripts/timed_load.py", ROOT / "scripts/serve_load.py",
                                     ROOT / "scripts/serve_prefill_probe.py", ROOT / "scripts/serve_context_matrix.py",
                                     ROOT / "scripts/qwen_yarn_release_check.py"]},
            "environment": "environment.json",
            "checkpoint_inventory_before_restore": "checkpoint-inventory.json",
            "telemetry": {"scope": "all serving ranks", "memory_interval_s": 1,
                          "gpu_interval_s": 5, "kernel_journal_rank": 2,
                          "management_transport": "configured SSH nodes"},
        })
    entries = json.loads((HERE / "matrix.json").read_text())
    failures = []
    for entry in entries:
        if args.only and entry["id"] not in args.only:
            continue
        subprocess.run([sys.executable, str(HERE / "checkpoint_storage.py"), "--check"], check=True)
        modes = ["default"] + (entry["modes"] if "modes" in args.stages else [])
        for mode in modes:
            try:
                measure(entry, mode, args.stages if mode == "default" else ["greedy"])
            except Exception as error:
                record = {"entry": entry["id"], "mode": mode, "time": campaign.now(), "error": str(error)}
                failures.append(record)
                with (HERE / "failures.jsonl").open("a") as output:
                    output.write(json.dumps(record) + "\n")
                print(record, flush=True)
                if isinstance(error, MemoryViolation):
                    raise
            finally:
                subprocess.run([sys.executable, str(HERE / "summarize.py")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(HERE / "checkpoint_storage.py"), "--cleanup-complete"], check=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
