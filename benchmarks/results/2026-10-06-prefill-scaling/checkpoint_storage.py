#!/usr/bin/env python3
"""Record disk space and remove session downloads after their full matrix passes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess

from result_sources import completed

HERE = Path(__file__).resolve().parent
NEW_MODELS = {
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "RadixArk/Qwen3.8-Flash-Next-NVFP4",
}
REMOTE = r'''
import json, pathlib, shutil, subprocess, sys
request = json.loads(sys.argv[1])
root = pathlib.Path.home() / ".cache/huggingface/hub"
before = shutil.disk_usage(root)
result = {"free_before": before.free, "total": before.total, "removed": []}
for name in request["remove"]:
    assert name.startswith("models--") and "/" not in name and ".." not in name
    resident_root = pathlib.Path.home() / ".cache/dgpp/matrix-2026-10-05"
    for parent, name_part in [(root, name), (resident_root, name.removeprefix("models--"))]:
        path = parent / name_part
        if not path.exists():
            continue
        assert path.is_dir() and not path.is_symlink()
        assert path.resolve().parent == parent.resolve()
        size = int(subprocess.check_output(["du", "-s", "-B1", str(path)], text=True).split()[0])
        shutil.rmtree(path)
        result["removed"].append({"path": str(path), "allocated_bytes": size})
result["free_after"] = shutil.disk_usage(root).free
print(json.dumps(result))
'''


def entry_complete(entry):
    root = HERE / "raw" / entry["id"] / "refresh"
    for mode in ["default", *entry["modes"]]:
        directory = root / mode
        jobs = ["greedy"]
        if mode == "default":
            jobs += ["prefill", *[f"context-{n}" for n in entry["context_targets"]]]
        if not all(completed(directory, job) is not None for job in jobs):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cleanup-complete", action="store_true")
    args = parser.parse_args()
    matrix = json.loads((HERE / "matrix.json").read_text())
    inventories = json.loads((HERE / "checkpoint-inventory.json").read_text())
    resident_before = json.loads((HERE / "resident-inventory-before-new-models.json").read_text())
    if any(node["campaign_resident_exists"] for node in resident_before):
        raise RuntimeError("campaign resident-cache path predates setup; refusing cleanup")
    removable = []
    cleaned_path = HERE / "cleaned-models.json"
    cleaned = json.loads(cleaned_path.read_text()) if cleaned_path.exists() else []
    if args.cleanup_complete:
        for model in sorted(NEW_MODELS):
            name = "models--" + model.replace("/", "--")
            if name in cleaned:
                continue
            entries = [entry for entry in matrix if entry["model"] == model]
            if entries and all(entry_complete(entry) for entry in entries):
                removable.append(name)

    def inspect(inventory):
        if inventory["returncode"] != 0:
            raise RuntimeError("initial inventory failed; refusing cleanup")
        initial = json.loads(inventory["stdout"])
        remove = [name for name in removable if name not in initial]
        request = json.dumps({"remove": remove}, separators=(",", ":"))
        import shlex
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "stephen@" + inventory["host"], "python3 - " + shlex.quote(request)]
        result = subprocess.run(command, input=REMOTE, text=True, capture_output=True,
                                timeout=300, check=True)
        return {"host": inventory["host"], **json.loads(result.stdout)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(inspect, inventories))
    record = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "eligible_models": removable, "nodes": records}
    with (HERE / "storage.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    if removable:
        cleaned_path.write_text(json.dumps(sorted(set(cleaned + removable)), indent=2) + "\n")
    for node in records:
        print(f"DISK {node['host']}: {node['free_after'] / 2**30:.1f} GiB free; "
              f"removed {len(node['removed'])} completed session downloads", flush=True)
    if any(node["free_after"] < 40 * 2**30 for node in records):
        raise RuntimeError("less than 40 GiB free; resolve storage before further benchmarks")


if __name__ == "__main__":
    main()
