"""Sync one Hugging Face snapshot and its blobs; peers never contact the Hub."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

import cluster_doctor

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def activate(snapshot):
    """Atomically update refs/main only after validating the checkpoint."""
    snapshot = Path(snapshot)
    cluster_doctor.checkpoint_size(snapshot)
    reference = snapshot.parent.parent / "refs/main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=reference.parent, delete=False) as output:
        output.write(snapshot.name + "\n")
        temporary = Path(output.name)
    try:
        os.replace(temporary, reference)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_files(snapshot):
    """List snapshot files and referenced blobs, never other models or HF tokens."""
    snapshot = Path(snapshot).resolve()
    repository = snapshot.parent.parent
    blobs = (repository / "blobs").resolve()
    files = set()
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(blobs) or Path(os.readlink(path)).is_absolute():
                raise ValueError(f"snapshot link must be relative and point into its own blobs directory: {path}")
            files.add(str(target.relative_to(repository)))
        elif path.is_dir():
            continue
        elif not path.is_file():
            raise ValueError(f"unsupported cache entry: {path}")
        files.add(str(path.relative_to(repository)))
    if not files or any("\n" in name or "\r" in name for name in files):
        raise ValueError("snapshot is empty or contains unsupported filenames")
    return "\n".join(sorted(files)) + "\n"


def ssh(target, command, **kwargs):
    result = subprocess.run(["ssh", *SSH_OPTIONS, target, command], capture_output=True,
                            text=True, timeout=120, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{target}: {result.stderr.strip() or 'SSH command failed'}")
    return result.stdout.strip()


def peer_cache(target, model, env, *, verify=False, revision=None):
    spec = {"env": env, "model": model, "verify": verify, "revision": revision}
    command = "python3 - cache --spec " + shlex.quote(json.dumps(spec))
    return json.loads(ssh(target, command, input=Path(cluster_doctor.__file__).read_text()))


def sync_snapshot(snapshot, target, env):
    snapshot = Path(snapshot).resolve()
    model = snapshot.parent.parent.name.removeprefix("models--").replace("--", "/", 1)
    manifest = snapshot_files(snapshot)
    print(f"Syncing {snapshot.name} to {target} (checking existing content may take a while)...", flush=True)
    destination = peer_cache(target, model, env)["repository"]
    if not Path(destination).is_absolute() or any(c in destination for c in "\0\r\n"):
        raise ValueError("peer returned an invalid cache path")
    temporary = ssh(target, "command -v rsync >/dev/null && mktemp -d /tmp/dgpp-cache-sync.XXXXXXXX")
    if not re.fullmatch(r"/tmp/dgpp-cache-sync\.[A-Za-z0-9]{8}", temporary):
        raise ValueError("peer returned an invalid helper directory")
    transport = shlex.join(["ssh", *SSH_OPTIONS])
    try:
        subprocess.run(["rsync", "-rt", "--protect-args", "-e", transport,
                        str(Path(__file__).resolve()), str(Path(cluster_doctor.__file__).resolve()),
                        f"{target}:{temporary}/"], check=True)
        ssh(target, "mkdir -p -- " + shlex.quote(destination))
        # Preserve relative snapshot symlinks and content-addressed blobs.
        # No --delete or --inplace: retain other revisions and keep partial
        # transfers from replacing complete files.
        subprocess.run(["rsync", "-rlt", "--links", "--checksum", "--protect-args",
                        "--info=progress2", "--partial-dir=.dgpp-partial", "--files-from=-", "-e", transport,
                        str(snapshot.parent.parent) + "/", f"{target}:{destination}/"],
                       input=manifest, text=True, check=True)
        remote_snapshot = str(Path(destination) / "snapshots" / snapshot.name)
        ssh(target, f"python3 -B {temporary}/cache_sync.py --activate " + shlex.quote(remote_snapshot))
        print(f"Synced and verified {target}: {remote_snapshot}", flush=True)
    finally:
        try:
            ssh(target, f"rm -f -- {temporary}/cache_sync.py {temporary}/cluster_doctor.py && rmdir -- {temporary}")
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"Warning: temporary helper cleanup failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", type=Path, required=True)
    args = parser.parse_args()
    try:
        activate(args.activate)
    except (OSError, ValueError) as error:
        parser.exit(2, f"cache activation: {error}\n")
