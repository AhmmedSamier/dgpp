"""Download or verify a complete checkpoint in DGPP's Hugging Face cache.

Authentication comes from Hugging Face's saved login or HF_TOKEN, never from
sourcing .env or placing a token on the command line. Run on each node; this
command does not copy credentials or weights between hosts.
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile

from cluster_doctor import cache_root, cached_snapshot, checkpoint_size
from site_env import config_path, deployment, cache_environment, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="defaults to the selected deployment's model")
    parser.add_argument("--config")
    parser.add_argument("--rank", type=int, default=0, help="use this node's cache override from site settings")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--activate", action="store_true", help="explicitly point refs/main at the downloaded revision after verification")
    parser.add_argument("--verify-only", action="store_true", help="read the active snapshot without network access or writes")
    args = parser.parse_args()
    values = settings()
    model = args.model or deployment(args.config or config_path(values))["model"]
    env = cache_environment(values, args.rank)
    root = args.cache_dir.expanduser() if args.cache_dir else cache_root(env)
    if args.verify_only:
        if args.activate or args.revision != "main":
            parser.error("--verify-only checks the active snapshot; do not combine with --activate or --revision")
        snapshot = cached_snapshot(model, root)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ValueError("install requirements-download.txt in a venv before downloading") from error
        snapshot = Path(snapshot_download(repo_id=model, revision=args.revision, cache_dir=str(root), max_workers=4))
    size = checkpoint_size(snapshot)
    if args.activate:
        reference = snapshot.parent.parent / "refs/main"
        reference.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=reference.parent, delete=False) as output:
            output.write(snapshot.name + "\n")
            temporary = Path(output.name)
        try:
            os.replace(temporary, reference)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified {model}: {snapshot}\n{size / 2**30:.1f} GiB of indexed weights; metadata and shard lengths are consistent.")
    print("This is a structural check, not a full content-hash verification.")
    if cached_snapshot(model, root).resolve() != snapshot.resolve():
        raise ValueError("a different snapshot is active; rerun with --activate if you intend to change the served revision")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"checkpoint setup: {error}", file=sys.stderr)
        raise SystemExit(2)
