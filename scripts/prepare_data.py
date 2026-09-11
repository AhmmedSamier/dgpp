"""Fetch pinned evaluation datasets or tokenize a local prefill text.

Downloads are explicit, never performed by a benchmark. Existing files are
preserved unless their content matches; use a different output directory to
keep multiple versions. A manifest records upstream revisions and SHA-256.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

from cluster_doctor import cache_root, cached_snapshot
from data_paths import data_dir, require_file
from site_env import config_path, deployment, cache_environment


DATASETS = {
    "gsm8k": ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
              "gsm8k_test.jsonl", "https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE"),
    "humaneval": ("https://raw.githubusercontent.com/openai/human-eval/"
                 "6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz",
                 "HumanEval.jsonl", "https://github.com/openai/human-eval/blob/6d43fb980f9fee3c892a914eda09951f772ad10d/LICENSE"),
}


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different content at {path}; choose another output path")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(data)
        temporary = Path(temp.name)
    try:
        # A hard link creates the destination atomically without clobbering a
        # file another process may have created since the existence check.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download(tasks, output):
    output = Path(output)
    for task in tasks:
        url, name, license_url = DATASETS[task]
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read(20_000_001)
        if len(payload) > 20_000_000:
            raise ValueError(f"unexpectedly large dataset download: {task}")
        data = gzip.decompress(payload) if url.endswith(".gz") else payload
        lines = [json.loads(line) for line in data.splitlines() if line.strip()]
        keys = {"question", "answer"} if task == "gsm8k" else {"task_id", "prompt", "test", "entry_point"}
        if not lines or any(not isinstance(line, dict) or not keys <= line.keys() for line in lines):
            raise ValueError(f"invalid dataset shape: {task}")
        manifest = {"source": url, "license": license_url, "rows": len(lines),
                    "sha256": hashlib.sha256(data).hexdigest()}
        write_new(output / name, data)
        write_new(output / (name + ".source.json"), (json.dumps(manifest, indent=2) + "\n").encode())
        print(f"{output / name}: {len(lines)} rows, SHA-256 {manifest['sha256']}")


def tokens(model, text, output):
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("install requirements-tools.txt in a venv to generate token IDs") from error
    snapshot = cached_snapshot(model, cache_root(cache_environment()))
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    ids = tokenizer.encode(require_file(text).read_text(), add_special_tokens=False).ids
    if not ids:
        raise ValueError("input text produced no tokens")
    write_new(output, (",".join(map(str, ids)) + "\n").encode())
    print(f"{output}: {len(ids)} token IDs for {model}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--tasks", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    fetch.add_argument("--out", type=Path)
    tokenize = commands.add_parser("tokens")
    tokenize.add_argument("--model")
    tokenize.add_argument("--text", type=Path, required=True)
    tokenize.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        download(args.tasks, args.out or data_dir())
    else:
        tokens(args.model or deployment(config_path())["model"], args.text, args.out or data_dir() / "hard_ids.csv")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"benchmark data: {error}", file=sys.stderr)
        raise SystemExit(2)
