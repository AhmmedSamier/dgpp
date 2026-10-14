#!/usr/bin/env python3
"""Prepare a serve-ready Qwen3-ASR checkpoint directory (docs/audio.md).

Upstream ships no dgpp-loadable tokenizer or template, so dgpp-serve
cannot point at a raw Hugging Face snapshot the way vLLM can. This script
closes that gap: it downloads the snapshots, symlinks the weights and
configs into --out, merges tokenizer.json
(tools/qwen_asr_tokenizer.py) and extracts chat_template.jinja, then
prints the serve command.

Usage:
  python3 tools/prepare_qwen_asr_checkpoint.py \
      --model Qwen/Qwen3-ASR-1.7B --out /models/qwen3-asr-1.7b
  python3 tools/prepare_qwen_asr_checkpoint.py \
      --snapshot /path/to/snapshot --in-place   # cluster snapshots

Requires the optional download dependencies
(requirements-download.txt: huggingface-hub). Serving itself needs none
of this: --out is plain files and symlinks.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN3_MODEL = "Qwen/Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--qwen3-model", default=QWEN3_MODEL,
                    help="tokenizer donor for tokenizer.json")
    ap.add_argument("--snapshot", default=None,
                    help="use this snapshot dir instead of downloading --model")
    ap.add_argument("--out", default=None,
                    help="serve-ready directory (symlinks + generated files)")
    ap.add_argument("--in-place", action="store_true",
                    help="write tokenizer.json and chat_template.jinja into "
                         "the snapshot itself instead of building --out "
                         "(for cluster snapshots synced to peers)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface-hub is required (requirements-download.txt)")

    if args.snapshot is not None:
        if not os.path.isdir(args.snapshot):
            sys.exit("missing snapshot dir: %s" % args.snapshot)
        snap = args.snapshot
    else:
        snap = snapshot_download(args.model, revision=args.revision)
    out_dir = snap if args.in_place else args.out
    if out_dir is None:
        sys.exit("need --out (or --in-place with --snapshot)")
    q3 = snapshot_download(args.qwen3_model, allow_patterns=["tokenizer.json"])
    q3_tok = os.path.join(q3, "tokenizer.json")
    if not os.path.isfile(q3_tok):
        sys.exit("no tokenizer.json in %s" % q3)

    os.makedirs(out_dir, exist_ok=True)
    if not args.in_place:
        for name in sorted(os.listdir(snap)):
            if name in ("tokenizer.json", "chat_template.jinja"):
                continue  # generated below, never symlinked
            src, dst = os.path.join(snap, name), os.path.join(out_dir, name)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)

    subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "qwen_asr_tokenizer.py"),
         "--snapshot", snap, "--qwen3-tokenizer", q3_tok,
         "--out", os.path.join(out_dir, "tokenizer.json")],
        check=True)
    with open(os.path.join(snap, "chat_template.json")) as f:
        template = json.load(f)["chat_template"]
    if not template.strip():
        sys.exit("empty chat_template in snapshot")
    with open(os.path.join(out_dir, "chat_template.jinja"), "w") as f:
        f.write(template)

    with open(os.path.join(out_dir, "tokenizer.json")) as f:
        json.load(f)  # must parse
    print("serve-ready checkpoint: %s" % out_dir)
    print("serve: ./build-release/dgpp-serve --checkpoint-dir %s --port 8000"
          % out_dir)


if __name__ == "__main__":
    main()
