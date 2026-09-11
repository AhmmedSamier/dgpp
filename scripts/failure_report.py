"""Request construction and replay checks for the rank-failure drill."""
import argparse
import glob
import json
import os
from pathlib import Path


def request_body(stream, max_tokens, prompt, model):
    return {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": stream}


def is_committed_prefix(committed, replay):
    prefix = Path(committed).read_text(encoding="utf-8")
    answer = Path(replay).read_text(encoding="utf-8")
    return bool(prefix) and answer.startswith(prefix)


def report_ops(directory, victim_rank):
    victim = f"serve_rank{victim_rank}.ops"  # killed -9: it wrote nothing this run
    files = sorted(f for f in glob.glob(os.path.join(directory, "serve_rank*.ops"))
                   if os.path.getsize(f) > 0 and os.path.basename(f) != victim)
    texts = {Path(f).name: Path(f).read_text(encoding="utf-8", errors="replace").splitlines()
             for f in files}
    if len(texts) < 2:
        print("  (fewer than two ops files; no cross-rank comparison)")
    else:
        n = min(len(v) for v in texts.values())
        base = next(iter(texts.values()))[:n]
        bad = [k for k, v in texts.items() if v[:n] != base]
        print(f"  op streams agree over the first {n} lines" if not bad else f"  FAIL: op streams disagree within the first {n} lines: {bad}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    body = commands.add_parser("request-body")
    body.add_argument("stream", choices=("true", "false"))
    body.add_argument("max_tokens", type=int)
    body.add_argument("prompt")
    body.add_argument("model")
    prefix = commands.add_parser("prefix")
    prefix.add_argument("committed")
    prefix.add_argument("replay")
    ops = commands.add_parser("ops")
    ops.add_argument("directory")
    ops.add_argument("victim_rank", type=int)
    args = parser.parse_args()
    if args.command == "request-body":
        print(json.dumps(request_body(args.stream == "true", args.max_tokens, args.prompt, args.model)))
    elif args.command == "prefix":
        return 0 if is_committed_prefix(args.committed, args.replay) else 1
    else:
        report_ops(args.directory, args.victim_rank)


if __name__ == "__main__":
    raise SystemExit(main())
