"""Prepare token-ID prompts and summarize prefill measurements."""
import argparse
from pathlib import Path
import re


def repeat_prompt(path, length):
    ids = Path(path).read_text().strip().replace("\n", ",").split(",")
    ids = [token for token in ids if token]
    return ",".join((ids * 20)[:length])


def report(path, length):
    match = re.search(r"prefill: (\d+) tokens in (\d+)ms", Path(path).read_text(errors="replace"))
    if match:
        tokens, milliseconds = map(int, match.groups())
        print(f"  {tokens:>5} tokens: {milliseconds:>7} ms prefill = {milliseconds / tokens:6.1f} ms/token")
    else:
        print(f"  n{length}: no prefill line")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("repeat-prompt", "summary"))
    parser.add_argument("path")
    parser.add_argument("length", type=int)
    args = parser.parse_args()
    if args.command == "repeat-prompt":
        print(repeat_prompt(args.path, args.length))
    else:
        report(args.path, args.length)


if __name__ == "__main__":
    main()
