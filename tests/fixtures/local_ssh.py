#!/usr/bin/env python3
"""Local SSH substitute for tiny-cache tests; never contacts a network host."""
import os
from pathlib import Path
import sys

args = sys.argv[1:]
user = None
while args and args[0] in ("-o", "-l"):
    if args[0] == "-l":
        user = args[1]
    args = args[2:]
host = args.pop(0) if args else ""
if not (host == "tester@fake-peer" or (host == "fake-peer" and user == "tester")):
    raise SystemExit("test SSH only accepts tester@fake-peer")
peer_home = Path(os.environ["DGPP_TEST_PEER_HOME"])
if not peer_home.is_dir():
    raise SystemExit("test peer home must already exist")
os.environ["HOME"] = str(peer_home)
os.environ.pop("HF_HOME", None)
os.environ.pop("HF_HUB_CACHE", None)
os.execv("/bin/sh", ["sh", "-c", " ".join(args)])
