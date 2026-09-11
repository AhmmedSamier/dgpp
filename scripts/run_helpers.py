"""Small process and timing helpers used by the Bash orchestration scripts."""
import argparse
import hashlib
from pathlib import Path
import random
import socket
import time


def port_open(port):
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("random-delay")
    path_id = commands.add_parser("path-id")
    path_id.add_argument("path")
    elapsed = commands.add_parser("elapsed")
    elapsed.add_argument("start", type=float)
    port = commands.add_parser("port-open")
    port.add_argument("port", type=int)
    args = parser.parse_args()
    if args.command == "random-delay":
        print(round(random.uniform(0.2, 1.5), 2))
    elif args.command == "path-id":
        print(hashlib.sha256(str(Path(args.path).resolve()).encode()).hexdigest()[:16])
    elif args.command == "elapsed":
        print(round(time.time() - args.start, 2))
    else:
        return 0 if port_open(args.port) else 1


if __name__ == "__main__":
    raise SystemExit(main())
