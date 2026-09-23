"""Record process provenance and rank logs; compare op streams after a clean stop."""

import argparse
import json
import pathlib
import shlex
import subprocess
from benchmark import ROOT, metrics

LIVE = r"""
import hashlib,json,pathlib,subprocess
pid=subprocess.check_output(['pgrep','-x','dgpp-serve'],text=True).strip()
assert pid.isdigit(), 'Expected exactly one serving process'
p=pathlib.Path('/proc')/pid
argv=(p/'cmdline').read_bytes().decode().split('\0')
config=pathlib.Path(argv[argv.index('--config')+1])
env=(p/'environ').read_bytes().decode().split('\0')
print(json.dumps(dict(pid=int(pid),sha256=hashlib.sha256((p/'exe').read_bytes()).hexdigest(),argv=argv,config=json.loads(config.read_text()),env=[x for x in env if x.startswith(('DGPP_MIMO_', 'DGPP_RESIDENT_CACHE=', 'DGPP_DENSE_GEMV_ROWS='))])))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["live", "stopped"])
    parser.add_argument("tag")
    parser.add_argument(
        "--hosts", nargs="+", required=True, help="SSH targets in rank order"
    )
    args = parser.parse_args()
    assert all(not h.startswith("-") for h in args.hosts)
    dest = ROOT / "raw" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / (args.phase + "-deployment.json")
    assert not output.exists(), "Preserve previous evidence"
    if args.phase == "live":
        ranks = []
        for host in args.hosts:
            result = subprocess.check_output(
                ["ssh", host, "python3 -c " + shlex.quote(LIVE)], text=True
            )
            ranks.append(json.loads(result))
        assert len({x["sha256"] for x in ranks}) == 1
        assert len({tuple(sorted(x["env"])) for x in ranks}) == 1
        record = {"hosts": args.hosts, "ranks": ranks, "metrics": metrics()}
    else:
        record = json.loads((dest / "live-deployment.json").read_text())
        assert record["hosts"] == args.hosts
        digests = []
        for rank, (host, entry) in enumerate(zip(args.hosts, record["ranks"])):
            argv = entry["argv"]
            remote_dir = pathlib.PurePosixPath(argv[argv.index("--config") + 1]).parent
            script = (
                "import pathlib,hashlib,json,subprocess\nassert subprocess.run(['pgrep','-x','dgpp-serve'],stdout=subprocess.DEVNULL).returncode == 1\np=pathlib.Path("
                + repr(str(remote_dir / f"serve_rank{rank}.ops"))
                + ")\nb=p.read_bytes()\nprint(json.dumps(dict(md5=hashlib.md5(b).hexdigest(),bytes=len(b))))"
            )
            digests.append(
                json.loads(
                    subprocess.check_output(
                        ["ssh", host, "python3 -c " + shlex.quote(script)], text=True
                    )
                )
            )
        record["op_streams"] = digests
    for rank, (host, entry) in enumerate(zip(args.hosts, record["ranks"])):
        argv = entry["argv"]
        remote_dir = pathlib.PurePosixPath(argv[argv.index("--config") + 1]).parent
        subprocess.run(
            [
                "scp",
                host + ":" + str(remote_dir / f"serve_r{rank}.log"),
                str(dest / f"rank{rank}-{args.phase}.log"),
            ],
            check=True,
        )
        if args.phase == "stopped":
            subprocess.run(
                [
                    "scp",
                    host + ":" + str(remote_dir / f"serve_rank{rank}.ops"),
                    str(dest / f"rank{rank}.ops"),
                ],
                check=True,
            )
    output.write_text(json.dumps(record, indent=2))
    if args.phase == "stopped":
        assert all(x["bytes"] for x in digests), "Empty op stream"
        assert (
            len({x["md5"] for x in digests}) == 1
        ), "Rank op streams differ; inspect logs"
    print("PASS", args.phase, "provenance/log capture", flush=True)


if __name__ == "__main__":
    main()
