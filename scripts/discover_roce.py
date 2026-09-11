"""List RoCE devices, Linux interfaces, IPs, MTUs and GID indices.

Run without arguments for this host, or with --config FILE to inspect every
node in a deployment over SSH. This is read-only: it does not change .env,
configure networking, open RDMA connections or require downloaded weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cluster_doctor
from site_env import resolve_config


def suggestions(inventory):
    devices = [row for row in inventory if row["usable"]]
    if not 1 <= len(devices) <= 2:
        return None
    result = {"DGPP_ROCE_DEVICES": " ".join(row["device"] for row in devices)}
    if all(len(row["gids"]) == 1 for row in devices):
        result["DGPP_ROCE_GID_INDICES"] = " ".join(str(row["gids"][0]["index"]) for row in devices)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="inspect the first world_size nodes; run on rank 0")
    parser.add_argument("--json", action="store_true", help="print machine-readable inventory")
    args = parser.parse_args(argv)
    cfg = resolve_config(args.config) if args.config else None
    if cfg:
        cluster_doctor.require_head(cfg["nodes"][0])
    nodes = cfg["nodes"] if cfg else ["localhost"]
    inventories = {}
    for rank, host in enumerate(nodes):
        if rank == 0:
            inventories[host] = cluster_doctor.roce_inventory()
        else:
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                     f"{cfg['ssh_user']}@{host}", "python3 - roce --spec '{}'"],
                                    input=Path(cluster_doctor.__file__).read_text(), text=True,
                                    capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{host}: {result.stderr.strip() or 'SSH discovery failed'}")
            inventories[host] = json.loads(result.stdout)
    if args.json:
        print(json.dumps(inventories, indent=2))
        return
    overrides = {}
    for rank, (host, rows) in enumerate(inventories.items()):
        print(f"\n{host}: verbs device / port -> Linux interface, IPs, MTU, RoCE-v2 GID")
        if not rows:
            print("  No RDMA devices found. Check the driver and network setup.")
        for row in rows:
            status = "usable" if row["usable"] else "not usable by DGPP"
            print(f"  {row['device']} / {row['port']}: {status}")
            for gid in row["gids"]:
                print(f"    {gid['interface'] or '?'}  {', '.join(gid['ips']) or gid['address']}  MTU {gid['mtu'] or '?'}  GID index {gid['index']} ({gid['address']})")
        setting = suggestions(rows)
        if setting is None:
            print("  Select one or two active port-1 devices with routable RoCE-v2 GIDs; no settings suggested.")
        elif rank == 0:
            for key, value in setting.items():
                print(f'  {key}="{value}"')
        else:
            overrides[host] = setting
    if overrides:
        print("\nPer-node candidate settings (merge with existing overrides):")
        print("DGPP_NODE_OVERRIDES='" + json.dumps(overrides, separators=(",", ":")) + "'")
    print("\nBefore copying settings: match lane order by subnet across nodes, not by device name.")
    print("Multiple GIDs require choosing the intended network. Discovery does not test RDMA connectivity.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"RoCE discovery: {error}", file=sys.stderr)
        raise SystemExit(2)
