#!/usr/bin/env python3
"""Read-only node inventory; run identically on each benchmark node."""
import datetime
import glob
import json
import os
from pathlib import Path
import platform
import re
import subprocess


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return {"error": str(error)}


def command(argv, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


r = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": platform.node(),
    "kernel": platform.uname()._asdict(),
    "os_release": read("/etc/os-release"),
    "dgx_release": "\n".join(line for line in str(read("/etc/dgx-release")).splitlines()
                               if "SERIAL_NUMBER=" not in line),
    "kernel_command_line": re.sub(r"root=\S+", "root=<node-root>", str(read("/proc/cmdline"))),
    "meminfo": read("/proc/meminfo"),
    "memory_pressure": read("/proc/pressure/memory"),
    "swap": read("/proc/swaps"),
    "vmstat": read("/proc/vmstat"),
    "uptime": read("/proc/uptime"),
    "dmi": {k: read("/sys/class/dmi/id/" + k) for k in (
        "bios_vendor", "bios_version", "bios_date", "board_vendor", "board_name",
        "board_version", "product_name", "product_version")},
    "cpu_governors": {p: read(p) for p in glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_governor")},
    "transparent_hugepages": read("/sys/kernel/mm/transparent_hugepage/enabled"),
    "system_settings": command(["sysctl", "vm.swappiness", "vm.overcommit_memory", "vm.overcommit_ratio",
        "vm.panic_on_oom", "vm.max_map_count", "vm.min_free_kbytes", "kernel.numa_balancing",
        "kernel.panic", "kernel.panic_on_oops", "kernel.softlockup_panic"]),
    "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,vbios_version,pstate,temperature.gpu,power.limit,power.draw,clocks.max.sm,clocks.current.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown", "--format=csv,noheader"]),
    "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]),
    "cuda_compiler": command(["/usr/local/cuda/bin/nvcc", "--version"]),
    "cuda_link": os.path.realpath("/usr/local/cuda"),
    "compiler": command(["g++", "--version"]),
    "rdma": command(["ibv_devinfo"]),
    "interfaces": command(["ip", "-j", "address", "show"]),
    "routes": command(["ip", "-j", "route", "show"]),
    "limits": command(["bash", "-lc", "ulimit -a"]),
    "disk": command(["df", "-B1", "/", "/tmp"]),
}
packages = command(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\t${db:Status-Status}\n"])
if packages.get("returncode") == 0:
    wanted = ("nvidia", "cuda", "cublas", "cudnn", "nccl", "libibverbs", "librdmacm", "rdma",
              "libmlx", "ibverbs", "ofed", "doca", "linux-image", "linux-modules", "linux-firmware",
              "libc6", "libstdc++", "libgcc", "systemd", "fwupd", "dgx", "mellanox")
    r["packages"] = [line for line in packages["stdout"].splitlines()
                     if line.endswith("\tinstalled") and any(s in line.split("\t")[0] for s in wanted)]
else:
    r["packages"] = packages
firmware = command(["fwupdmgr", "get-devices", "--json"], timeout=20)
try:
    devices = json.loads(firmware["stdout"])["Devices"]
    r["firmware"] = [{k: d[k] for k in ("Name", "Vendor", "Version", "VersionRaw", "VersionFormat",
                                           "Protocol", "Plugin", "Flags") if k in d} for d in devices]
except (KeyError, TypeError, json.JSONDecodeError):
    r["firmware"] = firmware
r["network_drivers"] = {Path(p).name: command(["ethtool", "-i", Path(p).name])
                        for p in glob.glob("/sys/class/net/en*")}
print(json.dumps(r, indent=2))
