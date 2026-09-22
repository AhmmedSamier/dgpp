# Cluster environment comparison

Captured on 2026-09-22 after Node 3 recovered from the second GLM-4.7 outage.
Node numbers here are the user's numbering; engine ranks are one less.
The inventory is read-only. No firmware, driver, kernel or system settings
were changed during this investigation.

## Findings

Node 3 differs in its GPU VBIOS, operating-system point release and several
userspace packages. The active kernel, GPU driver, CUDA runtime, cuBLAS and
RDMA stack match across the cluster. These differences are candidates for
investigation, not evidence that a particular component caused the outage.

The hardware is a mixed-vendor GB10 cluster. BIOS and embedded-controller
version strings are vendor-specific and cannot be ranked across vendors.

| Component | Node 1 (.11) | Node 2 (.12) | Node 3 (.13) | Node 4 (.14) |
|---|---|---|---|---|
| System | MSI EdgeXpert MS-C931 | ASUS GX10 | NVIDIA DGX Spark | AI TOP ATOM |
| Hostname | edgexpert-409e | gx10-e9cd | spark-a289 | aitopatom-4b80 |
| Ubuntu point release | 24.04.4 | 24.04.4 | 24.04.5 | 24.04.4 |
| Active kernel | 6.17.0-1026-nvidia | same | same | same |
| Active GPU driver | 580.173.02 | same | same | same |
| GPU VBIOS | 9A.0B.1E.00.00 | 9A.0B.1E.00.00 | 9A.0B.2D.00.00 | 9A.0B.25.00.00 |
| DMI BIOS version | 5.36_1.5.0 | GX10DGX.0104.2026.0326.1657 | 5.36_0ACUM018 | 5.36_0ACUM07 |
| DMI BIOS date | 2026-01-30 | 2026-03-26 | 2025-08-06 | 2026-04-21 |
| Embedded controller, raw reported version | 10500 | 0x02000005 | 0x03000508 | 0x03000302 |
| ConnectX-7 firmware | 28.45.4028 | same | same | same |
| CUDA compiler | 13.0.88 | same | same | same |
| CUDA runtime package | 13.0.96-1 | same | same | same |
| cuBLAS package | 13.1.1.3-1 | same | same | same |
| RDMA core / libibverbs | 50.0-2ubuntu0.2 | same | same | same |
| libstdc++ / libgcc | 14.2.0-4ubuntu2~24.04.1 | same | same | same |
| glibc | 2.39-0ubuntu8.7 | 2.39-0ubuntu8.7 | 2.39-0ubuntu8.9 | 2.39-0ubuntu8.7 |
| systemd | 255.4-1ubuntu8.16 | 255.4-1ubuntu8.16 | 255.4-1ubuntu8.17 | 255.4-1ubuntu8.16 |
| linux-firmware | 20240318.git3b128b60-0ubuntu2.27 | same as Node 1 | 20240318.git3b128b60.0ubuntu3.1 | same as Node 1 |
| NVIDIA container toolkit | 1.19.1-1 | 1.19.1-1 | 1.20.0-1 | 1.19.1-1 |

ELF build IDs also match for `libcudart`, `libcublasLt`, `libcuda`,
`libibverbs`, `libstdc++`, `libgcc_s`, `libnl` and `libnl-route`. The
`libc.so.6` build ID differs on Node 3, consistent with its package version.
The server runs directly on the hosts; the NVIDIA container toolkit is not
the serving runtime. HumanEval's Python containers execute on Node 1.

Node 3 has kernel 6.17.0-1032 packages installed, but is **running 1026**.
Its glibc, systemd and Linux firmware updates, and the 1032 installation,
date to September 12, before this benchmark campaign. See the
[package history and DMI record](raw/environment-audit/node3-update-history.txt).

## Memory and host settings

All nodes have about 121.6–121.7 GiB of OS-visible RAM and 16 GiB swap.
All use the performance CPU governor, `vm.swappiness=60`,
`vm.overcommit_memory=0`, `vm.panic_on_oom=0`, `kernel.numa_balancing=0`,
and transparent huge pages in `madvise` mode. Locked-memory limits are
unlimited, and the open-file limit is 500,000. Kernel command lines match
apart from each node's root-filesystem identity. The small difference in
`vm.min_free_kbytes` follows the small difference in OS-visible RAM.

At inventory time Node 1 was running the single-node Qwen quality evaluation;
the other nodes were idle. Current free-memory values are therefore not a
matched model-memory comparison. Node 3 had approximately 117.8 GiB
available, zero swap usage and no memory-pressure stalls in the current
averaging windows.

| Snapshot | Node 1 | Node 2 | Node 3 | Node 4 |
|---|---|---|---|---|
| Available RAM | 30.28 GiB | 117.61 GiB | 117.79 GiB | 117.61 GiB |
| Swap used | 206.83 MiB | 0.64 MiB | 0.00 MiB | 0.96 MiB |
| Root filesystem available | 116.2 GiB | 379.4 GiB | 688.1 GiB | 675.5 GiB |

These are unmatched load snapshots. All four nodes report zero memory
pressure in their 10-, 60- and 300-second averaging windows; cumulative
stall counters span different uptimes and are not directly comparable.

A spot check during Qwen YaRN's 512K cold prefill found 61.23 / 62.17 GiB
available on the two active nodes (Nodes 1 and 2), with zero pressure in
those averaging windows. GPU temperatures were 71 / 78°C, with no active
thermal-slowdown flags. These are single samples during the request, not
peak measurements. See the [saved snapshots](raw/qwen-yarn-w2/long/default/health-snapshots.jsonl).

More relevant are Node 3's persisted samples while GLM-4.7 was running:

| Sample time UTC | Available RAM | Swap used | Swap-in / swap-out | Time before outage |
|---|---|---|---|---|
| 03:20:04 | 30.38 GiB | 40 KiB | 0 / 0 pages/s | 7m 03s |
| 05:20:07 | 30.73 GiB | 0 | 0 / 0 pages/s | 4m 31s |

The sampled intervals also show no page-reclaim scans. These records argue
against sustained memory exhaustion, but do not measure peak memory during
the failing prefill. A transient allocation or driver failure remains
possible. The second attempt's recovered startup ledgers on the other three
ranks show about 30 GiB of available host memory after model and graph setup.

For comparison, the completed four-node GLM Flash FP8 performance run
reported only 4.56 GiB of available host memory on Node 3 after warm capture
at 02:19:35 UTC, then completed its 8K and 32K cold-prefill samples. This is
another reason not to attribute the GLM-4.7 outages to low steady-state
memory alone. It does not compare transient allocation demands between
models. The evidence is in the [Node 3 server log](raw/glm-flash-fp8-w4/performance/default/server/serve_r2.log)
and [prefill results](raw/glm-flash-fp8-w4/performance/default/prefill.json).

NVIDIA documents that Spark's CPU and GPU share physical DRAM and that CUDA
and OS memory figures have different accounting; those values must not be
added together as separate memory pools.
[NVIDIA UMA guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture).

## Network and crash evidence

Both active RoCE interfaces on every node report 200,000 Mb/s, full duplex,
MTU 9000 and RS FEC. The NIC driver is `mlx5_core` from the same active kernel,
with firmware 28.45.4028 and board identifier NVD0000000087. The sampled login
environments have no `LD_LIBRARY_PATH`, CUDA device overrides or selected
DGPP kernel/fabric overrides. The campaign separately sets
`CUDA_DEVICE_MAX_CONNECTIONS=32` on rank 0, as recorded in its manifest.

Node 3 also has an independent Wi-Fi interface, `wlP9s9`, with the DHCP
address `192.168.50.232` at inventory time. SSH through that address was
verified against Node 3's existing trusted host key. Subsequent four-node
runs collect Node 3's memory and kernel telemetry through this management
link, so loss of the RoCE interfaces alone need not cut off diagnostics.
The Wi-Fi address was not tested during the two earlier outages. Those
observations therefore establish loss of both fabric paths, not whether
the entire operating system had stopped responding.

The previous boot's saved journal contains no OOM-killer, GPU Xid or kernel
panic signature. Its last kernel message is at 05:15:31; its last saved
journal activity is an ordinary SSH session at 05:20:54. The outage occurred
at 05:24:38. Persistent crash storage is empty. Node 3 has zero crash-kernel
reservation (`/sys/kernel/kexec_crash_size=0`), so absence of a crash dump is
not evidence against a kernel failure.

Both outages occurred on GLM-4.7 at its first 8K cold-prefill request.
GLM-4.7 is now excluded at the user's request; other models remain in the
benchmark matrix. Neither memory pressure nor overheating is established.

Subsequent four-node runs collect host-memory and pressure counters
approximately every second, GPU temperature, power, clocks and thermal
flags approximately every five seconds, and Node 3's live kernel journal.
Collection begins before model startup and ends after deployment shutdown.
These lightweight probes run during timing; earlier measurements did not
have continuous telemetry. Each launch retains its own telemetry files.
The [telemetry summarizer](summarize_telemetry.py) derives per-node sampled
extrema and counter changes for completed launches in
`raw/telemetry-summary.json`. Its observation interval includes startup and
shutdown; sampled extrema do not establish transient peaks between samples.

The first completed four-node launch with this telemetry was full GLM-5.3
int4/int8, from 09:11 to 09:57 UTC. It passed all greedy and sampled timing
groups, all three cold-prefill requests at each of approximately 2K, 8K and
32K tokens, and its solo/batched exact-text check. All four operation streams
matched. Node 3's minimum sampled available memory was 3.876 GiB and its
maximum sampled GPU temperature was 81°C. Its OOM and swap-out counters did
not increase, memory PSI's sampled ten-second averages remained zero, and
neither sampled thermal-slowdown flag became active. The other nodes had
some swap-out activity over the launch interval; none recorded an OOM or
an active sampled thermal-slowdown flag. These observations establish a
successful run with another model and less steady memory headroom than the
GLM-4.7 attempts, while leaving the earlier outages' cause unresolved.
The [per-node summary](raw/telemetry-summary.json) and
[full GLM-5.3 run](raw/glm53-w4/performance/default/) retain the evidence.

During the FP8-KV GLM-5.3 sampled sweep, Node 3 ran its scheduled APT News
and ESM cache checks at 10:57:34 UTC. Both services finished successfully
within that second. Two AppArmor `perfmon` capability denials came from
their environment-detection helpers. PackageKit then started, and an NVIDIA
OTA availability check was visible in the process list. Package-install
logs remained unchanged since September 13; this was background update
checking, with no package installation recorded. These OS services were
not disabled for the campaign. The [saved service journal and log timestamps](raw/glm53-fp8kv-w4/performance/default/node3-background-services.log)
identify the activity; the kernel telemetry contains no GPU or fabric error
at that time.

The completed campaign did not reproduce the outage on another model.
Across all 21 completed four-node launches with continuous telemetry, no
node recorded an OOM kill or an active sampled thermal-slowdown flag.
Node 3's lowest sampled available memory was 3.312 GiB, its highest sampled
GPU temperature was 81°C, and all of its GPU telemetry queries succeeded.
These observations include loading and shutdown; they do not rule out
transient events between samples or establish the cause of the GLM-4.7
outages. The complete [per-launch summary](raw/telemetry-summary.json)
identifies the coverage and counter changes.

## Clock alignment

A read-only clock check at approximately 07:19 UTC found Node 2 about
12 seconds ahead of Node 1, with `NTPSynchronized=no`. Nodes 1, 3 and 4
reported synchronization, and the SSH sampling bounds put Nodes 3 and 4
within one second of Node 1. Cross-node wall-clock log timestamps therefore
need this qualification. Request durations use the client's monotonic
clock, and engine timings are local durations; these measurements do not
subtract wall-clock timestamps from different hosts. No clock settings
were changed. The [clock probe and raw offsets](raw/environment-audit/clock-offsets.json)
retain the local measurement bounds.

## Evidence

- [Identical inventory script](environment_audit.py) and [raw inventories](raw/environment-audit/).
- [Runtime library and network probe](raw/environment-audit/runtime_probe.py).
- [Node 3 memory history](raw/glm47-w4/performance/attempt-20260922T0524/node13-memory-sar.txt).
- [Previous-boot journal](raw/glm47-w4/performance/attempt-20260922T0524/node13-previous-boot-journal.txt).
- [Privileged crash-storage inspection](raw/glm47-w4/performance/attempt-20260922T0524/node13-privileged-crash-record.txt).
- [Interruption and recovery record](interruptions.md).
