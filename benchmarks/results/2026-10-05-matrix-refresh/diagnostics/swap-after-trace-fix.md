# Swap after the route-trace fix

The four-node GLM Flash hybrid launch that began at 05:34 UTC on
2026-10-06 is invalid. At 09:33:52 UTC, its root process reported
41,156 KiB of swap during the 512K cold request. The guard sent SIGINT,
the launcher stopped the cluster, and the campaign halted. No values
from that launch are included in the overview tables.

The two completed two-node deployments retain their passing memory,
rank-consistency and shutdown checks.

| Observation | Value |
|---|---|
| Root process | PID 442843 |
| Last sampled anonymous memory before reclaim | 1,187,008 KiB; flat for hours |
| Anonymous memory at violation | 1,145,852 KiB |
| Swap at violation | 41,156 KiB; exactly the anonymous-memory reduction |
| Data allocation at both samples | 1,264,288 KiB |
| Peers' maximum process swap | 0 KiB on all three peers |
| Maximum telemetry gaps | 1.06–1.09 seconds |
| Host reclaim burst | 28,874 pages swapped out; 63,213 anonymous pages scanned; direct and background reclaim both advanced |
| Host available memory immediately before the burst | About 3.1 GiB |

The trace fix removed the growing route-score vectors. This event came
from reclaim of existing anonymous memory, not renewed growth of those
vectors. The engine's optional `mlockall(MCL_CURRENT)` runs before model
construction, so it does not cover host allocations made later.

Subsequent launches use `DGPP_NO_SWAP=1`. The launcher starts each rank in
a systemd user scope with `MemorySwapMax=0`; the kernel therefore cannot
swap that scope's anonymous pages. OS swap remains available to other
processes. KV capacity, prefix cache, request slots and the serving binary
are unchanged. Telemetry records both the effective cgroup limit and its
swap usage, and rejects missing protection as well as nonzero swap.

The previous harness and manifest are preserved in
[`before-process-swap-policy/`](before-process-swap-policy/). Small-process
validation is recorded in
[`process-swap-policy-validation/`](process-swap-policy-validation/).
The failed launch is preserved as a whole in
[`attempts/glm-flash-hybrid-w4/default-20261006T094944333088Z/`](../attempts/glm-flash-hybrid-w4/default-20261006T094944333088Z/);
its exact probe, manifest, logs and failed memory check remain together.
The protected rerun began at 09:49:44 UTC and completed its default-mode
tests at 13:44 UTC, including all three 512K requests. Every rank reported
zero process and cgroup swap throughout; memory, operation-stream and
shutdown checks passed. See the [validated launch notes](../raw/glm-flash-hybrid-w4/refresh/default/run-notes.md).

The two previously completed deployments also had zero host swap-outs
throughout all eight mode launches, in addition to zero process swap:
[audit](completed-before-swap-policy.json).

Kernel reference: [`memory.swap.max`](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files).
