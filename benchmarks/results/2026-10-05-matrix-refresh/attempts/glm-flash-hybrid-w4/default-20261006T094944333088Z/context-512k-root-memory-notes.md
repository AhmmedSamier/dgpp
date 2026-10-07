# Rank 0 memory during the 512K cold request

At approximately 08:21 UTC, the root host's `MemAvailable` estimate fell
from about 3.6 GiB to 3.1 GiB. The serving process's memory counters did
not grow, and its `VmSwap` remained zero. No runtime or OS memory setting
was changed.

| Observation | Evidence |
|---|---|
| From 08:00 through 08:26 UTC, dgpp's sampled RSS, anonymous memory, shared memory and data size were unchanged | `node0-telemetry.jsonl`; RSS 4,136,964 KiB, anonymous 1,187,008 KiB, shared 2,242,308 KiB, data 1,264,288 KiB |
| The host swap-out counter remained 152,189,382 pages over that interval | `node0-telemetry.jsonl` |
| Memory-pressure averages remained zero | The same telemetry |
| The Normal zone had a watermark boost of 111,949 pages, or about 437 MiB with 4 KiB pages | `context-512k-root-watermarks.json` |
| Free pages changed much less than the available-memory estimate | At 08:20:01, free/available were 2,248/3,679 MiB; at 08:22:02, 2,092/3,194 MiB |
| The estimate later recovered to about 3.5 GiB as the watermark boost cleared, without intervention | `context-512k-root-watermarks-recovered.json`; Normal low watermark fell from 154,881 to 42,932 pages, exactly the previous boost |

Linux includes the zone's watermark boost in its low watermark, and the
available-memory calculation subtracts a watermark-dependent amount from
reclaimable memory. The observed boost is consistent with much of the
lower estimate. The subsequent snapshot confirms that the boost cleared;
its original trigger is not established. References: [kernel calculation](https://raw.githubusercontent.com/torvalds/linux/v6.11/mm/show_mem.c),
[watermark definitions](https://raw.githubusercontent.com/torvalds/linux/v6.11/include/linux/mmzone.h),
[kernel documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html#watermark-boost-factor).

The earlier `context-512k-live-memory.json` snapshot records all four ranks'
process memory, GPU state and host swap-counter deltas. Host counters
include other OS processes and do not identify which process owned those
pages. Per-process swap validation remains the benchmark gate.
