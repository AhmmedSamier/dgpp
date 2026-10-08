# Collective waits during the 128K prefill

The three 128K requests completed successfully. During the cold request,
the collective diagnostic logged waits above its 500 ms threshold. These
observations do not establish their cause.

| Observation | Evidence |
|---|---|
| Ranks 0, 2 and 3 each logged 17 warnings across the same 12 collective sequence numbers | `context-128k-collective-summary.json` |
| Maximum logged wait was about 1.5 seconds; later sequence numbers and prefill progress continued | Per-rank `context-128k-rankN-server.log` snapshots |
| Rank 1 logged no collective-wait warnings during this launch | The same snapshots; the pattern suggests later arrival by rank 1 |
| Serving-process swap remained zero on every rank; host swap-out counters did not increase during the 128K request | `nodeN-telemetry.jsonl` |
| Memory pressure averages were zero when inspected | `nodeN-telemetry.jsonl` |
| No GPU or transport kernel faults appeared in the inspected interval | `context-128k-rankN-kernel.log`; rank 0 has unrelated AppArmor records before this launch |
| A 77.76-second transport snapshot interval had small packet-sequence and congestion-counter increments, with no retransmission, timeout, CRC or ingress-discard increments | `context-128k-roce-before.txt`, `context-128k-roce-after.txt`, `context-128k-roce-delta.json` |
| CPU snapshots during the following 256K request showed the bus poller pinned to CPU 19 on each rank, and no competing workload dominating rank 1 | `context-256k-rankN-cpu.json` |
| The HTTP health check returned `ok` during the following 256K request | `context-256k-health-live.json` |

The warning timer starts when the bus thread picks up the collective request
(`src/net/collective_bus.cpp`, `collective_pass`); it is not a measurement of
wire latency alone. No profiler or runtime setting was changed for this
inspection. The measured prefill times include these waits.

Peer server logs append older launches. The summary filters log timestamps
at or after 2026-10-06 05:34 UTC to exclude the invalid attempt. Node wall
clocks differ: rank 1 is about 78 seconds ahead of rank 0, and rank 3 about
6 seconds behind it. Sequence numbers identify the corresponding warnings.
The full snapshots retain the original log contents.
