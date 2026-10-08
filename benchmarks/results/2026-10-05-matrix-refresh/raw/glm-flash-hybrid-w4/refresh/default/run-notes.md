# Four-node GLM Flash hybrid: validated default launch

This launch ran on 2026-10-06 from 09:49 to 13:44 UTC. It replaces the
invalid earlier launch; none of that launch's measurements were reused.

| Check | Result |
|---|---|
| Serving | C1/C2/C4/C8, five prompt classes, three repetitions: 60 request groups |
| Cold prefill | 2K/8K/32K, three uncached prompts each |
| Context | 0/32K/64K/128K/256K/512K, three requests each |
| Process swap | 0 KiB on every rank throughout monitoring |
| Cgroup swap | Effective limit 0 bytes and usage 0 bytes on every sample |
| Memory samples | 13,948–13,961 per rank; maximum gap 1.337 seconds |
| Rank operation streams | All four match: `14f7acb2d071b3251d1bef6d8e218d06` |
| Shutdown | Exit 0 |
| 512K input | 523,781 prompt tokens; cached tokens 0 / 523,776 / 523,776 |
| 512K cold prefill | 11,384.873 seconds |
| 512K decode | Median 45.247 ms/pass and 43.688 engine tokens/s |

The [memory check](memory-check.json), [rank check](rank-identity.json),
[512K samples](context-524288.json), and command receipts retain the evidence.
The serving binary, KV pool, prefix-cache allocation, and request slots
were unchanged. OS swap remained enabled; each dgpp rank ran in a scope
with `MemorySwapMax=0`.

## Collective waits

The [collective log summary](collective-observations.json) records waits
during long prefill. Cold-prefill timings include these waits. Their cause
has not been established; this launch had no process or cgroup swap.

| Rank | Warning lines | Distinct collective sequences | Maximum logged wait, ms |
|---|---:|---:|---:|
| 0 | 2 | 2 | 506 |
| 1 | 85 | 79 | 1,007 |
| 2 | 88 | 81 | 1,008 |
| 3 | 103 | 95 | 1,007 |

A warning is a periodic observation of an ongoing wait, not its final
duration. Several ranks can report the same collective. These counts
therefore cannot be added to estimate total elapsed stall time.

Peer logs append older launches. The summary excludes records before
09:45 on each node's local clock; the previous launch had ended before
09:36 on every node, and this launch began after 09:49 on every node.
Raw timestamps retain the nodes' clock offsets.
