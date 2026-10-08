# Preflight stopped before serving

No benchmark requests ran in this attempt. At 19:06:55 UTC the memory
monitor observed the launcher's short `dgpp-serve --version` process
outside a zero-swap scope and stopped it. The process itself reported
`VmSwap=0`; the shared parent session's swap usage was not its usage.
Preflight then failed its server-version check before any rank started.

Version probes now use the same `MemorySwapMax=0` protection as rank
launches. The memory monitor and its acceptance checks remain unchanged.
This entire attempt is retained as failed evidence and must not supply
benchmark results. The preceding default launch passed independently.
