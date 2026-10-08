# Version probes and the swap guard

The Qwen two-node plain-mode preflight stopped at 19:06:55 UTC on October 6.
The monitor saw the short `dgpp-serve --version` command outside a protected
scope and sent SIGINT. The launcher then failed its version check before
any serving rank started. No benchmark requests ran in this attempt.

The process reported zero `VmSwap`; its parent session cgroup's existing
swap usage belonged to the shared scope. Host swap-in and swap-out counters
were unchanged during the observed preflight. The preceding default launch
passed its own memory, rank-identity and shutdown checks and remains valid.

Preflight and status version probes now use `MemorySwapMax=0` whenever
`DGPP_NO_SWAP=1` is selected. The policy is forwarded to peer preflight
probes too. The monitor, benchmark binary, measurement clients and deployment
settings are unchanged. No policy exceptions or retries without protection
were added.

| Evidence | Record |
|---|---|
| Failed preflight | [Archived attempt](../../attempts/qwen-nvfp4-w2/plain-20261006T191602186592Z/INVALID.md) |
| Before the launcher change | [Manifest](manifest-before.json), [patch](launcher-patch-before.patch) |
| Tests | [Validation](validation.json): seven swap-policy tests pass; launcher suite runs 19 tests, with 18 passing and one skipped |
| Live guard | [Result](live-guard-result.json), [telemetry](live-guard-telemetry.jsonl): a three-second metadata probe stays protected throughout; the actual binary's version check also passes |

The resumed campaign starts with the interrupted plain-mode sweep. Its new
launch manifests record the updated launcher hashes; prior manifests remain
unchanged.
