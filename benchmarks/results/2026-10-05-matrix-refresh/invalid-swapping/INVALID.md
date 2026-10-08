# Invalid benchmark attempt

All measurements in this directory are invalid for the benchmark refresh. The four-node GLM Flash hybrid run showed active page-outs and several GiB of `VmSwap` in each `dgpp-serve` process. The campaign was stopped on 2026-10-06 at 02:47 UTC. Earlier runs lack complete per-process swap monitoring and must also be rerun.

OS swap remains enabled. Valid reruns must establish zero swap use by `dgpp` throughout loading and measurement. Original commands, results and diagnostics are preserved here; paths inside receipts refer to their original locations.
