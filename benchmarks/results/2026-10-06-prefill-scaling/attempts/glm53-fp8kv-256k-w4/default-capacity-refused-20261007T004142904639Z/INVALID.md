# Startup capacity refusal

This deployment ran no benchmark requests. Rank 0 refused the 256K FP8 configuration: 112.27 GiB planned + 4 GiB reserve exceeded 115.58 GiB available. The standalone memory-plan command runs before normal startup allocations and had overestimated the available margin.

The controller and failed launcher were interrupted during the HTTP-readiness wait. The normal down command collected all rank logs and confirmed no serving ranks remained. Memory telemetry continued through shutdown; every sampled process and cgroup swap counter stayed zero. Missing operation streams are expected because the service never became ready.

The active matrix replaces this unmeasured case with a separately named 256K FP4 KV configuration. Existing measured configurations and their results are unchanged. See `diagnostics/full-glm-256k-capacity/` in the campaign root for controller intervention and preliminary FP4 capacity records.
