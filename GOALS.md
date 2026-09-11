*The maintainer intends this document for local use only.*

# Project goal

Make DGPP the fastest inference engine for the models it supports on
NVIDIA DGX Spark, with particular attention to clustered deployments.

## Primary metric

Single-stream decode throughput per supported model. Verify regressions
before committing work of medium or large scope.

## Approach

- Use measured hardware bandwidth and compute limits to guide optimization.
- Write CUDA kernels suited to GB10 and each model's architecture.
- Test, profile and benchmark changes against representative workloads.
- Improve hardware utilization during decode and prefill.
- Share implementations where they suit the workload; specialize where
  the architecture or measurements justify it.

## Implemented support

Cluster configuration is selected at launch, so a site can keep separate
configs for each model. Qwen3.8-Flash-Next-FP8 is supported alongside
GLM-5.3-Flash and GLM-4.7; Qwen's NVFP4 variant also runs on one Spark.

## Further goals

- Improve weight-loading progress reports at startup.
- Increase prefill throughput, including latency under concurrent decode.
- Investigate viable quantized checkpoints for full GLM-5.3 on four Sparks.
- Add Qwen3.8-27B-FP8 support.

See [PLAN.md](PLAN.md) and [next steps](docs/next_steps.md) for the
implementation status and current work list.
