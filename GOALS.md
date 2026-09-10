*This file shall not be checked into the repository; it is a local-only manifesto written by the primary project author/maintainer*

# Project Goal

To be the single fasted inference engine specifically optimized for the Nvidia DGX Spark, especially clustered deployments.

# Golden Metric
* Single-stream decode throughput per supported model (regressions must be verified before any medium-to-large-scope body of work is committed)

# How we get there
* Measure what the hardware can actually do under synthetic workloads to establish "line-rate" (we already have this)
* Optimized CUDA kernels specifically targetting the GB10 platform
* Testing, profiling, benchmarking, and rapid iteration.
* Ensuring that the hardware is fully saturated during decode and prefill
* Custom implementations per-model / per-model-architecture. General purpose is only fine when general purpose is the optimal solution.

# Remaining goals (in no particular order):
* Add logged progress of weights loading during server startup
* cluster.json provided as a launch arg; longer-term there will be multiple models supported, so we want to allow users to maintain a local library of working configs.
* Optimize prefill throughput further
* Support for Qwen3.8 Flash Next FP8: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8
* Support for GLM 5.3 (full, not flash. FP8 won't fit on TP=4 spark, so need to research viable quants)
* Support for Qwen3.8-27b FP8
