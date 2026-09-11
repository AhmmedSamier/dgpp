# DGPP

DGPP is a C++/CUDA inference engine for NVIDIA DGX Spark (GB10) systems.
It serves GLM-5.3-Flash, Qwen3.8-Flash-Next and GLM-4.7 through an
OpenAI-compatible HTTP API, with tensor parallelism over RoCE for
multi-node deployments. Supported configurations use one, two or four
nodes, depending on the model and its memory requirements.

The repository contains the CUDA kernels, RDMA collectives, tokenizer,
Jinja chat-template interpreter, scheduler, prefix cache and HTTP service.
It uses the CUDA runtime, cuBLASLt and libibverbs. Rank 0 coordinates
requests through an admission journal; peers check their operation streams
against it throughout a run.

## Supported models and configurations

These serving configurations have deployment templates and recorded
measurements. World size is the number of ranks, with one DGX Spark per
rank; world 1 runs locally, while worlds 2 and 4 use tensor parallelism
over RoCE. Each quant links to its specific Hugging Face model card.

| Model | Quant / Hugging Face model card | World sizes | Example configuration |
|---|---|---|---|
| GLM-5.3-Flash | [unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8) | 4 | [Four nodes](deploy/cluster.example.json), with `model` set to the linked FP8 repository |
| GLM-5.3-Flash (hybrid) | [HawkBearPig/GLM-5.3-Flash-NVFP4-FP8](https://huggingface.co/HawkBearPig/GLM-5.3-Flash-NVFP4-FP8) | 4 | [Four nodes](deploy/cluster.nvfp4.example.json) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 2, 4 | [Two nodes](deploy/cluster_qwen_w2.example.json), [four nodes](deploy/cluster_qwen.example.json) |
| Qwen3.8-Flash-Next | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 1 | [One node, BF16 dense](deploy/cluster_qwen_spark1.example.json), [one node, FP8 dense](deploy/cluster_qwen_spark1_fp8.example.json) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) | 4 | [Four nodes](deploy/cluster_glm47.example.json) |

The single-Spark Qwen configurations require `engine.ngram_table: "mmap"`
to read the n-gram table from NVMe and `engine.decode_graph: true` for
resident graph serving. Its BF16-dense and FP8-dense templates use the
same Hugging Face quant; the latter sets `engine.dense_weights: "fp8"`
to encode dense projections at load. See
[the single-node guide](docs/qwen38_single_spark.md) for details.

The GLM-5.3 hybrid takes the main-stack routed experts from
[dabsLabs](https://huggingface.co/dabsLabs/GLM-5.3-Flash-NVFP4) and the
remaining tensors, including MTP, from
[Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8).
The default and NVFP4 cluster templates both use
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. Download it into the Hugging Face
cache on each node before serving. To use the FP8 release instead, set
`model` to `unsloth/GLM-5.3-Flash-FP8`.
To reproduce the hybrid from its sources, use
[the composition tool](tools/compose_nvfp4_hybrid.py) and
[the NVFP4 notes](docs/nvfp4_plan.md).

The linked templates enable MTP. Plain-decode and deeper-MTP variants are
also available in [deploy/](deploy/); [the benchmark tables](docs/benchmarks.md)
record the modes measured for each deployment.

## Highlights

- **Tensor-parallel resident serving** on four nodes: each rank holds its
  slice of the model resident on the GPU and boots from a per-rank image
  cache in 15–25 s.
- **Graph-based decode**, including collectives and MTP speculative
  decoding, with scalar or batched graphs selected for the active requests.
  Greedy MTP produces the same tokens as plain decode; sampled MTP
  preserves the target distribution.
- **Exact prefix caching**: matching token prefixes can reuse a stored
  session snapshot while preserving the cold-prefill result.
- **OpenAI-compatible Chat Completions**: streaming, constrained tool
  calls, `response_format` with supported JSON schemas,
  `reasoning_content`, logprobs, `stop`, `n`, `logit_bias`, usage details.
- **Deterministic across ranks**: admissions journaled from the head, every
  tick's operation-stream digest checked on every peer, and all ranks'
  complete streams compared at shutdown.
- **Process-failure detection**: a rank process exiting fails the service
  within seconds. Silent node loss is detected by the bus watchdog.
- **Deployment and monitoring**: a shared cluster config, versioned
  releases, periodic throughput logs and `/v1/metrics`. Startup checks
  the memory plan before allocation. Cache capacity is configurable,
  with BF16, FP8 or FP4 latent storage for GLM-5.3.

## Performance

These GLM-5.3-Flash-FP8 results were measured at TP=4 for the v1 sign-off.
See [the benchmark tables](docs/benchmarks.md) for other models and
configurations, and [the sign-off report](docs/signoff_v1.md) for the
workloads and validation behind these figures. They are historical
measurements, not performance guarantees for every configuration.

| measure | result |
|---|---|
| decode, one request, plain graph | 31.45 ms per token |
| decode, one request, MTP (greedy, 77–97 % drafts accepted by class) | 21.3–23.9 ms per token |
| aggregate, 8 live requests at T=1 | 76 tokens/s |
| aggregate, 4 live requests under MTP | 60 tokens/s |
| prefill, 256 / 2,048 / 32,768 tokens | 0.58 s / 1.7 s / 33.7 s |
| boot from the resident image cache | 15–25 s |

## Status

Version 0.1.0 (2026-09-06) completed milestones M0–M9 for GLM-5.3-Flash
serving, including prefix caching, transactional MTP and a one-hour soak.
The working tree also supports Qwen3.8-Flash-Next and GLM-4.7 through the
shared engine, scheduler and service. Quantized paths include NVFP4
weights and optional load-time FP8 encoding of Qwen's dense projections.
Qwen's NVFP4 checkpoint can run on one Spark with its n-gram table mapped
from NVMe; see [the single-node guide](docs/qwen38_single_spark.md).

[PLAN.md](PLAN.md) summarizes implementation status,
[CHANGELOG.md](CHANGELOG.md) records changes, and
[the remaining work](docs/next_steps.md) lists current limitations.

## Requirements

- NVIDIA DGX Spark (GB10) with CUDA 13 and an SM 12.1-capable compiler
- CMake 3.25 or newer and a C++20 compiler (GCC 13 tested)
- CUDA Runtime and cuBLASLt development files
- libibverbs headers and library, required by the current serving build,
  including world 1. `DGPP_ENABLE_IBV=OFF` omits serving and RDMA targets.
- Python 3.10 or newer for operations and test tooling. Reference-dump
  generators have optional dependencies such as PyTorch; see their help.

## Quick start

For a fresh machine, follow [Getting started](docs/getting-started.md):
dependencies, site configuration, checkpoint download, preflight, and a first
request. [Dependencies](docs/dependencies.md) separates runtime requirements
from optional evaluation and development tools.

### Build and test

```bash
cmake --preset ci
cmake --build --preset ci --target dgpp_serve_app -j 4
```

The presets are `release`, `debug`, `asan`, `ubsan` and `ci`
(`cmake --preset <name>`, `cmake --build --preset <name> -j`,
`ctest --preset <name>`). Build test targets before running CTest and select
the appropriate resource labels; GPU/RDMA suites need idle test hardware.
`ci-local.sh` builds and runs the full suite. With clang-format installed, `format` and
`format-check` targets exist. `docs/testing.md` lists the suites and what
each proves.

### Serve

1. Add the site settings from [.env.example](.env.example) to `.env`,
   preserving any existing credentials. Set `DGPP_NODES` in rank order
   and `DGPP_SSH_USER` for your machines. Copy a deployment template to
   `deploy/cluster.json` and choose its model, `world_size` and engine options.
2. Download the complete checkpoint on every node (see the setup guide),
   then run `scripts/dgpp-cluster doctor`. `scripts/dgpp-cluster up` stages the binary and config, boots
   rank 0, then the peers, and waits for the listening line.
3. Send a request:

   ```bash
   curl --fail http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "HawkBearPig/GLM-5.3-Flash-NVFP4-FP8", "reasoning_effort": "low", "max_tokens": 160,
     "messages": [{"role": "user", "content": "Name three primary colors, one per line."}]}'
   ```

4. `scripts/dgpp-cluster down` — drains, stops every rank and prints one
   md5 per rank's op stream. All ranks must have the same hash.

`docs/operations.md` is the operator's page: boot and stop, what a node
needs, what happens when a rank dies, how to tell the world is healthy.
HTTP defaults to localhost and has no authentication or TLS. Use an SSH
tunnel or an authenticated reverse proxy for remote access; see
[networking](docs/networking.md).

## Release and install

A release is a versioned tarball installed once per node. Starting an
installed release stages the configuration and uses the installed binary.

```bash
scripts/release.sh                                 # build the release preset, stage, verify, pack
scripts/dgpp-cluster install dist/dgpp-<version>.tar.zst   # every node: unpack under release_dir, verify
scripts/dgpp-cluster up --release <version>        # runs <release_dir>/dgpp-<version>/bin/dgpp-serve on every rank
scripts/dgpp-cluster releases                      # what each node has installed
```

Select a release with the config's `release` key or `--release`.

The version is the tree's, stamped at build time by `cmake/version.cmake`
into the binary (`dgpp-serve --version`; `0.1.0+g<sha12>`, `.dirty` when
the tree had uncommitted changes) and sent on the journal's settings
record, so a world of mixed versions refuses to form. Inside the tarball:

| path | what |
|---|---|
| `bin/dgpp-serve` | the server, rpath `$ORIGIN/../lib` |
| `lib/libcudart.so.13`, `lib/libcublasLt.so.13` | the CUDA runtime it was built against |
| `scripts/` | launcher, process/preflight/config helpers, checkpoint downloader and API check |
| `deploy/*.example.json` | all supported deployment templates |
| `README.md`, `docs/` | package-specific setup, dependency and networking guides |
| `MANIFEST`, `MANIFEST.sha256` | version, git sha, CUDA, build host and date; every other file's checksum |

Beyond the tarball a node needs the NVIDIA driver, rdma-core, libnl and
libstdc++, all part of the DGX OS image, plus the checkpoint and resident
image caches on its local NVMe. `install` unpacks under `paths.release_dir`
(`~/dgpp/releases/dgpp-<version>/`) and checks every file against the
manifest. Versions can be installed side by side; stop the running service
and select an older version to roll back. The launcher starts the service
on demand; no boot-time service units are included. When no release is
selected, `up` stages `build-ci/dgpp-serve` to the peers.

## Configuration

Site settings live in `.env` at the repository root, shared by every model
deployment. Scripts load it automatically through `scripts/site_env.py`
(shell scripts use `scripts/cluster_env.sh`). `DGPP_ENV_FILE` selects another
file; exported variables override file values. Values are literal, optionally
quoted: no shell commands or variable expansion are evaluated. Only the keys
below are loaded, so credentials such as `HF_ACCESS_TOKEN` stay out of the
scripts' environment and generated server config.

| `.env` key | purpose | default |
|---|---|---|
| `DGPP_NODES` | Space-separated addresses in rank order; the first node is the head | required |
| `DGPP_SSH_USER` | SSH/SCP login on the peers | current login when empty or absent |
| `DGPP_HTTP_PORT` | HTTP service port | 18080 |
| `DGPP_HTTP_BIND` | HTTP bind IPv4 address | `127.0.0.1` |
| `DGPP_FABRIC_PORT` | Fabric rendezvous port | 29970 |
| `DGPP_JOURNAL_PORT` | Journal port; must differ from the fabric port | 29971 |
| `DGPP_LOG_DIR` | Head logs, PID and collected peer logs | `~/dgpp/log` |
| `DGPP_STAGE_DIR` | Peer binary, runtime config and logs | `/tmp/bus4` |
| `DGPP_RELEASE_DIR` | Installed releases on each node | `~/dgpp/releases` |
| `DGPP_CLUSTER_CONFIG` | Default model deployment JSON; relative paths use the repository root | `deploy/cluster.json` |
| `DGPP_BUILD_DIR`, `DGPP_DATA_DIR` | Build and benchmark-data directories; relative to the repository root | `build-ci`, `data` |
| `HF_HUB_CACHE`, `HF_HOME` | Checkpoint cache; explicit hub cache takes precedence | `~/.cache/huggingface/hub` |
| `DGPP_RESIDENT_CACHE_DIR` | Local resident-image cache | `~/.cache/dgpp/resident` |
| `DGPP_ROCE_DEVICES` | Ordered verbs device names | discover active Ethernet devices |
| `DGPP_ROCE_GID_INDICES` | One GID index per explicit device | automatic RoCE-v2 selection |
| `DGPP_NODE_OVERRIDES` | JSON map of per-node NIC/GID/cache overrides | none |

Model settings live in deployment JSONs under `deploy/`. Each has a
`world_size` of 1, 2 or 4 and uses that many nodes from the beginning of
`DGPP_NODES`. Too few nodes is an error, not a fallback to a smaller world.
Use `--config FILE` with `dgpp-cluster` to override the selected deployment.
Wrappers that accept a config argument pass it through to their children.
Stage and release directories must be absolute or start with `~/`, without
spaces or shell syntax.
Log and staging directories are namespaced by deployment-file path;
`dgpp-cluster paths` prints the effective locations. An explicit `--log-dir`
is used as-is. `up` refuses an existing deployment unless `--replace` is given;
cleanup uses recorded process identity, never a binary-name kill.

The launcher resolves the deployment and site settings into
`<log_dir>/cluster.resolved.json` when starting the service. Both the head
and peers read that resolved config; the original JSON and `.env` are not
sent to peers. To inspect or use the runtime config with the native binary:

```bash
scripts/dgpp-cluster resolve --config deploy/cluster.nvfp4.json
# Save that JSON to a file before passing it to dgpp-serve --config.
```

The native binary reads resolved JSON, not `.env` or deployment templates.
Unknown keys and invalid types are rejected.

Rank 0 reads the model, ports and engine options, with command-line flags
after `--config` overriding the file. It sends the effective settings to
peers before model construction. Peers use their own files for bootstrap
addresses and local paths, then verify the shared settings by digest.
The defaults below come from `ClusterConfig::Engine`; deployment
templates set their serving options explicitly.

| key | required | what it does | default |
|---|---|---|---|
| `model` | yes | the Hugging Face model id every rank loads (`--model`) | — |
| `world_size` | yes | number of ranks: 1, 2 or 4; selects the first N entries of `DGPP_NODES` | — |
| `http.bind_host` | no | HTTP listening IPv4 address; overrides the site default | `DGPP_HTTP_BIND` |
| `http.port` | no | HTTP port; overrides the site default | `DGPP_HTTP_PORT` |
| `release` | no | the installed release version `up` runs on every rank (`<release_dir>/dgpp-<version>/bin/dgpp-serve`); launcher-only, `--release` overrides it; rolling back is naming the previous version | empty: the development binary `build-ci/dgpp-serve`, staged to the peers |
| `engine.max_concurrency` | no | Request slots per rank; graph row limits depend on the model (see below) | 8 |
| `engine.kv_capacity` | no | KV pool capacity in tokens per rank. The startup memory plan checks the configured model, slots and context before allocation | 8192 |
| `engine.kv_dtype` | no | GLM-5.3 latent storage: `bf16`, `fp8` or `fp4`. The index cache stays FP8; Qwen and GLM-4.7 K/V caches stay BF16 | `bf16` |
| `engine.mtp_depth` | no | Draft tokens per step, 1–3; requires `mtp`. Depth 1 verifies two rows. Deeper batching is model-dependent | 1 |
| `engine.ngram_table` | no | Qwen n-gram storage: `resident` copies the table to the GPU; `mmap` gathers rows from its NVMe-backed mapping | `resident` |
| `engine.dense_weights` | no | Qwen dense projections: `checkpoint` keeps BF16 weights; `fp8` encodes them to block FP8 at load time | `checkpoint` |
| `engine.default_max_tokens` | no | `max_tokens` for requests that omit it | 256 |
| `engine.queue_limit` | no | Maximum queued requests; excess requests receive 503 `overloaded` | 64 |
| `engine.max_connections` | no | rank 0's open-connection cap | 64 |
| `engine.no_eos` | no | ignore the model's end-of-sequence tokens (measurement runs only) | false |
| `engine.decode_graph` | no | Use the resident model and CUDA graph decode engine; supported at world sizes 1, 2 and 4 when the model fits | false |
| `engine.mtp` | no | Enable speculative decoding; requires `decode_graph` | false |
| `engine.graph_batch_min_live` | no | Live-request threshold for selecting a batch graph. The engine uses the smallest available batch covering the active slots; 0 selects min(2, `max_concurrency`) | 0 |
| `engine.sampling_candidates` | no | the sampled pick's per-rank candidate width in [1, 256]; narrower falls back to the exact gather more often | 128 |
| `engine.prefix_cache_gib` | no | Snapshot arena size per rank in GiB; slot size depends on the model and world size. 0 disables prefix caching | 1.5 |
| `engine.admission` | no | `full` reserves prompt plus `max_tokens` at admission; `grow` reserves prompt plus `admission_window` and grows on demand, shedding the youngest request at exhaustion | `full` |
| `engine.admission_window` | no | the tokens `grow` reserves past the prompt | 256 |
| `engine.bulk_pace_gbps` | no | the prefill all-reduce's sender pacing per queue pair; a negative value derives it from the port rate, 0 is unpaced | derived |
| `engine.bulk_inflight` | no | bulk stripes in flight per lane; a negative value takes the bus default (4) | default |
| `engine.rendezvous_timeout_ms` | no | how long the peers may take to join the bus world after rank 0 listens | 120000 |
| `engine.stats_interval_s` | no | the period of the throughput line in every rank's log; 0 turns it off | 10 |
| `engine.reasoning_in_content` | no | fold the reasoning into `content` with the model's own `</think>` instead of `reasoning_content` | false |
| `paths.resident_cache` | no | Resident image directory. Empty uses `~/.cache/dgpp/resident`; `DGPP_RESIDENT_CACHE_DIR` overrides both. Image size depends on the model and rank | `""` |

The application allows up to eight request slots. GLM-5.3-Flash and Qwen
use at most eight batched decode rows: eight plain requests or four
depth-1 MTP requests. Their deeper MTP steps use scalar graphs. GLM-4.7
supports batches of up to 32 rows, with `1 + mtp_depth` rows per request.
The startup log reports the selected shape and rejects unsupported sizes.

Sampling defaults come from the checkpoint's `generation_config.json`
and per-request fields. Flags such as `--temperature` override them for
measurement runs. See [operations](docs/operations.md) for memory planning,
MTP depth tradeoffs and deployment checks.

## Source layout

Model implementations share the engine, service, scheduler and text stack:

| directory | namespace | what |
|---|---|---|
| `src/serve/` | `dgpp::serve` | the HTTP server, the OpenAI-compatible service, the admission journal, the throughput line, the text frontend interface |
| `src/sched/` | `dgpp::sched` | the scheduler (admission, the tick, cancel and stop), the prefix cache's index, the `SchedulerEngine` interface every model implements |
| `src/sample/` | `dgpp::sample` | the sampler's host math: penalties, the logit bias, masks, top-k/p, the exact prefix decision |
| `src/text/` | `dgpp::text` | the tokenizer (HF tokenizer.json), the chat template (Jinja), the tool grammar and parser, the JSON-schema grammar |
| `src/kernels/` | `dgpp` | CUDA kernels; `pick.cu` and `sample_pick.cu` are the device pick and sampler any model's logits can use, the rest carry their model's name |
| `src/net/` | `dgpp::net` | the RoCE collective bus, TCP, the roster |
| `src/engine/` | `dgpp` | shared session interfaces, eager and graph engines, speculative decoding, memory plans and prefix arenas |
| `src/models/qwen/`, `src/models/glm4/` | `dgpp` | Qwen3.8-Flash-Next and GLM-4.7 configuration, loaders, layers and sessions |
| `src/models/glm/` | `dgpp` | GLM itself: the forward pass, the loader and resident image, the MoE and mHC layers, the eager and graph engine adapters, the MTP draft, the bus reducers |
| `src/models/` | `dgpp` | the KDA and DSA layer libraries and the quantized matrix, shared by any model that uses them |
| `src/loaders/`, `src/core/`, `src/common/` | `dgpp` | safetensors, the HF cache, JSON; arenas and tracing; logging and process memory |

The server binary is `dgpp-serve` (`apps/dgpp_serve.cpp`); the GLM
tools keep their names (`glm_gen_check`, `glm_forward_check`, …).

`tools/` holds the Python checkpoint and reference tooling the tests use;
`scripts/` the fabric and serving operations; `apps/` the binaries;
`benchmarks/` the probes and the engineering record; `deploy/` the cluster
config.

## Documentation

| page | what |
|---|---|
| `docs/operations.md` | booting, stopping and watching the serving world; memory and the image cache on a node; failure semantics |
| `docs/benchmarks.md` | every measured serving number: per model, world, concurrency and prompt class, with the method to reproduce each |
| `docs/signoff_v1.md` | the v1 performance and hardening sign-off, with every measurement |
| `docs/next_steps.md` | what is worth doing next, ranked by cost and benefit |
| `docs/tools.md` | serving, testing and diagnostic commands |
| `docs/testing.md` | the test suites and what each proves |
| `docs/numerics.md` | judging a numerics change; the sampling-width gate |
| `docs/mtp.md` | speculative decode with the MTP layer |
| `docs/measurements.md` | the curated platform measurements from the first milestones |
| `docs/checkpoint_budget.md` | the checkpoint's generated inventory and budget |
| `docs/batched_mtp_graph_stall.md` | a worked stall investigation, from symptom to root cause |
| `docs/qwen38_flash_next_plan.md` | the second family: Qwen3.8-Flash-Next's architecture, placement, decisions and progress |
| `docs/glm47_plan.md` | the third family: GLM-4.7 (NVFP4) — architecture, the modelopt format contract, decisions, gates and status |
| `DESIGN.md` | architecture and implementation contracts |
| `PLAN.md` | the milestones, their exit gates and status |
| `benchmarks/README.md`, `benchmarks/results/` | the probes, and the dated engineering record of every measurement and fix |
| `CHANGELOG.md` | the history by milestone |

## Contributing

`CONTRIBUTING.md` describes the build presets, the test discipline, the
evidence a fabric change needs, the code style and where things go.

## License

Apache License 2.0. See `LICENSE`.
