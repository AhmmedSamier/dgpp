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
| GLM-5.3-Flash | [unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8) | 4 | Copy the [base template](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1.example.json) to `cluster_glm-5.3-flash_fp8_w4_mtp1.json` and set `model` to the linked FP8 repository |
| GLM-5.3-Flash (hybrid) | [HawkBearPig/GLM-5.3-Flash-NVFP4-FP8](https://huggingface.co/HawkBearPig/GLM-5.3-Flash-NVFP4-FP8) | 4 | [Four nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.example.json) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 2, 4 | [Two nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w2_mtp1.example.json), [four nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w4_mtp1.example.json) |
| Qwen3.8-Flash-Next | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 1 | [One node, BF16 dense](deploy/cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1.example.json), [one node, FP8 dense](deploy/cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1_dense-fp8.example.json) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) | 4 | [Four nodes](deploy/cluster_glm-4.7_nvfp4_w4_mtp1.example.json) |

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
The base and large-cache GLM-5.3 templates both use
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. The setup command below downloads it
once on rank 0 and syncs the selected snapshot to peers. To use the FP8 release
instead, set
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

## Quickstart

Use an internet-connected Spark as rank 0. First get the source:

```bash
git clone https://github.com/HawkBearPig/dgpp.git
cd dgpp
```

Run the remaining commands in this same shell, after installing the
[dependencies on each node](docs/getting-started.md#1-install-the-dependencies).
This example serves GLM-5.3-Flash on four Sparks; choose a different
[deployment template](deploy/README.md) for another model or node count.
See [Getting started](docs/getting-started.md) for the full walkthrough and troubleshooting guidance.

### 1. Configure your deployment

Copy the model template and create your site file, preserving existing files.
Edit `.env` to set `DGPP_NODES` (rank 0 first) and `DGPP_SSH_USER`, then
[verify SSH-key access from rank 0 to each peer](docs/getting-started.md#3-set-your-node-addresses-and-ssh-user).

```bash
CONFIG=deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json
test -f "$CONFIG" || cp "${CONFIG%.json}.example.json" "$CONFIG"
test -f .env || cp .env.example .env
```

Set `CONFIG` again if you open a new shell; it is the deployment filename, not a persistent setting.

### 2. Select the RoCE lanes (multiple nodes only)

Discover the interfaces, then copy the appropriate device names and GID indices
into `.env`, matching lane subnet order across nodes; discovery does not test RDMA connectivity.

```bash
python3 scripts/discover_roce.py --config "$CONFIG"
```

### 3. Build

Build the server on rank 0; the launcher stages the executable on peers.
If CUDA is not found, see [compiler setup](docs/getting-started.md#5-build-the-server-on-rank-0).

```bash
cmake --preset ci
cmake --build --preset ci --target dgpp_serve_app -j 4
```

### 4. Download the checkpoint

Download once into rank 0's standard Hugging Face cache, then sync peers sequentially.
Add `--sync-only` if rank 0 already has the checkpoint.
For this example, allow roughly **250 GiB per node** for the checkpoint and
one resident cache; check [storage and offline options](docs/getting-started.md#6-download-once-and-sync-to-peers) first.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-download.txt
python scripts/download_model.py --config "$CONFIG"
```

### 5. Check and start

Fix any failed preflight checks, then start the deployment and wait for `READY`.
The API stays on localhost; it has no authentication or TLS.

```bash
python3 scripts/dgpp-cluster doctor --config "$CONFIG"
python3 scripts/dgpp-cluster up --config "$CONFIG"
```

### 6. Send a request

Ask the running model a question:

```bash
MODEL=$(curl --fail -s http://127.0.0.1:18080/v1/models | jq -r '.data[0].id')
jq -n --arg model "$MODEL" \
  '{model:$model,max_tokens:160,messages:[{role:"user",content:"Name three primary colors."}]}' \
  | curl --fail http://127.0.0.1:18080/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

Stop it when finished, using the same config:

```bash
python3 scripts/dgpp-cluster down --config "$CONFIG"
```

## Release and install

A release is a versioned tarball installed once per node. Starting an
installed release stages the configuration and uses the installed binary.

```bash
scripts/release.sh                                 # build the release preset, stage, verify, pack
scripts/dgpp-cluster install dist/dgpp-VERSION.tar.zst --config "$CONFIG"
scripts/dgpp-cluster up --release VERSION --config "$CONFIG"
scripts/dgpp-cluster releases --config "$CONFIG"
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
quoted: no shell commands or variable expansion are evaluated. The site helper
loads an allowlist of settings, so credentials such as `HF_ACCESS_TOKEN` stay out of the
scripts' environment and generated server config.

| `.env` key | purpose | default |
|---|---|---|
| `DGPP_NODES` | Space-separated hostnames/IPv4 addresses in rank order. The first is rank 0, where launch/download commands run. A deployment uses its first `world_size` nodes. These are host addresses, not RDMA device names. | required |
| `DGPP_SSH_USER` | Peer login for binary staging, process control, diagnostics and checkpoint sync. Needs SSH-key access and write access to the configured directories. | current login when empty or absent |
| `DGPP_HTTP_PORT` | Default client-facing API TCP port on rank 0; deployment `http.port` overrides it. | 18080 |
| `DGPP_HTTP_BIND` | Default IPv4 listening address on rank 0; deployment `http.bind_host` overrides it. Keep localhost unless you have arranged access protection. | `127.0.0.1` |
| `DGPP_FABRIC_PORT` | Rank-0 TCP rendezvous listener used to establish the inter-node transport. Peers must reach it; clients do not use it. Keep it private to the cluster. | 29970 |
| `DGPP_JOURNAL_PORT` | Rank-0 TCP listener that distributes ordered scheduler operations to peers. Must differ from the fabric/API ports; keep it private to the cluster. | 29971 |
| `DGPP_LOG_DIR` | Base directory on rank 0 for logs, process records and collected peer logs. The launcher adds a deployment-specific subdirectory. | `~/dgpp/log` |
| `DGPP_STAGE_DIR` | Base directory on peers for staged development binaries, runtime config and logs. The launcher adds a deployment-specific subdirectory; it is not the model cache. | `/tmp/bus4` |
| `DGPP_RELEASE_DIR` | Base directory on each node for versioned installed server releases. Use a persistent, writable location. | `~/dgpp/releases` |
| `DGPP_BUILD_DIR`, `DGPP_DATA_DIR` | Development binary and evaluation-data locations. Relative paths use the repository root; change only for a custom build/data layout. | `build-ci`, `data` |
| `HF_HUB_CACHE`, `HF_HOME` | Downloaded checkpoints. An explicit hub cache wins; otherwise use `HF_HOME/hub`. Leave unset for the standard cache or choose a disk with room for the full checkpoint on each node. | `~/.cache/huggingface/hub` |
| `DGPP_RESIDENT_CACHE_DIR` | Per-node disk cache of prepacked weight images for faster reloads, separate from the HF checkpoint. Takes precedence over deployment `paths.resident_cache`. | `~/.cache/dgpp/resident` |
| `DGPP_ROCE_DEVICES` | One or two ordered local verbs device names, not Linux interface names. Use `discover_roce.py`; match lane subnets in the same order across nodes. | discover active Ethernet devices |
| `DGPP_ROCE_GID_INDICES` | One RoCE-v2 address-table index per explicitly selected device, in the same order. Pin these when a device offers several networks; discover the values on each host. | automatic RoCE-v2 selection |
| `DGPP_NODE_OVERRIDES` | JSON map keyed by exact `DGPP_NODES` entries, with per-node device/GID/hub-cache/resident-cache values. Use when peers have different NIC names or disks; see `.env.example`. | none |

Model settings live in deployment JSONs under `deploy/`. Each has a
`world_size` of 1, 2 or 4 and uses that many nodes from the beginning of
`DGPP_NODES`. Too few nodes is an error, not a fallback to a smaller world.
Select the deployment explicitly with `--config FILE`.
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
scripts/dgpp-cluster resolve --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json
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

### Deployment and endpoint

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `model` | yes | Exact Hugging Face repository ID, such as `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. Selects the checkpoint tensors, tokenizer and chat template. It is not a local filesystem path or a generic quant name. | — |
| `world_size` | yes | Number of participating nodes/ranks: 1, 2 or 4. Uses the first N entries of `DGPP_NODES`; each rank stores its tensor-parallel share in memory. Choose a supported model/world pair, not an arbitrary smaller number to save machines. | — |
| `http.bind_host` | no | IPv4 address on rank 0 that accepts API connections. `127.0.0.1` is local-only; a LAN address exposes that interface; `0.0.0.0` exposes all IPv4 interfaces. The server has no authentication/TLS. Does not select the RoCE interface. | `DGPP_HTTP_BIND`, otherwise `127.0.0.1` |
| `http.port` | no | TCP port clients use for the API, in 1–65535. Change it if the default is occupied, then update client URLs. Overrides the site HTTP port and must differ from fabric/journal ports in a multi-node deployment. | `DGPP_HTTP_PORT`, otherwise 18080 |
| `release` | no | Installed software version to run on every node, not a model revision. Select a version under `DGPP_RELEASE_DIR/dgpp-VERSION`; use it to upgrade or roll back server binaries. `--release` overrides this field; `--bin` overrides binary selection. | Empty: use the development build |
| `paths.resident_cache` | no | Disk directory for prepacked per-rank weight images, which speed subsequent loads. This is separate from the Hugging Face download cache and consumes additional disk space. Prefer the shared `DGPP_RESIDENT_CACHE_DIR` site setting unless a deployment needs its own directory. | Empty: `~/.cache/dgpp/resident`; `DGPP_RESIDENT_CACHE_DIR` takes precedence |

### Request capacity and memory

For a first run, retain the template values. Tune `kv_capacity` for context
space and `max_concurrency` for simultaneous work; these are different limits.
Startup checks the combined memory plan before loading.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.max_concurrency` | no | Maximum actively executing requests, not TCP connections or queued requests. More slots can improve aggregate throughput but use more state/scratch memory and may increase per-request latency. Allowed range is 1–8, subject to model/MTP row limits below. | 8 |
| `engine.kv_capacity` | no | Shared context-token pool on each rank, across active requests—not a separate allowance for every request. A prompt and its generated answer must fit. Increase for longer contexts or more simultaneous context; memory use increases and allocation is rounded to model block boundaries. | 8192 tokens |
| `engine.kv_dtype` | no | GLM-5.3 latent-cache precision: `bf16`, `fp8`, or `fp4`. Lower precision reduces latent storage at a numerical-accuracy cost; it does not quantize model weights. The index cache stays FP8. Qwen and GLM-4.7 K/V caches remain BF16. | `bf16` |
| `engine.default_max_tokens` | no | Answer-token budget for requests that omit `max_tokens`. Clients may supply their own value; this is not a global hard limit. A larger default also reserves more context space under full admission. | 256 |
| `engine.queue_limit` | no | Maximum requests waiting for an execution slot or memory budget. Additional arrivals receive HTTP 503 `overloaded`. Increase to tolerate bursts, at the cost of longer waits—not higher execution capacity. | 64 |
| `engine.max_connections` | no | Maximum simultaneously open HTTP connections on rank 0, including idle keep-alive connections and streams. Excess connections receive 503. Size this separately from active request slots. | 64 |
| `engine.prefix_cache_gib` | no | Memory budget per rank for reusable prefix-state snapshots. Repeated conversation prefixes can skip prefill work; larger budgets retain more snapshots but leave less memory for other state. Set 0 to disable. This is not the on-disk resident weight cache. | 1.5 GiB |
| `engine.admission` | no | When to reserve context space. `full` reserves prompt plus the requested answer budget before admitting a request. `grow` starts with a smaller reservation and extends it during generation; if space runs out, the youngest request is shed. Use `full` for predictable reservations, `grow` to trade that guarantee for denser occupancy. | `full` |
| `engine.admission_window` | no | Answer-token reservation increment used by `grow` admission. Larger increments reduce growth frequency but reserve more space ahead of use. Has no effect under `full`. Must be positive. | 256 tokens |

### Execution and performance

These settings change how the model runs. Use the matching deployment template
first; change one setting at a time and measure the effect on your workload.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.decode_graph` | no | Use CUDA graph replay for decode to reduce CPU launch overhead. On one node, this selects resident graph serving instead of the eager streaming path. Required for MTP and the single-Spark Qwen templates; leave enabled for the documented serving configurations. | false |
| `engine.mtp` | no | Enable multi-token prediction: draft candidate tokens, then verify them with the main model. Can reduce decode time when drafts are accepted, but adds draft state and verification work. Requires `decode_graph`; not a larger request batch. | false |
| `engine.mtp_depth` | no | How many tokens to draft per speculative step, 1–3. Greater depth can accept more tokens per pass, but uses more verification rows and can waste work when drafts are rejected. Values above 1 require MTP; supported batching varies by model. | 1 |
| `engine.ngram_table` | no | Qwen n-gram embedding-table placement. `resident` keeps it in device-accessible memory; `mmap` leaves it on local NVMe and fetches needed rows through the host page cache. Use `mmap` for the single-Spark templates to fit the model; its speed depends on storage/page-cache behavior. No effect on GLM models. | `resident` |
| `engine.dense_weights` | no | Qwen dense-projection storage. `checkpoint` retains the checkpoint's BF16 form; `fp8` converts dense projections at load time to reduce their memory footprint, with quantization error. Does not select another HF repository or change the expert quant. No effect on GLM models. | `checkpoint` |
| `engine.graph_batch_min_live` | no | Active-request count at which decode switches from scalar to batched graphs. A lower threshold starts batching earlier; batching may improve throughput while doing extra padded-row work. 0 chooses min(2, `max_concurrency`); explicit values must be 1 through `max_concurrency`. | 0 (automatic) |
| `engine.sampling_candidates` | no | Number of candidate tokens gathered per rank on the sampled-token fast path, 1–256. Smaller values reduce routine work but may trigger more full-gather fallbacks. The fallback preserves sampling correctness; this is not the client's `top_k` parameter. | 128 |
| `engine.bulk_pace_gbps` | no | Sender pacing rate per queue pair for bulk/prefill communication, in gigabits per second. Negative derives the rate from link speed; 0 disables pacing. Override only when measuring network contention—this is not an API throughput limit. | -1 (automatic) |
| `engine.bulk_inflight` | no | Maximum in-flight bulk stripes per lane. More can keep the link busy but increase pressure on buffers and competing traffic. -1 uses the transport default (4); normally leave automatic. | -1 (automatic) |
| `engine.rendezvous_timeout_ms` | no | How long ranks may take to join the transport rendezvous. Increase for slow cold starts or delayed peers; it does not increase an HTTP request's timeout. | 120000 ms |
| `engine.stats_interval_s` | no | Interval between periodic server throughput/state log lines. Shorter intervals give finer operational visibility and more log output. Set 0 to disable periodic statistics. | 10 seconds |
| `engine.reasoning_in_content` | no | Put reasoning text in the response's `content`, separated by the model's `</think>` marker, instead of a separate `reasoning_content` field. Use only for clients that need that combined format; it does not disable reasoning. | false |
| `engine.no_eos` | no | Ignore the model's end-of-sequence token so measurement runs continue to their token budget. Leave false for normal serving, where a model should be allowed to finish its answer. | false |

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
