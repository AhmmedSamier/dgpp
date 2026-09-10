# DGPP

DGPP is a from-scratch C++/CUDA inference engine that serves
`unsloth/GLM-5.3-Flash-FP8` (and, on the same engine, `Qwen/Qwen3.8-Flash-Next-FP8`
and `nvidia/GLM-4.7-NVFP4`) across four NVIDIA DGX Spark (GB10) systems over a
RoCE fabric, behind an OpenAI-compatible HTTP API. Nothing sits underneath
it: the CUDA kernels, the RDMA collective bus, the tokenizer, the Jinja
chat-template interpreter, the scheduler, the prefix cache and the service
are all in this tree. Every rank executes an identical op stream by
construction, and the tests and the operations tooling prove it on every
run.

## Highlights

- **Tensor-parallel resident serving** on four nodes: each rank holds its
  slice of the model resident on the GPU and boots from a per-rank image
  cache in 15–25 s.
- **One CUDA graph per decode step**, the collectives as graph nodes, with
  adaptive scalar and row-batched variants; **MTP speculative decoding**
  inside the same replay, its transcript identical to plain decode.
- **An exact prefix cache**: a conversation's next turn attaches to a
  snapshot of the previous one, bitwise equal to a cold prefill.
- **The OpenAI Chat Completions contract**: streaming, tools with
  constrained decoding as a guarantee, `response_format` with JSON schemas,
  `reasoning_content`, logprobs, `stop`, `n`, `logit_bias`, usage details.
- **Deterministic across ranks**: admissions journaled from the head, every
  tick's op-stream fold checked on every peer, the four-way op-stream hash
  at shutdown.
- **Fail-fast failure semantics**, drilled: any rank's death fails the
  service within seconds and a restart reproduces the committed tokens.
- **Operable**: one cluster config, versioned releases installed once per
  node, a throughput line every 10 s in each rank's log, `/v1/metrics`;
  every boot checks its memory plan against the node before allocating,
  and the KV cache's dtype (bf16, fp8, fp4) is a config key.

## Performance

Measured on the four-node fabric at TP=4 (the method and every number:
`docs/signoff_v1.md`).

| measure | result |
|---|---|
| decode, one request, plain graph | 31.45 ms per token |
| decode, one request, MTP (greedy, 77–97 % drafts accepted by class) | 21.3–23.9 ms per token |
| aggregate, 8 live requests at T=1 | 76 tokens/s |
| aggregate, 4 live requests under MTP | 60 tokens/s |
| prefill, 256 / 2,048 / 32,768 tokens | 0.58 s / 1.7 s / 33.7 s |
| boot from the resident image cache | 15–25 s |

## Status

Version 0.1.0 (2026-09-06). Milestones M0–M9 of `PLAN.md` are closed:
the platform probes, the runtime and loader, the KDA and DSA operators,
the assembled forward, four-rank tensor parallelism over the collective
bus, generation and the API, the prefix cache, transactional MTP, and the
optimization and hardening pass with its one-hour soak. The engine serves
one model on one fabric; `docs/next_steps.md` ranks what remains by cost
and benefit.

## Requirements

- NVIDIA DGX Spark (GB10) with CUDA 13 and an SM 12.1-capable compiler
- CMake 3.24 or newer and a C++20 compiler
- CUDA Runtime and cuBLASLt development files
- libibverbs headers and library for the RoCE bus and probes; disable with
  `-DDGPP_ENABLE_IBV=OFF` when unavailable (the GDR and RC probes are then
  omitted)
- Python 3.10 or newer (standard library only) for the tooling and scripts

## Quick start

### Build and test

```bash
./scripts/ci-local.sh            # the ci preset: warnings as errors, build, the whole suite
```

The presets are `release`, `debug`, `asan`, `ubsan` and `ci`
(`cmake --preset <name>`, `cmake --build --preset <name> -j`,
`ctest --preset <name>`). Run a full build before `ctest`: the suite runs
whatever binaries exist. With clang-format installed, `format` and
`format-check` targets exist. `docs/testing.md` lists the suites and what
each proves.

### Serve

1. Copy `deploy/cluster.example.json` to `deploy/cluster.json` (local to
   your site, not tracked) and describe your world in it: the nodes in
   rank order, the model, the engine knobs (every key: the table below).
2. `scripts/dgpp-cluster up` — stages the binary and the config, boots
   rank 0, then the peers, and waits for the listening line.
3. Send a request:

   ```bash
   curl -s http://192.0.2.11:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "unsloth/GLM-5.3-Flash-FP8", "reasoning_effort": "low", "max_tokens": 160,
     "messages": [{"role": "user", "content": "Name three primary colors, one per line."}]}'
   ```

4. `scripts/dgpp-cluster down` — drains, stops every rank and prints one
   md5 per rank's op stream. The four must be identical.

`docs/operations.md` is the operator's page: boot and stop, what a node
needs, what happens when a rank dies, how to tell the world is healthy.

## Release and install

A release is one versioned tarball, installed once per node and run many
times; nothing is copied at boot.

```bash
scripts/release.sh                                 # build the release preset, stage, verify, pack
scripts/dgpp-cluster install dist/dgpp-<version>.tar.zst   # every node: unpack under release_dir, verify
scripts/dgpp-cluster up --release <version>        # runs <release_dir>/dgpp-<version>/bin/dgpp-serve on every rank
scripts/dgpp-cluster releases                      # what each node has installed
```

Which release runs is named, in the config's `release` key or with
`--release`, never inferred from a pointer on the nodes.

The version is the tree's, stamped at build time by `cmake/version.cmake`
into the binary (`dgpp-serve --version`; `0.1.0+g<sha12>`, `.dirty` when
the tree had uncommitted changes) and sent on the journal's settings
record, so a world of mixed versions refuses to form. Inside the tarball:

| path | what |
|---|---|
| `bin/dgpp-serve` | the server, rpath `$ORIGIN/../lib` |
| `lib/libcudart.so.13`, `lib/libcublasLt.so.13` | the CUDA runtime it was built against |
| `scripts/dgpp-cluster`, `scripts/serve_api_check.py` | the launcher and the request-field check |
| `deploy/cluster.example.json` | the config template a site copies and edits |
| `doc/README.md`, `doc/operations.md` | this page and the operator's page |
| `MANIFEST`, `MANIFEST.sha256` | version, git sha, CUDA, build host and date; every other file's checksum |

Beyond the tarball a node needs the NVIDIA driver, rdma-core, libnl and
libstdc++, all part of the DGX OS image, plus the checkpoint and resident
image caches on its local NVMe. `install` unpacks under `paths.release_dir`
(`~/dgpp/releases/dgpp-<version>/`) and checks every file against the
manifest; several versions sit side by side, and rolling back is naming
the previous one. There are no boot-time units by decision:
the world starts when an operator says `up`. With no release installed,
`up` stages the development binary `build-ci/dgpp-serve` to the peers as
before.

## Configuration

The serving world is described by one JSON file, `deploy/cluster.json`
(your site's copy of `deploy/cluster.example.json`; the copy is not
tracked), that every rank and the launcher read (`dgpp-serve --config FILE --rank R`
on each node; `scripts/dgpp-cluster up|down|status` boots, stops and
inspects the world from it). Flags given after `--config` override the
file. Every key is checked by name: an unknown key or a wrong type refuses
to start. The engine defaults below are the binary's own flag defaults;
the example states the production values explicitly. Every rank
digests its effective configuration (the model, the world, the fabric
ports and the engine knobs) and the journal's warm record refuses a world
whose ranks disagree.

| key | required | what it does | default |
|---|---|---|---|
| `model` | yes | the Hugging Face model id every rank loads (`--model`) | — |
| `nodes` | yes | the ranks' hosts in rank order; `nodes[0]` is the head (rank 0: the HTTP ingress and the journal); the world size is the list's length | — |
| `ssh_user` | no | the user the launcher uses for ssh and scp to the peers | the launcher's own user |
| `release` | no | the installed release version `up` runs on every rank (`<release_dir>/dgpp-<version>/bin/dgpp-serve`); launcher-only, `--release` overrides it; rolling back is naming the previous version | empty: the development binary `build-ci/dgpp-serve`, staged to the peers |
| `ports.http` | no | rank 0's OpenAI-compatible HTTP port | 18080 |
| `ports.fabric` | no | the bus rendezvous port rank 0 listens on and the peers connect to | 29970 |
| `ports.journal` | no | the admission journal's port (must differ from `ports.fabric`) | 29971 |
| `engine.max_concurrency` | no | request slots per rank (the decode rows; with `decode_graph`, slots × rows per request ≤ 8) | 8 |
| `engine.kv_capacity` | no | the KV pool in tokens per rank (a prompt plus its answer must fit). Before anything is allocated every rank checks its memory plan for this context against the node's free memory and refuses to boot, with the plan itemized and the largest context that would fit named, when it does not fit (`dgpp-serve --config … --rank R --memory-plan` runs the check alone) | 8192 |
| `engine.kv_dtype` | no | the latent cache's storage format: `bf16` (the parity gates' format), `fp8` (e4m3 with a per-row scale, half the bytes) or `fp4` (e2m1 in blocks of 16 with e4m3 block scales, ~0.28 of the bytes); the index cache stays fp8; smaller formats trade attention precision for context (`docs/operations.md`) | `bf16` |
| `engine.mtp_depth` | no | draft tokens per decode step with `mtp` (1–3): the verify runs 1 + depth rows, the draft block's chained rows propose the drafts after the first; depth 1 is the two-row step; depth 2 costs ~12 ms more per step (the third verify row's experts plus the chained block row) and pays only where the second draft lands often (measured 2026-09-06: code and JSON +4 %, prose −4 %), and past depth 1 every step is a scalar replay (the row batch is not built for it) — `docs/operations.md` | 1 |
| `engine.default_max_tokens` | no | `max_tokens` for requests that omit it | 256 |
| `engine.queue_limit` | no | the admission queue's bound; beyond it the door answers 503 `overloaded` | 64 |
| `engine.max_connections` | no | rank 0's open-connection cap | 64 |
| `engine.no_eos` | no | ignore the model's end-of-sequence tokens (measurement runs only) | false |
| `engine.decode_graph` | no | the one-graph decode step (the measured serving mode; needs a world larger than one) | false |
| `engine.mtp` | no | the speculative two-token step on the decode graph (needs `decode_graph`) | false |
| `engine.graph_batch_min_live` | no | live requests at which the adaptive engine switches from scalar graphs to a row batch — the smallest of the 2-slot, 3-slot and full batches that covers the live slots (2026-09-07); 0 means min(2, `max_concurrency`) | 0 |
| `engine.sampling_candidates` | no | the sampled pick's per-rank candidate width in [1, 256]; narrower falls back to the exact gather more often | 128 |
| `engine.prefix_cache_gib` | no | the prefix cache's snapshot arena per rank in GiB (one slot per session state, ~35 MiB each); 0 turns the cache off | 1.5 |
| `engine.admission` | no | `full` reserves prompt plus `max_tokens` at admission; `grow` reserves prompt plus `admission_window` and grows on demand, shedding the youngest request at exhaustion | `full` |
| `engine.admission_window` | no | the tokens `grow` reserves past the prompt | 256 |
| `engine.bulk_pace_gbps` | no | the prefill all-reduce's sender pacing per queue pair; a negative value derives it from the port rate, 0 is unpaced | derived |
| `engine.bulk_inflight` | no | bulk stripes in flight per lane; a negative value takes the bus default (4) | default |
| `engine.rendezvous_timeout_ms` | no | how long the peers may take to join the bus world after rank 0 listens | 120000 |
| `engine.stats_interval_s` | no | the period of the throughput line in every rank's log; 0 turns it off | 10 |
| `engine.reasoning_in_content` | no | fold the reasoning into `content` with the model's own `</think>` instead of `reasoning_content` | false |
| `paths.log_dir` | no | where the launcher writes the head's log, pid and every rank's fetched op stream and log | `~/dgpp/log` |
| `paths.stage_dir` | no | where the launcher puts the peers' binary and config (a `/tmp` path is emptied by a reboot and recreated at the next `up`) | `/tmp/bus4` |
| `paths.release_dir` | no | where installed releases live on each node (the install target) | `~/dgpp/releases` |
| `paths.resident_cache` | no | the resident image cache directory (~80 GiB per rank); empty means the binary's default, `~/.cache/dgpp/resident`, and `DGPP_RESIDENT_CACHE_DIR` in the environment wins over both | `""` |

Only rank 0 reads the `engine` keys, the model and the ports from the
file: it pushes them to every peer over the journal before any rank builds
its model, so the peers run what the head runs whatever their own file or
flags say (a peer's file supplies its bootstrap, `nodes[0]` and
`ports.journal`, and its `paths`). Sampling defaults (temperature, top-p
and the rest) are not cluster configuration: they come from the checkpoint's `generation_config.json`
and the per-request fields, with the `--temperature`-style flags as
overrides for measurement runs.

## Source layout

The tree separates what any model needs from what GLM is (the layout
split of 2026-09-06):

| directory | namespace | what |
|---|---|---|
| `src/serve/` | `dgpp::serve` | the HTTP server, the OpenAI-compatible service, the admission journal, the throughput line, the text frontend seam |
| `src/sched/` | `dgpp::sched` | the scheduler (admission, the tick, cancel and stop), the prefix cache's index, the `SchedulerEngine` seam every model implements |
| `src/sample/` | `dgpp::sample` | the sampler's host math: penalties, the logit bias, masks, top-k/p, the exact prefix decision |
| `src/text/` | `dgpp::text` | the tokenizer (HF tokenizer.json), the chat template (Jinja), the tool grammar and parser, the JSON-schema grammar |
| `src/kernels/` | `dgpp` | CUDA kernels; `pick.cu` and `sample_pick.cu` are the device pick and sampler any model's logits can use, the rest carry their model's name |
| `src/net/` | `dgpp::net` | the RoCE collective bus, TCP, the roster |
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
| `docs/signoff_v1.md` | the v1 performance and hardening sign-off, with every measurement |
| `docs/next_steps.md` | what is worth doing next, ranked by cost and benefit |
| `docs/tools.md` | every binary and script |
| `docs/testing.md` | the test suites and what each proves |
| `docs/numerics.md` | judging a numerics change; the sampling-width gate |
| `docs/mtp.md` | speculative decode with the MTP layer |
| `docs/measurements.md` | the curated platform measurements from the first milestones |
| `docs/checkpoint_budget.md` | the checkpoint's generated inventory and budget |
| `docs/batched_mtp_graph_stall.md` | a worked stall investigation, from symptom to root cause |
| `docs/qwen38_flash_next_plan.md` | the second family: Qwen3.8-Flash-Next's architecture, placement, decisions and progress |
| `docs/glm47_plan.md` | the third family: GLM-4.7 (NVFP4) — architecture, the modelopt format contract, decisions, gates and status |
| `DESIGN.md` | the architecture and every contract, kept as built |
| `PLAN.md` | the milestones, their exit gates and status |
| `benchmarks/README.md`, `benchmarks/results/` | the probes, and the dated engineering record of every measurement and fix |
| `CHANGELOG.md` | the history by milestone |

## Contributing

`CONTRIBUTING.md` describes the build presets, the test discipline, the
evidence a fabric change needs, the code style and where things go.

## License

Apache License 2.0. See `LICENSE`.
