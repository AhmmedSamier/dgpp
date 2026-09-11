# Getting started

DGPP targets Linux/aarch64 DGX Spark (GB10), with one rank per machine.
Other GPUs and architectures are not supported by the current build. Start
with a supported model/world combination; reducing world size is not a way
to make an arbitrary checkpoint fit. See the model-card table in the source
README and the templates in `deploy/`.

## 1. Prepare the machines

Install the [build and runtime dependencies](dependencies.md). Use a normal
user account. Run the launcher on rank 0; for multiple nodes, verify SSH to
each peer under that account, including host-key verification, before starting.
Peers need Python 3.10+, the runtime libraries, and the complete checkpoint.
A development launch copies the server binary, not its shared libraries.

## 2. Choose a deployment and site settings

From the repository root, copy `.env.example` to `.env` only if `.env` does
not exist. Otherwise merge the settings, keeping existing credentials.
Do not source `.env`: the scripts parse its allowlisted settings as literal data.

For a single Spark, set `DGPP_NODES="127.0.0.1"` and choose, for example,
`deploy/cluster_qwen_spark1_fp8.example.json`. Set
`DGPP_CLUSTER_CONFIG` to that relative path in `.env`, or copy the template
to `deploy/cluster.json` and edit your copy. Keep its mapped n-gram table
and graph settings: they are needed for this configuration to fit.

For two or four nodes, select the matching template and set `DGPP_NODES`
to their addresses in rank order and `DGPP_SSH_USER` to the remote login.
Each deployment uses the first `world_size` nodes. Configure the
[RoCE lanes and per-node paths](networking.md) before continuing.

Deployment JSONs support this optional HTTP section:

```json
"http": {"bind_host": "127.0.0.1", "port": 18080}
```

Absent fields use `DGPP_HTTP_BIND` and `DGPP_HTTP_PORT`, which default to
`127.0.0.1:18080`. The bind host must be an IPv4 address. Keep localhost
unless you have arranged network access controls; the server has no TLS
or authentication. This does not change the internal fabric addresses.

## 3. Build the server

```bash
cmake --preset ci
cmake --build --preset ci --target dgpp_serve_app -j 4
```

The output is `build-ci/dgpp-serve`. Set `DGPP_BUILD_DIR` if you use another
build directory. A packaged release does not need compilation; follow its
included README. Do not use `ci-local.sh` as an installation check: it also
runs GPU and fabric tests. See [test selection](testing.md).

## 4. Download and verify the checkpoint

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-download.txt
python scripts/download_model.py
python scripts/download_model.py --verify-only
```

Run the download on every participating node, using the same model and
revision. With shared site settings, pass `--rank N` on each peer to select
its cache override. Alternatively pass `--model ORG/NAME --cache-dir PATH`.
For gated repositories, authenticate locally with `hf auth login`; tokens
are not copied to peers or included in resolved JSON.

The helper downloads the full repository. DGPP reads `HF_HUB_CACHE`, then
`HF_HOME/hub`, then `~/.cache/huggingface/hub`. It uses `refs/main`, or the
only snapshot if that reference is absent; it will not guess between
multiple snapshots. To pin a revision, download with `--revision COMMIT
--activate` on each node. Activation explicitly changes that cache's
`refs/main`; stop any service using it first. `--verify-only` is offline
and checks metadata, indexed tensors and shard lengths, not content hashes.

Budget local NVMe for the full checkpoint on every node, plus resident image
caches and download headroom. Memory needs depend on model, rank count,
concurrency, context length and prefix capacity; there is no universal
free-memory threshold. A cold load takes longer than a resident-cache boot.

## 5. Check readiness, then start

```bash
python3 scripts/dgpp-cluster resolve
python3 scripts/dgpp-cluster doctor
python3 scripts/dgpp-cluster up
```

`doctor` checks the local node and SSH peers without installing files,
launching ranks, or killing processes. It checks GPU/platform, tools,
cache structure, writable storage, TCP bind availability, RoCE selection
and runtime libraries. `--local-only` omits SSH checks. It does not test
end-to-end RDMA traffic or prove that the requested memory plan fits.
`up` runs these checks automatically; `--skip-preflight` is an explicit
diagnostic escape hatch, not a way to bypass process ownership checks.

For a separate memory estimate, save `resolve` output to a new file and run
`build-ci/dgpp-serve --config FILE --rank 0 --memory-plan`. Check each rank's
plan for its intended machine. Startup also validates the plan before loading.

## 6. Send a request and stop

On rank 0 with the default endpoint:

```bash
curl --fail http://127.0.0.1:18080/v1/models
python3 scripts/serve_api_check.py 127.0.0.1 18080
python3 scripts/dgpp-cluster status
python3 scripts/dgpp-cluster down
```

The API check sends inference requests; it is not a passive health check.
Use the endpoint printed by `up` if you changed the bind or port. For remote
access, an SSH tunnel keeps the unauthenticated listener private:
`ssh -N -L 18080:127.0.0.1:18080 USER@HEAD`, then use localhost on your client.

`paths` prints the deployment's log and staging locations. They are
namespaced by the absolute deployment-file path. Use the same configuration,
site settings and explicit `--log-dir` when stopping as when starting.
`up` refuses a running deployment unless you pass `--replace`. Processes
started by older launchers have no identity records; stop them with their
original launcher before upgrading. The new launcher does not adopt them.

For benchmarks, prepare [datasets and fixtures](testing.md) first. For
ongoing use, see [operations](operations.md) and the source `scripts/README.md`.
