# DGPP packaged release

This package serves supported checkpoints on Linux/aarch64 DGX Spark (GB10).
It includes the server and CUDA runtime libraries, but not model weights.
Run the following commands from the extracted package root on rank 0.

## Getting started

### 1. Install the dependencies

1. On every node, check `nvidia-smi` works and Python is at least 3.10.
   Use the NVIDIA driver supplied with your DGX OS installation.
2. Install the runtime and transfer tools on every node:

   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-venv rdma-core libibverbs1 \
     ibverbs-providers libnl-3-200 libnl-route-3-200 libstdc++6 \
     openssh-client rsync curl zstd iproute2
   ```

The package bundles cudart and cuBLASLt; no compiler or CUDA toolkit is needed
to run it. Keep libibverbs installed even for a single node: the server links
it. Multi-node production uses it for RoCE communication. The build baseline
is Ubuntu 24.04; older distributions may not provide compatible glibc/libstdc++.

### 2. Verify the package and choose a deployment

1. Run `sha256sum -c MANIFEST.sha256`. Direct CMake installs have no manifest.
2. Copy **one** matching template to `deploy/cluster.json`. For example:

   ```bash
   # Four Sparks, GLM-5.3-Flash hybrid NVFP4/FP8:
   cp -n deploy/cluster.nvfp4.example.json deploy/cluster.json
   ```

   Other choices include `cluster_qwen_w2.example.json` for two Sparks and
   `cluster_qwen_spark1_fp8.example.json` for one.
3. Check the JSON's `model` and `world_size`. Leave its engine settings
   unchanged for the first run. Pass this file explicitly with `--config`.

### 3. Configure the nodes and fabric

1. Run `test -f .env || cp .env.example .env`.
2. Edit `DGPP_NODES` in `.env`: use space-separated addresses, rank 0 first,
   or `127.0.0.1` for a single node. Set `DGPP_SSH_USER` for peers.
3. Run `ssh USER@PEER_ADDRESS hostname` for each peer, verify its host key,
   and arrange SSH-key login without a password prompt.
4. For multiple nodes, run:

   ```bash
   python3 scripts/discover_roce.py --config deploy/cluster.json
   ```

5. Copy the appropriate device names and GID indices into `.env`. Match
   lane order by subnet across nodes; choose one or two lanes. Merge per-node
   differences into `DGPP_NODE_OVERRIDES`. See [networking](docs/networking.md).

Do not source `.env`; scripts parse it as data. Leave cache paths unset
to use the standard `~/.cache/huggingface/hub` on each node.

### 4. Download once and sync

On rank 0:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-download.txt
python scripts/download_model.py --config deploy/cluster.json
```

Use `hf auth login` first if authentication is required. Only rank 0 downloads
from Hugging Face; it syncs the selected snapshot and blobs to peers
sequentially using rsync/SSH. Credentials and other cached revisions are not
copied. Peers need space for the full checkpoint plus their resident cache.

If rank 0 already has the checkpoint, use `--sync-only` instead. Use
`--verify-only` to check active snapshots on all nodes without downloads or
writes. Add `--local-only` to download or verify without contacting peers.

### 5. Install and start

For a **single node**, use the extracted binary:

```bash
python3 scripts/dgpp-cluster doctor --config deploy/cluster.json --bin bin/dgpp-serve
python3 scripts/dgpp-cluster up --config deploy/cluster.json --bin bin/dgpp-serve
```

For **multiple nodes**, install the same tarball on each node using the
launcher. Replace `VERSION` with the value in `MANIFEST`:

```bash
python3 scripts/dgpp-cluster install /path/to/dgpp-VERSION.tar.zst --config deploy/cluster.json
python3 scripts/dgpp-cluster doctor --config deploy/cluster.json --release VERSION
python3 scripts/dgpp-cluster up --config deploy/cluster.json --release VERSION
```

Fix failed doctor checks before starting. Installation uses `DGPP_RELEASE_DIR`
and refuses to replace an existing version. Named releases use each node's
installed binary and bundled libraries; `--bin` stages only the executable,
so peers need compatible runtime libraries separately.

### 6. Check and stop the service

```bash
curl --fail http://127.0.0.1:18080/v1/models
python3 scripts/serve_api_check.py 127.0.0.1 18080
python3 scripts/dgpp-cluster status --config deploy/cluster.json
python3 scripts/dgpp-cluster down --config deploy/cluster.json
```

HTTP defaults to localhost; deployment `http.bind_host` and `http.port` select
another address or port. Use the same config and any `--log-dir` override for
start, status and stop. Use an SSH tunnel or authenticated TLS proxy for remote
clients; DGPP has no authentication or TLS.

Run `python3 scripts/dgpp-cluster paths --config deploy/cluster.json` to locate
logs. A running deployment must be stopped before another `up`, or explicitly
replaced with `up --replace`. Stop jobs from older launchers with their
original launcher; unrecorded processes are not adopted or killed.

Source-only benchmarks and reference generators are not included. Resident
weight caches are built locally at first load; allow additional NVMe space.
