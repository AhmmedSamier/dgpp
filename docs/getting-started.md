# Getting started

Run these steps on the first Spark in your cluster (rank 0). Commands assume
you are in the repository root. A supported deployment uses one GB10 per node
on Linux/aarch64; choose a model and node count from the
[supported configurations](../README.md#supported-models-and-configurations).
For the abbreviated command sequence, see the [quickstart](../README.md#quickstart).

## 1. Install the dependencies

1. On **every node**, check that the NVIDIA driver and CUDA 13 toolkit are installed:

   ```bash
   nvidia-smi
   nvcc --version
   ```

   Use the DGX OS driver/toolkit installation if either is missing. Keep the
   CUDA runtime and cuBLASLt versions compatible across nodes.

2. On **every node**, install the build and cluster tools:

   ```bash
   sudo apt-get update
   sudo apt-get install build-essential cmake ninja-build pkg-config \
     python3 python3-venv libibverbs-dev rdma-core libibverbs1 ibverbs-providers \
     openssh-client rsync curl jq zstd iproute2
   ```

3. Check `cmake --version` is at least 3.25 and `python3 --version` is at least
   3.10. GCC 13, CMake 3.28.3 and CUDA 13.0.88 on Ubuntu 24.04 are the tested
   baseline.

Keep libibverbs installed even for one node: the current server links it.
Multi-node production uses it for RoCE communication. Turning
`DGPP_ENABLE_IBV` off omits the server; it is not a single-node build option.
`rsync` is required on both ends of checkpoint transfers.

## 2. Copy a deployment template

1. Run **one** assignment matching your hardware. `CONFIG` is just a shell
   variable for the filename; each command below still passes `--config` explicitly.

   ```bash
   # One Spark: Qwen NVFP4 with FP8 dense projections.
   CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1_dense-fp8.json

   # Two Sparks: Qwen FP8.
   CONFIG=deploy/cluster_qwen-3.8-flash-next_fp8_w2_mtp1.json

   # Four Sparks: GLM-5.3-Flash hybrid NVFP4/FP8.
   CONFIG=deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json
   ```

2. Copy the matching template without replacing an existing local file:

   ```bash
   cp -n "${CONFIG%.json}.example.json" "$CONFIG"
   ```

3. Open the file named by `CONFIG`. Confirm `model` and `world_size` match your
   choice. Leave the engine settings unchanged for the first run. In a new
   terminal, set `CONFIG` again or pass the full filename to `--config`.

Names include the full model, checkpoint quant, world size and decode mode.
`plain` disables MTP; `mtp1`/`mtp2` specify the draft depth. `dense-fp8` means
load-time dense conversion, not a different checkpoint; `large-cache` selects
the larger KV/prefix budgets. See [deployment filenames](../deploy/README.md) for
the complete list and old-to-new names.

## 3. Set your node addresses and SSH user

1. Create the site file if it does not already exist:

   ```bash
   test -f .env || cp .env.example .env
   ```

2. Open `.env`. For **one Spark**, set `DGPP_NODES="127.0.0.1"`.
   For **multiple Sparks**, replace the example addresses with your machines'
   addresses, separated by spaces. Put the machine running these commands
   first. Every node must be able to reach the first address.
3. Set `DGPP_SSH_USER` to the login used on the other nodes. That account needs
   write access to its cache and staging directories.
4. For each peer, run `ssh USER@PEER_ADDRESS hostname`. Verify its host key and
   arrange SSH-key login so subsequent commands do not prompt for a password.

Keep any existing credentials in `.env`. Do not run `source .env`: scripts
read the allowed settings as data, without executing shell expressions.
The first `world_size` entries in `DGPP_NODES` participate in the deployment.

## 4. Discover the RoCE devices (multiple nodes only)

1. Run:

   ```bash
   python3 scripts/discover_roce.py --config "$CONFIG"
   ```

2. Read the device-to-interface mapping, IP addresses, MTUs and GID indices for
   each node. DGX Spark verbs names may look like `rocep1s0f0` and
   `roceP2p1s0f0`; copy your discovered names, not these examples.
3. Copy the suggested `DGPP_ROCE_DEVICES` and, where provided,
   `DGPP_ROCE_GID_INDICES` into `.env`. Choose one or two usable devices.
4. Match the **subnet order** across nodes. Lane 0 on each node must reach
   the other nodes' lane 0, and likewise for lane 1. Use the printed
   `DGPP_NODE_OVERRIDES` when device names or GID indices differ.

Discovery does not change networking or test RDMA traffic. If no usable device
appears, check cabling, interface IP assignments and the RDMA driver. If more
than two appear, select the pair connected to the intended fabric. With these
settings omitted, serving auto-selects active Ethernet RDMA devices; explicit
selection avoids choosing an unintended network.

## 5. Build the server on rank 0

```bash
cmake --preset ci
cmake --build --preset ci --target dgpp_serve_app -j 4
```

This creates `build-ci/dgpp-serve`. The launcher copies it to peers; their CUDA
and verbs runtime libraries must already be installed. Set `DGPP_BUILD_DIR`
in `.env` only if you built into a different directory.

## 6. Download once and sync to peers

1. On rank 0, prepare the downloader:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install -r requirements-download.txt
   ```

2. If the model requires authentication, run `hf auth login` on rank 0.
   Public checkpoints do not require a login.
3. Run **one** command on rank 0:

   ```bash
   python scripts/download_model.py --config "$CONFIG"
   ```

The script downloads the complete checkpoint to rank 0, then syncs its selected
snapshot and referenced blobs to each peer sequentially over rsync/SSH. Peers
do not download from Hugging Face. Matching files are checksum-checked and
reused; interrupted transfers can be resumed by rerunning the command.
Other models, other cached revisions, `.env`, and HF login tokens are not copied.

The default on each node is the standard `~/.cache/huggingface/hub` directory.
Leave the cache settings unset to use it. `HF_HUB_CACHE`, `HF_HOME`, and
per-node cache overrides are supported if you need another disk. Each node
needs space for the **full checkpoint**, its resident image cache, and temporary
transfer files—not just its tensor-parallel share of the download.

If the checkpoint is already downloaded on rank 0, skip the Hub entirely:

```bash
python scripts/download_model.py --config "$CONFIG" --sync-only
```

Check the active snapshot on every node without modifying files:

```bash
python scripts/download_model.py --config "$CONFIG" --verify-only
```

Use `--local-only` to verify or download on rank 0 without contacting peers.
For a pinned revision, use `--revision COMMIT --activate` with the download
command. Sync selects the same revision on peers after verification. Stop
deployments using the checkpoint before changing it. Structural verification
checks metadata and shard lengths; rsync checks file contents during sync.

## 7. Check the deployment and start it

```bash
python3 scripts/dgpp-cluster doctor --config "$CONFIG"
python3 scripts/dgpp-cluster up --config "$CONFIG"
```

Fix any failed preflight checks before starting. `doctor` checks GPU/platform,
libraries, cache files, writable paths, ports and RoCE selection without
starting ranks. Add `--local-only` to check rank 0 without SSH. It does not
prove end-to-end RDMA connectivity; startup checks the model's memory plan.

Wait for `READY`. A cold load takes longer than a resident-cache boot.
HTTP defaults to `127.0.0.1:18080`. To change it, add or edit the deployment's
`http` object, for example `"http": {"bind_host": "127.0.0.1", "port": 18081}`.
Keep localhost for now: the server has no TLS or authentication.

## 8. Send a request, then stop

1. On rank 0, check the endpoint and ask a question:

   ```bash
   curl --fail http://127.0.0.1:18080/v1/models
   MODEL=$(curl --fail -s http://127.0.0.1:18080/v1/models | jq -r '.data[0].id')
   jq -n --arg model "$MODEL" \
     '{model:$model,max_tokens:160,messages:[{role:"user",content:"Name three primary colors."}]}' \
     | curl --fail http://127.0.0.1:18080/v1/chat/completions \
       -H 'Content-Type: application/json' --data-binary @-
   ```

2. Inspect or stop the deployment with the same config:

   ```bash
   python3 scripts/dgpp-cluster status --config "$CONFIG"
   python3 scripts/dgpp-cluster down --config "$CONFIG"
   ```

Use the URL printed at startup if you changed the HTTP port or bind address.
For access from another machine, run
`ssh -N -L 18080:127.0.0.1:18080 USER@HEAD_ADDRESS` on that machine, then
use its localhost endpoint. Shared access needs an authenticated TLS proxy;
see [networking](networking.md).

To find logs, run `python3 scripts/dgpp-cluster paths --config "$CONFIG"`.
Use the same config path and any `--log-dir` override for start, status and stop.
`up` refuses an already running deployment; `up --replace` explicitly stops it
first. Stop processes created by an older launcher with that launcher before
upgrading—the current launcher does not adopt unrecorded processes.

## Optional development and evaluation tools

These are not needed to serve a model:

| Task | Additional setup |
|---|---|
| Host/Python tests | Build the test targets, then select the `host`/`python` CTest labels; see [testing](testing.md). `ci-local.sh` also runs GPU/RDMA suites, so use idle test hardware. |
| Tokenizer/template goldens and token-ID preparation | Install `requirements-tools.txt` in a venv. PyTorch reference modes need a compatible PyTorch installation separately. |
| Evaluation datasets | Run `python3 scripts/prepare_data.py download`. Token fixtures: `python3 scripts/prepare_data.py tokens --config "$CONFIG" --text /path/to/long-prompt.txt`. |
| HumanEval | Use an isolated evaluation environment and explicitly pass `--allow-code-execution`. Generated Python runs without a security sandbox. |
| Profiling | Install Nsight Systems (`nsys`). |
| Formatting | Install clang-format to enable the CMake formatting targets. |
| Release packaging | `tar`, `zstd`, `sha256sum`, Git and the CUDA toolkit on the packaging host. The package bundles cudart/cuBLASLt; peers still need the driver, verbs providers, libnl, libstdc++ and glibc. |
