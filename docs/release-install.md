# DGPP packaged release

This package serves supported checkpoints on Linux/aarch64 DGX Spark (GB10).
The binary and CUDA runtime libraries are included; model weights are not.
Read [runtime dependencies](docs/dependencies.md) before installing on peers.

From this package's root:

1. Verify `sha256sum -c MANIFEST.sha256` for a release tarball. A direct
   CMake install has no release manifest.
2. Merge `.env.example` into `.env`, preserving existing credentials.
   Set node addresses in rank order, SSH user, cache paths, and RoCE settings.
3. Choose a matching `deploy/*.example.json` and set `DGPP_CLUSTER_CONFIG`
   in `.env`, or copy it to `deploy/cluster.json`. World size must fit both
   the selected model and your machines.
4. Download the complete checkpoint on every node. Use a venv with
   `requirements-download.txt` and `python scripts/download_model.py`.
5. For a single node, run `python3 scripts/dgpp-cluster doctor --bin bin/dgpp-serve`,
   then `python3 scripts/dgpp-cluster up --bin bin/dgpp-serve`.

For multiple nodes, use `scripts/dgpp-cluster install PATH/TO/dgpp-VERSION.tar.zst`
to install the same verified package under `DGPP_RELEASE_DIR` on every node.
Installation refuses to replace an existing version directory.
Then run `scripts/dgpp-cluster doctor --release VERSION` and
`scripts/dgpp-cluster up --release VERSION`. The version is in `MANIFEST` and
`bin/dgpp-serve --version`. A named release uses each node's installed binary
and bundled libraries; a `--bin` launch stages only the executable.

HTTP defaults to localhost. Check `curl --fail http://127.0.0.1:18080/v1/models`.
Use `python3 scripts/serve_api_check.py 127.0.0.1 18080` for inference checks.
Stop with `scripts/dgpp-cluster down`, using the same config and log override.
Use an SSH tunnel or authenticated TLS proxy for remote clients; DGPP itself
has no authentication. See [network configuration](docs/networking.md).

The package includes all deployment templates, launcher modules, the download
helper, and runtime/network documentation. Source-only benchmarks and reference
generators are not included. `status` inspects recorded processes; `paths`
prints the deployment-specific log and staging directories. Use `--local-only`
with `doctor` to omit SSH checks. A running deployment must be stopped before
another `up`, or explicitly replaced with `up --replace`. Old launcher jobs
without process-identity records must be stopped using their original launcher.
Resident caches are built locally at first load; allow NVMe space beyond the
checkpoint itself.
