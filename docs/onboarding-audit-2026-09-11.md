# Fresh-checkout onboarding audit

Tested on 2026-09-11 against published commit
`21aa4e1aee662faefc0075d6c3c5f1b42080d999`.
The observations below describe that baseline; see
[follow-up verification](#follow-up-verification) for the subsequent fixes.

The documented sequence did not reach a running four-node service on the test
host. Source configuration failed until CUDA's compiler directory was added
to `PATH`; internet access and SSH trust also blocked the normal setup path.
After isolated workarounds, the server built successfully, local preflight
passed, and all seven Python CTest suites passed. Four-node startup, RDMA
traffic, checkpoint distribution and the first inference request remain
unverified by this audit.

## Test environment and boundaries

- Remote host: `stephen@192.168.88.12`, hostname `gx10-e9cd`, NVIDIA GB10,
  Linux/aarch64. This peer was treated as rank 0 for the new checkout.
- Checkout: `/tmp/dgpp-onboarding.7kcm7EiX/repo` on that host.
- Compiler/runtime baseline: GCC 13.3, CMake 3.28.3, Python 3.12.3 and
  CUDA 13.0.88. The documented system packages were already installed except
  `ninja-build`. The build used Make and succeeded without Ninja.
- This was a clean source checkout and a new Python venv, not a clean OS or
  an empty model cache. Existing cached checkpoints were reused read-only.
- The chosen deployment was
  `cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json`, copied from its
  template without changing engine settings.
- The test `.env` was created from scratch, with the test host first in
  `DGPP_NODES`. Log, staging and resident-cache paths were isolated under the
  test directory. No existing `.env`, token, SSH key or SSH configuration was
  copied or changed.
- No system packages were installed, network configuration changed, full
  model weights downloaded/transferred, or serving processes started/stopped.
  The memory-plan check exited without forming a world or loading weights.

## What happened

| Step | Observed result |
|---|---|
| Clone from GitHub on the peer | Failed: `Could not resolve host: github.com`. The peer had no default route; GitHub, PyPI and Hugging Face names did not resolve. |
| Obtain a clean checkout despite that failure | Cloned the public repository on the original head, made a Git bundle, transferred it, and cloned the bundle on the peer. Verified the published commit and a clean tracked worktree. This is a workaround, not a documented step. |
| Copy the deployment and site template | Worked. `cp -n` printed a portability warning. |
| Discover RoCE across the cluster | Failed on the first SSH peer with `Host key verification failed`. No inventory was printed, including the local inventory already collected. |
| Discover RoCE locally | Worked. Reported two active devices with GID index 3, on `192.168.88.*` and `192.168.89.*`. |
| `cmake --preset ci` | Failed with `Failed to detect a default CUDA architecture`. The configure log identified `CMAKE_CUDA_COMPILER-NOTFOUND`. |
| Configure with CUDA on `PATH`, then build | Passed with the workaround shown below and the documented `dgpp_serve_app -j 4` target. No source changes were needed. |
| Create the Python venv | Passed. |
| Install `requirements-download.txt` from PyPI | Failed on DNS, ultimately reporting `No matching distribution found`. Retries/timeouts were shortened for this test. |
| Install the same requirements from transferred wheels | Passed; installed `huggingface-hub` 1.31.0. `pip check` reported no broken requirements. This was another offline workaround. |
| Verify the existing local checkpoint | Passed with `--verify-only --local-only`, even before installing the HF package. Revision `17b8d4d83e33e836e9549d7e1a8d25b5117203c1`, 181.7 GiB of indexed weights. |
| Exercise the HF download helper without network or writes to weights | Passed with `HF_HUB_OFFLINE=1`, `--local-only`, and the already populated cache. This does not validate a first download. |
| Verify peer checkpoints | Local verification passed, then SSH host-key verification blocked the first peer. |
| Local doctor after the build | Passed with zero failed checks; warned about automatic RoCE selection. All binary dependencies resolved. |
| Cluster doctor after the build | All three remote probes failed SSH host-key verification. It also incorrectly reported a RoCE lane-count mismatch. |
| Memory plan for the default four-node template | Passed on the test rank: 92.46 GiB planned, plus 8 GiB headroom, against 116.59 GiB available. This does not establish that every rank fits. |
| Python tests from the clean checkout | All seven CTest suites passed. |
| Start, first request and shutdown | Not attempted after cluster preflight failed, following the guide's instruction to fix failed checks before starting. |

## Findings and suggested improvements

### 1. Explain how to make an installed CUDA toolkit discoverable

The [dependency instructions](getting-started.md#1-install-the-dependencies)
check `nvcc --version` and direct users to the DGX OS installation if it is
missing. Here, CUDA was already installed, but its compiler was absent from
the default SSH shell's `PATH`. `nvcc --version` failed, and the subsequent
CMake error mentioned architecture rather than the missing compiler.

The successful workaround was:

```bash
PATH=/usr/local/cuda/bin:$PATH cmake --fresh --preset ci
cmake --build --preset ci --target dgpp_serve_app -j 4
```

The first failed configure log is retained at
`/tmp/dgpp-onboarding.7kcm7EiX/failed-configure.yaml` on the peer.

Add a short installed-but-not-on-PATH branch, explain `CUDACXX` or the CUDA
installation path, and explain how to retry a failed configure. Ideally,
configuration should report a missing CUDA compiler directly.

### 2. Make rank-0 network requirements explicit

This host has reachable RoCE addresses but no internet route. Moving it to
the front of `DGPP_NODES` does not make it ready to fetch source, packages or
models. This is an environment limitation, not a Git or downloader defect.

The setup guide should say that the machine performing initial downloads
needs working DNS and outbound access to GitHub, the package repositories,
PyPI and Hugging Face. Peers need cluster connectivity, not independent Hub
downloads. Include a short pre-check and either an offline preparation path
or a clear statement that the quickstart assumes an internet-connected head.

The transferred Git bundle and wheelhouse demonstrated that source builds
and package installation can proceed without changing this peer's networking.

### 3. Give concrete SSH setup and recovery instructions

The [SSH section](getting-started.md#3-set-your-node-addresses-and-ssh-user)
correctly says to verify host keys and arrange key-based login. It does not
show how to do that or how to distinguish an unknown host key from failed
authentication. The test peer could receive SSH from the original head but
could not run the same checks in the opposite direction.

Spell out that access must work **from the selected rank 0 to every peer**.
Provide a verified-host-key workflow, key installation instructions, and a
batch-mode check before discovery/download. Do not suggest disabling host-key
verification. Authentication beyond the host-key failure was not tested here.

### 4. Do not turn an unsuccessful probe into a lane-count mismatch

The full doctor reported three SSH failures followed by:

```text
FAIL RoCE lane counts differ between nodes
```

In [cluster_doctor.py](../scripts/cluster_doctor.py), failed probes receive
an empty device list, and lane-count comparison includes those lists. No
remote device count was actually observed. This sends a newcomer toward
network-interface changes when the actionable problem is SSH trust.

Compare counts only for successfully inspected nodes, mark the remaining
counts unknown, and retain the preflight summary even when comparison fails.

### 5. Correct the missing-checkpoint recovery message

With `HF_HUB_CACHE` temporarily pointed at an empty test directory, local
doctor reproduced this instruction:

```text
download the complete checkpoint on this node
```

The message in [cluster_doctor.py](../scripts/cluster_doctor.py) conflicts
with the documented download-once/sync-peers workflow. It should direct users
to `download_model.py --config FILE` on rank 0, or `--sync-only` when the head
already has the snapshot. For a newcomer, the low-level `refs/main` error is
also less useful than a plain missing-checkpoint explanation plus that command.

### 6. Preserve useful RoCE output when one peer fails

[discover_roce.py](../scripts/discover_roce.py) collects all inventories
before printing anything. One SSH failure discards the useful local result
and prevents checking later peers. Running without `--config` did show the
local devices, but the error did not suggest that fallback.

Print available inventories alongside per-node errors and return failure if
any node remains unchecked. Label devices as locally eligible rather than
simply `usable`; neither local GIDs nor carrier establish cluster-wide RDMA
connectivity. Show the configured or automatically selected lane order
separately from other candidates.

### 7. Do not silently fall back when an explicit config argument is empty

In a fresh shell, the quickstart's `CONFIG` variable is unset. Reproducing
`dgpp-cluster resolve --config "$CONFIG"` then selected the base GLM filename,
not the large-cache filename used in the quickstart. Because only the latter
existed in the checkout, the result was a missing-config error naming an
unexpected file.

[dgpp-cluster](../scripts/dgpp-cluster) treats an explicitly empty argument
the same as an omitted one. If the base config existed, another deployment
could be selected instead; that consequence follows from the code and was
not tested by starting a service.

Reject an empty explicit `--config`, and add a brief same-shell note to the
quickstart. The full getting-started guide already explains resetting `CONFIG`
in a new terminal, but the abbreviated path omits that warning.

### Smaller clarity issues

- Neither entry point shows `git clone` and `cd`; both assume the reader
  already has the source. A short clone command would close that gap.
- The detailed guide puts three alternative `CONFIG=` assignments in one
  copyable block. Copying the block chooses the last, four-node deployment.
  Separate the alternatives or provide one clearly selected example.
- `cp -n` emitted `behavior of -n is non-portable and may change in future`.
  It worked here, but a guard followed by a normal copy would avoid warning
  about a basic setup command.
- Distinguish head build dependencies from peer runtime dependencies. The
  guide asks for compiler/toolkit/build packages on every node even though
  only rank 0 builds. Ninja was absent here and was not required by the
  documented preset's Make-based build.
- Make the download size visible before starting. The default checkpoint
  verified as 181.7 GiB, before the separate resident cache. The detailed
  guide explains the storage categories, but a newcomer still has to find
  a concrete size elsewhere. Doctor reports free space rather than checking
  whether the complete intended download and resident cache will fit.

## Follow-up order

First address CUDA discovery, rank-0 connectivity/SSH instructions, and the
two misleading doctor errors. Then improve discovery's partial results and
empty-config handling. Keep the README short by linking those recovery paths
from the relevant quickstart steps.

A full four-node acceptance run still needs verified SSH access from the
selected head to all peers. A genuine first-download test additionally needs
an empty cache and working outbound access, or an explicitly prepared offline
source. No claim of end-to-end inference validation should be made from this
audit alone.

The temporary checkout, build, venv, bundle and wheelhouse were retained for
reproduction. The remote checkout and venv occupy about 610 MiB; its wheelhouse
is about 6.8 MiB. Local transfer inputs are under
`/tmp/dgpp-onboarding-source.6rnts8D2`. No fixes were applied during the initial
audit.

## Follow-up verification

The follow-up changes address the seven findings above:

- CMake finds the conventional CUDA installation even when an SSH shell omits
  it from `PATH`. Explicit compiler/toolkit choices take precedence, and missing
  compilers produce a setup instruction rather than an architecture error.
- Setup now distinguishes head build tools from peer runtime dependencies and
  covers clone/cd, outbound connectivity, verified head-to-peer SSH login,
  offline preparation, separate deployment choices and non-overwriting copies.
- Doctor treats failed node inventories as unknown, compares only observed lane
  counts and always prints its summary. Missing-checkpoint recovery points to
  rank 0's download-and-sync command.
- Discovery retains successful inventories, continues after a failed peer,
  reports per-node errors and displays the configured/automatic device order.
  Its local-eligibility labels do not claim cluster connectivity.
- All five Python CLI entry points that accept `--config` reject empty or
  whitespace-only explicit arguments. The quickstart explains the shell
  variable's lifetime.
- Storage guidance shows the default checkpoint's measured size and a roughly
  250 GiB per-node planning budget for one deployment. Doctor still reports
  free space rather than enforcing a combined future disk-space estimate;
  the guide states this limitation explicitly.

The patched remote checkout configured with its unchanged SSH `PATH` and
rebuilt `dgpp_serve_app`. Discovery retained the local inventory while reporting
all three SSH trust failures. Cluster doctor reported exactly three failed
probes, unknown remote lane counts and its summary, without a false mismatch.
Pointing the local cache setting at an absent test directory produced the new
rank-0 recovery message. An empty explicit config failed at argument parsing.

The final remote rebuild passed all seven Python CTest suites, local doctor
reported zero failed checks and read-only checkpoint verification passed.
A separate configure with explicit `CUDACXX` also succeeded. The local
host-labelled CTest run passed all 14 suites. Regression coverage
includes partial discovery after SSH failure or timeout, automatic and explicit
device order, single-node discovery, real versus unknown lane-count differences,
checkpoint recovery, empty config arguments and missing CUDA setup. Existing
tests still check one Hub download followed by ordered peer transfers and use a
tiny fixture with real rsync to verify synchronization without real weights.

The test host's network and SSH trust were deliberately left unchanged.
A first internet download, real multi-node checkpoint distribution, RDMA startup
and inference still require a suitable test allocation; they are not established
by these checks.
