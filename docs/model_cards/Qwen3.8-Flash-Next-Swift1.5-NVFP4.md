# Swift 1.5 Qwen3.8-Flash-Next NVFP4

[kaushikvira/Qwen3.8-Flash-Next-swift15-nvfp4-dgpp](https://huggingface.co/kaushikvira/Qwen3.8-Flash-Next-swift15-nvfp4-dgpp)
is a community NVFP4 conversion of
[ukisai/Swift1.5-Qwen3.8-Flash-Next](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next),
a fine-tune of Qwen3.8-Flash-Next. The model and fine-tune belong to their
respective authors; kaushikvira contributed the conversion and deployment
template. DGPP serves it through the existing Qwen loader.

## Checkpoint and configuration

The checkpoint uses ModelOpt NVFP4 backbone experts with groups of 16, an
FP8 n-gram table and BF16 dense tensors. Its MTP experts are a fused BF16
gate/up pair and down projection, so the template sets
`engine.mtp_expert_format: "bf16_fused"`; the loader converts these draft
experts to FP8. The NVIDIA release instead ships separate FP8 draft experts,
so changing between these repositories also requires changing this setting.

The [two-node template](../../deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json)
maps the n-gram table from NVMe, encodes dense projections to FP8 at load,
uses the MMA vocabulary head and enables MTP depth 4. It has four request
slots, a shared 1,048,576-token BF16 KV pool and a 42 GiB prefix cache.
The pool is shared; the checkpoint's 262,144-token context limit still
applies to each request. Busy and idle prefill caps are both 4,096 tokens;
the current adaptive prefill controller operates within those caps.

The publisher supplies placeholder expert activation scales of `1.0`.
Use the default W4A16 expert path. The checkpoint has no calibrated scales
for the experimental `DGPP_MOE_W4A4=1` path.

## Run on two nodes

Configure the node addresses, SSH user and cache paths in `.env` as described
in [Getting started](../getting-started.md). The download command fetches
the checkpoint once on rank 0 and synchronizes it to the second node.
Use a Python environment with `requirements-download.txt` installed.

```bash
cp deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json \
   deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.json
CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.json
.venv/bin/python scripts/download_model.py --config "$CONFIG"
python3 scripts/dgpp-cluster up --config "$CONFIG"
python3 scripts/serve_api_check.py 127.0.0.1 18080
```

Use the configured HTTP port if it differs from 18080. To measure an idle
deployment, prepare the GSM8K prompt data and follow the
[benchmark procedure](../benchmarks.md#reproduce):

```bash
python3 scripts/prepare_data.py download --tasks gsm8k
python3 scripts/timed_load.py 127.0.0.1 18080 \
  --concurrency 1,2,4,8 --classes all --repeat 3 --max-tokens 256 \
  --json-out /tmp/swift15-decode.json
python3 scripts/serve_prefill_probe.py 127.0.0.1 18080 2048 8192 32768 \
  --repeat 3 --no-think --tag "swift15-$(date -u +%Y%m%dT%H%M%S)" \
  --json-out /tmp/swift15-prefill.json
python3 scripts/dgpp-cluster down --config "$CONFIG"
```

Keep detailed outputs outside tracked documentation. Report the source and
checkpoint revisions with any measurements; changing weights can change
MTP acceptance and throughput even when architecture and quantization match.

## Scope and limitations

The original contributor reported serving and throughput checks in
[PR #100](https://github.com/HawkBearPig/dgpp/pull/100). Those measurements
do not establish throughput equivalence to the NVIDIA release or validate
the upstream fine-tune's token-efficiency claim. A quality comparison needs
matched tasks, sampling settings and completion limits; a successful load
and API check alone cannot establish task accuracy.

The source model is gated and identifies its license as
`swift-open-license-1.0`; consult the upstream terms and the conversion's
model card. The template depends on the contributor's external repository.
