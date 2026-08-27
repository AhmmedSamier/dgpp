# GLM-5.3-Flash Checkpoint Budget Report

- Source: `/home/user/.cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-FP8/snapshots/a160e2291674d9e3e92e98fd82faa2544a2867a3`
- Files: 62 shards; tensors: 76108
- **Total on-disk weights: 328.3 GB**

## Bytes by class

| class | count | GB | share |
|---|---:|---:|---:|
| moe_expert | 72576 | 304.48 | 92.7% |
| mtp | 1760 | 7.49 | 2.3% |
| other | 692 | 5.81 | 1.8% |
| attn_lin | 68 | 4.56 | 1.4% |
| dense_mlp | 93 | 1.31 | 0.4% |
| lm_head | 1 | 1.27 | 0.4% |
| embed | 1 | 1.27 | 0.4% |
| shared_expert | 252 | 1.06 | 0.3% |
| attn_dsa | 177 | 0.90 | 0.3% |
| router | 84 | 0.10 | 0.0% |
| mhc | 270 | 0.07 | 0.0% |
| norm | 90 | 0.00 | 0.0% |
| attn_dsa_fp8 | 44 | 0.00 | 0.0% |

## By dtype

| dtype | GB |
|---|---:|
| F8_E4M3 | 314.40 |
| BF16 | 13.85 |
| F32 | 0.08 |

## Per-token active-traffic estimate (FP8 + BF16 mix, bs=1 decode)

- routed experts (42 MoE layers, top-8 of 288): **8.46 GB/token**
- shared expert (always-on): 1.06 GB/token
- dense MLPs (3 layers): 1.31 GB/token
- DSA attention projections: 0.90 GB/token (11 layers)
- linear-attn projections (BF16 in checkpoint): 4.56 GB/token (34 layers)
- linear-attn auxiliaries (A_log/dt/gates/convs): 5.81 GB/token
- lm_head matmul (bf16, full matrix): 1.27 GB/token
- router+mHC misc: 170 MB/token

- **Total ≈ 23.5 GB/token cluster-wide**
- TP=4 share ≈ **5.89 GB/node** → decode floor ≈ 26 ms/token @230 GB/s ⇒ ~39 tok/s pre-speculation
- ⚠ Optimization headroom: BF16 linear-attn projections could be requantized to FP8 (~saves most of the bf16 traffic) — quality-gated track for later.

## Unmatched names (692):
- `model.language_model.layers.0.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.0.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.0.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.0.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.0.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.0.self_attn.g_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.0.self_attn.g_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.0.self_attn.k_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.0.self_attn.o_proj.weight` [BF16] (4096, 8192)
- `model.language_model.layers.0.self_attn.q_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.0.self_attn.q_proj.weight` [BF16] (8192, 4096)
- `model.language_model.layers.0.self_attn.v_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.1.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.1.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.1.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.1.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.1.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.1.self_attn.g_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.1.self_attn.g_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.1.self_attn.k_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.1.self_attn.o_proj.weight` [BF16] (4096, 8192)
- `model.language_model.layers.1.self_attn.q_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.1.self_attn.q_proj.weight` [BF16] (8192, 4096)
- `model.language_model.layers.1.self_attn.v_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.10.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.10.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.10.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.10.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.10.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.10.self_attn.g_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.10.self_attn.g_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.10.self_attn.k_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.10.self_attn.o_proj.weight` [BF16] (4096, 8192)
- `model.language_model.layers.10.self_attn.q_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.10.self_attn.q_proj.weight` [BF16] (8192, 4096)
- `model.language_model.layers.10.self_attn.v_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.11.self_attn.o_proj.weight` [F8_E4M3] (4096, 16384)
- `model.language_model.layers.12.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.12.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.12.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.12.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.12.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.12.self_attn.g_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.12.self_attn.g_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.12.self_attn.k_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.12.self_attn.o_proj.weight` [BF16] (4096, 8192)
- `model.language_model.layers.12.self_attn.q_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.12.self_attn.q_proj.weight` [BF16] (8192, 4096)
- `model.language_model.layers.12.self_attn.v_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.13.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.13.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.13.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.13.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.13.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.13.self_attn.g_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.13.self_attn.g_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.13.self_attn.k_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.13.self_attn.o_proj.weight` [BF16] (4096, 8192)
- `model.language_model.layers.13.self_attn.q_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.13.self_attn.q_proj.weight` [BF16] (8192, 4096)
- `model.language_model.layers.13.self_attn.v_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.14.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.14.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.14.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.14.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.14.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.14.self_attn.g_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.14.self_attn.g_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.14.self_attn.k_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.14.self_attn.o_proj.weight` [BF16] (4096, 8192)
- `model.language_model.layers.14.self_attn.q_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.14.self_attn.q_proj.weight` [BF16] (8192, 4096)
- `model.language_model.layers.14.self_attn.v_conv1d.weight` [BF16] (8192, 1, 4)
- `model.language_model.layers.15.self_attn.o_proj.weight` [F8_E4M3] (4096, 16384)
- `model.language_model.layers.16.self_attn.A_log` [F32] (64,)
- `model.language_model.layers.16.self_attn.b_proj.weight` [BF16] (64, 4096)
- `model.language_model.layers.16.self_attn.dt_bias` [F32] (8192,)
- `model.language_model.layers.16.self_attn.f_a_proj.weight` [BF16] (128, 4096)
- `model.language_model.layers.16.self_attn.f_b_proj.weight` [BF16] (8192, 128)
- `model.language_model.layers.16.self_attn.g_a_proj.weight` [BF16] (128, 4096)
