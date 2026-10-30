#!/usr/bin/env python3
"""Reference goldens for the Nemotron-Transcribe host oracle (milestone 2 of
docs/nemotron_transcribe_plan.md).

Builds a tiny nemotron_h with the walk's real structure — Mamba/Attention/
MoE slots (the MoE with n_group=2/topk_group=1, so the group policy is
pinned beyond the release's degenerate 1/1), a full NeoX RoPE, the F32
A_log/D/dt_bias triad, the stacked relu2 experts and the score-correction
bias — fills it from a fixed seed in fp32, and records what
src/models/nemotron_transcribe/reference.cpp replays: the whole-model walk
(embeddings through norm_f and the head), and the primitives the walk hides
(causal depthwise conv + silu, the chunk-scan reference, the gated norm,
the router) on direct inputs.

Runs on CPU: the reference chain is the pure-torch fallback
(no mamba_ssm, no triton, no kernels package), which is exactly the
semantics the oracle mirrors. The goldens are fp32 against fp32 — the
bf16 checkpoint path is pinned separately by the text goldens and the
device parity of milestones 3/4.

Usage:
  python3 tests/python/nemotron_transcribe_ref_oracle.py --emit \
      tests/data/nemotron_transcribe_reference_oracle.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h import modeling_nemotron_h as M

SEED = 20261031
# The tiny walk: six slots, every mixer kind twice, the MoE group policy live.
LAYERS = ["mamba", "attention", "moe", "mamba", "moe", "attention"]
DIMS = dict(
    vocab_size=97,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    mamba_num_heads=4,
    mamba_head_dim=16,
    n_groups=2,
    ssm_state_size=32,
    conv_kernel=4,
    n_routed_experts=16,
    num_experts_per_tok=6,
    moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=48,
    n_group=2,
    topk_group=1,
)
IDS = [5, 42, 7, 7, 96, 0, 23, 17]
DT_MIN = 0.003


def build_model():
    cfg = NemotronHConfig(
        **DIMS,
        layers_block_type=LAYERS,
        mtp_layers_block_type=["attention", "moe"],
        attn_implementation="eager",  # the softmax oracle, not sdpa
        rope_theta=10000,
        partial_rotary_factor=1.0,
        attention_bias=False,
        mamba_proj_bias=False,
        use_conv_bias=True,
        mamba_hidden_act="silu",
        mlp_hidden_act="relu2",
        mlp_bias=False,
        time_step_min=DT_MIN,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        norm_eps=1e-5,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(SEED)
    model = M.NemotronHForCausalLM(cfg).to(torch.float32).eval()
    # Everything the post_init zeroed needs live fp32 values; the scales keep
    # the dt clamp, the relu2 dead band and the router ties informative.
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if "A_log" in name:
                p.normal_(0.0, 1.0)
            elif "dt_bias" in name or name.endswith(".D"):
                p.normal_(0.5, 1.0)
            elif "e_score_correction_bias" in name:
                p.normal_(0.0, 0.3)
            elif "conv1d.weight" in name:
                p.normal_(0.0, 0.5)
            elif "conv1d.bias" in name:
                p.normal_(0.0, 0.3)
            elif p.dim() == 1 and "weight" in name and "norm" in name:
                p.normal_(1.0, 0.1)
            elif p.dim() == 1 and "bias" in name:
                p.normal_(0.0, 0.1)
            elif "experts" in name or "_proj" in name or "embeddings" in name or "lm_head" in name:
                p.normal_(0.0, 0.35)
    return model


def t(x):
    return x.detach().cpu().to(torch.float32).flatten().tolist()


def tensor_rec(x):
    return {"shape": list(x.shape), "data": t(x)}


def capture_walk(model):
    """Run the fixed prompt through the model, recording per-slot residual
    in, slot-norm out (= mixer in), mixer out, and the internals that pin a
    whole primitive at once (in_proj rows, gated-norm in/gate, q/k/v/o, the
    router triple, the routed/shared expert sums)."""
    rec = {}
    handles = []

    def out_hook(key, name):
        def fn(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            rec[key] = tensor_rec(x)
        return fn

    def in_hook(key, idx=0):
        def fn(_mod, inp):
            rec[key] = tensor_rec(inp[idx])
        return fn

    cfg = model.config
    for i, kind in enumerate(cfg.layers_block_type):
        kind = {"linear_attention": "mamba", "full_attention": "attention"}.get(kind, kind)
        blk = model.model.layers[i]
        p = f"llm.model.layers.{i}."
        handles.append(blk.register_forward_pre_hook(in_hook(p + "residual_in")))
        handles.append(blk.norm.register_forward_hook(out_hook(p + "norm_out", "norm")))
        handles.append(blk.mixer.register_forward_hook(out_hook(p + "mixer_out", "mixer")))
        mx = blk.mixer
        if kind == "mamba":
            handles.append(mx.in_proj.register_forward_hook(out_hook(p + "mixer.in_proj", "ip")))
            handles.append(mx.norm.register_forward_pre_hook(
                lambda _m, inp, p=p: rec.__setitem__(p + "mixer.gated_norm_in", tensor_rec(inp[0]))))
            handles.append(mx.norm.register_forward_pre_hook(
                lambda _m, inp, p=p: rec.__setitem__(p + "mixer.gated_norm_gate", tensor_rec(inp[1]))))
        elif kind == "attention":
            for nm in ("q_proj", "k_proj", "v_proj"):
                handles.append(getattr(mx, nm).register_forward_hook(
                    out_hook(p + f"mixer.{nm}", nm)))
        else:
            def gate_hook(_mod, _inp, out, p=p):
                logits, weights, indices = out
                w, idx = weights.detach().cpu(), indices.detach().cpu()
                order = torch.argsort(idx, dim=-1)
                rec[p + "mixer.router"] = {
                    "logits": tensor_rec(logits),
                    "index": idx.gather(1, order).tolist(),
                    "weight": w.gather(1, order).tolist(),
                }
            handles.append(mx.gate.register_forward_hook(gate_hook))
            handles.append(mx.experts.register_forward_hook(out_hook(p + "mixer.experts_out", "e")))
            handles.append(mx.shared_experts.register_forward_hook(
                out_hook(p + "mixer.shared_out", "s")))

    with torch.no_grad():
        out = model(input_ids=torch.tensor([IDS]))
    for h in handles:
        h.remove()
    return {"logits": tensor_rec(out.logits), "rec": rec}


def primitive_cases():
    torch.manual_seed(SEED + 1)
    cases = {}
    # causal depthwise conv + silu (the causal_conv1d_fn fallback).
    seq, ch, kern = 6, 24, 4
    u = torch.randn(seq, ch)
    w = torch.randn(ch, 1, kern)
    b = torch.randn(ch)
    got = F.conv1d(u.T.unsqueeze(0), w, b, padding=kern - 1, groups=ch)[:, :, :seq].squeeze(0).T
    cases["conv"] = {
        "seq": seq, "channels": ch, "kernel": kern,
        "u": t(u), "weight": t(w), "bias": t(b), "out": t(F.silu(got)),
    }
    # the chunk-scan reference, multi-chunk: heads 4, groups 2, head_dim 8,
    # state 16, chunk 3, seq 7 — dt clamped at the release's floor.
    h, g, hd, st_, cs, sq = 4, 2, 8, 16, 3, 7
    x = torch.randn(1, sq, h, hd)
    dt = torch.randn(1, sq, h)
    A_log = torch.randn(h)
    A = -torch.exp(A_log)
    B = torch.randn(1, sq, g, st_)
    C = torch.randn(1, sq, g, st_)
    D = torch.randn(h)
    dt_bias = torch.randn(h)
    got = M.mamba2_chunk_scan(x, dt, A, B, C, chunk_size=cs, D=D, dt_bias=dt_bias,
                              dt_softplus=True, dt_limit=(DT_MIN, float("inf")))
    cases["scan"] = {
        "seq": sq, "heads": h, "head_dim": hd, "groups": g, "state": st_, "dt_min": DT_MIN,
        "x": t(x), "dt": t(dt), "A_log": t(A_log), "B": t(B), "C": t(C),
        "D": t(D), "dt_bias": t(dt_bias), "out": t(got),
    }
    # the gated norm (silu-gate before the group variance).
    n, grp, eps = 32, 16, 1e-5
    y = torch.randn(sq, n)
    gate = torch.randn(sq, n)
    nw = torch.randn(n) * 0.3 + 1.0
    mod = M.NemotronHRMSNormGated(n, group_size=grp, eps=eps)
    with torch.no_grad():
        mod.weight.copy_(nw)
        got = mod(y, gate)
    cases["gated_norm"] = {
        "rows": sq, "n": n, "group": grp, "eps": eps,
        "y": t(y), "gate": t(gate), "weight": t(nw), "out": t(got),
    }
    # the router standalone: group policy live (2 groups, topk_group 1),
    # bias-corrected selection, weights from the unbiased sigmoid.
    torch.manual_seed(SEED + 2)
    E, H, topk = 16, 64, 6
    r = M.NemotronHTopkRouter.__new__(M.NemotronHTopkRouter)
    torch.nn.Module.__init__(r)
    r.top_k, r.num_experts, r.hidden_dim = topk, E, H
    r.weight = torch.nn.Parameter(torch.randn(E, H) * 0.35)
    r.routed_scaling_factor = 2.5
    r.num_group, r.topk_group = 2, 1
    r.norm_topk_prob = True
    with torch.no_grad():
        r.e_score_correction_bias = torch.randn(E) * 0.3
    xin = torch.randn(3, H)
    with torch.no_grad():
        logits, weights, indices = r(xin)
    order = torch.argsort(indices, dim=-1)
    cases["router"] = {
        "E": E, "H": H, "topk": topk, "n_group": 2, "topk_group": 1,
        "x": t(xin), "gate": t(r.weight), "bias": t(r.e_score_correction_bias),
        "logits": t(logits),
        "index": indices.gather(1, order).tolist(),
        "weight": weights.gather(1, order).tolist(),
    }
    return cases


def weights_map(model):
    """The tiny model's state dict, renamed to the checkpoint's convention
    (embeddings->embed_tokens, norm_f->norm, the experts to the stacked bmm
    layout, the F32 triad under _fp32_params)."""
    want = {}
    for name, v in model.state_dict().items():
        if name.startswith("lm_head."):
            key = "llm." + name
        elif name == "model.embeddings.weight":
            key = "llm.model.embed_tokens.weight"
        elif name == "model.norm_f.weight":
            key = "llm.model.norm.weight"
        else:
            key = "llm." + name
        if ".mixer.experts.up_proj" in key:  # not shared_experts (rooted .mixer.)
            key = key.replace("experts.up_proj", "experts.gate_and_up_projs")
            v = v.transpose(1, 2)
        elif ".mixer.experts.down_proj" in key:
            key = key.replace("experts.down_proj", "experts.down_projs")
            v = v.transpose(1, 2)
        elif (".mixer.A_log" in key or ".mixer.D" in key or ".mixer.dt_bias" in key):
            leaf = key.rsplit(".", 1)[1]
            key = key[: -len(leaf)] + "_fp32_params." + leaf
        want[key] = {"dtype": "f32", "shape": list(v.shape), "data": t(v)}
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True, help="output json path")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and assert the walk logits reproduce")
    args = ap.parse_args()

    model = build_model()
    doc = {
        "generated_by": "tests/python/nemotron_transcribe_ref_oracle.py",
        "note": "tiny nemotron_h fp32 CPU goldens for the host oracle "
                "(docs/nemotron_transcribe_plan.md M2). Reference: HF "
                "transformers pure-torch nemotron_h fallback (no kernels "
                "package), the semantics reference.cpp mirrors.",
        "seed": SEED,
        "config": {
            "layers_block_type": LAYERS,
            "mtp_layers_block_type": ["attention", "moe"],
            **DIMS,
            "rope_theta": 10000,
            "time_step_min": DT_MIN,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "norm_eps": 1e-5,
            "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu",
        },
        "ids": IDS,
        "weights": weights_map(model),
        "walk": capture_walk(model),
        "cases": primitive_cases(),
    }
    out = Path(args.emit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(doc['weights'])} tensors)")
    if args.check:
        w2 = capture_walk(build_model())
        same = w2["logits"]["data"] == doc["walk"]["logits"]["data"]
        print("check: walk logits reproduce:", same)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
