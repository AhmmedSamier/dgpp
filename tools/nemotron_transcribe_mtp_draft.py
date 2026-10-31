#!/usr/bin/env python3
"""MTP draft-chain goldens for the Nemotron-3.5-Transcribe port (the fusion
leg of docs/nemotron_transcribe_plan.md — the MTP block the HF inference
graph does not carry).

Runs the real `llm.mtp.*` tensors — the repeated Nemotron-H draft head (the
`*E` pattern: the attention leg carrying the `enorm`/`hnorm`/`eh_proj` start
projections, the MoE leg carrying the `final_layernorm` end norm) — over
the 8-row greedy chain the card calls MTP8, starting from the text goldens'
final hidden (the post-norm row the draft head consumes, EAGLE-style) and
fed the target's own greedy continuation. One weight set, chained: the head
was trained at two depths with `use_repeated_layer`, so MTP8 reuses it eight
times — the chain needs no extra trained layers.

Per row the tool records the draft's top-5 (the row predicts the token
AFTER its fed token; the fed token sits at its plain absolute position for
RoPE, P+c for goldens token c), whether the argmax equals the target's
greedy next token (`accept`), and the row's output hidden — the fingerprint
a device port replays. The feed stays the goldens' greedy tokens regardless
of what the draft argmaxes, so the chain numbers are reproducible while the
accept column is the empirical quality read.

Sequential per row (row c's fusion needs row c-1's walk output), but the
attention legs recompute the causal prefix over the head's own KV — the
chain's KV cache only contains the chain's rows, the target's cache is not
visible to the draft.

Needs a GPU or ~6 GiB of host RAM: only the 18 MTP tensors plus the shared
embedding and LM head load (~5.8 GB bf16), not the 33 B backbone.

Usage:
  python3 tools/nemotron_transcribe_mtp_draft.py \
      --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
      --goldens tests/data/nemotron_transcribe_text_goldens.json \
      --out tests/data/nemotron_transcribe_mtp_draft.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

MTP_ROWS = 8  # the card's MTP8
HEADS, KV_HEADS, HEAD_DIM = 32, 2, 128
ROPE_THETA = 10000.0
TOPK, ROUTED_SCALE, NORM_TOPK = 6, 2.5, True


def snapshot_dir(model: str) -> Path:
    p = Path(model)
    if (p / "model.safetensors").is_file() or (p / "llm_backbone").is_dir():
        return p
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    repo = "models--" + model.replace("/", "--")
    snaps = sorted((home / "hub" / repo / "snapshots").iterdir())
    with_weights = [s for s in snaps if (s / "model.safetensors").is_file()]
    if len(with_weights) != 1:
        sys.exit(f"expected exactly one weighted snapshot for {model}, got {len(with_weights)}")
    return with_weights[0]


def load_tensors(weight: Path, dtype, device):
    """The 18 real MTP tensors (the `_extra_state` stubs carry nothing) plus
    the shared embedding and LM head, under their checkpoint names."""
    w = {}
    with safe_open(weight, framework="torch") as f:
        for key in f.keys():
            if "_extra_state" in key:
                continue
            if key.startswith("llm.mtp.") or key in (
                "llm.lm_head.weight",
                "llm.model.embed_tokens.weight",
            ):
                w[key] = f.get_tensor(key).to(dtype=dtype, device=device)
    expect = {
        "llm.mtp.layers.0.eh_proj.weight",
        "llm.mtp.layers.0.enorm.weight",
        "llm.mtp.layers.0.hnorm.weight",
        "llm.mtp.layers.0.mixer.q_proj.weight",
        "llm.mtp.layers.0.mixer.k_proj.weight",
        "llm.mtp.layers.0.mixer.v_proj.weight",
        "llm.mtp.layers.0.mixer.o_proj.weight",
        "llm.mtp.layers.0.norm.weight",
        "llm.mtp.layers.1.final_layernorm.weight",
        "llm.mtp.layers.1.mixer.experts.down_projs",
        "llm.mtp.layers.1.mixer.experts.gate_and_up_projs",
        "llm.mtp.layers.1.mixer.gate.e_score_correction_bias",
        "llm.mtp.layers.1.mixer.gate.weight",
        "llm.mtp.layers.1.mixer.shared_experts.down_proj.weight",
        "llm.mtp.layers.1.mixer.shared_experts.up_proj.weight",
        "llm.mtp.layers.1.norm.weight",
    }
    missing = expect - {k for k in w if k.startswith("llm.mtp.")}
    if missing:
        sys.exit(f"checkpoint lacks {len(missing)} MTP tensors, e.g. {sorted(missing)[:3]}")
    return w


def rmsnorm(weight, x, dtype):
    """NemotronHRMSNorm: fp32 variance, fp32 weight multiply, cast back."""
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return (weight.float() * (xf * torch.rsqrt(v + 1e-5))).to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_half(q, k, positions):
    """Full NeoX rotary (partial_rotary_factor 1.0, theta 10000)."""
    half = HEAD_DIM // 2
    inv = 1.0 / ROPE_THETA ** (
        torch.arange(half, device=positions.device, dtype=torch.float32) * 2 / HEAD_DIM
    )
    freqs = positions.float()[:, None] * inv[None, :]  # [T, half]
    cos = torch.cat([freqs.cos()] * 2, dim=-1)[None, :, None, :]  # [1, T, 1, D]
    sin = torch.cat([freqs.sin()] * 2, dim=-1)[None, :, None, :]
    return (q * cos + rotate_half(q) * sin).to(q.dtype), (k * cos + rotate_half(k) * sin).to(k.dtype)


def attention_row(w, x, pos, k_cache, v_cache):
    """The layers.0 leg's mixer for one [T, H] batch of chain rows: GQA 32/2
    eager attention over the head's OWN kv cache (the chain rows written so
    far — the target's cache is not visible to the draft), full NeoX rotary,
    scaling head_dim**-0.5, no q/k norms. Returns the projected output and
    the row's raw k/v for the cache append."""
    x = rmsnorm(w["llm.mtp.layers.0.norm.weight"], x, x.dtype)
    T = x.shape[0]
    q = F.linear(x, w["llm.mtp.layers.0.mixer.q_proj.weight"]).view(T, HEADS, HEAD_DIM).transpose(0, 1)[None]
    k = F.linear(x, w["llm.mtp.layers.0.mixer.k_proj.weight"]).view(T, KV_HEADS, HEAD_DIM).transpose(0, 1)[None]
    v = F.linear(x, w["llm.mtp.layers.0.mixer.v_proj.weight"]).view(T, KV_HEADS, HEAD_DIM).transpose(0, 1)[None]
    q, k = rope_half(q, k, pos)
    k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
    v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
    if k_cache is not None:
        k = torch.cat([k_cache.to(k.dtype), k], dim=2)
        v = torch.cat([v_cache.to(v.dtype), v], dim=2)
    ctx = k.shape[2]
    # Sequential concatenation of cached and fresh rows: a fresh row i (its
    # concat index is ctx - T + i) sees just its prefix in the concat order.
    mask = torch.triu(torch.full((T, ctx), float("-inf"), device=x.device,
                                 dtype=torch.float32), ctx - T + 1)[None]  # [1, T, ctx]
    scores = (q.float() @ k.float().transpose(-1, -2)) * HEAD_DIM ** -0.5 + mask
    attn = scores.softmax(-1)
    out = (attn @ v.float()).to(x.dtype).transpose(0, 1).reshape(T, HEADS * HEAD_DIM)
    return F.linear(out, w["llm.mtp.layers.0.mixer.o_proj.weight"]), k, v


def moe_row(w, x):
    """The layers.1 leg's mixer for [T, H]: the bias-corrected sigmoid router
    (top-6 of 128; the release's group policy is the degenerate 1/1), the
    stacked relu2 experts, and the shared expert outside the routed path —
    HF NemotronHMoE physics on the checkpoint's bmm layout. The MoE leg's
    prenorm (`llm.mtp.layers.1.norm`) applies here; the caller adds the
    residual."""
    x = rmsnorm(w["llm.mtp.layers.1.norm.weight"], x, x.dtype)
    logits = F.linear(x.float(), w["llm.mtp.layers.1.mixer.gate.weight"].float())
    scores = logits.sigmoid()
    choice = scores + w["llm.mtp.layers.1.mixer.gate.e_score_correction_bias"].float()
    topk_i = torch.topk(choice, k=TOPK, dim=-1, sorted=False)[1]
    topw = scores.gather(1, topk_i)
    if NORM_TOPK:
        topw = topw / (topw.sum(-1, keepdim=True) + 1e-20)
    topw = topw * ROUTED_SCALE
    up = w["llm.mtp.layers.1.mixer.experts.gate_and_up_projs"]      # [E, H, I]: x @ up[e]
    down = w["llm.mtp.layers.1.mixer.experts.down_projs"]          # [E, I, H]: h @ down[e]
    xf, upf, downf = x.float(), up.float(), down.float()
    # The router's top-6 only; gathering keeps the compute off the other 122.
    sel_up = upf[topk_i]                                           # [T, topk, H, I]
    sel_down = downf[topk_i]                                       # [T, topk, I, H]
    relu2 = torch.einsum("ti,teis->tes", xf, sel_up).clamp_min(0.0) ** 2
    all_out = torch.einsum("tes,teso->teo", relu2, sel_down)       # [T, topk, H]
    routed = (all_out * topw[..., None]).sum(1).to(x.dtype)
    shared = F.linear(
        F.linear(x, w["llm.mtp.layers.1.mixer.shared_experts.up_proj.weight"]).clamp_min(0.0) ** 2,
        w["llm.mtp.layers.1.mixer.shared_experts.down_proj.weight"],
    )
    return routed + shared


def chain_rows(w, case, feed, dtype, device):
    """The 8-row chain. First the draft head's own KV is warmed the way the
    serve prefill does (mtp_forward's warm draft rows q = 0..P-2: token
    q+1's embedding fused with the trunk row h_q, riding slot q+1 — the leg
    output is discarded, only the draft KV survives). The chain then starts
    from the trunk row h_{P-1} with the target's picked token, riding slot
    P-1 where the ring overwrite lands, and continues one slot per claim
    row — the GLM-DSA session_draft_chain convention: the row carrying the
    token the chain is claiming stands at its fill position minus one.

    Without a trunk capture the chain runs cold (empty draft history): the
    accept column then reports the chain-only quality, not the serve's."""
    P = len(case["prompt_ids"])
    trunk = case.get("trunk_hidden")
    has_warm = trunk is not None and len(trunk) == P
    if has_warm:
        h0 = torch.tensor(trunk[-1], dtype=dtype, device=device)
    else:
        h0 = torch.tensor(case["final_hidden"], dtype=dtype, device=device)
    prompt_ids = case["prompt_ids"]
    embed = w["llm.model.embed_tokens.weight"]
    lm_head = w["llm.lm_head.weight"]
    enorm = lambda x: rmsnorm(w["llm.mtp.layers.0.enorm.weight"], x, dtype)
    hnorm = lambda x: rmsnorm(w["llm.mtp.layers.0.hnorm.weight"], x, dtype)
    finorm = lambda x: rmsnorm(w["llm.mtp.layers.1.final_layernorm.weight"], x, dtype)
    eh = w["llm.mtp.layers.0.eh_proj.weight"]

    def fused_row(token, h):
        return F.linear(torch.cat([enorm(embed[torch.tensor([token], device=device)]),
                                   hnorm(h[None, :].to(device))], dim=-1), eh)

    k_cache = v_cache = None
    if has_warm:
        for q in range(P - 1):
            row_in = fused_row(prompt_ids[q + 1],
                               torch.tensor(trunk[q], dtype=dtype, device=device))
            _, kw, vw = attention_row(w, row_in, torch.tensor([q + 1], device=device),
                                      None, None)
            k_cache = kw if k_cache is None else torch.cat([k_cache, kw], dim=2)
            v_cache = vw if v_cache is None else torch.cat([v_cache, vw], dim=2)

    tokens = torch.tensor(feed, device=device)
    rows, h_prev = [], h0[None, :]
    for c in range(len(feed)):
        f = fused_row(tokens[c].item(), h_prev[0])
        if c == 0 and k_cache is not None:
            # Ring overwrite: the chain's first row claims the last warm
            # slot (P-1), so the warm row's K/V there gives way to fresh.
            k_cache, v_cache = k_cache[:, :, :-1], v_cache[:, :, :-1]
        attn, k_c, v_c = attention_row(w, f, torch.tensor([P - 1 + c], device=device),
                                       k_cache, v_cache)
        k_cache, v_cache = k_c, v_c
        post_attn = f + attn
        post_moe = post_attn + moe_row(w, post_attn)
        h_c = finorm(post_moe)
        logits = F.linear(h_c, lm_head).float()[0]
        top = torch.topk(logits, 5)
        rows.append({
            "token": feed[c],
            "argmax": int(logits.argmax()),
            "top5_ids": top.indices.tolist(),
            "top5_logits": [round(v, 5) for v in top.values.tolist()],
            "hidden": [round(v, 6) for v in h_c[0].float().tolist()],
        })
        h_prev = h_c
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    ap.add_argument("--goldens", default="tests/data/nemotron_transcribe_text_goldens.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    snap = snapshot_dir(args.model)
    w = load_tensors(snap / "model.safetensors", getattr(torch, args.dtype), args.device)
    goldens = json.loads(Path(args.goldens).read_text())

    out_cases = []
    with torch.no_grad():
        for case in goldens["cases"]:
            P = len(case["prompt_ids"])
            gen = case["generated_ids"]
            feed = gen[:MTP_ROWS]
            rows = chain_rows(w, case, feed, getattr(torch, args.dtype), args.device)
            for c, row in enumerate(rows):
                row["accept"] = row["argmax"] == gen[c + 1]
            out_cases.append({
                "prompt": case["prompt"],
                "prompt_ids": case["prompt_ids"],
                "generated_ids": case["generated_ids"],
                "warm": case.get("trunk_hidden") is not None,
                "rows": rows,
            })
            print(f"{case['prompt'][:35]!r}: "
                  f"accept {sum(r['accept'] for r in rows)}/{len(rows)}")

    doc = {
        "source": snap.name,
        "goldens": Path(args.goldens).name,
        "dtype": args.dtype,
        "device": args.device,
        "generated_by": "tools/nemotron_transcribe_mtp_draft.py",
        "mtp_rows": MTP_ROWS,
        "note": "the repeated MTP head chained 8 rows from the text "
                "goldens' final hidden; per-row top-5, accept flag and "
                "hidden fingerprint. Rows are the EAGLE-style step "
                "eh_proj([enorm(embed(t)) | hnorm(h)]) -> attention leg -> "
                "MoE leg -> final_layernorm, chain rows only in the "
                "attention window; row c rides position P-1+c (the GLM-DSA "
                "session_draft_chain convention — the row standing for the "
                "token the chain is claiming, one slot behind its fill "
                "position).",
        "cases": out_cases,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
