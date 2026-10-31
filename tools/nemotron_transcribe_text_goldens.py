#!/usr/bin/env python3
"""Text-mode goldens for the Nemotron-3.5-Transcribe port (milestone 2 of
docs/nemotron_transcribe_plan.md).

Rebuilds the HF `nemotron_h` LLM from the consolidated release checkpoint —
the `llm.*` tensors, remapped (the export names the stacked relu2 experts
`gate_and_up_projs`/`down_projs` in bmm layout, parks A_log/D/dt_bias under
`_fp32_params`, calls the embedding `embed_tokens` and the final norm
`norm`; HF wants `up_proj`/`down_proj` in linear layout, the mixer params at
the mixer root, `embeddings` and `norm_f`). The MTP block and everything
outside `llm.` are out of the base-LM oracle's scope.

Runs greedy continuation on fixed text-only prompts and records the prompt
ids, the generated ids, the last-token top-5 logits and the final hidden
state's fingerprint — what tests/unit/nemotron_transcribe_oracle_test.cpp
replays against the C++ fp32 walk.

Needs the GPU free (the LM is 33 B params; bf16 ≈ 66 GiB) or a host with
matching RAM:  --device cuda|cpu  --dtype bfloat16|float32.

Usage:
  python3 tools/nemotron_transcribe_text_goldens.py \
      --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
      --out tests/data/nemotron_transcribe_text_goldens.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

# (consolidated suffix, hf suffix) renames beyond the `llm.` strip.
RENAMES = {
    "model.embed_tokens.weight": "model.embeddings.weight",
    "model.norm.weight": "model.norm_f.weight",
}


def remap(key: str, shape: list[int]):
    """Map a consolidated `llm.*` tensor name to its HF parameter name and
    the transform the loader applies. Returns None for out-of-scope keys."""
    if not key.startswith("llm.") or key.endswith("._extra_state") or key.startswith("llm.mtp."):
        return None
    name = key[len("llm."):]
    if "._fp32_params." in name:
        name = name.replace("._fp32_params.", ".")
    if name.endswith("mixer.experts.gate_and_up_projs"):
        e, h, i = shape
        return (name[: -len("gate_and_up_projs")] + "up_proj", "transpose")
    if name.endswith("mixer.experts.down_projs"):
        e, i, h = shape
        return (name[: -len("down_projs")] + "down_proj", "transpose")
    return (RENAMES.get(name, name), "copy")

# Fixed text-only prompts (transcription-style instructions on empty
# context, plus plain prose continuation — the oracle needs no audio).
PROMPTS = [
    "The capital of France is",
    "Transcribe the following audio. Language: as spoken. Wording: verbatim.",
    "Once upon a time in a small town by the sea, there lived",
    "def fibonacci(n):\n    if n <= 1:\n        return",
]


def snapshot_dir(model: str) -> Path:
    p = Path(model)
    if (p / "model.safetensors").is_file() or (p / "llm_backbone").is_dir():
        return p
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    repo = "models--" + model.replace("/", "--")
    snaps = sorted((home / "hub" / repo / "snapshots").iterdir())
    # Re-uploads leave partial snapshots behind; only one carries the
    # 67.5 GB shard, and every consumer of this dir wants it.
    with_weights = [s for s in snaps if (s / "model.safetensors").is_file()]
    if len(with_weights) != 1:
        sys.exit(f"expected exactly one weighted snapshot for {model}, got {len(with_weights)}")
    return with_weights[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    snap = snapshot_dir(args.model)
    weight = snap / "model.safetensors"
    dtype = getattr(torch, args.dtype)

    cfg = NemotronHConfig.from_pretrained(snap / "llm_backbone")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    want = model.state_dict()
    loaded = {}
    with safe_open(weight, framework="torch") as f:
        for key in f.keys():
            hit = remap(key, list(f.get_slice(key).get_shape()))
            if hit is None:
                continue
            name, how = hit
            if name not in want:
                sys.exit(f"remapped {key} -> {name} is not an HF parameter")
            t = f.get_tensor(key)
            if how == "transpose":
                t = t.transpose(1, 2).contiguous()
            loaded[name] = t.to(dtype=dtype, device=args.device)
    missing = set(want) - set(loaded)
    if missing:
        sys.exit(f"checkpoint lacks {len(missing)} HF parameters, e.g. {sorted(missing)[:5]}")
    model.load_state_dict(loaded, assign=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(snap)

    cases = []
    with torch.no_grad():
        for text in PROMPTS:
            ids = tok(text, return_tensors="pt").input_ids.to(args.device)
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
            gen = out[0, ids.shape[1]:].tolist()
            last = model(ids, output_hidden_states=True)
            logits = last.logits[0, -1]
            top = torch.topk(logits.float(), 5)
            hidden = last.hidden_states[-1][0, -1]  # the final norm's output
            cases.append({
                "prompt": text,
                "prompt_ids": ids[0].tolist(),
                "generated_ids": gen,
                "top5_ids": top.indices.tolist(),
                "top5_logits": [round(v, 5) for v in top.values.tolist()],
                "final_hidden": [round(v, 6) for v in hidden.float().tolist()],
            })
            print(f"greedy {len(gen)} tokens for {text[:40]!r}")

    doc = {
        "source": args.model,
        "snapshot": snap.name,
        "dtype": args.dtype,
        "device": args.device,
        "generated_by": "tools/nemotron_transcribe_text_goldens.py",
        "note": "base nemotron_h LM only: the MTP block (llm.mtp.*) is not "
                "in the HF inference graph and is pinned separately "
                "(docs/nemotron_transcribe_plan.md milestones 4/6).",
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
