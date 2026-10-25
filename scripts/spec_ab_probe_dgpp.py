#!/usr/bin/env python3
"""DGPP side of the paired probe: exact per-request spec stats via metrics deltas.

Run after spec_ab_probe_prompts.py against the dgpp-serve that carries the
drafter under test; per-request alpha/tau come from the /metrics spec_decode
deltas around each single request. The prompt cells are imported from
spec_ab_probe_prompts, never restated: the pairing means nothing unless both
engines saw the same bytes.

Point --url at the dgpp-serve port when one is up (its /metrics and /v1/models
are its own). Behind a swap proxy, --model names which deployment the numbers
belong to; the script checks the id is served before sampling anything rather
than attributing one model's counters to another.
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ab_probe_prompts import CELLS  # noqa: E402


def served_ids(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
        return sorted({m["id"] for m in json.load(r).get("data", [])})


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=120) as r:
        return json.load(r)["scheduler"]["spec_decode"]


def run(cells, out, base, model):
    results = []
    for c in cells:
        m0 = metrics(base)
        d0, a0 = m0["num_draft_tokens_total"], m0["num_accepted_tokens_total"]
        n0 = m0["num_drafts_total"]
        p0 = list(m0["num_draft_tokens_per_pos_total"])
        pa0 = list(m0["num_accepted_tokens_per_pos_total"])
        t0 = time.time()
        body = {"model": model, "temperature": 0, "stream": False,
                "max_tokens": c["max_tokens"], "messages": c["messages"]}
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        wall = time.time() - t0
        m1 = metrics(base)
        d1, a1 = m1["num_draft_tokens_total"], m1["num_accepted_tokens_total"]
        p1 = list(m1["num_draft_tokens_per_pos_total"])
        pa1 = list(m1["num_accepted_tokens_per_pos_total"])
        drafts, accepted = d1 - d0, a1 - a0
        per_pos = [b - a for a, b in zip(pa0, pa1)]
        pos_drafts = [b - a for a, b in zip(p0, p1)]
        # τ = accepted/(steps) : steps = num_new_drafts; accepted counts the bonus token, as the bench does
        steps = m1["num_drafts_total"] - n0
        tau = None if not steps else accepted / steps
        results.append({
            "cell": c["cell"], "model": model, "ok": True, "wall_s": round(wall, 2),
            "steps": steps, "drafts": drafts, "accepted": accepted,
            "alpha": None if drafts == 0 else round(accepted / drafts, 4),
            "tau": None if not steps else round(accepted / steps, 3),
            "per_pos_accepted": per_pos, "per_pos_drafts": pos_drafts,
            "usage": resp.get("usage"),
        })
        print(f"{c['cell']:>16}  α={results[-1]['alpha']}  τ={results[-1]['tau']}  "
              f"steps={steps} drafts={drafts} accepted={accepted} wall={wall:.1f}s")
    json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DGPP side of the paired spec probe.")
    ap.add_argument("out", help="where to write the per-cell JSON")
    ap.add_argument("--url", default="http://localhost:8000",
                    help="the dgpp-serve base URL (its own port when one is running)")
    ap.add_argument("--model", default=None, help="defaults to the single served id")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    ids = served_ids(base)
    model = a.model
    if model is None:
        if len(ids) != 1:
            sys.exit(f"{base} reports {len(ids)} model ids {ids}: name the deployment with --model "
                     f"(a swap proxy lists every configured model, not the loaded one)")
        model = ids[0]
    elif ids and model not in ids:
        sys.exit(f"{base} does not serve '{model}' (it reports {ids})")
    print(f"# {len(CELLS)} cells against {base} as '{model}'")
    run(CELLS, a.out, base, model)
