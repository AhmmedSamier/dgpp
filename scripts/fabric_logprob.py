#!/usr/bin/env python3
"""Teacher-forced log-probability gate for fabric runs.

    fabric_logprob.py NEW_DIR [REF_DIR] [--max-mean-nll-delta X]
                                        [--max-token-delta Y]

A run made with `glm_gen_check --teacher-file F` logs, per rank and step,

    [tf] rank r step s: target T argmax A lmax M lse L target_logit G

where L is the rank's log-sum-exp over ITS vocabulary slice and G is the
target's logit when the target lives in that slice ("nan" otherwise). The
global log p(target) at a step is

    G(owner rank) - logaddexp(L_0, L_1, ..., L_{world-1})

This tool joins the ranks, prints the text's NLL / perplexity / top-1
accuracy, and — given a REF_DIR made with the same text and prompt — the
per-token log-prob deltas and a verdict. The thresholds are the numerics
contract for reassociated kernels: the logits are bf16, so single-token
deltas of ~0.1 nat are rounding (a bf16 ulp at |logit| 16 is 0.125); the
MEAN delta over thousands of tokens is the signal, and 0.02 nat is ~2% of
perplexity — well inside what two bf16 implementations of one model differ
by, and far below anything that moves a benchmark.

Texts (benchmarks/): teacher_text.txt (556 tokens of prose, ppl ~2.6 —
quick), teacher_text_hard.txt (7331 tokens of dense technical prose the
model has never seen, ppl ~10.8 — the sensitive instrument: flat
distributions amplify kernel differences), teacher_text_memorized.txt
(6549 tokens of Conan Doyle, ppl 1.04 — the confident regime, where nothing
should move). Each takes its token count x the step time: ~4 minutes for
the long ones.
"""
import argparse
import math
import os
import re
import sys

RE_TF = re.compile(
    r"\[tf\] rank (\d+) step (\d+): target (\d+) argmax (\d+) lmax ([-\d.]+) "
    r"lse ([-\d.]+) target_logit ([-\d.]+|nan)")


def load_run(directory):
    """step -> {"target", "argmax", "lse": [per rank], "target_logit"}"""
    steps = {}
    ranks = set()
    for name in sorted(os.listdir(directory)):
        m = re.fullmatch(r"r(\d+)\.log", name)
        if not m:
            continue
        rank = int(m.group(1))
        ranks.add(rank)
        with open(os.path.join(directory, name), errors="replace") as f:
            for line in f:
                g = RE_TF.search(line)
                if not g:
                    continue
                step = int(g.group(2))
                rec = steps.setdefault(
                    step, {"target": int(g.group(3)), "argmax": int(g.group(4)),
                           "lse": {}, "target_logit": None})
                rec["lse"][rank] = float(g.group(6))
                if g.group(7) != "nan":
                    rec["target_logit"] = float(g.group(7))
    if not steps:
        sys.exit(f"{directory}: no [tf] lines (run with --teacher-file)")
    return steps, sorted(ranks)


def logaddexp_all(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def logprobs(steps, ranks):
    """step -> (logprob, top1_hit); incomplete steps are skipped with a note."""
    out = {}
    for step, rec in sorted(steps.items()):
        if set(rec["lse"]) != set(ranks) or rec["target_logit"] is None:
            print(f"  step {step}: incomplete (ranks {sorted(rec['lse'])}, "
                  f"target logit {'present' if rec['target_logit'] is not None else 'missing'}) — skipped")
            continue
        lse = logaddexp_all(list(rec["lse"].values()))
        out[step] = (rec["target_logit"] - lse, rec["argmax"] == rec["target"])
    return out


def summarize(label, lp):
    n = len(lp)
    nll = -sum(v for v, _ in lp.values())
    hits = sum(1 for _, h in lp.values() if h)
    print(f"{label}: {n} tokens, NLL {nll:.3f}, mean {nll / n:.4f} nat/token, "
          f"perplexity {math.exp(nll / n):.3f}, top-1 {hits}/{n} "
          f"({100.0 * hits / n:.1f}%)")
    worst = sorted(lp.items(), key=lambda kv: kv[1][0])[:5]
    print("  least likely positions: " +
          ", ".join(f"step {s} ({v:.2f})" for s, (v, _) in worst))
    return nll / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_dir")
    ap.add_argument("ref_dir", nargs="?")
    ap.add_argument("--max-mean-nll-delta", type=float, default=0.02,
                    help="|mean NLL(new) - mean NLL(ref)| bound, nats (0.02)")
    ap.add_argument("--big-delta", type=float, default=1.0,
                    help="a single-token |delta log p| above this is a 'big' "
                         "move (1.0 nat)")
    ap.add_argument("--max-big-delta-rate", type=float, default=0.01,
                    help="bound on the fraction of tokens with a big move "
                         "(0.01). A rounding-level change at a MoE router's "
                         "top-k boundary swaps an expert and moves that "
                         "position's logits by O(1) — rare by nature, and no "
                         "bound on the single worst token survives it; the "
                         "RATE of such positions is what a defect would raise. "
                         "Calibration (2026-09-03, round 11's reassociations "
                         "vs the bit-exact baseline): 0.41% on the hard text, "
                         "0.12% on the memorized one.")
    args = ap.parse_args()

    new_steps, new_ranks = load_run(args.new_dir)
    new_lp = logprobs(new_steps, new_ranks)
    if not new_lp:
        sys.exit("no complete steps")
    new_mean = summarize(f"new ({args.new_dir}, {len(new_ranks)} ranks)", new_lp)
    if args.ref_dir is None:
        return 0

    ref_steps, ref_ranks = load_run(args.ref_dir)
    ref_lp = logprobs(ref_steps, ref_ranks)
    ref_mean = summarize(f"ref ({args.ref_dir}, {len(ref_ranks)} ranks)", ref_lp)

    common = sorted(set(new_lp) & set(ref_lp))
    mismatched = [s for s in common
                  if new_steps[s]["target"] != ref_steps[s]["target"]]
    if mismatched:
        sys.exit(f"the runs scored different texts (targets differ at steps "
                 f"{mismatched[:5]}...)")
    deltas = [(s, new_lp[s][0] - ref_lp[s][0]) for s in common]
    if not deltas:
        sys.exit("no common steps")
    n = len(deltas)
    mean_delta = sum(d for _, d in deltas) / n
    # Standard error of the mean: is the shift distinguishable from the
    # per-token rounding noise it is averaged over?
    var = sum((d - mean_delta) ** 2 for _, d in deltas) / max(n - 1, 1)
    sem = math.sqrt(var / n)
    mean_abs = sum(abs(d) for _, d in deltas) / n
    big = sorted((sd for sd in deltas if abs(sd[1]) > args.big_delta),
                 key=lambda sd: -abs(sd[1]))
    big_rate = len(big) / n
    flips = sum(1 for s in common if new_lp[s][1] != ref_lp[s][1])
    print(f"delta over {n} common tokens: mean {mean_delta:+.5f} nat "
          f"(+-{sem:.5f} s.e.; NLL {-mean_delta * n:+.3f} total), mean |delta| "
          f"{mean_abs:.4f}; top-1 flips {flips}; big moves (>{args.big_delta} "
          f"nat) {len(big)} = {100 * big_rate:.2f}%")
    for step, d in big[:5]:
        print(f"  step {step}: log p {ref_lp[step][0]:.3f} -> {new_lp[step][0]:.3f} "
              f"({d:+.3f}); target {new_steps[step]['target']}, argmax "
              f"{ref_steps[step]['argmax']} -> {new_steps[step]['argmax']}")
    mean_nll_delta = new_mean - ref_mean
    ok = (abs(mean_nll_delta) <= args.max_mean_nll_delta
          and big_rate <= args.max_big_delta_rate)
    print(f"verdict: {'PASS' if ok else 'FAIL'} (mean NLL delta "
          f"{mean_nll_delta:+.5f} vs {args.max_mean_nll_delta}; big-move rate "
          f"{100 * big_rate:.2f}% vs {100 * args.max_big_delta_rate:.2f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
