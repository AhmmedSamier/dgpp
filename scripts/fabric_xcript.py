#!/usr/bin/env python3
"""Judge two fabric runs' transcripts.

Numerics changes within floating-point rounding may flip a greedy pick at
a near-tie, after which the transcripts diverge for good. This tool says
whether a divergence is that (a flip at a small global top-2 margin) or
something to worry about (a flip at a wide margin).

    fabric_xcript.py REF_DIR NEW_DIR [--steps N]

Reads rN.log from both directories: the '[gen] rank r step s: token T
(local slice [a,b) best B logit L second S)' lines give every rank's local
best and runner-up per step; the global runner-up is the best of the other
ranks' bests and the winner rank's second. Prints the first divergence
(step, tokens, margin) and the margin distribution of the identical prefix.
"""
import os
import re
import sys

RE_GEN = re.compile(
    r"\[gen\] rank (\d+) (?:step|prefill\+pick) (\d+): token (\d+) .*?best (\d+) "
    r"logit ([-\d.]+)(?: second ([-\d.inf]+))?")


def load_run(directory):
    """rank -> {step: (token, best_id, best_logit, second_logit)}"""
    ranks = {}
    for name in sorted(os.listdir(directory)):
        m = re.fullmatch(r"r(\d+)\.log", name)
        if not m:
            continue
        rank = int(m.group(1))
        steps = {}
        with open(os.path.join(directory, name), errors="replace") as f:
            for line in f:
                g = RE_GEN.search(line)
                if not g:
                    continue
                step = int(g.group(2))
                second = float(g.group(6)) if g.group(6) else float("-inf")
                steps[step] = (int(g.group(3)), int(g.group(4)),
                               float(g.group(5)), second)
        ranks[rank] = steps
    return ranks


def global_margin(ranks, step):
    """Top-1 minus top-2 logit across the ranks' slices at `step`."""
    bests = []
    for rank, steps in ranks.items():
        if step not in steps:
            return None
        _, best_id, best_logit, second = steps[step]
        bests.append((best_logit, -best_id, rank, second))
    bests.sort(reverse=True)
    winner = bests[0]
    runner = max([b[0] for b in bests[1:]] + [winner[3]])
    return winner[0] - runner


def bf16_ulp(x):
    """One bf16 ulp (8 significand bits) at magnitude x — the rounding
    scale of the residual stream that feeds the head. The logits themselves
    are fp32 since 2026-09-03, so an exact tie no longer happens, but a
    margin inside one bf16 ulp is still what a rounding-level kernel change
    can flip."""
    import math
    if x == 0 or math.isinf(x) or math.isnan(x):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def tokens_of(ranks):
    steps = ranks[0]
    return [steps[s][0] for s in sorted(steps)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    ref = load_run(args[0])
    new = load_run(args[1])
    if 0 not in ref or 0 not in new:
        print("missing r0.log in one of the directories")
        return 2
    ref_toks, new_toks = tokens_of(ref), tokens_of(new)
    n = min(len(ref_toks), len(new_toks))
    first = next((i for i in range(n) if ref_toks[i] != new_toks[i]), None)
    steps_sorted = sorted(ref[0])[:n]
    margins = [global_margin(ref, s) for s in steps_sorted]
    margins = [m for m in margins if m is not None]
    if margins:
        ms = sorted(margins)
        ties = sum(1 for s in steps_sorted
                   if (m := global_margin(ref, s)) is not None
                   and m <= 1.5 * bf16_ulp(ref[0][s][2]))
        print(f"reference margins over {len(ms)} steps: min {ms[0]:.4f} "
              f"p10 {ms[len(ms)//10]:.4f} median {ms[len(ms)//2]:.4f}; "
              f"steps decided by <= 1 bf16 ulp: {ties}")
    if first is None:
        print(f"IDENTICAL over {n} steps")
        return 0
    step = sorted(ref[0])[first]
    m_ref = global_margin(ref, step)
    m_new = global_margin(new, step)
    print(f"DIVERGENCE at step {step} (token index {first} of {n}): "
          f"ref token {ref_toks[first]} new token {new_toks[first]}")
    print(f"  global top-2 margin at that step: ref {m_ref:.4f}, new {m_new:.4f}")
    # A pick decided within one bf16 ulp of the winner (0.125 at |logit|
    # ~20) is a coin the hidden state's last rounding flips — expected after any
    # reassociation. Wider than a couple of ulps is not rounding.
    ulp = bf16_ulp(ref[0][step][2])
    near = min(m_ref, m_new) <= 1.5 * ulp
    verdict = (f"near-tie flip (<= 1 bf16 ulp = {ulp:.4f}; rounding-level)"
               if near else "WIDE-MARGIN FLIP — investigate")
    print(f"  verdict: {verdict}")
    for r in sorted(ref):
        if step in ref[r] and step in new[r]:
            a, b = ref[r][step], new[r][step]
            print(f"  r{r}: ref best {a[1]} {a[2]:.4f} (2nd {a[3]:.4f}) | "
                  f"new best {b[1]} {b[2]:.4f} (2nd {b[3]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
