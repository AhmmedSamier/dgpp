#!/usr/bin/env python3
"""Shared reader for fabric run directories (r0.log .. rN.log).

`scripts/fabric_run.sh --fetch-logs` leaves one log per rank in the run
directory. The per-step record lines glm_gen_check writes on every rank
are parsed here, once, so each analysis tool (fabric_xcript.py,
fabric_logprob.py, whatever judges MTP acceptance next) is its arithmetic
and nothing else.

    from fabric_logs import load_gen, load_teacher, bf16_ulp

    gen = load_gen("/mnt/ramlog/fabric48")      # rank -> step -> GenLine
    tf = load_teacher("/mnt/ramlog/tfR_f32")    # rank -> step -> TeacherLine

Both return {} for a directory without the lines; the callers decide what
that means. Steps are keyed by the app's step number (0 = the pick off the
prefill's logits).
"""
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TypeVar

# '[gen] rank r step s: token T (local slice [a,b) best B logit L second S)'
# — `second` (the slice's runner-up) arrived 2026-09-02; older logs lack it.
RE_GEN = re.compile(
    r"\[gen\] rank (\d+) (?:step|prefill\+pick) (\d+): token (\d+) .*?best (\d+) "
    r"logit ([-\d.]+)(?: second ([-\d.inf]+))?")

# '[tf] rank r step s: target T argmax A lmax M lse L target_logit G|nan'
RE_TEACHER = re.compile(
    r"\[tf\] rank (\d+) step (\d+): target (\d+) argmax (\d+) lmax ([-\d.]+) "
    r"lse ([-\d.]+) target_logit ([-\d.]+|nan)")


@dataclass(frozen=True)
class GenLine:
    rank: int
    step: int
    token: int         # the global pick (identical on every rank)
    best_id: int       # this rank's slice argmax
    best_logit: float
    second: float      # this rank's slice runner-up (-inf when unlogged)


@dataclass(frozen=True)
class TeacherLine:
    rank: int
    step: int
    target: int
    argmax: int                    # the global pick at this step
    lmax: float                    # this rank's slice max
    lse: float                     # this rank's slice log-sum-exp
    target_logit: Optional[float]  # None unless the target is in the slice


T = TypeVar("T")


def rank_logs(directory: str) -> Dict[int, str]:
    """rank -> path for every rN.log in the directory."""
    logs = {}
    for name in os.listdir(directory):
        m = re.fullmatch(r"r(\d+)\.log", name)
        if m:
            logs[int(m.group(1))] = os.path.join(directory, name)
    return dict(sorted(logs.items()))


def scan(directory: str, pattern: "re.Pattern[str]",
         make: Callable[["re.Match[str]"], T]) -> Dict[int, Dict[int, T]]:
    """rank -> step -> make(match) over every line matching `pattern`.

    A step logged twice on one rank keeps the last line (a restarted run
    appending to the same log; the later run is the one that finished).
    """
    out: Dict[int, Dict[int, T]] = {}
    for rank, path in rank_logs(directory).items():
        steps: Dict[int, T] = {}
        with open(path, errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    rec = make(m)
                    steps[getattr(rec, "step")] = rec
        out[rank] = steps
    return out


def _gen_of(m: "re.Match[str]") -> GenLine:
    return GenLine(rank=int(m.group(1)), step=int(m.group(2)),
                   token=int(m.group(3)), best_id=int(m.group(4)),
                   best_logit=float(m.group(5)),
                   second=float(m.group(6)) if m.group(6) else float("-inf"))


def _teacher_of(m: "re.Match[str]") -> TeacherLine:
    g7 = m.group(7)
    return TeacherLine(rank=int(m.group(1)), step=int(m.group(2)),
                       target=int(m.group(3)), argmax=int(m.group(4)),
                       lmax=float(m.group(5)), lse=float(m.group(6)),
                       target_logit=None if g7 == "nan" else float(g7))


def load_gen(directory: str) -> Dict[int, Dict[int, GenLine]]:
    return scan(directory, RE_GEN, _gen_of)


def load_teacher(directory: str) -> Dict[int, Dict[int, TeacherLine]]:
    return scan(directory, RE_TEACHER, _teacher_of)


def bf16_ulp(x: float) -> float:
    """One bf16 ulp (8 significand bits) at magnitude x — the rounding scale
    of the residual stream that feeds the head. The logits themselves are
    fp32 since 2026-09-03, so an exact tie no longer happens, but a margin
    inside one bf16 ulp is still what a rounding-level kernel change can
    flip."""
    if x == 0 or math.isinf(x) or math.isnan(x):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def logaddexp_all(values) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))
