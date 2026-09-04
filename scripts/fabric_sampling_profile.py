#!/usr/bin/env python3
"""Choose the M6 distributed-sampling candidate width from fabric evidence.

    python3 scripts/fabric_sampling_profile.py RUN_DIR [RUN_DIR ...]
        [--top-p 0.95] [--max-fallback-rate 0.01]

Each directory must come from a complete ``glm_gen_check --teacher-file F
--sampling-profile`` run with fetched rank logs. The app measures the exact
full-distribution probability mass represented by global top-k for
k={32,64,128,256}. This tool first requires identical, contiguous positions
and masses on every rank, combines the supplied teacher texts, and selects the
smallest width whose exact-gather fallback rate is at most the requested bound.
"""
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from fabric_logs import load_sample_mass, load_sample_mass_summary


TOP_KS = (32, 64, 128, 256)


@dataclass(frozen=True)
class RunProfile:
    directory: str
    ranks: Tuple[int, ...]
    by_step: Dict[int, Tuple[float, float, float, float]]


@dataclass(frozen=True)
class WidthSummary:
    k: int
    positions: int
    minimum: float
    mean: float
    p50: float
    p99: float
    fallbacks: int

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.positions


def load_run(directory: str) -> RunProfile:
    """Load one run, refusing missing/divergent rank evidence."""
    per_rank = load_sample_mass(directory)
    final = load_sample_mass_summary(directory)
    if not per_rank or not any(per_rank.values()):
        raise ValueError(
            f"{directory}: no [sample_mass] lines "
            "(run with --teacher-file and --sampling-profile)")

    ranks = tuple(sorted(per_rank))
    if ranks != tuple(range(ranks[-1] + 1)):
        raise ValueError(f"{directory}: rank logs are not contiguous: {ranks}")
    empty = [rank for rank, lines in per_rank.items() if not lines]
    if empty:
        raise ValueError(f"{directory}: no sampling lines for ranks {empty}")

    first_rank = ranks[0]
    steps = tuple(sorted(per_rank[first_rank]))
    if steps != tuple(range(len(steps))):
        raise ValueError(
            f"{directory}: rank {first_rank} steps are not contiguous from 0")

    reference = {}
    for step in steps:
        line = per_rank[first_rank][step]
        if line.rank != first_rank:
            raise ValueError(
                f"{directory}: r{first_rank}.log contains rank {line.rank} "
                f"at step {step}")
        _validate_masses(directory, first_rank, step, line.masses)
        reference[step] = line.masses

    expected_steps = set(steps)
    for rank in ranks[1:]:
        actual_steps = set(per_rank[rank])
        if actual_steps != expected_steps:
            missing = sorted(expected_steps - actual_steps)
            extra = sorted(actual_steps - expected_steps)
            raise ValueError(
                f"{directory}: rank {rank} step mismatch; "
                f"missing={missing[:8]} extra={extra[:8]}")
        for step in steps:
            line = per_rank[rank][step]
            if line.rank != rank:
                raise ValueError(
                    f"{directory}: r{rank}.log contains rank {line.rank} "
                    f"at step {step}")
            _validate_masses(directory, rank, step, line.masses)
            if line.masses != reference[step]:
                raise ValueError(
                    f"{directory}: rank {rank} disagrees with rank "
                    f"{first_rank} at step {step}: "
                    f"{line.masses} != {reference[step]}")

    for rank in ranks:
        summaries = final.get(rank, {})
        if set(summaries) != set(TOP_KS):
            raise ValueError(
                f"{directory}: rank {rank} lacks the four final "
                "[sample_mass_summary] lines (run incomplete)")
        for k, marker in summaries.items():
            if marker.rank != rank or marker.positions != len(steps):
                raise ValueError(
                    f"{directory}: rank {rank} has an invalid k={k} final "
                    f"summary (logged rank {marker.rank}, positions "
                    f"{marker.positions}, expected {len(steps)})")
    return RunProfile(directory=directory, ranks=ranks, by_step=reference)


def _validate_masses(directory: str, rank: int, step: int,
                     masses: Tuple[float, ...]) -> None:
    if (any(not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in masses)
            or any(right < left for left, right in zip(masses, masses[1:]))):
        raise ValueError(
            f"{directory}: invalid or non-monotonic masses at rank {rank} "
            f"step {step}: {masses}")


def _percentile(sorted_values: List[float], fraction: float) -> float:
    # Match glm_gen_check's diagnostic summary: floor(fraction*n) as a
    # zero-based index, capped at the final observation.
    return sorted_values[min(len(sorted_values) - 1,
                             int(fraction * len(sorted_values)))]


def summarize(profiles: Iterable[RunProfile],
              top_p: float = 0.95) -> Dict[int, WidthSummary]:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0,1]")
    points = [masses for profile in profiles
              for _, masses in sorted(profile.by_step.items())]
    if not points:
        raise ValueError("no sampling-profile positions")

    out = {}
    for index, k in enumerate(TOP_KS):
        values = sorted(point[index] for point in points)
        fallbacks = sum(value < top_p for value in values)
        out[k] = WidthSummary(
            k=k, positions=len(values), minimum=values[0],
            mean=sum(values) / len(values),
            p50=_percentile(values, 0.50),
            p99=_percentile(values, 0.99), fallbacks=fallbacks)
    return out


def choose_width(summaries: Dict[int, WidthSummary],
                 max_fallback_rate: float = 0.01) -> Optional[int]:
    if not 0.0 <= max_fallback_rate <= 1.0:
        raise ValueError("max_fallback_rate must be in [0,1]")
    return next((k for k in TOP_KS
                 if summaries[k].fallback_rate <= max_fallback_rate), None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-fallback-rate", type=float, default=0.01)
    args = parser.parse_args(argv)

    try:
        profiles = [load_run(directory) for directory in args.run_dirs]
        summaries = summarize(profiles, args.top_p)
        selected = choose_width(summaries, args.max_fallback_rate)
    except (OSError, ValueError) as error:
        print(f"sampling profile: {error}", file=sys.stderr)
        return 2

    for profile in profiles:
        per_run = summarize([profile], args.top_p)
        rates = ", ".join(
            f"k={k}:{100.0 * per_run[k].fallback_rate:.3f}%"
            for k in TOP_KS)
        print(f"{profile.directory}: {len(profile.by_step)} positions, "
              f"ranks {','.join(map(str, profile.ranks))}; fallback {rates}")
    print(f"combined: {next(iter(summaries.values())).positions} positions; "
          f"T=1 top_p={args.top_p:g}")
    for k in TOP_KS:
        summary = summaries[k]
        print(f"  k={k:3d}: mass min/mean/p50/p99="
              f"{summary.minimum:.6f}/{summary.mean:.6f}/"
              f"{summary.p50:.6f}/{summary.p99:.6f}; fallback "
              f"{summary.fallbacks}/{summary.positions} "
              f"({100.0 * summary.fallback_rate:.3f}%)")

    if selected is None:
        print("verdict: FAIL — no measured width meets fallback-rate bound "
              f"{100.0 * args.max_fallback_rate:.3f}%")
        return 1
    print(f"verdict: k={selected} is the smallest measured width at or below "
          f"{100.0 * args.max_fallback_rate:.3f}% fallback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
