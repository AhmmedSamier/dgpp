#!/usr/bin/env python3
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fabric_sampling_profile as profile  # noqa: E402


def write_run(directory, ranks, complete=True):
    """ranks maps rank -> iterable of (step, four masses)."""
    root = Path(directory)
    for rank, points in ranks.items():
        with (root / f"r{rank}.log").open("w") as out:
            for step, masses in points:
                out.write(
                    "2026-09-03 info [sample_mass] "
                    f"rank {rank} step {step}: "
                    f"k32 {masses[0]:.17f} k64 {masses[1]:.17f} "
                    f"k128 {masses[2]:.17f} k256 {masses[3]:.17f}\n")
            if complete:
                for k in profile.TOP_KS:
                    out.write(
                        "2026-09-03 info [sample_mass_summary] "
                        f"rank {rank}: T=1 top_p=0.95 k={k} "
                        f"positions={len(points)} mass min/mean/p50/p99="
                        f"0/0/0/0 fallback=0/{len(points)} (0%)\n")


class SamplingProfileTest(unittest.TestCase):
    def test_combined_summary_selects_smallest_passing_width(self):
        points_a = [
            (0, (0.90, 0.96, 0.98, 0.99)),
            (1, (0.94, 0.949, 0.97, 0.999)),
        ]
        points_b = [(0, (0.93, 0.951, 0.96, 0.98))]
        with tempfile.TemporaryDirectory() as a:
            with tempfile.TemporaryDirectory() as b:
                write_run(a, {0: points_a, 1: points_a})
                write_run(b, {0: points_b, 1: points_b})
                summaries = profile.summarize(
                    [profile.load_run(a), profile.load_run(b)], top_p=0.95)

        self.assertEqual(summaries[32].fallback_rate, 1.0)
        self.assertEqual(summaries[64].fallback_rate, 1.0 / 3.0)
        self.assertEqual(summaries[128].fallback_rate, 0.0)
        self.assertEqual(profile.choose_width(summaries, 0.01), 128)

    def test_rank_mass_disagreement_is_rejected(self):
        rank0 = [(0, (0.90, 0.95, 0.98, 0.99))]
        rank1 = [(0, (0.90, 0.95, 0.97, 0.99))]
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, {0: rank0, 1: rank1})
            with self.assertRaisesRegex(ValueError, "disagrees"):
                profile.load_run(directory)

    def test_missing_rank_step_is_rejected(self):
        rank0 = [
            (0, (0.90, 0.95, 0.98, 0.99)),
            (1, (0.91, 0.96, 0.99, 1.00)),
        ]
        rank1 = rank0[:1]
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, {0: rank0, 1: rank1})
            with self.assertRaisesRegex(ValueError, "step mismatch"):
                profile.load_run(directory)

    def test_invalid_mass_order_is_rejected(self):
        points = [(0, (0.90, 0.89, 0.98, 0.99))]
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, {0: points})
            with self.assertRaisesRegex(ValueError, "non-monotonic"):
                profile.load_run(directory)

    def test_missing_final_summary_is_rejected_as_incomplete(self):
        points = [(0, (0.90, 0.95, 0.98, 0.99))]
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, {0: points}, complete=False)
            with self.assertRaisesRegex(ValueError, "run incomplete"):
                profile.load_run(directory)

    def test_cli_returns_failure_when_no_width_meets_bound(self):
        points = [(0, (0.10, 0.20, 0.30, 0.40))]
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, {0: points})
            with redirect_stdout(io.StringIO()) as output:
                status = profile.main([directory])
        self.assertEqual(status, 1)
        self.assertIn("verdict: FAIL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
