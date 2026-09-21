"""The head gate must fail closed on incomplete, mismatched or drifting data."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qwen_head_compare import COMPACT_SHAPES, compare


def records(rank, mode):
    result = [("begin", dict(rank=rank, world=2, mode=mode, repeats=2, capacity=16,
                             boundary_tokens=128, prefill_trials=1, yarn=0, corpora=1,
                             manifest="abc", vocab_begin=rank * 2, vocab_count=2, vocab=4))]
    count = 0
    for repeat in range(2):
        for lane, widths in (("verify", (4, 6, 8, 12, 16)), ("prefill", (1, 4, 5, 8, 9, 16, 17))):
            for width in widths:
                positions = width if lane == "prefill" else {4: 92, 6: 72, 8: 56, 12: 60, 16: 48}[width]
                changed = mode == "mma" and 4 < width <= 16
                for index in range(positions):
                    result.append(("score", dict(rank=rank, repeat=repeat, corpus="fixture", lane=lane,
                                                 width=width, index=index, target=0,
                                                 lse=math.log(math.exp(1 - 2 * rank) + math.exp(-2 * rank)),
                                                 target_logit=(1.001 if changed else 1.0) if rank == 0 else None,
                                                 top=[[2 * rank, 1 - 2 * rank], [2 * rank + 1, -2 * rank]],
                                                 hidden="same", logits="changed" if changed else "same")))
                    count += 1
                result.append(("case", dict(rank=rank, repeat=repeat, corpus="fixture", lane=lane,
                                            width=width, count=positions, tokens=128, token_hash="tokens")))
    result.append(("end", dict(rank=rank, count=count)))
    return result


class HeadCompareTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runs = {mode: {rank: records(rank, mode) for rank in range(2)} for mode in ("gemv", "mma")}

    def run_gate(self):
        for mode, ranks in self.runs.items():
            directory = Path(self.directory.name) / mode
            directory.mkdir(exist_ok=True)
            for rank, entries in ranks.items():
                (directory / f"r{rank}.log").write_text("".join(
                    f"[head_{kind}] {json.dumps(record)}\n" for kind, record in entries))
        return compare(str(Path(self.directory.name) / "gemv"), str(Path(self.directory.name) / "mma"))

    def test_complete_sharded_repeated_comparison(self):
        report = self.run_gate()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 12)
        expected_nll = math.log(sum(math.exp(x) for x in (1, 0, -1, -2))) - 1
        self.assertAlmostEqual(report["cases"][0]["gemv_mean_nll"], expected_nll)
        self.assertAlmostEqual(report["cases"][2]["mean_nll_delta"], -0.001)

    def test_truncation(self):
        self.runs["mma"][1].pop()
        with self.assertRaisesRegex(ValueError, "begin or end"):
            self.run_gate()

    def test_duplicate_score(self):
        self.runs["mma"][0].insert(2, copy.deepcopy(self.runs["mma"][0][1]))
        with self.assertRaisesRegex(ValueError, "duplicate score"):
            self.run_gate()

    def test_unchanged_mode_drift(self):
        for kind, record in self.runs["mma"][0]:
            if kind == "score" and record["repeat"] == 1:
                record["logits"] = "nondeterministic"
                break
        with self.assertRaisesRegex(ValueError, "repeat differs"):
            self.run_gate()

    def test_missing_entire_case(self):
        entries = self.runs["mma"][0]
        self.runs["mma"][0] = [(kind, record) for kind, record in entries
                               if not (kind in ("case", "score") and record["lane"] == "verify" and record["width"] == 6)]
        self.runs["mma"][0][-1][1]["count"] -= 144
        with self.assertRaisesRegex(ValueError, "missing or unexpected case"):
            self.run_gate()

    def test_rank_target_mismatch(self):
        for kind, record in self.runs["mma"][1]:
            if kind == "score":
                record["target"] = 1
        with self.assertRaisesRegex(ValueError, "rank target mismatch"):
            self.run_gate()

    def test_hidden_state_drift_between_modes(self):
        for entries in self.runs["mma"].values():
            for kind, record in entries:
                if kind == "score":
                    record["hidden"] = "different"
        with self.assertRaisesRegex(ValueError, "identical targets and hidden"):
            self.run_gate()

    def test_numerical_regression(self):
        for kind, record in self.runs["mma"][0]:
            if kind == "score" and 4 < record["width"] <= 16:
                record["target_logit"] -= 0.1
        self.assertFalse(self.run_gate()["passed"])

    def test_dispatch_control_drift(self):
        for kind, record in self.runs["mma"][0]:
            if kind == "score" and record["width"] == 4:
                record["logits"] = "wrong-dispatch"
        with self.assertRaisesRegex(ValueError, "dispatch control changed"):
            self.run_gate()

    def change_winner(self, margin):
        for mode in ("gemv", "mma"):
            for kind, record in self.runs[mode][0]:
                if kind == "score" and record["lane"] == "verify" and record["width"] == 6:
                    winner = int(mode == "mma")
                    record["top"] = [[winner, 1.0], [1 - winner, 1.0 - margin]]

    def test_near_tie_top1_flip(self):
        self.change_winner(0.01)
        report = self.run_gate()
        self.assertTrue(report["passed"])
        self.assertEqual(sum(c["top1_changes"] for c in report["cases"]), 72)

    def test_wide_margin_top1_flip(self):
        self.change_winner(0.1)
        with self.assertRaisesRegex(ValueError, "wide-margin top-1 flip"):
            self.run_gate()

    def test_near_tie_winners_on_different_shards(self):
        for mode, ranks in self.runs.items():
            for rank, entries in ranks.items():
                for kind, record in entries:
                    if kind != "score" or record["lane"] != "verify" or record["width"] != 6:
                        continue
                    best = 1.0 if rank == int(mode == "mma") else 0.99
                    second = -0.5 - rank
                    record["top"] = [[2 * rank, best], [2 * rank + 1, second]]
                    record["lse"] = math.log(math.exp(best) + math.exp(second))
                    record["target_logit"] = best if rank == 0 else None
        report = self.run_gate()
        self.assertTrue(report["passed"])
        self.assertEqual(sum(c["top1_changes"] for c in report["cases"]), 72)
        case = next(c for c in report["cases"] if c["lane"] == "verify" and c["width"] == 6)
        self.assertAlmostEqual(case["mean_nll_delta"], 0.01)

    def test_large_token_changes_cannot_cancel_in_the_mean(self):
        for kind, record in self.runs["mma"][0]:
            if kind == "score" and record["lane"] == "verify" and record["width"] == 6:
                record["target_logit"] += 1.1 if record["index"] % 2 else -1.1
        report = self.run_gate()
        case = next(c for c in report["cases"] if c["lane"] == "verify" and c["width"] == 6)
        self.assertLess(abs(case["mean_nll_delta"]), 0.02)
        self.assertEqual(case["big_delta_rate"], 1.0)
        self.assertFalse(report["passed"])

    def test_missing_rank(self):
        del self.runs["mma"][1]
        with self.assertRaisesRegex(ValueError, "missing rank"):
            self.run_gate()



class CompactCompareTest(unittest.TestCase):
    run_gate = HeadCompareTest.run_gate

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runs = {}
        for mode, policy in (("gemv", "off"), ("mma", "on")):
            self.runs[mode] = {}
            for rank in range(2):
                header = records(rank, "gemv")[0]
                header[1].update(compaction=policy, capacity=64, prefill_trials=0)
                entries = [header]
                count = 0
                for repeat in range(2):
                    for (lane, width), (requests, per) in COMPACT_SHAPES.items():
                        n = 1024 if per == 4 else 128
                        positions = ((n // requests - 17) // per) * width
                        for index in range(positions):
                            changed = policy == "on" and not lane.startswith("dense")
                            entries.append(("score", dict(
                                rank=rank, repeat=repeat, corpus="fixture", lane=lane, width=width,
                                index=index, target=0, lse=math.log(math.exp(1 - 2 * rank) + math.exp(-2 * rank)),
                                target_logit=(1.001 if changed else 1.0) if rank == 0 else None,
                                top=[[2 * rank, 1 - 2 * rank], [2 * rank + 1, -2 * rank]],
                                hidden="changed" if changed else "same", logits="changed" if changed else "same")))
                            count += 1
                        entries.append(("case", dict(rank=rank, repeat=repeat, corpus="fixture", lane=lane,
                                                     width=width, count=positions, tokens=1024, token_hash="tokens")))
                entries.append(("end", dict(rank=rank, count=count)))
                self.runs[mode][rank] = entries

    def test_compaction_allows_shape_rounding_and_checks_dense_controls(self):
        result = self.run_gate()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 6)
        for case in result["cases"]:
            self.assertAlmostEqual(case["mean_nll_delta"], 0 if case["lane"].startswith("dense") else -0.001)

    def test_truncated_compact_run(self):
        self.runs["mma"][1].pop()
        with self.assertRaisesRegex(ValueError, "begin or end"):
            self.run_gate()

    def test_compaction_repeat_drift(self):
        for kind, record in self.runs["mma"][0]:
            if kind == "score" and record["repeat"] == 1:
                record["logits"] = "drift"
                break
        with self.assertRaisesRegex(ValueError, "repeat differs"):
            self.run_gate()

    def test_dense_mapping_control_drift(self):
        for entries in self.runs["mma"].values():
            for kind, record in entries:
                if kind == "score" and record["lane"] == "dense16":
                    record["hidden"] = "drift"
        with self.assertRaisesRegex(ValueError, "dispatch control changed"):
            self.run_gate()

    def test_sparse_numerical_regression(self):
        for kind, record in self.runs["mma"][0]:
            if kind == "score" and record["lane"] == "sparse2":
                record["target_logit"] -= .1
        self.assertFalse(self.run_gate()["passed"])

    def test_missing_sparse_case(self):
        for ranks in self.runs.values():
            for rank, entries in ranks.items():
                removed = sum(kind == "score" and record["lane"] == "sparse2" for kind, record in entries)
                ranks[rank] = [(kind, record) for kind, record in entries
                               if not (kind in ("score", "case") and record["lane"] == "sparse2")]
                ranks[rank][-1][1]["count"] -= removed
        with self.assertRaisesRegex(ValueError, "missing or unexpected case"):
            self.run_gate()

if __name__ == "__main__":
    unittest.main()
