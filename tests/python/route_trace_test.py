#!/usr/bin/env python3
"""Tests for the route-trace traffic model (M4 deliverable 5): the format is
pinned by golden bytes shared with tests/unit/glm_trace_test.cpp, and the
analysis is pinned to DESIGN §3's two anchors — a uniformly random trace
must reproduce the hypergeometric busiest-rank occupancy (3.515) and the
7.457 GB / 32.42 ms critical path; a single-rank trace must reproduce the
12.198 GB worst-placement row."""
import random
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import route_trace_traffic as rtt  # noqa: E402

GOLDEN = (
    b"DGPPTC1\x01"
    + struct.pack("<I", 1)             # num_layers
    + struct.pack("<IIQ", 3, 2, 3)     # layer_idx=3, top_k=2, tokens=3
    + struct.pack("<6i", 0, 5, 5, 0, 7, 3)
    + struct.pack("<6f", 0.25, 0.75, 0.5, 0.5, 1.0, 0.0)
)


class FormatTest(unittest.TestCase):
    def test_golden_bytes_parse(self):
        # Same literal the C++ writer test asserts; if either side changes
        # the format, one of the two tests fails.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(GOLDEN)
            path = f.name
        try:
            layers = rtt.read_trace(path)
            self.assertEqual(len(layers), 1)
            l = layers[0]
            self.assertEqual((l.layer_idx, l.top_k, l.tokens), (3, 2, 3))
            self.assertEqual(l.ids, [0, 5, 5, 0, 7, 3])
            self.assertEqual(l.weights, [0.25, 0.75, 0.5, 0.5, 1.0, 0.0])
        finally:
            os.unlink(path)

    def test_bad_magic_rejected(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"NOTMAGIC" + GOLDEN[8:])
            path = f.name
        try:
            with self.assertRaises(ValueError):
                rtt.read_trace(path)
        finally:
            os.unlink(path)


def write_trace(path, layers):
    """Write a DGPPTC1 trace: layers = [(layer_idx, top_k, ids, weights)]."""
    out = [rtt.MAGIC, struct.pack("<I", len(layers))]
    for layer_idx, top_k, ids, weights in layers:
        tokens = len(ids) // top_k
        out.append(struct.pack("<IIQ", layer_idx, top_k, tokens))
        out.append(struct.pack(f"<{len(ids)}i", *ids))
        out.append(struct.pack(f"<{len(weights)}f", *weights))
    with open(path, "wb") as f:
        f.write(b"".join(out))


class AnalysisTest(unittest.TestCase):
    def test_uniform_trace_reproduces_hypergeometric_model(self):
        rng = random.Random(42)
        tokens, top_k = 512, 8
        layers = []
        for l in range(rtt.N_MOE_LAYERS):
            ids = []
            for _ in range(tokens):
                ids.extend(sorted(rng.sample(range(rtt.N_EXPERTS), top_k)))
            layers.append((l, top_k, ids, [1.0 / top_k] * len(ids)))
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            write_trace(path, layers)
            r = rtt.analyze(rtt.read_trace(path))
            self.assertAlmostEqual(r["mean_busiest"], rtt.UNIFORM_BUSIEST,
                                   delta=0.06)
            self.assertAlmostEqual(r["critical_gb"], rtt.CRITICAL_GB_UNIFORM,
                                   delta=0.15)
        finally:
            os.unlink(path)

    def test_single_rank_trace_reproduces_worst_placement(self):
        # Every token selects experts 0..7 (rank 0): busiest = 8 exactly —
        # DESIGN §3's "all selected experts on one rank: 12.198 GB" row.
        tokens, top_k = 64, 8
        ids = []
        for _ in range(tokens):
            ids.extend(range(top_k))
        layers = [(l, top_k, list(ids), [0.125] * len(ids))
                  for l in range(rtt.N_MOE_LAYERS)]
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            write_trace(path, layers)
            r = rtt.analyze(rtt.read_trace(path))
            self.assertEqual(r["mean_busiest"], 8.0)
            self.assertAlmostEqual(r["critical_gb"], 12.198, delta=0.05)
        finally:
            os.unlink(path)

    def test_expert_bytes_match_checkpoint_budget(self):
        # 42 layers x 288 experts x expert_bytes must be the routed share of
        # the checkpoint (304.48-304.52 GB in docs/checkpoint_budget.md).
        total = rtt.N_MOE_LAYERS * rtt.N_EXPERTS * rtt.expert_bytes()
        self.assertGreater(total, 304.4e9)
        self.assertLess(total, 304.6e9)


if __name__ == "__main__":
    unittest.main()
