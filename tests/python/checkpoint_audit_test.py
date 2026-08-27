#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import checkpoint_audit as audit  # noqa: E402


class ClassificationTest(unittest.TestCase):
    def setUp(self):
        self.layers = ["linear_attention", "deepseek_sparse_attention"]

    def test_model_sections_are_mutually_exclusive(self):
        cases = {
            "model.visual.blocks.0.mlp.gate_proj.weight": "vision",
            "model.language_model.embed_tokens.weight": "embed",
            "lm_head.weight": "lm_head",
            "model.language_model.norm.weight": "norm",
            "model.language_model.layers.2.mlp.experts.0.down_proj.weight": "mtp",
            "model.language_model.layers.0.self_attn.q_proj.weight": "kda",
            "model.language_model.layers.1.self_attn.q_a_proj.weight": "dsa",
            "model.language_model.layers.1.self_attn.indexer.wk.weight": "dsa_indexer",
            "model.language_model.layers.0.mlp.gate_proj.weight": "dense_mlp",
            "model.language_model.layers.1.mlp.experts.7.down_proj.weight": "routed_expert",
            "model.language_model.layers.1.mlp.shared_experts.down_proj.weight": "shared_expert",
            "model.language_model.layers.1.mlp.gate.weight": "router",
            "model.language_model.layers.1.hc_attn_base": "mhc",
            "model.language_model.layers.1.input_layernorm.weight": "norm",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(audit.classify(name, self.layers), expected)

    def test_unknown_tensor_is_not_silently_folded_into_text_traffic(self):
        self.assertEqual(audit.classify("unexpected.weight", self.layers), "other")


class ExpertOccupancyTest(unittest.TestCase):
    def test_glm_geometry_matches_exact_hypergeometric_result(self):
        self.assertAlmostEqual(
            audit.expected_busiest_rank(288, 8, 4), 3.515121767306306, places=12
        )

    def test_one_rank_and_one_selection(self):
        self.assertEqual(audit.expected_busiest_rank(288, 8, 1), 8.0)
        self.assertEqual(audit.expected_busiest_rank(288, 1, 4), 1.0)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.expected_busiest_rank(4, 5, 2)


class ExpertGeometryTest(unittest.TestCase):
    @staticmethod
    def inventory(sizes=(10, 10)):
        return {
            f"model.language_model.layers.3.mlp.experts.{expert}.weight": {
                "class": "routed_expert",
                "nbytes": size,
            }
            for expert, size in enumerate(sizes)
        }

    def test_complete_equal_whole_experts_pass(self):
        self.assertEqual(audit.validate_expert_geometry(self.inventory(), 2), 1)

    def test_missing_or_unequal_experts_fail(self):
        with self.assertRaises(ValueError):
            audit.validate_expert_geometry(self.inventory((10,)), 2)
        with self.assertRaises(ValueError):
            audit.validate_expert_geometry(self.inventory((10, 11)), 2)


if __name__ == "__main__":
    unittest.main()
