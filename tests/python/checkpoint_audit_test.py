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


class Glm53ClassificationTest(unittest.TestCase):
    """The full GLM-5.3 (GlmMoeDsa) inventory: main and draft layers share
    the `model.layers.N.` prefix; the draft is every N >= num_hidden_layers."""

    def test_classes_are_mutually_exclusive(self):
        cases = {
            "lm_head.weight": "lm_head",
            "model.embed_tokens.weight": "embed",
            "model.norm.weight": "norm",
            "model.layers.0.self_attn.q_a_proj.weight": "dsa_attention",
            "model.layers.3.self_attn.kv_b_proj.weight_packed": "dsa_attention",
            "model.layers.3.self_attn.kv_b_proj.weight_scale": "dsa_attention",
            "model.layers.3.self_attn.q_a_layernorm.weight": "norm",
            "model.layers.6.self_attn.indexer.k_norm.bias": "dsa_indexer",
            "model.layers.0.mlp.down_proj.weight": "dense_mlp",
            "model.layers.3.mlp.experts.255.up_proj.weight_shape": "routed_expert",
            "model.layers.3.mlp.shared_experts.gate_proj.weight_packed": "shared_expert",
            "model.layers.3.mlp.gate.weight": "router",
            "model.layers.3.mlp.gate.e_score_correction_bias": "router",
            "model.layers.3.input_layernorm.weight": "norm",
            "model.layers.78.mlp.experts.0.gate_proj.weight": "mtp",
            "model.layers.78.eh_proj.weight": "mtp",
            "model.layers.78.self_attn.indexer.wk.weight": "mtp",
            "unexpected.weight": "other",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(audit.classify_glm53(name, 78), expected)


class Glm53PackedContractTest(unittest.TestCase):
    """compressed-tensors pack-quantized triples: I32 words [N, K*bits/32],
    BF16 scales [N, K/64], I64 shape [2], the width of the one config group
    that targets the module, nothing a group targets stored plain."""

    GROUPS = [
        (audit.re.compile(r"model\.layers\.(?:[3-9]|[1-6][0-9]|7[0-7])\.self_attn\.(?:q_a_proj|o_proj)$"), 8),
        (audit.re.compile(r"model\.layers\.(?:[3-9]|[1-6][0-9]|7[0-7])\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"), 4),
    ]

    @staticmethod
    def triple(base, n, k, bits, scale_dtype="BF16", shape_dtype="I64"):
        return {
            base + ".weight_packed": ("f", "I32", (n, k * bits // 32)),
            base + ".weight_scale": ("f", scale_dtype, (n, k // 64)),
            base + ".weight_shape": ("f", shape_dtype, (2,)),
        }

    def test_valid_triples_report_their_width_and_geometry(self):
        tensors = {}
        tensors.update(self.triple("model.layers.3.self_attn.q_a_proj", 2048, 6144, 8))
        tensors.update(self.triple("model.layers.3.mlp.experts.7.down_proj", 6144, 2048, 4))
        tensors["model.layers.0.self_attn.q_a_proj.weight"] = ("f", "BF16", (2048, 6144))
        triples = audit.validate_glm53_packed(tensors, self.GROUPS)
        self.assertEqual(triples["model.layers.3.self_attn.q_a_proj"],
                         {"bits": 8, "n": 2048, "k": 6144, "words": 2048 * 1536 * 4, "scales": 2048 * 96 * 2})
        self.assertEqual(triples["model.layers.3.mlp.experts.7.down_proj"]["bits"], 4)
        self.assertEqual(triples["model.layers.3.mlp.experts.7.down_proj"]["k"], 2048)

    def test_violations_are_rejected(self):
        base = "model.layers.3.self_attn.q_a_proj"
        incomplete = self.triple(base, 2048, 6144, 8)
        del incomplete[base + ".weight_shape"]
        wrong_width = self.triple(base, 2048, 6144, 4)          # the group says 8
        wrong_scale = self.triple(base, 2048, 6144, 8, scale_dtype="F32")
        untargeted = self.triple("model.layers.3.self_attn.kv_a_proj_with_mqa", 576, 6144, 8)
        both = self.triple(base, 2048, 6144, 8)
        both[base + ".weight"] = ("f", "BF16", (2048, 6144))
        plain_in_group = {base + ".weight": ("f", "BF16", (2048, 6144))}
        for label, tensors in [("incomplete", incomplete), ("wrong width", wrong_width),
                               ("wrong scale dtype", wrong_scale), ("untargeted module", untargeted),
                               ("packed and plain", both), ("plain where the group packs", plain_in_group)]:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    audit.validate_glm53_packed(tensors, self.GROUPS)

    def test_quant_groups_come_from_the_config(self):
        config = {"quantization_config": {"format": "pack-quantized", "config_groups": {
            "group_0": {"targets": ["re:model\\.layers\\.3\\.self_attn\\.o_proj$"],
                        "weights": {"num_bits": 8, "group_size": 64, "strategy": "group", "symmetric": True, "type": "int"}}}}}
        groups = audit.glm53_quant_groups(config)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][1], 8)
        self.assertTrue(groups[0][0].search("model.layers.3.self_attn.o_proj"))
        with self.assertRaises(ValueError):
            audit.glm53_quant_groups({"quantization_config": {"format": "float-quantized"}})


if __name__ == "__main__":
    unittest.main()
