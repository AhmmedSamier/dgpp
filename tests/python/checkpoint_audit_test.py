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


class Dsv41ClassificationTest(unittest.TestCase):
    """DeepSeek-V4.1-Flash (2026-09-13): backbone layers under `layers.N.`,
    the DSpark draft under `mtp.S.`, the Engram tables apart from the Engram
    projections, the vision tower its own class."""

    def test_classes_are_mutually_exclusive(self):
        cases = {
            "head.weight": "lm_head",
            "embed.weight": "embed",
            "norm.weight": "norm",
            "layers.3.attn.wq_b.weight": "attention",
            "layers.3.attn.wo_a.scale": "attention",
            "layers.3.attn.attn_sink": "attention",
            "layers.3.attn.q_norm.weight": "norm",
            "layers.2.attn.indexer.wk.weight": "indexer",
            "layers.2.attn.compressor.wgate.weight": "compressor",
            "layers.3.ffn.experts.383.w2.scale": "routed_expert",
            "layers.3.ffn.shared_experts.w1.weight": "shared_expert",
            "layers.3.ffn.gate.bias_vl": "router",
            "layers.3.hc_ffn_fn": "mhc",
            "layers.14.engram.embed.scale": "engram_table",
            "layers.14.engram.wkv.weight": "engram",
            "layers.14.engram.q_weight": "engram",
            "mtp.0.main_proj.weight": "draft",
            "mtp.2.markov_head.head.weight": "draft",
            "mtp.1.ffn.experts.5.w3.weight": "draft_expert",
            "vision.blocks.0.attn.wqkv.weight": "vision",
            "aligner.w1.bias": "vision",
            "image_newline": "vision",
            "layers.40.attn.wq_a.weight": "other",
            "unexpected.weight": "other",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(audit.classify_dsv41(name, 40), expected)


class Dsv41PairContractTest(unittest.TestCase):
    """The release's quantized pairs: fp8 `.weight` F8_E4M3 [N, K] with an
    F8_E8M0 `.scale` on the 32x32 grid, MXFP4 `.weight` I8 [N, K/2] with a
    `.scale` [N, K/32], the Engram tables [rows, 256] with [rows, 8]."""

    @staticmethod
    def fp8(base, n, k):
        return {base + ".weight": ("f", "F8_E4M3", (n, k)), base + ".scale": ("f", "F8_E8M0", (-(-n // 32), -(-k // 32)))}

    @staticmethod
    def fp4(base, n, k):
        return {base + ".weight": ("f", "I8", (n, k // 2)), base + ".scale": ("f", "F8_E8M0", (n, k // 32))}

    def test_valid_pairs_report_their_format(self):
        tensors = {}
        tensors.update(self.fp8("layers.3.attn.wq_b", 32768, 1280))
        tensors.update(self.fp4("layers.3.ffn.experts.7.w2", 5120, 2304))
        tensors["layers.1.engram.embed.weight"] = ("f", "F8_E4M3", (1000, 256))
        tensors["layers.1.engram.embed.scale"] = ("f", "F8_E8M0", (1000, 8))
        tensors["layers.3.attn_norm.weight"] = ("f", "BF16", (5120,))
        pairs = audit.validate_dsv41_pairs(tensors)
        self.assertEqual(pairs["layers.3.attn.wq_b"]["fmt"], "fp8")
        self.assertEqual(pairs["layers.3.ffn.experts.7.w2"], {"fmt": "mxfp4", "n": 5120, "k": 2304,
                                                                "payload": 5120 * 1152, "scales": 5120 * 72})
        self.assertEqual(pairs["layers.1.engram.embed"]["fmt"], "engram")

    def test_violations_are_rejected(self):
        no_scale = {"layers.3.attn.wq_b.weight": ("f", "F8_E4M3", (32768, 1280))}
        wrong_grid = self.fp8("layers.3.attn.wq_b", 32768, 1280)
        wrong_grid["layers.3.attn.wq_b.scale"] = ("f", "F8_E8M0", (256, 10))
        wrong_fp4 = self.fp4("layers.3.ffn.experts.7.w2", 5120, 2304)
        wrong_fp4["layers.3.ffn.experts.7.w2.scale"] = ("f", "F8_E8M0", (5120, 144))
        orphan_scale = {"layers.3.attn.wq_b.scale": ("f", "F8_E8M0", (1024, 40))}
        stray_e8m0 = {"layers.3.something": ("f", "F8_E8M0", (4,))}
        bf16_scale = self.fp8("layers.3.attn.wq_b", 32768, 1280)
        bf16_scale["layers.3.attn.wq_b.scale"] = ("f", "BF16", (1024, 40))
        for label, tensors in [("payload without scale", no_scale), ("wrong fp8 grid", wrong_grid),
                               ("wrong fp4 scale", wrong_fp4), ("scale without payload", orphan_scale),
                               ("e8m0 outside a pair", stray_e8m0), ("bf16 scale", bf16_scale)]:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    audit.validate_dsv41_pairs(tensors)


if __name__ == "__main__":
    unittest.main()
