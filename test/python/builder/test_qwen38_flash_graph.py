# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from builders.qwen3_8_flash.graph import Weight, build_target_graph, validate_config
from builders.qwen3_8_flash.quantization import Int2Weights, SafetensorsCheckpoint, prepack_int2
from builders.qwen3_8_flash.weights import TargetWeights, bind_weights, quantize_fp8, rotary_cache, tensor_bytes
from onnx import TensorProto as T
from onnx import helper
from safetensors.torch import save_file


def target_config():
    return {
        "model_type": "qwen4_exp",
        "quantization_config": {"config_groups": {"experts": {"weights": {"num_bits": 4, "group_size": 16}}}},
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "ple_embed_dim": 2560,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "indexer_compress_ratio": 4,
            "indexer_budget": 2048,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "max_position_embeddings": 262144,
            "vocab_size": 248320,
            "ple_layer_ids": [2],
            "output_gate_type": "sigmoid",
            "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)],
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
            "rms_norm_eps": 1e-6,
            "norm_topk_prob": True,
        },
    }


class TargetGraphTests(unittest.TestCase):
    def test_gdn_decay_uses_a_and_beta_uses_b(self):
        model, _ = build_target_graph(target_config())
        gdns = [node for node in model.graph.node if node.op_type == "GatedDeltaNet"]
        self.assertEqual(len(gdns), 36)
        for node in gdns:
            self.assertTrue(node.input[4].endswith(".in_proj_a.out"))
            self.assertTrue(node.input[5].endswith(".in_proj_b.out"))

    def test_native_context_and_s4_contract(self):
        model, weights = build_target_graph(target_config())
        operations = Counter(node.op_type for node in model.graph.node)
        for op, count in (
            ("QMoE", 48),
            ("GatedDeltaNet", 36),
            ("CausalConvWithState", 37),
            ("DynamicSparseAttention", 12),
            ("PackedSparseAttentionIndexer", 12),
            ("HyperConnectionLowRankMix", 96),
            ("FoldMid2", 1),
            ("FoldTail", 1),
        ):
            self.assertEqual(operations[op], count, op)
        inputs = {value.name: value for value in model.graph.input}
        outputs = {value.name for value in model.graph.output}
        self.assertEqual(inputs["past_key_values.3.key"].type.tensor_type.shape.dim[2].dim_value, 262148)
        self.assertEqual(inputs["past_key_values.3.index_key"].type.tensor_type.shape.dim[1].dim_value, 65537)
        for value in inputs:
            if not value.startswith("past_key_values.") or value.endswith((".key", ".value", ".index_key")):
                continue
            for tag in ("present.", "present_mid.", "present_mid2.", "present_mid3."):
                self.assertIn(tag + value.removeprefix("past_key_values."), outputs)
        self.assertIn("aux_hidden_states", outputs)
        self.assertIn("fold.tail_done", outputs)
        self.assertEqual({node.domain for node in model.graph.node}, {"", "com.microsoft"})
        self.assertEqual({p.domain: p.version for p in model.opset_import}, {"": 21, "com.microsoft": 1})
        self.assertEqual(len({w.name for w in weights}), len(weights))
        self.assertEqual(sum(w.transform.startswith("cache_scale:") for w in weights), 24)
        self.assertEqual(sum(w.transform.startswith("experts:") for w in weights), 192)

    def test_graph_is_topologically_complete_and_deterministic(self):
        config = target_config()
        original = copy.deepcopy(config)
        model, weights = build_target_graph(config)
        again, again_weights = build_target_graph(config)
        self.assertEqual(config, original)
        self.assertEqual(model.SerializeToString(), again.SerializeToString())
        self.assertEqual(weights, again_weights)
        known = (
            {v.name for v in model.graph.input} | {t.name for t in model.graph.initializer} | {w.name for w in weights}
        )
        for node in model.graph.node:
            for name in node.input:
                self.assertTrue(not name or name in known, (node.name, name))
            for name in node.output:
                if name:
                    self.assertNotIn(name, known)
                    known.add(name)
        self.assertTrue(all(value.name in known for value in model.graph.output))

    def test_fresh_calibration_graph_is_not_folded_or_int8(self):
        model, weights = build_target_graph(target_config(), calibration=True, calibration_capacity=2048)
        self.assertFalse(any(node.op_type.startswith("Fold") for node in model.graph.node))
        self.assertFalse(any(w.transform.startswith("cache_scale:") for w in weights))
        outputs = {v.name for v in model.graph.output}
        for layer in range(3, 48, 4):
            for kind in ("key", "value"):
                self.assertIn(f"calibration.{layer}.{kind}.amax", outputs)
                value = next(v for v in model.graph.input if v.name == f"past_key_values.{layer}.{kind}")
                self.assertEqual(value.type.tensor_type.elem_type, T.BFLOAT16)
                self.assertEqual(value.type.tensor_type.shape.dim[2].dim_value, 2048)

    def test_unsupported_model_config_is_rejected(self):
        for key in ("max_position_embeddings", "num_experts", "hc_count"):
            config = target_config()
            config["text_config"][key] -= 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_config(config)
        config = target_config()
        config["text_config"]["layer_types"][0] = "full_attention"
        with self.assertRaisesRegex(ValueError, "ordering"):
            build_target_graph(config)
        with self.assertRaisesRegex(ValueError, "capacity"):
            build_target_graph(target_config(), calibration=True, calibration_capacity=3)


class TargetWeightTests(unittest.TestCase):
    def test_swiglu_interleaves_gate_and_up_rows_before_prepacking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specifications = [
                Weight("fc1.weight", T.UINT8, (512, 16, 16), "0", "experts:fc1:experts_weights"),
                Weight("fc1.scale", T.BFLOAT16, (512, 16, 1), "0", "experts:fc1:scales"),
                Weight("fc2.weight", T.UINT8, (512, 8, 16), "0", "experts:fc2:experts_weights"),
                Weight("fc2.scale", T.BFLOAT16, (512, 8, 1), "0", "experts:fc2:scales"),
            ]
            converted = {}
            for index, field in enumerate(("gate_proj", "up_proj", "down_proj"), 1):
                packed = torch.arange(index * 16, index * 16 + 128).to(torch.uint8).reshape(8, 16)
                scales = torch.arange(index * 8, index * 8 + 8).to(torch.bfloat16).reshape(8, 1)
                converted[field] = Int2Weights(packed, scales, 0.0, 0.0)
            with TargetWeights(None, root, target_config()) as writer:
                for spec in specifications:
                    offset = writer.align()
                    length = int(np.prod(spec.shape)) * (2 if spec.dtype == T.BFLOAT16 else 1)
                    writer.entries[spec.name] = {"offset": offset, "length": length}
                    writer.stream.seek(length - 1, 1)
                    writer.stream.write(b"\0")
                writer.write_expert({spec.transform: spec for spec in specifications}, 0, converted)
                entries = writer.entries
            expected = torch.empty((16, 16), dtype=torch.uint8)
            expected[0::2] = converted["gate_proj"].packed
            expected[1::2] = converted["up_proj"].packed
            entry = entries["fc1.weight"]
            actual = np.memmap(
                root / "model.onnx.data", mode="r", dtype=np.uint8, offset=entry["offset"], shape=(16, 16)
            )
            np.testing.assert_array_equal(actual, prepack_int2(expected).numpy())
            entry = entries["fc1.scale"]
            actual = np.memmap(root / "model.onnx.data", mode="r", dtype=np.uint16, offset=entry["offset"], shape=(16,))
            expected = torch.empty((16,), dtype=torch.bfloat16)
            expected[0::2] = converted["gate_proj"].scales.flatten()
            expected[1::2] = converted["up_proj"].scales.flatten()
            np.testing.assert_array_equal(actual, expected.view(torch.uint16).numpy())

    def test_singleton_transpose_serialization(self):
        weight = torch.arange(2560).reshape(1, 2560).to(torch.bfloat16).T
        self.assertEqual(tensor_bytes(weight), weight.reshape(-1).view(torch.uint16).numpy().tobytes())

    def test_fp8_block_conversion_and_roundtrip(self):
        weights = torch.randn((7, 256), generator=torch.Generator().manual_seed(987))
        quantized, scales = quantize_fp8(weights, 128)
        expected_scales = weights.reshape(7, 2, 128).abs().amax(-1) / 448
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(quantized.shape, weights.shape)
        self.assertTrue(torch.isfinite(quantized.float()).all())
        with self.assertRaisesRegex(ValueError, "divisible"):
            quantize_fp8(weights[:, :255], 128)
        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_fp8(torch.full((1, 128), float("nan")), 128)

    def test_rotary_cache(self):
        for width in (32, 64):
            cos, sin = (rotary_cache(16, width, kind) for kind in ("cos", "sin"))
            self.assertEqual(cos.dtype, torch.bfloat16)
            self.assertEqual(cos.shape, (16, width))
            self.assertTrue(torch.all(cos[0] == 1))
            self.assertTrue(torch.all(sin[0] == 0))
            theta = torch.arange(32).float() * (2 / 64)
            expected = (1 / (10000000**theta)).cos().to(torch.bfloat16)
            if width == 64:
                expected = expected.repeat(2)
            torch.testing.assert_close(cos[1], expected)
        with self.assertRaises(ValueError):
            rotary_cache(0, 32, "cos")

    def test_streaming_serialization_and_external_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            value = torch.arange(32).reshape(2, 16).to(torch.bfloat16)
            save_file({"matrix": value, "norm": value.flatten().clone()}, raw / "model.safetensors")
            specs = [
                Weight("linear.weight", T.FLOAT8E4M3FN, (2, 16), "matrix", "fp8:16"),
                Weight("linear.scale", T.FLOAT, (2, 1), "matrix", "fp8_scale:16"),
                Weight("norm", T.BFLOAT16, (32,), "norm", "norm"),
                Weight("transposed", T.BFLOAT16, (16, 2), "matrix", "transpose"),
            ]
            with SafetensorsCheckpoint(raw) as checkpoint, TargetWeights(checkpoint, root, target_config()) as writer:
                writer.materialize(specs)
                entries = copy.deepcopy(writer.entries)
                with self.assertRaisesRegex(ValueError, "Invalid target weight"):
                    writer.write(specs[0], value)
            for entry in entries.values():
                self.assertEqual(entry["offset"] % 4096, 0)
                self.assertLessEqual(entry["offset"] + entry["length"], (root / "model.onnx.data").stat().st_size)
            model = helper.make_model(helper.make_graph([], "weights", [], [], []))
            bind_weights(model, specs, entries)
            path = root / "weights.onnx"
            onnx.save(model, path)
            loaded = onnx.load(path)
            values = {t.name: onnx.numpy_helper.to_array(t) for t in loaded.graph.initializer}
            np.testing.assert_array_equal(
                values["transposed"].view(np.uint16), value.T.contiguous().view(torch.uint16).numpy()
            )
            self.assertEqual(json.loads((root / "target_weights.json").read_text()), entries)
            with self.assertRaises(FileExistsError), TargetWeights(None, root, target_config()):
                pass


if __name__ == "__main__":
    unittest.main()
