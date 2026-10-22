#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from optimize_ported_package import (
    deduplicate_standard_expressions,
    fuse_attention_rotary,
    fuse_causal_conv_projection,
    fuse_causal_conv_state_updates,
    fuse_fp8_projection_slices,
    fuse_fp8_projection_split,
    fuse_gdn_gate_casts,
    fuse_gdn_qkv_split,
    fuse_hyper_connection_postmix,
    fuse_hyper_connections,
    fuse_rms_input_slices,
    fuse_rms_output_casts,
    fuse_rotary_casts,
    fuse_shared_expert_residual,
    fuse_silu_gate,
    fuse_standard_rms_norm,
    infer_value_types,
    optimize_model,
    quantize_prefill_outputs,
    refresh_weight_pool,
    share_rotary_cache_expands,
    untile_weights,
    write_performance_metadata,
)


class RotaryEmbedding(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        position_ids,
        cos_cache,
        sin_cache,
        num_heads=None,
        rotary_embedding_dim=None,
        interleaved=0,
        use_fp32_arithmetic=0,
        **kwargs,
    ):
        assert not interleaved and x.shape[0] == 1 and position_ids.tolist() == [0]
        dtype = x.dtype
        if use_fp32_arithmetic:
            x, cos_cache, sin_cache = (array.astype(np.float32) for array in (x, cos_cache, sin_cache))
        heads = x.reshape(1, x.shape[1], num_heads, -1)
        half = rotary_embedding_dim // 2
        cos = cos_cache[None, :, None, :]
        sin = sin_cache[None, :, None, :]
        result = heads.copy()
        result[..., :half] = heads[..., :half] * cos - heads[..., half : 2 * half] * sin
        result[..., half : 2 * half] = heads[..., half : 2 * half] * cos + heads[..., :half] * sin
        return (result.reshape(x.shape).astype(dtype),)


def capture_value(model, value, dtype=TensorProto.FLOAT):
    branch = helper.make_graph(
        [helper.make_node("Identity", [value], ["branch_capture"])],
        "capture",
        [],
        [helper.make_tensor_value_info("branch_capture", dtype, None)],
    )
    condition = "__capture_condition"
    output = "__capture_output"
    model.graph.initializer.append(numpy_helper.from_array(np.array(True), condition))
    model.graph.node.append(helper.make_node("If", [condition], [output], then_branch=branch, else_branch=branch))
    model.graph.output.append(helper.make_tensor_value_info(output, dtype, None))
    return model


class OptimizePortedPackageTest(unittest.TestCase):
    def test_conv_state_update_fusion_numerics_and_guards(self):
        class CausalConvWithState(OpRun):
            op_domain = "com.microsoft"

            def _run(
                self,
                x,
                weight,
                bias,
                past,
                dilation=1,
                state_channels_last=-1,
                input_channels=0,
                state_length=0,
                activation="none",
                **kwargs,
            ):
                state = past.transpose(0, 2, 1) if state_channels_last == 0 else past
                sequence = np.concatenate([state, x], axis=1)
                history = state.shape[1]
                result = np.zeros_like(x)
                for t in range(x.shape[1]):
                    for k in range(weight.shape[2]):
                        result[:, t] += sequence[:, t + k * dilation] * weight[:, 0, k]
                if bias is not None:
                    result += bias
                if activation == "silu":
                    result = result / (1 + np.exp(-result))
                present = sequence[:, -history:]
                first = sequence[:, 1 : history + 1]
                if state_channels_last == 0:
                    return result, present.transpose(0, 2, 1), first.transpose(0, 2, 1)
                return result, present

        def graph(length=2, extra=None, mutation=None, captured=False):
            constants = [numpy_helper.from_array(np.linspace(-0.3, 0.4, 12).astype(np.float32).reshape(4, 1, 3), "w")]
            nodes = [
                helper.make_node("Transpose", ["past"], ["past_cl"], perm=[0, 2, 1]),
                helper.make_node(
                    "CausalConvWithState",
                    ["x", "w", "", "past_cl"],
                    ["y", "carry"],
                    domain="com.microsoft",
                    channels_last=1,
                    dilation=2,
                    activation="silu",
                    name="conv",
                ),
            ]

            def add_slice(source, output, start, end, axis, step=1):
                names = [output + suffix for suffix in ("_start", "_end", "_axis", "_step")]
                constants.extend(
                    numpy_helper.from_array(np.array([v], np.int64), name)
                    for name, v in zip(names, (start, end, axis, step), strict=True)
                )
                nodes.append(helper.make_node("Slice", [source, *names], [output]))

            add_slice("past", "tail", 1, 2**31 - 1, 2)
            add_slice("x", "first", 0, 1, 1)
            nodes.extend(
                [
                    helper.make_node("Transpose", ["first"], ["first_cf"], perm=[0, 2, 1]),
                    helper.make_node("Concat", ["tail", "first_cf"], ["mid"], axis=2),
                    helper.make_node("Transpose", ["carry"], ["present"], perm=[0, 2, 1]),
                ]
            )
            if mutation == "dilation":
                nodes[1].attribute.extend([helper.make_attribute("state_window", 1)])
            if mutation == "tail":
                constants[1].CopyFrom(numpy_helper.from_array(np.array([2], np.int64), "tail_start"))
            if mutation == "layout":
                nodes[0].attribute[0].ints[:] = [0, 1, 2]
            outputs = [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, length, 4]),
                helper.make_tensor_value_info("present", TensorProto.FLOAT, [2, 4, 4]),
                helper.make_tensor_value_info("mid", TensorProto.FLOAT, [2, 4, 4]),
            ]
            if extra:
                nodes.append(helper.make_node("Identity", [extra], ["observed"]))
                outputs.append(helper.make_tensor_value_info("observed", TensorProto.FLOAT, None))
            if captured:
                body = helper.make_graph(
                    [helper.make_node("Identity", ["tail"], ["captured"])],
                    "body",
                    [],
                    [helper.make_tensor_value_info("captured", TensorProto.FLOAT, [2, 4, 3])],
                )
                nodes.append(helper.make_node("If", ["condition"], ["branch"], then_branch=body, else_branch=body))
                constants.append(numpy_helper.from_array(np.array(True), "condition"))
                outputs.append(helper.make_tensor_value_info("branch", TensorProto.FLOAT, [2, 4, 3]))
            input_shape = (
                None if mutation == "unknown_rank" else ([2, length, 2, 2] if mutation == "rank4" else [2, length, 4])
            )
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "state",
                    [
                        helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                        helper.make_tensor_value_info("past", TensorProto.FLOAT, [2, 4, 4]),
                    ],
                    outputs,
                    constants,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            for length in (1, 2, 7):
                model = graph(length)
                feed = {
                    "x": np.linspace(-1, 1, 2 * length * 4).astype(np.float32).reshape(2, length, 4),
                    "past": np.linspace(-0.7, 0.8, 32).astype(np.float32).reshape(2, 4, 4),
                }
                expected = ReferenceEvaluator(model, new_ops=[CausalConvWithState]).run(None, feed)
                onnx.save(model, source)
                if path.exists():
                    path.unlink()
                os.link(source, path)
                before = source.read_bytes()
                self.assertEqual(fuse_causal_conv_state_updates(path), 1)
                self.assertEqual(source.read_bytes(), before)
                fused = onnx.load(path)
                self.assertEqual(len(fused.graph.node), 1)
                self.assertEqual(list(fused.graph.node[0].output), ["y", "present", "mid"])
                actual = ReferenceEvaluator(fused, new_ops=[CausalConvWithState]).run(None, feed)
                for a, b in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(fuse_causal_conv_state_updates(path), 0)
            for extra in ("past_cl", "carry", "tail", "first", "first_cf"):
                onnx.save(graph(extra=extra), path)
                self.assertEqual(fuse_causal_conv_state_updates(path), 0)
            for mutation in ("dilation", "tail", "layout", "unknown_rank", "rank4"):
                onnx.save(graph(mutation=mutation), path)
                self.assertEqual(fuse_causal_conv_state_updates(path), 0)
            onnx.save(graph(captured=True), path)
            self.assertEqual(fuse_causal_conv_state_updates(path), 0)

    def fp8_model(self, scale_dtype=np.float32, output="output", missing_shape=False):
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("DequantizeLinear", ["weights", "scales"], ["float_weights"], axis=1),
                    helper.make_node("Cast", ["float_weights"], ["bf16_weights"], to=TensorProto.BFLOAT16),
                    helper.make_node("MatMul", ["input", "bf16_weights"], [output]),
                ],
                "fp8",
                [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 2, 16])],
                [helper.make_tensor_value_info(output, TensorProto.BFLOAT16, None if missing_shape else [1, 2, 512])],
                [
                    numpy_helper.from_array(np.ones((16, 512), dtype=ml_dtypes.float8_e4m3fn), "weights"),
                    numpy_helper.from_array(np.ones(512, dtype=scale_dtype), "scales"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        )

    def test_fp8_scale_types_and_external_length(self):
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.onnx"
                model = self.fp8_model(scale_dtype=dtype)
                tensor = model.graph.initializer[0]
                (path.parent / "weights.bin").write_bytes(b"pad" + tensor.raw_data)
                onnx.external_data_helper.set_external_data(tensor, location="weights.bin", offset=3)
                tensor.ClearField("raw_data")
                onnx.save(model, path)
                result = optimize_model(path)
                self.assertEqual(result, {"fused_fp8_matmuls": 1} if dtype is np.float32 else {})
                loaded = onnx.load(path)
                if dtype is np.float32:
                    weights = numpy_helper.to_array(loaded.graph.initializer[0])
                    self.assertEqual(weights.shape, (512, 16))
                    np.testing.assert_array_equal(weights.view(np.uint8), np.full((512, 16), 0x38, np.uint8))
                else:
                    self.assertEqual(
                        [node.op_type for node in loaded.graph.node], ["DequantizeLinear", "Cast", "MatMul"]
                    )

    def test_fp8_generated_names_preserve_existing_values_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            model = self.fp8_model()
            for name in ("weights.performance_nk", "weights.performance_nk_1"):
                model.graph.initializer.append(numpy_helper.from_array(np.array([7], np.float32), name))
                model.graph.node.append(helper.make_node("Identity", [name], [name + ".observed"]))
            model.graph.node.append(helper.make_node("Identity", ["input"], ["scales.performance_n1"]))
            capture_value(model, "scales.performance_n1_1", TensorProto.BFLOAT16)
            model.graph.initializer.append(
                numpy_helper.from_array(np.ones((1,), dtype=ml_dtypes.bfloat16), "scales.performance_n1_1")
            )
            onnx.save(model, path)
            provenance = {}
            self.assertEqual(optimize_model(path, provenance), {"fused_fp8_matmuls": 1})
            optimized = onnx.load(path)
            node = next(n for n in optimized.graph.node if n.op_type == "MatMulBlockQuantizedFp8Weight")
            self.assertEqual(list(node.input[1:]), ["weights.performance_nk_2", "scales.performance_n1_2"])
            self.assertEqual(set(provenance), set(node.input[1:]))
            tensors = {t.name: numpy_helper.to_array(t) for t in optimized.graph.initializer}
            for name in ("weights.performance_nk", "weights.performance_nk_1"):
                np.testing.assert_array_equal(tensors[name], [7])
            names = [t.name for t in optimized.graph.initializer] + [
                out for n in optimized.graph.node for out in n.output
            ]
            self.assertEqual(len(names), len(set(names)))

    def test_head_quantization_after_fp8_fusion_without_output_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            onnx.save(self.fp8_model(output="value", missing_shape=True), path)
            self.assertEqual(optimize_model(path), {"fused_fp8_matmuls": 1})
            self.assertEqual(quantize_prefill_outputs(path, {"value": np.ones(512, np.float32)}, head=True), 1)
            model = onnx.load(path)
            onnx.checker.check_model(model)
            self.assertEqual(model.graph.output[0].type.tensor_type.elem_type, TensorProto.INT8)
            self.assertEqual(model.graph.node[0].output[0], "value.performance_unquantized")

    def test_head_output_capture_rejects_quantization_without_modifying_model(self):
        for name in ("key", "value"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.onnx"
                model = self.fp8_model(output=name)
                capture_value(model, name, TensorProto.BFLOAT16)
                onnx.save(model, path)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "captured head output"):
                    quantize_prefill_outputs(path, {name: np.ones(512, np.float32)}, head=True)
                self.assertEqual(path.read_bytes(), before)

    def test_rms_input_slice_fusion_numerics_and_guards(self):
        class BranchwiseRMSNorm(OpRun):
            op_domain = "com.microsoft"

            def _run(self, x, scale, input_start=0, input_width=0, epsilon=1e-6, **kwargs):
                if input_width:
                    x = x[..., input_start : input_start + input_width]
                mean = np.mean(x * x, axis=-1, keepdims=True)
                y = (x / np.sqrt(mean + np.float32(epsilon))) * scale
                return y, y.astype(np.float32)

        def graph(
            *,
            start=3,
            end=19,
            axis=-1,
            step=1,
            mode=1,
            branches=0,
            observed=False,
            captured=False,
            shared=False,
            unknown=False,
        ):
            width = end - start if 0 <= start < end <= 32 and step == 1 else 16
            constants = [
                numpy_helper.from_array(np.array(values, np.int64), name)
                for name, values in (
                    ("starts", [start]),
                    ("ends", [end]),
                    ("axis", [axis]),
                    ("steps", [step]),
                    ("shape", [0, 0, 3, -1 if unknown else 32]),
                )
            ]
            constants.append(numpy_helper.from_array(np.linspace(0.5, 1.5, width).astype(np.float32), "scale"))
            nodes = [
                helper.make_node("Reshape", ["packed", "shape"], ["heads"]),
                helper.make_node("Slice", ["heads", "starts", "ends", "axis", "steps"], ["selected"]),
                helper.make_node(
                    "BranchwiseRMSNorm",
                    ["selected", "scale"],
                    ["y", "wide"],
                    domain="com.microsoft",
                    use_onnx_arithmetic=mode,
                    num_branches=branches,
                ),
            ]
            outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("y", "wide")]
            if observed:
                outputs.append(helper.make_tensor_value_info("selected", TensorProto.FLOAT, None))
            if shared:
                nodes.append(helper.make_node("Identity", ["selected"], ["shared"]))
            if captured:
                branch = helper.make_graph(
                    [helper.make_node("Identity", ["selected"], ["capture"])],
                    "branch",
                    [],
                    [helper.make_tensor_value_info("capture", TensorProto.FLOAT, None)],
                )
                constants.append(numpy_helper.from_array(np.array(True), "condition"))
                nodes.append(
                    helper.make_node("If", ["condition"], ["captured"], then_branch=branch, else_branch=branch)
                )
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "slice_rms",
                    [
                        helper.make_tensor_value_info(
                            "packed", TensorProto.FLOAT, ["batch", "sequence", "unknown" if unknown else 96]
                        )
                    ],
                    outputs,
                    constants,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            rng = np.random.default_rng(51)
            for start, axis in ((0, 3), (3, -1)):
                original = graph(start=start, axis=axis)
                onnx.save(original, source)
                before = source.read_bytes()
                os.link(source, path)
                self.assertEqual(fuse_rms_input_slices(path), 1)
                self.assertEqual(source.read_bytes(), before)
                optimized = onnx.load(path)
                self.assertEqual(fuse_rms_input_slices(path), 0)
                self.assertFalse(any(node.op_type == "Slice" for node in optimized.graph.node))
                self.assertEqual(next(op.version for op in optimized.opset_import if op.domain == "com.microsoft"), 2)
                for batch, sequence in ((1, 1), (2, 3), (3, 9)):
                    feeds = {"packed": rng.normal(size=(batch, sequence, 96)).astype(np.float32)}
                    expected = ReferenceEvaluator(original, new_ops=[BranchwiseRMSNorm]).run(None, feeds)
                    actual = ReferenceEvaluator(optimized, new_ops=[BranchwiseRMSNorm]).run(None, feeds)
                    for a, b in zip(expected, actual, strict=True):
                        np.testing.assert_array_equal(a, b)
                path.unlink()
            for options in (
                {"observed": True},
                {"captured": True},
                {"shared": True},
                {"unknown": True},
                {"start": -1},
                {"end": 40},
                {"axis": 2},
                {"step": 2},
                {"mode": 0},
                {"branches": 3},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    before = path.read_bytes()
                    self.assertEqual(fuse_rms_input_slices(path), 0)
                    self.assertEqual(path.read_bytes(), before)

    def test_fixed_generic_contracts_prove_types_without_opaque_op_assumptions(self):
        def graph(dtype=TensorProto.BFLOAT16, declared=0):
            nodes = [
                helper.make_node("Opaque", ["input"], ["opaque"], domain="test.opaque"),
                helper.make_node("SigmoidGateMul", ["opaque", "projection"], ["gated"], domain="com.microsoft"),
                helper.make_node("Slice", ["projection", "starts", "ends", "axes"], ["selected"]),
            ]
            info = [helper.make_tensor_value_info("opaque", declared, None)] if declared else []
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "proof",
                    [
                        helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 2, 8]),
                        helper.make_tensor_value_info("projection", dtype, [1, 2, 8]),
                    ],
                    [],
                    [
                        numpy_helper.from_array(np.array(values, np.int64), name)
                        for name, values in (("starts", [0]), ("ends", [4]), ("axes", [2]))
                    ],
                    value_info=info,
                ),
                opset_imports=[
                    helper.make_opsetid("", 21),
                    helper.make_opsetid("com.microsoft", 1),
                    helper.make_opsetid("test.opaque", 1),
                ],
            )

        types = infer_value_types(graph())
        for name in ("opaque", "projection", "gated", "selected"):
            self.assertEqual(types[name], TensorProto.BFLOAT16)
        for model in (graph(dtype=TensorProto.FLOAT), graph(declared=TensorProto.FLOAT)):
            with self.assertRaisesRegex(ValueError, "SigmoidGateMul requires BF16"):
                infer_value_types(model)
        model = graph()
        del model.graph.node[1]
        self.assertNotIn("opaque", infer_value_types(model))

    def test_rotary_cast_fusion_preserves_float_arithmetic_and_external_values(self):
        def graph(
            *,
            external=False,
            captured=False,
            widened_output=False,
            input_type=TensorProto.FLOAT16,
            output_type=TensorProto.FLOAT16,
            norm=False,
        ):
            nodes = [
                helper.make_node(
                    "BranchwiseRMSNorm", ["input"], ["rounded", "float"], domain="com.microsoft", use_onnx_arithmetic=1
                )
                if norm
                else helper.make_node("Cast", ["input"], ["float"], to=TensorProto.FLOAT),
                helper.make_node("Reshape", ["float", "flat_shape"], ["flat"]),
            ]
            for name in ("cos", "sin"):
                nodes.extend(
                    [
                        helper.make_node("Cast", [name], [name + "_float"], to=TensorProto.FLOAT),
                        helper.make_node("Expand", [name + "_float", "expanded_shape"], [name + "_expanded"]),
                        helper.make_node("Reshape", [name + "_expanded", "cache_shape"], [name + "_cache"]),
                    ]
                )
            nodes.extend(
                [
                    helper.make_node(
                        "RotaryEmbedding",
                        ["flat", "positions", "cos_cache", "sin_cache"],
                        ["rotated"],
                        domain="com.microsoft",
                        num_heads=2,
                        rotary_embedding_dim=32,
                        interleaved=0,
                    ),
                    helper.make_node("Reshape", ["rotated", "output_shape"], ["view"]),
                    helper.make_node("Cast", ["view"], ["output"], to=output_type),
                ]
            )
            outputs = [helper.make_tensor_value_info("output", output_type, [1, 2, 2, 64])]
            if external:
                outputs.append(helper.make_tensor_value_info("rotated", TensorProto.FLOAT, [1, 2, 128]))
            if widened_output:
                outputs.append(helper.make_tensor_value_info("float", TensorProto.FLOAT, [1, 2, 2, 64]))
            tensors = [
                numpy_helper.from_array(np.array(values, np.int64), name)
                for name, values in (
                    ("flat_shape", [1, -1, 128]),
                    ("cache_shape", [-1, 16]),
                    ("expanded_shape", [1, 2, 16]),
                    ("output_shape", [1, 2, 2, 64]),
                    ("positions", [0]),
                )
            ]
            if captured:
                captured_name = captured if isinstance(captured, str) else "rotated"
                captured_shape = {"rotated": [1, 2, 128], "float": [1, 2, 2, 64], "cos_float": [1, 1, 16]}[
                    captured_name
                ]
                branch = helper.make_graph(
                    [helper.make_node("Identity", [captured_name], ["captured_rotary"])],
                    "branch",
                    [],
                    [helper.make_tensor_value_info("captured_rotary", TensorProto.FLOAT, captured_shape)],
                )
                tensors.append(numpy_helper.from_array(np.array(True), "condition"))
                nodes.append(
                    helper.make_node("If", ["condition"], ["captured_result"], then_branch=branch, else_branch=branch)
                )
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "cast_rotary",
                    [
                        helper.make_tensor_value_info("input", input_type, [1, 2, 2, 64]),
                        helper.make_tensor_value_info("cos", input_type, [1, 1, 16]),
                        helper.make_tensor_value_info("sin", input_type, [1, 1, 16]),
                    ],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_rotary_casts(path), 1)
            self.assertEqual(fuse_rotary_casts(path), 0)
            self.assertEqual(source.read_bytes(), original)
            model = onnx.load(path)
            self.assertFalse(any(node.op_type == "Cast" for node in model.graph.node))
            names = [node.name for node in model.graph.node if node.name]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(next(opset.version for opset in model.opset_import if opset.domain == "com.microsoft"), 2)
            rng = np.random.default_rng(42)
            feed = {
                "input": rng.normal(size=(1, 2, 2, 64)).astype(np.float16),
                "cos": rng.normal(size=(1, 1, 16)).astype(np.float16),
                "sin": rng.normal(size=(1, 1, 16)).astype(np.float16),
            }
            expected = ReferenceEvaluator(str(source), new_ops=[RotaryEmbedding]).run(None, feed)[0]
            actual = ReferenceEvaluator(str(path), new_ops=[RotaryEmbedding]).run(None, feed)[0]
            np.testing.assert_array_equal(actual, expected)
            legacy = graph()
            legacy.graph.node.extend(
                [helper.make_node("Identity", ["input"], ["untouched"], name="cos_expanded_ActivationStorage")]
            )
            rotary = next(node for node in legacy.graph.node if node.op_type == "RotaryEmbedding")
            rotary.attribute.append(helper.make_attribute("use_fp32_arithmetic", 0))
            next(opset for opset in legacy.opset_import if opset.domain == "com.microsoft").version = 2
            onnx.save(legacy, path)
            self.assertEqual(fuse_rotary_casts(path), 1)
            model = onnx.load(path)
            names = [node.name for node in model.graph.node if node.name]
            self.assertEqual(len(names), len(set(names)))
            rotary = next(node for node in model.graph.node if node.op_type == "RotaryEmbedding")
            self.assertEqual([attr.i for attr in rotary.attribute if attr.name == "use_fp32_arithmetic"], [1])
            for options in (
                {"external": True},
                {"captured": True},
                {"input_type": TensorProto.FLOAT},
                {"output_type": TensorProto.BFLOAT16},
            ):
                onnx.save(graph(**options), path)
                original = path.read_bytes()
                self.assertEqual(fuse_rotary_casts(path), 0, options)
                self.assertEqual(path.read_bytes(), original)
            for norm in (False, True):
                onnx.save(graph(norm=norm, widened_output=True), path)
                self.assertEqual(fuse_rotary_casts(path), 1)
                model = onnx.load(path)
                self.assertIn("float", {name for node in model.graph.node for name in node.output})
            for captured in ("float", "cos_float"):
                onnx.save(graph(captured=captured), path)
                self.assertEqual(fuse_rotary_casts(path), 1)
                model = onnx.load(path)
                self.assertIn(captured, {name for node in model.graph.node for name in node.output})
            onnx.save(graph(norm=True), path)
            self.assertEqual(fuse_rotary_casts(path), 1)
            model = onnx.load(path)
            self.assertEqual(list(model.graph.node[0].output), ["rounded"])

    def test_rotary_cache_sharing_proves_token_dimensions_and_preserves_outputs(self):
        def graph(second_batch="batch", second_sequence="sequence", protected=False, captured=False):
            nodes = []
            tensors = [
                numpy_helper.from_array(np.array([0, 1], np.int64), "token_axes"),
                numpy_helper.from_array(np.array([2], np.int64), "cache_width"),
            ]
            for name in ("x", "y"):
                nodes.extend(
                    [
                        helper.make_node("Shape", [name], [name + "_shape"]),
                        helper.make_node("Gather", [name + "_shape", "token_axes"], [name + "_tokens"], axis=0),
                        helper.make_node(
                            "Concat", [name + "_tokens", "cache_width"], [name + "_expanded_shape"], axis=0
                        ),
                        helper.make_node("Expand", ["cache", name + "_expanded_shape"], [name + "_expanded"]),
                        helper.make_node("Identity", [name + "_expanded"], [name + "_output"]),
                    ]
                )
            outputs = [helper.make_tensor_value_info(name + "_output", TensorProto.FLOAT, None) for name in ("x", "y")]
            if protected:
                outputs.append(helper.make_tensor_value_info("y_expanded_shape", TensorProto.INT64, [3]))
            if captured:
                branch = helper.make_graph(
                    [helper.make_node("Identity", ["y_expanded_shape"], ["captured_shape"])],
                    "branch",
                    [],
                    [helper.make_tensor_value_info("captured_shape", TensorProto.INT64, [3])],
                )
                tensors.append(numpy_helper.from_array(np.array(True), "condition"))
                nodes.append(
                    helper.make_node("If", ["condition"], ["captured_result"], then_branch=branch, else_branch=branch)
                )
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "cache",
                    [
                        helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "sequence", 2, 4]),
                        helper.make_tensor_value_info("y", TensorProto.FLOAT, [second_batch, second_sequence, 3, 4]),
                        helper.make_tensor_value_info("cache", TensorProto.FLOAT, ["batch", 1, 2]),
                    ],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(share_rotary_cache_expands(path), 1)
            self.assertEqual(deduplicate_standard_expressions(path), {"Expand": 1})
            self.assertEqual(share_rotary_cache_expands(path), 0)
            self.assertEqual(source.read_bytes(), original)
            for batch, sequence in ((1, 1), (2, 3), (3, 9)):
                feed = {
                    "x": np.zeros((batch, sequence, 2, 4), np.float32),
                    "y": np.zeros((batch, sequence, 3, 4), np.float32),
                    "cache": np.arange(batch * 2, dtype=np.float32).reshape(batch, 1, 2),
                }
                for expected, actual in zip(
                    ReferenceEvaluator(str(source)).run(None, feed),
                    ReferenceEvaluator(str(path)).run(None, feed),
                    strict=True,
                ):
                    np.testing.assert_array_equal(actual, expected)
            for options in (
                {"second_batch": "other_batch"},
                {"second_sequence": "other_sequence"},
                {"second_batch": None},
                {"protected": True},
                {"captured": True},
            ):
                onnx.save(graph(**options), path)
                before = path.read_bytes()
                self.assertEqual(share_rotary_cache_expands(path), 0, options)
                self.assertEqual(path.read_bytes(), before)

    def test_rms_output_cast_fusion_preserves_rounded_and_external_values(self):
        def graph(mode=1, cast_type=TensorProto.FLOAT):
            return helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node(
                            "BranchwiseRMSNorm",
                            ["input", "scale"],
                            ["rounded"],
                            domain="com.microsoft",
                            use_onnx_arithmetic=mode,
                        ),
                        helper.make_node("Cast", ["rounded"], ["widened"], to=cast_type),
                    ],
                    "widening",
                    [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 2, 256])],
                    [
                        helper.make_tensor_value_info("rounded", TensorProto.BFLOAT16, [1, 2, 256]),
                        helper.make_tensor_value_info("widened", cast_type, [1, 2, 256]),
                    ],
                    [numpy_helper.from_array(np.ones(256, np.float32), "scale")],
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_rms_output_casts(path), 1)
            self.assertEqual(source.read_bytes(), original)
            nodes = onnx.load(path).graph.node
            self.assertEqual(len(nodes), 1)
            self.assertEqual(list(nodes[0].output), ["rounded", "widened"])
            self.assertEqual(fuse_rms_output_casts(path), 0)
            for mode, cast_type in ((0, TensorProto.FLOAT), (1, TensorProto.FLOAT16)):
                onnx.save(graph(mode, cast_type), path)
                self.assertEqual(fuse_rms_output_casts(path), 0)

    def test_standard_rms_fusion_preserves_boundaries_and_guards(self):
        def graph(
            *, external=False, captured=False, width=256, dtype=TensorProto.BFLOAT16, epsilon=1e-6, epsilon_rank=0
        ):
            nodes = [
                helper.make_node("Reshape", ["input", "shape"], ["heads"]),
                helper.make_node("Cast", ["heads"], ["float"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["float", "float"], ["square"]),
                helper.make_node("ReduceMean", ["square", "axes"], ["mean"], keepdims=1),
                helper.make_node("Add", ["mean", "epsilon"], ["adjusted"]),
                helper.make_node("Sqrt", ["adjusted"], ["root"]),
                helper.make_node("Div", ["float", "root"], ["normalized"]),
                helper.make_node("Mul", ["normalized", "scale"], ["scaled"]),
                helper.make_node("Cast", ["scaled"], ["output"], to=dtype),
            ]
            outputs = [helper.make_tensor_value_info("output", dtype, [1, 2, 3, 256])]
            if external:
                outputs.append(helper.make_tensor_value_info("root", TensorProto.FLOAT, [1, 2, 3, 1]))
            tensors = [
                numpy_helper.from_array(np.array([0, 0, 3, 256], np.int64), "shape"),
                numpy_helper.from_array(np.array([-1], np.int64), "axes"),
                numpy_helper.from_array(np.full((1,) * epsilon_rank, epsilon, np.float32), "epsilon"),
                numpy_helper.from_array(np.ones(width, np.float32), "scale"),
            ]
            if captured:
                branch = helper.make_graph(
                    [helper.make_node("Identity", ["root"], ["captured_root"])],
                    "branch",
                    [],
                    [helper.make_tensor_value_info("captured_root", TensorProto.FLOAT, [1, 2, 3, 1])],
                )
                tensors.append(numpy_helper.from_array(np.array(True), "condition"))
                nodes.append(
                    helper.make_node("If", ["condition"], ["captured_result"], then_branch=branch, else_branch=branch)
                )
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "normalization",
                    [helper.make_tensor_value_info("input", dtype, [1, 2, 768])],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_standard_rms_norm(path), 1)
            self.assertEqual(source.read_bytes(), original)
            fused = onnx.load(path).graph.node[-1]
            self.assertEqual(fused.op_type, "BranchwiseRMSNorm")
            self.assertEqual(list(fused.input), ["heads", "scale"])
            self.assertEqual(list(fused.output), ["output"])
            self.assertEqual(
                {attr.name: helper.get_attribute_value(attr) for attr in fused.attribute}["use_onnx_arithmetic"], 1
            )
            self.assertEqual(fuse_standard_rms_norm(path), 0)
            for options in (
                {"external": True},
                {"captured": True},
                {"width": 128},
                {"dtype": TensorProto.FLOAT},
                {"epsilon": -1e-6},
                {"epsilon": np.nan},
                {"epsilon_rank": 5},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_standard_rms_norm(path), 0)
            for dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
                onnx.save(graph(dtype=dtype), path)
                self.assertEqual(fuse_standard_rms_norm(path), 1)
            standard_only = graph()
            standard_only.ClearField("opset_import")
            standard_only.opset_import.append(helper.make_opsetid("", 21))
            onnx.save(standard_only, path)
            self.assertEqual(fuse_standard_rms_norm(path), 1)
            optimized = onnx.load(path)
            self.assertEqual(
                next(opset.version for opset in optimized.opset_import if opset.domain == "com.microsoft"),
                2,
            )

    def test_gdn_qkv_split_requires_complete_private_compatible_views(self):
        def graph(
            *,
            external=False,
            captured=False,
            axis=2,
            sizes=(256, 256, 512),
            q_shape=(0, 0, 2, 128),
            state_shape=(1, 4, 128, 128),
            channels_last=1,
        ):
            tensors = [numpy_helper.from_array(np.array(sizes, np.int64), "sizes")]
            nodes = [
                helper.make_node(
                    "CausalConvWithState",
                    ["input", "weight"],
                    ["packed"],
                    domain="com.microsoft",
                    channels_last=channels_last,
                ),
                helper.make_node("Split", ["packed", "sizes"], ["q", "k", "v"], axis=axis),
            ]
            for name, shape in zip(("q", "k", "v"), (q_shape, (0, 0, 2, 128), (0, 0, 4, 128)), strict=False):
                tensors.append(numpy_helper.from_array(np.array(shape, np.int64), name + "_shape"))
                nodes.append(helper.make_node("Reshape", [name, name + "_shape"], [name + "_heads"]))
            nodes.append(
                helper.make_node(
                    "GatedDeltaNet",
                    ["q_heads", "k_heads", "v_heads", "", "decay", "beta", "state"],
                    ["output"],
                    domain="com.microsoft",
                )
            )
            outputs = [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 2, 4, 128])]
            if external:
                outputs.append(helper.make_tensor_value_info("k", TensorProto.BFLOAT16, [1, 2, 256]))
            if captured:
                branch = helper.make_graph(
                    [helper.make_node("Identity", ["k"], ["captured_key"])],
                    "branch",
                    [],
                    [helper.make_tensor_value_info("captured_key", TensorProto.BFLOAT16, [1, 2, 256])],
                )
                tensors.append(numpy_helper.from_array(np.array(True), "condition"))
                nodes.append(
                    helper.make_node("If", ["condition"], ["captured_result"], then_branch=branch, else_branch=branch)
                )
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "packed",
                    [
                        helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 2, 1024]),
                        helper.make_tensor_value_info("state", TensorProto.FLOAT, state_shape),
                    ],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_gdn_qkv_split(path), 1)
            self.assertEqual(source.read_bytes(), original)
            nodes = onnx.load(path).graph.node
            self.assertEqual([node.op_type for node in nodes], ["CausalConvWithState", "GatedDeltaNet"])
            self.assertEqual(list(nodes[-1].input)[:3], ["packed", "", ""])
            self.assertEqual(fuse_gdn_qkv_split(path), 0)
            for options in (
                {"external": True},
                {"captured": True},
                {"axis": 1},
                {"sizes": (256, 128, 512)},
                {"q_shape": (0, 0, 1, 256)},
                {"state_shape": (1, 2, 128, 128)},
                {"channels_last": 0},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_gdn_qkv_split(path), 0)

    def test_standard_expression_reuse_is_exact_atomic_and_idempotent(self):
        nodes = []
        for index in range(2):
            nodes += [
                helper.make_node("Cast", ["input"], [f"half{index}"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", [f"half{index}"], [f"rounded{index}"], to=TensorProto.FLOAT),
                helper.make_node("Unsqueeze", [f"rounded{index}", "axes"], [f"row{index}"]),
                helper.make_node("Expand", [f"row{index}", "shape"], [f"expanded{index}"]),
            ]
        nodes.append(helper.make_node("Add", ["expanded0", "expanded1"], ["output"]))
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "reuse",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
                [
                    numpy_helper.from_array(np.array([0], np.int64), "axes"),
                    numpy_helper.from_array(np.array([2, 4], np.int64), "shape"),
                ],
                value_info=[helper.make_tensor_value_info("half1", TensorProto.FLOAT16, [4])],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(model, source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(deduplicate_standard_expressions(path), {"Cast": 2, "Unsqueeze": 1, "Expand": 1})
            self.assertEqual(source.read_bytes(), original)
            optimized = onnx.load(path)
            onnx.checker.check_model(optimized)
            self.assertEqual(len(optimized.graph.value_info), 0)
            self.assertEqual(deduplicate_standard_expressions(path), {})
            for values in ([-0.0, 0.0, 1.0003, -3.14159], [np.inf, -np.inf, np.nan, 0.00000001]):
                feeds = {"input": np.array(values, np.float32)}
                before = ReferenceEvaluator(model).run(None, feeds)[0]
                after = ReferenceEvaluator(optimized).run(None, feeds)[0]
                self.assertEqual(before.tobytes(), after.tobytes())

    def test_standard_expression_reuse_preserves_attributes_domains_and_outputs(self):
        nodes = [
            helper.make_node("Cast", ["input"], ["first"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["input"], ["second"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["other"], ["third"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["input"], ["fourth"], to=TensorProto.FLOAT, domain="custom"),
            helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT),
            helper.make_node("RandomUniform", [], ["random0"], shape=[4]),
            helper.make_node("RandomUniform", [], ["random1"], shape=[4]),
        ]
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "guards",
                [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [4]) for name in ("input", "other")],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
            ),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("custom", 1)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            onnx.save(model, path)
            original = path.read_bytes()
            self.assertEqual(deduplicate_standard_expressions(path), {})
            self.assertEqual(path.read_bytes(), original)

    def test_standard_expression_reuse_preserves_subgraph_captures(self):
        branch = helper.make_graph(
            [helper.make_node("Identity", ["captured"], ["branch_output"])],
            "branch",
            [],
            [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [4])],
        )
        nodes = [
            helper.make_node("Cast", ["input"], ["first"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["input"], ["captured"], to=TensorProto.FLOAT),
            helper.make_node("If", ["condition"], ["output"], then_branch=branch, else_branch=branch),
        ]
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "captures",
                [
                    helper.make_tensor_value_info("input", TensorProto.FLOAT16, [4]),
                    helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
                ],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            onnx.save(model, path)
            original = path.read_bytes()
            self.assertEqual(deduplicate_standard_expressions(path), {})
            self.assertEqual(path.read_bytes(), original)
            onnx.checker.check_model(onnx.load(path))

    def test_gdn_gate_casts_require_matching_lossless_input_types(self):
        def graph(
            *,
            query_type=TensorProto.BFLOAT16,
            gate_type=TensorProto.BFLOAT16,
            cast_type=TensorProto.FLOAT,
            external=False,
            beta_cast=True,
        ):
            nodes = [
                helper.make_node("Cast", ["decay"], ["decay_float"], name="decay_cast", to=cast_type),
                helper.make_node("Cast", ["beta"], ["beta_float"], name="beta_cast", to=cast_type),
                helper.make_node(
                    "GatedDeltaNet",
                    ["q", "k", "v", "", "decay_float", "beta_float" if beta_cast else "beta"],
                    ["output"],
                    name="gdn",
                    domain="com.microsoft",
                ),
            ]
            inputs = [helper.make_tensor_value_info(name, query_type, [1, 2, 1, 128]) for name in ("q", "k", "v")]
            inputs += [helper.make_tensor_value_info(name, gate_type, [1, 2, 1]) for name in ("decay", "beta")]
            outputs = [helper.make_tensor_value_info("output", query_type, [1, 2, 1, 128])]
            if external:
                outputs.append(helper.make_tensor_value_info("decay_float", cast_type, [1, 2, 1]))
            return helper.make_model(
                helper.make_graph(nodes, "gates", inputs, outputs),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_gdn_gate_casts(path), 2)
            self.assertEqual(source.read_bytes(), original)
            nodes = onnx.load(path).graph.node
            self.assertEqual(len(nodes), 1)
            self.assertEqual(list(nodes[0].input)[4:6], ["decay", "beta"])
            self.assertEqual(fuse_gdn_gate_casts(path), 0)
            onnx.save(graph(external=True), path)
            self.assertEqual(fuse_gdn_gate_casts(path), 1)
            self.assertEqual([node.op_type for node in onnx.load(path).graph.node], ["Cast", "GatedDeltaNet"])
            onnx.save(capture_value(graph(), "decay_float", TensorProto.FLOAT), path)
            self.assertEqual(fuse_gdn_gate_casts(path), 1)
            self.assertIn("decay_float", {name for node in onnx.load(path).graph.node for name in node.output})
            for options in (
                {"query_type": TensorProto.FLOAT16},
                {"gate_type": TensorProto.FLOAT},
                {"cast_type": TensorProto.FLOAT16},
                {"beta_cast": False},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_gdn_gate_casts(path), 0)
            onnx.save(graph(query_type=TensorProto.FLOAT16, gate_type=TensorProto.FLOAT16), path)
            self.assertEqual(fuse_gdn_gate_casts(path), 2)

    def test_fp8_projection_slices_preserve_values_and_indexed_consumers(self):
        class MatMulBlockQuantizedFp8Weight(OpRun):
            op_domain = "com.microsoft"

            def _run(self, a, b, scale, split_sizes=None):
                output = a @ (b * scale).T
                if split_sizes is None:
                    return (output,)
                return tuple(np.split(output, np.cumsum(split_sizes)[:-1], axis=-1))

        class CausalConvWithState(OpRun):
            op_domain = "com.microsoft"

            def _run(
                self,
                x,
                weight,
                channels_last=None,
                ndim=None,
                input_channels=None,
                input_channel_offset=0,
                activation=None,
            ):
                # Only the indexed input contract is under test; CUDA tests cover convolution arithmetic.
                return (x[..., input_channel_offset : input_channel_offset + input_channels] * weight,)

        class SigmoidGateMul(OpRun):
            op_domain = "com.microsoft"

            def _run(
                self, attention, projection, head_dim=None, num_heads=None, q_stride=None, tok_stride=None, q_offset=0
            ):
                flat = projection.reshape(-1)
                indices = (
                    np.arange(attention.size // (num_heads * head_dim))[:, None, None] * tok_stride
                    + q_offset
                    + np.arange(num_heads)[None, :, None] * q_stride
                    + head_dim
                    + np.arange(head_dim)[None, None, :]
                )
                gates = flat[indices].reshape(attention.shape)
                return (attention / (1 + np.exp(-gates)),)

        rng = np.random.default_rng(17)
        for consumer in ("conv", "gate"):
            for batch, sequence in ((1, 1), (1, 2), (2, 5)):
                with self.subTest(consumer=consumer, batch=batch, sequence=sequence):
                    weight = rng.normal(size=(11, 4)).astype(np.float32)
                    tensors = [
                        numpy_helper.from_array(weight, "weight"),
                        numpy_helper.from_array(np.ones((11, 1), np.float32), "scale"),
                    ]
                    nodes = [
                        helper.make_node(
                            "MatMulBlockQuantizedFp8Weight",
                            ["input", "weight", "scale"],
                            ["projected"],
                            domain="com.microsoft",
                            name="projection",
                        )
                    ]
                    ranges = ((0, 3), (7, 11)) if consumer == "conv" else ((0, 3), (3, 7), (7, 11))
                    for index, (start, stop) in enumerate(ranges):
                        for suffix, value in (("start", start), ("end", stop), ("axis", 2)):
                            tensors.append(numpy_helper.from_array(np.array([value], np.int64), f"{suffix}{index}"))
                        nodes.append(
                            helper.make_node(
                                "Slice",
                                ["projected", f"start{index}", f"end{index}", f"axis{index}"],
                                [f"slice{index}"],
                                name=f"slice{index}",
                            )
                        )
                    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, sequence, 4])]
                    feeds = {"input": rng.normal(size=(batch, sequence, 4)).astype(np.float32)}
                    if consumer == "conv":
                        tensors.append(numpy_helper.from_array(np.arange(1, 5, dtype=np.float32), "conv_weight"))
                        nodes.append(
                            helper.make_node(
                                "CausalConvWithState",
                                ["projected", "conv_weight"],
                                ["indexed"],
                                name="conv",
                                domain="com.microsoft",
                                ndim=1,
                                channels_last=1,
                                input_channel_offset=3,
                                input_channels=4,
                            )
                        )
                    else:
                        inputs.append(
                            helper.make_tensor_value_info("attention", TensorProto.FLOAT, [batch, sequence, 2])
                        )
                        feeds["attention"] = rng.normal(size=(batch, sequence, 2)).astype(np.float32)
                        nodes.append(
                            helper.make_node(
                                "SigmoidGateMul",
                                ["attention", "projected"],
                                ["indexed"],
                                name="gate",
                                domain="com.microsoft",
                                head_dim=2,
                                num_heads=1,
                                q_offset=3,
                                q_stride=4,
                                tok_stride=11,
                            )
                        )
                    outputs = [
                        helper.make_tensor_value_info(f"slice{i}", TensorProto.FLOAT, None) for i in range(len(ranges))
                    ]
                    outputs.append(helper.make_tensor_value_info("indexed", TensorProto.FLOAT, None))
                    model = helper.make_model(
                        helper.make_graph(nodes, "values", inputs, outputs, tensors),
                        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                    )
                    ops = [MatMulBlockQuantizedFp8Weight, CausalConvWithState, SigmoidGateMul]
                    expected = ReferenceEvaluator(model, new_ops=ops).run(None, feeds)
                    with tempfile.TemporaryDirectory() as temporary:
                        path = Path(temporary) / "model.onnx"
                        onnx.save(model, path)
                        self.assertEqual(fuse_fp8_projection_slices(path), 1)
                        actual = ReferenceEvaluator(onnx.load(path), new_ops=ops).run(None, feeds)
                        for left, right in zip(expected, actual, strict=False):
                            np.testing.assert_array_equal(left, right)

    def test_fp8_attention_slices_rebase_and_rank_guards(self):
        def graph(*, rank=3, offset=1, stride=8, external=False, captured=False, axis=2, end=5):
            nodes = [
                helper.make_node(
                    "MatMulBlockQuantizedFp8Weight",
                    ["input", "weight", "scale"],
                    ["projected"],
                    name="projection",
                    domain="com.microsoft",
                ),
            ]
            tensors = [
                helper.make_tensor("weight", TensorProto.FLOAT8E4M3FN, [8, 16], [0] * 128),
                numpy_helper.from_array(np.ones((8, 1), np.float32), "scale"),
            ]
            for index, (begin, stop) in enumerate(((0, 1), (1, end), (5, 8))):
                for suffix, value in (("start", begin), ("end", stop), ("axis", axis)):
                    tensors.append(numpy_helper.from_array(np.array([value], np.int64), f"{suffix}{index}"))
                nodes.append(
                    helper.make_node(
                        "Slice",
                        ["projected", f"start{index}", f"end{index}", f"axis{index}"],
                        [f"part{index}"],
                        name=f"slice{index}",
                    )
                )
            nodes.append(
                helper.make_node(
                    "SigmoidGateMul",
                    ["attention", "projected"],
                    ["output"],
                    domain="com.microsoft",
                    name="gate",
                    num_heads=1,
                    head_dim=2,
                    q_stride=4,
                    tok_stride=stride,
                    q_offset=offset,
                )
            )
            outputs = [
                helper.make_tensor_value_info(name, TensorProto.BFLOAT16, None)
                for name in ("part0", "part1", "part2", "output")
            ]
            if external:
                outputs.append(helper.make_tensor_value_info("projected", TensorProto.BFLOAT16, None))
            shape = None if rank is None else [1] * (rank - 1) + [16]
            model = helper.make_model(
                helper.make_graph(
                    nodes,
                    "attention_slices",
                    [
                        helper.make_tensor_value_info("input", TensorProto.BFLOAT16, shape),
                        helper.make_tensor_value_info("attention", TensorProto.BFLOAT16, [1, 1, 2]),
                    ],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )
            return capture_value(model, "projected", TensorProto.BFLOAT16) if captured else model

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            onnx.save(graph(), path)
            # The custom projection's rank is propagated from the known activation rank.
            self.assertEqual(fuse_fp8_projection_slices(path), 1)
            projection, gate = onnx.load(path).graph.node
            self.assertEqual(list(projection.output), ["part0", "part1", "part2"])
            self.assertEqual(gate.input[1], "part1")
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in gate.attribute}
            self.assertEqual((attrs["q_offset"], attrs["tok_stride"]), (0, 4))
            for options in (
                {"rank": None},
                {"rank": 2},
                {"offset": 0},
                {"stride": 7},
                {"external": True},
                {"captured": True},
                {"axis": 1},
                {"end": 4},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_fp8_projection_slices(path), 0)

    def test_fp8_conv_slices_partitions_and_guards(self):
        def graph(*, axis=2, step=1, start=4, end=7, extra=False, external=False, channels_last=1):
            nodes = [
                helper.make_node(
                    "MatMulBlockQuantizedFp8Weight",
                    ["input", "weight", "scale"],
                    ["projected"],
                    name="projection",
                    domain="com.microsoft",
                ),
                helper.make_node(
                    "CausalConvWithState",
                    ["projected", "conv_weight"],
                    ["conv"],
                    name="conv",
                    domain="com.microsoft",
                    ndim=1,
                    channels_last=channels_last,
                    input_channels=4,
                    input_channel_offset=0,
                ),
                helper.make_node("Slice", ["projected", "start", "end", "axis", "step"], ["slice"], name="slice"),
            ]
            tensors = [
                helper.make_tensor("weight", TensorProto.FLOAT8E4M3FN, [7, 16], [0] * 112),
                numpy_helper.from_array(np.ones((7, 1), np.float32), "scale"),
            ]
            for name, value in (("start", start), ("end", end), ("axis", axis), ("step", step)):
                tensors.append(numpy_helper.from_array(np.array([value], np.int64), name))
            outputs = [helper.make_tensor_value_info(n, TensorProto.BFLOAT16, None) for n in ("conv", "slice")]
            if external:
                outputs.append(helper.make_tensor_value_info("projected", TensorProto.BFLOAT16, None))
            if extra:
                nodes.append(helper.make_node("Identity", ["projected"], ["extra"]))
                outputs.append(helper.make_tensor_value_info("extra", TensorProto.BFLOAT16, None))
            return helper.make_model(
                helper.make_graph(
                    nodes,
                    "slices",
                    [
                        helper.make_tensor_value_info("input", TensorProto.BFLOAT16, None),
                        helper.make_tensor_value_info("conv_weight", TensorProto.BFLOAT16, None),
                    ],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_fp8_projection_slices(path), 1)
            self.assertEqual(source.read_bytes(), original)
            model = onnx.load(path)
            self.assertEqual(len(model.graph.node), 2)
            projection, conv = model.graph.node
            self.assertEqual(list(projection.output), ["projected.indexed_partition", "slice"])
            self.assertEqual(helper.get_attribute_value(projection.attribute[-1]), [4, 3])
            self.assertEqual(conv.input[0], projection.output[0])
            self.assertEqual(fuse_fp8_projection_slices(path), 0)
            for options in (
                {"axis": 1},
                {"step": 2},
                {"start": 3},
                {"start": 5},
                {"end": 8},
                {"end": 6},
                {"extra": True},
                {"external": True},
                {"channels_last": 0},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_fp8_projection_slices(path), 0)
            onnx.save(graph(axis=-1), path)
            self.assertEqual(fuse_fp8_projection_slices(path), 1)

    def test_fp8_projection_split_guards_and_atomic_save(self):
        def graph(*, axis=-1, widths=(1, 7, 9), external=False, rank=None, captured=False):
            nodes = [
                helper.make_node(
                    "MatMulBlockQuantizedFp8Weight",
                    ["input", "weight", "scale"],
                    ["projected"],
                    domain="com.microsoft",
                    name="projection",
                    block_size=256,
                ),
                helper.make_node("Split", ["projected", "widths"], ["x", "y", "z"], name="split", axis=axis),
            ]
            tensors = [
                helper.make_tensor("weight", TensorProto.FLOAT8E4M3FN, [17, 256], [0] * (17 * 256)),
                numpy_helper.from_array(np.ones((17, 1), np.float32), "scale"),
                numpy_helper.from_array(np.array(widths, np.int64), "widths"),
            ]
            outputs = [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, None) for name in ("x", "y", "z")]
            if external:
                outputs.append(helper.make_tensor_value_info("projected", TensorProto.BFLOAT16, None))
            shape = None if rank is None else [1] * (rank - 1) + [256]
            model = helper.make_model(
                helper.make_graph(
                    nodes,
                    "split",
                    [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, shape)],
                    outputs,
                    tensors,
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )
            return capture_value(model, "projected", TensorProto.BFLOAT16) if captured else model

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_fp8_projection_split(path), 1)
            self.assertEqual(source.read_bytes(), original)
            optimized = onnx.load(path)
            self.assertEqual(len(optimized.graph.node), 1)
            projection = optimized.graph.node[0]
            self.assertEqual(list(projection.output), ["x", "y", "z"])
            self.assertEqual(
                {attr.name: helper.get_attribute_value(attr) for attr in projection.attribute},
                {"block_size": 256, "split_sizes": [1, 7, 9]},
            )
            self.assertEqual(fuse_fp8_projection_split(path), 0)
            for options in (
                {"axis": 0},
                {"axis": 0, "rank": 2},
                {"widths": (1, 7, 8)},
                {"widths": (0, 8, 9)},
                {"widths": (8, 9)},
                {"external": True},
                {"captured": True},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_fp8_projection_split(path), 0)
            onnx.save(graph(axis=2, rank=3), path)
            self.assertEqual(fuse_fp8_projection_split(path), 1)

    def test_silu_gate_fusion_type_width_and_external_user_guards(self):
        def graph(*, external=None, bad_width=False, dtype=TensorProto.BFLOAT16):
            nodes = [
                helper.make_node("Split", ["input", "sizes"], ["gate", "up"], axis=-1),
                helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
                helper.make_node("Mul", ["gate", "sigmoid"], ["silu"]),
                helper.make_node("Mul", ["silu", "up"], ["output"], name="gated"),
            ]
            inputs = [helper.make_tensor_value_info("input", dtype, ["batch", "sequence", 8])]
            outputs = [helper.make_tensor_value_info("output", dtype, None)]
            if external:
                nodes.append(helper.make_node("Identity", [external], ["external"]))
                outputs.append(helper.make_tensor_value_info("external", dtype, None))
            sizes = numpy_helper.from_array(np.array([3, 5] if bad_width else [4, 4], np.int64), "sizes")
            return helper.make_model(
                helper.make_graph(nodes, "silu", inputs, outputs, [sizes]),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_silu_gate(path), 1)
            self.assertEqual(source.read_bytes(), original)
            model = onnx.load(path)
            self.assertEqual(len(model.graph.node), 2)
            fused = model.graph.node[-1]
            self.assertEqual(fused.op_type, "SigmoidGateMul")
            self.assertEqual(list(fused.input), ["up", "gate"])
            self.assertEqual(
                {a.name: helper.get_attribute_value(a) for a in fused.attribute}, {"silu_gate": 1, "sigmoid_in_fp32": 0}
            )
            self.assertEqual(fuse_silu_gate(path), 0)
            for options in (
                {"external": "sigmoid"},
                {"external": "silu"},
                {"bad_width": True},
                {"dtype": TensorProto.FLOAT16},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_silu_gate(path), 0)
            onnx.save(graph(external="gate"), path)
            self.assertEqual(fuse_silu_gate(path), 1)
            onnx.save(capture_value(graph(), "silu", TensorProto.BFLOAT16), path)
            self.assertEqual(fuse_silu_gate(path), 0)
            standard_only = graph()
            standard_only.ClearField("opset_import")
            standard_only.opset_import.append(helper.make_opsetid("", 21))
            onnx.save(standard_only, path)
            self.assertEqual(fuse_silu_gate(path), 1)
            self.assertIn(
                ("com.microsoft", 1),
                {(opset.domain, opset.version) for opset in onnx.load(path).opset_import},
            )

    def test_shared_expert_residual_fusion_guards_and_rounding_mode(self):
        def graph(*, external=None, different_root=False, bad_gate_width=False):
            def op(kind, inputs, output, domain="", **attrs):
                return helper.make_node(kind, inputs, [output], name=output, domain=domain, **attrs)

            ms = "com.microsoft"
            nodes = [
                op("MatMulBlockQuantizedFp8Weight", ["hidden", "packed_weight", "packed_scale"], "packed", ms),
                helper.make_node("Split", ["packed", "split_sizes"], ["router", "gate", "up"], axis=-1),
                op("Sigmoid", ["gate"], "gate_sigmoid"),
                op("Mul", ["gate", "gate_sigmoid"], "silu"),
                op("Mul", ["silu", "up"], "gated_up"),
                op("MatMulBlockQuantizedFp8Weight", ["gated_up", "down_weight", "down_scale"], "down", ms),
                op("QMoE", ["hidden", "router"], "routed", ms),
                op("MatMul", ["other" if different_root else "hidden", "gate_weight"], "gate_logit"),
                op("Sigmoid", ["gate_logit"], "scale"),
                op("Mul", ["down", "scale"], "contribution"),
                op("Add", ["routed", "contribution"], "output"),
            ]
            tensors = [
                helper.make_tensor("packed_weight", TensorProto.FLOAT8E4M3FN, [12, 4], [0] * 48),
                helper.make_tensor("down_weight", TensorProto.FLOAT8E4M3FN, [4, 4], [0] * 16),
                helper.make_tensor(
                    "gate_weight",
                    TensorProto.BFLOAT16,
                    [4, 2 if bad_gate_width else 1],
                    [0] * (8 if bad_gate_width else 4),
                ),
                numpy_helper.from_array(np.ones((12, 1), dtype=np.float32), "packed_scale"),
                numpy_helper.from_array(np.ones((4, 1), dtype=np.float32), "down_scale"),
                numpy_helper.from_array(np.array([4, 4, 4], dtype=np.int64), "split_sizes"),
            ]
            inputs = [
                helper.make_tensor_value_info(n, TensorProto.BFLOAT16, ["batch", "sequence", 4])
                for n in ("hidden", "other")
            ]
            outputs = [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, ["batch", "sequence", 4])]
            if external:
                nodes.append(op("Identity", [external], "external"))
                outputs.append(helper.make_tensor_value_info("external", TensorProto.BFLOAT16, None))
            return helper.make_model(
                helper.make_graph(nodes, "shared_expert", inputs, outputs, tensors),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(ms, 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_shared_expert_residual(path), 1)
            self.assertEqual(source.read_bytes(), original)
            optimized = onnx.load(path)
            onnx.checker.check_model(optimized)
            fused = next(n for n in optimized.graph.node if n.op_type == "SigmoidGateMul")
            self.assertEqual(list(fused.input), ["down", "gate_logit", "routed"])
            self.assertEqual(list(fused.output), ["output"])
            self.assertEqual(
                {a.name: helper.get_attribute_value(a) for a in fused.attribute},
                {"broadcast_gate": 1, "sigmoid_in_fp32": 0},
            )
            self.assertIn("gate_sigmoid", {name for n in optimized.graph.node for name in n.output})
            self.assertEqual(fuse_shared_expert_residual(path), 0)
            for options in (
                {"external": "scale"},
                {"external": "contribution"},
                {"different_root": True},
                {"bad_gate_width": True},
            ):
                with self.subTest(options=options):
                    onnx.save(graph(**options), path)
                    self.assertEqual(fuse_shared_expert_residual(path), 0)
            onnx.save(graph(external="down"), path)
            self.assertEqual(fuse_shared_expert_residual(path), 1)
            onnx.save(capture_value(graph(), "contribution", TensorProto.BFLOAT16), path)
            self.assertEqual(fuse_shared_expert_residual(path), 0)

    def test_attention_rotary_fusion_rounding_broadcast_and_guards(self):
        def graph(*, external=False, malformed=False, shared=False, captured=False, folded=False):
            initializers, nodes = [], []
            half = 32 if folded else 4
            width = 4 * half

            def const(name, value, dtype=np.int64):
                initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
                return name

            def node(kind, inputs, output, **attrs):
                nodes.append(helper.make_node(kind, inputs, [output], name=output, **attrs))
                return output

            def sliced(source, output, start, end):
                return node(
                    "Slice",
                    [
                        source,
                        const(output + "_start", [start]),
                        const(output + "_end", [end]),
                        const(output + "_axis", [3]),
                    ],
                    output,
                )

            node("Reshape", ["packed", const("head_shape", [0, 0, 2, width * 2])], "heads")
            sliced("heads", "selected", 0, width)
            node("Cast", ["selected"], "raw", to=TensorProto.FLOAT)
            node("Mul", ["raw", "raw"], "square")
            node("ReduceMean", ["square", const("norm_axis", [-1])], "mean", keepdims=1)
            node("Add", ["mean", const("epsilon", 1e-6, np.float32)], "mean_eps")
            node("Sqrt", ["mean_eps"], "rms")
            node("Div", ["raw", "rms"], "scaled")
            node("Mul", ["scaled", const("gamma", np.linspace(0.5, 1.5, width), np.float32)], "norm")
            node("Cast", ["norm"], "rounded", to=TensorProto.BFLOAT16)
            node("Cast", ["rounded"], "float_norm", to=TensorProto.FLOAT)
            if folded:
                nodes.append(
                    helper.make_node(
                        "FoldMid2",
                        ["opaque_input"] * 10,
                        ["ids", "length", "slots", "seqlens", "cos_bf16", "sin_bf16", "sel"],
                        domain="com.microsoft",
                    )
                )
                node("Cast", ["cos_bf16"], "cos", to=TensorProto.FLOAT)
                node("Cast", ["sin_bf16"], "sin", to=TensorProto.FLOAT)
            node("Unsqueeze", ["cos", const("head_axis", [2])], "cos_heads")
            node("Unsqueeze", ["sin", "head_axis"], "sin_heads")
            sliced("float_norm", "a", 0, half)
            sliced("float_norm", "b", half, 2 * half)
            sliced("float_norm", "tail", 2 * half, width)
            node("Mul", ["a", "cos_heads"], "ac")
            node("Mul", ["b", "sin_heads"], "bs")
            node("Mul", ["b", "cos_heads"], "bc")
            node("Mul", ["b" if malformed else "a", "sin_heads"], "as")
            node("Sub", ["ac", "bs"], "first")
            node("Add", ["bc", "as"], "second")
            node("Concat", ["first", "second", "tail"], "rotated", axis=3)
            node("Cast", ["rotated"], "output", to=TensorProto.BFLOAT16)
            inputs = [
                helper.make_tensor_value_info("packed", TensorProto.FLOAT, ["batch", "sequence", width * 4]),
            ]
            if folded:
                inputs.append(helper.make_tensor_value_info("opaque_input", TensorProto.INT64, None))
            else:
                inputs.extend(
                    [
                        helper.make_tensor_value_info(
                            "cos", TensorProto.FLOAT, ["cache_batch", "cache_sequence", half]
                        ),
                        helper.make_tensor_value_info(
                            "sin", TensorProto.FLOAT, ["cache_batch", "cache_sequence", half]
                        ),
                    ]
                )
            outputs = [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, ["batch", "sequence", 2, width])]
            if shared:
                node("Identity", ["cos_heads"], "external_cache")
                outputs.append(helper.make_tensor_value_info("external_cache", TensorProto.FLOAT, None))
            if captured:
                branch = helper.make_graph(
                    [helper.make_node("Identity", ["a"], ["captured_a"])],
                    "branch",
                    [],
                    [helper.make_tensor_value_info("captured_a", TensorProto.FLOAT, None)],
                )
                const("condition", True, np.bool_)
                nodes.append(
                    helper.make_node("If", ["condition"], ["captured_result"], then_branch=branch, else_branch=branch)
                )
            if external:
                node("Identity", ["a"], "external")
                outputs.append(helper.make_tensor_value_info("external", TensorProto.FLOAT, None))
            return helper.make_model(
                helper.make_graph(nodes, "rotary", inputs, outputs, initializers),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            original = graph()
            onnx.save(original, source)
            source_bytes = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_attention_rotary(path), 1)
            self.assertEqual(source.read_bytes(), source_bytes)
            optimized = onnx.load(path)
            onnx.checker.check_model(optimized)
            self.assertEqual(fuse_attention_rotary(path), 0)
            self.assertEqual(sum(n.op_type == "RotaryEmbedding" for n in optimized.graph.node), 1)
            for name in ("square", "mean", "mean_eps", "rms", "scaled", "norm", "rounded", "float_norm", "output"):
                self.assertEqual(
                    next(n for n in original.graph.node if name in n.output),
                    next(n for n in optimized.graph.node if name in n.output),
                )
            reference = ReferenceEvaluator(original)
            fused = ReferenceEvaluator(optimized, new_ops=[RotaryEmbedding])
            rng = np.random.default_rng(13)
            for batch, length, broadcast_batch, broadcast_sequence in (
                (1, 1, False, False),
                (1, 9, False, False),
                (2, 9, False, False),
                (2, 9, True, False),
                (2, 9, True, True),
            ):
                with self.subTest(batch=batch, length=length, broadcast=(broadcast_batch, broadcast_sequence)):
                    shape = (1 if broadcast_batch else batch, 1 if broadcast_sequence else length, 4)
                    feeds = {
                        "packed": rng.normal(size=(batch, length, 64)).astype(np.float32),
                        "cos": rng.normal(size=shape).astype(np.float32),
                        "sin": rng.normal(size=shape).astype(np.float32),
                    }
                    np.testing.assert_array_equal(reference.run(None, feeds)[0], fused.run(None, feeds)[0])
            for options in ({"external": True}, {"malformed": True}, {"captured": True}):
                onnx.save(graph(**options), path)
                self.assertEqual(fuse_attention_rotary(path), 0)
            shared = graph(shared=True)
            onnx.save(shared, path)
            self.assertEqual(fuse_attention_rotary(path), 1)
            model = onnx.load(path)
            self.assertTrue(any("cos_heads" in node.output for node in model.graph.node))
            for expected, actual in zip(
                ReferenceEvaluator(shared).run(None, feeds),
                ReferenceEvaluator(model, new_ops=[RotaryEmbedding]).run(None, feeds),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            onnx.save(graph(folded=True), path)
            self.assertEqual(fuse_attention_rotary(path), 1)
            model = onnx.load(path)
            self.assertEqual(fuse_attention_rotary(path), 0)
            self.assertFalse(any(info.name == "cos_bf16" for info in model.graph.value_info))
            types = infer_value_types(model)
            self.assertEqual(types["cos_bf16"], TensorProto.BFLOAT16)
            self.assertEqual(types["sin_bf16"], TensorProto.BFLOAT16)
            for dtype, shape in ((TensorProto.FLOAT, [1, 2, 32]), (TensorProto.BFLOAT16, [1, 3, 32])):
                model = graph(folded=True)
                model.graph.value_info.append(helper.make_tensor_value_info("cos_bf16", dtype, shape))
                onnx.save(model, path)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "FoldMid2 rotary outputs require"):
                    fuse_attention_rotary(path)
                self.assertEqual(path.read_bytes(), before)

            model = graph()
            model.ClearField("opset_import")
            model.opset_import.append(helper.make_opsetid("", 21))
            first_concat = next(node for node in model.graph.node if node.op_type == "Concat")
            concat_index = list(model.graph.node).index(first_concat)
            collision = f"Concat_{concat_index}_Rotary_shape"
            model.graph.initializer.append(numpy_helper.from_array(np.array([1], np.int64), collision))
            second = copy.deepcopy(model.graph)
            remapped = {name: name + "_second" for node in second.node for name in node.output}
            remapped.update({tensor.name: tensor.name + "_second" for tensor in second.initializer})
            for node in second.node:
                node.input[:] = [remapped.get(name, name) for name in node.input]
                node.output[:] = [remapped[name] for name in node.output]
                if node.name:
                    node.name += "_second"
            for tensor in second.initializer:
                tensor.name = remapped[tensor.name]
            second_output = copy.deepcopy(second.output[0])
            second_output.name = remapped[second_output.name]
            for node in model.graph.node:
                if node.op_type == "Concat":
                    node.name = ""
            for node in second.node:
                if node.op_type == "Concat":
                    node.name = ""
            model.graph.node.extend(second.node)
            model.graph.initializer.extend(second.initializer)
            model.graph.output.append(second_output)
            onnx.save(model, path)
            self.assertEqual(fuse_attention_rotary(path), 2)
            optimized = onnx.load(path)
            rotary_nodes = [node for node in optimized.graph.node if node.op_type == "RotaryEmbedding"]
            self.assertEqual(len(rotary_nodes), 2)
            self.assertEqual(len({node.name for node in rotary_nodes}), 2)
            all_values = [name for node in optimized.graph.node for name in node.output]
            all_values.extend(tensor.name for tensor in optimized.graph.initializer)
            self.assertEqual(len(all_values), len(set(all_values)))
            self.assertIn(
                ("com.microsoft", 1),
                {(opset.domain, opset.version) for opset in optimized.opset_import},
            )

    def test_conv_projection_fusion_guards_and_shared_source(self):
        def graph(extra_output=None, malformed=False, captured=False):
            constants = []
            nodes = []

            def slice_node(source, output, start, end):
                names = [output + suffix for suffix in ("_start", "_end", "_axes")]
                constants.extend(
                    numpy_helper.from_array(np.array([v], dtype=np.int64), name)
                    for name, v in zip(names, (start, end, 2), strict=False)
                )
                nodes.append(helper.make_node("Slice", [source, *names], [output], name=output))

            slice_node("packed", "selected", 2, 9)
            nodes.append(helper.make_node("Transpose", ["selected"], ["cf"], perm=[0, 2, 1]))
            slice_node("past", "tail", -3, 2**30)
            nodes.append(
                helper.make_node(
                    "CausalConvWithState",
                    ["cf", "weight", "", "tail"],
                    ["conv_cf", "unused_carry"],
                    name="conv",
                    domain="com.microsoft",
                    activation="silu",
                    channels_last=0,
                )
            )
            nodes.append(helper.make_node("Transpose", ["conv_cf"], ["output"], perm=[0, 2, 1]))
            nodes.append(helper.make_node("Concat", ["past", "cf"], ["full"], axis=2))
            slice_node("full", "present", -4, 2**30)
            slice_node("cf", "first", 0, 2 if malformed else 1)
            nodes.append(helper.make_node("Concat", ["past", "first"], ["mid"], axis=2))
            slice_node("mid", "present_mid", -4, 2**30)
            constants.append(numpy_helper.from_array(np.zeros((7, 1, 4), dtype=np.float32), "weight"))
            inputs = [
                helper.make_tensor_value_info("packed", TensorProto.FLOAT, [2, 9, 13]),
                helper.make_tensor_value_info("past", TensorProto.FLOAT, [2, 7, 4]),
            ]
            outputs = [
                helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in ("output", "present", "present_mid")
            ]
            if extra_output:
                nodes.append(helper.make_node("Identity", [extra_output], ["external"]))
                outputs.append(helper.make_tensor_value_info("external", TensorProto.FLOAT, None))
            model = helper.make_model(
                helper.make_graph(nodes, "conv", inputs, outputs, constants),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )
            return capture_value(model, "cf") if captured else model

        with tempfile.TemporaryDirectory() as temporary:
            source, path = Path(temporary) / "source.onnx", Path(temporary) / "model.onnx"
            onnx.save(graph(), source)
            original = source.read_bytes()
            os.link(source, path)
            self.assertEqual(fuse_causal_conv_projection(path), 1)
            self.assertEqual(source.read_bytes(), original)
            optimized = onnx.load(path)
            self.assertEqual(len(optimized.graph.node), 1)
            node = optimized.graph.node[0]
            self.assertEqual(list(node.input), ["packed", "weight", "", "past"])
            self.assertEqual(list(node.output), ["output", "present", "present_mid"])
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            self.assertEqual(
                (
                    attrs["input_channels"],
                    attrs["input_channel_offset"],
                    attrs["state_length"],
                    attrs["state_channels_last"],
                    attrs["channels_last"],
                ),
                (7, 2, 4, 0, 1),
            )
            self.assertEqual(fuse_causal_conv_projection(path), 0)
            for name in ("selected", "cf", "tail", "conv_cf", "unused_carry", "full", "first", "mid"):
                with self.subTest(external_user=name):
                    onnx.save(graph(extra_output=name), path)
                    self.assertEqual(fuse_causal_conv_projection(path), 0)
            onnx.save(graph(captured=True), path)
            self.assertEqual(fuse_causal_conv_projection(path), 0)
            onnx.save(graph(malformed=True), path)
            self.assertEqual(fuse_causal_conv_projection(path), 0)
            for name in ("present", "present_mid"):
                for captured in (False, True):
                    with self.subTest(early_output=name, captured=captured):
                        model = graph()
                        if captured:
                            capture_value(model, name)
                        else:
                            model.graph.node.append(helper.make_node("Identity", [name], ["early"]))
                        nodes = list(model.graph.node)
                        del model.graph.node[:]
                        model.graph.node.extend(nodes[:2] + nodes[5:] + nodes[2:5])
                        onnx.save(model, path)
                        before = path.read_bytes()
                        self.assertEqual(fuse_causal_conv_projection(path), 0)
                        self.assertEqual(path.read_bytes(), before)

    def test_postmix_preserves_residual_users_and_graph_output(self):
        for extra_mix in (False, True):
            with self.subTest(extra_mix=extra_mix), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.onnx"
                post_inputs = ["streams", "previous", "previous_gate"] + (["mix_matrix"] if extra_mix else [])
                nodes = [
                    helper.make_node(
                        "HyperConnectionPostMix",
                        post_inputs,
                        ["mixed"],
                        name="postmix",
                        domain="com.microsoft",
                        num_branches=4,
                    ),
                    helper.make_node(
                        "HyperConnectionLowRankMix",
                        ["mixed", "norm", "down", "up", "inject"],
                        ["pre", "gate"],
                        name="hc",
                        domain="com.microsoft",
                        num_branches=4,
                    ),
                    helper.make_node("Identity", ["mixed"], ["residual"], name="residual"),
                ]
                inputs = [
                    helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("streams", "previous", "previous_gate", "norm", "down", "up", "inject", "mix_matrix")
                ]
                outputs = [
                    helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("pre", "gate", "mixed", "residual")
                ]
                onnx.save(helper.make_model(helper.make_graph(nodes, "postmix", inputs, outputs)), path)
                self.assertEqual(fuse_hyper_connection_postmix(path), 0 if extra_mix else 1)
                optimized = onnx.load(path)
                if not extra_mix:
                    self.assertEqual(
                        [n.op_type for n in optimized.graph.node], ["HyperConnectionLowRankMix", "Identity"]
                    )
                    fused = optimized.graph.node[0]
                    self.assertEqual(
                        list(fused.input), ["streams", "norm", "down", "up", "inject", "previous", "previous_gate"]
                    )
                    self.assertEqual(list(fused.output), ["pre", "gate", "mixed"])
                    self.assertEqual(fuse_hyper_connection_postmix(path), 0)

        for early_consumer, has_streams_out in ((True, False), (False, True)):
            with (
                self.subTest(early_consumer=early_consumer, has_streams_out=has_streams_out),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "model.onnx"
                nodes = [
                    helper.make_node(
                        "HyperConnectionPostMix",
                        ["streams", "previous", "previous_gate"],
                        ["mixed"],
                        domain="com.microsoft",
                        num_branches=4,
                    )
                ]
                if early_consumer:
                    nodes.append(helper.make_node("Identity", ["mixed"], ["residual"]))
                nodes.append(
                    helper.make_node(
                        "HyperConnectionLowRankMix",
                        ["mixed", "norm", "down", "up", "inject"],
                        ["pre", "gate", "streams_out"] if has_streams_out else ["pre", "gate"],
                        domain="com.microsoft",
                        num_branches=4,
                    )
                )
                outputs = [
                    helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in (
                        "pre",
                        "gate",
                        *(("residual",) if early_consumer else ()),
                        *(("streams_out",) if has_streams_out else ()),
                    )
                ]
                model = helper.make_model(
                    helper.make_graph(
                        nodes,
                        "postmix_safety",
                        [
                            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                            for name in (
                                "streams",
                                "previous",
                                "previous_gate",
                                "norm",
                                "down",
                                "up",
                                "inject",
                            )
                        ],
                        outputs,
                    ),
                    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                )
                for node in model.graph.node:
                    node.name = ""
                onnx.save(model, path)
                self.assertEqual(fuse_hyper_connection_postmix(path), 0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captured.onnx"
            nodes = [
                helper.make_node(
                    "HyperConnectionPostMix",
                    ["streams", "previous", "previous_gate"],
                    ["mixed"],
                    domain="com.microsoft",
                    num_branches=4,
                ),
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    ["mixed", "norm", "down", "up", "inject"],
                    ["pre", "gate"],
                    domain="com.microsoft",
                    num_branches=4,
                ),
            ]
            inputs = [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                for name in ("streams", "previous", "previous_gate", "norm", "down", "up", "inject")
            ]
            model = helper.make_model(
                helper.make_graph(
                    nodes,
                    "postmix_capture",
                    inputs,
                    [
                        helper.make_tensor_value_info("pre", TensorProto.FLOAT, None),
                        helper.make_tensor_value_info("gate", TensorProto.FLOAT, None),
                    ],
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )
            capture_value(model, "mixed")
            model.graph.node.insert(1, model.graph.node.pop())
            onnx.save(model, path)
            self.assertEqual(fuse_hyper_connection_postmix(path), 0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unnamed.onnx"
            nodes = [
                helper.make_node(
                    "HyperConnectionPostMix",
                    ["streams", "previous", "previous_gate"],
                    ["mixed"],
                    domain="com.microsoft",
                    num_branches=4,
                ),
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    ["mixed", "norm", "down", "up", "inject"],
                    ["pre", "gate"],
                    domain="com.microsoft",
                    num_branches=4,
                ),
                helper.make_node("Identity", ["mixed"], ["residual"]),
            ]
            for node in nodes:
                node.name = ""
            model = helper.make_model(
                helper.make_graph(
                    nodes,
                    "postmix_unnamed",
                    [
                        helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                        for name in (
                            "streams",
                            "previous",
                            "previous_gate",
                            "norm",
                            "down",
                            "up",
                            "inject",
                        )
                    ],
                    [
                        helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                        for name in ("pre", "gate", "residual")
                    ],
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            )
            onnx.save(model, path)
            self.assertEqual(fuse_hyper_connection_postmix(path), 1)
            optimized = onnx.load(path).graph.node
            self.assertEqual([node.op_type for node in optimized], ["HyperConnectionLowRankMix", "Identity"])
            self.assertEqual(list(optimized[0].output), ["pre", "gate", "mixed"])

    def test_hyper_connection_reuses_fused_operator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            specs = [
                ("BranchwiseRMSNorm", ["input", "norm_weight"], ["norm"], {"epsilon": 1e-6, "num_branches": 4}),
                ("MatMul", ["norm", "input_mix_weight_down"], ["down"], {}),
                ("ScaledSiLU", ["down"], ["silu"], {"alpha": 0.25}),
                ("MatMul", ["silu", "up_weight"], ["up"], {}),
                ("Sigmoid", ["up"], ["sigmoid"], {}),
                (
                    "HyperConnectionPreMix",
                    ["norm", "sigmoid"],
                    ["output"],
                    {"num_branches": 4, "reduction_scale": 0.25},
                ),
                ("MatMul", ["norm", "block_inject_weight"], ["inject"], {}),
                ("Div", ["inject", "divisor"], ["div"], {}),
                ("Sigmoid", ["div"], ["gate"], {}),
                ("Mul", ["gate", "multiplier"], ["post"], {}),
            ]
            nodes = [
                helper.make_node(
                    op,
                    inputs,
                    outputs,
                    name=outputs[0],
                    domain="com.microsoft"
                    if op in ("BranchwiseRMSNorm", "ScaledSiLU", "HyperConnectionPreMix")
                    else "",
                    **attrs,
                )
                for op, inputs, outputs, attrs in specs
            ]
            constants = [
                numpy_helper.from_array(np.ones(shape, np.float32), name)
                for name, shape in (
                    ("norm_weight", [8]),
                    ("input_mix_weight_down", [8, 16]),
                    ("up_weight", [16, 8]),
                    ("block_inject_weight", [8, 4]),
                    ("divisor", []),
                    ("multiplier", []),
                )
            ]
            graph = helper.make_graph(
                nodes,
                "hc",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8])],
                [
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2]),
                    helper.make_tensor_value_info("post", TensorProto.FLOAT, [1, 1, 4]),
                ],
                constants,
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
            )
            for dtype in (np.float32, ml_dtypes.bfloat16):
                variant = copy.deepcopy(model)
                variant.graph.input[0].type.tensor_type.elem_type = (
                    TensorProto.FLOAT if dtype is np.float32 else TensorProto.BFLOAT16
                )
                for tensor in variant.graph.initializer:
                    if tensor.name not in ("divisor", "multiplier"):
                        tensor.CopyFrom(
                            numpy_helper.from_array(numpy_helper.to_array(tensor).astype(dtype), tensor.name)
                        )
                norm = variant.graph.node[0]
                attrs = [attr for attr in norm.attribute if attr.name != "epsilon"]
                del norm.attribute[:]
                norm.attribute.extend(attrs)
                onnx.save(variant, path)
                self.assertEqual(fuse_hyper_connections(path), 1)
                fused = onnx.load(path).graph.node[0]
                epsilon = next(a.f for a in fused.attribute if a.name == "epsilon")
                self.assertAlmostEqual(epsilon, 1e-5)
                for mismatch in ("shared", "dtype", "layout", "unknown", "produced"):
                    with self.subTest(dtype=dtype, mismatch=mismatch):
                        rejected = copy.deepcopy(variant)
                        scale = rejected.graph.initializer[0]
                        if mismatch == "shared":
                            scale.CopyFrom(numpy_helper.from_array(np.ones(2, dtype=dtype), scale.name))
                        elif mismatch == "dtype":
                            other_dtype = ml_dtypes.bfloat16 if dtype is np.float32 else np.float32
                            scale.CopyFrom(numpy_helper.from_array(np.ones(8, dtype=other_dtype), scale.name))
                        elif mismatch == "layout":
                            scale.dims[:] = [4, 2]
                        elif mismatch == "unknown":
                            rejected.graph.input[0].type.tensor_type.shape.dim[-1].ClearField("dim_value")
                        else:
                            weight = rejected.graph.initializer[1]
                            weight.name = "down_source"
                            rejected.graph.node.insert(
                                1, helper.make_node("Identity", ["down_source"], ["input_mix_weight_down"])
                            )
                        onnx.save(rejected, path)
                        self.assertEqual(fuse_hyper_connections(path), 0)
                        self.assertEqual(
                            [node.op_type for node in onnx.load(path).graph.node],
                            [node.op_type for node in rejected.graph.node],
                        )
            onnx.save(model, path)
            self.assertEqual(fuse_hyper_connections(path), 1)
            optimized = onnx.load(path)
            self.assertEqual(len(optimized.graph.node), 1)
            fused = optimized.graph.node[0]
            self.assertEqual(fused.op_type, "HyperConnectionLowRankMix")
            self.assertEqual(list(fused.output), ["output", "post"])
            self.assertEqual(
                list(fused.input), ["input", "norm_weight", "input_mix_weight_down", "up_weight", "block_inject_weight"]
            )
            self.assertEqual(helper.get_attribute_value(next(a for a in fused.attribute if a.name == "gate_div")), 1.0)

            unnamed_path = Path(temporary) / "unnamed.onnx"
            for node in model.graph.node:
                node.name = ""
            model.graph.node.append(helper.make_node("Identity", ["input"], ["unrelated"]))
            onnx.save(model, unnamed_path)
            self.assertEqual(fuse_hyper_connections(unnamed_path), 1)
            unnamed_nodes = onnx.load(unnamed_path).graph.node
            self.assertEqual([node.op_type for node in unnamed_nodes], ["HyperConnectionLowRankMix", "Identity"])
            self.assertEqual(list(unnamed_nodes[0].output), ["output", "post"])

    def test_metadata_does_not_overwrite_hardlinked_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "source", Path(temporary) / "destination"
            source.mkdir()
            destination.mkdir()
            configuration = {
                "model": {
                    "qwen38_graph": {"prefill": {}},
                    "decoder": {"session_options": {"provider_options": [{"cuda": {"enable_cuda_graph": "0"}}]}},
                }
            }
            manifest = {"experimental_graph_rewrites": {}, "unported_operator_findings": {}}
            for name, value in (("genai_config.json", configuration), ("package_manifest.json", manifest)):
                (source / name).write_text(json.dumps(value))
                os.link(source / name, destination / name)
            provider = source / "provider.so"
            provider.write_bytes(b"old provider")
            (destination / "provider").mkdir()
            os.link(provider, destination / "provider/libonnxruntime_providers_cuda.so")
            updated = Path(temporary) / "updated.so"
            updated.write_bytes(b"updated runtime")
            for label in ("target", "dense", "head_r2", "head_kv"):
                (destination / label).mkdir()
                onnx.save(helper.make_model(helper.make_graph([], label, [], [])), destination / label / "model.onnx")
            results = {"target/model.onnx": {"deduplicated_standard_expressions": {"Cast": 2}}}
            write_performance_metadata(destination, source, results, {}, updated, updated, updated)
            self.assertEqual(json.loads((source / "genai_config.json").read_text()), configuration)
            self.assertEqual(json.loads((source / "package_manifest.json").read_text()), manifest)
            self.assertEqual(provider.read_bytes(), b"old provider")
            self.assertEqual(
                (destination / "provider/libonnxruntime_providers_cuda.so").read_bytes(), b"updated runtime"
            )
            configured = json.loads((destination / "genai_config.json").read_text())["model"]
            self.assertTrue(configured["qwen38_graph"]["warmup"])
            self.assertEqual(
                configured["decoder"]["session_options"]["provider_options"][0]["cuda"]["enable_cuda_graph"], "1"
            )
            metadata = json.loads((destination / "package_manifest.json").read_text())
            self.assertEqual(metadata["performance_rewrites"], results)
            self.assertIn("SharedStandardExpressions", metadata["experimental_graph_rewrites"])

    def test_untile_preserves_bits_and_layout(self):
        values = np.arange(3276800, dtype=np.uint32).astype(np.uint16)
        expected = (
            values.reshape(8, 20, 10, 128, 16).transpose(0, 2, 3, 1, 4).reshape(10240, 320),
            values.reshape(160, 10, 32, 4, 16).transpose(1, 2, 3, 0, 4).reshape(320, 10240),
        )
        for kind in (0, 1):
            np.testing.assert_array_equal(untile_weights(values, kind), expected[kind])
        with self.assertRaises(ValueError):
            untile_weights(values, 2)
        with self.assertRaises(ValueError):
            untile_weights(values[:16], 0)

    def test_prefill_retains_float_recurrent_outputs(self):
        for head in (False, True):
            with self.subTest(head=head), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.onnx"
                shape = [1, 2, 3, 256] if not head else [1, 3, 512]
                name = "present.0.key" if not head else "key"
                graph = helper.make_graph(
                    [helper.make_node("Identity", ["input"], [name])],
                    "prefill",
                    [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                    [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
                onnx.save(model, path)
                scales = np.linspace(0.25, 2.0, 512, dtype=np.float32)
                self.assertEqual(quantize_prefill_outputs(path, {name: scales}, head=head), 1)
                optimized = onnx.load(path)
                onnx.checker.check_model(optimized)
                values = np.linspace(-300, 300, np.prod(shape), dtype=np.float32).reshape(shape)
                results = ReferenceEvaluator(optimized).run(None, {"input": values})
                flattened = values if head else values.transpose(0, 2, 1, 3).reshape(1, 3, 512)
                expected = np.clip(np.rint(flattened / scales), -128, 127).astype(np.int8)
                if not head:
                    expected = expected.reshape(1, 3, 2, 256).transpose(0, 2, 1, 3)
                    np.testing.assert_array_equal(results[0], values)
                    self.assertEqual(optimized.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)
                    self.assertEqual(optimized.graph.output[1].name, name + ".quantized")
                np.testing.assert_array_equal(results[-1], expected)

    def test_fp8_weight_transpose_and_scale_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            weights = (np.arange(128).reshape(16, 8) / 16).astype(ml_dtypes.float8_e4m3fn)
            scales = np.linspace(0.25, 2, 8, dtype=np.float32)
            graph = helper.make_graph(
                [
                    helper.make_node("DequantizeLinear", ["weights", "scales"], ["float_weights"], name="dq", axis=1),
                    helper.make_node("Cast", ["float_weights"], ["bf16_weights"], name="cast", to=TensorProto.BFLOAT16),
                    helper.make_node("MatMul", ["input", "bf16_weights"], ["output"], name="matmul"),
                ],
                "fp8",
                [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 16])],
                [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 8])],
                [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(scales, "scales")],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
            )
            onnx.save(model, path)
            self.assertEqual(optimize_model(path), {"fused_fp8_matmuls": 1})
            optimized = onnx.load(path)
            self.assertEqual([n.op_type for n in optimized.graph.node], ["MatMulBlockQuantizedFp8Weight"])
            constants = {t.name: numpy_helper.to_array(t) for t in optimized.graph.initializer}
            np.testing.assert_array_equal(constants["weights.performance_nk"].view(np.uint8), weights.T.view(np.uint8))
            np.testing.assert_array_equal(constants["scales.performance_n1"], scales.reshape(8, 1))

    def test_optimizer_handles_unnamed_fp8_nodes_captures_and_missing_opset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            weights = np.ones((16, 8), dtype=ml_dtypes.float8_e4m3fn)
            graph = helper.make_graph(
                [
                    helper.make_node("DequantizeLinear", ["weights", "scales"], ["float_weights"], axis=1),
                    helper.make_node("Cast", ["float_weights"], ["bf16_weights"], to=TensorProto.BFLOAT16),
                    helper.make_node("MatMul", ["input", "bf16_weights"], ["output"]),
                    helper.make_node("Identity", ["input"], ["unrelated"]),
                ],
                "unnamed_fp8",
                [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 16])],
                [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 8])],
                [
                    numpy_helper.from_array(weights, "weights"),
                    numpy_helper.from_array(np.ones(8, np.float32), "scales"),
                ],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
            capture_value(model, "bf16_weights", TensorProto.BFLOAT16)
            onnx.save(model, path)
            self.assertEqual(optimize_model(path), {"fused_fp8_matmuls": 1})
            optimized = onnx.load(path)
            self.assertEqual(
                [node.op_type for node in optimized.graph.node],
                ["DequantizeLinear", "Cast", "MatMulBlockQuantizedFp8Weight", "Identity", "If"],
            )
            self.assertIn(
                ("com.microsoft", 1),
                {(opset.domain, opset.version) for opset in optimized.opset_import},
            )

    def test_optimizer_and_hyper_connections_keep_branch_only_initializers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            model = helper.make_model(
                helper.make_graph(
                    [],
                    "captured_initializer",
                    [],
                    [],
                    [numpy_helper.from_array(np.array([3.0], np.float32), "branch_only")],
                ),
                opset_imports=[helper.make_opsetid("", 21)],
            )
            capture_value(model, "branch_only")
            onnx.save(model, path)
            self.assertEqual(optimize_model(path), {})
            self.assertIn("branch_only", {tensor.name for tensor in onnx.load(path).graph.initializer})
            self.assertEqual(fuse_hyper_connections(path), 0)
            self.assertIn("branch_only", {tensor.name for tensor in onnx.load(path).graph.initializer})

    def test_pool_rebuild_shares_equivalent_transformed_weights_by_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "source", Path(temporary) / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "weights_pool.txt").write_text(
                "qwen38_weights_pool 1\n"
                "pool 0 17 0 128 2 16 8 target/weights.bin\n"
                "pool 1 1 128 32 1 8 target/scales.bin\n"
                "session target weights 0\nsession dense weights 0\n"
                "session target scales 1\nsession dense scales 1\n"
            )
            provenance = {}
            for label in ("target", "dense"):
                folder = destination / label
                folder.mkdir()
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node("DequantizeLinear", ["weights", "scales"], ["float_weights"], axis=1),
                            helper.make_node("Cast", ["float_weights"], ["bf16_weights"], to=TensorProto.BFLOAT16),
                            helper.make_node("MatMul", ["input", "bf16_weights"], ["output"]),
                        ],
                        label,
                        [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 16])],
                        [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 8])],
                        [
                            numpy_helper.from_array(np.ones((16, 8), dtype=ml_dtypes.float8_e4m3fn), "weights"),
                            numpy_helper.from_array(np.ones(8, np.float32), "scales"),
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                )
                onnx.save(model, folder / "model.onnx")
                provenance[label] = {}
                self.assertEqual(
                    optimize_model(folder / "model.onnx", provenance[label]),
                    {"fused_fp8_matmuls": 1},
                )
            for label in ("head_r2", "head_kv"):
                folder = destination / label
                folder.mkdir()
                onnx.save(helper.make_model(helper.make_graph([], label, [], [])), folder / "model.onnx")

            result = refresh_weight_pool(source, destination, provenance)
            manifest = (destination / "weights_pool.txt").read_text()
            self.assertEqual(result["initializers"], 2)
            self.assertIn("session target weights.performance_nk", manifest)
            self.assertIn("session dense weights.performance_nk", manifest)
            self.assertIn("session target scales.performance_n1", manifest)
            self.assertIn("session dense scales.performance_n1", manifest)

            dense_weight = provenance["dense"]["weights.performance_nk"]
            provenance["dense"]["weights.performance_nk"] = (
                dense_weight[0],
                ("different_transform", *dense_weight[1][1:]),
            )
            refresh_weight_pool(source, destination, provenance)
            manifest = (destination / "weights_pool.txt").read_text()
            self.assertNotIn("session target weights.performance_nk", manifest)
            self.assertNotIn("session dense weights.performance_nk", manifest)

    def test_pool_uses_declared_sharing_not_inode_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "source", Path(temporary) / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "weights_pool.txt").write_text(
                "qwen38_weights_pool 1\n"
                "pool 0 1 0 4 1 1 target/weights.bin\n"
                "pool 1 1 4 4 1 1 target/weights.bin\n"
                "session target weight 0\nsession dense weight 0\n"
                "session target removed 1\nsession dense removed 1\n"
            )
            for label in ("target", "dense", "head_r2", "head_kv"):
                folder = destination / label
                folder.mkdir()
                model = helper.make_model(
                    helper.make_graph([], label, [], [], [numpy_helper.from_array(np.ones(1, np.float32), "weight")])
                )
                onnx.save(model, folder / "model.onnx")
            result = refresh_weight_pool(source, destination)
            self.assertEqual(result["initializers"], 1)
            manifest = (destination / "weights_pool.txt").read_text()
            self.assertIn("session target weight 0", manifest)
            self.assertIn("session dense weight 0", manifest)
            self.assertNotIn("removed", manifest)
            self.assertEqual(result["gib"], 4 / 2**30)


if __name__ == "__main__":
    unittest.main()
