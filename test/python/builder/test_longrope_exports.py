# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from builder_test_utils import load_builder_module


def export_caches(tmp_path, ep="cpu", graph_capture=False, paged=False):
    model = load_builder_module("base").Model(graph_name="longrope_caches")
    model.io_dtype = ir.DataType.FLOAT
    model.context_length = 32
    model.original_context_length = 8
    model.head_size = 8
    model.ep = ep
    model.ep_attrs = {"webgpu": {}, "trt-rtx": {}}
    model.extra_options = {"enable_webgpu_graph": graph_capture}
    model.use_paged_attention = paged
    model.attention_attrs = {"op_type": "PagedAttention" if paged else "GroupQueryAttention", "paged_block_size": 1}
    input_name = "block_table" if paged else "input_ids"
    model.input_names = {input_name: input_name}
    model.input_types = {input_name: ir.DataType.INT32 if paged else ir.DataType.INT64}
    model.input_shapes = {input_name: [1, "sequence_length"]}
    model.output_names = {"cos_cache": "cos_cache", "sin_cache": "sin_cache"}
    model.output_types = dict.fromkeys(model.output_names, ir.DataType.FLOAT)
    model.output_shapes = {name: ["cache_length", 4] for name in model.output_names}
    model.rope_attrs = {
        "partial_rotary_factor": 1.0,
        "theta": 10000.0,
        "t_dtype": torch.float32,
        "position_scale": 1,
        "cast": {"use_fp32": False},
        "create_caches": True,
        "multi_cache": {
            "long_factor": torch.full((4,), 2.0),
            "short_factor": torch.ones(4),
            "long_mscale": 2.0,
            "short_mscale": 1.0,
        },
    }
    model.make_rotary_embedding_multi_cache()
    model.make_inputs_and_outputs()
    path = tmp_path / "model.onnx"
    model.save_onnx_model(model.model, path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def expected_caches(long):
    positions = np.arange(32 if long else 8, dtype=np.float32)
    frequencies = 10000.0 ** (-np.arange(0, 8, 2, dtype=np.float32) / 8)
    angles = np.outer(positions, frequencies / (2 if long else 1))
    scale = 2 if long else 1
    return np.cos(angles) * scale, np.sin(angles) * scale


def test_longrope_scalar_and_metadata_survive_export(tmp_path):
    session = export_caches(tmp_path)
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["com.microsoft.genai.rope_context_length_input"] == "rope_context_length"
    assert metadata["com.microsoft.genai.original_context_length"] == "8"
    scalar = next(value for value in session.get_inputs() if value.name == "rope_context_length")
    assert scalar.type == "tensor(int64)"
    assert scalar.shape == []

    for context_length in (7, 8, 9):
        outputs = session.run(
            None,
            {
                "input_ids": np.ones((1, context_length), dtype=np.int64),
                "rope_context_length": np.asarray(context_length, dtype=np.int64),
            },
        )
        for actual, expected in zip(outputs, expected_caches(context_length > 8), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("ep", "graph_capture"), [("dml", False), ("trt-rtx", False), ("webgpu", True)])
def test_concatenated_longrope_exports_keep_existing_interface(tmp_path, ep, graph_capture):
    session = export_caches(tmp_path, ep=ep, graph_capture=graph_capture)
    assert [value.name for value in session.get_inputs()] == ["input_ids"]
    assert "com.microsoft.genai.rope_context_length_input" not in session.get_modelmeta().custom_metadata_map
    outputs = session.run(None, {"input_ids": np.ones((1, 9), dtype=np.int64)})
    for actual, short, long in zip(outputs, expected_caches(False), expected_caches(True), strict=True):
        np.testing.assert_allclose(actual, np.concatenate((short, long)), rtol=1e-5, atol=1e-6)


def test_paged_longrope_export_keeps_block_table_selection(tmp_path):
    session = export_caches(tmp_path, paged=True)
    assert [value.name for value in session.get_inputs()] == ["block_table"]
    assert "com.microsoft.genai.rope_context_length_input" not in session.get_modelmeta().custom_metadata_map
    for blocks in (8, 9):
        outputs = session.run(None, {"block_table": np.zeros((1, blocks), dtype=np.int32)})
        for actual, expected in zip(outputs, expected_caches(blocks > 8), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
