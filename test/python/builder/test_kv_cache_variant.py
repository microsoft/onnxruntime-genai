# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import sys
from pathlib import Path

import numpy as np
import onnx

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
sys.path.insert(0, str(MODELS_DIR))

from kv_cache_variant import KVCacheVariant  # noqa: E402


def _external_initializer(name: str, shape: list[int], offset: int, length: int) -> onnx.TensorProto:
    initializer = onnx.numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name)
    onnx.external_data_helper.set_external_data(
        initializer,
        location="model.onnx.data",
        offset=offset,
        length=length,
    )
    initializer.ClearField("raw_data")
    initializer.data_location = onnx.TensorProto.EXTERNAL
    return initializer


def _cache_value(name: str, elem_type: int, width: int) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(name, elem_type, ["num_blocks", "block_size", 1, width])


def _write_source_model(tmp_path: Path, head_size: int) -> tuple[Path, Path]:
    scale_bytes = head_size * 4
    k_scale = _external_initializer("model.layers.3.attn.k_scale", [1, 1, head_size], 0, scale_bytes)
    v_scale = _external_initializer("model.layers.3.attn.v_scale", [1, 1, head_size], scale_bytes, scale_bytes)
    weight = _external_initializer("model.layers.3.weight", [4], 32, 16)
    node = onnx.helper.make_node(
        "PagedAttention",
        ["past_key_values.3.key", "past_key_values.3.value", k_scale.name, v_scale.name],
        ["present.3.key", "present.3.value"],
        name="/model/layers.3/attn/PagedAttention",
        domain="com.microsoft",
        num_heads=1,
        kv_num_heads=1,
        k_quant_type="PER_CHANNEL",
        v_quant_type="PER_CHANNEL",
    )
    graph = onnx.helper.make_graph(
        [node],
        "test",
        [
            _cache_value("past_key_values.3.key", onnx.TensorProto.INT8, head_size),
            _cache_value("past_key_values.3.value", onnx.TensorProto.INT8, head_size),
        ],
        [
            _cache_value("present.3.key", onnx.TensorProto.INT8, head_size),
            _cache_value("present.3.value", onnx.TensorProto.INT8, head_size),
        ],
        [k_scale, v_scale, weight],
    )
    source_path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), source_path)
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(
        json.dumps(
            {
                "scales": {"k_scales": [[0.1] * head_size], "v_scales": [[0.2] * head_size]},
                "layer_ids": [3],
                "qmax": 128.0,
            }
        ),
        encoding="utf-8",
    )
    return source_path, scale_path


def test_create_int4_kv_cache_variant_packs_odd_head_size(tmp_path):
    source_path, scale_path = _write_source_model(tmp_path, head_size=3)
    output_path = tmp_path / "model_int4.onnx"

    KVCacheVariant("int4_per_channel").create(source_path, output_path, scale_path)

    variant = onnx.load(output_path, load_external_data=False)
    attributes = {attribute.name: attribute.s for attribute in variant.graph.node[0].attribute}
    assert attributes["k_cache_dtype"] == b"int4"
    assert attributes["v_cache_dtype"] == b"int4"
    for value_info in (*variant.graph.input, *variant.graph.output):
        assert value_info.type.tensor_type.elem_type == onnx.TensorProto.UINT8
        assert value_info.type.tensor_type.shape.dim[-1].dim_value == 2


def test_create_kv_cache_variant_reuses_external_weights_and_inlines_scales(tmp_path):
    k_scale = _external_initializer("model.layers.3.attn.k_scale", [1, 1, 4], 0, 16)
    v_scale = _external_initializer("model.layers.3.attn.v_scale", [1, 1, 4], 16, 16)
    weight = _external_initializer("model.layers.3.weight", [4], 32, 16)
    node = onnx.helper.make_node(
        "PagedAttention",
        ["past_key_values.3.key", "past_key_values.3.value", k_scale.name, v_scale.name],
        ["present.3.key", "present.3.value"],
        name="/model/layers.3/attn/PagedAttention",
        domain="com.microsoft",
        num_heads=1,
        kv_num_heads=1,
        k_quant_type="PER_CHANNEL",
        v_quant_type="PER_CHANNEL",
        k_cache_dtype="int4",
        v_cache_dtype="int4",
    )
    graph = onnx.helper.make_graph(
        [node],
        "test",
        [
            _cache_value("past_key_values.3.key", onnx.TensorProto.UINT8, 2),
            _cache_value("past_key_values.3.value", onnx.TensorProto.UINT8, 2),
        ],
        [
            _cache_value("present.3.key", onnx.TensorProto.UINT8, 2),
            _cache_value("present.3.value", onnx.TensorProto.UINT8, 2),
        ],
        [k_scale, v_scale, weight],
    )
    source_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model_32gib.onnx"
    onnx.save(onnx.helper.make_model(graph), source_path)
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(
        json.dumps(
            {
                "model": {
                    "scales": {
                        "k_scales": [[0.1, 0.2, 0.3, 0.4]],
                        "v_scales": [[0.5, 0.6, 0.7, 0.8]],
                    },
                    "layer_ids": [3],
                    "qmax": 128.0,
                }
            }
        ),
        encoding="utf-8",
    )

    KVCacheVariant("int8_per_channel").create(source_path, output_path, scale_path)

    variant = onnx.load(output_path, load_external_data=False)
    attributes = {attribute.name for attribute in variant.graph.node[0].attribute}
    assert "k_cache_dtype" not in attributes
    assert "v_cache_dtype" not in attributes
    for value_info in (*variant.graph.input, *variant.graph.output):
        assert value_info.type.tensor_type.elem_type == onnx.TensorProto.INT8
        assert value_info.type.tensor_type.shape.dim[-1].dim_value == 4
    initializers = {initializer.name: initializer for initializer in variant.graph.initializer}
    assert initializers[k_scale.name].data_location == onnx.TensorProto.DEFAULT
    assert initializers[v_scale.name].data_location == onnx.TensorProto.DEFAULT
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(initializers[k_scale.name]),
        np.asarray([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32),
    )
    assert initializers[weight.name].data_location == onnx.TensorProto.EXTERNAL
    assert {entry.key: entry.value for entry in initializers[weight.name].external_data} == {
        "location": "model.onnx.data",
        "offset": "32",
        "length": "16",
    }
