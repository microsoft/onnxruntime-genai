# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

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


def _write_source_model(tmp_path: Path, head_size: int, layer_ids=(3,)) -> tuple[Path, Path]:
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
    template = copy.deepcopy(graph)
    del graph.node[:]
    del graph.input[:]
    del graph.output[:]
    del graph.initializer[:]
    for layer_id in layer_ids:
        layer = copy.deepcopy(template)
        for layer_node in layer.node:
            layer_node.name = layer_node.name.replace("layers.3", f"layers.{layer_id}")
            for names in (layer_node.input, layer_node.output):
                for index, name in enumerate(names):
                    names[index] = name.replace(".3.", f".{layer_id}.")
        for values in (layer.input, layer.output, layer.initializer):
            for value in values:
                value.name = value.name.replace(".3.", f".{layer_id}.")
        graph.node.extend(layer.node)
        graph.input.extend(layer.input)
        graph.output.extend(layer.output)
        graph.initializer.extend(layer.initializer)
    source_path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), source_path)
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [[0.1] * head_size for layer_id in layer_ids],
                    "v_scales": [[0.2] * head_size for layer_id in layer_ids],
                },
                "layer_ids": list(layer_ids),
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
    weights = np.arange(12, dtype=np.float32)
    (tmp_path / "model.onnx.data").write_bytes(weights.tobytes())
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
    loaded_variant = onnx.load(output_path, load_external_data=True)
    loaded_weights = next(tensor for tensor in loaded_variant.graph.initializer if tensor.name == weight.name)
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(loaded_weights), weights[8:])


def test_create_kv_cache_variant_rejects_different_output_directory(tmp_path):
    source_path, scale_path = _write_source_model(tmp_path, head_size=4)
    output_path = tmp_path / "variant" / "model.onnx"

    with pytest.raises(ValueError, match="must share a directory"):
        KVCacheVariant("int4_per_channel").create(source_path, output_path, scale_path)

    assert not output_path.exists()


def test_create_kv_cache_variant_rejects_overwriting_source(tmp_path):
    source_path, scale_path = _write_source_model(tmp_path, head_size=4)
    source_bytes = source_path.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite the source graph"):
        KVCacheVariant("int4_per_channel").create(source_path, source_path, scale_path)

    assert source_path.read_bytes() == source_bytes


def test_create_kv_cache_variant_maps_scales_by_layer_id(tmp_path):
    source_path, scale_path = _write_source_model(tmp_path, head_size=4, layer_ids=(3, 7))
    scale_data = json.loads(scale_path.read_text())
    scale_data["layer_ids"] = [7, 3]
    scale_data["scales"] = {
        "k_scales": [[0.7] * 4, [0.3] * 4],
        "v_scales": [[1.7] * 4, [1.3] * 4],
    }
    scale_path.write_text(json.dumps(scale_data))
    output_path = tmp_path / "variant.onnx"

    KVCacheVariant("int8_per_channel").create(source_path, output_path, scale_path)

    variant = onnx.load(output_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in variant.graph.initializer}
    for layer_id in (3, 7):
        for kind, offset in (("k", 0), ("v", 1)):
            scale = initializers[f"model.layers.{layer_id}.attn.{kind}_scale"]
            np.testing.assert_allclose(onnx.numpy_helper.to_array(scale), layer_id / 10 + offset)


@pytest.mark.parametrize("layer_ids", [[3, 3], [3, 8], [3], [3, 7, 8]])
def test_create_kv_cache_variant_rejects_mismatched_scale_layers(tmp_path, layer_ids):
    source_path, scale_path = _write_source_model(tmp_path, head_size=4, layer_ids=(3, 7))
    scale_data = json.loads(scale_path.read_text())
    scale_data["layer_ids"] = layer_ids
    scale_path.write_text(json.dumps(scale_data))

    with pytest.raises(ValueError, match="Scale layer_ids must match"):
        KVCacheVariant("int8_per_channel").create(source_path, tmp_path / "variant.onnx", scale_path)


@pytest.mark.parametrize("kind", ["k", "v"])
@pytest.mark.parametrize("shape", [[4], [1, 2, 2], [1, 1, 3], [2, 1, 2], [0, 1, 4]])
def test_create_kv_cache_variant_rejects_inconsistent_later_scale_shape(tmp_path, kind, shape):
    source_path, scale_path = _write_source_model(tmp_path, head_size=4, layer_ids=(3, 7))
    model = onnx.load(source_path, load_external_data=False)
    scale = next(tensor for tensor in model.graph.initializer if tensor.name == f"model.layers.7.attn.{kind}_scale")
    del scale.dims[:]
    scale.dims.extend(shape)
    onnx.save(model, source_path)
    output_path = tmp_path / "variant.onnx"

    with pytest.raises(ValueError, match=f"KV-cache scale .*layers.7.attn.{kind}_scale.*shape"):
        KVCacheVariant("int4_per_channel").create(source_path, output_path, scale_path)

    assert not output_path.exists()
