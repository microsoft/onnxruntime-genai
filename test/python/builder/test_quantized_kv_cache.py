# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for quantized KV cache support in the model builder.

The quantized KV cache stores the past/present key and value tensors in a
compressed integer/FP8 encoding instead of the model's floating-point I/O dtype.
It is selected via the ``kv_cache_quant_scheme`` extra option and is only valid
with ``GroupQueryAttention``. These tests exercise the builder plumbing
standalone (no model download):

* ``check_extra_options`` validation of ``kv_cache_quant_scheme``.
* ``make_kv_cache_init`` (cache dtype, bit width, quant granularity,
  int4 head-size packing, past/present buffer sharing).
* ``make_kv_cache_scale_initializers`` (per-tensor vs per-channel scale sizes,
  calibrated per-layer scales from a JSON file, and their validation).
* ``make_group_query_attention`` wiring of the per-layer k/v scale inputs and
  the quant attributes onto the GQA node.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(MODELS_DIR))


def _load_base_module():
    models_package = sys.modules.setdefault("models", types.ModuleType("models"))
    models_package.__path__ = [str(MODELS_DIR)]
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]

    if "models.builders.base" in sys.modules:
        return sys.modules["models.builders.base"]
    spec = importlib.util.spec_from_file_location("models.builders.base", BUILDERS_DIR / "base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.builders.base"] = module
    spec.loader.exec_module(module)
    return module


def _load_builder_entrypoint_module():
    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model
Qwen35TextModel = importlib.import_module("models.builders.qwen").Qwen35TextModel
OnnxSourceModel = importlib.import_module("models.builders.onnx_source").OnnxSourceModel


@pytest.fixture
def paged_onnx_source(tmp_path):
    source = Model(graph_name="source")
    source.graph.opset_imports["com.microsoft"] = 1
    source.model.metadata_props["source_record"] = "preserved"
    inputs = {
        "qkv": (ir.DataType.BFLOAT16, [3, 1024]),
        "past_keys_0": (ir.DataType.BFLOAT16, [1, 256, 1, 128]),
        "past_values_0": (ir.DataType.BFLOAT16, [1, 256, 1, 128]),
        "cumulative_sequence_lengths": (ir.DataType.INT32, [2]),
        "past_sequence_lengths": (ir.DataType.INT32, [1]),
        "block_table": (ir.DataType.INT32, [1, 1]),
        "attention_metadata": (ir.DataType.INT32, [3]),
    }
    for name, (dtype, shape) in inputs.items():
        source.graph.inputs.append(source.make_value(name, dtype, shape))
    source.make_initializer(np.arange(512, dtype=np.int32).astype(np.uint8), "packed_weight")
    source.make_node(
        "PagedAttention",
        inputs=[
            "qkv",
            "",
            "",
            "past_keys_0",
            "past_values_0",
            "cumulative_sequence_lengths",
            "past_sequence_lengths",
            "block_table",
            *([""] * 8),
            "attention_metadata",
        ],
        outputs=["attention", "present_keys_0", "present_values_0"],
        name="attention",
        domain="com.microsoft",
        num_heads=6,
        kv_num_heads=1,
    )
    for name, shape in (
        ("attention", [3, 768]),
        ("present_keys_0", [1, 256, 1, 128]),
        ("present_values_0", [1, 256, 1, 128]),
    ):
        source.graph.outputs.append(source.make_value(name, ir.DataType.BFLOAT16, shape))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "model.onnx"
    source.save_onnx_model(source.model, source_path)
    decoder = {
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "head_size": 128,
        "inputs": {"past_key_names": "past_keys_%d", "past_value_names": "past_values_%d"},
        "outputs": {"present_key_names": "present_keys_%d", "present_value_names": "present_values_%d"},
    }
    return source_path, decoder


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("filename", [None, "imported.onnx"])
def test_onnx_source_preserves_weights_and_applies_cache_policy(tmp_path, paged_onnx_source, quantized, filename):
    source_path, decoder = paged_onnx_source
    options = {}
    if quantized:
        scale_path = tmp_path / "scales.json"
        scale_path.write_text(
            json.dumps(
                {
                    "scales": {"k_scales": [[0.125] * 128], "v_scales": [[0.25] * 128]},
                    "qmax": 128,
                    "layer_ids": [0],
                }
            )
        )
        options = {"kv_cache_quant_scheme": "int8_per_channel", "kv_cache_scale_file": str(scale_path)}
    if filename is not None:
        options["filename"] = filename
    source_bytes = source_path.read_bytes()
    source_data = source_path.with_suffix(".onnx.data").read_bytes()
    model = OnnxSourceModel.from_pretrained(source_path, decoder, options)
    assert model.filename == (filename or source_path.name)
    assert model.kv_cache_attrs == {
        "quant_scheme": "int8_per_channel" if quantized else "none",
        "quant_mode": "PER_CHANNEL",
        "scales_path": options.get("kv_cache_scale_file", ""),
    }
    exported_path = model.save_model(tmp_path / "exported")
    assert exported_path.name == model.filename
    exported = ir.load(exported_path)
    assert exported.metadata_props["source_record"] == "preserved"
    assert source_path.read_bytes() == source_bytes
    assert source_path.with_suffix(".onnx.data").read_bytes() == source_data
    np.testing.assert_array_equal(
        exported.graph.initializers["packed_weight"].const_value.numpy(),
        np.arange(512, dtype=np.int32).astype(np.uint8),
    )
    cache_dtype = ir.DataType.INT8 if quantized else ir.DataType.BFLOAT16
    assert {
        value.dtype for value in exported.graph.inputs if value.name.startswith(("past_keys_", "past_values_"))
    } == {cache_dtype}
    assert {value.dtype for value in exported.graph.outputs if value.name.startswith("present_")} == {cache_dtype}
    assert exported.graph.inputs[0].dtype == ir.DataType.BFLOAT16
    attention = next(node for node in exported.graph if node.op_type == "PagedAttention")
    assert attention.attributes["num_heads"].as_int() == 6
    assert attention.attributes["kv_num_heads"].as_int() == 1
    assert len(attention.inputs) == 17
    assert len(exported.graph) == 1
    if quantized:
        assert (model.num_layers, model.num_kv_heads, model.head_size) == (1, 1, 128)
        assert model.input_names["past_key_values.key"] == {0: "past_keys_0"}
        assert attention.attributes["k_quant_type"].as_string() == "PER_CHANNEL"
        for index, expected in ((14, 0.125), (15, 0.25)):
            scale = attention.inputs[index].const_value.numpy()
            assert scale.shape == (1, 1, 128)
            np.testing.assert_array_equal(scale, np.full((1, 1, 128), expected, dtype=np.float32))
    else:
        assert "k_quant_type" not in attention.attributes
    assert all(value.const_value.location == f"{model.filename}.data" for value in exported.graph.initializers.values())


def test_onnx_source_uses_sparse_paged_kv_state_groups(tmp_path, paged_onnx_source):
    source_path, decoder = paged_onnx_source
    source = OnnxSourceModel.from_pretrained(source_path)
    for value in (*source.graph.inputs, *source.graph.outputs):
        if value.name.startswith(("past_keys_", "past_values_", "present_keys_", "present_values_")):
            value.name = value.name.replace("_0", "_3")
    hybrid_path = source.save_model(tmp_path / "hybrid")
    decoder["num_hidden_layers"] = 4
    decoder["state_groups"] = [
        {"kind": "paged_kv", "layer_ids": [3]},
        {"kind": "fixed_conv", "layer_ids": [0, 1, 2]},
        {"kind": "fixed_recurrent", "layer_ids": [0, 1, 2]},
    ]
    scales_path = tmp_path / "scales.json"
    scales_path.write_text(
        json.dumps(
            {
                "scales": {"k_scales": [[0.125] * 128], "v_scales": [[0.25] * 128]},
                "qmax": 128,
                "layer_ids": [3],
            }
        )
    )
    model = OnnxSourceModel.from_pretrained(
        hybrid_path,
        decoder,
        {"kv_cache_quant_scheme": "int8_per_channel", "kv_cache_scale_file": str(scales_path)},
    )
    assert model.input_names["past_key_values.key"] == {3: "past_keys_3"}
    exported = ir.load(model.save_model(tmp_path / "exported"))
    attention = next(node for node in exported.graph if node.op_type == "PagedAttention")
    assert attention.inputs[3].name == "past_keys_3"
    assert attention.inputs[3].dtype == attention.inputs[4].dtype == ir.DataType.INT8
    assert (attention.inputs[14].name, attention.inputs[15].name) == model.get_kv_cache_scale_names(3)
    np.testing.assert_array_equal(attention.inputs[14].const_value.numpy(), np.full((1, 1, 128), 0.125))
    np.testing.assert_array_equal(attention.inputs[15].const_value.numpy(), np.full((1, 1, 128), 0.25))


@pytest.mark.parametrize("topology", ["output_alias", "output_consumer", "input_consumer"])
def test_onnx_source_rejects_unsupported_cache_conversion_topology(tmp_path, paged_onnx_source, topology):
    onnx = importlib.import_module("onnx")
    source_path, decoder = paged_onnx_source
    source = OnnxSourceModel.from_pretrained(source_path)
    cache_shape = [1, 256, 1, 128]
    if topology == "output_alias":
        source.rename_value("present_keys_0", "raw_keys")
        source.graph.outputs[:] = [value for value in source.graph.outputs if value.name != "raw_keys"]
        source.make_node("Identity", inputs=["raw_keys"], outputs=["present_keys_0"], name="cache_alias")
        source.graph.outputs.append(source.make_value("present_keys_0", ir.DataType.BFLOAT16, cache_shape))
    else:
        cache_name = "past_keys_0" if topology == "input_consumer" else "present_keys_0"
        source.make_node(
            "Cast", inputs=[cache_name], outputs=["floating_cache"], name="floating_consumer", to=ir.DataType.FLOAT
        )
        source.graph.outputs.append(source.make_value("floating_cache", ir.DataType.FLOAT, cache_shape))
    topology_path = source.save_model(tmp_path / "topology")
    onnx.checker.check_model(topology_path)
    scales_path = tmp_path / "scales.json"
    scales_path.write_text(json.dumps({"scales": {"k_scales": [[0.125] * 128], "v_scales": [[0.25] * 128]}}))

    with pytest.raises(ValueError, match=r"cache output|cache consumer"):
        OnnxSourceModel.from_pretrained(
            topology_path,
            decoder,
            {"kv_cache_quant_scheme": "int8_per_channel", "kv_cache_scale_file": str(scales_path)},
        )


def test_onnx_source_preserves_existing_int8_target(tmp_path, paged_onnx_source):
    source_path, decoder = paged_onnx_source
    scales = tmp_path / "scales.json"
    scales.write_text(
        json.dumps(
            {
                "scales": {"k_scales": [[0.125] * 128], "v_scales": [[0.25] * 128]},
                "qmax": 128,
                "layer_ids": [0],
            }
        )
    )
    quantized = OnnxSourceModel.from_pretrained(
        source_path,
        decoder,
        {
            "kv_cache_quant_scheme": "int8_per_channel",
            "kv_cache_scale_file": str(scales),
        },
    ).save_model(tmp_path / "quantized")
    imported = OnnxSourceModel.from_pretrained(quantized)
    exported = ir.load(imported.save_model(tmp_path / "imported"))
    assert exported.metadata_props["onnxruntime_genai.kv_cache_quant_scheme"] == "int8_per_channel"
    attention = next(node for node in exported.graph if node.op_type == "PagedAttention")
    assert attention.inputs[3].dtype == attention.inputs[4].dtype == ir.DataType.INT8
    assert len(attention.inputs) == 17
    np.testing.assert_array_equal(
        attention.inputs[14].const_value.numpy(), np.full((1, 1, 128), 0.125, dtype=np.float32)
    )


@pytest.mark.parametrize("scheme", ["int4_per_channel", "int8_per_token"])
def test_onnx_source_rejects_unimplemented_cache_formats(paged_onnx_source, scheme):
    source_path, decoder = paged_onnx_source
    with pytest.raises(ValueError, match="supports only none or int8_per_channel"):
        OnnxSourceModel.from_pretrained(source_path, decoder, {"kv_cache_quant_scheme": scheme})


@pytest.mark.parametrize("invalid_attribute", [False, True])
def test_onnx_source_validates_ort_normalization_without_registry_leaks(tmp_path, invalid_attribute):
    onnx = importlib.import_module("onnx")
    previously_registered = onnx.defs.has("SimplifiedLayerNormalization")
    source = Model(graph_name="normalization_source")
    source.graph.inputs.append(source.make_value("input", ir.DataType.FLOAT, [1, 4]))
    source.make_initializer(np.ones(4, dtype=np.float32), "scale")
    source.make_node(
        "SimplifiedLayerNormalization",
        inputs=["input", "scale"],
        outputs=["output"],
        name="normalization",
        epsilon=1e-5,
        **({"invalid_attribute": 1} if invalid_attribute else {}),
    )
    source.graph.outputs.append(source.make_value("output", ir.DataType.FLOAT, [1, 4]))
    source_path = tmp_path / "normalization.onnx"
    source.save_onnx_model(source.model, source_path)
    if invalid_attribute:
        with pytest.raises(ValueError, match="not a valid ONNX model"):
            OnnxSourceModel.from_pretrained(source_path)
    else:
        model = OnnxSourceModel.from_pretrained(source_path)
        assert next(iter(model.graph)).op_type == "SimplifiedLayerNormalization"
    assert onnx.defs.has("SimplifiedLayerNormalization") == previously_registered


def test_onnx_source_keeps_shape_initializers_loadable(tmp_path):
    ort = importlib.import_module("onnxruntime")
    source = Model(graph_name="shape_source")
    source.graph.inputs.append(source.make_value("input", ir.DataType.FLOAT, [2]))
    source.make_initializer(np.asarray([0], dtype=np.int64), "axis_zero")
    source.make_node("Unsqueeze", inputs=["input", "axis_zero"], outputs=["output"], name="unsqueeze")
    source.graph.outputs.append(source.make_value("output", ir.DataType.FLOAT, [1, 2]))
    source_path = tmp_path / "shape.onnx"
    source.save_onnx_model(source.model, source_path)
    exported_path = OnnxSourceModel.from_pretrained(source_path).save_model(tmp_path / "exported")
    exported = ir.load(exported_path)
    assert not isinstance(exported.graph.initializers["axis_zero"].const_value, ir.ExternalTensor)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"input": np.asarray([1, 2], dtype=np.float32)})[0], [[1, 2]])


@pytest.mark.parametrize(
    "node_names",
    [
        ("source/1/Add", None, "source/1/Add/1"),
        (None, "source/0/Add", "source/0/Add/1"),
        (None, None, None),
    ],
)
def test_onnx_source_preserves_colliding_named_and_unnamed_nodes(tmp_path, node_names):
    onnx = importlib.import_module("onnx")
    ort = importlib.import_module("onnxruntime")
    source = Model(graph_name="node_name_source")
    previous = source.make_value("input", ir.DataType.FLOAT, [2])
    source.graph.inputs.append(previous)
    source.make_initializer(np.ones(2, dtype=np.float32), "increment")
    for index, name in enumerate(node_names):
        output = source.make_value(f"sum_{index}", ir.DataType.FLOAT, [2])
        source.graph.append(
            ir.node("Add", inputs=[previous, source.graph.initializers["increment"]], outputs=[output], name=name)
        )
        previous = output
    source.graph.outputs.append(previous)
    source_path = tmp_path / "node_names.onnx"
    source.save_onnx_model(source.model, source_path)
    proto = onnx.load(source_path, load_external_data=False)
    for node, original_name in zip(proto.graph.node, node_names, strict=True):
        if original_name is None:
            node.ClearField("name")
    source_path.write_bytes(proto.SerializeToString())
    onnx.checker.check_model(source_path)

    exported_path = OnnxSourceModel.from_pretrained(source_path).save_model(tmp_path / "exported")
    exported = ir.load(exported_path)
    names = [node.name for node in exported.graph]
    assert len(names) == len(set(names)) == len(node_names)
    assert all(name in names for name in node_names if name)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"input": np.asarray([1, 2], dtype=np.float32)})[0], [4, 5])


def test_onnx_source_preserves_folded_constant_initializer(tmp_path):
    ort = importlib.import_module("onnxruntime")
    source = Model(graph_name="folded_constant_source")
    constant_name = "/model/constants/INT64/[0]"
    input_value = source.make_value("input", ir.DataType.FLOAT, [2])
    output_value = source.make_value("output", ir.DataType.FLOAT, [2, 1])
    source.graph.inputs.append(input_value)
    source.make_initializer(np.asarray([1], dtype=np.int64), constant_name)
    source.graph.append(
        ir.node(
            "Unsqueeze",
            inputs=[input_value, source.graph.initializers[constant_name]],
            outputs=[output_value],
            name="unsqueeze",
        )
    )
    source.graph.outputs.append(output_value)
    source_path = tmp_path / "folded.onnx"
    source.save_onnx_model(source.model, source_path)

    exported_path = OnnxSourceModel.from_pretrained(source_path).save_model(tmp_path / "exported")
    exported = ir.load(exported_path)
    assert [node.op_type for node in exported.graph] == ["Unsqueeze"]
    initializer = exported.graph.initializers[constant_name]
    assert initializer.producer() is None
    np.testing.assert_array_equal(initializer.const_value.numpy(), [1])
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"input": np.asarray([1, 2], dtype=np.float32)})[0], [[1], [2]])


@pytest.mark.parametrize("operation", ["SequenceConstruct", "SequenceAt", "Identity"])
def test_onnx_source_preserves_sequence_types(tmp_path, operation):
    onnx = importlib.import_module("onnx")
    ort = importlib.import_module("onnxruntime")
    source = Model(graph_name="sequence_source")
    input_value = source.make_value("input", ir.DataType.FLOAT, [2])
    output_value = source.make_value("output", ir.DataType.FLOAT, [2])
    if operation != "SequenceConstruct":
        input_value.type = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
    if operation != "SequenceAt":
        output_value.type = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
    source.graph.inputs.append(input_value)
    inputs = ["input"]
    if operation == "SequenceAt":
        source.make_initializer(np.asarray(0, dtype=np.int64), "index")
        inputs.append("index")
    source.make_node(operation, inputs=inputs, outputs=["output"], name="sequence")
    source.graph.outputs.append(output_value)
    source_path = tmp_path / "sequence.onnx"
    source.save_onnx_model(source.model, source_path)
    onnx.checker.check_model(source_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    feed = np.asarray([1, 2], dtype=np.float32)
    if operation != "SequenceConstruct":
        feed = [feed, np.asarray([3, 4], dtype=np.float32)]
    original = ort.InferenceSession(str(source_path), options, providers=["CPUExecutionProvider"])
    expected = original.run(None, {"input": feed})[0]

    exported_path = OnnxSourceModel.from_pretrained(source_path).save_model(tmp_path / "exported")
    onnx.checker.check_model(exported_path)
    session = ort.InferenceSession(str(exported_path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": feed})[0]
    if operation == "SequenceAt":
        np.testing.assert_array_equal(actual, expected)
    else:
        assert len(actual) == len(expected)
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_tensor, expected_tensor)


def test_onnx_source_preserves_typed_empty_float_attributes(tmp_path):
    onnx = importlib.import_module("onnx")
    source = Model(graph_name="typed_attribute_source")
    source.make_node(
        "Constant", inputs=[], outputs=["output"], name="empty_floats",
        value_floats=ir.Attr("value_floats", ir.AttributeType.FLOATS, []),
    )
    source.graph.outputs.append(source.make_value("output", ir.DataType.FLOAT, [0]))
    source_path = tmp_path / "empty.onnx"
    source.save_onnx_model(source.model, source_path)
    onnx.checker.check_model(source_path)

    exported_path = OnnxSourceModel.from_pretrained(source_path).save_model(tmp_path / "exported")
    onnx.checker.check_model(exported_path)
    exported = ir.load(exported_path)
    attribute = next(iter(exported.graph)).attributes["value_floats"]
    assert attribute.type == ir.AttributeType.FLOATS
    assert attribute.value == () or attribute.value == []


def test_onnx_source_rejects_sparse_initializers_before_ir_conversion(tmp_path):
    onnx = importlib.import_module("onnx")
    source = Model(graph_name="sparse_source")
    source.graph.inputs.append(source.make_value("input", ir.DataType.FLOAT, [2]))
    source.make_node("Add", inputs=["input", "sparse_weight"], outputs=["output"], name="add")
    source.graph.outputs.append(source.make_value("output", ir.DataType.FLOAT, [2]))
    proto = ir.serde.serialize_model(source.model)
    sparse = proto.graph.sparse_initializer.add()
    sparse.dims.append(2)
    sparse.values.name = "sparse_weight"
    sparse.values.data_type = onnx.TensorProto.FLOAT
    sparse.values.dims.append(1)
    sparse.values.float_data.append(3.0)
    sparse.indices.data_type = onnx.TensorProto.INT64
    sparse.indices.dims.append(1)
    sparse.indices.int64_data.append(1)
    source_path = tmp_path / "sparse.onnx"
    source_path.write_bytes(proto.SerializeToString())
    onnx.checker.check_model(source_path)

    with pytest.raises(ValueError, match="unsupported sparse initializers"):
        OnnxSourceModel.from_pretrained(source_path)
    assert onnx.load(source_path).graph.sparse_initializer[0].values.name == "sparse_weight"


def _check_extra_options(extra_options, precision, execution_provider):
    # The CLI-level `check_extra_options` loads the Hugging Face config; stub it out
    # so these validation-only tests need no model download.
    builder_module.get_hf_details = lambda *args, **kwargs: {
        "hf_config": types.SimpleNamespace(tie_word_embeddings=True)
    }
    builder_module.check_extra_options("dummy-model", "", "", precision, execution_provider, "", extra_options)


# ===========================================================================
# check_extra_options: kv_cache_quant_scheme validation
# ===========================================================================


@pytest.mark.parametrize(
    "quant_type",
    [
        "none",
        "int8_per_tensor",
        "int8_per_channel",
        "int4_per_tensor",
        "int4_per_channel",
        "fp8_per_tensor",
        "fp8_per_channel",
    ],
)
def test_valid_kv_cache_quant_schemes_are_accepted(quant_type):
    kv = {"kv_cache_quant_scheme": quant_type}
    _check_extra_options(kv, "fp16", "cuda")
    assert kv["kv_cache_quant_scheme"] == quant_type


def test_invalid_kv_cache_quant_scheme_is_rejected():
    with pytest.raises(ValueError, match="kv_cache_quant_scheme must be one of"):
        _check_extra_options({"kv_cache_quant_scheme": "int3_per_tensor"}, "fp16", "cuda")


def test_kv_cache_quant_scheme_is_lowercased():
    kv = {"kv_cache_quant_scheme": "INT8_Per_Tensor"}
    _check_extra_options(kv, "fp16", "cuda")
    assert kv["kv_cache_quant_scheme"] == "int8_per_tensor"


@pytest.mark.parametrize("execution_provider", ["cpu", "cuda"])
def test_quantized_kv_cache_allowed_on_cpu_and_cuda(execution_provider):
    kv = {"kv_cache_quant_scheme": "fp8_per_tensor"}
    _check_extra_options(kv, "fp16", execution_provider)
    assert kv["kv_cache_quant_scheme"] == "fp8_per_tensor"


@pytest.mark.parametrize("execution_provider", ["webgpu", "dml", "rocm"])
def test_quantized_kv_cache_rejected_on_unsupported_ep(execution_provider):
    with pytest.raises(ValueError, match="only supported for the CPU and CUDA"):
        _check_extra_options({"kv_cache_quant_scheme": "int8_per_tensor"}, "fp16", execution_provider)


@pytest.mark.parametrize("execution_provider", ["webgpu", "dml", "cpu", "cuda"])
def test_none_kv_cache_quant_scheme_allowed_on_any_ep(execution_provider):
    # `none` is a no-op and must not be restricted to the CPU/CUDA EPs.
    kv = {"kv_cache_quant_scheme": "none"}
    _check_extra_options(kv, "fp16", execution_provider)
    assert kv["kv_cache_quant_scheme"] == "none"


# ===========================================================================
# make_kv_cache_init: cache dtype / bit width / packing
# ===========================================================================


def _make_kv_model(
    kv_cache_quant_type="none",
    ep="cpu",
    op_type="GroupQueryAttention",
    head_size=16,
    num_kv_heads=2,
    num_layers=3,
    extra_options=None,
    use_paged_attention=False,
):
    model = Model.__new__(Model)
    model.ep = ep
    model.head_size = head_size
    model.num_kv_heads = num_kv_heads
    model.num_layers = num_layers
    model.extra_options = extra_options if extra_options is not None else {}
    model.kv_cache_attrs = {
        "quant_scheme": kv_cache_quant_type,
        "quant_type": (
            ir.DataType.INT8
            if kv_cache_quant_type.startswith("int8")
            else ir.DataType.INT4
            if kv_cache_quant_type.startswith("int4")
            else ir.DataType.FLOAT8E4M3FN
            if kv_cache_quant_type.startswith("fp8")
            else None
        ),
        "bit_width": 4 if kv_cache_quant_type.startswith("int4") else 8,
        "quant_mode": "PER_TOKEN"
        if kv_cache_quant_type.endswith("per_token")
        else "PER_CHANNEL"
        if kv_cache_quant_type.endswith("per_channel")
        else "PER_TENSOR",
        "scales_path": model.extra_options.get("kv_cache_scale_file", ""),
    }
    model.attention_attrs = {"op_type": op_type}
    model.use_paged_attention = use_paged_attention
    model.input_names = {
        "past_key_values.key": {layer_id: f"past_key_values.{layer_id}.key" for layer_id in range(num_layers)},
        "past_key_values.value": {layer_id: f"past_key_values.{layer_id}.value" for layer_id in range(num_layers)},
    }
    model.input_types = {}
    model.output_types = {}
    model.input_shapes = {
        "past_key_values.key": ["batch_size", num_kv_heads, "past_sequence_length", head_size],
        "past_key_values.value": ["batch_size", num_kv_heads, "past_sequence_length", head_size],
    }
    model.output_shapes = {
        "present.key": ["batch_size", num_kv_heads, "total_sequence_length", head_size],
        "present.value": ["batch_size", num_kv_heads, "total_sequence_length", head_size],
    }
    return model


def _initialize_kv_cache(model):
    make_scale_initializers = model.make_kv_cache_scale_initializers
    model.make_kv_cache_scale_initializers = lambda: None
    try:
        model.make_kv_cache_init()
    finally:
        model.make_kv_cache_scale_initializers = make_scale_initializers


def test_quantized_kv_cache_requires_group_query_or_paged_attention():
    model = _make_kv_model(kv_cache_quant_type="int8_per_tensor", op_type="MultiHeadAttention")
    with pytest.raises(ValueError, match="requires GroupQueryAttention or PagedAttention"):
        _initialize_kv_cache(model)


@pytest.mark.parametrize("quant_type", ["int8_per_tensor", "int8_per_channel", "fp8_per_tensor", "fp8_per_channel"])
def test_paged_attention_accepts_int8_and_fp8_kv_cache(quant_type):
    model = _make_kv_model(
        kv_cache_quant_type=quant_type,
        ep="cuda",
        op_type="PagedAttention",
        use_paged_attention=True,
    )
    _initialize_kv_cache(model)

    expected_dtype = ir.DataType.FLOAT8E4M3FN if quant_type.startswith("fp8") else ir.DataType.INT8
    assert model.input_types["past_key_values.key"] == expected_dtype
    assert model.output_types["present.value"] == expected_dtype


@pytest.mark.parametrize("quant_type", ["int4_per_tensor", "int4_per_channel"])
def test_paged_attention_accepts_int4_kv_cache(quant_type):
    model = _make_kv_model(
        kv_cache_quant_type=quant_type,
        ep="cuda",
        op_type="PagedAttention",
        use_paged_attention=True,
    )
    _initialize_kv_cache(model)
    assert model.input_types["past_key_values.key"] == ir.DataType.UINT8
    assert model.input_shapes["past_key_values.key"][-1] == 8


@pytest.mark.parametrize("ep", ["webgpu", "dml", "trt-rtx"])
def test_quantized_kv_cache_init_rejects_unsupported_ep(ep):
    # `check_extra_options` only guards the CLI path, so the builder itself must also reject
    # unsupported EPs for direct `create_model()` callers.
    model = _make_kv_model(kv_cache_quant_type="int8_per_tensor", ep=ep)
    with pytest.raises(ValueError, match="only supported for the CPU and CUDA"):
        _initialize_kv_cache(model)


def test_kv_cache_quant_schemes_constant_matches_builder_validation():
    # `check_extra_options` in builder.py is the single place that validates the option and
    # reads this constant from `quantization`, which is the source of truth.
    assert set(builder_module.KV_CACHE_QUANT_SCHEMES) == {
        "none",
        "int4_per_token",
        "int8_per_token",
        "int8_per_tensor",
        "int8_per_channel",
        "int4_per_tensor",
        "int4_per_channel",
        "fp8_per_tensor",
        "fp8_per_channel",
    }


@pytest.mark.parametrize(
    "quant_type, expected_dtype, expected_bits, expected_quant",
    [
        ("int8_per_tensor", ir.DataType.INT8, 8, "PER_TENSOR"),
        ("int8_per_channel", ir.DataType.INT8, 8, "PER_CHANNEL"),
        ("int4_per_tensor", ir.DataType.UINT8, 4, "PER_TENSOR"),
        ("int4_per_channel", ir.DataType.UINT8, 4, "PER_CHANNEL"),
        ("fp8_per_tensor", ir.DataType.FLOAT8E4M3FN, 8, "PER_TENSOR"),
        ("fp8_per_channel", ir.DataType.FLOAT8E4M3FN, 8, "PER_CHANNEL"),
    ],
)
def test_make_kv_cache_init_sets_dtype_and_metadata(quant_type, expected_dtype, expected_bits, expected_quant):
    model = _make_kv_model(kv_cache_quant_type=quant_type)
    _initialize_kv_cache(model)

    assert model.kv_cache_attrs["bit_width"] == expected_bits
    assert model.kv_cache_attrs["quant_mode"] == expected_quant
    for io in ("past_key_values.key", "past_key_values.value"):
        assert model.input_types[io] == expected_dtype
    for io in ("present.key", "present.value"):
        assert model.output_types[io] == expected_dtype


def test_int4_kv_cache_packs_two_elements_per_byte():
    model = _make_kv_model(kv_cache_quant_type="int4_per_tensor", head_size=16)
    _initialize_kv_cache(model)

    # int4 stores two elements per byte -> last dim is ceil(head_size / 2).
    assert model.input_shapes["past_key_values.key"][-1] == 8
    assert model.input_shapes["past_key_values.value"][-1] == 8
    assert model.output_shapes["present.key"][-1] == 8
    assert model.output_shapes["present.value"][-1] == 8


@pytest.mark.parametrize("scheme", ["int4_per_token", "int8_per_token"])
def test_dynamic_paged_scales_need_no_calibration(scheme):
    model = _make_kv_model(
        scheme,
        ep="cuda",
        op_type="PagedAttention",
        use_paged_attention=True,
        extra_options={"kv_cache_rotation": "hadamard"},
    )
    model.output_names = {
        f"present.{side}": {layer: f"present.{layer}.{side}" for layer in range(model.num_layers)}
        for side in ("key", "value")
    }
    model.make_kv_cache_init()
    assert model.kv_cache_attrs["rotation"] == "HADAMARD"
    assert model.input_types["past_key_values.key_scale"] == ir.DataType.FLOAT16
    assert model.input_shapes["past_key_values.key_scale"] == ["num_blocks", "block_size", 2]
    assert model.output_names["present.value_scale"][2] == "present.2.value_scale"
    assert model.get_kv_cache_scale_inputs(layer_id=0) == ("", "")


def test_per_token_scales_reject_nonpaged_export():
    model = _make_kv_model("int4_per_token", ep="cuda")
    with pytest.raises(ValueError, match="requires CUDA PagedAttention"):
        model.make_kv_cache_init()


def test_hadamard_rejects_per_channel_scales():
    model = _make_kv_model(
        "int4_per_channel",
        ep="cuda",
        op_type="PagedAttention",
        use_paged_attention=True,
        extra_options={"kv_cache_rotation": "hadamard"},
    )
    with pytest.raises(ValueError, match="per-channel"):
        model.make_kv_cache_init()


def test_int4_kv_cache_packs_odd_head_size_with_ceiling():
    model = _make_kv_model(kv_cache_quant_type="int4_per_channel", head_size=15)
    _initialize_kv_cache(model)

    # ceil(15 / 2) == 8
    assert model.input_shapes["past_key_values.key"][-1] == 8
    assert model.output_shapes["present.value"][-1] == 8


@pytest.mark.parametrize("quant_type", ["int8_per_tensor", "fp8_per_tensor"])
def test_non_int4_kv_cache_does_not_pack_head_size(quant_type):
    model = _make_kv_model(kv_cache_quant_type=quant_type, head_size=16)
    _initialize_kv_cache(model)

    # int8/fp8 store one byte per element, so the head-size dim is unchanged.
    assert model.input_shapes["past_key_values.key"][-1] == 16
    assert model.output_shapes["present.value"][-1] == 16


@pytest.mark.parametrize(
    "ep, expected_share",
    [
        ("cuda", True),
        ("cpu", False),
    ],
)
def test_quantized_kv_cache_past_present_share_buffer(ep, expected_share):
    model = _make_kv_model(kv_cache_quant_type="int8_per_tensor", ep=ep)
    _initialize_kv_cache(model)
    assert model.past_present_share_buffer is expected_share


# ===========================================================================
# make_kv_cache_scale_initializers: scale sizes, calibration file, validation
# ===========================================================================


def _capture_initializers(model):
    captured = {}

    def fake_make_initializer(tensor, name, to=None):
        captured[name] = np.asarray(tensor, dtype=np.float32)

    model.make_initializer = fake_make_initializer
    return captured


def test_per_tensor_scale_initializers_are_scalar_per_layer(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"scales": {"k_scales": [0.1, 0.2], "v_scales": [0.3, 0.4]}}))
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=2,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    # One k_scale and one v_scale per layer.
    assert set(captured) == {
        "model.layers.0.attn.k_scale",
        "model.layers.0.attn.v_scale",
        "model.layers.1.attn.k_scale",
        "model.layers.1.attn.v_scale",
    }
    for arr in captured.values():
        assert arr.size == 1


def test_per_channel_scale_initializers_span_num_kv_heads_times_head_size(tmp_path):
    scale_size = 2 * 16
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [[0.1] * scale_size],
                    "v_scales": [[0.2] * scale_size],
                }
            }
        )
    )
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_channel",
        num_kv_heads=2,
        head_size=16,
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    for arr in captured.values():
        assert arr.size == 2 * 16
        # GroupQueryAttention only looks at the element count, so the scales stay flat.
        assert arr.shape == (2 * 16,)


def test_paged_per_channel_scale_initializers_use_canonical_shape(tmp_path):
    # PagedAttention validates the per-channel scale shape and requires (kv_num_heads, 1, head_size),
    # unlike GroupQueryAttention which only checks the element count.
    scale_size = 2 * 16
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [[0.1] * scale_size],
                    "v_scales": [[0.2] * scale_size],
                }
            }
        )
    )
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_channel",
        ep="cuda",
        op_type="PagedAttention",
        num_kv_heads=2,
        head_size=16,
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
        use_paged_attention=True,
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    assert set(captured) == {"model.layers.0.attn.k_scale", "model.layers.0.attn.v_scale"}
    for arr in captured.values():
        assert arr.shape == (2, 1, 16)


def test_graph_named_scale_section_is_selected(tmp_path):
    scale_size = 2 * 16
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "mtp": {
                    "scales": {
                        "k_scales": [[0.1] * scale_size],
                        "v_scales": [[0.2] * scale_size],
                    }
                }
            }
        )
    )
    model = Model.__new__(Model)
    model.kv_cache_attrs = {
        "quant_scheme": "int8_per_channel",
        "quant_mode": "PER_CHANNEL",
        "scales_path": str(scale_file),
    }
    model.use_paged_attention = True
    model.filename = "mtp.onnx"
    model.num_kv_heads = 2
    model.head_size = 16
    model.num_layers = 1
    model.input_names = {"past_key_values.key": ["past_key_values.0.key"]}
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    assert set(captured) == {"model.layers.0.attn.k_scale", "model.layers.0.attn.v_scale"}
    for arr in captured.values():
        assert arr.shape == (2, 1, 16)


def test_paged_per_tensor_scale_initializers_stay_scalar(tmp_path):
    # The canonical (kv_num_heads, 1, head_size) reshape must only apply to per-channel scales.
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"scales": {"k_scales": [0.1], "v_scales": [0.2]}}))
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        ep="cuda",
        op_type="PagedAttention",
        num_kv_heads=2,
        head_size=16,
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
        use_paged_attention=True,
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    for arr in captured.values():
        assert arr.shape == (1,)


def test_missing_scale_file_is_rejected():
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=1,
    )
    _capture_initializers(model)

    with pytest.raises(ValueError, match="kv_cache_scale_file"):
        model.make_kv_cache_scale_initializers()


def test_calibrated_per_layer_scales_are_loaded_from_file(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [0.1, 0.2],
                    "v_scales": [0.3, 0.4],
                }
            }
        )
    )
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=2,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    np.testing.assert_allclose(captured["model.layers.0.attn.k_scale"], 0.1)
    np.testing.assert_allclose(captured["model.layers.1.attn.k_scale"], 0.2)
    np.testing.assert_allclose(captured["model.layers.0.attn.v_scale"], 0.3)
    np.testing.assert_allclose(captured["model.layers.1.attn.v_scale"], 0.4)


def test_calibration_qmax_retargets_scales_to_the_requested_bit_width(tmp_path):
    # A scale is threshold / qmax, so an int8-calibrated file (qmax 128) reused for an int4
    # cache (qmax 8) must scale up by 16x. Without this the cache clips hard.
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"qmax": 128, "scales": {"k_scales": [0.1], "v_scales": [0.2]}}))
    model = _make_kv_model(
        kv_cache_quant_type="int4_per_tensor",
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    np.testing.assert_allclose(captured["model.layers.0.attn.k_scale"], 1.6)
    np.testing.assert_allclose(captured["model.layers.0.attn.v_scale"], 3.2)


def test_calibration_qmax_matching_the_scheme_leaves_scales_unchanged(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"qmax": 128, "scales": {"k_scales": [0.1], "v_scales": [0.2]}}))
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    np.testing.assert_allclose(captured["model.layers.0.attn.k_scale"], 0.1)
    np.testing.assert_allclose(captured["model.layers.0.attn.v_scale"], 0.2)


@pytest.mark.parametrize("qmax", [0, -8, "128", True, float("inf")])
def test_invalid_calibration_qmax_is_rejected(tmp_path, qmax):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"qmax": qmax, "scales": {"k_scales": [0.1], "v_scales": [0.2]}}))
    model = _make_kv_model(
        kv_cache_quant_type="int4_per_tensor",
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    _capture_initializers(model)

    with pytest.raises(ValueError, match="qmax must be"):
        model.make_kv_cache_scale_initializers()


@pytest.mark.parametrize("scale,qmax", [(3e38, 128), (1e-44, 0.01), (1.0, 1e308)])
def test_rescaled_calibration_scales_must_remain_finite_and_positive(tmp_path, scale, qmax):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"qmax": qmax, "scales": {"k_scales": [scale], "v_scales": [scale]}}))
    model = _make_kv_model(
        kv_cache_quant_type="int4_per_tensor",
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    _capture_initializers(model)

    with pytest.raises(ValueError, match="Rescaled kv_cache scale.*finite positive"):
        model.make_kv_cache_scale_initializers()


def test_sparse_layer_ids_map_scales_to_model_layers(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [0.1, 0.2],
                    "v_scales": [0.3, 0.4],
                },
                "layer_ids": [1, 3],
            }
        )
    )
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=4,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    model.kv_quant_type = "PER_TENSOR"
    model.input_names["past_key_values.key"] = {1: "past_key_values.1.key", 3: "past_key_values.3.key"}
    model.input_names["past_key_values.value"] = {1: "past_key_values.1.value", 3: "past_key_values.3.value"}
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    assert set(captured) == {
        "model.layers.1.attn.k_scale",
        "model.layers.1.attn.v_scale",
        "model.layers.3.attn.k_scale",
        "model.layers.3.attn.v_scale",
    }
    np.testing.assert_allclose(captured["model.layers.1.attn.k_scale"], 0.1)
    np.testing.assert_allclose(captured["model.layers.3.attn.k_scale"], 0.2)
    np.testing.assert_allclose(captured["model.layers.1.attn.v_scale"], 0.3)
    np.testing.assert_allclose(captured["model.layers.3.attn.v_scale"], 0.4)


def test_sparse_layer_ids_must_match_model_kv_layers(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [0.1, 0.2],
                    "v_scales": [0.3, 0.4],
                },
                "layer_ids": [0, 2],
            }
        )
    )
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=4,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    model.kv_quant_type = "PER_TENSOR"
    model.input_names["past_key_values.key"] = {1: "past_key_values.1.key", 3: "past_key_values.3.key"}
    model.input_names["past_key_values.value"] = {1: "past_key_values.1.value", 3: "past_key_values.3.value"}
    _capture_initializers(model)

    with pytest.raises(ValueError, match="must match the model's KV-cache layers"):
        model.make_kv_cache_scale_initializers()


@pytest.mark.parametrize(
    "layer_ids,error",
    [
        ([], "must not be empty"),
        ([1, 1], "must not contain duplicates"),
        ([True], "must be a list of integer"),
        ([4], r"must be in \[0, 4\)"),
    ],
)
def test_invalid_sparse_layer_ids_are_rejected(tmp_path, layer_ids, error):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(
        json.dumps(
            {
                "scales": {
                    "k_scales": [0.1] * len(layer_ids),
                    "v_scales": [0.2] * len(layer_ids),
                },
                "layer_ids": layer_ids,
            }
        )
    )
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=4,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    model.kv_quant_type = "PER_TENSOR"
    _capture_initializers(model)

    with pytest.raises(ValueError, match=error):
        model.make_kv_cache_scale_initializers()


def test_calibrated_per_channel_scales_are_loaded_from_file(tmp_path):
    k_vec = [0.1, 0.2, 0.3, 0.4]  # num_kv_heads(2) * head_size(2)
    v_vec = [0.5, 0.6, 0.7, 0.8]
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"scales": {"k_scales": [k_vec], "v_scales": [v_vec]}}))

    model = _make_kv_model(
        kv_cache_quant_type="int8_per_channel",
        num_kv_heads=2,
        head_size=2,
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    np.testing.assert_allclose(captured["model.layers.0.attn.k_scale"], k_vec)
    np.testing.assert_allclose(captured["model.layers.0.attn.v_scale"], v_vec)


def test_scale_file_with_wrong_number_of_layers_is_rejected(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"scales": {"k_scales": [0.1], "v_scales": [0.2]}}))

    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=3,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    _capture_initializers(model)

    with pytest.raises(ValueError, match="must provide 3 per-layer scales"):
        model.make_kv_cache_scale_initializers()


def test_scale_file_with_wrong_per_channel_size_is_rejected(tmp_path):
    # Per-channel scale vector has the wrong length (expected num_kv_heads*head_size == 4).
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"scales": {"k_scales": [[0.1, 0.2]], "v_scales": [[0.3, 0.4]]}}))

    model = _make_kv_model(
        kv_cache_quant_type="int8_per_channel",
        num_kv_heads=2,
        head_size=2,
        num_layers=1,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    _capture_initializers(model)

    with pytest.raises(ValueError, match="expected 4"):
        model.make_kv_cache_scale_initializers()


# ===========================================================================
# make_group_query_attention: scale inputs + quant attributes on the GQA node
# ===========================================================================


def _make_gqa_model(kv_cache_quant_type="none", kv_quant_type="PER_TENSOR", kv_cache_bit_width=8):
    model = Model.__new__(Model)
    model.num_attn_heads = 8
    model.num_kv_heads = 2
    model.head_size = 16
    model.window_size = -1
    model.attention_attrs = {
        "op_type": "GroupQueryAttention",
        "scale": 0.125,
        "softcap": 0.0,
        "use_rope_in_attn": True,
        "qk_norm_epsilon": 1e-6,
    }
    model.rope_attrs = {"interleaved": 0}
    model.io_dtype = ir.DataType.FLOAT16
    model.kv_cache_attrs = {
        "quant_scheme": kv_cache_quant_type,
        "quant_mode": kv_quant_type,
        "bit_width": kv_cache_bit_width,
    }
    model.nodes = []

    def make_node(op_type, inputs, outputs, name, domain="", **attributes):
        model.nodes.append({"op_type": op_type, "inputs": inputs, "outputs": outputs, "attributes": attributes})

    model.make_node = make_node
    model.make_value = lambda *args, **kwargs: None
    return model


# The 12 fixed GQA inputs preceding the optional k/v scales and q/k norm weights.
_GQA_BASE_INPUT_COUNT = 12


def test_paged_dynamic_int4_node_wires_scale_aliases():
    model = _make_gqa_model("int4_per_token", "PER_TOKEN", 4)
    model.kv_cache_attrs["rotation"] = "HADAMARD"
    model.input_names = {
        f"past_key_values.{side}_scale": {3: f"past_key_values.3.{side}_scale"} for side in ("key", "value")
    }
    model.output_names = {f"present.{side}_scale": {3: f"present.3.{side}_scale"} for side in ("key", "value")}
    model.make_paged_attention(
        "/paged",
        layer_id=3,
        q_path="q",
        k_path="k",
        v_path="v",
        past_k="past_key_values.3.key",
        past_v="past_key_values.3.value",
        present_k="present.3.key",
        present_v="present.3.value",
        cumulative_sequence_lengths="cu",
        past_sequence_lengths="past",
        block_table="blocks",
        attention_metadata="metadata",
    )
    node = model.nodes[-1]
    assert node["inputs"][14:19] == ["", "", "metadata", "past_key_values.3.key_scale", "past_key_values.3.value_scale"]
    assert node["outputs"][3:] == ["present.3.key_scale", "present.3.value_scale"]
    assert node["attributes"]["k_cache_dtype"] == "int4"
    assert node["attributes"]["v_cache_dtype"] == "int4"
    assert node["attributes"]["qk_rotation"] == "HADAMARD"
    assert node["attributes"]["v_rotation"] == "HADAMARD"


def test_plain_gqa_has_no_scale_inputs_or_quant_attributes():
    model = _make_gqa_model(kv_cache_quant_type="none")

    model.make_group_query_attention("/gqa", layer_id=0, q_path="q", k_path="k", v_path="v")

    node = model.nodes[-1]
    assert len(node["inputs"]) == _GQA_BASE_INPUT_COUNT
    assert "k_quant_type" not in node["attributes"]
    assert "kv_cache_bit_width" not in node["attributes"]


def test_quantized_gqa_appends_per_layer_scale_inputs():
    model = _make_gqa_model(kv_cache_quant_type="int8_per_tensor")

    model.make_group_query_attention("/gqa", layer_id=3, q_path="q", k_path="k", v_path="v")

    inputs = model.nodes[-1]["inputs"]
    assert inputs[_GQA_BASE_INPUT_COUNT] == "model.layers.3.attn.k_scale"
    assert inputs[_GQA_BASE_INPUT_COUNT + 1] == "model.layers.3.attn.v_scale"


def test_quantized_gqa_sets_quant_attributes():
    model = _make_gqa_model(kv_cache_quant_type="int4_per_channel", kv_quant_type="PER_CHANNEL", kv_cache_bit_width=4)

    model.make_group_query_attention("/gqa", layer_id=0, q_path="q", k_path="k", v_path="v")

    attrs = model.nodes[-1]["attributes"]
    assert attrs["k_quant_type"] == "PER_CHANNEL"
    assert attrs["v_quant_type"] == "PER_CHANNEL"
    assert attrs["kv_cache_bit_width"] == 4


def test_quantized_gqa_requires_layer_id():
    model = _make_gqa_model(kv_cache_quant_type="int8_per_tensor")

    with pytest.raises(ValueError, match="layer_id is required"):
        model.make_group_query_attention("/gqa", q_path="q", k_path="k", v_path="v")


def test_quantized_gqa_scales_precede_qk_norm_weights_without_placeholders():
    model = _make_gqa_model(kv_cache_quant_type="int8_per_tensor")

    model.make_group_query_attention(
        "/gqa",
        layer_id=1,
        q_path="q",
        k_path="k",
        v_path="v",
        q_norm_weight="q_norm",
        k_norm_weight="k_norm",
    )

    inputs = model.nodes[-1]["inputs"]
    # Quantized path uses the real scale inputs (no empty placeholders) directly
    # followed by the q/k norm weights.
    assert inputs[_GQA_BASE_INPUT_COUNT] == "model.layers.1.attn.k_scale"
    assert inputs[_GQA_BASE_INPUT_COUNT + 1] == "model.layers.1.attn.v_scale"
    assert inputs[_GQA_BASE_INPUT_COUNT + 2] == "q_norm"
    assert inputs[_GQA_BASE_INPUT_COUNT + 3] == "k_norm"


def test_plain_gqa_with_qk_norm_uses_empty_scale_placeholders():
    model = _make_gqa_model(kv_cache_quant_type="none")

    model.make_group_query_attention(
        "/gqa",
        layer_id=0,
        q_path="q",
        k_path="k",
        v_path="v",
        q_norm_weight="q_norm",
        k_norm_weight="k_norm",
    )

    inputs = model.nodes[-1]["inputs"]
    # Non-quantized path keeps empty k/v scale slots before the norm weights.
    assert inputs[_GQA_BASE_INPUT_COUNT] == ""
    assert inputs[_GQA_BASE_INPUT_COUNT + 1] == ""
    assert inputs[_GQA_BASE_INPUT_COUNT + 2] == "q_norm"
    assert inputs[_GQA_BASE_INPUT_COUNT + 3] == "k_norm"
