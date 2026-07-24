# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for quantized KV cache support in the model builder.

The quantized KV cache stores the past/present key and value tensors in a
compressed integer/FP8 encoding instead of the model's floating-point I/O dtype.
It is selected via the ``kv_cache_quant_type`` extra option and is only valid
with ``GroupQueryAttention``. These tests exercise the builder plumbing
standalone (no model download):

* ``check_extra_options`` validation of ``kv_cache_quant_type``.
* ``make_quantized_kv_cache_init`` (cache dtype, bit width, quant granularity,
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
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_base_module():
    sys.modules.setdefault("models", types.ModuleType("models"))
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]

    spec = importlib.util.spec_from_file_location("models.builders.base", BUILDERS_DIR / "base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.builders.base"] = module
    spec.loader.exec_module(module)
    return module


def _load_builder_entrypoint_module():
    # `builder.py` imports the concrete model classes via `from builders import (...)`.
    # Provide a stub `builders` module so we can import the lightweight helpers
    # (`check_extra_options`) without pulling in every model builder.
    builders_stub = types.ModuleType("builders")

    def _stub_getattr(name):  # PEP 562: satisfies `from builders import <ModelClass>`
        return type(name, (), {})

    builders_stub.__getattr__ = _stub_getattr
    sys.modules["builders"] = builders_stub

    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model


# ===========================================================================
# check_extra_options: kv_cache_quant_type validation
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
def test_valid_kv_cache_quant_types_are_accepted(quant_type):
    kv = {"kv_cache_quant_type": quant_type}
    builder_module.check_extra_options(kv, "fp16", "cuda")
    assert kv["kv_cache_quant_type"] == quant_type


def test_invalid_kv_cache_quant_type_is_rejected():
    with pytest.raises(ValueError, match="kv_cache_quant_type must be one of"):
        builder_module.check_extra_options({"kv_cache_quant_type": "int3_per_tensor"}, "fp16", "cuda")


def test_kv_cache_quant_type_is_lowercased():
    kv = {"kv_cache_quant_type": "INT8_Per_Tensor"}
    builder_module.check_extra_options(kv, "fp16", "cuda")
    assert kv["kv_cache_quant_type"] == "int8_per_tensor"


@pytest.mark.parametrize("execution_provider", ["cpu", "cuda"])
def test_quantized_kv_cache_allowed_on_cpu_and_cuda(execution_provider):
    kv = {"kv_cache_quant_type": "fp8_per_tensor"}
    builder_module.check_extra_options(kv, "fp16", execution_provider)
    assert kv["kv_cache_quant_type"] == "fp8_per_tensor"


@pytest.mark.parametrize("execution_provider", ["webgpu", "dml", "rocm"])
def test_quantized_kv_cache_rejected_on_unsupported_ep(execution_provider):
    with pytest.raises(ValueError, match="only supported for the CPU and CUDA"):
        builder_module.check_extra_options({"kv_cache_quant_type": "int8_per_tensor"}, "fp16", execution_provider)


@pytest.mark.parametrize("execution_provider", ["webgpu", "dml", "cpu", "cuda"])
def test_none_kv_cache_quant_type_allowed_on_any_ep(execution_provider):
    # `none` is a no-op and must not be restricted to the CPU/CUDA EPs.
    kv = {"kv_cache_quant_type": "none"}
    builder_module.check_extra_options(kv, "fp16", execution_provider)
    assert kv["kv_cache_quant_type"] == "none"


# ===========================================================================
# make_quantized_kv_cache_init: cache dtype / bit width / packing
# ===========================================================================


def _make_kv_model(
    kv_cache_quant_type="none",
    ep="cpu",
    op_type="GroupQueryAttention",
    head_size=16,
    num_kv_heads=2,
    num_layers=3,
    extra_options=None,
):
    model = Model.__new__(Model)
    model.ep = ep
    model.head_size = head_size
    model.num_kv_heads = num_kv_heads
    model.num_layers = num_layers
    model.kv_cache_quant_type = kv_cache_quant_type
    model.extra_options = extra_options if extra_options is not None else {}
    model.attention_attrs = {"op_type": op_type}
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


def test_quantized_kv_cache_requires_group_query_attention():
    model = _make_kv_model(kv_cache_quant_type="int8_per_tensor", op_type="MultiHeadAttention")
    with pytest.raises(ValueError, match="requires GroupQueryAttention"):
        model.make_quantized_kv_cache_init()


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
def test_make_quantized_kv_cache_init_sets_dtype_and_metadata(
    quant_type, expected_dtype, expected_bits, expected_quant
):
    model = _make_kv_model(kv_cache_quant_type=quant_type)
    model.make_quantized_kv_cache_init()

    assert model.kv_cache_bit_width == expected_bits
    assert model.kv_quant_type == expected_quant
    for io in ("past_key_values.key", "past_key_values.value"):
        assert model.input_types[io] == expected_dtype
    for io in ("present.key", "present.value"):
        assert model.output_types[io] == expected_dtype


def test_int4_kv_cache_packs_two_elements_per_byte():
    model = _make_kv_model(kv_cache_quant_type="int4_per_tensor", head_size=16)
    model.make_quantized_kv_cache_init()

    # int4 stores two elements per byte -> last dim is ceil(head_size / 2).
    assert model.input_shapes["past_key_values.key"][-1] == 8
    assert model.input_shapes["past_key_values.value"][-1] == 8
    assert model.output_shapes["present.key"][-1] == 8
    assert model.output_shapes["present.value"][-1] == 8


def test_int4_kv_cache_packs_odd_head_size_with_ceiling():
    model = _make_kv_model(kv_cache_quant_type="int4_per_channel", head_size=15)
    model.make_quantized_kv_cache_init()

    # ceil(15 / 2) == 8
    assert model.input_shapes["past_key_values.key"][-1] == 8
    assert model.output_shapes["present.value"][-1] == 8


@pytest.mark.parametrize("quant_type", ["int8_per_tensor", "fp8_per_tensor"])
def test_non_int4_kv_cache_does_not_pack_head_size(quant_type):
    model = _make_kv_model(kv_cache_quant_type=quant_type, head_size=16)
    model.make_quantized_kv_cache_init()

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
    model.make_quantized_kv_cache_init()
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


def test_per_tensor_scale_initializers_are_scalar_per_layer():
    model = _make_kv_model(kv_cache_quant_type="int8_per_tensor", num_layers=2)
    model.kv_quant_type = "PER_TENSOR"
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    # One k_scale and one v_scale per layer.
    assert set(captured) == {
        "/model/kv_cache_scales/k_scale.0",
        "/model/kv_cache_scales/v_scale.0",
        "/model/kv_cache_scales/k_scale.1",
        "/model/kv_cache_scales/v_scale.1",
    }
    for arr in captured.values():
        assert arr.size == 1


def test_per_channel_scale_initializers_span_num_kv_heads_times_head_size():
    model = _make_kv_model(kv_cache_quant_type="int8_per_channel", num_kv_heads=2, head_size=16, num_layers=1)
    model.kv_quant_type = "PER_CHANNEL"
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    for arr in captured.values():
        assert arr.size == 2 * 16


def test_default_scale_value_is_used_when_no_calibration_file():
    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=1,
        extra_options={"kv_cache_scale": "0.125"},
    )
    model.kv_quant_type = "PER_TENSOR"
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    for arr in captured.values():
        np.testing.assert_allclose(arr, 0.125)


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
    model.kv_quant_type = "PER_TENSOR"
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    np.testing.assert_allclose(captured["/model/kv_cache_scales/k_scale.0"], 0.1)
    np.testing.assert_allclose(captured["/model/kv_cache_scales/k_scale.1"], 0.2)
    np.testing.assert_allclose(captured["/model/kv_cache_scales/v_scale.0"], 0.3)
    np.testing.assert_allclose(captured["/model/kv_cache_scales/v_scale.1"], 0.4)


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
    model.kv_quant_type = "PER_CHANNEL"
    captured = _capture_initializers(model)

    model.make_kv_cache_scale_initializers()

    np.testing.assert_allclose(captured["/model/kv_cache_scales/k_scale.0"], k_vec)
    np.testing.assert_allclose(captured["/model/kv_cache_scales/v_scale.0"], v_vec)


def test_scale_file_with_wrong_number_of_layers_is_rejected(tmp_path):
    scale_file = tmp_path / "kv_scales.json"
    scale_file.write_text(json.dumps({"scales": {"k_scales": [0.1], "v_scales": [0.2]}}))

    model = _make_kv_model(
        kv_cache_quant_type="int8_per_tensor",
        num_layers=3,
        extra_options={"kv_cache_scale_file": str(scale_file)},
    )
    model.kv_quant_type = "PER_TENSOR"
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
    model.kv_quant_type = "PER_CHANNEL"
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
    model.kv_cache_quant_type = kv_cache_quant_type
    model.kv_quant_type = kv_quant_type
    model.kv_cache_bit_width = kv_cache_bit_width
    model.nodes = []

    def make_node(op_type, inputs, outputs, name, domain="", **attributes):
        model.nodes.append({"op_type": op_type, "inputs": inputs, "attributes": attributes})

    model.make_node = make_node
    model.make_value = lambda *args, **kwargs: None
    return model


# The 12 fixed GQA inputs preceding the optional k/v scales and q/k norm weights.
_GQA_BASE_INPUT_COUNT = 12


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
    assert inputs[_GQA_BASE_INPUT_COUNT] == "/model/kv_cache_scales/k_scale.3"
    assert inputs[_GQA_BASE_INPUT_COUNT + 1] == "/model/kv_cache_scales/v_scale.3"


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
    assert inputs[_GQA_BASE_INPUT_COUNT] == "/model/kv_cache_scales/k_scale.1"
    assert inputs[_GQA_BASE_INPUT_COUNT + 1] == "/model/kv_cache_scales/v_scale.1"
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
