# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Per-layer state manifest and I/O registration for Qwen-3.8 Flash Next (``Qwen4Exp``).

Qwen4Exp carries four kinds of per-layer state:

* paged KV blocks and a QSA indexer key cache on full-attention layers,
* short-conv + recurrent state on linear-attention layers,
* a PLE short-conv state and an n-gram token history on PLE layers.

If the manifest and the graph I/O names disagree the runtime silently allocates the wrong
buffers, so both are asserted here.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))

sys.modules.setdefault("models", types.ModuleType("models"))
_builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
_builders_package.__path__ = [str(BUILDERS_DIR)]


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


_load_builder_module("base")
_load_builder_module("quant_config")
_load_builder_module("qwen")
qwen4exp = _load_builder_module("qwen4exp")

Qwen4ExpTextModel = qwen4exp.Qwen4ExpTextModel

NUM_LAYERS = 8
HIDDEN = 32
HC_COUNT = 4
NGRAM_SIZE = 4
PLE_CONV_KERNEL = 3
# Every 4th layer is full attention, matching the published hybrid ratio.
LAYER_TYPES = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(NUM_LAYERS)]
PLE_LAYERS = [0, 1]


def _make_model(use_paged_attention=True, indexer_head_dim=64, ple_layers=PLE_LAYERS, layer_types=None):
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.use_paged_attention = use_paged_attention
    model.layer_types = list(layer_types or LAYER_TYPES)
    model.num_layers = len(model.layer_types)
    model.hidden_size = HIDDEN
    model.hc_count = HC_COUNT
    model.hc_hidden_size = HC_COUNT * HIDDEN
    model.indexer_head_dim = indexer_head_dim
    model.indexer_kv_heads = 1
    model.ngram_size = NGRAM_SIZE
    model.ple_conv_kernel_size = PLE_CONV_KERNEL
    model.ple_conv_state_len = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE
    model.ple_layer_index = {layer_id: idx for idx, layer_id in enumerate(ple_layers)}
    model.input_names = {}
    model.output_names = {}
    model.input_types = {}
    model.input_shapes = {}
    model.output_types = {}
    model.output_shapes = {}
    return model


#####################################################################################
# State manifest
#####################################################################################


def test_paged_manifest_lists_every_qwen4exp_state_kind_in_order():
    model = _make_model()
    inputs, outputs = {}, {}

    groups = model.make_decoder_state_groups(inputs, outputs)

    assert [group["kind"] for group in groups] == [
        "paged_kv",
        "fixed_conv",
        "fixed_recurrent",
        "fixed_ple_conv",
        "fixed_ple_tokens",
        "paged_indexer_key",
    ]


def test_paged_manifest_assigns_each_group_the_right_layers():
    model = _make_model()
    groups = {group["kind"]: group["layer_ids"] for group in model.make_decoder_state_groups({}, {})}

    full_attention = [i for i, lt in enumerate(LAYER_TYPES) if lt == "full_attention"]
    linear_attention = [i for i, lt in enumerate(LAYER_TYPES) if lt == "linear_attention"]

    assert groups["paged_kv"] == full_attention
    assert groups["paged_indexer_key"] == full_attention, "the QSA indexer only exists on full-attention layers"
    assert groups["fixed_conv"] == linear_attention
    assert groups["fixed_recurrent"] == linear_attention
    assert groups["fixed_ple_conv"] == PLE_LAYERS
    assert groups["fixed_ple_tokens"] == PLE_LAYERS


def test_paged_manifest_registers_the_new_state_name_templates():
    model = _make_model()
    inputs, outputs = {}, {}

    model.make_decoder_state_groups(inputs, outputs)

    assert inputs["past_ple_conv_names"] == "past_key_values.%d.ple_conv_state"
    assert inputs["past_ple_token_names"] == "past_key_values.%d.ple_tokens"
    assert inputs["past_indexer_key_names"] == "past_key_values.%d.indexer_key"
    assert outputs["present_ple_conv_names"] == "present.%d.ple_conv_state"
    assert outputs["present_ple_token_names"] == "present.%d.ple_tokens"
    assert outputs["present_indexer_key_names"] == "present.%d.indexer_key"


def test_all_full_attention_model_still_emits_a_paged_kv_group():
    # `Qwen35TextModel` returns an empty manifest when there is no linear attention, but a
    # Qwen4Exp model still owns PLE and indexer state, so the KV group must be described.
    model = _make_model(layer_types=["full_attention"] * NUM_LAYERS)

    groups = model.make_decoder_state_groups({}, {})

    assert [group["kind"] for group in groups] == ["paged_kv", "fixed_ple_conv", "fixed_ple_tokens", "paged_indexer_key"]
    assert groups[0]["layer_ids"] == list(range(NUM_LAYERS))


def test_model_without_ple_or_indexer_keeps_the_qwen35_manifest():
    model = _make_model(indexer_head_dim=0, ple_layers=[])

    groups = model.make_decoder_state_groups({}, {})

    assert [group["kind"] for group in groups] == ["paged_kv", "fixed_conv", "fixed_recurrent"]


def test_non_paged_builder_emits_no_manifest():
    model = _make_model(use_paged_attention=False)

    assert model.make_decoder_state_groups({}, {}) == []


#####################################################################################
# Graph I/O registration
#####################################################################################


def test_indexer_key_cache_io_is_added_only_for_full_attention_layers():
    model = _make_model()
    model._add_ple_and_indexer_cache_io()

    for layer_id, layer_type in enumerate(LAYER_TYPES):
        key = f"past_state.{layer_id}.indexer_key"
        if layer_type == "full_attention":
            assert model.input_names[key] == f"past_key_values.{layer_id}.indexer_key"
            assert model.output_names[f"present_state.{layer_id}.indexer_key"] == f"present.{layer_id}.indexer_key"
        else:
            assert key not in model.input_names


def test_indexer_key_cache_shape_uses_the_indexer_geometry():
    model = _make_model()
    model._add_ple_and_indexer_cache_io()

    layer_id = LAYER_TYPES.index("full_attention")
    assert model.input_shapes[f"past_state.{layer_id}.indexer_key"] == [
        "batch_size",
        1,
        "total_sequence_length",
        64,
    ]
    assert model.input_types[f"past_state.{layer_id}.indexer_key"] == ir.DataType.FLOAT16


def test_ple_state_io_is_added_only_for_ple_layers():
    model = _make_model()
    model._add_ple_and_indexer_cache_io()

    for layer_id in range(NUM_LAYERS):
        present = f"present_state.{layer_id}.ple_conv"
        if layer_id in PLE_LAYERS:
            assert model.output_names[present] == f"present.{layer_id}.ple_conv_state"
        else:
            assert present not in model.output_names


def test_ple_conv_state_is_widened_and_dilation_sized():
    model = _make_model()
    model._add_ple_and_indexer_cache_io()

    # The conv runs on the concatenated hyper-connection streams and is dilated by ngram_size,
    # so the cached receptive field is (kernel - 1) * ngram_size.
    assert model.input_shapes["past_state.0.ple_conv"] == [
        "batch_size",
        HC_COUNT * HIDDEN,
        (PLE_CONV_KERNEL - 1) * NGRAM_SIZE,
    ]


def test_ple_token_history_is_int32_and_ngram_sized():
    model = _make_model()
    model._add_ple_and_indexer_cache_io()

    assert model.input_types["past_state.0.ple_tokens"] == ir.DataType.INT32
    assert model.output_types["present_state.0.ple_tokens"] == ir.DataType.INT32
    assert model.input_shapes["past_state.0.ple_tokens"] == ["batch_size", NGRAM_SIZE - 1]


def test_state_io_shapes_are_not_aliased_between_input_and_output():
    model = _make_model()
    model._add_ple_and_indexer_cache_io()

    for key in ("past_state.0.ple_conv", "past_state.0.ple_tokens"):
        present_key = key.replace("past_state", "present_state")
        assert model.input_shapes[key] == model.output_shapes[present_key]
        assert model.input_shapes[key] is not model.output_shapes[present_key]
