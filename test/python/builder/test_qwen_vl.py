# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
qwen_module = _load_builder_module("qwen")
Model = base_module.Model
Qwen25VLTextModel = qwen_module.Qwen25VLTextModel
Qwen3VLTextModel = qwen_module.Qwen3VLTextModel


def _initialize_qwen_model(monkeypatch, model_class):
    def initialize_base(self, *_args, **_kwargs):
        self.layernorm_attrs = {
            "cast": {
                "use_fp32": False,
                "root_input": False,
                "skip_input": False,
                "output_0": False,
                "output_3": False,
            }
        }
        self.rope_attrs = {
            "mrope_layout": 0,
            "cast": {"use_fp32": False, "root_input": False, "output_0": False},
        }
        self.attention_attrs = {"q_norm": False, "k_norm": False}

    monkeypatch.setattr(Model, "__init__", initialize_base)
    return model_class(types.SimpleNamespace(), ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", "", {})


@pytest.mark.parametrize(
    "model_class,expected_layout,uses_qk_norm,casts_rope_input,casts_layernorm_skip",
    [
        (Qwen25VLTextModel, 0, False, True, True),
        (Qwen3VLTextModel, 1, True, False, False),
    ],
)
def test_qwen_vl_configures_variant_specific_mrope(
    monkeypatch, model_class, expected_layout, uses_qk_norm, casts_rope_input, casts_layernorm_skip
):
    model = _initialize_qwen_model(monkeypatch, model_class)

    assert model.rope_attrs["mrope_layout"] == expected_layout
    assert model.rope_attrs["cast"] == {
        "use_fp32": True,
        "root_input": casts_rope_input,
        "output_0": True,
    }
    assert model.layernorm_attrs["cast"]["use_fp32"]
    assert model.layernorm_attrs["cast"]["output_3"] is casts_layernorm_skip
    assert model.attention_attrs["q_norm"] is uses_qk_norm
    assert model.attention_attrs["k_norm"] is uses_qk_norm


@pytest.mark.parametrize("model_class", [Qwen25VLTextModel, Qwen3VLTextModel])
def test_qwen_vl_uses_separate_qkv_and_mrope(monkeypatch, model_class):
    model = _initialize_qwen_model(monkeypatch, model_class)

    assert not model.is_packed_matmul_supported()
    assert not model.is_fused_rope_supported()


@pytest.mark.parametrize("model_class", [Qwen25VLTextModel, Qwen3VLTextModel])
def test_qwen_vl_decoder_uses_3d_position_ids(monkeypatch, model_class):
    model = model_class.__new__(model_class)
    model.input_shapes = {"position_ids": ["batch_size", "sequence_length"]}
    monkeypatch.setattr(Model, "make_inputs_and_outputs", lambda self: None)

    model.make_inputs_and_outputs()

    assert model.input_shapes["position_ids"] == [3, "batch_size", "sequence_length"]


@pytest.mark.parametrize(
    "layout,sections",
    [
        (0, [16, 24, 24]),
        (1, [24, 20, 20]),
    ],
)
def test_qwen_vl_emits_layout_specific_mrotary_embedding(layout, sections):
    model = Model.__new__(Model)
    model.head_size = 128
    model.rope_attrs = {
        "interleaved": 0,
        "rotary_embedding_dim": 0,
        "mrope_section": sections,
        "mrope_layout": layout,
    }
    model.nodes = []
    model.values = []

    def make_node(op_type, inputs, outputs, name, domain, **attributes):
        model.nodes.append(
            {
                "op_type": op_type,
                "inputs": inputs,
                "outputs": outputs,
                "name": name,
                "domain": domain,
                "attributes": attributes,
            }
        )

    model.make_node = make_node
    model.make_value = lambda name, dtype, shape: model.values.append((name, dtype, shape))

    model.make_mrotary_embedding(
        "/model/layers.0/attn/q_rotary/MRotaryEmbedding",
        "q",
        "q_rotated",
        position_ids="position_ids",
        cos_cache_name="cos_cache",
        sin_cache_name="sin_cache",
        num_heads=16,
        dtype=ir.DataType.FLOAT,
    )

    node = model.nodes[0]
    assert node["op_type"] == "MRotaryEmbedding"
    assert node["domain"] == "com.microsoft"
    assert node["inputs"] == ["q", "position_ids", "cos_cache", "sin_cache"]
    assert node["attributes"]["num_heads"] == 16
    assert node["attributes"]["mrope_section"] == sections
    assert node["attributes"]["mrope_layout"] == layout
    assert model.values == [("q_rotated", ir.DataType.FLOAT, ["batch_size", "sequence_length", 2048])]


def test_qwen3_vl_keeps_qk_norm_outputs_in_fp32(monkeypatch):
    model = Qwen3VLTextModel.__new__(Qwen3VLTextModel)
    model.layernorm_attrs = {"cast": {"output_0": True}}
    model.attention_attrs = {"q_path": "q_norm", "k_path": "k_norm"}
    model.values = {
        "q_norm": types.SimpleNamespace(dtype=ir.DataType.FLOAT16),
        "k_norm": types.SimpleNamespace(dtype=ir.DataType.FLOAT16),
    }
    observed_cast_setting = []

    def make_qk_norm(self, *_args):
        observed_cast_setting.append(self.layernorm_attrs["cast"]["output_0"])

    monkeypatch.setattr(Model, "make_qk_norm", make_qk_norm)

    model.make_qk_norm(0, object())

    assert observed_cast_setting == [False]
    assert model.values["q_norm"].dtype == ir.DataType.FLOAT
    assert model.values["k_norm"].dtype == ir.DataType.FLOAT
    assert model.layernorm_attrs["cast"]["output_0"]
