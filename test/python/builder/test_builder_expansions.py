# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
from pathlib import Path

import pytest

from models.builders.base import Model
from models.builders.expansions.standard import Standard
from models.builders.expansions.webgpu import WebGPU
from models.builders.qwen import Qwen3VLTextModel
from models.builders.qwen3_5 import Qwen35MoETextModel, Qwen35MTPModel, Qwen35TextModel
from models.builders.qwen3_8 import Qwen4ExpMTPTextModel, Qwen4ExpTextModel, Qwen4ExpVisionModel, Qwen38Engram
from models.builders.qwen3_8_graph import Qwen4ExpGraphModel


@pytest.mark.parametrize(
    ("model_class", "methods"),
    [
        (
            Model,
            [
                "uses_windowed_kv_cache",
                "make_decoder_state_groups",
                "make_paged_key_value_state_group",
                "make_key_value_cache_names",
                "is_windowed_paged_layer",
                "has_windowed_paged_layers",
                "make_key_value_cache_shape",
            ],
        ),
        (
            Model,
            ["make_causal_conv_state_at_last_real_token", "make_varlen_causal_conv_with_state", "make_gated_rms_norm"],
        ),
        (
            Model,
            [
                "make_mrotary_embedding",
                "make_rotary_embedding_multi_cache",
                "make_paged_rotary_multi_cache_subgraph",
                "make_qk_norm",
            ],
        ),
        (Model, ["make_gated_delta_net", "make_varlen_gated_delta_net", "make_linear_attention_gate"]),
        (Model, ["make_sparse_attention", "make_paged_attention", "make_attention"]),
        (Model, ["prunes_hidden_rows", "make_selected_hidden_rows", "make_last_real_token_position"]),
        (Model, ["make_cache_names", "make_inputs_init"]),
        (
            Qwen35TextModel,
            [
                "make_attention_input_proj",
                "make_attention_output_proj",
                "make_qwen_gated_delta_net",
                "make_linear_attention_input_proj",
                "make_linear_attention_output_proj",
                "make_decoder_state_groups",
            ],
        ),
    ],
)
def test_builder_methods_retain_original_grouping(model_class, methods):
    positions = [getattr(model_class, name).__code__.co_firstlineno for name in methods]
    assert positions == sorted(positions)


@pytest.mark.parametrize("capture", [False, True])
def test_webgpu_graph_capture_override_lives_in_webgpu(capture):
    model = object.__new__(Model)
    model.ep = "webgpu"
    model.extra_options = {"enable_webgpu_graph": capture}

    model.make_ep_expansions_init()

    expected = (
        WebGPU.make_attention_mask_graph_capture_reformatting_for_gqa
        if capture
        else Model.make_attention_mask_reformatting_for_gqa
    )
    assert model.make_attention_mask_reformatting_for_gqa.__func__ is expected
    assert (
        WebGPU.make_attention_mask_graph_capture_reformatting_for_gqa.__module__ == "models.builders.expansions.webgpu"
    )
    assert not hasattr(Standard, "make_attention_mask_reformatting_for_gqa")


@pytest.mark.parametrize(
    ("model_class", "methods"),
    [
        (Qwen3VLTextModel, ["make_qk_norm"]),
        (Qwen35TextModel, ["make_attention_input_proj", "make_attention_output_proj"]),
        (Qwen35MoETextModel, ["make_moe_preprocessing", "make_moe_router", "make_moe_subgraph", "make_shared_expert"]),
        (
            Qwen4ExpTextModel,
            [
                "make_moe_preprocessing",
                "make_moe_expert_names",
                "make_moe_op",
                "make_attention_input_proj",
                "make_gated_rms_norm",
            ],
        ),
        (Qwen4ExpGraphModel, ["make_cast"]),
        (Qwen4ExpVisionModel, ["make_attention", "make_gelu"]),
    ],
)
def test_qwen_overrides_are_defined_directly_in_builders(model_class, methods):
    for method in methods:
        assert method in model_class.__dict__
        assert getattr(model_class, method).__module__ == model_class.__module__


@pytest.mark.parametrize(
    ("model_class", "methods"),
    [
        (
            Model,
            [
                "make_paged_attention",
                "make_paged_engine_config",
                "make_varlen_gated_delta_net",
                "make_selected_counts",
                "get_mrope_owners",
            ],
        ),
        (
            Qwen35TextModel,
            ["make_qwen_gated_delta_net", "make_linear_attention_output_proj", "make_linear_attention_qkv_z_proj"],
        ),
        (Qwen35MTPModel, ["make_mtp_input_projection", "make_mtp_embedding"]),
        (
            Qwen4ExpTextModel,
            [
                "make_linear_attention_qkv_z_proj",
                "make_qwen_sparse_attention",
                "make_hyper_connection_mix",
                "make_ple_embedding",
                "make_ple_gate",
            ],
        ),
        (Qwen4ExpGraphModel, ["make_linear", "make_layer_norm", "make_binary"]),
        (Qwen4ExpVisionModel, ["make_position_embeddings", "make_rotary_embeddings", "make_frame_attention_mask"]),
        (Qwen4ExpMTPTextModel, ["make_mtp_input_projection"]),
    ],
)
def test_non_overriding_helpers_are_defined_in_builders(model_class, methods):
    for method in methods:
        assert method in model_class.__dict__
        assert getattr(model_class, method).__module__.startswith("models.builders.")
        assert ".expansions." not in getattr(model_class, method).__module__


def test_standard_helpers_live_with_their_expansions():
    for name in (
        "make_skip_simplified_layer_norm",
        "make_skip_layer_norm",
        "make_simplified_layer_norm",
        "make_mrope_positions",
        "make_mrope_cache",
        "make_mrope_rotation",
        "make_mrope_output",
        "make_l2_normalize",
    ):
        assert name in Standard.__dict__
        assert name not in Model.__dict__
    assert Qwen35TextModel.make_l2_normalize is Standard.make_l2_normalize
    assert not hasattr(Model, "make_offset_rmsnorm")
    assert not hasattr(Model, "make_expansion_constant")
    assert not any(name.endswith("_expansion") for name in Model.__dict__)


@pytest.mark.parametrize("ep", ["cuda", "cpu", "webgpu", "dml", "trt-rtx"])
def test_hyper_connection_overrides_are_provider_specific(ep):
    model = object.__new__(Qwen4ExpTextModel)
    model.ep = ep
    model.extra_options = {}
    model.make_ep_expansions_init()
    for name in (
        "make_branchwise_rms_norm",
        "make_scaled_silu",
        "make_hyper_connection_pre_mix",
        "make_hyper_connection_post_mix",
    ):
        expected = Standard if ep in {"dml", "trt-rtx"} else Qwen4ExpTextModel
        assert getattr(model, name).__func__ is getattr(expected, name)


def test_expansion_overrides_have_existing_builder_counterparts():
    builders = Path(__file__).parents[3] / "src/python/py/models/builders"
    names = {
        method.name
        for path in builders.glob("*.py")
        for cls in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
    }
    names.update(
        {
            "make_skip_simplified_layer_norm",
            "make_skip_layer_norm",
            "make_simplified_layer_norm",
            "make_mrope_positions",
            "make_mrope_cache",
            "make_mrope_rotation",
            "make_mrope_output",
            "make_l2_normalize",
        }
    )
    for path in (builders / "expansions").glob("*.py"):
        for cls in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(cls, ast.ClassDef):
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef):
                        name = (
                            "make_attention_mask_reformatting_for_gqa"
                            if cls.name == "WebGPU"
                            and method.name == "make_attention_mask_graph_capture_reformatting_for_gqa"
                            else method.name
                        )
                        assert name in names, f"{path.name}:{cls.name}.{method.name}"
                    elif isinstance(method, ast.Assign) and isinstance(method.value, ast.Attribute):
                        for target in method.targets:
                            assert isinstance(target, ast.Name)
                            assert target.id in names, f"{path.name}:{cls.name}.{target.id}"


def test_qwen38_super_dispatch_preserves_qwen35_projection_fallback():
    model = object.__new__(Qwen4ExpTextModel)

    assert (
        super(Qwen4ExpTextModel, model).make_attention_input_proj.__func__ is Qwen35TextModel.make_attention_input_proj
    )
    assert super(Qwen4ExpTextModel, model).make_moe_preprocessing.__func__ is Qwen35MoETextModel.make_moe_preprocessing
    assert super(Qwen35TextModel, model).make_attention_input_proj.__func__ is Model.make_attention_input_proj
    assert (
        super(Qwen4ExpTextModel, model).make_linear_attention_qkv_z_proj.__func__
        is Qwen35TextModel.make_linear_attention_qkv_z_proj
    )


def test_qwen_expansion_modules_are_removed():
    expansions = Path(__file__).parents[3] / "src/python/py/models/builders/expansions"
    assert not any((expansions / name).exists() for name in ("qwen.py", "qwen3_5.py", "qwen3_8.py"))


@pytest.mark.parametrize("paged", [False, True])
def test_qwen38_engram_dispatches_to_dense_or_paged_builder(monkeypatch, paged):
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = paged
    ple = object()
    calls = []

    def dense(self, layer_id, weights, root_input):
        calls.append(("dense", self, layer_id, weights, root_input))
        return "dense_output"

    def packed(self, layer_id, weights, root_input):
        calls.append(("paged", self, layer_id, weights, root_input))
        return "paged_output"

    monkeypatch.setattr(Qwen38Engram, "make_ple", dense)
    monkeypatch.setattr(Qwen4ExpTextModel, "make_paged_ple", packed)
    kind = "paged" if paged else "dense"
    assert model.make_ple(2, ple, "hidden_states") == f"{kind}_output"
    assert calls == [(kind, model, 2, ple, "hidden_states")]
    assert Qwen38Engram.__module__ == "models.builders.qwen3_8"
    assert "make_paged_ple" in Qwen4ExpTextModel.__dict__
