# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from models.builders.base import Model
from models.builders.expansions.engine import Engine
from models.builders.expansions.engram import Qwen38Engram as PagedEngram
from models.builders.expansions.mtp import Qwen35MTP, Qwen38MTP
from models.builders.expansions.qwen3_5 import Qwen35, Qwen35MoE
from models.builders.expansions.qwen3_8 import Qwen38, Qwen38Graph, Qwen38Vision
from models.builders.qwen3_5 import Qwen35MoETextModel, Qwen35MTPModel, Qwen35TextModel
from models.builders.qwen3_8 import Qwen4ExpMTPTextModel, Qwen4ExpTextModel, Qwen4ExpVisionModel, Qwen38Engram


@pytest.mark.parametrize(
    ("model_class", "expansion_class", "method"),
    [
        (Model, Engine, "make_paged_attention"),
        (Model, Engine, "make_paged_inputs_init"),
        (Model, Engine, "make_paged_outputs_init"),
        (Model, Engine, "make_paged_engine_config"),
        (Model, Engine, "make_paged_mrope_inputs"),
        (Model, Engine, "make_selected_hidden_rows"),
        (Model, Engine, "make_paged_rotary_multi_cache_subgraph"),
        (Model, Engine, "make_varlen_causal_conv_with_state"),
        (Model, Engine, "make_varlen_gated_delta_net"),
        (Qwen35TextModel, Qwen35, "make_attention_input_proj"),
        (Qwen35TextModel, Qwen35, "make_qwen_gated_delta_net"),
        (Qwen35TextModel, Qwen35, "make_linear_attention_output_proj"),
        (Qwen35MoETextModel, Qwen35MoE, "make_moe_router"),
        (Qwen35MoETextModel, Qwen35MoE, "make_shared_expert"),
        (Qwen35MTPModel, Qwen35MTP, "make_mtp_input_projection"),
        (Qwen4ExpTextModel, Qwen38, "make_attention_input_proj"),
        (Qwen4ExpTextModel, Qwen38, "make_linear_attention_qkv_z_proj"),
        (Qwen4ExpTextModel, Qwen38, "make_linear_attention_a_b_proj"),
        (Qwen4ExpTextModel, Qwen38, "make_qwen_sparse_attention"),
        (Qwen4ExpTextModel, Qwen38, "make_hyper_connection_mix"),
        (Qwen4ExpTextModel, Qwen35, "make_qwen_gated_delta_net"),
        (Qwen4ExpVisionModel, Qwen38Graph, "make_linear"),
        (Qwen4ExpVisionModel, Qwen38Vision, "make_position_embeddings"),
        (Qwen4ExpVisionModel, Qwen38Vision, "make_attention"),
        (Qwen4ExpMTPTextModel, Qwen38MTP, "make_mtp_input_projection"),
    ],
)
def test_model_uses_architecture_expansion(model_class, expansion_class, method):
    assert method not in model_class.__dict__
    assert getattr(model_class, method) is getattr(expansion_class, method)


def test_qwen38_super_dispatch_preserves_qwen35_projection_fallback():
    model = object.__new__(Qwen4ExpTextModel)

    assert super(Qwen38, model).make_attention_input_proj.__func__ is Qwen35.make_attention_input_proj
    assert super(Qwen38, model).make_moe_preprocessing.__func__ is Qwen35MoE.make_moe_preprocessing
    assert super(Qwen35, model).make_attention_input_proj.__func__ is Model.make_attention_input_proj


@pytest.mark.parametrize("expansion_class", [Qwen38Graph, Qwen38Vision])
def test_qwen38_auxiliary_expansions_live_in_architecture_module(expansion_class):
    assert expansion_class.__module__ == "models.builders.expansions.qwen3_8"


@pytest.mark.parametrize("paged", [False, True])
def test_qwen38_engram_dispatches_to_dense_builder_or_paged_expansion(monkeypatch, paged):
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
    monkeypatch.setattr(PagedEngram, "make_ple", packed)
    kind = "paged" if paged else "dense"
    assert model.make_ple(2, ple, "hidden_states") == f"{kind}_output"
    assert calls == [(kind, model, 2, ple, "hidden_states")]
    assert Qwen38Engram.__module__ == "models.builders.qwen3_8"
    assert PagedEngram.__module__ == "models.builders.expansions.engram"
