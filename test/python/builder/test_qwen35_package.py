# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import SimpleNamespace

import onnx_ir as ir
import pytest
from _builder_test_utils import load_builder_module


@pytest.mark.parametrize(
    ("component", "class_name"),
    [
        ("text", "Qwen35TextModel"),
        ("text", "Qwen35MoETextModel"),
        ("model", "Qwen35Model"),
        ("model", "Qwen35MoEModel"),
        ("mtp", "Qwen35MTPModel"),
        ("mtp", "Qwen35DenseMTPModel"),
    ],
)
def test_qwen35_package_preserves_public_class_exports(component, class_name):
    package = load_builder_module("qwen3_5")
    owner = load_builder_module(f"qwen3_5.{component}")
    legacy = load_builder_module("qwen")
    builder_class = getattr(owner, class_name)

    assert getattr(package, class_name) is builder_class
    assert getattr(legacy, class_name) is builder_class
    assert class_name in package.__all__
    assert builder_class.__module__ == f"models.builders.qwen3_5.{component}"


@pytest.mark.parametrize(
    ("class_name", "decoder_name", "mtp_name"),
    [
        ("Qwen35Model", "Qwen35TextModel", "Qwen35DenseMTPModel"),
        ("Qwen35MoEModel", "Qwen35MoETextModel", "Qwen35MTPModel"),
    ],
)
def test_qwen35_composite_dispatch_retains_constructor_globals(monkeypatch, class_name, decoder_name, mtp_name):
    package = load_builder_module("qwen3_5")
    builder_class = getattr(package, class_name)
    decoder, mtp = object(), object()
    monkeypatch.setitem(builder_class.__init__.__globals__, decoder_name, decoder)
    monkeypatch.setitem(builder_class.__init__.__globals__, mtp_name, mtp)
    model = builder_class.__new__(builder_class)

    assert model.get_decoder_model_class() is decoder
    assert model.get_mtp_model_class() is mtp


@pytest.mark.parametrize("class_name", ["Qwen35MTPModel", "Qwen35DenseMTPModel"])
def test_qwen35_mtp_hidden_states_output_uses_normalization_producer(monkeypatch, class_name):
    package = load_builder_module("qwen3_5")
    builder_class = getattr(package, class_name)
    model = builder_class.__new__(builder_class)
    model.model = SimpleNamespace(graph=ir.Graph(inputs=[], outputs=[], nodes=[]))
    model.values = {}
    model.node_names = set()
    model.layernorm_attrs = {}
    model.io_dtype = ir.DataType.FLOAT
    model.mtp_weights = SimpleNamespace(layers=[object()], norm=object(), lm_head=object())
    hidden_shape = [1, 2, 4]
    placeholder = model.make_value("hidden_states_out", model.io_dtype, hidden_shape)

    monkeypatch.setattr(model, "make_inputs_and_outputs", lambda: None)
    monkeypatch.setattr(model, "load_mtp_weights", lambda path: None)
    monkeypatch.setattr(model, "make_preprocessing_nodes", lambda: None)
    monkeypatch.setattr(model, "make_mtp_input_projection", lambda: "projected")
    monkeypatch.setattr(model, "make_layer", lambda *args: None)
    monkeypatch.setattr(model, "make_postprocessing_nodes", lambda: None)
    monkeypatch.setattr(model, "make_hidden_state_shape", lambda: hidden_shape)

    def make_layernorm(*args, **kwargs):
        model.make_node("Sigmoid", ["projected"], ["normalized"], name="normalization")
        model.layernorm_attrs["output_0"] = "normalized"

    def make_lm_head(weight):
        model.make_node("Relu", ["normalized"], ["logits"], name="lm_head")

    monkeypatch.setattr(model, "make_layernorm", make_layernorm)
    monkeypatch.setattr(model, "make_lm_head", make_lm_head)
    model.make_model("unused-checkpoint")

    graph = model.model.graph
    hidden = graph.outputs[0]
    assert hidden.name == "hidden_states_out"
    assert hidden is model.values["hidden_states_out"]
    assert hidden is not placeholder
    assert hidden.producer() is graph[0]
    assert graph[1].inputs[0] is hidden
    assert "normalized" not in model.values
    assert hidden.dtype == model.io_dtype
    assert list(hidden.shape) == hidden_shape
    assert [node.op_type for node in graph] == ["Sigmoid", "Relu"]
