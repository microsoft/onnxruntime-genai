# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Builder-only coverage; no model download or execution provider is required."""

from types import SimpleNamespace

import pytest
from _builder_test_utils import load_builder_module

Qwen35TextModel = load_builder_module("qwen").Qwen35TextModel
Qwen4ExpTextModel = load_builder_module("qwen").Qwen4ExpTextModel


@pytest.mark.parametrize("model_type", [Qwen35TextModel, Qwen4ExpTextModel])
@pytest.mark.parametrize("ep", ["cpu", "cuda", "dml", "trt-rtx", "webgpu"])
def test_default_preserves_kernel_chunk(model_type, ep):
    model = model_type.__new__(model_type)
    model.ep = ep
    model.layer_types = ["linear_attention"]
    model.make_gated_delta_net_init({})
    assert model.gated_delta_net_attrs == {"chunk_size": 64}


@pytest.mark.parametrize("value", [16, "16", 64, "64"])
def test_webgpu_chunk_selection(value):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.ep = "webgpu"
    model.layer_types = ["linear_attention"]
    model.make_gated_delta_net_init({"webgpu_gdn_chunk_size": value})
    attrs = model.make_gated_delta_net_attributes(model.gated_delta_net_attrs)
    assert attrs["chunk_size"] == int(value)
    assert attrs["update_rule"] == "gated_delta"


@pytest.mark.parametrize("value", [True, False, None, 0, -16, 32, 16.0, "16.0", "bad"])
def test_invalid_chunk_rejected(value):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.ep = "webgpu"
    model.layer_types = ["linear_attention"]
    with pytest.raises(ValueError, match="must be 16 or 64"):
        model.make_gated_delta_net_init({"webgpu_gdn_chunk_size": value})


@pytest.mark.parametrize("ep", ["cpu", "cuda", "dml", "trt-rtx"])
def test_override_rejected_on_other_providers(ep):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.ep = ep
    model.layer_types = ["linear_attention"]
    with pytest.raises(ValueError, match="requires the WebGPU"):
        model.make_gated_delta_net_init({"webgpu_gdn_chunk_size": 16})


def test_override_requires_linear_attention():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.ep = "webgpu"
    model.layer_types = ["full_attention"]
    with pytest.raises(ValueError, match="requires a Qwen linear-attention layer"):
        model.make_gated_delta_net_init({"webgpu_gdn_chunk_size": 16})


@pytest.mark.parametrize("class_name", ["Qwen35MTPModel", "Qwen35DenseMTPModel", "Qwen4ExpMTPTextModel"])
def test_target_chunk_policy_not_forwarded_to_mtp(monkeypatch, class_name):
    builder_class = getattr(load_builder_module("qwen"), class_name)
    options = {"webgpu_gdn_chunk_size": 16}
    config = SimpleNamespace(text_config=SimpleNamespace())

    def initialize_parent(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        assert "webgpu_gdn_chunk_size" not in extra_options
        raise RuntimeError("parent construction reached")

    # Dense Qwen MTP inherits the MoE MTP constructor, which calls the text parent.
    parent = load_builder_module("qwen").Qwen35MoETextModel
    if class_name == "Qwen4ExpMTPTextModel":
        parent = Qwen4ExpTextModel
    monkeypatch.setattr(parent, "__init__", initialize_parent)
    with pytest.raises(RuntimeError, match="parent construction reached"):
        builder_class(config, None, None, "webgpu", "", options)
    assert options == {"webgpu_gdn_chunk_size": 16}
