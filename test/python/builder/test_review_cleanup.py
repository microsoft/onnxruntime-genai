# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
from pathlib import Path

import onnx_ir as ir
import pytest
import torch
from transformers import LlamaConfig

from models.builders.base import Model


def make_model(extra_options=None):
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
    )
    return Model(config, ir.DataType.FLOAT16, ir.DataType.INT4, "cuda", "", extra_options or {})


@pytest.mark.parametrize("qmoe", [-1, 0, 1])
@pytest.mark.parametrize("matmul", [0, 1, 2])
def test_packing_layout_uses_flags_not_build_host(monkeypatch, qmoe, matmul):
    def reject_host_query(*args, **kwargs):
        pytest.fail("Weight layout must not query build-host GPU capabilities")

    monkeypatch.setattr(torch.cuda, "get_device_capability", reject_host_query)
    monkeypatch.setattr(torch.cuda, "get_device_properties", reject_host_query)
    model = make_model({"qmoe_weights_prepacked": qmoe, "matmulnbits_weights_prepacked": matmul})
    assert model.matmul_attrs["weights_prepacked"] == matmul
    assert model.moe_attrs["weights_prepacked"] == qmoe


def test_reshape_always_uses_named_intermediate_output():
    model = make_model()
    model.make_value("hidden", ir.DataType.FLOAT16, [1, 2, 8])
    model.make_reshape("/reshape", ["hidden", "/model/constants/INT64/[1, 2, 8]"], ir.DataType.FLOAT16, [1, 2, 8])
    reshape = next(node for node in model.model.graph if node.op_type == "Reshape")
    assert reshape.outputs[0].name == "/reshape/output_0"
    assert model.values["/reshape/output_0"] is reshape.outputs[0]


def test_paged_io_and_mrope_helpers_are_inlined():
    for name in (
        "make_paged_bindings_init",
        "make_paged_inputs_init",
        "make_paged_outputs_init",
        "make_paged_mrope_inputs",
        "make_paged_mrope_output",
        "is_mlp_gate_up_fusion_supported",
        "make_offset_rmsnorm",
        "make_expansion_constant",
    ):
        assert not hasattr(Model, name)
    path = Path(__file__).parents[3] / "src/python/py/models/builders/base.py"
    model_class = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Model"
    )
    methods = [node.name for node in model_class.body if isinstance(node, ast.FunctionDef)]
    assert methods.index("make_inputs_init") == methods.index("make_cache_names") + 1
    assert methods.index("make_outputs_init") == methods.index("make_inputs_init") + 1


def test_mlp_fusion_has_no_builder_configuration_flag():
    assert "fuse_gate_up" not in make_model().mlp_attrs
