# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest

from models.builders.qwen import Qwen35MTPModel

HIDDEN_SIZE = 8


def _mtp_head(use_paged_attention):
    model = object.__new__(Qwen35MTPModel)
    model.io_dtype = ir.DataType.FLOAT
    model.onnx_dtype = ir.DataType.FLOAT
    model.hidden_size = HIDDEN_SIZE
    model.use_paged_attention = use_paged_attention
    model.hidden_rows_dim = "num_tokens"
    model.values = {}
    model.node_names = set()
    model.initializers = {}
    model.layernorm_attrs = {"add_offset": 0, "epsilon": 1e-6}

    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21}, name="mtp_shape_test")
    model.model = ir.Model(graph, ir_version=10)
    return model, graph


@pytest.mark.parametrize(
    "use_paged_attention, expected",
    [
        (False, ["batch_size", "sequence_length", HIDDEN_SIZE]),
        (True, ["num_tokens", HIDDEN_SIZE]),
    ],
)
def test_input_projection_norms_follow_the_model_token_layout(use_paged_attention, expected):
    model, graph = _mtp_head(use_paged_attention)
    model.make_value("root", model.io_dtype, shape=expected)
    model.input_names = {"hidden_states": "root"}
    model.layernorm_attrs["add_offset"] = 1.0
    norm = SimpleNamespace(weight=np.arange(HIDDEN_SIZE, dtype=np.float32) - 1.0)
    model.mtp_weights = SimpleNamespace(pre_fc_norm_embedding=norm, pre_fc_norm_hidden=norm, fc=object())
    model.make_mtp_embedding = lambda basename: "embedding"
    model.make_matmul = lambda weight, name, root_input: name

    model.make_mtp_input_projection()

    for location in ("pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        output = f"/model/mtp/{location}/output_0"
        assert [str(dim) for dim in model.values[output].shape] == [str(dim) for dim in expected]
        np.testing.assert_array_equal(
            graph.initializers[f"model.mtp.{location}.weight"].const_value.numpy(), np.arange(HIDDEN_SIZE)
        )
    assert all(graph.node(index).domain == "" for index in (0, 1))
    assert all(graph.node(index).op_type == "SimplifiedLayerNormalization" for index in (0, 1))


@pytest.mark.parametrize(
    "use_paged_attention, expected",
    [
        (False, ["batch_size", "sequence_length", 2 * HIDDEN_SIZE]),
        (True, ["num_tokens", 2 * HIDDEN_SIZE]),
    ],
)
def test_input_projection_concat_widens_only_the_feature_axis(use_paged_attention, expected):
    model, _ = _mtp_head(use_paged_attention)

    # The MTP head concatenates the normalized embedding with the target's hidden states, so the
    # concat is twice as wide on the last axis but keeps the surrounding token layout.
    assert model.make_hidden_state_shape(last_dim=2 * model.hidden_size) == expected


def test_paged_head_declares_no_three_dimensional_hidden_state():
    model, _ = _mtp_head(use_paged_attention=True)

    assert len(model.make_hidden_state_shape()) == 2
    assert len(model.make_hidden_state_shape(last_dim=2 * model.hidden_size)) == 2
