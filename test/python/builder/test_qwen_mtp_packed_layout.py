# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

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
def test_offset_rmsnorm_follows_the_model_token_layout(use_paged_attention, expected):
    model, _ = _mtp_head(use_paged_attention)
    model.make_value("root", model.io_dtype, shape=expected)

    output = model.make_offset_rmsnorm("/model/mtp/pre_fc_norm_hidden", "root", np.zeros(HIDDEN_SIZE, dtype=np.float32))

    assert [str(dim) for dim in model.values[output].shape] == [str(dim) for dim in expected]


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
