# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""state_window=0 must not be written as a node attribute.

The default extra option is 0, which means the legacy unwindowed conv/recurrent
state I/O. ONNX Runtime 1.25.1 through 1.28 reject an unrecognized state_window
attribute even when its value is 0, so the builder should omit it unless a
window was requested.
"""

from __future__ import annotations

import pytest
from _builder_test_utils import load_builder_module

base_module = load_builder_module("base")
Model = base_module.Model


class _Recorder:
    make_causal_conv_with_state = Model.make_causal_conv_with_state
    make_linear_attention = Model.make_linear_attention

    def __init__(self, state_window=0):
        self.context_length_attrs = {"state_window": state_window}
        self.io_dtype = None
        self.linear_value_dim = 16
        self.output_shapes = {"present.conv": ["batch_size", 4, 3]}
        self.nodes = []

    def make_node(self, op_type, inputs, outputs, *, name, domain="", **attributes):
        self.nodes.append({"op_type": op_type, "name": name, "domain": domain, "attributes": attributes})

    def make_value(self, *args, **kwargs):
        pass


def _emit_causal_conv(state_window):
    model = _Recorder(state_window)
    model.make_causal_conv_with_state(
        "/model/layers.0/conv/CausalConvWithState",
        root_input="x",
        weight="w",
        bias="b",
        past_conv_state="past",
        present_conv_state="present",
        channels=4,
    )
    return model.nodes


def _emit_linear_attention(state_window):
    model = _Recorder(state_window)
    model.make_linear_attention(
        "/model/layers.0/linear_attn/LinearAttention",
        q_path="q",
        k_path="k",
        v_path="v",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="decay",
        beta="beta",
        q_num_heads=2,
        kv_num_heads=2,
    )
    return model.nodes


@pytest.mark.parametrize(
    "emit, op_type", [(_emit_causal_conv, "CausalConvWithState"), (_emit_linear_attention, "LinearAttention")]
)
def test_zero_state_window_is_not_written(emit, op_type):
    nodes = emit(0)
    assert len(nodes) == 1
    assert nodes[0]["op_type"] == op_type
    assert "state_window" not in nodes[0]["attributes"]


@pytest.mark.parametrize(
    "emit, op_type", [(_emit_causal_conv, "CausalConvWithState"), (_emit_linear_attention, "LinearAttention")]
)
def test_positive_state_window_is_written(emit, op_type):
    nodes = emit(8)
    assert len(nodes) == 1
    assert nodes[0]["op_type"] == op_type
    assert nodes[0]["attributes"]["state_window"] == 8
