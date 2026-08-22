# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import onnx_ir as ir
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.base import Model
from models.builders.qwen import Qwen35TextModel


def _record(model, calls, name):
    def call(self, *args, **kwargs):
        calls.append((name, args, kwargs))

    return MethodType(call, model)


def test_qwen_gated_delta_net_uses_raw_gates_and_windowed_checkpoints():
    model = object.__new__(Qwen35TextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.linear_key_dim = 8
    model.linear_value_dim = 8
    model.linear_num_key_heads = 2
    model.linear_num_value_heads = 2
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 4
    model._state_window = 4
    model._state_window_dims = [4]

    calls = []
    for name in ("make_node", "make_value", "make_reshape", "make_cast", "make_initializer", "make_gated_delta_net"):
        setattr(model, name, _record(model, calls, name))

    output = model._make_gated_delta_net(
        0,
        SimpleNamespace(A_log=torch.ones(2), dt_bias=torch.ones(2)),
        "conv_output",
        "/beta",
        "/decay",
    )

    assert output == "/model/layers.0/linear_attn/gdn_out/Reshape/output_0"
    casts = [call for call in calls if call[0] == "make_cast"]
    assert [(args[1], args[2]) for _, args, _ in casts] == [
        ("/decay/output_0", ir.DataType.FLOAT),
        ("/beta/output_0", ir.DataType.FLOAT),
    ]

    _, args, kwargs = next(call for call in calls if call[0] == "make_gated_delta_net")
    assert args == ("/model/layers.0/linear_attn/GatedDeltaNet",)
    assert kwargs["initial_state"] == "past_key_values.0.recurrent_state"
    assert kwargs["final_state"] == ""
    assert kwargs["checkpoints"] == "present.0.recurrent_state"
    assert kwargs["state_checkpoints"] == 4
    assert kwargs["state_shape"] == [4, "batch_size", 2, 4, 4]
    assert kwargs["gate_activation"] == "qwen"
    assert kwargs["beta_activation"] == "sigmoid"
    assert kwargs["qk_l2_norm"] == 1
    assert kwargs["a_log"] == "model.layers.0.linear_attn.A_log"
    assert kwargs["dt_bias"] == "model.layers.0.linear_attn.dt_bias"


def test_gated_delta_net_helper_emits_checkpoint_only_state_output():
    model = object.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    calls = []
    model.make_node = _record(model, calls, "make_node")
    model.make_value = _record(model, calls, "make_value")
    state_shape = [4, "batch_size", 2, 4, 4]

    model.make_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        decay="decay",
        beta="beta",
        initial_state="past",
        final_state="",
        checkpoints="present",
        state_checkpoints=4,
        a_log="a_log",
        dt_bias="dt_bias",
        update_rule="gated_delta",
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        scale=0.0,
        output_shape=["batch_size", "sequence_length", 2, 4],
        state_shape=state_shape,
        checkpoints_shape=state_shape,
    )

    _, args, kwargs = calls[0]
    assert args == ("GatedDeltaNet",)
    assert kwargs["inputs"] == ["q", "k", "v", "", "decay", "beta", "past", "a_log", "dt_bias"]
    assert kwargs["outputs"] == ["/gdn/output_0", "", "present"]
    assert kwargs["domain"] == "com.microsoft"
    assert kwargs["state_checkpoints"] == 4
    assert calls[-1] == ("make_value", ("present", ir.DataType.FLOAT), {"shape": state_shape})
