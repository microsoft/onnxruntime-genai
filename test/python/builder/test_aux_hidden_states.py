# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(MODELS_DIR))


def _load_base_module():
    sys.modules.setdefault("models", types.ModuleType("models"))
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]

    spec = importlib.util.spec_from_file_location("models.builders.base", BUILDERS_DIR / "base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.builders.base"] = module
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
Model = base_module.Model


def _outputs_model(aux_option, *, num_layers=8, use_paged_attention=False):
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.layer_types = ["full_attention"]
    model.num_layers = num_layers
    model.hidden_size = 16
    model.vocab_size = 32
    model.num_kv_heads = 2
    model.head_size = 8
    model.use_paged_attention = use_paged_attention
    model.output_names = {"hidden_states": "hidden_states", "logits": "logits"}
    model.output_types = {"hidden_states": ir.DataType.FLOAT16, "logits": ir.DataType.FLOAT16}
    model.output_shapes = {
        "hidden_states": ["batch_size", "sequence_length", model.hidden_size],
        "logits": ["batch_size", "sequence_length", model.vocab_size],
    }
    model.extra_options = {} if aux_option is None else {"aux_hidden_state_layers": aux_option}
    return model


def test_absent_option_adds_no_auxiliary_output():
    model = _outputs_model(None)

    model.make_outputs_init()

    assert model.aux_hidden_state_layers == []
    assert "aux_hidden_states" not in model.output_names


@pytest.mark.parametrize("aux_option", ["", "   "])
def test_blank_option_adds_no_auxiliary_output(aux_option):
    model = _outputs_model(aux_option)

    model.make_outputs_init()

    assert model.aux_hidden_state_layers == []
    assert "aux_hidden_states" not in model.output_names


@pytest.mark.parametrize("aux_option", ["1,two,3", None])
def test_non_integer_layer_option_reports_its_name(aux_option):
    model = _outputs_model(None)
    model.extra_options["aux_hidden_state_layers"] = aux_option

    with pytest.raises(ValueError, match="aux_hidden_state_layers must be a comma-separated list of integers"):
        model.make_outputs_init()


def test_option_registers_one_output_wide_enough_for_every_tap():
    model = _outputs_model("1, 3, 5")

    model.make_outputs_init()

    assert model.aux_hidden_state_layers == [1, 3, 5]
    assert model.output_names["aux_hidden_states"] == "aux_hidden_states"
    assert model.output_types["aux_hidden_states"] == ir.DataType.FLOAT16
    assert model.output_shapes["aux_hidden_states"] == ["batch_size", "sequence_length", 3 * model.hidden_size]


def test_paged_attention_uses_the_packed_two_dimensional_shape():
    model = _outputs_model("1,2", use_paged_attention=True)

    model.make_outputs_init()

    assert model.output_shapes["aux_hidden_states"] == ["num_tokens", 2 * model.hidden_size]


# Layer 0's incoming residual stream is the raw embedding, which is not produced by any skip
# layer norm, so it has no tap to read.
@pytest.mark.parametrize("aux_option", ["0", "0,2", "8", "9", "-1"])
def test_layers_outside_the_tappable_range_are_rejected(aux_option):
    model = _outputs_model(aux_option, num_layers=8)

    with pytest.raises(ValueError, match="outside"):
        model.make_outputs_init()


def _concat_model(tap_dtype, num_taps=2, hidden_size=4):
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT
    model.hidden_size = hidden_size
    model.use_paged_attention = False
    model.values = {}
    model.node_names = set()
    model.aux_hidden_state_layers = list(range(1, num_taps + 1))
    model.aux_hidden_state_taps = {
        layer_id: (f"tap_{layer_id}", tap_dtype) for layer_id in model.aux_hidden_state_layers
    }
    model.output_names = {"aux_hidden_states": "aux_hidden_states"}
    model.output_shapes = {"aux_hidden_states": ["batch_size", "sequence_length", num_taps * hidden_size]}

    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21}, name="aux_hidden_states_test")
    model.model = ir.Model(graph, ir_version=10)
    for layer_id in model.aux_hidden_state_layers:
        graph.inputs.append(
            model.make_value(f"tap_{layer_id}", tap_dtype, ["batch_size", "sequence_length", hidden_size])
        )
    return model, graph


def test_no_taps_emits_no_concat():
    model, graph = _concat_model(ir.DataType.FLOAT)
    model.aux_hidden_state_layers = []

    model.make_aux_hidden_states()

    assert len(graph) == 0


def test_a_layer_without_a_tap_is_reported():
    model, _ = _concat_model(ir.DataType.FLOAT)
    del model.aux_hidden_state_taps[2]

    with pytest.raises(ValueError, match=r"aux_hidden_state_layers \[2\]"):
        model.make_aux_hidden_states()


def test_taps_are_concatenated_in_declared_layer_order(tmp_path):
    model, graph = _concat_model(ir.DataType.FLOAT, num_taps=3)

    model.make_aux_hidden_states()
    graph.outputs.append(model.values["aux_hidden_states"])

    model_path = tmp_path / "aux_hidden_states.onnx"
    ir.save(model.model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    taps = {f"tap_{i}": np.full((1, 2, 4), float(i), dtype=np.float32) for i in (1, 2, 3)}
    (aux_hidden_states,) = session.run(None, taps)

    assert aux_hidden_states.shape == (1, 2, 12)
    np.testing.assert_array_equal(aux_hidden_states, np.concatenate([taps[f"tap_{i}"] for i in (1, 2, 3)], axis=-1))


def test_a_tap_in_another_dtype_is_cast_to_the_model_io_dtype(tmp_path):
    model, graph = _concat_model(ir.DataType.FLOAT16, num_taps=2)

    model.make_aux_hidden_states()
    graph.outputs.append(model.values["aux_hidden_states"])

    assert [node.op_type for node in graph] == ["Cast", "Cast", "Concat"]

    model_path = tmp_path / "aux_hidden_states_cast.onnx"
    ir.save(model.model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    taps = {f"tap_{i}": np.full((1, 2, 4), float(i), dtype=np.float16) for i in (1, 2)}
    (aux_hidden_states,) = session.run(None, taps)

    assert aux_hidden_states.dtype == np.float32
    assert aux_hidden_states.shape == (1, 2, 8)


def test_layernorm_tap_reuses_the_casted_residual(monkeypatch):
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.include_hidden_states = False
    model.exclude_lm_head = False
    model.aux_hidden_state_layers = [1]
    model.aux_hidden_state_taps = {}
    model.layernorm_attrs = {
        "root_input": "root_input",
        "skip_input": "skip_input",
        "add_offset": 0,
        "epsilon": 1e-6,
        "last_layernorm": False,
        "cast": {
            "use_fp32": True,
            "root_input": False,
            "skip_input": False,
            "output_0": False,
            "output_3": True,
        },
    }
    model.values = {
        "root_input": types.SimpleNamespace(dtype=ir.DataType.FLOAT16, shape=["batch_size", "sequence_length", 4]),
        "skip_input": types.SimpleNamespace(dtype=ir.DataType.FLOAT16, shape=["batch_size", "sequence_length", 4]),
    }

    def make_value(name, dtype, shape):
        value = types.SimpleNamespace(dtype=dtype, shape=shape)
        model.values[name] = value
        return value

    monkeypatch.setattr(model, "make_initializer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_value", make_value)
    monkeypatch.setattr(model, "make_layernorm_subgraph", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_hidden_state_shape", lambda: ["batch_size", "sequence_length", 4])

    model.make_layernorm(
        1,
        types.SimpleNamespace(weight=np.ones(4, dtype=np.float32)),
        skip=True,
        simple=True,
        location="input",
    )

    assert model.aux_hidden_state_taps[1] == ("/model/layers.1/input_layernorm/output_3", ir.DataType.FLOAT16)


def test_the_mtp_head_does_not_inherit_the_target_tap_set():
    from models.builders.qwen import Qwen35MoEModel  # noqa: PLC0415 -- avoids importing qwen at module load

    model = object.__new__(Qwen35MoEModel)
    model.mtp_attrs = {
        "io_dtype": None,
        "onnx_dtype": None,
        "extra_options": copy.deepcopy({"aux_hidden_state_layers": "5,19,33"}),
    }
    model.resolve_mtp_model_config = lambda _options: None
    model.drop_unusable_mtp_kv_scales = lambda _options: None
    model.get_mtp_model_class = lambda: lambda *args, **kwargs: types.SimpleNamespace()

    model.make_mtp_model(None, None, None, None, None, {})

    # The head has one layer, so inheriting a tap set naming layer 5 would fail its range check.
    assert "aux_hidden_state_layers" not in model.mtp_attrs["extra_options"]
