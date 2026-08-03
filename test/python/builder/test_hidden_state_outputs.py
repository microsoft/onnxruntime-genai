# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_base_module():
    sys.modules.setdefault("models", types.ModuleType("models"))
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]

    spec = importlib.util.spec_from_file_location("models.builders.base", BUILDERS_DIR / "base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.builders.base"] = module
    spec.loader.exec_module(module)
    return module


def _load_builder_entrypoint_module():
    builders_stub = types.ModuleType("builders")

    def _stub_getattr(name):
        return type(name, (), {})

    builders_stub.__getattr__ = _stub_getattr
    builders_stub.__path__ = [str(BUILDERS_DIR)]
    sys.modules["builders"] = builders_stub

    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model


@pytest.mark.parametrize(
    "value, error",
    [
        ("", "non-empty"),
        ("2,,14", "integers"),
        ("2,a,14", "integers"),
        ("-1,2", "non-negative"),
        ("2,2", "duplicate"),
        ("14,2", "strictly increasing"),
        ("2,28", "less than the exported layer count"),
    ],
)
def test_parse_output_hidden_states_layers_rejects_invalid_values(value, error):
    with pytest.raises(ValueError, match=error):
        builder_module.parse_output_hidden_states_layers(value, 28)


def test_parse_output_hidden_states_layers_preserves_execution_order():
    assert builder_module.parse_output_hidden_states_layers("2, 14,25", 28) == [2, 14, 25]


def _make_output_model(layer_ids):
    model = Model.__new__(Model)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.BFLOAT16
    model.hidden_size = 2048
    model.extra_options = {"output_hidden_states_layers": layer_ids}
    model.output_names = {
        "hidden_states": "hidden_states",
        "logits": "logits",
        "present.key": [],
        "present.value": [],
    }
    model.output_types = {
        "hidden_states": model.io_dtype,
        "logits": model.io_dtype,
        "present.key": model.io_dtype,
        "present.value": model.io_dtype,
    }
    model.output_shapes = {
        "hidden_states": ["batch_size", "sequence_length", model.hidden_size],
        "logits": ["batch_size", "sequence_length", 151936],
        "present.key": [],
        "present.value": [],
    }
    return model


def test_make_outputs_init_adds_selected_hidden_state_metadata():
    model = _make_output_model([2, 14, 25])

    model.make_outputs_init()

    for layer_id in (2, 14, 25):
        name = f"hidden_states_before_layer_{layer_id}"
        assert model.output_names[name] == name
        assert model.output_types[name] == ir.DataType.BFLOAT16
        assert model.output_shapes[name] == ["batch_size", "sequence_length", 2048]
    assert list(model.output_names)[-3:] == [
        "hidden_states_before_layer_2",
        "hidden_states_before_layer_14",
        "hidden_states_before_layer_25",
    ]


def test_make_outputs_init_is_unchanged_without_selected_layers():
    model = _make_output_model([])

    model.make_outputs_init()

    assert all(not name.startswith("hidden_states_before_layer_") for name in model.output_names)


@pytest.mark.parametrize(
    "skip, expected_hidden_states",
    [
        (False, "embedding_output"),
        (True, "/model/layers.14/input_layernorm/output_3"),
    ],
)
def test_input_layernorm_captures_the_pre_layer_residual(skip, expected_hidden_states):
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.BFLOAT16
    model.hidden_size = 2048
    model.use_paged_attention = False
    model.output_hidden_states_layers = [14]
    model.layernorm_attrs = {
        "root_input": "embedding_output",
        "skip_input": "previous_mlp_output",
        "epsilon": 1e-6,
        "add_offset": 0,
        "last_layernorm": False,
        "cast": {
            "use_fp32": False,
            "root_input": False,
            "skip_input": False,
            "output_0": False,
            "output_3": False,
        },
    }
    model.values = {
        "embedding_output": ir.Value(
            name="embedding_output",
            type=ir.TensorType(ir.DataType.BFLOAT16),
            shape=ir.Shape(["batch_size", "sequence_length", 2048]),
        ),
        "previous_mlp_output": ir.Value(
            name="previous_mlp_output",
            type=ir.TensorType(ir.DataType.BFLOAT16),
            shape=ir.Shape(["batch_size", "sequence_length", 2048]),
        ),
    }
    model.make_initializer = lambda *_args, **_kwargs: None
    model.make_layernorm_subgraph = lambda *_args, **_kwargs: None
    captured = []
    model.make_hidden_states_layer_output = lambda layer_id, hidden_states: captured.append((layer_id, hidden_states))
    layernorm = types.SimpleNamespace(weight=0)

    model.make_layernorm(14, layernorm, skip=skip, simple=True, location="input")

    assert captured == [(14, expected_hidden_states)]


def test_hidden_state_output_identity_uses_stable_public_name():
    model = Model.__new__(Model)
    model.node_names = set()
    model.values = {}
    model.graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21}, name="test_graph")
    model.model = ir.Model(model.graph, ir_version=10)
    source = model.make_value(
        "/model/layers.14/input_layernorm/output_3",
        ir.DataType.BFLOAT16,
        ["batch_size", "sequence_length", 2048],
    )
    output = model.make_value(
        "hidden_states_before_layer_14",
        ir.DataType.BFLOAT16,
        ["batch_size", "sequence_length", 2048],
    )
    model.graph.inputs.append(source)
    model.graph.outputs.append(output)

    model.make_hidden_states_layer_output(14, source.name)

    node = model.graph.node(0)
    assert node.op_type == "Identity"
    assert node.inputs[0].name == source.name
    assert node.outputs[0].name == "hidden_states_before_layer_14"
