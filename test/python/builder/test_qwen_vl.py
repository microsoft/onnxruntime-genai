# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest
import torch

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parent))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
qwen_module = _load_builder_module("qwen")
Model = base_module.Model
Qwen25VLTextModel = qwen_module.Qwen25VLTextModel
Qwen3VLTextModel = qwen_module.Qwen3VLTextModel
Qwen35TextModel = qwen_module.Qwen35TextModel
Qwen35MoETextModel = qwen_module.Qwen35MoETextModel


def _initialize_qwen_model(monkeypatch, model_class):
    def initialize_base(self, *_args, **_kwargs):
        self.layernorm_attrs = {
            "cast": {
                "use_fp32": False,
                "root_input": False,
                "skip_input": False,
                "output_0": False,
                "output_3": False,
            }
        }
        self.rope_attrs = {
            "mrope_layout": 0,
            "cast": {"use_fp32": False, "root_input": False, "output_0": False},
        }
        self.attention_attrs = {"q_norm": False, "k_norm": False}

    monkeypatch.setattr(Model, "__init__", initialize_base)
    return model_class(types.SimpleNamespace(), ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", "", {})


@pytest.mark.parametrize(
    "model_class,expected_layout,uses_qk_norm,casts_rope_input,casts_layernorm_skip",
    [
        (Qwen25VLTextModel, 0, False, True, True),
        (Qwen3VLTextModel, 1, True, False, False),
    ],
)
def test_qwen_vl_configures_variant_specific_mrope(
    monkeypatch, model_class, expected_layout, uses_qk_norm, casts_rope_input, casts_layernorm_skip
):
    model = _initialize_qwen_model(monkeypatch, model_class)

    assert model.rope_attrs["mrope_layout"] == expected_layout
    assert model.rope_attrs["cast"] == {
        "use_fp32": True,
        "root_input": casts_rope_input,
        "output_0": True,
    }
    assert model.layernorm_attrs["cast"]["use_fp32"]
    assert model.layernorm_attrs["cast"]["output_3"] is casts_layernorm_skip
    assert model.attention_attrs["q_norm"] is uses_qk_norm
    assert model.attention_attrs["k_norm"] is uses_qk_norm


@pytest.mark.parametrize("model_class", [Qwen25VLTextModel, Qwen3VLTextModel])
def test_qwen_vl_uses_separate_qkv_and_mrope(monkeypatch, model_class):
    model = _initialize_qwen_model(monkeypatch, model_class)

    assert not model.is_packed_matmul_supported()
    assert not model.is_fused_rope_supported()


@pytest.mark.parametrize("model_class", [Qwen25VLTextModel, Qwen3VLTextModel])
def test_qwen_vl_decoder_uses_3d_position_ids(monkeypatch, model_class):
    model = model_class.__new__(model_class)
    model.input_shapes = {"position_ids": ["batch_size", "sequence_length"]}
    monkeypatch.setattr(Model, "make_inputs_and_outputs", lambda self: None)

    model.make_inputs_and_outputs()

    assert model.input_shapes["position_ids"] == [3, "batch_size", "sequence_length"]


@pytest.mark.parametrize(
    "layout,sections",
    [
        (0, [16, 24, 24]),
        (1, [24, 20, 20]),
    ],
)
def test_qwen_vl_emits_layout_specific_mrotary_embedding(layout, sections):
    model = Model.__new__(Model)
    model.head_size = 128
    model.rope_attrs = {
        "interleaved": 0,
        "rotary_embedding_dim": 0,
        "mrope_section": sections,
        "mrope_layout": layout,
    }
    model.nodes = []
    model.values = []

    def make_node(op_type, inputs, outputs, name, domain, **attributes):
        model.nodes.append(
            {
                "op_type": op_type,
                "inputs": inputs,
                "outputs": outputs,
                "name": name,
                "domain": domain,
                "attributes": attributes,
            }
        )

    model.make_node = make_node
    model.make_value = lambda name, dtype, shape: model.values.append((name, dtype, shape))

    model.make_mrotary_embedding(
        "/model/layers.0/attn/q_rotary/MRotaryEmbedding",
        "q",
        "q_rotated",
        position_ids="position_ids",
        cos_cache_name="cos_cache",
        sin_cache_name="sin_cache",
        num_heads=16,
        dtype=ir.DataType.FLOAT,
    )

    node = model.nodes[0]
    assert node["op_type"] == "MRotaryEmbedding"
    assert node["domain"] == "com.microsoft"
    assert node["inputs"] == ["q", "position_ids", "cos_cache", "sin_cache"]
    assert node["attributes"]["num_heads"] == 16
    assert node["attributes"]["mrope_section"] == sections
    assert node["attributes"]["mrope_layout"] == layout
    assert model.values == [("q_rotated", ir.DataType.FLOAT, ["batch_size", "sequence_length", 2048])]


def test_qwen3_vl_keeps_qk_norm_outputs_in_fp32(monkeypatch):
    model = Qwen3VLTextModel.__new__(Qwen3VLTextModel)
    model.layernorm_attrs = {"cast": {"output_0": True}}
    model.attention_attrs = {"q_path": "q_norm", "k_path": "k_norm"}
    model.values = {
        "q_norm": types.SimpleNamespace(dtype=ir.DataType.FLOAT16),
        "k_norm": types.SimpleNamespace(dtype=ir.DataType.FLOAT16),
    }
    observed_cast_setting = []

    def make_qk_norm(self, *_args):
        observed_cast_setting.append(self.layernorm_attrs["cast"]["output_0"])

    monkeypatch.setattr(Model, "make_qk_norm", make_qk_norm)

    model.make_qk_norm(0, object())

    assert observed_cast_setting == [False]
    assert model.values["q_norm"].dtype == ir.DataType.FLOAT
    assert model.values["k_norm"].dtype == ir.DataType.FLOAT
    assert model.layernorm_attrs["cast"]["output_0"]


def test_qwen35_configures_interleaved_partial_mrope(monkeypatch):
    def initialize_base(self, config, *_args, **_kwargs):
        self.head_size = 128
        self.q_size = 2048
        self.input_shapes = {"position_ids": ["batch_size", "sequence_length"]}
        self.layernorm_attrs = {"cast": {}, "add_offset": 0}
        self.attention_attrs = {"q_norm": False, "k_norm": False}
        self.rope_attrs = {
            "op_type": "RotaryEmbedding",
            "partial_rotary_factor": config.partial_rotary_factor,
            "rotary_embedding_dim": int(self.head_size * config.partial_rotary_factor),
            "mrope_section": [],
            "mrope_layout": 0,
            "cast": {"use_fp32": False, "root_input": False, "output_0": False},
        }
        self.make_rope_init(config)

    config = types.SimpleNamespace(
        num_hidden_layers=1,
        rope_parameters={
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
        },
    )
    monkeypatch.setattr(Model, "__init__", initialize_base)

    model = Qwen35TextModel(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", "", {})

    assert model.q_size == 2048
    assert model.rope_attrs["op_type"] == "MRotaryEmbedding"
    assert model.rope_attrs["mrope_section"] == [11, 11, 10]
    assert model.rope_attrs["mrope_layout"] == 1
    assert model.rope_attrs["rotary_embedding_dim"] == 32
    assert model.rope_attrs["cast"] == {"use_fp32": True, "root_input": True, "output_0": True}
    assert not model.is_fused_rope_supported()


def test_qwen35_genai_config_includes_recurrent_cache_names(monkeypatch, tmp_path):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.hf_token = None
    model.hf_remote = False
    model.num_layers = 24
    model.layer_types = ["linear_attention"] * 24
    model.model_type = "qwen3_5_text"
    model.input_names = {}
    model.output_names = {}
    observed = {}

    config = types.SimpleNamespace(save_pretrained=lambda _path: None)
    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    def capture_names(self, *_args, **_kwargs):
        observed["inputs"] = self.input_names.copy()
        observed["outputs"] = self.output_names.copy()

    monkeypatch.setattr(Model, "make_genai_config", capture_names)

    model.make_genai_config("model", {}, tmp_path)

    assert observed["inputs"]["past.conv"] == "past.%d.conv"
    assert observed["inputs"]["past.recurrent"] == "past.%d.recurrent"
    assert observed["outputs"]["present.conv"] == "present.%d.conv"
    assert observed["outputs"]["present.recurrent"] == "present.%d.recurrent"


def test_qwen35_applies_mrope_to_q_and_k(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.attention_attrs = {
        "rope": True,
        "use_rope_in_attn": False,
        "q_path": "q",
        "k_path": "k",
    }
    model.input_names = {"position_ids": "position_ids"}
    model.rope_attrs = {"op_type": "MRotaryEmbedding"}
    calls = []

    def make_rotary_embedding_op(name, root_input, position_ids):
        calls.append((name, root_input, position_ids))

    monkeypatch.setattr(model, "make_rotary_embedding_op", make_rotary_embedding_op)

    model.make_attention_qk_rope(2)

    assert calls == [
        ("/model/layers.2/attn/q_rotary/MRotaryEmbedding", "q", "position_ids"),
        ("/model/layers.2/attn/k_rotary/MRotaryEmbedding", "k", "position_ids"),
    ]
    assert model.attention_attrs["q_path"] == "/model/layers.2/attn/q_rotary/MRotaryEmbedding/output_0"
    assert model.attention_attrs["k_path"] == "/model/layers.2/attn/k_rotary/MRotaryEmbedding/output_0"


def test_qwen35_attention_input_proj_splits_per_head_gate(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.num_attn_heads = 16
    model.head_size = 128
    model.io_dtype = ir.DataType.FLOAT16
    model.attention_attrs = {}
    calls = []

    def make_base_input_proj(self, layer_id, attention, root_input, **kwargs):
        calls.append(("base_input", layer_id, root_input))
        self.attention_attrs["q_path"] = "q_gate"

    monkeypatch.setattr(Model, "make_attention_input_proj", make_base_input_proj)
    monkeypatch.setattr(
        model,
        "make_reshape",
        lambda name, inputs, dtype, shape: calls.append(("reshape", name, inputs, shape)),
    )
    monkeypatch.setattr(
        model,
        "make_split",
        lambda name, inputs, outputs, dtypes, shapes, axis: calls.append(("split", name, inputs, outputs, shapes, axis)),
    )

    model.make_attention_input_proj(3, object(), "hidden_states")

    assert calls[0] == ("base_input", 3, "hidden_states")
    assert calls[1][0:3] == (
        "reshape",
        "/model/layers.3/attn/q_gate/Reshape",
        ["q_gate", "/model/constants/INT64/[0, 0, 16, 256]"],
    )
    assert calls[2][0:2] == ("split", "/model/layers.3/attn/q_gate/Split")
    assert model.attention_attrs["q_path"] == "/model/layers.3/attn/q_proj/Reshape/output_0"
    assert model.attention_attrs["gate_path"] == "/model/layers.3/attn/gate/Reshape/output_0"


def test_qwen35_attention_output_proj_gates_before_base_projection(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.num_attn_heads = 16
    model.head_size = 128
    model.io_dtype = ir.DataType.FLOAT16
    model.attention_attrs = {
        "op_type": "GroupQueryAttention",
        "gate_path": "gate",
    }
    calls = []

    monkeypatch.setattr(
        model,
        "make_sigmoid",
        lambda name, root_input, dtype, shape: calls.append(("sigmoid", name, root_input, shape)),
    )
    monkeypatch.setattr(
        model,
        "make_mul",
        lambda name, inputs, dtype, shape: calls.append(("mul", name, inputs, shape)),
    )

    def make_base_output_proj(self, layer_id, attention, root_input, **kwargs):
        calls.append(("base_output", layer_id, root_input, self.attention_attrs["o_path"]))

    monkeypatch.setattr(Model, "make_attention_output_proj", make_base_output_proj)

    model.make_attention_output_proj(3, object(), "hidden_states")

    assert calls == [
        ("sigmoid", "/model/layers.3/attn/gate/Sigmoid", "gate", ["batch_size", "sequence_length", 2048]),
        (
            "mul",
            "/model/layers.3/attn/gate/Mul",
            [
                "/model/layers.3/attn/GroupQueryAttention/output_0",
                "/model/layers.3/attn/gate/Sigmoid/output_0",
            ],
            ["batch_size", "sequence_length", 2048],
        ),
        ("base_output", 3, "hidden_states", "/model/layers.3/attn/gate/Mul/output_0"),
    ]


@pytest.mark.parametrize(
    "layer_type,expected_attribute",
    [("linear_attention", "linear_attn"), ("full_attention", "self_attn")],
)
def test_qwen35_selects_layer_attention_module(layer_type, expected_attribute):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.layer_types = [layer_type]
    layer = types.SimpleNamespace(linear_attn=object(), self_attn=object())

    assert model.get_attention_module(0, layer) is getattr(layer, expected_attribute)
    assert "make_layer" not in Qwen35TextModel.__dict__


def test_qwen35_moe_delegates_only_layer_mlp(monkeypatch):
    model = Qwen35MoETextModel.__new__(Qwen35MoETextModel)
    layer = types.SimpleNamespace(mlp=object())
    calls = []
    monkeypatch.setattr(
        model,
        "make_moe",
        lambda layer_id, mlp, root_input: calls.append((layer_id, mlp, root_input)),
    )

    model.make_layer_mlp(2, layer, "hidden_states")

    assert calls == [(2, layer.mlp, "hidden_states")]
    assert "make_layer" not in Qwen35MoETextModel.__dict__


@pytest.mark.parametrize(
    "method_name,expected_op_type,inputs,attributes",
    [
        (
            "make_gated_rms_norm",
            "GatedRMSNorm",
            ["x", "scale", "gate"],
            {"epsilon": 1e-6},
        ),
        (
            "make_gated_add",
            "GatedAdd",
            ["x", "y", "gate"],
            {},
        ),
    ],
)
def test_gated_contrib_op_emitters(method_name, expected_op_type, inputs, attributes):
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    nodes = []
    values = []
    model.make_node = lambda op_type, inputs, outputs, name, domain, **kwargs: nodes.append(
        (op_type, inputs, outputs, name, domain, kwargs)
    )
    model.make_value = lambda name, dtype, shape: values.append((name, dtype, shape))
    shape = ["batch_size", "sequence_length", 2048]

    if method_name == "make_gated_rms_norm":
        model.make_gated_rms_norm("/gated", inputs[0], inputs[1], inputs[2], shape, epsilon=1e-6)
    else:
        model.make_gated_add("/gated", inputs[0], inputs[1], inputs[2], shape)

    assert nodes == [(expected_op_type, inputs, ["/gated/output_0"], "/gated", "com.microsoft", attributes)]
    assert values == [("/gated/output_0", ir.DataType.FLOAT16, shape)]


def test_linear_attention_gate_emitter():
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    nodes = []
    values = []
    model.make_node = lambda op_type, inputs, outputs, name, domain, **kwargs: nodes.append(
        (op_type, inputs, outputs, name, domain, kwargs)
    )
    model.make_value = lambda name, dtype, shape: values.append((name, dtype, shape))
    shape = ["batch_size", "sequence_length", 16]

    model.make_linear_attention_gate("/gate", "a", "dt_bias", "decay_scale", "b", shape)

    assert nodes == [
        (
            "LinearAttentionGate",
            ["a", "dt_bias", "decay_scale", "b"],
            ["/gate/output_0", "/gate/output_1"],
            "/gate",
            "com.microsoft",
            {},
        )
    ]
    assert values == [
        ("/gate/output_0", ir.DataType.FLOAT16, shape),
        ("/gate/output_1", ir.DataType.FLOAT16, shape),
    ]


def test_qwen35_linear_attention_uses_fused_gate(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.linear_key_dim = 8
    model.linear_value_dim = 16
    model.linear_num_value_heads = 4
    model.linear_key_head_dim = 2
    attention = types.SimpleNamespace(
        dt_bias=torch.ones(4),
        A_log=torch.zeros(4),
    )
    initializers = []
    gate_calls = []
    monkeypatch.setattr(model, "make_split", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_l2_normalize", lambda name, _input: f"{name}/output_0")
    monkeypatch.setattr(model, "make_mul", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model,
        "make_initializer",
        lambda tensor, name, to: initializers.append((tensor, name, to)),
    )
    monkeypatch.setattr(
        model,
        "make_linear_attention_gate",
        lambda name, **kwargs: gate_calls.append((name, kwargs)),
    )

    outputs = model.make_linear_attention_normalize_and_gate(
        3,
        attention,
        "conv_output",
        "/b_proj/MatMul",
        "/a_proj/MatMul",
    )

    assert [item[1:] for item in initializers] == [
        ("model.layers.3.linear_attn.dt_bias", ir.DataType.FLOAT),
        ("model.layers.3.linear_attn.neg_exp_A", ir.DataType.FLOAT),
    ]
    torch.testing.assert_close(initializers[1][0], -torch.ones(4))
    assert gate_calls == [
        (
            "/model/layers.3/linear_attn/LinearAttentionGate",
            {
                "a": "/a_proj/MatMul/output_0",
                "dt_bias": "model.layers.3.linear_attn.dt_bias",
                "decay_scale": "model.layers.3.linear_attn.neg_exp_A",
                "b": "/b_proj/MatMul/output_0",
                "shape": ["batch_size", "sequence_length", 4],
            },
        )
    ]
    assert outputs[3:] == (
        "/model/layers.3/linear_attn/LinearAttentionGate/output_0",
        "/model/layers.3/linear_attn/LinearAttentionGate/output_1",
    )


def test_qwen35_linear_attention_uses_gated_rms_norm(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.linear_value_dim = 2048
    model.layernorm_attrs = {"epsilon": 1e-6}
    model.layernorm_attrs["skip_input"] = ""
    attention = types.SimpleNamespace(
        norm=types.SimpleNamespace(weight=torch.ones(128)),
        out_proj=object(),
    )
    calls = []
    monkeypatch.setattr(
        model,
        "make_initializer",
        lambda tensor, name, to: calls.append(("initializer", name, to, tuple(tensor.shape))),
    )
    monkeypatch.setattr(
        model,
        "make_gated_rms_norm",
        lambda name, root_input, scale, gate, shape, epsilon: calls.append(
            ("gated_rms_norm", name, root_input, scale, gate, shape, epsilon)
        ),
    )
    monkeypatch.setattr(
        model,
        "make_matmul",
        lambda matmul, name, root_input: calls.append(("matmul", name, root_input)) or name,
    )

    model.make_linear_attention_output_proj(4, attention, "linear_output", "/z_proj/MatMul")

    assert calls[1] == (
        "gated_rms_norm",
        "/model/layers.4/linear_attn/GatedRMSNorm",
        "linear_output",
        "model.layers.4.linear_attn.norm.weight",
        "/z_proj/MatMul/output_0",
        ["batch_size", "sequence_length", 2048],
        1e-6,
    )
    assert calls[2] == (
        "matmul",
        "/model/layers.4/linear_attn/out_proj/MatMul",
        "/model/layers.4/linear_attn/GatedRMSNorm/output_0",
    )


def test_qwen35_moe_combines_shared_expert_with_gated_add(monkeypatch):
    model = Qwen35MoETextModel.__new__(Qwen35MoETextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.hidden_size = 16
    model.moe_intermediate_size = 4
    model.moe_attrs = {
        "op_type": "MoE",
        "num_experts": 2,
    }
    model.layernorm_attrs = {"skip_input": ""}
    mlp = types.SimpleNamespace(
        gate=object(),
        experts=types.SimpleNamespace(
            gate_up_proj=torch.zeros(2, 8, 16),
            down_proj=torch.zeros(2, 16, 4),
        ),
        shared_expert=object(),
        shared_expert_gate=object(),
    )
    calls = []
    monkeypatch.setattr(model, "make_matmul", lambda *_args: "/router/MatMul")
    monkeypatch.setattr(model, "make_reshape", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_initializer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_moe_op", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model, "make_shared_expert", lambda *_args: ("shared_output", "sigmoid_gate"))
    monkeypatch.setattr(
        model,
        "make_gated_add",
        lambda name, root_input, scaled_input, gate, shape: calls.append(
            (name, root_input, scaled_input, gate, shape)
        ),
    )

    model.make_moe(1, mlp, "hidden_states")

    assert calls == [
        (
            "/model/layers.1/moe/GatedAdd",
            "/model/layers.1/moe/MoE/output_0",
            "shared_output",
            "sigmoid_gate",
            ["batch_size", "sequence_length", 16],
        )
    ]
