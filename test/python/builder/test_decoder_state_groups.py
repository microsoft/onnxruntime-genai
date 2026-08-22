# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


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
Qwen35TextModel = qwen_module.Qwen35TextModel


class _NoGenerationConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise FileNotFoundError("no generation_config.json")


def _make_config_model(model_type, layer_types=None, use_paged_attention=True):
    model = model_type.__new__(model_type)
    model.hf_token = None
    model.hf_remote = False
    model.ep = "cuda"
    model.ep_attrs = {"cuda": {}}
    model.extra_options = {}
    model.use_paged_attention = use_paged_attention
    model.past_present_share_buffer = False
    model.context_length = 262144
    model.filename = "model.onnx"
    model.head_size = 256
    model.hidden_size = 5120
    model.num_attn_heads = 24
    model.num_kv_heads = 4
    model.num_layers = 64
    model.model_type = "Qwen3_5_textForCausalLM"
    model.vocab_size = 248320
    model.window_size = None
    model.use_windowed_paged_kv_cache = False
    model.eps_with_windowed_kv_cache = {"cuda"}
    model.attention_attrs = {"paged_block_size": 256}
    model.input_names = {
        "input_ids": "input_ids",
        "block_table": "block_table",
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "attention_metadata": "attention_metadata",
        "past_key_values.key": "past_key_values.%d.key",
        "past_key_values.value": "past_key_values.%d.value",
    }
    model.output_names = {
        "logits": "logits",
        "present.key": "present.%d.key",
        "present.value": "present.%d.value",
    }
    if layer_types is not None:
        model.layer_types = layer_types
    return model


def _write_config(monkeypatch, tmp_path, model):
    hf_config = SimpleNamespace(eos_token_id=[248044], bos_token_id=248045, pad_token_id=248044)
    monkeypatch.setattr(base_module, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: hf_config))
    monkeypatch.setattr(base_module, "GenerationConfig", _NoGenerationConfig)
    make_config = Model.make_genai_config.__get__(model)
    make_config("model_name_or_path", {}, str(tmp_path))
    return json.loads((tmp_path / "genai_config.json").read_text())


def test_common_paged_builder_preserves_legacy_manifest_absence(monkeypatch, tmp_path):
    config = _write_config(monkeypatch, tmp_path, _make_config_model(Model))

    assert "state_groups" not in config["model"]["decoder"]


def test_common_nonpaged_builder_preserves_manifest_absence(monkeypatch, tmp_path):
    config = _write_config(
        monkeypatch,
        tmp_path,
        _make_config_model(Model, use_paged_attention=False),
    )

    assert "state_groups" not in config["model"]["decoder"]


def test_qwen_all_attention_builder_preserves_legacy_manifest_absence(monkeypatch, tmp_path):
    config = _write_config(
        monkeypatch,
        tmp_path,
        _make_config_model(Qwen35TextModel, layer_types=["full_attention"] * 64),
    )

    assert "state_groups" not in config["model"]["decoder"]


def test_qwen38_official_geometry_emits_exact_sparse_groups(monkeypatch, tmp_path):
    layer_types = ["full_attention" if (layer_id + 1) % 4 == 0 else "linear_attention" for layer_id in range(64)]
    config = _write_config(
        monkeypatch,
        tmp_path,
        _make_config_model(Qwen35TextModel, layer_types=layer_types),
    )

    groups = config["model"]["decoder"]["state_groups"]
    assert [group["kind"] for group in groups] == [
        "paged_kv",
        "fixed",
        "fixed",
    ]
    assert groups[0]["layer_ids"] == list(range(3, 64, 4))
    assert groups[1]["layer_ids"] == [i for i in range(64) if (i + 1) % 4 != 0]
    assert groups[2]["layer_ids"] == groups[1]["layer_ids"]
    assert groups[1]["bindings"]["state"] == {
        "input": "past_key_values.%d.conv_state",
        "output": "present.%d.conv_state",
    }
    assert groups[2]["bindings"]["state"] == {
        "input": "past_key_values.%d.recurrent_state",
        "output": "present.%d.recurrent_state",
    }


def test_qwen38_layer_types_support_reduced_official_fixture():
    official_layer_types = [
        "full_attention" if (layer_id + 1) % 4 == 0 else "linear_attention" for layer_id in range(64)
    ]
    config = SimpleNamespace(text_config=SimpleNamespace(layer_types=official_layer_types))

    assert Qwen35TextModel._resolve_layer_types(config, 4) == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


def test_qwen38_layer_types_reject_invalid_configurations():
    short_config = SimpleNamespace(text_config=SimpleNamespace(layer_types=["linear_attention"]))
    with pytest.raises(ValueError, match="1 entries"):
        Qwen35TextModel._resolve_layer_types(short_config, 2)

    unknown_config = SimpleNamespace(text_config=SimpleNamespace(layer_types=["unknown"]))
    with pytest.raises(ValueError, match="Unsupported"):
        Qwen35TextModel._resolve_layer_types(unknown_config, 1)


def _recording_model():
    model = Model.__new__(Model)
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.nodes = []
    model.values = []
    model.casts = []
    model.make_node = lambda op_type, **kwargs: model.nodes.append((op_type, kwargs))
    model.make_value = lambda name, dtype, shape: model.values.append((name, dtype, shape))
    model.make_cast = lambda name, input_name, dtype, shape: (
        model.casts.append((name, input_name, dtype, shape)),
        model.values.append((f"{name}/output_0", dtype, shape)),
    )
    return model


def test_varlen_causal_conv_emits_reviewed_required_state_contract():
    model = _recording_model()
    model.make_varlen_causal_conv_with_state(
        "/conv",
        root_input="input",
        weight="weight",
        cumulative_sequence_length="cu",
        bias="",
        past_conv_state="past",
        present_conv_state="present",
        output_shape=["num_tokens", 8],
        present_conv_shape=["batch_size", 8, 3],
    )

    op_type, node = model.nodes[0]
    assert op_type == "VarlenCausalConvWithState"
    assert node["inputs"] == ["input", "weight", "cu", "", "past"]
    assert node["outputs"] == ["/conv/output_0", "present"]
    assert node["activation"] == "silu"
    assert node["max_checkpoints"] == 0
    assert "ndim" not in node
    assert "state_window" not in node


def test_varlen_linear_attention_emits_reviewed_fp32_state_contract():
    model = _recording_model()
    state_shape = ["batch_size", 16, 128, 128]
    model.make_varlen_linear_attention(
        "/linear",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="decay",
        beta="beta",
        gate_shape=["num_tokens", 16],
        output_shape=["num_tokens", 16, 128],
        present_recurrent_shape=state_shape,
        update_rule="gated_delta",
        scale=1.0,
    )

    op_type, node = model.nodes[0]
    assert op_type == "VarlenLinearAttention"
    assert node["inputs"] == ["q", "k", "v", "cu", "past", "decay", "beta"]
    assert node["outputs"] == ["/linear/output_0", "present"]
    assert node["update_rule"] == "gated_delta"
    assert node["scale"] == 1.0
    assert node["decay_activation"] == "none"
    assert node["beta_activation"] == "none"
    assert node["max_checkpoints"] == 0
    assert "q_num_heads" not in node
    assert "kv_num_heads" not in node
    assert "state_window" not in node
    assert model.values[-1] == ("present", base_module.ir.DataType.FLOAT, state_shape)


def test_gated_delta_net_evaluation_preserves_precomputed_gate_contract():
    model = _recording_model()
    state_shape = ["batch_size", 16, 128, 128]
    model.make_varlen_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="decay",
        beta="beta",
        gate_shape=["num_tokens", 16],
        output_shape=["num_tokens", 16, 128],
        present_recurrent_shape=state_shape,
        scale=1.0,
    )

    op_type, node = model.nodes[0]
    assert model.casts == [
        (
            "/gdn/decay_fp32/Cast",
            "decay",
            base_module.ir.DataType.FLOAT,
            ["num_tokens", 16],
        ),
        (
            "/gdn/beta_fp32/Cast",
            "beta",
            base_module.ir.DataType.FLOAT,
            ["num_tokens", 16],
        ),
    ]
    assert op_type == "GatedDeltaNet"
    assert node["inputs"] == [
        "q",
        "k",
        "v",
        "cu",
        "/gdn/decay_fp32/Cast/output_0",
        "/gdn/beta_fp32/Cast/output_0",
        "past",
    ]
    assert node["outputs"] == ["/gdn/output_0", "present"]
    assert node["update_rule"] == "gated_delta"
    assert node["scale"] == 1.0
    assert node["gate_activation"] == "none"
    assert node["beta_activation"] == "none"
    assert node["qk_l2_norm"] == 0
    assert node["chunk_size"] == 64
    assert node["state_checkpoints"] == 0
    assert model.values[-1] == ("present", base_module.ir.DataType.FLOAT, state_shape)


def test_gated_delta_net_evaluation_rejects_bf16_but_accepts_fp32():
    model = _recording_model()
    model.io_dtype = base_module.ir.DataType.BFLOAT16
    with pytest.raises(ValueError, match="does not support bfloat16"):
        model.make_varlen_gated_delta_net("/gdn")

    model.io_dtype = base_module.ir.DataType.FLOAT
    model.make_varlen_gated_delta_net(
        "/gdn_fp32",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="decay",
        beta="beta",
        gate_shape=["num_tokens", 16],
        output_shape=["num_tokens", 16, 128],
        present_recurrent_shape=["batch_size", 16, 128, 128],
    )


def test_qwen_packed_linear_attention_reshapes_thd_and_uses_v_major_state():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.linear_conv_dim = 48
    model.linear_value_dim = 24
    model.linear_num_value_heads = 3
    model.linear_num_key_heads = 2
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.linear_conv_kernel_dim = 4
    model._state_window_dims = []
    model._state_window = 0
    model.input_names = {"cumulative_sequence_lengths": "cu"}
    model._leading_dims = lambda: ["num_tokens"]
    model._make_linear_attention_projections = lambda *args: ("z", "b", "a", "conv_input", "weight")
    model.make_initializer = lambda *args, **kwargs: None
    conv_calls = []
    model.make_varlen_causal_conv_with_state = lambda name, **kwargs: conv_calls.append((name, kwargs))
    model._make_linear_attention_normalize_and_gate = lambda *args: ("q_flat", "k_flat", "v_flat", "decay", "beta")
    reshapes = []
    model.make_reshape = lambda name, inputs, dtype, shape: reshapes.append((name, inputs, dtype, shape))
    linear_calls = []
    model.make_varlen_gated_delta_net = lambda name, **kwargs: linear_calls.append((name, kwargs))
    outputs = []
    model._make_linear_attention_output = lambda *args: outputs.append(args)

    model._make_linear_attention(1, SimpleNamespace(), "root")

    assert conv_calls[0][1]["output_shape"] == ["num_tokens", 48]
    assert [reshape[3] for reshape in reshapes[:3]] == [
        ["num_tokens", 2, 4],
        ["num_tokens", 2, 4],
        ["num_tokens", 3, 8],
    ]
    linear = linear_calls[0][1]
    assert linear_calls[0][0].endswith("/GatedDeltaNet")
    assert linear["q_path"].endswith("/q_thd/Reshape/output_0")
    assert linear["k_path"].endswith("/k_thd/Reshape/output_0")
    assert linear["v_path"].endswith("/v_thd/Reshape/output_0")
    assert linear["output_shape"] == ["num_tokens", 3, 8]
    assert linear["present_recurrent_shape"] == ["batch_size", 3, 8, 4]
    assert linear["gate_shape"] == ["num_tokens", 3]
    assert reshapes[-1][3] == ["num_tokens", 24]
    assert outputs[0][2].endswith("/linear_attention_output/Reshape/output_0")


def test_qwen_packed_recurrent_io_is_fp32_v_major():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.layer_types = ["linear_attention"]
    model._state_window_dims = []
    model.linear_conv_dim = 48
    model.linear_conv_kernel_dim = 4
    model.linear_num_value_heads = 3
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.input_names = {
        "past_key_values.key": ["past.0.key"],
        "past_key_values.value": ["past.0.value"],
    }
    model.output_names = {
        "present.key": ["present.0.key"],
        "present.value": ["present.0.value"],
    }
    model.input_types = {}
    model.input_shapes = {}
    model.output_types = {}
    model.output_shapes = {}

    model._setup_hybrid_cache_io()

    assert model.input_types["past_state.0.conv"] == base_module.ir.DataType.FLOAT16
    assert model.input_shapes["past_state.0.conv"] == ["batch_size", 48, 3]
    assert model.input_types["past_state.0.recurrent"] == base_module.ir.DataType.FLOAT
    assert model.input_shapes["past_state.0.recurrent"] == ["batch_size", 3, 8, 4]
    assert model.output_types["present_state.0.recurrent"] == base_module.ir.DataType.FLOAT
    assert model.output_shapes["present_state.0.recurrent"] == ["batch_size", 3, 8, 4]
