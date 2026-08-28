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
    model.use_windowed_paged_kv_cache = False
    model.past_present_share_buffer = False
    model.context_length = 262144
    model.context_length_attrs = {
        "state_window": 0,
        "state_window_dims": [],
        "state_update_capacity": 0,
        "window_kv_cache": True,
        "window_kv_cache_slack": 0,
    }
    model.filename = "model.onnx"
    model.head_size = 256
    model.hidden_size = 5120
    model.num_attn_heads = 24
    model.num_kv_heads = 4
    model.num_layers = 64
    model.model_type = "Qwen3_5_textForCausalLM"
    model.vocab_size = 248320
    model.window_size = None
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
        "past.conv": "past.%d.conv",
        "past.recurrent": "past.%d.recurrent",
    }
    model.output_names = {
        "logits": "logits",
        "present.key": "present.%d.key",
        "present.value": "present.%d.value",
        "present.conv": "present.%d.conv",
        "present.recurrent": "present.%d.recurrent",
    }
    model.layer_types = layer_types if layer_types is not None else ["full_attention"] * model.num_layers
    return model


def _write_config(monkeypatch, tmp_path, model):
    hf_config = SimpleNamespace(eos_token_id=[248044], bos_token_id=248045, pad_token_id=248044)
    monkeypatch.setattr(base_module, "GenerationConfig", _NoGenerationConfig)
    make_config = Model.make_genai_config.__get__(model)
    make_config(hf_config, {}, str(tmp_path))
    return json.loads((tmp_path / "genai_config.json").read_text())


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


def test_common_paged_builder_emits_paged_kv_group(monkeypatch, tmp_path):
    config = _write_config(monkeypatch, tmp_path, _make_config_model(Model))

    assert config["model"]["decoder"]["state_groups"] == [{"kind": "paged_kv", "layer_ids": list(range(64))}]


def test_common_nonpaged_builder_preserves_manifest_absence(monkeypatch, tmp_path):
    config = _write_config(
        monkeypatch,
        tmp_path,
        _make_config_model(Model, use_paged_attention=False),
    )

    assert "state_groups" not in config["model"]["decoder"]


def test_qwen_all_attention_builder_emits_paged_kv_group(monkeypatch, tmp_path):
    config = _write_config(
        monkeypatch,
        tmp_path,
        _make_config_model(Qwen35TextModel, layer_types=["full_attention"] * 64),
    )

    assert config["model"]["decoder"]["state_groups"] == [{"kind": "paged_kv", "layer_ids": list(range(64))}]


@pytest.mark.parametrize(
    ("layer_types", "expected"),
    [
        (["sliding_attention"], [{"kind": "paged_kv", "layer_ids": [0]}]),
        (["conv"], [{"kind": "fixed_conv", "layer_ids": [0]}]),
        (
            ["linear_attention"],
            [
                {"kind": "fixed_conv", "layer_ids": [0]},
                {"kind": "fixed_recurrent", "layer_ids": [0]},
            ],
        ),
    ],
)
def test_state_groups_are_added_only_for_matching_layers(layer_types, expected):
    model = _make_config_model(Model, layer_types=layer_types)

    assert model.make_decoder_state_groups({}, {}) == expected


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
        "fixed_conv",
        "fixed_recurrent",
    ]
    assert groups[0]["layer_ids"] == list(range(3, 64, 4))
    assert groups[1]["layer_ids"] == [i for i in range(64) if (i + 1) % 4 != 0]
    assert groups[2]["layer_ids"] == groups[1]["layer_ids"]
    decoder = config["model"]["decoder"]
    assert decoder["inputs"]["past_conv_names"] == "past.%d.conv"
    assert decoder["inputs"]["past_recurrent_names"] == "past.%d.recurrent"
    assert decoder["outputs"]["present_conv_names"] == "present.%d.conv"
    assert decoder["outputs"]["present_recurrent_names"] == "present.%d.recurrent"


def test_qwen38_compact_state_update_bindings_are_emitted(monkeypatch, tmp_path):
    model = _make_config_model(Qwen35TextModel, layer_types=["linear_attention", "full_attention"])
    model.context_length_attrs["state_update_capacity"] = 3
    model.input_names["state_update.capture_count"] = "state_update_capture_count"
    model.input_names["state_update.active"] = "state_update_active"
    model.output_names["state_update.conv_value"] = {0: "state_update.0.conv_value"}
    model.output_names["state_update.recurrent_capsule"] = {0: "state_update.0.recurrent_capsule"}

    decoder = _write_config(monkeypatch, tmp_path, model)["model"]["decoder"]

    assert decoder["state_update_capacity"] == 3
    assert decoder["inputs"]["state_update_capture_count"] == "state_update_capture_count"
    assert decoder["inputs"]["state_update_active"] == "state_update_active"
    assert decoder["outputs"]["state_update_conv_value_names"] == "state_update.%d.conv_value"
    assert decoder["outputs"]["state_update_recurrent_capsule_names"] == "state_update.%d.recurrent_capsule"


def test_qwen38_capacity_without_bindings_is_not_recorded(monkeypatch, tmp_path):
    model = _make_config_model(Qwen35TextModel, layer_types=["linear_attention", "full_attention"])
    model.context_length_attrs["state_update_capacity"] = 3

    decoder = _write_config(monkeypatch, tmp_path, model)["model"]["decoder"]

    assert "state_update_capacity" not in decoder


def test_varlen_ops_emit_compact_state_updates_at_exact_slots():
    model = _recording_model()
    model.make_varlen_causal_conv_with_state(
        "/conv",
        root_input="x",
        weight="weight",
        cumulative_sequence_length="cu",
        bias="bias",
        past_conv_state="past",
        present_conv_state="present",
        state_update_capture_count="capture_count",
        state_update_capacity=3,
        state_update_value="state_update.0.conv_value",
        output_shape=["num_tokens", 48],
        present_conv_shape=["batch_size", 48, 3],
        state_update_value_shape=["batch_size", 3, 48],
    )
    conv = model.nodes[-1][1]
    assert conv["inputs"] == ["x", "weight", "cu", "bias", "past", "capture_count"]
    assert conv["outputs"] == ["/conv/output_0", "present", "state_update.0.conv_value"]
    assert conv["state_update_capacity"] == 3

    model.make_varlen_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="a",
        beta="b",
        a_log="a_log",
        dt_bias="dt_bias",
        gate_shape=["num_tokens", 3],
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        scale=0.0,
        state_update_capture_count="capture_count",
        state_update_active="active",
        state_update_capacity=3,
        state_update_capsule="state_update.0.recurrent_capsule",
        state_update_capsule_shape=["batch_size", 105],
        output_shape=["num_tokens", 3, 8],
        present_recurrent_shape=["batch_size", 3, 8, 4],
    )
    gdn = model.nodes[-1][1]
    assert gdn["inputs"] == [
        "q",
        "k",
        "v",
        "cu",
        "/gdn/decay_fp32/Cast/output_0",
        "/gdn/beta_fp32/Cast/output_0",
        "past",
        "a_log",
        "dt_bias",
        "capture_count",
        "active",
    ]
    assert gdn["outputs"] == ["/gdn/output_0", "present", "state_update.0.recurrent_capsule"]
    assert gdn["state_update_capacity"] == 3
    assert gdn["scale"] == 0.0


def test_varlen_ops_omit_compact_state_updates_at_zero_capacity():
    model = _recording_model()
    model.make_varlen_causal_conv_with_state(
        "/conv",
        root_input="x",
        weight="weight",
        cumulative_sequence_length="cu",
        bias="bias",
        past_conv_state="past",
        present_conv_state="present",
        state_update_capacity=0,
        output_shape=["num_tokens", 48],
        present_conv_shape=["batch_size", 48, 3],
    )
    conv = model.nodes[-1][1]
    assert conv["inputs"] == ["x", "weight", "cu", "bias", "past"]
    assert conv["outputs"] == ["/conv/output_0", "present"]
    assert "state_update_capacity" not in conv

    model.make_varlen_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="a",
        beta="b",
        a_log="a_log",
        dt_bias="dt_bias",
        gate_shape=["num_tokens", 3],
        state_update_capacity=0,
        output_shape=["num_tokens", 3, 8],
        present_recurrent_shape=["batch_size", 3, 8, 4],
    )
    gdn = model.nodes[-1][1]
    assert gdn["inputs"] == [
        "q",
        "k",
        "v",
        "cu",
        "/gdn/decay_fp32/Cast/output_0",
        "/gdn/beta_fp32/Cast/output_0",
        "past",
        "a_log",
        "dt_bias",
    ]
    assert gdn["outputs"] == ["/gdn/output_0", "present"]
    assert "state_update_capacity" not in gdn


def test_varlen_gated_delta_net_requires_compact_state_update_input_pair():
    model = _recording_model()
    common = {
        "q_path": "q",
        "k_path": "k",
        "v_path": "v",
        "cumulative_sequence_length": "cu",
        "past_recurrent_state": "past",
        "present_recurrent_state": "present",
        "decay": "a",
        "beta": "b",
        "a_log": "a_log",
        "dt_bias": "dt_bias",
        "gate_shape": ["num_tokens", 3],
        "state_update_capacity": 3,
        "state_update_capsule": "capsule",
        "state_update_capsule_shape": ["batch_size", 105],
        "output_shape": ["num_tokens", 3, 8],
        "present_recurrent_shape": ["batch_size", 3, 8, 4],
    }

    with pytest.raises(ValueError, match="state_update_capture_count and state_update_active are required"):
        model.make_varlen_gated_delta_net("/gdn", **common)

    with pytest.raises(
        ValueError, match="state_update_capture_count and state_update_active must be provided together"
    ):
        model.make_varlen_gated_delta_net("/gdn", state_update_capture_count="capture_count", **common)


def test_dense_and_packed_gated_delta_net_share_one_attribute_policy():
    model = _recording_model()
    shared = dict(
        q_path="q",
        k_path="k",
        v_path="v",
        decay="a",
        beta="b",
        gate_shape=["batch_size", "sequence_length", 3],
        output_shape=["batch_size", "sequence_length", 3, 8],
    )
    model.make_gated_delta_net("/gdn", initial_state="past", final_state="present", state_shape=[3], **shared)
    dense = model.nodes[-1][1]

    model.make_varlen_gated_delta_net(
        "/varlen_gdn",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        present_recurrent_shape=[3],
        **shared,
    )
    packed = model.nodes[-1][1]

    attributes = ("update_rule", "scale", "gate_activation", "beta_activation", "qk_l2_norm", "chunk_size")
    assert {key: dense[key] for key in attributes} == {key: packed[key] for key in attributes}
    assert dense["scale"] == 0.0
    assert dense["chunk_size"] == 64


def test_dense_gated_delta_net_casts_gates_to_float32():
    model = _recording_model()
    model.make_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        initial_state="past",
        final_state="present",
        decay="a",
        beta="b",
        gate_shape=["batch_size", "sequence_length", 3],
        output_shape=["batch_size", "sequence_length", 3, 8],
        state_shape=["batch_size", 3, 8, 4],
    )

    assert model.casts == [
        ("/gdn/decay_fp32/Cast", "a", base_module.ir.DataType.FLOAT, ["batch_size", "sequence_length", 3]),
        ("/gdn/beta_fp32/Cast", "b", base_module.ir.DataType.FLOAT, ["batch_size", "sequence_length", 3]),
    ]
    assert model.nodes[-1][1]["inputs"][4:7] == [
        "/gdn/decay_fp32/Cast/output_0",
        "/gdn/beta_fp32/Cast/output_0",
        "past",
    ]


def test_dense_gated_delta_net_supports_bfloat16_io():
    model = _recording_model()
    model.io_dtype = base_module.ir.DataType.BFLOAT16
    model.make_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        initial_state="past",
        final_state="present",
        decay="a",
        beta="b",
        gate_shape=["batch_size", "sequence_length", 3],
        output_shape=["batch_size", "sequence_length", 3, 8],
        state_shape=["batch_size", 3, 8, 4],
    )

    assert model.values[-2] == (
        "/gdn/output_0",
        base_module.ir.DataType.BFLOAT16,
        ["batch_size", "sequence_length", 3, 8],
    )
    assert model.values[-1] == (
        "present",
        base_module.ir.DataType.FLOAT,
        ["batch_size", 3, 8, 4],
    )


@pytest.mark.parametrize(
    ("use_paged_attention", "linear_attn_op", "state_window", "ep", "message"),
    [
        (True, "linear_attention", 0, "webgpu", "CUDA execution provider"),
        (False, "gated_delta_net", 0, "webgpu", "CUDA execution provider"),
        (True, "linear_attention", 1, "cuda", "require state_window=0"),
        (False, "gated_delta_net", 1, "cuda", "require state_window=0"),
    ],
)
def test_qwen35_gated_delta_net_option_validation(
    use_paged_attention,
    linear_attn_op,
    state_window,
    ep,
    message,
):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    with pytest.raises(ValueError, match=message):
        model.validate_gated_delta_net_options(
            use_paged_attention,
            linear_attn_op,
            state_window,
            ep,
        )


def _packed_gated_delta_net_model(
    capacity=3,
    layer_types=("linear_attention", "full_attention"),
    io_dtype=base_module.ir.DataType.FLOAT16,
):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.linear_attn_op = "gated_delta_net"
    model.ep = "cuda"
    model.context_length_attrs = {"state_window": 0, "state_update_capacity": capacity}
    model.io_dtype = io_dtype
    model.layer_types = list(layer_types)
    model.linear_conv_dim = 48
    model.linear_key_dim = 8
    model.linear_value_dim = 24
    model.linear_conv_kernel_dim = 4
    model.linear_num_key_heads = 2
    model.linear_num_value_heads = 3
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.input_names = {}
    model.input_types = {"past.conv": model.io_dtype, "past.recurrent": model.io_dtype}
    model.input_shapes = {"past.conv": [4, "old"], "past.recurrent": [4, "old"]}
    model.output_names = {}
    model.output_types = {"present.conv": model.io_dtype, "present.recurrent": model.io_dtype}
    model.output_shapes = {"present.conv": [4, "old"], "present.recurrent": [4, "old"]}
    return model


def test_qwen38_packed_bfloat16_io_keeps_recurrent_state_float32():
    model = _packed_gated_delta_net_model(io_dtype=base_module.ir.DataType.BFLOAT16)

    model.configure_gated_delta_net_io()

    assert model.input_types["past.conv"] == base_module.ir.DataType.BFLOAT16
    assert model.output_types["present.conv"] == base_module.ir.DataType.BFLOAT16
    assert model.input_types["past.recurrent"] == base_module.ir.DataType.FLOAT
    assert model.output_types["present.recurrent"] == base_module.ir.DataType.FLOAT


def test_qwen38_packed_io_uses_unwindowed_v_major_state_and_compact_updates():
    model = _packed_gated_delta_net_model()

    model.configure_gated_delta_net_io()

    assert model.input_shapes["past.conv"] == ["batch_size", 48, 3]
    assert model.output_shapes["present.conv"] == ["batch_size", 48, 3]
    assert model.input_types["past.recurrent"] == base_module.ir.DataType.FLOAT
    assert model.input_shapes["past.recurrent"] == ["batch_size", 3, 8, 4]
    assert model.output_shapes["state_update.conv_value"] == ["batch_size", 3, 48]
    assert model.output_shapes["state_update.recurrent_capsule"] == [
        "batch_size",
        3 * (3 + 2 * 4 + 3 * 8),
    ]
    assert model.output_names["state_update.recurrent_capsule"] == {0: "state_update.0.recurrent_capsule"}


def test_qwen38_capacity_is_cleared_without_linear_attention_layers():
    model = _packed_gated_delta_net_model(layer_types=("full_attention",))

    model.configure_gated_delta_net_io()

    assert model.context_length_attrs["state_update_capacity"] == 0
    assert "state_update.capture_count" not in model.input_names


def test_qwen38_packed_layer_reuses_declared_state_update_bindings(monkeypatch):
    model = _packed_gated_delta_net_model()
    model.configure_gated_delta_net_io()
    model.input_names["cumulative_sequence_lengths"] = "cumulative_sequence_lengths"
    model.input_names["past.recurrent"] = {0: "past"}
    model.output_names["present.recurrent"] = {0: "present"}
    model.make_split = lambda *args, **kwargs: None
    model.make_reshape = lambda *args, **kwargs: None
    model.make_initializer = lambda *args, **kwargs: None
    calls = []
    monkeypatch.setattr(Model, "make_varlen_gated_delta_net", lambda self, name, **kwargs: calls.append(kwargs))

    model.make_gated_delta_net_layer(0, SimpleNamespace(A_log="a_log", dt_bias="dt_bias"), "conv", "b", "a")

    gdn = calls[0]
    assert gdn["state_update_capture_count"] == model.input_names["state_update.capture_count"]
    assert gdn["state_update_active"] == model.input_names["state_update.active"]
    assert gdn["state_update_capsule"] == model.output_names["state_update.recurrent_capsule"][0]
    assert gdn["state_update_capsule_shape"] == model.output_shapes["state_update.recurrent_capsule"]
    assert gdn["present_recurrent_shape"] == model.output_shapes["present.recurrent"]
    assert model.make_conv_state_update_kwargs(0) == {
        "state_update_capacity": 3,
        "state_update_capture_count": model.input_names["state_update.capture_count"],
        "state_update_value": model.output_names["state_update.conv_value"][0],
        "state_update_value_shape": model.output_shapes["state_update.conv_value"],
    }


def test_qwen38_dense_gated_delta_net_exports_raw_a_log(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = False
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.linear_key_dim = 8
    model.linear_value_dim = 24
    model.linear_num_key_heads = 2
    model.linear_num_value_heads = 3
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.input_names = {"past.recurrent": {1: "past"}}
    model.output_names = {"present.recurrent": {1: "present"}}
    model.output_shapes = {"present.recurrent": ["batch_size", 3, 8, 4]}
    model.make_split = lambda *args, **kwargs: None
    model.make_reshape = lambda *args, **kwargs: None
    initializers = []
    model.make_initializer = lambda *args, **kwargs: initializers.append((args, kwargs))
    calls = []
    monkeypatch.setattr(Model, "make_gated_delta_net", lambda self, name, **kwargs: calls.append((name, kwargs)))

    a_log = qwen_module.torch.tensor([0.25, -0.5, 0.0])
    output = model.make_gated_delta_net_layer(
        1,
        SimpleNamespace(A_log=a_log, dt_bias="dt_bias_value"),
        "conv_output",
        "b",
        "a",
    )

    gdn = calls[0][1]
    assert gdn["a_log"] == "model.layers.1.linear_attn.A_log"
    assert gdn["state_shape"] == ["batch_size", 3, 8, 4]
    assert gdn["gate_shape"] == ["batch_size", "sequence_length", 3]
    assert "state_update_capacity" not in gdn
    exported_a_log = next(args[0] for args, _ in initializers if args[1] == gdn["a_log"])
    qwen_module.torch.testing.assert_close(exported_a_log, a_log)
    assert output.endswith("/gdn_out/Reshape/output_0")
