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

    assert config["model"]["decoder"]["state_groups"] == [
        {
            "kind": "paged_kv",
            "layer_ids": list(range(64)),
            "bindings": {
                "key": {"input": "past_key_values.%d.key", "output": "present.%d.key"},
                "value": {"input": "past_key_values.%d.value", "output": "present.%d.value"},
            },
        }
    ]


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

    assert config["model"]["decoder"]["state_groups"][0]["kind"] == "paged_kv"
    assert config["model"]["decoder"]["state_groups"][0]["layer_ids"] == list(range(64))


@pytest.mark.parametrize(
    ("layer_types", "expected"),
    [
        (
            ["sliding_attention"],
            [
                {
                    "kind": "paged_kv",
                    "layer_ids": [0],
                    "bindings": {
                        "key": {"input": "past_key_values.%d.key", "output": "present.%d.key"},
                        "value": {"input": "past_key_values.%d.value", "output": "present.%d.value"},
                    },
                }
            ],
        ),
        (
            ["conv"],
            [
                {
                    "kind": "fixed",
                    "layer_ids": [0],
                    "bindings": {"state": {"input": "past.%d.conv", "output": "present.%d.conv"}},
                }
            ],
        ),
        (
            ["linear_attention"],
            [
                {
                    "kind": "fixed",
                    "layer_ids": [0],
                    "bindings": {"state": {"input": "past.%d.conv", "output": "present.%d.conv"}},
                },
                {
                    "kind": "fixed",
                    "layer_ids": [0],
                    "bindings": {"state": {"input": "past.%d.recurrent", "output": "present.%d.recurrent"}},
                },
            ],
        ),
    ],
)
def test_state_groups_are_added_only_for_matching_layers(layer_types, expected):
    model = _make_config_model(Model, layer_types=layer_types)

    inputs = {
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value",
        "past_conv_names": "past.%d.conv",
        "past_recurrent_names": "past.%d.recurrent",
    }
    outputs = {
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value",
        "present_conv_names": "present.%d.conv",
        "present_recurrent_names": "present.%d.recurrent",
    }
    assert model.make_decoder_state_groups(inputs, outputs) == expected


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
    decoder = config["model"]["decoder"]
    assert decoder["inputs"]["past_conv_names"] == "past.%d.conv"
    assert decoder["inputs"]["past_recurrent_names"] == "past.%d.recurrent"
    assert decoder["outputs"]["present_conv_names"] == "present.%d.conv"
    assert decoder["outputs"]["present_recurrent_names"] == "present.%d.recurrent"


def test_qwen38_compact_state_groups_are_checkpoint_free(monkeypatch, tmp_path):
    model = _make_config_model(
        Qwen35TextModel,
        layer_types=["linear_attention", "full_attention"],
    )
    model.num_layers = 2
    model.state_update_capacity = 3
    model.linear_num_key_heads = 2
    config = _write_config(monkeypatch, tmp_path, model)

    conv_group, recurrent_group = config["model"]["decoder"]["state_groups"][1:]
    assert conv_group["state_update"] == {
        "kind": "causal_conv",
        "capacity": 3,
        "capture_count": "state_update_capture_count",
        "active": "state_update_active",
        "value": "state_update.%d.conv_value",
    }
    assert recurrent_group["state_update"] == {
        "kind": "gated_delta_net",
        "capacity": 3,
        "capture_count": "state_update_capture_count",
        "active": "state_update_active",
        "capsule": "state_update.%d.recurrent_capsule",
        "key_head_count": 2,
    }
    for group in (conv_group, recurrent_group):
        assert "checkpoints" not in group["bindings"]["state"]
        assert "checkpoint_count" not in group
        assert "checkpoint_alignment" not in group


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
    assert "arithmetic_mode" not in gdn


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


def test_varlen_ops_reject_obsolete_checkpoint_outputs():
    model = _recording_model()
    with pytest.raises(ValueError, match="checkpoint outputs are no longer supported"):
        model.make_varlen_causal_conv_with_state(
            "/conv",
            prefix_conv_state="checkpoint",
        )
    with pytest.raises(ValueError, match="checkpoint outputs are no longer supported"):
        model.make_varlen_gated_delta_net(
            "/gdn",
            checkpoints="checkpoint",
        )


def test_varlen_gated_delta_net_rejects_separate_state_update_outputs():
    model = _recording_model()
    with pytest.raises(ValueError, match="separate state-update outputs are no longer supported"):
        model.make_varlen_gated_delta_net(
            "/gdn",
            state_update_decay="decay_update",
        )


@pytest.mark.parametrize(
    ("capacity", "use_paged_attention", "linear_attn_op", "state_window", "message"),
    [
        (-1, True, "gated_delta_net", 4, "between 0 and 8"),
        (9, True, "gated_delta_net", 10, "between 0 and 8"),
        (3, False, "gated_delta_net", 4, "use_paged_attention=true"),
        (3, True, "linear_attention", 4, "linear_attn_op=gated_delta_net"),
        (3, True, "gated_delta_net", 3, "state_update_capacity \\+ 1"),
    ],
)
def test_qwen38_compact_state_update_option_validation(
    capacity,
    use_paged_attention,
    linear_attn_op,
    state_window,
    message,
):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    with pytest.raises(ValueError, match=message):
        model.validate_state_update_options(
            capacity,
            use_paged_attention,
            linear_attn_op,
            state_window,
        )


@pytest.mark.parametrize("capacity", [True, 1.0, 1.5, "1.5", None])
def test_qwen38_compact_state_update_capacity_requires_integer(capacity):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    with pytest.raises(ValueError, match="must be an integer"):
        model.parse_state_update_capacity(capacity)

    assert model.parse_state_update_capacity("3") == 3


def test_qwen38_packed_io_uses_unwindowed_v_major_state_and_compact_updates():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.linear_attn_op = "gated_delta_net"
    model.state_update_capacity = 3
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.layer_types = ["linear_attention", "full_attention"]
    model.linear_conv_dim = 48
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


def test_qwen38_dense_gated_delta_net_exports_raw_a_log(monkeypatch):
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.linear_key_dim = 8
    model.linear_value_dim = 24
    model.linear_num_key_heads = 2
    model.linear_num_value_heads = 3
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.input_names = {"past.recurrent": {1: "past"}}
    model.output_names = {"present.recurrent": {1: "present"}}
    model.make_split = lambda *args, **kwargs: None
    model.make_reshape = lambda *args, **kwargs: None
    model.make_cast = lambda *args, **kwargs: None
    initializers = []
    model.make_initializer = lambda *args, **kwargs: initializers.append((args, kwargs))
    calls = []
    monkeypatch.setattr(Model, "make_gated_delta_net", lambda self, name, **kwargs: calls.append((name, kwargs)))

    a_log = qwen_module.torch.tensor([0.25, -0.5, 0.0])
    output = model.make_dense_gated_delta_net(
        1,
        SimpleNamespace(A_log=a_log, dt_bias="dt_bias_value"),
        "conv_output",
        "b",
        "a",
    )

    gdn = calls[0][1]
    assert "arithmetic_mode" not in gdn
    assert gdn["a_log"] == "model.layers.1.linear_attn.A_log"
    assert gdn["state_shape"] == ["batch_size", 3, 8, 4]
    exported_a_log = next(args[0] for args, _ in initializers if args[1] == gdn["a_log"])
    qwen_module.torch.testing.assert_close(exported_a_log, a_log)
    assert output.endswith("/gdn_out/Reshape/output_0")
