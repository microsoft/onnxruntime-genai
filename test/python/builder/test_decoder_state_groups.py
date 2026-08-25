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
    model.linear_num_key_heads = 16
    model.model_type = "Qwen3_5_textForCausalLM"
    model.vocab_size = 248320
    model.window_size = None
    model.use_windowed_paged_kv_cache = False
    model.eps_with_windowed_kv_cache = {"cuda"}
    model.attention_attrs = {"paged_block_size": 256}
    model._state_window = 0
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
    assert "checkpoint_count" not in groups[1]
    assert "checkpoint_count" not in groups[2]
    assert "state_update" not in groups[1]
    assert "state_update" not in groups[2]
    assert "mixed_batch_checkpoints" not in config["model"]["decoder"]


def test_qwen38_state_window_emits_checkpoint_bindings(monkeypatch, tmp_path):
    layer_types = ["full_attention" if (layer_id + 1) % 4 == 0 else "linear_attention" for layer_id in range(64)]
    model = _make_config_model(Qwen35TextModel, layer_types=layer_types)
    model._state_window = 4
    config = _write_config(monkeypatch, tmp_path, model)

    groups = config["model"]["decoder"]["state_groups"]
    assert groups[1]["checkpoint_count"] == 4
    assert groups[1]["checkpoint_alignment"] == "left"
    assert groups[1]["bindings"]["state"]["checkpoints"] == "checkpoints.%d.conv_state"
    assert groups[2]["checkpoint_count"] == 4
    assert groups[2]["checkpoint_alignment"] == "right"
    assert groups[2]["bindings"]["state"]["checkpoints"] == "checkpoints.%d.recurrent_state"
    # mixed_batch_checkpoints stays opt-in: keeping drafts alive in a mixed prefill+decode step
    # still corrupts state even with per-request operator checkpoints. See
    # dev/docs/memory/qwen_3.8_27b_nvfp4_gdn_paged_dflash2_hybrid_dispatch_design.md section 6.2.
    assert "mixed_batch_checkpoints" not in config["model"]["decoder"]


def test_qwen38_compact_state_updates_coexist_with_checkpoints(monkeypatch, tmp_path):
    layer_types = ["linear_attention", "full_attention"]
    model = _make_config_model(Qwen35TextModel, layer_types=layer_types)
    model.num_layers = 2
    model._state_window = 4
    model.state_update_capacity = 3
    config = _write_config(monkeypatch, tmp_path, model)

    conv_group, recurrent_group = config["model"]["decoder"]["state_groups"][1:]
    assert conv_group == {
        "kind": "fixed",
        "layer_ids": [0],
        "bindings": {
            "state": {
                "input": "past_key_values.%d.conv_state",
                "output": "present.%d.conv_state",
                "checkpoints": "checkpoints.%d.conv_state",
            }
        },
        "checkpoint_count": 4,
        "checkpoint_alignment": "left",
        "state_update": {
            "kind": "causal_conv",
            "capacity": 3,
            "capture_count": "state_update_capture_count",
            "active": "state_update_active",
            "value": "state_update.%d.conv_value",
        },
    }
    assert recurrent_group == {
        "kind": "fixed",
        "layer_ids": [0],
        "bindings": {
            "state": {
                "input": "past_key_values.%d.recurrent_state",
                "output": "present.%d.recurrent_state",
                "checkpoints": "checkpoints.%d.recurrent_state",
            }
        },
        "checkpoint_count": 4,
        "checkpoint_alignment": "right",
        "state_update": {
            "kind": "gated_delta_net",
            "capacity": 3,
            "capture_count": "state_update_capture_count",
            "active": "state_update_active",
            "capsule": "state_update.%d.recurrent_capsule",
            "key_head_count": 16,
        },
    }
    assert "mixed_batch_checkpoints" not in config["model"]["decoder"]


def test_qwen38_compact_deployment_omits_dense_checkpoints(monkeypatch, tmp_path):
    layer_types = ["linear_attention", "full_attention"]
    model = _make_config_model(Qwen35TextModel, layer_types=layer_types)
    model.num_layers = 2
    model._state_window = 4
    model.state_update_capacity = 3
    model.state_update_keep_checkpoints = False
    config = _write_config(monkeypatch, tmp_path, model)

    for group in config["model"]["decoder"]["state_groups"][1:]:
        assert "checkpoints" not in group["bindings"]["state"]
        assert "checkpoint_count" not in group
        assert "checkpoint_alignment" not in group
        assert group["state_update"]["capacity"] == 3


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
    with pytest.raises(ValueError, match=message):
        Qwen35TextModel._validate_state_update_options(
            capacity,
            use_paged_attention,
            linear_attn_op,
            state_window,
        )


@pytest.mark.parametrize("capacity", [True, 1.5, "1.5", None])
def test_qwen38_compact_state_update_capacity_requires_integer(capacity):
    with pytest.raises(ValueError, match="must be an integer"):
        Qwen35TextModel._parse_state_update_capacity(capacity)

    assert Qwen35TextModel._parse_state_update_capacity("3") == 3


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


def test_gated_delta_net_fused_gate_inputs_use_schema_slots_before_capture_count():
    model = _recording_model()
    model.make_varlen_gated_delta_net(
        "/gdn",
        q_path="q_raw",
        k_path="k_raw",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        decay="a_raw",
        beta="b_raw",
        a_log="a_log",
        dt_bias="dt_bias",
        gate_shape=["num_tokens", 16],
        gate_activation="qwen",
        beta_activation="sigmoid",
        arithmetic_mode="compatibility",
        qk_l2_norm=1,
        scale=0.0,
        state_update_capture_count="capture_count",
        state_update_active="state_update_active",
        state_update_capacity=7,
        state_update_decay="update_decay",
        state_update_key="update_key",
        state_update_delta="update_delta",
        state_update_shapes=[
            ["batch_size", 7, 16],
            ["batch_size", 7, 16, 128],
            ["batch_size", 7, 16, 128],
        ],
        output_shape=["num_tokens", 16, 128],
        present_recurrent_shape=["batch_size", 16, 128, 128],
    )

    node = model.nodes[-1][1]
    assert node["inputs"] == [
        "q_raw",
        "k_raw",
        "v",
        "cu",
        "/gdn/decay_fp32/Cast/output_0",
        "/gdn/beta_fp32/Cast/output_0",
        "past",
        "a_log",
        "dt_bias",
        "capture_count",
        "state_update_active",
    ]
    assert node["gate_activation"] == "qwen"
    assert node["beta_activation"] == "sigmoid"
    assert node["arithmetic_mode"] == "compatibility"
    assert node["qk_l2_norm"] == 1
    assert node["scale"] == 0.0


def test_varlen_ops_emit_compact_state_update_contract_at_exact_slots():
    model = _recording_model()
    model.make_varlen_causal_conv_with_state(
        "/conv",
        root_input="x",
        weight="w",
        bias="b",
        cumulative_sequence_length="cu",
        past_conv_state="past",
        present_conv_state="present",
        prefix_conv_state="checkpoint",
        max_checkpoints=4,
        state_update_capture_count="state_update_capture_count",
        state_update_capacity=3,
        state_update_value="state_update.0.conv_value",
        output_shape=["num_tokens", 48],
        present_conv_shape=["batch_size", 48, 3],
        prefix_conv_shape=[4, "batch_size", 48, 3],
        state_update_value_shape=["batch_size", 3, 48],
    )
    conv = model.nodes[-1][1]
    assert conv["inputs"] == ["x", "w", "cu", "b", "past", "state_update_capture_count"]
    assert conv["outputs"] == [
        "/conv/output_0",
        "present",
        "checkpoint",
        "state_update.0.conv_value",
    ]
    assert conv["state_update_capacity"] == 3
    assert model.values[-1] == (
        "state_update.0.conv_value",
        base_module.ir.DataType.FLOAT16,
        ["batch_size", 3, 48],
    )

    model.make_varlen_causal_conv_with_state(
        "/compact_conv",
        root_input="x",
        weight="w",
        bias="b",
        cumulative_sequence_length="cu",
        past_conv_state="past",
        present_conv_state="present",
        state_update_capture_count="state_update_capture_count",
        state_update_capacity=3,
        state_update_value="state_update.0.conv_value",
        output_shape=["num_tokens", 48],
        present_conv_shape=["batch_size", 48, 3],
        state_update_value_shape=["batch_size", 3, 48],
    )
    compact_conv = model.nodes[-1][1]
    assert compact_conv["outputs"] == [
        "/compact_conv/output_0",
        "present",
        "",
        "state_update.0.conv_value",
    ]

    model.make_varlen_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        checkpoints="checkpoint",
        state_checkpoints=4,
        decay="decay",
        beta="beta",
        gate_shape=["num_tokens", 3],
        state_update_capture_count="state_update_capture_count",
        state_update_capacity=3,
        state_update_decay="state_update.0.recurrent_decay",
        state_update_key="state_update.0.recurrent_key",
        state_update_delta="state_update.0.recurrent_delta",
        output_shape=["num_tokens", 3, 8],
        present_recurrent_shape=["batch_size", 3, 8, 4],
        checkpoints_shape=[4, "batch_size", 3, 8, 4],
        state_update_shapes=[
            ["batch_size", 3, 3],
            ["batch_size", 3, 2, 4],
            ["batch_size", 3, 3, 8],
        ],
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
        "",
        "",
        "state_update_capture_count",
    ]
    assert gdn["outputs"] == [
        "/gdn/output_0",
        "present",
        "checkpoint",
        "state_update.0.recurrent_decay",
        "state_update.0.recurrent_key",
        "state_update.0.recurrent_delta",
    ]
    assert gdn["state_update_capacity"] == 3
    assert model.values[-3:] == [
        ("state_update.0.recurrent_decay", base_module.ir.DataType.FLOAT, ["batch_size", 3, 3]),
        ("state_update.0.recurrent_key", base_module.ir.DataType.FLOAT, ["batch_size", 3, 2, 4]),
        ("state_update.0.recurrent_delta", base_module.ir.DataType.FLOAT, ["batch_size", 3, 3, 8]),
    ]


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
    model.state_update_capacity = 3
    model.gdn_arithmetic_mode = "compatibility"
    model.input_names = {"cumulative_sequence_lengths": "cu"}
    model._leading_dims = lambda: ["num_tokens"]
    model._make_linear_attention_projections = lambda *args: ("z", "b", "a", "conv_input", "weight")
    initializers = []
    model.make_initializer = lambda *args, **kwargs: initializers.append((args, kwargs))
    conv_calls = []
    model.make_varlen_causal_conv_with_state = lambda name, **kwargs: conv_calls.append((name, kwargs))
    model._make_linear_attention_normalize_and_gate = lambda *args: pytest.fail(
        "packed GDN must fuse normalization and gates"
    )
    nodes = []
    model.make_node = lambda op_type, inputs, outputs, **kwargs: nodes.append((op_type, inputs, outputs, kwargs))
    model.make_value = lambda *args, **kwargs: None
    reshapes = []
    model.make_reshape = lambda name, inputs, dtype, shape: reshapes.append((name, inputs, dtype, shape))
    linear_calls = []
    model.make_varlen_gated_delta_net = lambda name, **kwargs: linear_calls.append((name, kwargs))
    outputs = []
    model._make_linear_attention_output = lambda *args: outputs.append(args)

    a_log = qwen_module.torch.tensor([0.25, -0.5, 0.0])
    model._make_linear_attention(1, SimpleNamespace(A_log=a_log, dt_bias="dt_bias_value"), "root")

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
    assert linear["decay"] == "a/output_0"
    assert linear["beta"] == "b/output_0"
    assert linear["gate_activation"] == "qwen"
    assert linear["beta_activation"] == "sigmoid"
    assert linear["arithmetic_mode"] == "compatibility"
    assert linear["a_log"] == "model.layers.1.linear_attn.neg_exp_A"
    assert linear["qk_l2_norm"] == 1
    assert linear["scale"] == 0.0
    assert linear["state_update_capacity"] == 3
    assert linear["state_update_capsule"] == "state_update.1.recurrent_capsule"
    assert linear["state_update_capsule_shape"] == ["batch_size", 3 * (3 + 2 * 4 + 3 * 8)]
    assert linear["output_shape"] == ["num_tokens", 3, 8]
    assert linear["present_recurrent_shape"] == ["batch_size", 3, 8, 4]
    assert linear["gate_shape"] == ["num_tokens", 3]
    decay_scale = next(args[0] for args, _ in initializers if args[1] == linear["a_log"])
    qwen_module.torch.testing.assert_close(decay_scale, -a_log.exp())
    assert reshapes[-1][3] == ["num_tokens", 24]
    assert outputs[0][2].endswith("/linear_attention_output/Reshape/output_0")


def test_qwen_packed_recurrent_io_is_fp32_v_major():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.layer_types = ["linear_attention"]
    model._state_window_dims = []
    model._state_window = 0
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
    assert "checkpoint_state.0.conv" not in model.output_names


def test_qwen_packed_state_window_declares_checkpoint_outputs():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.layer_types = ["linear_attention"]
    model._state_window_dims = []
    model._state_window = 4
    model.linear_conv_dim = 48
    model.linear_conv_kernel_dim = 4
    model.linear_num_value_heads = 3
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.input_names = {"past_key_values.key": [], "past_key_values.value": []}
    model.output_names = {"present.key": [], "present.value": []}
    model.input_types = {}
    model.input_shapes = {}
    model.output_types = {}
    model.output_shapes = {}

    model._setup_hybrid_cache_io()

    # The committed state keeps its unwindowed shape; only the extra outputs carry the window.
    assert model.input_shapes["past_state.0.recurrent"] == ["batch_size", 3, 8, 4]
    assert model.output_shapes["present_state.0.recurrent"] == ["batch_size", 3, 8, 4]
    assert model.output_names["checkpoint_state.0.conv"] == "checkpoints.0.conv_state"
    assert model.output_shapes["checkpoint_state.0.conv"] == [4, "batch_size", 48, 3]
    assert model.output_types["checkpoint_state.0.conv"] == base_module.ir.DataType.FLOAT16
    assert model.output_names["checkpoint_state.0.recurrent"] == "checkpoints.0.recurrent_state"
    assert model.output_shapes["checkpoint_state.0.recurrent"] == [4, "batch_size", 3, 8, 4]
    assert model.output_types["checkpoint_state.0.recurrent"] == base_module.ir.DataType.FLOAT


def test_qwen_packed_compact_state_update_declares_exact_graph_io():
    model = Qwen35TextModel.__new__(Qwen35TextModel)
    model.use_paged_attention = True
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.layer_types = ["linear_attention"]
    model._state_window_dims = []
    model._state_window = 4
    model.state_update_capacity = 3
    model.linear_conv_dim = 48
    model.linear_conv_kernel_dim = 4
    model.linear_num_key_heads = 2
    model.linear_num_value_heads = 3
    model.linear_key_head_dim = 4
    model.linear_value_head_dim = 8
    model.input_names = {"past_key_values.key": [], "past_key_values.value": []}
    model.output_names = {"present.key": [], "present.value": []}
    model.input_types = {}
    model.input_shapes = {}
    model.output_types = {}
    model.output_shapes = {}

    model._setup_hybrid_cache_io()

    assert model.input_names["state_update.capture_count"] == "state_update_capture_count"
    assert model.input_types["state_update.capture_count"] == base_module.ir.DataType.INT32
    assert model.input_shapes["state_update.capture_count"] == ["batch_size"]
    assert model.input_names["state_update.active"] == "state_update_active"
    assert model.input_types["state_update.active"] == base_module.ir.DataType.INT32
    assert model.input_shapes["state_update.active"] == [1]
    assert model.output_shapes["state_update.0.conv_value"] == ["batch_size", 3, 48]
    assert model.output_types["state_update.0.conv_value"] == base_module.ir.DataType.FLOAT16
    assert model.output_shapes["state_update.0.recurrent_capsule"] == [
        "batch_size",
        3 * (3 + 2 * 4 + 3 * 8),
    ]
    assert model.output_types["state_update.0.recurrent_capsule"] == base_module.ir.DataType.FLOAT


def test_varlen_ops_emit_checkpoint_outputs():
    model = _recording_model()
    model.make_varlen_causal_conv_with_state(
        "/conv",
        root_input="x",
        weight="w",
        bias="b",
        cumulative_sequence_length="cu",
        past_conv_state="past",
        present_conv_state="present",
        prefix_conv_state="prefix",
        max_checkpoints=4,
        output_shape=["num_tokens", 48],
        present_conv_shape=["batch_size", 48, 3],
        prefix_conv_shape=[4, "batch_size", 48, 3],
    )
    conv = model.nodes[-1][1]
    assert conv["outputs"] == ["/conv/output_0", "present", "prefix"]
    assert conv["max_checkpoints"] == 4
    assert model.values[-1] == ("prefix", base_module.ir.DataType.FLOAT16, [4, "batch_size", 48, 3])

    model.make_varlen_gated_delta_net(
        "/gdn",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_length="cu",
        past_recurrent_state="past",
        present_recurrent_state="present",
        checkpoints="ckpt",
        state_checkpoints=4,
        decay="decay",
        beta="beta",
        gate_shape=["num_tokens", 3],
        output_shape=["num_tokens", 3, 8],
        present_recurrent_shape=["batch_size", 3, 8, 4],
        checkpoints_shape=[4, "batch_size", 3, 8, 4],
    )
    gdn = model.nodes[-1][1]
    assert gdn["outputs"] == ["/gdn/output_0", "present", "ckpt"]
    assert gdn["state_checkpoints"] == 4
    assert model.values[-1] == ("ckpt", base_module.ir.DataType.FLOAT, [4, "batch_size", 3, 8, 4])


def test_varlen_ops_reject_half_configured_checkpoints():
    model = _recording_model()
    with pytest.raises(ValueError, match="must be set together"):
        model.make_varlen_causal_conv_with_state(
            "/conv",
            root_input="x",
            weight="w",
            bias="b",
            cumulative_sequence_length="cu",
            past_conv_state="past",
            present_conv_state="present",
            max_checkpoints=4,
            output_shape=["num_tokens", 48],
            present_conv_shape=["batch_size", 48, 3],
        )
    with pytest.raises(ValueError, match="must be set together"):
        model.make_varlen_gated_delta_net(
            "/gdn",
            q_path="q",
            k_path="k",
            v_path="v",
            cumulative_sequence_length="cu",
            past_recurrent_state="past",
            present_recurrent_state="present",
            checkpoints="ckpt",
            decay="decay",
            beta="beta",
            gate_shape=["num_tokens", 3],
            output_shape=["num_tokens", 3, 8],
            present_recurrent_shape=["batch_size", 3, 8, 4],
        )
