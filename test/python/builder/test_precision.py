# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for `--precision int8` support in the model builder.

int8 precision maps `onnx_dtype` to INT8/UINT8 (mirroring int4 -> INT4/UINT4), builds a
float graph, and quantizes the dense weights to 8-bit `MatMulNBits` at save time via
`to_nbits`. These tests exercise the precision plumbing standalone (no model download).
"""

from __future__ import annotations

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


def test_paged_attention_metadata_is_int32_triplet(monkeypatch):
    base = _load_base_module()
    monkeypatch.setattr(base.Model, "make_ep_expansions_init", lambda self: None)
    monkeypatch.setattr(base.Model, "make_inputs_init", lambda self: None)
    config = types.SimpleNamespace(
        architectures=["TestModel"],
        hidden_act="silu",
        hidden_size=64,
        intermediate_size=128,
        max_position_embeddings=1024,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=2,
        vocab_size=256,
        _name_or_path="test",
    )

    model = base.Model(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {})
    other_model = base.Model(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {})

    assert model.input_types["attention_metadata"] == ir.DataType.INT32
    assert model.input_shapes["attention_metadata"] == [3]
    assert model.input_shapes["attention_metadata"] is not other_model.input_shapes["attention_metadata"]


def test_num_hidden_layers_truncates_configured_layer_types(monkeypatch):
    base = _load_base_module()
    monkeypatch.setattr(base.Model, "make_ep_expansions_init", lambda self: None)
    monkeypatch.setattr(base.Model, "make_inputs_init", lambda self: None)
    config = types.SimpleNamespace(
        architectures=["TestModel"],
        hidden_act="silu",
        hidden_size=64,
        intermediate_size=128,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=1024,
        num_attention_heads=8,
        num_hidden_layers=4,
        num_key_value_heads=2,
        vocab_size=256,
        _name_or_path="test",
    )

    model = base.Model(
        config,
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT16,
        "cuda",
        None,
        {"num_hidden_layers": 2},
    )

    assert model.num_layers == 2
    assert model.layer_types == ["linear_attention", "linear_attention"]


def _load_builder_entrypoint_module():
    # `builder.py` imports the concrete model classes via `from builders import (...)`.
    # Provide a stub `builders` module so we can import the lightweight precision helpers
    # (`set_onnx_dtype` / `set_io_dtype` / `check_extra_options`) without pulling in every
    # model builder.
    builders_stub = types.ModuleType("builders")

    def _stub_getattr(name):  # PEP 562: satisfies `from builders import <ModelClass>`
        return type(name, (), {})

    builders_stub.__getattr__ = _stub_getattr
    # Submodule imports (e.g. `from quantization import ...`) must resolve to the
    # real, dependency-free modules rather than the catch-all above.
    builders_stub.__path__ = [str(BUILDERS_DIR)]
    sys.modules["builders"] = builders_stub

    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model


def test_add_special_token_ids_uses_first_available_candidate():
    config = types.SimpleNamespace()
    tokenizer = types.SimpleNamespace(
        get_vocab=lambda: {
            "<tool_call>": "10",
            "<|tool_call|>": "11",
            "<|/tool_call|>": "12",
            "<think>": "13",
            "</think>": "14",
        }
    )

    builder_module.add_special_token_ids(config, tokenizer)

    assert config.bot_token_id == 10
    assert config.eot_token_id == 12
    assert config.bor_token_id == 13
    assert config.eor_token_id == 14


def test_add_special_token_ids_omits_tokens_not_in_vocabulary():
    config = types.SimpleNamespace()
    tokenizer = types.SimpleNamespace(get_vocab=lambda: {})

    builder_module.add_special_token_ids(config, tokenizer)

    assert not hasattr(config, "bot_token_id")
    assert not hasattr(config, "eot_token_id")
    assert not hasattr(config, "bor_token_id")
    assert not hasattr(config, "eor_token_id")


# ---------------------------------------------------------------------------
# int8 precision maps onnx_dtype to INT8/UINT8 (like int4 -> INT4/UINT4).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "is_symmetric, expected",
    [
        (True, ir.DataType.INT8),
        (False, ir.DataType.UINT8),
    ],
)
def test_int8_onnx_dtype_is_int8(is_symmetric, expected):
    assert builder_module.set_onnx_dtype("int8", {"is_symmetric": is_symmetric}) == expected


@pytest.mark.parametrize(
    "is_symmetric, expected",
    [
        (True, ir.DataType.INT4),
        (False, ir.DataType.UINT4),
    ],
)
def test_int4_onnx_dtype_is_still_int4(is_symmetric, expected):
    assert builder_module.set_onnx_dtype("int4", {"is_symmetric": is_symmetric}) == expected


@pytest.mark.parametrize(
    "execution_provider, expected",
    [
        ("cpu", ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT16),
        ("webgpu", ir.DataType.FLOAT16),
    ],
)
def test_int8_io_dtype_is_not_forced_to_fp32(execution_provider, expected):
    # int8 must not assume FP32 I/O everywhere: GPU/WebGPU use FP16, only CPU uses FP32.
    assert builder_module.set_io_dtype("int8", execution_provider, {}) == expected


# ---------------------------------------------------------------------------
# int8's INT8/UINT8 onnx_dtype routes through the MatMulNBits builders, which
# fall back to a float MatMul when the source model is not already quantized.
# ---------------------------------------------------------------------------


def _make_bare_model(onnx_dtype, quant_attrs=None):
    model = Model.__new__(Model)
    model.onnx_dtype = onnx_dtype
    model.quant_attrs = quant_attrs if quant_attrs is not None else {"use_qdq": False}
    return model


@pytest.mark.parametrize("onnx_dtype", [ir.DataType.INT8, ir.DataType.UINT8])
def test_make_matmul_op_int8_falls_back_to_float_when_not_quantized(monkeypatch, onnx_dtype):
    model = _make_bare_model(onnx_dtype)
    sentinel = object()
    monkeypatch.setattr(model, "make_matmul_float", lambda *a, **k: sentinel)

    assert model.make_matmul_op(object(), "/lm_head/MatMul", "root") is sentinel


@pytest.mark.parametrize("onnx_dtype", [ir.DataType.INT8, ir.DataType.UINT8])
def test_make_packed_matmul_int8_falls_back_to_float_when_not_quantized(monkeypatch, onnx_dtype):
    model = _make_bare_model(onnx_dtype)
    sentinel = object()
    monkeypatch.setattr(model, "make_packed_matmul_float", lambda *a, **k: sentinel)

    assert model.make_packed_matmul(object(), object(), object(), "/attn/qkv/MatMul", "root") is sentinel


# ---------------------------------------------------------------------------
# `to_nbits` forwards the requested weight bit width to `MatMulNBitsQuantizer`.
# ---------------------------------------------------------------------------


def _make_quant_model(bits):
    model = Model.__new__(Model)
    model.model = object()
    model.ep = "cpu"  # keep the CUDA prepack post-pass a no-op
    model.matmul_attrs = {"weights_prepacked": 0}
    model.quantization_algo = "default"
    model.int4_customized_weight_config = {}
    model.quant_attrs = {
        "bits": bits,
        "matmul_block_size": 32,
        "is_symmetric": True,
        "accuracy_level": 4,
        "nodes_to_exclude": [],
        "use_qdq": False,
        "op_types_to_quantize": ("MatMul",),
        "algo_config": None,
    }
    return model


class _FakeQuantizer:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self.model = types.SimpleNamespace(model="quantized-proto")

    def process(self):
        pass


@pytest.mark.parametrize("bits", [4, 8])
def test_to_nbits_forwards_requested_bits(monkeypatch, bits):
    _FakeQuantizer.captured = None
    monkeypatch.setattr(base_module, "MatMulNBitsQuantizer", _FakeQuantizer)
    monkeypatch.setattr(base_module.ir, "to_proto", lambda m: m)
    monkeypatch.setattr(base_module.ir, "from_proto", lambda p: p)

    model = _make_quant_model(bits)
    result = model.to_nbits()

    assert _FakeQuantizer.captured is not None
    assert _FakeQuantizer.captured["bits"] == bits
    assert result == "quantized-proto"


def _run_check_extra_options(
    monkeypatch,
    extra_options,
    *,
    precision="int4",
    execution_provider="cpu",
    tie_word_embeddings=True,
    layer_types=None,
):
    # Avoid Hugging Face network/config loading and provide only the config fields needed.
    fake_config = types.SimpleNamespace(tie_word_embeddings=tie_word_embeddings, layer_types=layer_types)

    def _fake_get_hf_details(*_args, **_kwargs):
        return {
            "extra_kwargs": {},
            "hf_name": "fake-model",
            "hf_config": fake_config,
        }

    monkeypatch.setattr(builder_module, "get_hf_details", _fake_get_hf_details)
    builder_module.check_extra_options(
        model_name="fake-model",
        input_path="/tmp/fake-model",
        output_dir="/tmp/fake-output",
        precision=precision,
        execution_provider=execution_provider,
        cache_dir="/tmp/fake-cache",
        extra_options=extra_options,
    )


def test_mtp_quant_config_json_is_parsed(monkeypatch):
    options = {"mtp_quant_config": '{"io_dtype":"bf16","weights":{"type":"int4"}}'}

    _run_check_extra_options(monkeypatch, options)

    assert options["mtp_quant_config"].io_dtype == "bf16"
    assert options["mtp_quant_config"].weights.type == "int4"


def test_parse_extra_options_preserves_equals_inside_json(monkeypatch):
    captured = {}

    def fake_check_extra_options(*args):
        captured.update(args[-1])

    monkeypatch.setattr(builder_module, "check_extra_options", fake_check_extra_options)
    builder_module.parse_extra_options(
        "model",
        "input",
        "output",
        "int4",
        "cuda",
        "cache",
        ['mtp_quant_config={"weights":{"overrides":[{"match":{"name":"name=a"},"exclude":true}]}}'],
    )

    assert captured["mtp_quant_config"] == ('{"weights":{"overrides":[{"match":{"name":"name=a"},"exclude":true}]}}')


def test_qwen35_moe_architecture_selects_composite_builder(monkeypatch, tmp_path):
    captured = {}

    class FakeQwen35MoEModel:
        exclude_embeds = False

        def __init__(self, *args):
            captured["args"] = args

        def make_genai_config(self, *args):
            captured["genai_config"] = args

        def save_processing(self, *args):
            captured["processing"] = args

    config = types.SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
    monkeypatch.setattr(builder_module, "Qwen35MoEModel", FakeQwen35MoEModel)

    builder_module.create_model(
        "fake-model",
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "fp16",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        hf_details={"extra_kwargs": {}, "hf_name": "fake-model", "hf_config": config},
    )

    assert captured["args"][0] is config
    assert captured["args"][5]["config_only"] is True
    assert captured["genai_config"][0] is config
    assert captured["processing"][0] == "fake-model"


def test_state_window_must_be_non_negative(monkeypatch):
    with pytest.raises(ValueError, match="non-negative integer"):
        _run_check_extra_options(monkeypatch, {"state_window": "-1"})


def test_state_window_is_normalized(monkeypatch):
    options = {"state_window": "3"}

    _run_check_extra_options(monkeypatch, options)

    assert options["state_window"] == 3


def test_num_hidden_layers_rejects_more_layers_than_configured(monkeypatch):
    with pytest.raises(ValueError, match="layer_types has 1 entries"):
        _run_check_extra_options(
            monkeypatch,
            {"num_hidden_layers": "2"},
            layer_types=["full_attention"],
        )


def test_num_hidden_layers_is_normalized(monkeypatch):
    options = {"num_hidden_layers": "1"}

    _run_check_extra_options(monkeypatch, options, layer_types=["full_attention"])

    assert options["num_hidden_layers"] == 1


# ---------------------------------------------------------------------------
# int8 rejects the unsupported QDQ format (8-bit MatMulNBits is QOperator-only).
# ---------------------------------------------------------------------------


def test_int8_with_qdq_is_rejected(monkeypatch):
    with pytest.raises(NotImplementedError, match="QDQ"):
        _run_check_extra_options(monkeypatch, {"use_qdq": "true"}, precision="int8")


def test_int4_with_qdq_is_allowed(monkeypatch):
    # QDQ is only rejected for int8; int4 still supports it.
    _run_check_extra_options(monkeypatch, {"use_qdq": "true"}, precision="int4")


@pytest.mark.parametrize(
    "quant_type",
    ["int8_per_tensor", "int8_per_channel", "int4_per_tensor", "int4_per_channel", "fp8_per_tensor", "fp8_per_channel"],
)
def test_kv_cache_quant_scheme_is_accepted_for_supported_providers(monkeypatch, quant_type):
    options = {"kv_cache_quant_scheme": quant_type.upper()}

    _run_check_extra_options(monkeypatch, options, precision="fp16", execution_provider="cuda")

    assert options["kv_cache_quant_scheme"] == quant_type


def test_kv_cache_quant_scheme_rejects_unsupported_value(monkeypatch):
    with pytest.raises(ValueError, match="kv_cache_quant_scheme must be one of"):
        _run_check_extra_options(
            monkeypatch, {"kv_cache_quant_scheme": "int6_per_tensor"}, precision="fp16", execution_provider="cuda"
        )


def test_quantized_kv_cache_rejects_unsupported_provider(monkeypatch):
    with pytest.raises(ValueError, match="only supported for the CPU and CUDA"):
        _run_check_extra_options(
            monkeypatch, {"kv_cache_quant_scheme": "int8_per_tensor"}, precision="fp16", execution_provider="webgpu"
        )


def test_shared_embeddings_with_untied_weights_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="tie_word_embeddings=false"):
        _run_check_extra_options(
            monkeypatch,
            {"shared_embeddings": "true"},
            precision="int4",
            tie_word_embeddings=False,
        )


def test_shared_embeddings_with_tied_weights_is_accepted(monkeypatch):
    # Should not raise when tie_word_embeddings=True
    _run_check_extra_options(
        monkeypatch,
        {"shared_embeddings": "true"},
        precision="int4",
        tie_word_embeddings=True,
    )


def test_shared_embeddings_defaults_to_tied_when_config_ties_embeddings(monkeypatch):
    # When shared_embeddings is not specified, it defaults to tie_word_embeddings value
    # Should not raise because shared_embeddings will default to True when tie_word_embeddings=True
    _run_check_extra_options(
        monkeypatch,
        {},
        precision="int4",
        tie_word_embeddings=True,
    )


def test_shared_embeddings_defaults_to_false_when_config_doesnt_tie_embeddings(monkeypatch):
    # When shared_embeddings is not specified and tie_word_embeddings=False,
    # shared_embeddings will default to False
    _run_check_extra_options(
        monkeypatch,
        {},
        precision="int4",
        tie_word_embeddings=False,
    )


def test_shared_embeddings_handles_none_tie_word_embeddings(monkeypatch):
    # When tie_word_embeddings is None, it should default to False
    with pytest.raises(ValueError, match="tie_word_embeddings=false"):
        _run_check_extra_options(
            monkeypatch,
            {"shared_embeddings": "true"},
            precision="int4",
            tie_word_embeddings=None,
        )


def test_hidden_state_shape_defaults_to_non_paged_for_bare_model():
    model = Model.__new__(Model)
    model.use_paged_attention = False
    model.hidden_size = 64
    assert model.make_hidden_state_shape() == ["batch_size", "sequence_length", 64]


def test_hidden_state_shape_uses_flat_token_axis_for_paged_model():
    model = Model.__new__(Model)
    model.use_paged_attention = True
    model.hidden_size = 64
    assert model.make_hidden_state_shape() == ["num_tokens", 64]
    assert model.make_hidden_state_shape(seq_dim="batch_size") == ["batch_size", 64]


@pytest.mark.parametrize(
    "extra_options, logits_first_dim",
    [
        ({"include_hidden_states": True}, "num_tokens"),
        ({"include_hidden_states": True, "prune_lm_head": True}, "batch_size"),
    ],
)
def test_paged_attention_uses_flat_hidden_states_output_shape(extra_options, logits_first_dim):
    model = Model.__new__(Model)
    model.use_paged_attention = True
    model.io_dtype = ir.DataType.FLOAT16
    model.hidden_size = 64
    model.vocab_size = 128
    model.num_kv_heads = 4
    model.head_size = 16
    model.layer_types = ["full_attention"]
    model.extra_options = extra_options
    model.output_names = {
        "hidden_states": "hidden_states",
        "logits": "logits",
        "present.conv": {},
        "present.recurrent": {},
    }
    model.output_types = {"logits": ir.DataType.FLOAT16}
    model.output_shapes = {
        "hidden_states": ["batch_size", "sequence_length", model.hidden_size],
        "logits": ["batch_size", "sequence_length", model.vocab_size],
        "present.key": [],
        "present.value": [],
    }

    model.make_outputs_init()

    assert model.output_shapes["hidden_states"] == ["num_tokens", model.hidden_size]
    assert model.output_shapes["logits"] == [logits_first_dim, model.vocab_size]
    assert model.prune_lm_head is (logits_first_dim == "batch_size")


@pytest.mark.parametrize(
    "prune_lm_head, logits_first_dim, expected_rows",
    [
        (True, "batch_size", [1, 4, 5]),
        (False, "num_tokens", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_paged_attention_lm_head_pruning(monkeypatch, tmp_path, prune_lm_head, logits_first_dim, expected_rows):
    model = Model.__new__(Model)
    model.use_paged_attention = True
    model.prune_lm_head = prune_lm_head
    model.io_dtype = ir.DataType.FLOAT
    model.hidden_size = 3
    model.vocab_size = 3
    model.input_names = {"cumulative_sequence_lengths": "cumulative_sequence_lengths"}
    model.output_types = {"logits": ir.DataType.FLOAT}
    model.output_shapes = {"logits": [logits_first_dim, model.vocab_size]}
    model.layernorm_attrs = {"output_0": "hidden_states"}
    model.lm_head_attrs = {"scale": 1, "mask": None, "softcap": 0.0}
    model.values = {}
    model.node_names = set()
    graph = ir.Graph(
        inputs=(),
        outputs=(),
        nodes=(),
        opset_imports={"": 21},
        name="paged_logits_test",
    )
    model.model = ir.Model(graph, ir_version=10)
    graph.inputs.append(model.make_value("hidden_states", ir.DataType.FLOAT, ["num_tokens", model.hidden_size]))
    graph.inputs.append(model.make_value("cumulative_sequence_lengths", ir.DataType.INT32, ["batch_size + 1"]))

    def make_matmul(_lm_head, name, root_input, **_kwargs):
        model.make_node("Identity", inputs=[root_input], outputs=["logits"], name=name)
        model.make_value("logits", ir.DataType.FLOAT, [logits_first_dim, model.vocab_size])
        return name

    monkeypatch.setattr(model, "make_matmul", make_matmul)
    model.make_lm_head(types.SimpleNamespace(bias=None))
    graph.outputs.append(model.make_value("logits"))

    model_path = tmp_path / "paged_logits.onnx"
    ir.save(model.model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    hidden_states = np.arange(18, dtype=np.float32).reshape(6, model.hidden_size)
    cumulative_sequence_lengths = np.array([0, 2, 5, 6], dtype=np.int32)
    (logits,) = session.run(
        None,
        {
            "hidden_states": hidden_states,
            "cumulative_sequence_lengths": cumulative_sequence_lengths,
        },
    )

    np.testing.assert_array_equal(logits, hidden_states[expected_rows])
    assert session.get_outputs()[0].shape == [logits_first_dim, model.vocab_size]
    assert model.output_shapes["logits"] == [logits_first_dim, model.vocab_size]


@pytest.mark.parametrize(
    "extra_options, error",
    [
        ({"use_paged_attention": "true", "paged_block_size": "0"}, "paged_block_size"),
        ({"use_paged_attention": "true", "paged_block_size": "128"}, "paged_block_size"),
        ({"use_paged_attention": "true", "paged_chunk_size": "0"}, "paged_chunk_size"),
        ({"use_paged_attention": "true", "paged_chunk_size": "-1"}, "paged_chunk_size"),
        ({"use_paged_attention": "true", "paged_chunk_size": "abc"}, "paged_chunk_size"),
        ({"use_paged_attention": "true", "max_batch_size": "-1"}, "max_batch_size"),
        ({"use_paged_attention": "true", "max_batch_size": "257"}, "max_batch_size"),
        ({"use_paged_attention": "true", "gpu_utilization_factor": "0"}, "gpu_utilization_factor"),
        ({"use_paged_attention": "true", "gpu_utilization_factor": "1.1"}, "gpu_utilization_factor"),
    ],
)
def test_paged_attention_rejects_invalid_engine_options(monkeypatch, extra_options, error):
    with pytest.raises(ValueError, match=error):
        _run_check_extra_options(monkeypatch, extra_options, precision="bf16", execution_provider="cuda")


def test_paged_attention_normalizes_engine_options(monkeypatch):
    extra_options = {
        "use_paged_attention": "true",
        "paged_block_size": "512",
        "paged_chunk_size": "64",
        "gpu_utilization_factor": "0.75",
        "max_batch_size": "32",
    }
    _run_check_extra_options(monkeypatch, extra_options, precision="bf16", execution_provider="cuda")
    assert extra_options["paged_block_size"] == 512
    assert extra_options["paged_chunk_size"] == 64
    assert extra_options["gpu_utilization_factor"] == 0.75
    assert extra_options["max_batch_size"] == 32


@pytest.mark.parametrize(
    "option_value, expected",
    [
        ("true", True),
        ("false", False),
    ],
)
def test_paged_attention_accepts_lm_head_pruning_option(monkeypatch, option_value, expected):
    extra_options = {
        "use_paged_attention": "true",
        "prune_lm_head": option_value,
    }
    _run_check_extra_options(monkeypatch, extra_options, precision="bf16", execution_provider="cuda")
    assert extra_options["prune_lm_head"] is expected


@pytest.mark.parametrize("option", ["exclude_embeds", "exclude_lm_head"])
def test_paged_attention_rejects_incompatible_graph_interfaces(monkeypatch, option):
    with pytest.raises(ValueError, match=option):
        _run_check_extra_options(
            monkeypatch,
            {"use_paged_attention": "true", option: "true"},
            precision="bf16",
            execution_provider="cuda",
        )
