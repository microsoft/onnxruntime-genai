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
    builders_package = sys.modules.setdefault(
        "models.builders", types.ModuleType("models.builders")
    )
    builders_package.__path__ = [str(BUILDERS_DIR)]
    spec = importlib.util.spec_from_file_location(
        "models.builders.base", BUILDERS_DIR / "base.py"
    )
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
    spec = importlib.util.spec_from_file_location(
        "models_builder_entrypoint", MODELS_DIR / "builder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model


def _make_input_model(include_attention_bias: bool):
    model = Model.__new__(Model)
    model.extra_options = {"include_attention_bias": include_attention_bias}
    model.use_paged_attention = False
    model.input_names = {
        "input_ids": "input_ids",
        "inputs_embeds": "inputs_embeds",
        "attention_mask": "attention_mask",
        "attention_bias": "attention_bias",
        "position_ids": "position_ids",
        "block_table": "block_table",
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
    }
    return model


def test_attention_bias_input_is_opt_in():
    model = _make_input_model(False)

    model.make_inputs_init()

    assert "attention_bias" not in model.input_names


def test_attention_bias_input_is_retained_when_enabled():
    model = _make_input_model(True)

    model.make_inputs_init()

    assert model.input_names["attention_bias"] == "attention_bias"


class _FakeAttentionInitModel:
    make_attention_init = Model.make_attention_init

    def __init__(self, gqa_supported: bool):
        self.use_paged_attention = False
        self.extra_options = {"include_attention_bias": True}
        self.include_attention_bias = True
        self.num_attn_heads = 8
        self.num_kv_heads = 2
        self.head_size = 16
        self.ep = "cpu"
        self.io_dtype = ir.DataType.FLOAT
        self.input_names = {
            "attention_bias": "attention_bias",
            "position_ids": "position_ids",
        }
        self.rope_attrs = {}
        self.matmul_attrs = {"use_lora": False}
        self.attention_attrs = {
            "op_type": "MultiHeadAttention",
            "q_norm": False,
            "k_norm": False,
            "rope": True,
            "use_rope_in_attn": False,
        }
        self._gqa_supported = gqa_supported

    def is_gqa_supported(self):
        return self._gqa_supported

    def is_packed_attn_supported(self):
        return False

    def is_fused_rope_supported(self):
        return True


def test_attention_bias_forces_explicit_rope_and_position_ids():
    model = _FakeAttentionInitModel(gqa_supported=True)

    model.make_attention_init(types.SimpleNamespace())

    assert model.attention_attrs["op_type"] == "GroupQueryAttention"
    assert not model.attention_attrs["use_rope_in_attn"]
    assert model.input_names["position_ids"] == "position_ids"


def test_attention_bias_rejects_non_gqa_attention():
    model = _FakeAttentionInitModel(gqa_supported=False)

    with pytest.raises(
        ValueError, match="include_attention_bias requires GroupQueryAttention"
    ):
        model.make_attention_init(types.SimpleNamespace())


def _check_options(extra_options, monkeypatch):
    monkeypatch.setattr(
        builder_module,
        "get_hf_details",
        lambda *_args, **_kwargs: {
            "hf_config": types.SimpleNamespace(tie_word_embeddings=False)
        },
    )
    builder_module.check_extra_options(
        model_name="model",
        input_path="model",
        output_dir="output",
        precision="fp32",
        execution_provider="cpu",
        cache_dir="cache",
        extra_options=extra_options,
    )


def test_attention_bias_option_parses_boolean(monkeypatch):
    options = {"include_attention_bias": "true"}

    _check_options(options, monkeypatch)

    assert options["include_attention_bias"] is True


def test_attention_bias_rejects_paged_attention(monkeypatch):
    with pytest.raises(
        ValueError,
        match="use_paged_attention cannot be combined with include_attention_bias",
    ):
        _check_options(
            {
                "use_paged_attention": "true",
                "include_attention_bias": "true",
            },
            monkeypatch,
        )
