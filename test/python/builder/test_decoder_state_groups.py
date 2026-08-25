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
    model.use_windowed_paged_kv_cache = False
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
        "fixed_conv",
        "fixed_recurrent",
    ]
    assert groups[0]["layer_ids"] == list(range(3, 64, 4))
    assert groups[1]["layer_ids"] == [i for i in range(64) if (i + 1) % 4 != 0]
    assert groups[2]["layer_ids"] == groups[1]["layer_ids"]
    decoder = config["model"]["decoder"]
    assert decoder["inputs"]["past_conv_names"] == "past_key_values.%d.conv_state"
    assert decoder["inputs"]["past_recurrent_names"] == "past_key_values.%d.recurrent_state"
    assert decoder["outputs"]["present_conv_names"] == "present.%d.conv_state"
    assert decoder["outputs"]["present_recurrent_names"] == "present.%d.recurrent_state"


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
