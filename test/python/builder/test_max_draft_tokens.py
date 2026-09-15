# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the ``max_draft_tokens`` builder option.

The option writes a ``speculative`` block into ``genai_config.json`` so that a
speculative-decoding model can ship with a verified proposal width instead of
falling back to the runtime default. It is deliberately independent of the
drafter's exported geometry, so it applies to any model and any drafter.
"""

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
Model = base_module.Model


class _NoGenerationConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise FileNotFoundError("no generation_config.json")


def _write_genai_config(monkeypatch, out_dir, extra_options):
    hf_config = SimpleNamespace(bos_token_id=None, eos_token_id=[2], pad_token_id=None)
    monkeypatch.setattr(base_module, "GenerationConfig", _NoGenerationConfig)

    model = Model.__new__(Model)
    model.context_length_attrs = {
        "state_window": 0,
        "state_window_dims": [],
        "window_kv_cache_slack": 0,
        "window_kv_cache": True,
    }
    model.hf_token = None
    model.hf_remote = False
    model.ep = "cuda"
    model.ep_attrs = {"cuda": {}}
    model.extra_options = extra_options
    model.matmul_attrs = {"weights_prepacked": 0}
    model.use_paged_attention = False
    model.past_present_share_buffer = True
    model.context_length = 1024
    model.filename = "model.onnx"
    model.head_size = 16
    model.hidden_size = 128
    model.num_attn_heads = 8
    model.num_kv_heads = 2
    model.num_layers = 4
    model.model_type = "TestForCausalLM"
    model.vocab_size = 32
    model.window_size = 0
    model.is_local = lambda layer_id: False
    model.input_names = {"input_ids": "input_ids", "past_key_values.key": [], "past_key_values.value": []}
    model.output_names = {"logits": "logits", "present.key": [], "present.value": []}

    model.make_genai_config(hf_config, {}, str(out_dir))
    return json.loads((Path(out_dir) / "genai_config.json").read_text())


def test_genai_config_omits_speculative_by_default(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, {})

    assert "speculative" not in config


@pytest.mark.parametrize("value", [1, 6, 16])
def test_genai_config_emits_max_draft_tokens(monkeypatch, tmp_path, value):
    config = _write_genai_config(monkeypatch, tmp_path, {"max_draft_tokens": value})

    assert config["speculative"] == {"max_draft_tokens": value}


def test_genai_config_accepts_max_draft_tokens_as_a_string(monkeypatch, tmp_path):
    # Olive and the command line both hand extra options through as strings.
    config = _write_genai_config(monkeypatch, tmp_path, {"max_draft_tokens": "6"})

    assert config["speculative"] == {"max_draft_tokens": 6}


@pytest.mark.parametrize("value", [0, -1, 17])
def test_genai_config_rejects_out_of_range_max_draft_tokens(monkeypatch, tmp_path, value):
    with pytest.raises(ValueError, match="max_draft_tokens must be between 1 and 16"):
        _write_genai_config(monkeypatch, tmp_path, {"max_draft_tokens": value})


@pytest.mark.parametrize("value", [6.5, "6.5", 0.5, "1e1", "", "six"])
def test_genai_config_rejects_non_integer_max_draft_tokens(monkeypatch, tmp_path, value):
    # The runtime parses this field as an integer, so truncating 6.5 to 6 would silently
    # export a width the caller never asked for.
    with pytest.raises(ValueError, match="max_draft_tokens must be an integer between 1 and 16"):
        _write_genai_config(monkeypatch, tmp_path, {"max_draft_tokens": value})
