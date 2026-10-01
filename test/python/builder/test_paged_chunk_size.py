# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the ``paged_chunk_size`` builder option.

The option writes ``search.chunk_size``, which caps the prompt tokens a single request
contributes to one engine step. Before this was fixed it was only written for models whose
sliding-window layers are served from a ring of blocks, so passing it for any other paged
model was silently ignored.
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
    spec = importlib.util.spec_from_file_location(
        f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py"
    )
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


def _write_genai_config(monkeypatch, out_dir, extra_options, use_paged_attention=True):
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
    model.use_paged_attention = use_paged_attention
    model.attention_attrs = {"paged_block_size": 256}
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
    model.layer_types = ["full_attention"] * 4
    model.is_local = lambda layer_id: False
    model.input_names = {
        "input_ids": "input_ids",
        "past_key_values.key": [],
        "past_key_values.value": [],
    }
    if use_paged_attention:
        model.input_names.update(
            {
                "block_table": "block_table",
                "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                "past_sequence_lengths": "past_sequence_lengths",
                "attention_metadata": "attention_metadata",
            }
        )
    model.output_names = {"logits": "logits", "present.key": [], "present.value": []}

    model.make_genai_config(hf_config, {}, str(out_dir))
    return json.loads((Path(out_dir) / "genai_config.json").read_text())


def test_genai_config_omits_chunk_size_by_default(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, {"use_paged_attention": True})

    assert "chunk_size" not in config["search"]


@pytest.mark.parametrize("value", [256, 512, 1024])
def test_genai_config_emits_chunk_size_without_a_ring(monkeypatch, tmp_path, value):
    # No sliding-window layers here, which is the case that used to drop the option.
    config = _write_genai_config(
        monkeypatch, tmp_path, {"use_paged_attention": True, "paged_chunk_size": value}
    )

    assert config["search"]["chunk_size"] == value


def test_genai_config_accepts_chunk_size_as_a_string(monkeypatch, tmp_path):
    # Olive and the command line both hand extra options through as strings.
    config = _write_genai_config(
        monkeypatch, tmp_path, {"use_paged_attention": True, "paged_chunk_size": "512"}
    )

    assert config["search"]["chunk_size"] == 512


def test_genai_config_ignores_chunk_size_without_paged_attention(monkeypatch, tmp_path):
    config = _write_genai_config(
        monkeypatch,
        tmp_path,
        {"paged_chunk_size": 512},
        use_paged_attention=False,
    )

    assert "chunk_size" not in config["search"]
