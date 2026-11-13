# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""``genai_config.json`` must carry the sampling defaults from the checkpoint's ``generation_config.json``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from _builder_test_utils import load_builder_module

transformers = pytest.importorskip("transformers")
if getattr(transformers, "__spec__", None) is None:
    # _builder_test_utils installs a placeholder module when Transformers is missing; it has no GenerationConfig.
    pytest.skip("transformers is not installed", allow_module_level=True)

base_module = load_builder_module("base")
Model = base_module.Model


def _write_genai_config(monkeypatch, out_dir, generation_config):
    hf_config = SimpleNamespace(bos_token_id=None, eos_token_id=[2], pad_token_id=None)
    monkeypatch.setattr(
        base_module, "GenerationConfig", SimpleNamespace(from_pretrained=lambda *args, **kwargs: generation_config)
    )

    model = Model.__new__(Model)
    model.context_length_attrs = {
        "state_window": 0,
        "state_window_dims": [],
        "window_kv_cache_slack": 0,
        "window_kv_cache": True,
    }
    model.model_name_or_path = "fake-model"
    model.resolve_special_token_ids = lambda config, extra_kwargs: (None, [2], None)  # Avoid loading a tokenizer
    model.hf_token = None
    model.hf_remote = False
    model.ep = "cpu"
    model.ep_attrs = {"cpu": {}}
    model.extra_options = {}
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
    return json.loads((Path(out_dir) / "genai_config.json").read_text())["search"]


def test_search_defaults_follow_generation_config(monkeypatch, tmp_path):
    # Qwen2.5-style sampling defaults
    generation_config = transformers.GenerationConfig(
        do_sample=True, temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.05
    )
    search = _write_genai_config(monkeypatch, tmp_path, generation_config)

    assert search["do_sample"] is True
    assert search["temperature"] == 0.7
    assert search["top_k"] == 20
    assert search["top_p"] == 0.8
    assert search["repetition_penalty"] == 1.05


def test_repetition_penalty_defaults_to_one_without_generation_config_value(monkeypatch, tmp_path):
    search = _write_genai_config(monkeypatch, tmp_path, transformers.GenerationConfig())

    assert search["repetition_penalty"] == 1.0
