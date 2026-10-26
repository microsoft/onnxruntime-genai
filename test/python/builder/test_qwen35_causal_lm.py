# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Text-only Qwen3.5 checkpoints (`Qwen3_5ForCausalLM`) use the dense Qwen3.5 builder.

`Qwen3_5ForCausalLM.save_pretrained` writes `architectures: ["Qwen3_5ForCausalLM"]` with a
`qwen3_5_text` config, which is what a text-only fine-tune of Qwen3.5 ships as. Its decoder is
the same as the `Qwen3_5ForConditionalGeneration` language model, so it must not be rejected as
an unsupported architecture.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
sys.path.insert(0, str(MODELS_DIR))


def _load_builder_entrypoint_module():
    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder_module = _load_builder_entrypoint_module()


@pytest.mark.parametrize("architecture", ["Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"])
@pytest.mark.parametrize("exclude_embeds,expected_type", [(True, "qwen3_5"), (False, "qwen3_5_text")])
def test_qwen35_text_only_architecture_uses_qwen35_builder(
    monkeypatch, tmp_path, architecture, exclude_embeds, expected_type
):
    captured = {}

    class FakeQwen35Model:
        model_type = "qwen3_5"

        def __init__(self, *args):
            captured["args"] = args
            self.exclude_embeds = exclude_embeds

        def make_genai_config(self, *args):
            captured["model_type"] = self.model_type

        def save_processing(self, *args):
            pass

    config = types.SimpleNamespace(architectures=[architecture])
    monkeypatch.setattr(builder_module, "Qwen35Model", FakeQwen35Model)

    builder_module.create_model(
        "Qwen/Qwen3.5-text",
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "int4",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        hf_details={"extra_kwargs": {}, "hf_name": "Qwen/Qwen3.5-text", "hf_config": config},
    )

    assert captured["args"][0] is config
    assert captured["model_type"] == expected_type
