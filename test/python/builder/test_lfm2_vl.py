# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM2-VL model builder wiring.

LFM2-VL reuses the dense LFM2 builder for its language model. The checkpoint nests that decoder under
`model.language_model`, so the builder has to load it through `Lfm2VlForConditionalGeneration`, find
its `embedding_norm` there, and label the export `lfm2_vl` (a vision pipeline stage) or
`lfm2_vl_text` (a text-only decoder) depending on `exclude_embeds`.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(MODELS_DIR))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


def _load_builder_entrypoint_module():
    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model


@pytest.mark.parametrize("exclude_embeds,expected_type", [(True, "lfm2_vl"), (False, "lfm2_vl_text")])
def test_lfm2_vl_architecture_uses_lfm2_builder_and_labels_the_export(
    monkeypatch, tmp_path, exclude_embeds, expected_type
):
    captured = {}

    class FakeLFM2Model:
        model_type = "lfm2"

        def __init__(self, *args):
            captured["args"] = args
            self.exclude_embeds = exclude_embeds

        def make_genai_config(self, *args):
            captured["model_type"] = self.model_type

        def save_processing(self, *args):
            pass

    config = types.SimpleNamespace(architectures=["Lfm2VlForConditionalGeneration"])
    monkeypatch.setattr(importlib.import_module("builders"), "LFM2Model", FakeLFM2Model)

    builder_module.create_model(
        "LiquidAI/LFM2.5-VL-1.6B",
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "int4",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        hf_details={"extra_kwargs": {}, "hf_name": "LiquidAI/LFM2.5-VL-1.6B", "hf_config": config},
    )

    assert captured["args"][0] is config
    assert captured["model_type"] == expected_type


@pytest.mark.parametrize("model_type", ["lfm2_vl", "lfm2_vl_text"])
def test_lfm2_vl_loads_the_vlm_transformers_model(monkeypatch, model_type):
    # The auto-class lookup is a substring match, so the single "lfm2_vl" entry must also cover
    # "lfm2_vl_text"; falling through to AutoModelForCausalLM cannot load a VLM checkpoint.
    calls = []

    class FakeLfm2VlModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(cls)
            return cls()

    class FakeCausalLM(FakeLfm2VlModel):
        pass

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.Lfm2VlForConditionalGeneration = FakeLfm2VlModel
    transformers_stub.AutoModelForCausalLM = FakeCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    model = Model.__new__(Model)
    model.model_type = model_type
    model.model_name_or_path = "LiquidAI/LFM2.5-VL-1.6B"
    model.cache_dir = "/cache"
    model.hf_token = True
    model.hf_remote = False
    model.quant_type = None
    model.num_layers = 16
    model.extra_options = {}

    loaded_model = model.load_weights("")

    assert calls == [FakeLfm2VlModel]
    assert not isinstance(loaded_model, FakeCausalLM)


def test_has_final_norm_finds_the_nested_lfm2_vl_embedding_norm():
    # LFM2 applies `embedding_norm` after the last layer; in LFM2-VL it lives under `language_model`.
    embedding_norm = object()
    other_module = object()
    language_model = types.SimpleNamespace(embedding_norm=embedding_norm)
    orig_model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=language_model))

    model = Model.__new__(Model)

    assert model.has_final_norm(embedding_norm, orig_model)
    assert not model.has_final_norm(other_module, orig_model)
