# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for metadata saved by the model builder."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import onnx_ir as ir
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GenerationConfig, PreTrainedTokenizerFast

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(TEST_ROOT, "src", "python", "py", "models")
BUILDERS_DIR = os.path.join(MODELS_DIR, "builders")
with open(os.path.join(TEST_ROOT, "VERSION_INFO"), encoding="utf-8") as version_info:
    GENAI_VERSION = version_info.read().strip()
sys.path.insert(0, MODELS_DIR)


def _load_base_module():
    sys.modules.setdefault("models", types.ModuleType("models"))
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [BUILDERS_DIR]

    if "models.builders.base" in sys.modules:
        return sys.modules["models.builders.base"]
    spec = importlib.util.spec_from_file_location("models.builders.base", os.path.join(BUILDERS_DIR, "base.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.builders.base"] = module
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
Model = base_module.Model


def _make_empty_onnx_model():
    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22})
    return ir.Model(graph, ir_version=10, producer_name="onnxruntime-genai")


def test_stamp_build_metadata_embeds_commit_in_producer_version(monkeypatch):
    result = types.SimpleNamespace(stdout="0123456789abcdef\n")
    monkeypatch.setattr(base_module.subprocess, "run", lambda *args, **kwargs: result)
    monkeypatch.setattr(base_module.Model, "get_genai_version", classmethod(lambda cls: GENAI_VERSION))
    model = _make_empty_onnx_model()

    Model.stamp_build_metadata(model)

    assert model.producer_version == f"{GENAI_VERSION}+0123456"
    assert not model.metadata_props


def test_stamp_build_metadata_uses_version_when_commit_is_unavailable(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(base_module.subprocess, "run", missing_git)
    monkeypatch.setattr(base_module.Model, "get_genai_version", classmethod(lambda cls: GENAI_VERSION))
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", types.SimpleNamespace(__commit__=""))
    model = _make_empty_onnx_model()

    Model.stamp_build_metadata(model)

    assert model.producer_version == GENAI_VERSION
    assert not model.metadata_props


def test_get_genai_version_uses_installed_package_metadata(monkeypatch):
    monkeypatch.setattr(base_module.os.path, "exists", lambda _: False)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", types.SimpleNamespace(__version__=GENAI_VERSION))

    assert Model.get_genai_version() == GENAI_VERSION


def test_get_genai_commit_uses_installed_package_metadata(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(base_module.subprocess, "run", missing_git)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", types.SimpleNamespace(__commit__="0123456789abcdef"))

    assert Model.get_genai_commit() == "0123456789abcdef"


@pytest.fixture
def processing_model(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(source)
    model = Model.__new__(Model)
    model.model_name_or_path = str(source)
    model.hf_token = False
    model.hf_remote = False
    model.context_length = 128
    return model


@pytest.mark.parametrize("do_sample", [None, False, True])
def test_save_processing_preserves_generation_config(tmp_path, processing_model, do_sample):
    metadata = {"top_p": 0.95, "top_k": 20, "eos_token_id": [1, 2]}
    if do_sample is not None:
        metadata["do_sample"] = do_sample
    source = tmp_path / "source" / "generation_config.json"
    source.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    original = GenerationConfig.from_pretrained(source.parent, local_files_only=True)

    processing_model.save_processing(processing_model.model_name_or_path, {"local_files_only": True}, str(output))

    saved = GenerationConfig.from_pretrained(output, local_files_only=True)
    assert saved.do_sample is original.do_sample
    assert saved.top_p == metadata["top_p"]
    assert saved.top_k == metadata["top_k"]
    assert saved.eos_token_id == metadata["eos_token_id"]
    assert json.loads(source.read_text(encoding="utf-8")) == metadata
    tokenizer = PreTrainedTokenizerFast.from_pretrained(output, local_files_only=True)
    assert tokenizer.model_max_length == processing_model.context_length
    assert tokenizer.eos_token_id == 1


def test_save_processing_without_generation_config(tmp_path, processing_model):
    output = tmp_path / "output"
    output.mkdir()

    processing_model.save_processing(processing_model.model_name_or_path, {"local_files_only": True}, str(output))

    assert not (output / "generation_config.json").exists()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(output, local_files_only=True)
    assert tokenizer.model_max_length == processing_model.context_length
    assert tokenizer.eos_token_id == 1
