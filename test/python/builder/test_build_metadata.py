# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for ONNX export provenance stamped by the model builder."""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import onnx_ir as ir

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