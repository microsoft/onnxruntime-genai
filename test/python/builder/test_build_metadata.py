# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for ONNX export provenance stamped by the model builder."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
GENAI_VERSION = (Path(__file__).parents[3] / "VERSION_INFO").read_text(encoding="utf-8").strip()
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


base_module = _load_base_module()
Model = base_module.Model


def _make_empty_onnx_model():
    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22})
    return ir.Model(graph, ir_version=10, producer_name="onnxruntime-genai")


def test_stamp_build_metadata_sets_genai_version_and_commit(monkeypatch):
    result = types.SimpleNamespace(stdout="0123456789abcdef\n")
    monkeypatch.setattr(base_module.subprocess, "run", lambda *args, **kwargs: result)
    monkeypatch.setattr(base_module.Model, "get_genai_version", lambda self: GENAI_VERSION)
    builder = Model.__new__(Model)
    model = _make_empty_onnx_model()

    builder.stamp_build_metadata(model)

    assert model.producer_version == GENAI_VERSION
    assert model.metadata_props["producer_commit"] == "0123456789abcdef"


def test_stamp_build_metadata_handles_missing_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(base_module.subprocess, "run", missing_git)
    monkeypatch.setattr(base_module.Model, "get_genai_version", lambda self: GENAI_VERSION)
    builder = Model.__new__(Model)
    model = _make_empty_onnx_model()

    builder.stamp_build_metadata(model)

    assert model.producer_version == GENAI_VERSION
    assert "producer_commit" not in model.metadata_props


def test_get_genai_version_uses_installed_package_metadata(monkeypatch):
    class MissingVersionInfo:
        def resolve(self):
            return self

        @property
        def parents(self):
            return [self] * 6

        def __truediv__(self, _):
            return self

        def read_text(self, **_):
            raise FileNotFoundError

    monkeypatch.setattr(base_module, "Path", lambda _: MissingVersionInfo())
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", types.SimpleNamespace(__version__=GENAI_VERSION))

    assert Model.__new__(Model).get_genai_version() == GENAI_VERSION


def test_get_genai_commit_uses_installed_package_metadata(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(base_module.subprocess, "run", missing_git)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", types.SimpleNamespace(__commit__="0123456789abcdef"))

    assert Model.__new__(Model).get_genai_commit() == "0123456789abcdef"