# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Helpers that load the ``tools/qwen38_graph`` scripts in the unit tests.

The scripts import each other by bare module name and pull in ``onnxruntime_genai`` (and, depending on the script, ``torch``,
``transformers`` or the research evaluation libraries). The tests only exercise their pure logic, so ``load_tools`` replaces those
imports with stubs for the duration of the import and removes the scripts, the stubs and the ``sys.path`` entry again afterwards.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools" / "qwen38_graph"
KV_CALIBRATION_SOURCE = REPO_ROOT / "src" / "python" / "py" / "models" / "quantization" / "kv_cache_calibration.py"


def make_onnxruntime_genai_stub():
    og = types.ModuleType("onnxruntime_genai")
    og.FinishReason = types.SimpleNamespace(EOS="eos", LENGTH="length")
    og.EngineEventFlags = types.SimpleNamespace(FAILED=1, CAPACITY_BLOCKED=2, TOKEN=4, TURN_FINISHED=8)
    return og


def make_kv_calibration_stubs():
    """Stubs for the imports of ``calibrate_qwen38_kv.py``.

    The numpy-only helpers of ``kv_cache_calibration.py`` are loaded from the repository sources, so the tests use the code of this
    checkout rather than an installed wheel.
    """
    name = "onnxruntime_genai.models.quantization.kv_cache_calibration"
    spec = importlib.util.spec_from_file_location(name, KV_CALIBRATION_SOURCE)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    return {
        "onnxruntime_genai.models": types.ModuleType("onnxruntime_genai.models"),
        "onnxruntime_genai.models.quantization": types.ModuleType("onnxruntime_genai.models.quantization"),
        name: calibration,
        "torch": types.ModuleType("torch"),
        "transformers": transformers,
    }


def _drop_tool_modules():
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve().parent == TOOLS_DIR:
            del sys.modules[name]


@contextlib.contextmanager
def load_tools(*names, stubs=None):
    """Yields the ``tools/qwen38_graph`` modules ``names``, imported with ``stubs`` (module name -> module) installed.

    ``onnxruntime_genai`` is always replaced by a minimal stub unless ``stubs`` provides one.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TOOLS_DIR))
        patch.setitem(sys.modules, "onnxruntime_genai", make_onnxruntime_genai_stub())
        for name, module in (stubs or {}).items():
            patch.setitem(sys.modules, name, module)
        _drop_tool_modules()
        try:
            yield [importlib.import_module(name) for name in names]
        finally:
            _drop_tool_modules()
