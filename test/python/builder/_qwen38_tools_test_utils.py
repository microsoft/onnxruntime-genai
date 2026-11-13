# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Helpers that load the ``tools/python/qwen38_flash_next`` scripts in the unit tests.

The scripts import each other by bare module name and pull in ``onnxruntime_genai`` (and, depending on the script, ``torch``,
``transformers`` or the research evaluation libraries). The tests only exercise their pure logic, so ``load_tools`` replaces those
imports with stubs for the duration of the import and removes the scripts, the stubs and the ``sys.path`` entry again afterwards.

The module also describes the ``model.qwen38_graph`` section of ``genai_config.json`` that the ``qwen38_flash_graph`` engine
accepts, so that the tests can assert that the tools emit nothing the engine's config parser rejects.
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
TOOLS_DIR = REPO_ROOT / "tools" / "python" / "qwen38_flash_next"
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
    """Yields the ``tools/python/qwen38_flash_next`` modules ``names``, imported with ``stubs`` (module name -> module) installed.

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


# The keys below ``model.qwen38_graph`` that the engine's config parser (src/config.cpp, model type ``qwen38_flash_graph``) knows.
# It rejects every other key when the package is loaded, e.g. the former ``fold.mtp`` / ``fold.drafter`` head sessions,
# ``fold.force_reject``, ``mtp.skip_refeed``, ``prefill_chunk_tokens`` and the obsolete top-level CUDA graph id keys.
# Positive, distinct legacy ``fold.target_graph_id`` / ``fold.head_r2.graph_id`` are still accepted. A leaf maps to ``None``;
# the ``session_options`` objects are standard session options, which this description does not look into.
QWEN38_GRAPH_KEYS = {
    "plane_library": None,
    "ple_table": None,
    "kv_capacity": None,
    "max_batch_size": None,
    "warmup": None,
    "head": {"filename_s1": None, "filename_s2": None, "session_options": None},
    "mtp": {"enabled": None, "prefill_head": None},
    "fold": {
        "enabled": None,
        "target_graph_id": None,
        "head_r2": {"filename": None, "graph_id": None, "session_options": None},
    },
    "prefill": {
        "enabled": None,
        "chunk": None,
        "weights_pool": None,
        "constants": None,
        "warmup_lengths": None,
        "dense": {"filename": None, "session_options": None},
        "head_kv": {"filename": None, "session_options": None},
    },
}

# The keys of earlier ``qwen38_graph`` configs that the parser no longer accepts, as dotted paths below ``model.qwen38_graph``.
REMOVED_QWEN38_GRAPH_KEYS = (
    "mtp.skip_refeed",
    "fold.force_reject",
    "fold.mtp",
    "fold.drafter",
    "prefill_chunk_tokens",
    "target_s1_graph_id",
    "target_s2_graph_id",
    "head_s1_graph_id",
    "head_s2_graph_id",
)


def unknown_qwen38_graph_keys(section, known=QWEN38_GRAPH_KEYS, path="model.qwen38_graph"):
    """Returns the dotted paths of the keys of ``section`` (a ``model.qwen38_graph`` object or overlay) the engine rejects."""
    unknown = []
    for key, value in section.items():
        if key not in known:
            unknown.append(f"{path}.{key}")
        elif isinstance(known[key], dict):
            if isinstance(value, dict):
                unknown += unknown_qwen38_graph_keys(value, known[key], f"{path}.{key}")
            else:
                unknown.append(f"{path}.{key} is not an object")
    return unknown


def prefill_chunk_problem(section):
    """Returns why the engine rejects ``prefill.chunk`` of the complete ``model.qwen38_graph`` object ``section``, or None.

    ``chunk`` 0 (or absent) selects 8192 tokens with ``prefill.enabled`` and 2048 otherwise; a negative value is rejected;
    the chunk has to be at most ``kv_capacity`` with ``prefill.enabled`` and even without it.
    """
    prefill = section.get("prefill", {})
    chunk = prefill.get("chunk", 0)
    if chunk < 0:
        return "prefill.chunk is negative"
    enabled = prefill.get("enabled", False)
    chunk = chunk or (8192 if enabled else 2048)
    if enabled:
        return None if chunk <= section.get("kv_capacity", 0) else f"prefill.chunk {chunk} exceeds kv_capacity"
    return None if chunk % 2 == 0 else f"prefill.chunk {chunk} is odd while prefill.enabled is false"


_NO_MEMORY_BOUND = 2**64 - 1


def _eligibilities_intersect(a, b):
    """``EligibilitiesIntersect`` of ``src/config.cpp``: whether one device could satisfy both eligibility objects."""

    def lower(key):
        return max(a.get(key, 0), b.get(key, 0))

    def upper(key):
        return min(a.get(key, _NO_MEMORY_BOUND), b.get(key, _NO_MEMORY_BOUND))

    minimum_free = lower("minimum_free_device_memory_bytes")
    maximum_total = upper("maximum_total_device_memory_bytes")
    maximum_host = upper("maximum_effective_host_memory_bytes")
    integrated = (a.get("is_integrated"), b.get("is_integrated"))
    return (
        lower("minimum_total_device_memory_bytes") <= maximum_total
        and minimum_free <= upper("maximum_free_device_memory_bytes")
        and minimum_free <= maximum_total
        and lower("minimum_effective_host_memory_bytes") <= maximum_host
        and lower("minimum_available_host_memory_bytes") <= maximum_host
        and (None in integrated or integrated[0] == integrated[1])
    )


def runtime_profiles_problem(profiles):
    """Returns why the engine's ``ValidateRuntimeProfiles`` (``src/config.cpp``) rejects ``runtime_profiles``, or None.

    Every profile needs a unique id and ``minimum_total_device_memory_bytes``, no maximum may be below its minimum,
    and no device may satisfy the eligibility of two profiles.
    """
    ids = set()
    for profile in profiles:
        profile_id = profile.get("id")
        if not profile_id:
            return "runtime profile id must not be empty"
        if profile_id in ids:
            return f"duplicate runtime profile id: {profile_id}"
        ids.add(profile_id)
        eligibility = profile.get("eligibility", {})
        if "minimum_total_device_memory_bytes" not in eligibility:
            return f"runtime profile '{profile_id}' is missing minimum_total_device_memory_bytes"
        for kind in ("total_device", "free_device", "effective_host"):
            if eligibility.get(f"maximum_{kind}_memory_bytes", _NO_MEMORY_BOUND) < eligibility.get(
                f"minimum_{kind}_memory_bytes", 0
            ):
                return f"runtime profile '{profile_id}' has maximum_{kind}_memory_bytes below its minimum"
        if eligibility.get("minimum_free_device_memory_bytes", 0) > eligibility.get(
            "maximum_total_device_memory_bytes", _NO_MEMORY_BOUND
        ):
            return f"runtime profile '{profile_id}' requires more free device memory than its maximum total"
        if eligibility.get("minimum_available_host_memory_bytes", 0) > eligibility.get(
            "maximum_effective_host_memory_bytes", _NO_MEMORY_BOUND
        ):
            return (
                f"runtime profile '{profile_id}' requires more available host memory "
                "than its maximum effective host memory"
            )
    for index, first in enumerate(profiles):
        for second in profiles[index + 1 :]:
            if _eligibilities_intersect(first["eligibility"], second["eligibility"]):
                return f"runtime profile eligibility ranges overlap: '{first['id']}' and '{second['id']}'"
    return None
