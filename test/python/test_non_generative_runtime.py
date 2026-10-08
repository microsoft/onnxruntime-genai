# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import onnxruntime_genai as og
import pytest


class _NativeSession:
    def __init__(
        self,
        path,
        providers,
        cache_capacity,
        cache_capacity_bytes,
        prefix_reuse=True,
        prefix_cache_capacity=32,
        prefix_cache_capacity_bytes=512 * 1024 * 1024,
    ):
        self.path = path
        self.providers = providers
        self.capacity = cache_capacity
        self.capacity_bytes = cache_capacity_bytes
        self.requests = []
        self.cleared = 0
        self.prefix_reuse_enabled = prefix_reuse
        self.prefix_reuse_status = "compatible explicit state I/O"
        self.prefix_capacity = prefix_cache_capacity
        self.prefix_capacity_bytes = prefix_cache_capacity_bytes

    def run(self, request):
        self.requests.append(request)
        return {"answer": {"type": "noul", "noul": 0.75}}

    def cache_stats(self):
        return {
            "hits": 2,
            "misses": 1,
            "evictions": 0,
            "entries": 1,
            "bytes": 8,
            "capacity": self.capacity,
            "capacity_bytes": self.capacity_bytes,
        }

    def clear_cache(self):
        self.cleared += 1

    def invalidate_cache(self):
        self.cleared += 1

    def set_cache_capacity(self, capacity, capacity_bytes):
        self.capacity = capacity
        self.capacity_bytes = capacity_bytes

    def prefix_cache_stats(self):
        return {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "entries": 0,
            "bytes": 0,
            "capacity": self.prefix_capacity,
            "capacity_bytes": self.prefix_capacity_bytes,
            "prefix_runs": 0,
            "branch_runs": 0,
            "fallback_runs": 0,
        }

    def set_prefix_cache_capacity(self, capacity, capacity_bytes):
        self.prefix_capacity = capacity
        self.prefix_capacity_bytes = capacity_bytes


@pytest.fixture
def runtime():
    previous_package = sys.modules.get("onnxruntime_genai")
    previous_native = sys.modules.get("onnxruntime_genai.onnxruntime_genai")
    native = types.ModuleType("onnxruntime_genai.onnxruntime_genai")
    native.ComponentSession = object
    native._RankingSession = _NativeSession
    native._DecisionSession = _NativeSession
    package = types.ModuleType("onnxruntime_genai")
    package.__path__ = []
    sys.modules["onnxruntime_genai"] = package
    sys.modules["onnxruntime_genai.onnxruntime_genai"] = native
    source = Path(__file__).parents[2] / "src/python/py/non_generative.py"
    spec = importlib.util.spec_from_file_location("_non_generative_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if previous_package is None:
        sys.modules.pop("onnxruntime_genai", None)
    else:
        sys.modules["onnxruntime_genai"] = previous_package
    if previous_native is None:
        sys.modules.pop("onnxruntime_genai.onnxruntime_genai", None)
    else:
        sys.modules["onnxruntime_genai.onnxruntime_genai"] = previous_native


@pytest.mark.parametrize(
    ("session_name", "method_name"),
    (("RankingSession", "rank"), ("DecisionSession", "decide")),
)
def test_python_session_is_thin_native_dict_adapter(runtime, session_name, method_name):
    request = {
        "state": {"weather": "rain"},
        "questions": {"q": {"type": "noul", "instructions": "Take umbrella?"}},
    }
    session = getattr(runtime, session_name)("/models/package", providers=["cuda"])
    assert getattr(session, method_name)(request) == {"answer": {"type": "noul", "noul": 0.75}}
    assert session._native.path == "/models/package"
    assert session._native.providers == ["cuda"]
    assert session._native.requests == [request]


@pytest.mark.parametrize("session_name", ["RankingSession", "DecisionSession"])
def test_python_cache_controls(runtime, session_name):
    session = getattr(runtime, session_name)("/models/package", cache_capacity=3, cache_capacity_bytes=99)
    assert session.cache_stats["capacity"] == 3
    session.clear_cache()
    session.invalidate_cache()
    assert session._native.cleared == 2
    session.set_cache_capacity(0, 0)
    assert session.cache_stats["capacity"] == 0
    if session_name == "DecisionSession":
        assert session.prefix_reuse_enabled
        assert session.prefix_reuse_status == "compatible explicit state I/O"
        session.prefix_reuse_enabled = False
        session.set_prefix_cache_capacity(1, 8)
        assert session.prefix_cache_stats["capacity"] == 1


def test_precomputed_actions_remain_pinned_across_clear():
    root_value = os.getenv("ORT_GENAI_NON_GENERATIVE_TEST_ROOT")
    if not root_value:
        pytest.skip("set ORT_GENAI_NON_GENERATIVE_TEST_ROOT to exported packages")
    package = (
        Path(root_value) / "clm-v0.1-8b-fp16-bf16-fallback"
    )
    if not (package / "precomputed_action_projections.bin").is_file():
        pytest.skip("precomputed CLM package is unavailable")
    session = og.RankingSession(package, providers=["cuda"])
    initial = session.cache_stats
    if initial["entries"] == 0:
        pytest.skip("package has no precomputed actions")

    session.clear_cache()

    assert session.cache_stats["entries"] == initial["entries"]


@pytest.mark.skipif(
    os.getenv("ORT_GENAI_RUN_NON_GENERATIVE_INTEGRATION") != "1",
    reason="requires opt-in multi-gigabyte model packages",
)
@pytest.mark.parametrize("provider", [None, "cuda"])
def test_exported_packages_cpu_cuda_parity(provider):
    if provider == "cuda" and os.getenv("ORT_GENAI_RUN_NON_GENERATIVE_CUDA") != "1":
        pytest.skip("set ORT_GENAI_RUN_NON_GENERATIVE_CUDA=1 for CUDA")
    root_value = os.getenv("ORT_GENAI_NON_GENERATIVE_TEST_ROOT")
    if not root_value:
        pytest.skip("set ORT_GENAI_NON_GENERATIVE_TEST_ROOT to exported packages")
    root = Path(root_value)
    kwargs = {"providers": [provider]} if provider else {}
    clm_request = json.loads((root / "clm-request.json").read_text())
    kev_request = json.loads((root / "kev-request.json").read_text())
    assert og.RankingSession(root / "clm-v0.1-8b-fp32", **kwargs).rank(clm_request)
    full = og.DecisionSession(root / "kev-4b-fp32", prefix_reuse=False, **kwargs)
    optimized = og.DecisionSession(root / "kev-4b-fp32", prefix_reuse=True, **kwargs)
    start = time.perf_counter()
    full_result = full.decide(kev_request)
    full_seconds = time.perf_counter() - start
    start = time.perf_counter()
    optimized_result = optimized.decide(kev_request)
    optimized_seconds = time.perf_counter() - start
    assert optimized_result == full_result
    start = time.perf_counter()
    assert optimized.decide(kev_request) == full_result
    cached_seconds = time.perf_counter() - start
    stats = optimized.prefix_cache_stats
    assert stats["prefix_runs"] == 1
    assert stats["hits"] == 1
    print(
        {
            "provider": provider or "cpu",
            "full_seconds": full_seconds,
            "optimized_seconds": optimized_seconds,
            "cached_seconds": cached_seconds,
        }
    )
