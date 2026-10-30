# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import onnxruntime_genai as og
import pytest


def test_component_runtime_primitives_are_not_public():
    assert not hasattr(og, "ComponentSession")
    assert not hasattr(og.onnxruntime_genai, "ComponentSession")
    assert not hasattr(og.onnxruntime_genai, "_DirectoryTokenizer")


class _Config:
    def __init__(self, path):
        self.path = path
        self.providers = []

    def clear_providers(self):
        self.providers.clear()

    def append_provider(self, provider):
        self.providers.append(provider)


class _NativeSession:
    def __init__(self, config):
        self.config = config
        self.requests = []

    def run(self, request):
        value = json.loads(request)
        self.requests.append(value)
        if "candidates" in value:
            return json.dumps(
                {
                    "model": "clm",
                    "ranked": [
                        {"rank": 1, "key": "yes", "probability": 0.75},
                        {"rank": 2, "key": "no", "probability": 0.25},
                    ],
                }
            )
        return json.dumps(
            {
                "model": "kev",
                "answers": {"q": {"type": "noul", "noul": 0.75}},
            }
        )


@pytest.fixture
def runtime():
    previous_package = sys.modules.get("onnxruntime_genai")
    previous_native = sys.modules.get("onnxruntime_genai.onnxruntime_genai")
    native = types.ModuleType("onnxruntime_genai.onnxruntime_genai")
    native.Config = _Config
    native._NonGenerativeSession = _NativeSession
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


def test_python_session_is_thin_native_dict_adapter(runtime):
    ranking_request = {
        "state": {"weather": "rain"},
        "instructions": "Choose",
        "candidates": {"yes": "Take umbrella", "no": "Leave umbrella"},
    }
    decision_request = {
        "state": {"weather": "rain"},
        "questions": {"q": {"type": "noul", "instructions": "Take umbrella?"}},
    }
    config = runtime.Config("/models/package")
    config.append_provider("cuda")
    session = runtime.NonGenerativeSession(config)
    assert session.run(ranking_request)["ranked"][0] == {
        "rank": 1,
        "key": "yes",
        "probability": 0.75,
    }
    assert session.run(decision_request)["answers"]["q"] == {
        "type": "noul",
        "noul": 0.75,
    }
    assert session._native.config is config
    assert config.path == "/models/package"
    assert config.providers == ["cuda"]
    assert session._native.requests == [ranking_request, decision_request]


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
    def config(package):
        value = og.Config(str(package))
        if provider:
            value.clear_providers()
            value.append_provider(provider)
        return value

    clm_fixture = json.loads((root / "clm-request.json").read_text())
    clm_request = {
        "state": clm_fixture["context"],
        "instructions": clm_fixture["question"],
        "temperature": clm_fixture["temperature"],
        "candidates": {
            str(index): value for index, value in enumerate(clm_fixture["answers"])
        },
    }
    kev_request = json.loads((root / "kev-request.json").read_text())
    assert og.NonGenerativeSession(config(root / "clm-v0.1-8b-fp32")).run(clm_request)
    decision = og.NonGenerativeSession(config(root / "kev-4b-fp32"))
    first = decision.run(kev_request)
    assert decision.run(kev_request) == first
