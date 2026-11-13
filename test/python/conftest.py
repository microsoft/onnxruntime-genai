# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import functools
import logging
import os

import pytest
from _test_utils import register_plugin_providers

register_plugin_providers(logging.getLogger(__name__))


def pytest_addoption(parser):
    parser.addoption(
        "--test_models",
        help="Path to the current working directory",
        type=str,
        default=None,
    )
    from integration import models  # noqa: PLC0415

    group = parser.getgroup("integration")
    group.addoption(
        "--model-root",
        action="store",
        default=None,
        help="Root directory containing the foundrylocalmodels layout (overrides ORTGENAI_MODEL_ROOT).",
    )
    group.addoption(
        "--model",
        action="append",
        default=[],
        choices=list(models.MODELS),
        help="Logical model id to test (repeatable). Defaults to every entry in MODELS.",
    )
    group.addoption(
        "--execution-provider",
        action="append",
        default=[],
        choices=list(models.DEVICE_DIRNAMES),
        help="Execution providers to test (repeatable). Defaults to cpu only.",
    )
    group.addoption(
        "--dflash2-webgpu-model",
        default=None,
        help="Exported paged Qwen3.8 WebGPU DFlash2 model directory for opt-in precision, token and activity tests.",
    )
    group.addoption(
        "--run-engine-tests",
        action="store_true",
        default=False,
        help="Run the opt-in real-model Engine tests in engine/test_integration_engine.py.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "engine: paged-attention Engine integration test; opt in with --run-engine-tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-engine-tests"):
        return
    skip_engine = pytest.mark.skip(reason="Engine integration tests are opt-in; pass --run-engine-tests.")
    for item in items:
        if item.get_closest_marker("engine") is not None:
            item.add_marker(skip_engine)


def get_path_for_model(data_path, model_name, precision, device):
    if not data_path:
        pytest.skip("--test_models not provided")
    model_path = os.path.join(data_path, model_name, precision, device)
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_name} not found at {model_path}")
    return model_path


@pytest.fixture
def phi2_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-2",
        "int4",
    )


@pytest.fixture
def phi3_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-3-mini",
        "int4",
    )


@pytest.fixture
def phi4_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-4-mini",
        "int4",
    )


@pytest.fixture
def gemma_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "gemma",
        "int4",
    )


@pytest.fixture
def llama_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "llama",
        "int4",
    )


@pytest.fixture
def qwen_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "qwen-2.5-0.5b",
        "int4",
    )


@pytest.fixture
def qwen_graph_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "qwen-2.5-0.5b-graph",
        "int4",
    )


@pytest.fixture
def phi4_graph_for(request):
    return functools.partial(
        get_path_for_model,
        request.config.getoption("--test_models"),
        "phi-4-mini-graph",
        "int4",
    )


@pytest.fixture
def path_for_model(request):
    return functools.partial(get_path_for_model, request.config.getoption("--test_models"))


# ASR streaming model directories exercised by the streaming-ASR tests.
ASR_SPEECH_MODELS = [
    "nemotron-speech-streaming",
    "moonshine-streaming-small",
    "moonshine-streaming-tiny",
]


@pytest.fixture(params=ASR_SPEECH_MODELS)
def asr_speech_model_path(request):
    """Return the path to an ASR streaming model directory, or skip if not available.

    Parametrized over every known ASR streaming model so that each test using this
    fixture runs once per model (and is skipped for any model that isn't present).
    """
    test_data = request.config.getoption("--test_models")
    model_path = os.path.join(test_data, request.param)
    if not os.path.exists(model_path):
        pytest.skip(f"ASR speech model not found at {model_path}")
    return model_path


@pytest.fixture
def parakeet_tdt_model_path(request):
    """Return the path to a parakeet-tdt model directory, or skip if not available."""
    test_data = request.config.getoption("--test_models")
    model_path = os.path.join(test_data, "parakeet-tdt")
    if not os.path.exists(os.path.join(model_path, "genai_config.json")):
        pytest.skip(f"Parakeet TDT model not found at {model_path}")
    return model_path


@pytest.fixture
def test_data_path(request):
    return request.config.getoption("--test_models")
