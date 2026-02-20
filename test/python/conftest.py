# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import functools
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test_models",
        help="Path to the current working directory",
        type=str,
        required=True,
    )


def get_path_for_model(data_path, model_name, precision, device):
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
def path_for_model(request):
    return functools.partial(get_path_for_model, request.config.getoption("--test_models"))


@pytest.fixture
def nemotron_speech_model_path(request):
    """Return the path to a nemotron_speech model directory, or skip if not available."""
    # Check environment variable first
    env_path = os.environ.get("NEMOTRON_SPEECH_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check test_models directory
    test_data = request.config.getoption("--test_models")
    candidates = [
        os.path.join(test_data, "nemotron-speech-streaming"),
        os.path.join(test_data, "nemotron-speech-streaming-en-0.6b"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Check CI data path
    ci_data = os.path.join(os.path.abspath(os.sep), "data", "ortgenai", "onnx")
    ci_candidate = os.path.join(ci_data, "nemotron-speech-streaming")
    if os.path.isdir(ci_candidate):
        return ci_candidate

    pytest.skip("Nemotron speech model not found. Set NEMOTRON_SPEECH_MODEL_PATH env var.")


@pytest.fixture
def test_data_path(request):
    return request.config.getoption("--test_models")
