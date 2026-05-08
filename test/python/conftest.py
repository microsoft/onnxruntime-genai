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
    test_data = request.config.getoption("--test_models")
    model_path = os.path.join(test_data, "nemotron-speech-streaming")
    if not os.path.exists(model_path):
        pytest.skip(f"Nemotron speech model not found at {model_path}")
    return model_path


@pytest.fixture
def parakeet_tdt_model_path(request):
    """Return the path to a parakeet_tdt model directory, or skip if not available.

    Accepts any of these layouts for --test_models:
      <test_models>/genai_config.json
      <test_models>/fp32/genai_config.json
      <test_models>/parakeet-tdt/genai_config.json
      <test_models>/parakeet-tdt/fp32/genai_config.json
    """
    test_data = request.config.getoption("--test_models")
    candidates = [
        test_data,
        os.path.join(test_data, "fp32"),
        os.path.join(test_data, "parakeet-tdt"),
        os.path.join(test_data, "parakeet-tdt", "fp32"),
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "genai_config.json")):
            return cand
    pytest.skip(f"Parakeet TDT model not found under {test_data} (looked in {candidates})")


@pytest.fixture
def test_data_path(request):
    """Path containing the bundled audio fixtures (audios/jfk.flac, etc.).

    Defaults to the in-repo `test/test_models` directory so tests work even
    when --test_models points at an external model directory.
    """
    test_data = request.config.getoption("--test_models")
    if os.path.isdir(os.path.join(test_data, "audios")):
        return test_data
    repo_default = os.path.join(os.path.dirname(__file__), "..", "test_models")
    return os.path.abspath(repo_default)
