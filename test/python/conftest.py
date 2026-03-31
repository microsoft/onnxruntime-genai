# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import functools
import os

import onnxruntime_genai as og
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


def create_model_for_device(model_path, device):
    """Create an og.Model, disabling graph capture for DML to avoid
    failures when not all nodes are partitioned to the DML EP."""
    config = og.Config(model_path)
    config.clear_providers()
    if device == "dml":
        config.append_provider("dml")
        config.set_provider_option("dml", "enable_graph_capture", "0")
    else:
        config.append_provider(device)
    return og.Model(config)


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
def test_data_path(request):
    return request.config.getoption("--test_models")
