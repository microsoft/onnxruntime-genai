# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
import sys

import pytest

from _test_utils import run_subprocess


def pytest_addoption(parser):
    parser.addoption(
        "--test_models",
        help="Path to the current working directory",
        type=str,
        required=True,
    )


def download_phi2(model_path):
    # python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cpu -o model_path
    device = "cpu"  # FIXME: "cuda" if og.is_cuda_available() else "cpu"
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-m",
        "microsoft/phi-2",
        "-p",
        "int4",
        "-e",
        device,
        "-o",
        model_path,
    ]
    run_subprocess(command).check_returncode()


@pytest.fixture
def test_data_path(request):
    def _get_model_path(model_name=None):
        if not model_name:
            return request.config.getoption("--test_models")

        if model_name == "phi-2":
            model_path = os.path.join(
                request.config.getoption("--test_models"), "phi-2"
            )
            if not os.path.exists(model_path):
                download_phi2(model_path)
            return model_path
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    return _get_model_path
