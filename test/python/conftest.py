# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test_models",
        help="Path to the current working directory",
        type=str,
        required=True,
    )


@pytest.fixture
def test_data_path(request):
    return request.config.getoption("--test_models")
