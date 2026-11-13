# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pytest configuration for the integration suite."""

from __future__ import annotations

import pytest

from . import models, resolver


def pytest_generate_tests(metafunc):
    if "model_path" not in metafunc.fixturenames:
        return
    if "device" in metafunc.fixturenames:
        devices = metafunc.config.getoption("--execution-provider") or ["cpu"]
        metafunc.parametrize("device", devices)
    if "model" in metafunc.fixturenames:
        chosen = metafunc.config.getoption("--model") or list(models.MODELS)
        metafunc.parametrize("model", chosen)


@pytest.fixture
def model_path(device, model, pytestconfig):
    return resolver.get_path_for(model, device, model_root=pytestconfig.getoption("--model-root"))
