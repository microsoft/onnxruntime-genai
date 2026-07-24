# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pytest configuration for the integration suite."""

from __future__ import annotations

import pytest

from . import models, resolver


def pytest_addoption(parser):
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


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        devices = metafunc.config.getoption("--execution-provider") or ["cpu"]
        metafunc.parametrize("device", devices)
    if "model" in metafunc.fixturenames:
        chosen = metafunc.config.getoption("--model") or list(models.MODELS)
        metafunc.parametrize("model", chosen)


@pytest.fixture
def model_path(device, model, pytestconfig):
    return resolver.get_path_for(model, device, model_root=pytestconfig.getoption("--model-root"))
