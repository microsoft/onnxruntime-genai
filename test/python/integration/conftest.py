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
        help="Logical model id to test (repeatable). Defaults to the models eligible for each suite.",
    )
    group.addoption(
        "--execution-provider",
        action="append",
        default=[],
        choices=list(models.DEVICE_DIRNAMES),
        help="Execution providers to test (repeatable). Defaults to cpu only.",
    )
    group.addoption(
        "--run-multimodal-tests",
        action="store_true",
        default=False,
        help="Run pinned real vision-model continuation tests (not part of the text/Engine suites).",
    )
    group.addoption(
        "--multimodal-output-dir",
        default="build/multimodal-integration-results",
        help="Directory for real-model images and per-session ORT partition profiles.",
    )
    group.addoption(
        "--run-engine-tests",
        action="store_true",
        default=False,
        help=(
            "Run the paged-attention Engine integration tests "
            "(test_integration_engine.py). Skipped by default so the "
            "text-generation pipeline, which collects the whole directory, "
            "does not require the pinned paged model artifact."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "multimodal: real vision-model integration test; opt in with --run-multimodal-tests.",
    )
    config.addinivalue_line(
        "markers",
        "engine: paged-attention Engine integration test; opt in with --run-engine-tests.",
    )


def pytest_collection_modifyitems(config, items):
    for marker, option in (("engine", "--run-engine-tests"), ("multimodal", "--run-multimodal-tests")):
        if not config.getoption(option):
            skip = pytest.mark.skip(reason=f"{marker} integration tests are opt-in; pass {option}.")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        devices = metafunc.config.getoption("--execution-provider") or ["cpu"]
        metafunc.parametrize("device", devices)
    if "model" in metafunc.fixturenames:
        is_multimodal = metafunc.definition.get_closest_marker("multimodal") is not None
        eligible = models.multimodal if is_multimodal else [m for m in models.MODELS if m not in models.multimodal]
        chosen = metafunc.config.getoption("--model") or eligible
        metafunc.parametrize("model", [m for m in chosen if m in eligible])


@pytest.fixture
def model_path(device, model, pytestconfig):
    return resolver.get_path_for(model, device, model_root=pytestconfig.getoption("--model-root"))
