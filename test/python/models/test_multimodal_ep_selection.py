# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _test_utils as utils
import onnxruntime as ort
import onnxruntime_genai as og
import pytest


def test_explicit_multimodal_providers_keep_cpu(monkeypatch):
    monkeypatch.setenv("ORTGENAI_MULTIMODAL_TEST_EPS", "CUDA,webgpu,cuda")
    assert utils.multimodal_test_devices() == ("cpu", "cuda", "webgpu")


@pytest.mark.parametrize("value", ["", "cpu,", "unknown", "dml"])
def test_unknown_or_unsupported_multimodal_provider_fails(monkeypatch, value):
    monkeypatch.setenv("ORTGENAI_MULTIMODAL_TEST_EPS", value)
    with pytest.raises(ValueError, match="Unknown multimodal"):
        utils.multimodal_test_devices()


def test_compiled_webgpu_support_does_not_admit_cpu_fallback(monkeypatch):
    monkeypatch.setattr(og, "is_webgpu_available", lambda: True)
    monkeypatch.setattr(utils, "register_plugin_ep", lambda provider: False)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        utils.require_execution_provider("webgpu")


def test_cuda_plugin_requires_genai_device_library(monkeypatch):
    monkeypatch.setattr(utils, "register_plugin_ep", lambda provider: True)
    monkeypatch.setattr(og, "is_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GenAI device library"):
        utils.require_execution_provider("cuda")


def test_plugin_registration_errors_are_not_skipped(monkeypatch):
    def fail(provider):
        raise RuntimeError("plugin initialization failed")

    monkeypatch.setattr(utils, "register_plugin_ep", fail)
    with pytest.raises(RuntimeError, match="plugin initialization failed"):
        utils.require_execution_provider("webgpu")
