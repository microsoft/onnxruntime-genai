# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Execution-provider availability helpers shared by the integration tests.

Kept in one module (rather than copied into each test file) so the one-time
WebGPU plug-in registration is genuinely one-time within a pytest process:
registering the same EP library twice throws, which is exactly what happened
when two test files each carried their own registration helper.
"""

from __future__ import annotations

import onnxruntime_genai as og

# Known (platform, device, model) combinations that don't fit on the agent's
# GPU memory. TODO: re-enable these once the GPU agents have more VRAM. The
# Windows CUDA pool (onnxruntime-Win2022-GPU-A10) and the Linux CUDA pool
# (onnxruntime-Linux-GPU-A10) only expose ~4 GB to the job.
VRAM_CONSTRAINED_SKIPS: set[tuple[str, str, str]] = {
    ("win32", "cuda", "ministral-3-3b-Instruct-2512"),
    ("win32", "cuda", "Phi-4-mini-instruct"),
    ("linux", "cuda", "ministral-3-3b-Instruct-2512"),
    ("linux", "cuda", "Phi-4-mini-instruct"),
}

def register_webgpu_plugin_once() -> bool:
    """Register the onnxruntime-ep-webgpu plug-in once per process.

    The base onnxruntime package doesn't ship a WebGPU EP; the plug-in package
    provides it as a separate shared library that must be registered with ORT
    GenAI before ``append_provider("webgpu")`` works. Returns True if the EP is
    available (registration succeeded or already happened), False if the plug-in
    package isn't installed. The once-per-process flag lives on the function so a
    single shared instance guards the registration for every importer.
    """
    if getattr(register_webgpu_plugin_once, "_registered", False):
        return True
    try:
        import onnxruntime_ep_webgpu as webgpu_ep  # noqa: PLC0415
    except ImportError:
        return False
    og.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())
    register_webgpu_plugin_once._registered = True
    return True


def ep_available(device: str) -> bool:
    """Whether ``device`` can actually run in this build/environment."""
    if device == "cpu":
        return True
    if device == "cuda":
        return og.is_cuda_available()
    if device == "webgpu":
        return register_webgpu_plugin_once()
    return False
