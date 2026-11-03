# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize(
    "package,dependency",
    [
        ("onnxruntime-genai", "onnxruntime"),
        ("onnxruntime-genai-cuda", "onnxruntime-gpu"),
        ("onnxruntime-genai-directml", "onnxruntime-directml"),
        ("onnxruntime-genai-trt-rtx", "onnxruntime-trt-rtx"),
        ("onnxruntime-genai-winml", "onnxruntime"),
    ],
)
def test_sdk_wheel_excludes_ort_and_declares_runtime_dependency(tmp_path, package, dependency):
    template = Path(__file__).resolve().parents[2] / "src/python/setup.py.in"
    setup = template.read_text(encoding="utf-8").replace("@TARGET_NAME@", package).replace("@VERSION_INFO@", "1.0.0")
    (tmp_path / "setup.py").write_text(setup, encoding="utf-8")
    (tmp_path / "package_description.md").write_text("Test package", encoding="utf-8")
    package_dir = tmp_path / "onnxruntime_genai"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    genai_files = {
        "onnxruntime-genai.dll",
        "onnxruntime-genai-cuda.dll",
        "onnxruntime_genai.pyd",
        "libonnxruntime-genai.so",
        "libonnxruntime-genai-cuda.so",
        "libonnxruntime-genai.dylib",
    }
    ort_files = {
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
        "libonnxruntime.so",
        "libonnxruntime.so.1.30.0",
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime.dylib",
        "libonnxruntime.1.dylib",
    }
    for name in genai_files | ort_files:
        (package_dir / name).write_bytes(b"native payload")
    environment = os.environ.copy()
    environment["ONNXRUNTIME_VERSION"] = "1.30.0"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", "dist", "."],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with ZipFile(wheel) as archive:
        paths = set(archive.namelist())
        assert all(f"onnxruntime_genai/{name}" in paths for name in genai_files)
        assert not any(f"onnxruntime_genai/{name}" in paths for name in ort_files)
        (metadata_path,) = (name for name in paths if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
    assert metadata["Name"] == package
    assert f"{dependency}>=1.30.0" in metadata.get_all("Requires-Dist")
