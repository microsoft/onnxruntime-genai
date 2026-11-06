# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Keep nested builder packages in the installed Python wheel."""

import runpy
import shutil
from pathlib import Path

import pytest
import setuptools


@pytest.mark.parametrize("telemetry_enabled", [False, True])
def test_wheel_discovers_all_model_packages(tmp_path, monkeypatch, telemetry_enabled):
    root = Path(__file__).resolve().parents[3]
    python_source = root / "src" / "python"
    package = tmp_path / "onnxruntime_genai"
    shutil.copytree(python_source / "py", package, ignore=shutil.ignore_patterns("__pycache__"))
    (package / "__init__.py").touch()
    shutil.copyfile(python_source / "package_description.md", tmp_path / "package_description.md")
    template = (python_source / "setup.py.in").read_text(encoding="utf-8")
    setup_file = tmp_path / "setup.py"
    setup_file.write_text(
        template.replace("@TARGET_NAME@", "onnxruntime-genai")
        .replace("@VERSION_INFO@", "0.0.0")
        .replace("@PYTHON_TELEMETRY_ENABLED@", str(telemetry_enabled)),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(setup_file))

    packages = set(captured["packages"])
    assert ("onnxruntime_genai.telemetry" in packages) is telemetry_enabled
    assert "onnxruntime_genai.models.builders.qwen3_8" in packages
    assert "onnxruntime_genai.models.builders.qwen3_5" in packages
    for initializer in (package / "models").rglob("__init__.py"):
        expected = ".".join(initializer.parent.relative_to(tmp_path).parts)
        assert expected in packages
    assert all(name == "onnxruntime_genai" or name.startswith("onnxruntime_genai.") for name in packages)
