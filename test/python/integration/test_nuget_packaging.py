# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import importlib
from pathlib import Path
from xml.etree import ElementTree

import pytest


@pytest.fixture
def generators(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/nuget"))
    return (
        importlib.import_module("generate_nuspec_for_native_nuget"),
        importlib.import_module("generate_nuspec_for_custom_nuget"),
    )


@pytest.mark.parametrize("with_ort", [False, True])
def test_cuda_arm64_nuspec_keeps_addon_without_bundling_ort(tmp_path, generators, with_ort):
    native, _ = generators
    directory = tmp_path / "build/linux-arm64/Release"
    directory.mkdir(parents=True)
    core_files = {"libonnxruntime-genai.so", "libonnxruntime-genai-cuda.so"}
    files = set(core_files)
    if with_ort:
        files.update(
            {
                "libonnxruntime.so",
                "libonnxruntime.so.1",
                "libonnxruntime.so.1.30.0",
                "libonnxruntime_providers_shared.so",
                "libonnxruntime_providers_cuda.so",
                "ORT-LICENSE",
                "ORT-ThirdPartyNotices.txt",
            }
        )
    for name in files:
        (directory / name).touch()
    args = argparse.Namespace(
        sources_path=str(tmp_path),
        native_build_path="build",
        build_config="Release",
        package_name="Microsoft.ML.OnnxRuntimeGenAI.Cuda",
    )
    lines = []
    native.generate_files(lines, args)
    entries = ElementTree.fromstring("\n".join(lines)).findall("file")
    payload = {Path(entry.attrib["src"]).name for entry in entries if entry.attrib["target"].startswith("runtimes")}
    assert payload == core_files
    assert not any(entry.attrib["target"].startswith("licenses\\onnxruntime") for entry in entries)


@pytest.mark.parametrize(
    "rid,core,addon,ort_files",
    [
        (
            "win-x64",
            "onnxruntime-genai.dll",
            "onnxruntime-genai-cuda.dll",
            ["onnxruntime.dll", "onnxruntime.lib", "onnxruntime_providers_cuda.dll"],
        ),
        (
            "win-arm64",
            "onnxruntime-genai.dll",
            "onnxruntime-genai-cuda.dll",
            ["onnxruntime.dll", "onnxruntime_providers_shared.dll"],
        ),
        (
            "osx-arm64",
            "libonnxruntime-genai.dylib",
            "libonnxruntime-genai-cuda.dylib",
            ["libonnxruntime.dylib", "libonnxruntime.1.dylib"],
        ),
        (
            "linux-x64",
            "libonnxruntime-genai.so",
            "libonnxruntime-genai-cuda.so",
            ["libonnxruntime.so", "libonnxruntime_providers_shared.so"],
        ),
        (
            "linux-arm64",
            "libonnxruntime-genai.so",
            "libonnxruntime-genai-cuda.so",
            ["libonnxruntime.so.1.30.0", "libonnxruntime_providers_cuda.so"],
        ),
    ],
)
def test_foundry_nuspec_excludes_ort_and_addon(tmp_path, generators, rid, core, addon, ort_files):
    _, custom = generators
    directory = tmp_path / rid / "lib"
    directory.mkdir(parents=True)
    core_alias = core.replace("onnxruntime-genai", "onnxruntime_genai")
    for name in [core, core_alias, addon, *ort_files]:
        (directory / name).touch()
    args = argparse.Namespace(
        root_dir=str(tmp_path),
        package_name="Microsoft.ML.OnnxRuntimeGenAI.Foundry",
        **{
            platform.replace("-", "_"): str(tmp_path / platform)
            for platform in ["win-x64", "win-arm64", "osx-arm64", "linux-x64", "linux-arm64"]
        },
    )
    lines = []
    custom.generate_files(lines, args)
    entries = ElementTree.fromstring("\n".join(lines)).findall("file")
    payload = [entry.attrib["target"] for entry in entries if entry.attrib["target"].startswith("runtimes")]
    assert set(payload) == {f"runtimes/{rid}/native/{core}", f"runtimes/{rid}/native/{core_alias}"}


@pytest.mark.parametrize("rid", ["win-x64", "win-arm64"])
def test_winml_nuspec_excludes_ort_and_keeps_sdk_dependency(tmp_path, generators, rid):
    winml = importlib.import_module("generate_nuspec_for_winml_nuget")
    directory = tmp_path / rid / "lib"
    directory.mkdir(parents=True)
    for name in [
        "onnxruntime-genai.dll",
        "onnxruntime_genai.dll",
        "onnxruntime.dll",
        "onnxruntime.lib",
        "onnxruntime_providers_shared.dll",
    ]:
        (directory / name).touch()
    args = argparse.Namespace(
        root_dir=str(tmp_path),
        package_name="Microsoft.ML.OnnxRuntimeGenAI.WinML",
        package_version="1.0.0",
        ort_package_name="Microsoft.Windows.AI.MachineLearning",
        ort_package_version="2.0.0",
        win_x64=str(tmp_path / "win-x64"),
        win_arm64=str(tmp_path / "win-arm64"),
    )
    package = ElementTree.fromstring("\n".join(winml.generate_nuspec(args)))
    payload = [
        entry.attrib["target"]
        for entry in package.findall("files/file")
        if entry.attrib["target"].startswith("runtimes")
    ]
    assert set(payload) == {
        f"runtimes/{rid}/native/onnxruntime-genai.dll",
        f"runtimes/{rid}/native/onnxruntime_genai.dll",
    }
    dependencies = package.findall("metadata/dependencies/group/dependency")
    assert any(
        entry.attrib == {"id": "Microsoft.Windows.AI.MachineLearning", "version": "2.0.0"} for entry in dependencies
    )
