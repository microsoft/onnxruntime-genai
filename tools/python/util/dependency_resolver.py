# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import platform
import shutil
from os import PathLike, listdir
from os.path import isfile
from pathlib import Path

import requests

from .logger import get_logger
from .platform_helpers import is_linux, is_windows, is_windows_arm

_log = get_logger("util.dependency_resolver")


def _download_ort(
    use_cuda: bool, use_rocm: bool, use_dml: bool, destination_dir: PathLike
):
    def _lib_path():
        plat = "linux" if is_linux() else "win" if is_windows() else "osx"
        mach = None
        if platform.machine().lower() == "x86_64" or platform.machine().lower() == "amd64":
            mach = "x64"
        elif platform.machine().lower() == "aarch64" or platform.machine().lower() == "arm64":
            mach = "arm64"
        else:
            raise NotImplementedError(
                f"Unsupported machine architecture: {platform.machine()}"
            )

        return destination_dir / "ort" / "runtimes" / (plat + "-" + mach) / "native"

    package_name = None
    if use_cuda:
        if is_linux():
            package_name = "Microsoft.ML.OnnxRuntime.Gpu.Linux"
        elif is_windows():
            package_name = "Microsoft.ML.OnnxRuntime.Gpu.Windows"
        else:
            raise NotImplementedError("ORT with CUDA is not supported on this platform")
    elif use_rocm:
        package_name = "Microsoft.ML.OnnxRuntime.Rocm"
    elif use_dml:
        package_name = "Microsoft.ML.OnnxRuntime.DirectML"
    else:
        package_name = "Microsoft.ML.OnnxRuntime"
        if is_windows_arm():
            package_name = "Microsoft.ML.OnnxRuntime.QNN"

    package_path = destination_dir / f"{package_name}.zip"
    if package_path.exists():
        _log.info(f"Package {package_name} already downloaded")
        return _lib_path()

    organization = "aiinfra"
    feed_name = "ORT-Nightly"
    version_fetch_url = f"https://feeds.dev.azure.com/{organization}/PublicPackages/_apis/packaging/Feeds/{feed_name}/packages?packageNameQuery={package_name}&api-version=6.0-preview.1"

    version = requests.get(version_fetch_url).json()["value"][0]["versions"][0][
        "normalizedVersion"
    ]
    feed_project = "2692857e-05ef-43b4-ba9c-ccf1c22c437c"
    feed_id = "7982ae20-ed19-4a35-a362-a96ac99897b7"
    package_url = f"https://pkgs.dev.azure.com/{organization}/{feed_project}/_apis/packaging/feeds/{feed_id}/nuget/packages/{package_name}/versions/{version}/content?api-version=6.0-preview.1"

    _log.info(f"Downloading {package_name} version {version}")
    with open(package_path, "wb") as f:
        f.write(requests.get(package_url).content)

    unpacked_dir = destination_dir / "ort"
    shutil.unpack_archive(package_path, unpacked_dir)

    return _lib_path()


def _download_dml(destination_dir: PathLike):
    def _lib_path():
        mach = None
        if platform.machine().lower() == "x86_64" or platform.machine().lower() == "amd64":
            mach = "x64"
        elif platform.machine().lower() == "aarch64" or platform.machine().lower() == "arm64":
            mach = "arm64"
        else:
            raise NotImplementedError(
                f"Unsupported machine architecture: {platform.machine()}"
            )

        return destination_dir / "dml" / "bin" / (mach + "-win") / "DirectML.dll"

    dml_version = "1.15.2"
    dml_package_name = "Microsoft.AI.DirectML"
    dml_package_url = (
        f"https://www.nuget.org/api/v2/package/{dml_package_name}/{dml_version}"
    )
    package_path = destination_dir / f"{dml_package_name}.zip"
    if package_path.exists():
        _log.info(f"Package {dml_package_name} already downloaded")
        return _lib_path()

    _log.info(f"Downloading {dml_package_name} version {dml_version}")
    with open(package_path, "wb") as f:
        f.write(requests.get(dml_package_url).content)

    unpacked_dir = destination_dir / "dml"
    shutil.unpack_archive(package_path, unpacked_dir)

    return _lib_path()


def _download_d3d12(destination_dir: PathLike):
    def _lib_path():
        mach = None
        if platform.machine().lower() == "x86_64" or platform.machine().lower() == "amd64":
            mach = "x64"
        elif platform.machine().lower() == "aarch64" or platform.machine().lower() == "arm64":
            mach = "arm64"
        else:
            raise NotImplementedError(
                f"Unsupported machine architecture: {platform.machine()}"
            )

        return (
            destination_dir
            / "d3d12"
            / "build"
            / "native"
            / "bin"
            / mach
            / "D3D12Core.dll"
        )

    d3d12_version = "1.614.1"
    d3d12_package_name = "Microsoft.Direct3D.D3D12"
    d3d12_package_url = (
        f"https://www.nuget.org/api/v2/package/{d3d12_package_name}/{d3d12_version}"
    )
    package_path = destination_dir / f"{d3d12_package_name}.zip"
    if package_path.exists():
        _log.info(f"Package {d3d12_package_name} already downloaded")
        return _lib_path()

    _log.info(f"Downloading {d3d12_package_name} version {d3d12_version}")
    with open(package_path, "wb") as f:
        f.write(requests.get(d3d12_package_url).content)

    unpacked_dir = destination_dir / "d3d12"
    shutil.unpack_archive(package_path, unpacked_dir)

    return _lib_path()


def download_dependencies(
    use_cuda: bool, use_rocm: bool, use_dml: bool, destination_dir: PathLike
):
    dependencies_dir = destination_dir / "dependencies"
    if not dependencies_dir.exists():
        dependencies_dir.mkdir(parents=True)

    ort_lib_dir = _download_ort(use_cuda, use_rocm, use_dml, dependencies_dir)
    libs = listdir(ort_lib_dir)
    for file_name in libs:
        if isfile(Path(ort_lib_dir) / file_name):
            shutil.copy(Path(ort_lib_dir) / file_name, destination_dir)

    if use_dml:
        dml_lib_path = _download_dml(dependencies_dir)
        shutil.copy(dml_lib_path, destination_dir)

        d3d12_lib_path = _download_d3d12(dependencies_dir)
        shutil.copy(d3d12_lib_path, destination_dir)

    return dependencies_dir


def copy_dependencies(lib_dir: PathLike, destination_dir: PathLike):
    libs = listdir(lib_dir)
    for file_name in libs:
        shutil.copy(Path(lib_dir) / file_name, destination_dir)
