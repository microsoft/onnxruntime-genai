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


def _download_ort(use_cuda: bool, use_dml: bool, destination_dir: PathLike):
    def _lib_path():
        plat = "linux" if is_linux() else "win" if is_windows() else "osx"
        mach = None
        if platform.machine().lower() == "x86_64" or platform.machine().lower() == "amd64":
            mach = "x64"
        elif platform.machine().lower() == "aarch64" or platform.machine().lower() == "arm64":
            mach = "arm64"
        else:
            raise NotImplementedError(f"Unsupported machine architecture: {platform.machine()}")

        # macOS ort packages no longer contain x64 binaries. In case we are running on x64 macOS, we need to download arm64 binaries.
        if plat == "osx":
            mach = "arm64"

        return destination_dir / "ort" / "runtimes" / (plat + "-" + mach) / "native"

    package_name = None
    if use_cuda:
        if is_linux():
            package_name = "Microsoft.ML.OnnxRuntime.Gpu.Linux"
        elif is_windows():
            if is_windows_arm():
                package_name = "Microsoft.ML.OnnxRuntime"
            else:
                package_name = "Microsoft.ML.OnnxRuntime.Gpu.Windows"
        else:
            raise NotImplementedError("ORT with CUDA is not supported on this platform")
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

    version = requests.get(version_fetch_url).json()["value"][0]["versions"][0]["normalizedVersion"]
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
            raise NotImplementedError(f"Unsupported machine architecture: {platform.machine()}")

        return destination_dir / "dml" / "bin" / (mach + "-win") / "DirectML.dll"

    dml_version = "1.15.2"
    dml_package_name = "Microsoft.AI.DirectML"
    dml_package_url = f"https://www.nuget.org/api/v2/package/{dml_package_name}/{dml_version}"
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
            raise NotImplementedError(f"Unsupported machine architecture: {platform.machine()}")

        return destination_dir / "d3d12" / "build" / "native" / "bin" / mach / "D3D12Core.dll"

    d3d12_version = "1.614.1"
    d3d12_package_name = "Microsoft.Direct3D.D3D12"
    d3d12_package_url = f"https://www.nuget.org/api/v2/package/{d3d12_package_name}/{d3d12_version}"
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


def download_dependencies(use_cuda: bool, use_dml: bool, destination_dir: PathLike):
    dependencies_dir = destination_dir / "dependencies"
    if not dependencies_dir.exists():
        dependencies_dir.mkdir(parents=True)

    ort_lib_dir = _download_ort(use_cuda, use_dml, dependencies_dir)
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


_ORT_NIGHTLY_FEED_URL = "https://pkgs.dev.azure.com/aiinfra/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_apis/packaging/feeds/7982ae20-ed19-4a35-a362-a96ac99897b7"
_CUDA_PLUGIN_EP_FEED_URL = "https://pkgs.dev.azure.com/aiinfra/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8"
_ORT_VERSION = "1.29.0"
_CUDA_PLUGIN_EP_VERSION = "0.1.0"

_ENGINE_BENCHMARK_PACKAGES = {
    "Microsoft.ML.OnnxRuntime": (
        f"{_ORT_NIGHTLY_FEED_URL}/nuget/packages/Microsoft.ML.OnnxRuntime"
        f"/versions/{_ORT_VERSION}/content?api-version=6.0-preview.1"
    ),
    "Microsoft.ML.OnnxRuntime.EP.Cuda12.linux-x64": (
        f"{_CUDA_PLUGIN_EP_FEED_URL}/nuget/packages/Microsoft.ML.OnnxRuntime.EP.Cuda12.linux-x64"
        f"/versions/{_CUDA_PLUGIN_EP_VERSION}/content?api-version=6.0-preview.1"
    ),
}


def _download_and_unpack_nupkg(package_name: str, package_url: str, destination_dir: Path) -> Path:
    unpacked_dir = destination_dir / package_name
    if unpacked_dir.exists():
        _log.info(f"Package {package_name} already downloaded")
        return unpacked_dir

    _log.info(f"Downloading {package_name} from {package_url}")
    response = requests.get(package_url)
    response.raise_for_status()
    package_path = destination_dir / f"{package_name}.zip"
    with open(package_path, "wb") as f:
        f.write(response.content)

    shutil.unpack_archive(package_path, unpacked_dir, format="zip")
    return unpacked_dir


def setup_engine_benchmark_dependencies(genai_lib_dir: PathLike, destination_dir: PathLike) -> Path:
    """
    Populate the engine_benchmark output directory with the shared libraries it loads at runtime:
    the ONNX Runtime linux-x64 libraries and CUDA execution provider plugin from the ORT-Nightly
    feed, plus the locally built onnxruntime-genai libraries.
    """
    genai_lib_dir = Path(genai_lib_dir)
    destination_dir = Path(destination_dir)
    # Downloaded packages are kept in a subdir; only the .so files are copied next to the executable.
    dependencies_dir = destination_dir / "dependencies"
    dependencies_dir.mkdir(parents=True, exist_ok=True)

    for package_name, package_url in _ENGINE_BENCHMARK_PACKAGES.items():
        package_dir = _download_and_unpack_nupkg(package_name, package_url, dependencies_dir)
        _log.info(f"Extracting {package_name} .so files to {destination_dir}")
        for lib in package_dir.rglob("linux-x64/native/*"):
            if lib.is_file():
                shutil.copy(lib, destination_dir)

    # ORT is loaded by soname, which the nuget package only ships as the unversioned file.
    unversioned_ort = destination_dir / "libonnxruntime.so"
    symlink_ort = destination_dir / "libonnxruntime.so.1"
    _log.info(f"Creating symlink {unversioned_ort.name} --> {symlink_ort.name}")
    symlink_ort.unlink(missing_ok=True)
    symlink_ort.symlink_to(unversioned_ort.name)

    _log.info(f"Copying local genai and genai-cuda builds to {destination_dir}")
    for genai_lib_name in ("libonnxruntime-genai.so", "libonnxruntime-genai-cuda.so"):
        genai_lib = genai_lib_dir / genai_lib_name
        if genai_lib.is_file():
            shutil.copy(genai_lib, destination_dir)
        else:
            _log.warning(f"{genai_lib} not found; skipping.")

    return destination_dir
