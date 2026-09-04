# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from os import PathLike, listdir
from os.path import isfile
from pathlib import Path
from xml.etree import ElementTree

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


def _restore_dml_dependencies(
    destination_dir: PathLike,
    nuget_config_file: PathLike | None = None,
    nuget_package_source: str | None = None,
):
    mach = None
    if platform.machine().lower() == "x86_64" or platform.machine().lower() == "amd64":
        mach = "x64"
    elif platform.machine().lower() == "aarch64" or platform.machine().lower() == "arm64":
        mach = "arm64"
    else:
        raise NotImplementedError(f"Unsupported machine architecture: {platform.machine()}")

    destination_dir = Path(destination_dir)
    packages = (
        ("Microsoft.AI.DirectML", "1.15.2"),
        ("Microsoft.Direct3D.D3D12", "1.614.1"),
    )
    dml_root = destination_dir / f"{packages[0][0]}.{packages[0][1]}"
    d3d12_root = destination_dir / f"{packages[1][0]}.{packages[1][1]}"
    dml_lib_path = dml_root / "bin" / f"{mach}-win" / "DirectML.dll"
    d3d12_lib_path = d3d12_root / "build" / "native" / "bin" / mach / "D3D12Core.dll"

    if dml_lib_path.exists() and d3d12_lib_path.exists():
        _log.info("DirectML dependencies already restored")
        return dml_lib_path, d3d12_lib_path

    nuget_path = shutil.which("nuget") or shutil.which("nuget.exe")
    if not nuget_path:
        raise RuntimeError("nuget or nuget.exe must be available on PATH to restore DirectML dependencies")

    packages_config = destination_dir / "dml-packages.config"
    packages_element = ElementTree.Element("packages")
    for package_name, package_version in packages:
        ElementTree.SubElement(
            packages_element,
            "package",
            id=package_name,
            version=package_version,
            targetFramework="native",
        )
    ElementTree.ElementTree(packages_element).write(packages_config, encoding="utf-8", xml_declaration=True)

    command = [
        nuget_path,
        "restore",
        str(packages_config),
        "-PackagesDirectory",
        str(destination_dir),
        "-NonInteractive",
    ]
    if nuget_package_source:
        command.extend(["-Source", nuget_package_source])
    if nuget_config_file:
        command.extend(["-ConfigFile", str(nuget_config_file)])

    _log.info("Restoring DirectML dependencies with NuGet")
    subprocess.run(command, check=True)

    if not dml_lib_path.exists():
        raise RuntimeError(f"NuGet restore did not produce {dml_lib_path}")
    if not d3d12_lib_path.exists():
        raise RuntimeError(f"NuGet restore did not produce {d3d12_lib_path}")

    return dml_lib_path, d3d12_lib_path


def download_dependencies(
    use_cuda: bool,
    use_dml: bool,
    destination_dir: PathLike,
    nuget_config_file: PathLike | None = None,
    nuget_package_source: str | None = None,
):
    dependencies_dir = destination_dir / "dependencies"
    if not dependencies_dir.exists():
        dependencies_dir.mkdir(parents=True)

    ort_lib_dir = _download_ort(use_cuda, use_dml, dependencies_dir)
    libs = listdir(ort_lib_dir)
    for file_name in libs:
        if isfile(Path(ort_lib_dir) / file_name):
            shutil.copy(Path(ort_lib_dir) / file_name, destination_dir)

    if use_dml:
        dml_lib_path, d3d12_lib_path = _restore_dml_dependencies(
            dependencies_dir,
            nuget_config_file,
            nuget_package_source,
        )
        shutil.copy(dml_lib_path, destination_dir)
        shutil.copy(d3d12_lib_path, destination_dir)

    return dependencies_dir


def copy_dependencies(lib_dir: PathLike, destination_dir: PathLike):
    libs = listdir(lib_dir)
    for file_name in libs:
        shutil.copy(Path(lib_dir) / file_name, destination_dir)


# ADO Feed: aiinfra / PublicPackages / ORT-Nightly
_ORT_FEED_URL = "https://pkgs.dev.azure.com/aiinfra/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_apis/packaging/feeds/7982ae20-ed19-4a35-a362-a96ac99897b7"
_ORT_VERSION = "1.29.0"

# ADO Feed: aiinfra / PublicPackages / onnxruntime-cuda-12
_CUDA_PLUGIN_EP_FEED_URL = "https://pkgs.dev.azure.com/aiinfra/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8"
_CUDA_PLUGIN_EP_VERSION = "0.1.0"

_ENGINE_BENCHMARK_PACKAGES = {
    "Microsoft.ML.OnnxRuntime": (
        f"{_ORT_FEED_URL}/nuget/packages/Microsoft.ML.OnnxRuntime"
        f"/versions/{_ORT_VERSION}/content?api-version=6.0-preview.1"
    ),
    "Microsoft.ML.OnnxRuntime.EP.Cuda12.linux-x64": (
        f"{_CUDA_PLUGIN_EP_FEED_URL}/nuget/packages/Microsoft.ML.OnnxRuntime.EP.Cuda12.linux-x64"
        f"/versions/{_CUDA_PLUGIN_EP_VERSION}/content?api-version=6.0-preview.1"
    ),
}


def _download_and_unpack_nupkg(package_name: str, version: str, package_url: str, destination_dir: Path) -> Path:
    unpacked_dir = destination_dir / package_name
    if unpacked_dir.exists():
        _log.info(f"Package {package_name} already downloaded")
        return unpacked_dir

    _log.info(f"Downloading {package_name} {version}...")
    with requests.get(package_url, stream=True, timeout=60) as response:
        response.raise_for_status()  # raises a 4xx or 5xx (client/server error) if encountered
        package_path = destination_dir / f"{package_name}.zip"
        with open(package_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    shutil.unpack_archive(package_path, unpacked_dir, format="zip")
    return unpacked_dir


def setup_engine_benchmark_dependencies(genai_lib_dir: PathLike, destination_dir: PathLike) -> Path:
    """
    Populate the engine_benchmark output directory with the shared libraries it loads at runtime:
    the **linux-x64** ONNX Runtime and CUDA execution provider plugin libraries from the ORT-Nightly
    feed, plus the locally built onnxruntime-genai libraries.
    """
    genai_lib_dir = Path(genai_lib_dir)
    destination_dir = Path(destination_dir)
    # Downloaded packages are kept in a subdir; only the .so files are copied next to the executable.
    dependencies_dir = destination_dir / "dependencies"
    dependencies_dir.mkdir(parents=True, exist_ok=True)

    for package_name, package_url in _ENGINE_BENCHMARK_PACKAGES.items():
        version = _ORT_VERSION if package_name == "Microsoft.ML.OnnxRuntime" else _CUDA_PLUGIN_EP_VERSION
        package_dir = _download_and_unpack_nupkg(package_name, version, package_url, dependencies_dir)
        _log.info(f"Extracting {package_name} .so files to {destination_dir}")
        for lib in package_dir.rglob("linux-x64/native/*"):
            if lib.is_file():
                shutil.copy(lib, destination_dir)

    # ORT is loaded by soname, which the nuget package only ships as the unversioned file.
    unversioned_ort = destination_dir / "libonnxruntime.so"
    symlink_ort = destination_dir / "libonnxruntime.so.1"
    _log.info(f"Creating symlink {symlink_ort.name} -> {unversioned_ort.name}")
    symlink_ort.unlink(missing_ok=True)
    symlink_ort.symlink_to(unversioned_ort.name)

    _log.info(f"Copying local genai and genai-cuda builds to {destination_dir}")
    patchelf = shutil.which("patchelf")  # patchelf is a utility to modify the dynamic linker and RPATH of ELF executables
    if patchelf is None:
        raise RuntimeError("patchelf is required to stage benchmark dependencies")

    for genai_lib_name in ("libonnxruntime-genai.so", "libonnxruntime-genai-cuda.so"):
        genai_lib = genai_lib_dir / genai_lib_name
        if not genai_lib.is_file():
            raise RuntimeError(f"Required GenAI library not found: {genai_lib}")

        staged_lib = shutil.copy(genai_lib, destination_dir)
        # The build bakes the configure-time ORT path into RPATH, which beats LD_LIBRARY_PATH and
        # would load that ORT instead of the pinned one staged here.
        subprocess.run([patchelf, "--set-rpath", "$ORIGIN", staged_lib], check=True)
        rpath = subprocess.check_output([patchelf, "--print-rpath", staged_lib], text=True).strip()
        if rpath != "$ORIGIN":
            raise RuntimeError(f"Staged library integrity check failed: {staged_lib} has RPATH '{rpath}'")

    required_libraries = (
        "libonnxruntime.so",
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime-genai.so",
        "libonnxruntime-genai-cuda.so",
    )
    missing_libraries = [name for name in required_libraries if not (destination_dir / name).is_file()]
    if missing_libraries:
        raise RuntimeError(f"Missing staged benchmark libraries: {', '.join(missing_libraries)}")
    if not symlink_ort.is_symlink() or symlink_ort.readlink() != Path(unversioned_ort.name):
        raise RuntimeError("Staged ORT soname link integrity check failed")

    # Read back by the benchmark so results record which packages they ran against.
    versions_path = destination_dir / "versions.json"
    versions_path.write_text(
        json.dumps({"ort_version": _ORT_VERSION, "cuda_plugin_ep_version": _CUDA_PLUGIN_EP_VERSION}, indent=2)
    )

    return destination_dir
