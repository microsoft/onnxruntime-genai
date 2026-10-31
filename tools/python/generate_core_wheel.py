# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from packaging.utils import parse_wheel_filename
from packaging.version import Version
from wheel.wheelfile import WheelFile

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "onnxruntime_genai_core"
PLATFORMS = {
    "win-x64": ("win_amd64", "onnxruntime-genai.dll"),
    "win-arm64": ("win_arm64", "onnxruntime-genai.dll"),
    "osx-arm64": ("macosx_12_0_arm64", "libonnxruntime-genai.dylib"),
    "linux-x64": ("linux_x86_64", "libonnxruntime-genai.so"),
    "linux-arm64": ("linux_aarch64", "libonnxruntime-genai.so"),
}
MANYLINUX_PLATFORMS = {
    "linux-x64": "manylinux_2_28_x86_64",
    "linux-arm64": "manylinux_2_34_aarch64",
}


def generate_wheel(core_dir: Path, rid: str, package_version: str, output_dir: Path) -> Path:
    platform_tag, library_name = PLATFORMS[rid]
    library = core_dir / "lib" / library_name
    if not library.is_file():
        raise FileNotFoundError(f"Missing GenAI core library for {rid}: {library}")

    if rid != "osx-arm64":
        config = core_dir / "lib" / "cmake" / "onnxruntime-genai" / "onnxruntime-genaiConfig.cmake"
        cuda_library = library.with_name(library_name.replace("genai", "genai-cuda"))
        if not re.search(r"set\(\s*onnxruntime-genai_USE_CUDA\s+1\s*\)", config.read_text(encoding="utf-8")):
            raise ValueError(f"A CUDA-built GenAI core is required for {rid}")
        if not cuda_library.is_file():
            raise FileNotFoundError(f"Missing CUDA add-on in the source core package for {rid}: {cuda_library}")

    version = str(Version(package_version))
    wheel_tag = f"py3-none-{platform_tag}"
    dist_info = f"{PACKAGE_NAME}-{version}.dist-info"
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = output_dir / f"{PACKAGE_NAME}-{version}-{wheel_tag}.whl"
    with WheelFile(str(wheel_path), "w") as wheel:
        wheel.writestr(f"{PACKAGE_NAME}/__init__.py", "")
        # Preserve the signed core bytes; the add-on and ORT are supplied at runtime.
        wheel.write(str(library), f"{PACKAGE_NAME}/bin/{library_name}")
        wheel.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            "Name: onnxruntime-genai-core\n"
            f"Version: {version}\n"
            "Summary: ONNX Runtime GenAI native core for dynamic execution providers\n"
            "License: MIT\n",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\nGenerator: onnxruntime-genai\nRoot-Is-Purelib: false\nTag: {wheel_tag}\n",
        )
        for name in ("LICENSE", "ThirdPartyNotices.txt"):
            wheel.write(str(REPO_ROOT / name), f"{dist_info}/{name}")
    return wheel_path


def repair_manylinux_wheel(wheel_path: Path, rid: str, output_dir: Path) -> Path:
    platform_tag = MANYLINUX_PLATFORMS[rid]
    name, version, _, _ = parse_wheel_filename(wheel_path.name)
    with tempfile.TemporaryDirectory(prefix="genai-core-repaired-") as directory:
        repaired_dir = Path(directory)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "auditwheel",
                "repair",
                "--plat",
                platform_tag,
                # These libraries are supplied by the application, not bundled in the core wheel.
                "--exclude",
                "libonnxruntime.so*",
                "--exclude",
                "libonnxruntime_providers_*.so*",
                "--exclude",
                "libonnxruntime-genai-cuda.so",
                "--wheel-dir",
                str(repaired_dir),
                str(wheel_path),
            ],
            check=True,
        )
        wheels = list(repaired_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one repaired core wheel for {rid}; found {len(wheels)}")
        repaired = wheels[0]
        repaired_name, repaired_version, _, tags = parse_wheel_filename(repaired.name)
        if (
            (repaired_name, repaired_version) != (name, version)
            or not any(tag.platform == platform_tag for tag in tags)
            or any(
                tag.interpreter != "py3" or tag.abi != "none" or not tag.platform.startswith("manylinux")
                for tag in tags
            )
        ):
            raise RuntimeError(f"Unexpected repaired core wheel tags or version: {repaired.name}")
        with ZipFile(repaired) as wheel:
            if f"{PACKAGE_NAME}/bin/libonnxruntime-genai.so" not in wheel.namelist():
                raise RuntimeError("Repaired core wheel is missing the GenAI library")
            for path in wheel.namelist():
                filename = Path(path).name
                if filename.startswith(("libonnxruntime-genai-cuda", "libonnxruntime_providers_")) or re.match(
                    r"libonnxruntime(?:-[0-9a-f]+)?\.so", filename
                ):
                    raise RuntimeError(f"Repaired core wheel contains a runtime-managed library: {filename}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = output_dir / repaired.name
        shutil.copy2(repaired, result)
        return result


def main():
    parser = argparse.ArgumentParser(description="Package a prebuilt GenAI core without its CUDA add-on or ORT.")
    parser.add_argument("--core_dir", required=True, type=Path, help="Installed core CMake package directory.")
    parser.add_argument("--rid", required=True, choices=PLATFORMS)
    parser.add_argument("--package_version", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.rid in MANYLINUX_PLATFORMS:
        with tempfile.TemporaryDirectory(prefix="genai-core-unrepaired-") as directory:
            wheel = generate_wheel(args.core_dir, args.rid, args.package_version, Path(directory))
            print(repair_manylinux_wheel(wheel, args.rid, args.output_dir))
    else:
        print(generate_wheel(args.core_dir, args.rid, args.package_version, args.output_dir))


if __name__ == "__main__":
    main()
