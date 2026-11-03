# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from elftools.elf.dynamic import DynamicSection
from elftools.elf.elffile import ELFFile
from packaging.utils import parse_wheel_filename

CORE_PATH = "onnxruntime_genai_core/bin/libonnxruntime-genai.so"
MACHINES = {"x64": "EM_X86_64", "arm64": "EM_AARCH64"}
RUNTIME_LIBRARIES = (
    "libonnxruntime",
    "libcuda",
    "libcublas",
    "libcudnn",
    "libcufft",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libcupti",
    "libnv",
)


def verify_elf_payload(wheel_path: Path, arch: str):
    with ZipFile(wheel_path) as wheel:
        if CORE_PATH not in wheel.namelist():
            raise ValueError("Core wheel is missing the GenAI library")
        for path in wheel.namelist():
            if ".so" not in Path(path).suffixes:
                continue
            if path != CORE_PATH and Path(path).name.startswith(RUNTIME_LIBRARIES):
                raise ValueError(f"Core wheel bundles a runtime-managed library: {path}")
            elf = ELFFile(io.BytesIO(wheel.read(path)))
            if elf.elfclass != 64 or not elf.little_endian or elf["e_machine"] != MACHINES[arch]:
                raise ValueError(f"Unexpected ELF architecture for {arch}: {path}")
            needed = [
                tag.needed
                for section in elf.iter_sections()
                if isinstance(section, DynamicSection)
                for tag in section.iter_tags("DT_NEEDED")
            ]
            unexpected = [name for name in needed if name.startswith(RUNTIME_LIBRARIES)]
            if unexpected:
                raise ValueError(f"Runtime-managed DT_NEEDED dependencies in {path}: {unexpected}")


def verify_native_loading(wheel_path: Path, ort_home: Path):
    ort_libraries = sorted((ort_home / "lib").glob("libonnxruntime.so*"))
    if not ort_libraries:
        raise FileNotFoundError(f"ORT runtime library missing from {ort_home}")
    environment = os.environ.copy()
    environment["ORT_LIB_PATH"] = str(ort_libraries[0].resolve())
    with tempfile.TemporaryDirectory(prefix="genai-core-install-") as directory:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                directory,
                str(wheel_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                (
                    "import ctypes, pathlib, sys; "
                    "sys.path.insert(0, sys.argv[1]); "
                    "import onnxruntime_genai_core; "
                    "core = pathlib.Path(onnxruntime_genai_core.__file__).parent / 'bin' / 'libonnxruntime-genai.so'; "
                    "library = ctypes.CDLL(str(core)); "
                    "library.OgaShutdown.argtypes = []; "
                    "library.OgaShutdown.restype = None; "
                    "library.OgaShutdown(); "
                    "print('Installed core wheel imports and loads with externally supplied ORT')"
                ),
                directory,
            ],
            env=environment,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Check and load a repaired native-only Linux core wheel.")
    parser.add_argument("--wheel_dir", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=MACHINES)
    parser.add_argument("--ort_home", required=True, type=Path)
    args = parser.parse_args()
    wheels = list(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one repaired core wheel; found {len(wheels)}")
    wheel = wheels[0]
    name, _, _, tags = parse_wheel_filename(wheel.name)
    expected = {"x64": "manylinux_2_28_x86_64", "arm64": "manylinux_2_34_aarch64"}[args.arch]
    if name != "onnxruntime-genai-core" or not any(tag.platform == expected for tag in tags):
        raise ValueError(f"Unexpected repaired wheel: {wheel.name}")
    verify_elf_payload(wheel, args.arch)
    verify_native_loading(wheel, args.ort_home)


if __name__ == "__main__":
    main()
