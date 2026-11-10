# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[4] / "tools/python"), *sys.path]):
    dependencies = importlib.import_module("util.dependency_resolver")
    build_spec = importlib.util.spec_from_file_location("genai_build", Path(__file__).resolve().parents[4] / "build.py")
    build_driver = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(build_driver)


class MacDependenciesTests(unittest.TestCase):
    def test_build_stages_dependencies_next_to_executable_for_each_generator(self):
        for generator, config in (
            ("Xcode", "Release"),
            ("Xcode", "Debug"),
            ("Ninja", "Release"),
            ("Unix Makefiles", "Release"),
        ):
            with self.subTest(generator=generator, config=config), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                build_dir = root / "build"
                genai_dir = build_dir / config if generator == "Xcode" else build_dir
                executable_dir = build_dir / "benchmark" / "engine"
                if generator == "Xcode":
                    executable_dir /= config
                genai_dir.mkdir(parents=True)
                executable_dir.mkdir(parents=True)
                (root / "sdk/lib").mkdir(parents=True)
                (root / "sdk/lib/libonnxruntime.dylib").write_bytes(b"local ORT")
                (genai_dir / "libonnxruntime-genai.dylib").write_bytes(b"local GenAI")
                args = argparse.Namespace(
                    cmake_path=Path("cmake"),
                    build_dir=build_dir,
                    config=config,
                    cmake_generator=generator,
                    parallel=False,
                    skip_wheel=True,
                    ort_home=root / "sdk",
                    build_csharp=False,
                    build_engine_benchmark=True,
                )
                library = Mock()
                library.OrtGetApiBase.return_value.contents.GetVersionString.return_value = b"1.32.0"
                with (
                    patch.object(build_driver.util, "is_windows", return_value=False),
                    patch.object(build_driver.util, "is_linux", return_value=False),
                    patch.object(build_driver.util, "is_mac", return_value=True),
                    patch.object(build_driver.util, "run") as run,
                    patch.object(dependencies.platform, "system", return_value="Darwin"),
                    patch("ctypes.CDLL", return_value=library),
                ):
                    build_driver.build(args, {})
                run.assert_called_once_with(["cmake", "--build", str(build_dir), "--config", config], env={})
                self.assertEqual((executable_dir / "libonnxruntime.dylib").read_bytes(), b"local ORT")
                self.assertEqual((executable_dir / "libonnxruntime-genai.dylib").read_bytes(), b"local GenAI")
                self.assertEqual(json.loads((executable_dir / "versions.json").read_text())["ort_version"], "1.32.0")
                if generator == "Xcode":
                    self.assertFalse((executable_dir.parent / "versions.json").exists())

    def test_macos_requires_explicit_matching_sdk(self):
        with (
            patch.object(dependencies.platform, "system", return_value="Darwin"),
            self.assertRaisesRegex(ValueError, "--ort_home"),
        ):
            dependencies.setup_engine_benchmark_dependencies("build", "out")

    def test_macos_reports_missing_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(dependencies.platform, "system", return_value="Darwin"),
                self.assertRaisesRegex(FileNotFoundError, "libonnxruntime.dylib"),
            ):
                dependencies.setup_engine_benchmark_dependencies(root, root / "out", ort_home=root)

    def test_macos_stages_local_runtime_and_version_without_cuda_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sdk/lib").mkdir(parents=True)
            (root / "genai").mkdir()
            (root / "sdk/lib/libonnxruntime.dylib").write_bytes(b"local ORT")
            (root / "genai/libonnxruntime-genai.dylib").write_bytes(b"local GenAI")
            library = Mock()
            library.OrtGetApiBase.return_value.contents.GetVersionString.return_value = b"1.32.0"
            with (
                patch.object(dependencies.platform, "system", return_value="Darwin"),
                patch("ctypes.CDLL", return_value=library),
                patch.object(dependencies, "_download_and_unpack_nupkg") as download,
            ):
                output = dependencies.setup_engine_benchmark_dependencies(
                    root / "genai", root / "out", ort_home=root / "sdk"
                )
            download.assert_not_called()
            self.assertEqual((output / "libonnxruntime.dylib").read_bytes(), b"local ORT")
            self.assertEqual((output / "libonnxruntime-genai.dylib").read_bytes(), b"local GenAI")
            self.assertEqual(json.loads((output / "versions.json").read_text())["ort_version"], "1.32.0")


if __name__ == "__main__":
    unittest.main()
