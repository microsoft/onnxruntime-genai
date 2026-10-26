# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[4] / "tools/python"), *sys.path]):
    dependencies = importlib.import_module("util.dependency_resolver")


class MacDependenciesTests(unittest.TestCase):
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
