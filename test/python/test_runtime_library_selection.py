# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / "src/python/py/_dll_directory.py"
SPEC = importlib.util.spec_from_file_location("genai_dll_directory", SOURCE)
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)


class RuntimeLibrarySelectionTest(unittest.TestCase):
    def test_explicit_runtime_takes_precedence(self):
        for platform in ("linux", "darwin"):
            with (
                self.subTest(platform=platform),
                mock.patch.object(LOADER.sys, "platform", platform),
                mock.patch.dict(os.environ, {"ORT_LIB_PATH": "/fresh/libonnxruntime.so"}, clear=True),
                mock.patch.object(LOADER.ctypes, "CDLL") as load,
                mock.patch.object(LOADER.importlib.util, "find_spec") as find,
            ):
                LOADER.add_onnxruntime_dependency("onnxruntime-genai-cuda")
                load.assert_called_once_with("/fresh/libonnxruntime.so")
                find.assert_not_called()
                self.assertEqual(os.environ["ORT_LIB_PATH"], "/fresh/libonnxruntime.so")

    def test_invalid_explicit_runtime_does_not_fall_back(self):
        with (
            mock.patch.object(LOADER.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"ORT_LIB_PATH": "/missing/runtime.so"}, clear=True),
            mock.patch.object(LOADER.ctypes, "CDLL", side_effect=OSError("cannot load runtime")),
            mock.patch.object(LOADER.importlib.util, "find_spec") as find,
        ):
            with self.assertRaisesRegex(OSError, "cannot load runtime"):
                LOADER.add_onnxruntime_dependency("onnxruntime-genai-cuda")
            find.assert_not_called()

    def test_empty_explicit_runtime_is_rejected(self):
        with (
            mock.patch.object(LOADER.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"ORT_LIB_PATH": ""}, clear=True),
            mock.patch.object(LOADER.ctypes, "CDLL") as load,
        ):
            with self.assertRaisesRegex(ImportError, "must not be empty"):
                LOADER.add_onnxruntime_dependency("onnxruntime-genai-cuda")
            load.assert_not_called()

    def test_unset_override_preserves_installed_package_selection(self):
        with (
            mock.patch.object(LOADER.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                LOADER.importlib.util,
                "find_spec",
                return_value=mock.Mock(submodule_search_locations=["/installed/onnxruntime"]),
            ),
            mock.patch.object(LOADER.glob, "glob", return_value=["/installed/onnxruntime/capi/runtime.so"]),
            mock.patch.object(LOADER.ctypes, "CDLL") as load,
        ):
            LOADER.add_onnxruntime_dependency("onnxruntime-genai-cuda")
            load.assert_called_once_with("/installed/onnxruntime/capi/runtime.so")
            self.assertEqual(os.environ["ORT_LIB_PATH"], "/installed/onnxruntime/capi/runtime.so")


if __name__ == "__main__":
    unittest.main()
