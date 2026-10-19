#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""CPU-only package assembly regressions; no checkpoint, CUDA or research dependencies."""

import contextlib
import errno
import importlib.util
import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper


def load_builder(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bp = load_builder("build_package")
v12 = load_builder("build_package_v12")
v12b = load_builder("build_package_v12b")


class PackageAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, name):
        path = self.root / name
        path.mkdir()
        (path / "model.onnx").write_bytes(b"graph")
        (path / "weights.bin").write_bytes(b"weights")
        return path

    def external_model(self, directory, name="weight"):
        tensor = helper.make_tensor(name, TensorProto.FLOAT, [2], [1.0, 2.0])
        tensor.ClearField("float_data")
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in {"location": "weights.bin", "offset": "0", "length": "8"}.items():
            item = tensor.external_data.add()
            item.key, item.value = key, value
        model = helper.make_model(helper.make_graph([], "weights", [], [], [tensor]))
        onnx.save(model, directory / "model.onnx")

    def test_rejects_existing_bundle_without_modifying_it(self):
        source = self.bundle("source")
        target = self.bundle("target")
        (target / "model.onnx").write_bytes(b"old graph")
        for copy in (False, True):
            with self.subTest(copy=copy), self.assertRaisesRegex(SystemExit, "already exists"):
                bp.link_tree(source, target, copy)
            self.assertEqual((target / "model.onnx").read_bytes(), b"old graph")
            self.assertEqual((target / "weights.bin").read_bytes(), b"weights")

    def test_preflight_checks_all_destinations_before_copying(self):
        source = self.bundle("source")
        output = self.root / "out"
        (output / "mtp_head").mkdir(parents=True)
        for builder, flag in ((bp, "--target-bundle"), (v12, "--fold-bundle")):
            with self.subTest(builder=builder.__name__):
                collision = output / ("mtp_head" if builder is bp else "head_r2")
                collision.mkdir(exist_ok=True)
                head_flag = "--head-bundle" if builder is bp else "--head-r2-bundle"
                argv = ["builder", "--out", str(output), flag, str(source), head_flag, str(source)]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "already exists"):
                    builder.main()
                self.assertFalse((output / "target").exists())
                self.assertFalse((output / "libs").exists())
                self.assertFalse((output / "genai_config.json").exists())

    def test_preserves_file_and_directory_symlinks(self):
        source = self.bundle("source")
        external = self.root / "external"
        external.mkdir()
        (external / "weight.bin").write_bytes(b"external")
        (source / "external_weights").symlink_to("../external", target_is_directory=True)
        (source / "linked.bin").symlink_to("external_weights/weight.bin")
        (source / "dangling").symlink_to("missing", target_is_directory=True)
        for copy in (False, True):
            target = self.root / f"linked-{copy}"
            bp.link_tree(source, target, copy)
            for name in ("external_weights", "linked.bin", "dangling"):
                self.assertTrue((target / name).is_symlink())
                self.assertEqual(os.readlink(target / name), os.readlink(source / name))
            self.assertEqual((target / "external_weights/weight.bin").read_bytes(), b"external")
            self.assertEqual((target / "linked.bin").read_bytes(), b"external")

    def test_rejects_existing_dangling_bundle_symlink(self):
        source = self.bundle("source")
        target = self.root / "target"
        target.symlink_to("missing", target_is_directory=True)
        with self.assertRaisesRegex(SystemExit, "already exists"):
            bp.link_tree(source, target, False)
        self.assertEqual(os.readlink(target), "missing")

    def test_rejects_nested_destinations_before_any_writes(self):
        source = self.bundle("source")
        alias = self.root / "alias"
        alias.symlink_to(source, target_is_directory=True)
        for base in (source, alias):
            for copy in (False, True):
                destination = base / "out/target"
                with self.subTest(base=base, copy=copy), self.assertRaisesRegex(SystemExit, "inside a source"):
                    bp.link_tree(source, destination, copy)
                self.assertFalse((source / "out").exists())
        other = self.bundle("other")
        with self.assertRaisesRegex(SystemExit, "inside a source"):
            bp.validate_bundle_destinations([(other, source / "out/target"), (source, self.root / "head")])
        self.assertFalse((source / "out").exists())
        for builder, flag in ((bp, "--target-bundle"), (v12, "--fold-bundle")):
            head_flag = "--head-bundle" if builder is bp else "--head-r2-bundle"
            argv = ["builder", "--out", str(source / "out"), flag, str(source), head_flag, str(other)]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "inside a source"):
                builder.main()
            self.assertFalse((source / "out").exists())

    def test_bf16_constants_support_typed_inline_and_external_without_length(self):
        values = [1.0, -2.0, 0.5, 0.0]
        expected = struct.pack("<4H", 0x3F80, 0xC000, 0x3F00, 0)
        tensor = helper.make_tensor("constant", TensorProto.BFLOAT16, [2, 2], values)
        self.assertTrue(tensor.int32_data)
        self.assertEqual(v12._tensor_bytes(tensor, self.root), expected)
        tensor.ClearField("int32_data")
        tensor.raw_data = expected
        self.assertEqual(v12._tensor_bytes(tensor, self.root), expected)
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in {"location": "constant.bin", "offset": "3"}.items():
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
        (self.root / "constant.bin").write_bytes(b"pad" + expected)
        self.assertEqual(v12._tensor_bytes(tensor, self.root), expected)
        entry = tensor.external_data.add()
        entry.key, entry.value = "length", str(len(expected))
        (self.root / "constant.bin").write_bytes(b"pad" + expected + b"suffix")
        self.assertEqual(v12._tensor_bytes(tensor, self.root), expected)

    def test_qsa_constants_extract_typed_inline_bf16_weights_and_rotary_tables(self):
        paths = []
        for label, prefix in (("target", ""), ("head", "head.")):
            constants = [
                helper.make_tensor(prefix + "layers.0.norm", TensorProto.BFLOAT16, [2], [1.0, -2.0]),
                helper.make_tensor("cos", TensorProto.BFLOAT16, [33028, 1], [1.0] * 33028),
                helper.make_tensor("sin", TensorProto.BFLOAT16, [33028, 1], [0.5] * 33028),
            ]
            node = helper.make_node(
                "PackedQSAIndexer", ["x", "y", "z", *(t.name for t in constants)], ["out"], epsilon=1e-5
            )
            model = helper.make_model(helper.make_graph([node], label, [], [], constants))
            path = self.root / f"{label}.onnx"
            onnx.save(model, path)
            paths.append(path)
        summary = v12.write_qsa_constants(self.root, *paths)
        self.assertEqual(summary["width"], 1)
        self.assertTrue(summary["head_tables_equal_target"])
        blob = (self.root / "qsa_consts.bin").read_bytes()
        expected = {
            "knorm.0": struct.pack("<2H", 0x3F80, 0xC000),
            "cos": struct.pack("<H", 0x3F80) * 33028,
            "sin": struct.pack("<H", 0x3F00) * 33028,
        }
        entries = (self.root / "qsa_consts.txt").read_text().splitlines()[4:]
        self.assertEqual(len(entries), 6)
        for line in entries:
            _, _, name, offset, length = line.split()
            self.assertEqual(blob[int(offset) : int(offset) + int(length)], expected[name])

    def test_shared_pool_survives_copy_and_cross_device_fallback(self):
        first = self.bundle("first")
        second = self.bundle("second")
        (first / "weights.bin").write_bytes(b"12345678")
        (second / "weights.bin").unlink()
        os.link(first / "weights.bin", second / "weights.bin")
        self.external_model(first, "target_weight")
        self.external_model(second, "head_weight")
        for mode in ("link", "copy", "fallback"):
            with self.subTest(mode=mode):
                output = self.root / mode
                output.mkdir()
                with contextlib.ExitStack() as stack:
                    if mode == "fallback":
                        stack.enter_context(patch.object(os, "link", side_effect=OSError(errno.EXDEV, "cross-device")))
                    bp.link_tree(first, output / "target", mode == "copy")
                    bp.link_tree(second, output / "head", mode == "copy")
                summary = v12.write_pool(
                    output,
                    {"target": output / "target/model.onnx", "head_r2": output / "head/model.onnx"},
                    {"target": first / "model.onnx", "head_r2": second / "model.onnx"},
                )
                self.assertEqual(summary["initializers"], 1)
                self.assertEqual(summary["sessions"], {"target": 1, "head_r2": 1})
                manifest = (output / "weights_pool.txt").read_text()
                self.assertIn("session target target_weight 0", manifest)
                self.assertIn("session head_r2 head_weight 0", manifest)

    def test_pool_does_not_merge_unrelated_same_named_weights(self):
        first = self.bundle("first")
        second = self.bundle("second")
        for path in (first, second):
            (path / "weights.bin").write_bytes(b"12345678")
            self.external_model(path)
        output = self.root / "out"
        output.mkdir()
        for label, source in (("target", first), ("head", second)):
            bp.link_tree(source, output / label, True)
        summary = v12.write_pool(
            output,
            {"target": output / "target/model.onnx", "head": output / "head/model.onnx"},
            {"target": first / "model.onnx", "head": second / "model.onnx"},
        )
        self.assertEqual(summary["initializers"], 0)

    def test_empty_pool_round_trip_through_optimizer(self):
        optimizer = load_builder("optimize_ported_package")
        output = self.root / "empty_pool"
        output.mkdir()
        models = {}
        for label in ("target", "dense", "head_r2", "head_kv"):
            directory = output / label
            directory.mkdir()
            path = directory / "model.onnx"
            onnx.save(helper.make_model(helper.make_graph([], label, [], [])), path)
            models[label] = path
        self.assertEqual(v12.write_pool(output, models)["initializers"], 0)
        self.assertEqual((output / "weights_pool.txt").read_text(), "qwen38_weights_pool 1\n\n")
        self.assertEqual(optimizer.refresh_weight_pool(output, output), {"initializers": 0, "gib": 0})
        self.assertEqual((output / "weights_pool.txt").read_text(), "qwen38_weights_pool 1\n")

    def test_constants_use_packaged_target(self):
        source = self.bundle("source")
        tokenizers = self.root / "tokenizers"
        tokenizers.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            (tokenizers / name).write_text("{}")
        library = self.root / "lib.so"
        library.write_bytes(b"library")
        output = self.root / "out"
        argv = [
            "builder",
            "--out",
            str(output),
            "--skip-ple",
            "--no-pool",
            "--copy",
            "--plane-lib",
            str(library),
            "--tokenizer-from",
            str(tokenizers),
            "--checkpoint",
            str(tokenizers),
        ]
        for flag in ("fold-bundle", "head-r2-bundle", "dense-bundle", "head-kv-bundle"):
            argv.extend([f"--{flag}", str(source)])
        for name in v12.LIBS:
            argv.extend([f"--lib-{name.replace('_', '-')}", str(library)])
        with (
            patch.object(sys, "argv", argv),
            patch.object(v12, "write_qsa_constants", return_value={}) as constants,
            patch.object(v12, "DECODE_REF_MODEL", library, create=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            v12.main()
        constants.assert_called_once_with(output, output / "target/model.onnx", output / "head_r2/model.onnx")

    def test_v12b_checks_provider_before_building(self):
        with (
            patch.object(
                sys, "argv", ["builder", "--out", str(self.root / "out"), "--provider", str(self.root / "missing.so")]
            ),
            patch.object(v12b.v12, "main") as build,
            self.assertRaisesRegex(SystemExit, "missing provider"),
        ):
            v12b.main()
        build.assert_not_called()

    def test_v12b_help_needs_no_research_files(self):
        with (
            patch.object(sys, "argv", ["builder", "--help"]),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            v12b.main()
        self.assertEqual(error.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
