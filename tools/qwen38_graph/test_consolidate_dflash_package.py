# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from consolidate_dflash_package import DataWriter, consolidate, deduplicate_initializers, rewrite_pool


def external_tensor(name, location, offset, length):
    tensor = TensorProto(name=name, data_type=TensorProto.UINT8, dims=[length],
                         data_location=TensorProto.EXTERNAL)
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        item = tensor.external_data.add()
        item.key, item.value = key, value
    return tensor


def package(path):
    path.mkdir()
    (path / "weights.data").write_bytes(b"unused" + b"ABCD" + b"tail")
    (path / "ple.data").write_bytes(b"header" + b"01234567" + b"unused")
    target = helper.make_model(helper.make_graph([], "target", [], [], [
        external_tensor("weight", "weights.data", 6, 4)]))
    onnx.save_model(target, path / "target.onnx")
    draft = helper.make_model(helper.make_graph([], "draft", [], [], [
        external_tensor("draft", "dflash2.onnx.data", 0, 3)]))
    onnx.save_model(draft, path / "dflash2.onnx")
    (path / "dflash2.onnx.data").write_bytes(b"xyz")
    (path / "plane.so").write_bytes(b"plane")
    (path / "weights_pool.txt").write_text(
        "qwen38_weights_pool 1\npool 0 2 6 4 1 4 weights.data\nsession target weight 0\n")
    (path / "ple_table.txt").write_text("qwen38_ple_table 1\nhead_dim 2\nshard ple.data 6 4\n")
    config = {"model": {"decoder": {"filename": "target.onnx"}, "dflash2": {"filename": "dflash2.onnx"},
                        "qwen38_graph": {"plane_library": "plane.so", "ple_table": "ple_table.txt",
                                         "fold": {"enabled": True, "head_r2": {"filename": "unused.onnx"}},
                                         "mtp": {"enabled": True, "prefill_head": False},
                                         "prefill": {"enabled": True, "unified": True,
                                                     "weights_pool": "weights_pool.txt",
                                                     "constants": "unused.bin"}}}}
    (path / "genai_config.json").write_text(json.dumps(config))


class ConsolidationTests(unittest.TestCase):
    def test_shared_ranges_and_ple_are_packed_and_draft_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            package(source)
            original = (source / "genai_config.json").read_bytes()
            report = consolidate(source, output)
            target = onnx.load(output / "model.onnx", load_external_data=False)
            metadata = {item.key: item.value for item in target.graph.initializer[0].external_data}
            self.assertEqual(metadata["location"], "model.onnx.data")
            self.assertEqual(onnx.load(output / "model.onnx").graph.initializer[0].raw_data, b"ABCD")
            pool = (output / "weights_pool.txt").read_text().splitlines()[1].split()
            self.assertEqual(pool[3], metadata["offset"])
            shard = (output / "ple_table.txt").read_text().splitlines()[-1].split()
            with (output / shard[1]).open("rb") as stream:
                stream.seek(int(shard[2]))
                self.assertEqual(stream.read(8), b"01234567")
            self.assertEqual(len(report["verified_ranges"]), 2)
            self.assertEqual((output / "dflash2.onnx").read_bytes(), (source / "dflash2.onnx").read_bytes())
            self.assertEqual((output / "dflash2.onnx.data").read_bytes(), b"xyz")
            config = json.loads((output / "genai_config.json").read_text())["model"]
            self.assertEqual(config["decoder"]["filename"], "model.onnx")
            self.assertNotIn("head_r2", config["qwen38_graph"]["fold"])
            self.assertNotIn("constants", config["qwen38_graph"]["prefill"])
            self.assertNotIn("plane_library", config["qwen38_graph"])
            self.assertFalse((output / "plane.so").exists())
            self.assertEqual((source / "genai_config.json").read_bytes(), original)

    def test_integrated_package_does_not_require_a_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            package(source)
            config = json.loads((source / "genai_config.json").read_text())
            del config["model"]["qwen38_graph"]["plane_library"]
            (source / "genai_config.json").write_text(json.dumps(config))
            (source / "plane.so").unlink()
            consolidate(source, output)
            self.assertFalse(list(output.rglob("*.so")))

    def test_existing_destination_and_unsupported_execution_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            package(source)
            with self.assertRaises(FileExistsError):
                consolidate(source, source)
            config = json.loads((source / "genai_config.json").read_text())
            config["model"]["qwen38_graph"]["prefill"]["unified"] = False
            (source / "genai_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unified prefill"):
                consolidate(source, root / "output")
            self.assertFalse((root / "output").exists())

    def test_ranges_paths_and_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "weights").write_bytes(b"ABCD")
            (root / "outside").write_bytes(b"outside")
            writer = DataWriter(source, root / "packed")
            try:
                for offset, length in ((-1, 1), (0, 0), (0, 5)):
                    with self.assertRaisesRegex(ValueError, "Invalid external"):
                        writer.copy_range("weights", offset, length)
                with self.assertRaisesRegex(ValueError, "escapes package"):
                    writer.copy_range("../outside", 0, 1)
                writer.copy_range("weights", 0, 4)
                writer.stream.flush()
                with writer.destination.open("r+b") as stream:
                    stream.write(b"X")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    writer.verify()
            finally:
                writer.close()

    def test_pool_cannot_reintroduce_inactive_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package(root / "source")
            path = root / "source/weights_pool.txt"
            path.write_text(path.read_text().replace("session target", "session head_r2"))
            writer = DataWriter(root / "source", root / "packed")
            try:
                with self.assertRaisesRegex(ValueError, "Only target"):
                    rewrite_pool(path, writer)
            finally:
                writer.close()

    def test_nested_graph_weights_are_relocated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            package(source)
            target = onnx.load(source / "target.onnx", load_external_data=False)
            branch = helper.make_graph([], "branch", [], [], [
                external_tensor("nested", "weights.data", 6, 4)])
            target.graph.node.append(helper.make_node("If", ["condition"], ["result"],
                                                       then_branch=branch, else_branch=branch))
            onnx.save_model(target, source / "target.onnx")
            consolidate(source, output)
            packed = onnx.load(output / "model.onnx", load_external_data=False)
            for attribute in packed.graph.node[0].attribute:
                metadata = {item.key: item.value for item in attribute.g.initializer[0].external_data}
                self.assertEqual(metadata["location"], "model.onnx.data")
                self.assertEqual(metadata["offset"], "0")

    def test_shape_constants_remain_embedded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            package(source)
            target = onnx.load(source / "target.onnx", load_external_data=False)
            shape = helper.make_tensor("axes", TensorProto.INT64, [1], (1).to_bytes(8, "little"), raw=True)
            target.graph.initializer.append(shape)
            onnx.save_model(target, source / "target.onnx")
            consolidate(source, output)
            packed = onnx.load(output / "model.onnx", load_external_data=False)
            self.assertEqual(packed.graph.initializer[1], shape)

    def test_only_identical_duplicate_initializers_can_be_removed(self):
        tensor = external_tensor("weight", "weights.data", 0, 4)
        graph = helper.make_graph([], "duplicate", [], [], [tensor, tensor])
        self.assertEqual(deduplicate_initializers(graph), 1)
        self.assertEqual(len(graph.initializer), 1)
        graph.initializer.append(external_tensor("weight", "other.data", 0, 4))
        with self.assertRaisesRegex(ValueError, "Conflicting initializers"):
            deduplicate_initializers(graph)


if __name__ == "__main__":
    unittest.main()
