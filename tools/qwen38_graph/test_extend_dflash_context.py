# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from extend_dflash_context import CONTEXT, GUARD, resize_target, rotary_rows, update_rotary_pool


class ContextTests(unittest.TestCase):
    def model(self):
        nodes = [helper.make_node("PackedSparseAttentionIndexer", [], [], state_capacity=32769)
                 for _ in range(12)]
        nodes.append(helper.make_node("FoldMid2", [], [], head_capacity=131076))
        graph = helper.make_graph(nodes, "capacity",
                                  [helper.make_tensor_value_info("kv", onnx.TensorProto.INT8, [1, 2, 131076, 256]),
                                   helper.make_tensor_value_info("index", onnx.TensorProto.BFLOAT16, [1, 32769, 128])],
                                  [], [numpy_helper.from_array(np.asarray([131076], np.int32),
                                                              "generic_guarded_dsa_total_sequence_length")])
        return helper.make_model(graph)

    def test_all_capacity_contracts_are_updated_together(self):
        model = self.model()
        changes = resize_target(model, 131076)
        self.assertEqual(changes["index_attributes"], 12)
        self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[2].dim_value, CONTEXT + GUARD)
        self.assertEqual(model.graph.input[1].type.tensor_type.shape.dim[1].dim_value, 65537)
        self.assertEqual(numpy_helper.to_array(model.graph.initializer[0]).tolist(), [CONTEXT + GUARD])
        self.assertEqual(model.graph.node[-1].attribute[0].i, CONTEXT + GUARD)

    def test_unexpected_capacity_cannot_be_silently_replaced(self):
        model = self.model()
        model.graph.node[0].attribute[0].i = 9
        with self.assertRaisesRegex(ValueError, "index state capacity"):
            resize_target(model, 131076)

    def test_rotary_generation_is_position_invariant_and_covers_guard(self):
        for draft in (False, True):
            width, theta = (64, 10000.0) if draft else (32, 10000000.0)
            full = rotary_rows(CONTEXT - 1, CONTEXT + GUARD, width, theta, draft)
            last = rotary_rows(CONTEXT + GUARD - 1, CONTEXT + GUARD, width, theta, draft)
            for a, b in zip(full, last):
                self.assertEqual(a.shape, (GUARD + 1, width))
                np.testing.assert_array_equal(a[-1:], b)

    def test_shared_pool_cannot_override_extended_rotary_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights_pool.txt"
            path.write_text("qwen38_weights_pool 1\npool 0 16 0 16777216 2 262144 32 model.onnx.data\n"
                            "session target head.mtp.rotary_emb.cos_cache 0\n")
            tensor = onnx.TensorProto(name="head.mtp.rotary_emb.cos_cache",
                                      data_type=onnx.TensorProto.BFLOAT16, dims=[CONTEXT + GUARD, 32],
                                      data_location=onnx.TensorProto.EXTERNAL)
            for key, value in (("location", "model.onnx.data"), ("offset", "1234"),
                               ("length", str((CONTEXT + GUARD) * 64))):
                entry = tensor.external_data.add()
                entry.key, entry.value = key, value
            model = helper.make_model(helper.make_graph([], "pool", [], [], [tensor]))
            update_rotary_pool(path, model)
            fields = path.read_text().splitlines()[1].split()
            self.assertEqual(fields[3], "1234")
            self.assertEqual(fields[4], str((CONTEXT + GUARD) * 64))
            self.assertEqual(fields[6:8], [str(CONTEXT + GUARD), "32"])


if __name__ == "__main__":
    unittest.main()
