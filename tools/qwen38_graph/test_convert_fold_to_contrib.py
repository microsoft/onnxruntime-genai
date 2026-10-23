# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from convert_fold_to_contrib import convert
from onnx import TensorProto, helper


def legacy_model():
    mid = helper.make_node(
        "FoldMid2",
        ["tokens", "ids", "status", "ctl", "ctx", "prefixes", "prefix2", "ptrs", "cos", "sin", "hidden"],
        ["head_ids", "head_c", "slots", "lengths", "cos_out", "sin_out", "sel", "head_hidden"],
        domain="com.microsoft",
        head_rows=4,
        head_capacity=131076,
    )
    tail = helper.make_node("FoldTail", ["sel", "status", "ctl", "ptrs"], ["done"], domain="com.microsoft")
    graph = helper.make_graph(
        [mid, tail],
        "legacy",
        [helper.make_tensor_value_info("ctl", TensorProto.INT64, [16])],
        [helper.make_tensor_value_info("done", TensorProto.INT32, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 2)])


class ConversionTest(unittest.TestCase):
    def test_explicit_state_contract_and_existing_inputs_are_preserved(self):
        model = legacy_model()
        inputs = list(model.graph.node[0].input)
        result = convert(model)
        mid, tail = result.graph.node
        self.assertEqual(list(mid.input), inputs)
        self.assertEqual(len(mid.output), 10)
        self.assertEqual(tail.input[2], mid.output[8])
        self.assertEqual(len(tail.output), 2)
        self.assertEqual(result.opset_import[0].version, 1)
        info = {value.name: value for value in result.graph.value_info}
        self.assertEqual(info[mid.output[8]].type.tensor_type.shape.dim[0].dim_value, 16)

    def test_version_one_import_is_preserved(self):
        model = legacy_model()
        model.opset_import[0].version = 1
        self.assertEqual(convert(model).opset_import[0].version, 1)

    def test_repeat_conversion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy multi-row"):
            convert(convert(legacy_model()))

    def test_different_control_buffers_are_rejected(self):
        model = legacy_model()
        model.graph.node[1].input[2] = "other"
        with self.assertRaisesRegex(ValueError, "share"):
            convert(model)


if __name__ == "__main__":
    unittest.main()
