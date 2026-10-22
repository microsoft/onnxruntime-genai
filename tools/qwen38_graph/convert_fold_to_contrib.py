# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Convert the legacy in-place folded graph to contrib FoldMid2/FoldTail v2."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def convert(model):
    mid_nodes = [node for node in model.graph.node
                 if node.domain == "com.microsoft" and node.op_type == "FoldMid2"]
    tail_nodes = [node for node in model.graph.node
                  if node.domain == "com.microsoft" and node.op_type == "FoldTail"]
    if len(mid_nodes) != 1 or len(tail_nodes) != 1:
        raise ValueError("Expected exactly one FoldMid2 and one FoldTail")
    mid, tail = mid_nodes[0], tail_nodes[0]
    if len(mid.input) != 11 or len(mid.output) != 8 or len(tail.input) != 4 or len(tail.output) != 1:
        raise ValueError("Expected legacy multi-row fold graph with hidden-state bridge and result publication")
    if tail.input[2] != mid.input[3]:
        raise ValueError("FoldMid2 and FoldTail must share the legacy control input")
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    state_names = ["fold.contrib.mid_ctl", "fold.contrib.ctx_commit", "fold.contrib.tail_ctl"]
    if any(name in names for name in state_names):
        raise ValueError("Contrib state names already exist")
    mid.output.extend(state_names[:2])
    tail.input[2] = state_names[0]
    tail.output.append(state_names[2])
    control = next(value for value in model.graph.input if value.name == mid.input[3])
    for name in (state_names[0], state_names[2]):
        value = model.graph.value_info.add()
        value.name = name
        value.type.CopyFrom(control.type)
    model.graph.value_info.append(helper.make_tensor_value_info(state_names[1], TensorProto.INT64, [2]))
    imports = [opset for opset in model.opset_import if opset.domain == "com.microsoft"]
    if len(imports) != 1 or imports[0].version not in (1, 2):
        raise ValueError("Expected a single com.microsoft opset-1 or opset-2 import")
    imports[0].version = 2
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input.resolve().parent != args.output.resolve().parent:
        raise ValueError("Output must share the input directory to preserve external-weight paths")
    model = convert(onnx.load(args.input, load_external_data=False))
    onnx.save_model(model, args.output)


if __name__ == "__main__":
    main()
