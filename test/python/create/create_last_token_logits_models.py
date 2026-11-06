# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate tiny, executable CPU fixtures for the native Logits component tests."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def create_model(path, *, pruned, indices_type=None):
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch", "sequence"])]
    if indices_type is not None:
        inputs.append(helper.make_tensor_value_info("last_token_indices", indices_type, ["batch", 1, 1]))
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["tokens"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["tokens", "axes"], ["hidden"]),
    ]
    selected = "hidden"
    if pruned:
        nodes.append(helper.make_node("GatherElements", ["hidden", "last_token_indices"], ["selected"], axis=1))
        selected = "selected"
    nodes.append(helper.make_node("Concat", [selected] * 4, ["logits"], axis=2))
    graph = helper.make_graph(
        nodes,
        "last_token_logits",
        inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1 if pruned else "sequence", 4])],
        [helper.make_tensor("axes", TensorProto.INT64, [1], [2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
        producer_name="onnxruntime-genai",
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models/engine/last-token-logits",
    )
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    create_model(output / "pruned.onnx", pruned=True, indices_type=TensorProto.INT32)
    create_model(output / "invalid-index-type.onnx", pruned=True, indices_type=TensorProto.INT64)
    create_model(output / "unpruned-with-indices.onnx", pruned=False, indices_type=TensorProto.INT32)
    create_model(output / "full.onnx", pruned=False)


if __name__ == "__main__":
    main()
