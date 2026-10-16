# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Real conditional/cache fixtures for the optional decoder cache-phase input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from create_cache_dispatch_models import create_model
from create_multimodal_cache_model import create_auxiliary


def value(name, dtype, shape):
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def create(directory: Path, *, multimodal=False, alias="has_previous_state", rank=0):
    create_model(
        directory, recurrent=True, model_type="nemotron_h_omni" if multimodal else "decoder", multimodal=multimodal
    )
    if multimodal:
        create_auxiliary(directory, vision=True, retention=True)
        create_auxiliary(directory, vision=False)
        embedding = ir.load(directory / "embedding.onnx")
        features = next(x for x in embedding.graph.inputs if x.name == "image_features")
        features.dtype = ir.DataType.BFLOAT16
        converted = value("float_features", ir.DataType.FLOAT, ["num_image_tokens", 2])
        original_nodes = list(embedding.graph)
        for node in original_nodes:
            for i, inp in enumerate(node.inputs):
                if inp is features:
                    node.replace_input_with(i, converted)
        embedding.graph.insert_before(
            original_nodes[0], ir.Node("", "Cast", [features], [ir.AttrInt64("to", 1)], outputs=[converted])
        )
        ir.save(embedding, directory / "embedding.onnx")
    model = ir.load(directory / "decoder.onnx")
    graph = model.graph
    phase = value(alias, ir.DataType.BOOL, [] if rank == 0 else [1])
    graph.inputs.append(phase)
    original_nodes = list(graph)
    for past in list(graph.inputs):
        if not past.name.startswith("past_key_values."):
            continue
        shape = value(f"{past.name}_shape", ir.DataType.INT64, [len(past.shape)])
        selected = value(f"{past.name}_selected", past.dtype, list(past.shape))
        then_out = value(f"{past.name}_then", past.dtype, list(past.shape))
        else_out = value(f"{past.name}_else", past.dtype, list(past.shape))
        then = ir.Graph([], [then_out], nodes=[ir.Node("", "Identity", [past], outputs=[then_out])], name="cached")
        otherwise = ir.Graph(
            [], [else_out], nodes=[ir.Node("", "ConstantOfShape", [shape], outputs=[else_out])], name="fresh"
        )
        # False does not read dirty cache data; true consumes the preceding outputs.
        prefix = [
            ir.Node("", "Shape", [past], outputs=[shape]),
            ir.Node(
                "",
                "If",
                [phase],
                [ir.AttrGraph("then_branch", then), ir.AttrGraph("else_branch", otherwise)],
                outputs=[selected],
            ),
        ]
        for node in original_nodes:
            for i, inp in enumerate(node.inputs):
                if inp is past:
                    node.replace_input_with(i, selected)
        graph.insert_before(original_nodes[0], prefix)
    observed = value("observed_phase", ir.DataType.BOOL, [] if rank == 0 else [1])
    graph.append(ir.Node("", "Identity", [phase], outputs=[observed]))
    graph.outputs.append(observed)
    if not multimodal:
        offset = value("cache_offset", ir.DataType.INT64, [])
        observed_offset = value("observed_offset", ir.DataType.INT64, [])
        graph.inputs.append(offset)
        graph.append(ir.Node("", "Identity", [offset], outputs=[observed_offset]))
        graph.outputs.append(observed_offset)

        def constant(name, data):
            tensor = ir.tensor(np.asarray(data, dtype=np.int64), name=name)
            result = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=tensor.shape, const_value=tensor)
            graph.initializers[name] = result
            return result

        failure = value("fail_requested", ir.DataType.BOOL, [])
        reshape_shape = value("execution_shape", ir.DataType.INT64, [1])
        checked = value("execution_check", ir.DataType.INT64, [1])
        graph.append(ir.Node("", "Equal", [offset, constant("failure_offset", -1)], outputs=[failure]))
        graph.append(
            ir.Node(
                "",
                "Where",
                [failure, constant("invalid_shape", [-2]), constant("valid_shape", [1])],
                outputs=[reshape_shape],
            )
        )
        graph.append(ir.Node("", "Reshape", [constant("execution_data", [0]), reshape_shape], outputs=[checked]))
        graph.outputs.append(checked)
    ir.save(model, directory / "decoder.onnx")
    onnx.checker.check_model(directory / "decoder.onnx", full_check=True)
    config = json.loads((directory / "genai_config.json").read_text())
    config["model"]["decoder"]["inputs"]["has_previous_state"] = alias
    (directory / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "models" / "cache-dispatch"
    create(root / "phase")
    create(root / "phase-alias", alias="cache_is_valid")
    create(root / "phase-multimodal", multimodal=True)
    create(root / "phase-rank", rank=1)
