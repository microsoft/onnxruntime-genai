# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Create the prepared-image three-model Nemotron-Omni cache plumbing fixture.

This tests generic runtime roles and named tensors, not C-RADIO preprocessing,
video pruning, native Mamba2 mathematics or packed checkpoint weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from create_cache_dispatch_models import create_model


def create_auxiliary(directory: Path, *, vision: bool, retention: bool = False) -> None:
    nodes = []
    initializers = []

    def value(name, dtype, shape):
        return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

    def constant(name, data, dtype=np.int64):
        tensor = ir.tensor(np.asarray(data, dtype=dtype), name=name)
        result = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=tensor.shape, const_value=tensor)
        initializers.append(result)
        return result

    def op(kind, args, *, name=None, dtype=ir.DataType.FLOAT, shape=(), attrs=()):
        result = value(name or f"v_{len(nodes)}", dtype, shape)
        nodes.append(ir.Node("", kind, args, attrs, outputs=[result]))
        return result

    if vision:
        pixels = value("pixel_values", ir.DataType.FLOAT, ["num_images", 3, 2, 2])
        inputs = [pixels]
        means = op(
            "ReduceMean",
            [pixels, constant("pixel_axes", [1, 2, 3])],
            shape=["num_images"],
            attrs=[ir.AttrInt64("keepdims", 0)],
        )
        rows = op("Unsqueeze", [means, constant("feature_axis", [1])], shape=["num_images", 1])
        features = op(
            "Concat",
            [rows, rows],
            name="image_features",
            shape=["num_images", 2],
            attrs=[ir.AttrInt64("axis", 1)],
        )
        outputs = [features]
        if retention:
            is_video = value("is_video", ir.DataType.BOOL, [])
            inputs.append(is_video)
            count = op("Size", [means], dtype=ir.DataType.INT64)
            indices = op(
                "Range",
                [constant("start", 0), count, constant("step", 1)],
                dtype=ir.DataType.INT64,
                shape=["num_images"],
            )
            image_mask = op("Equal", [indices, indices], dtype=ir.DataType.BOOL, shape=["num_images"])
            video_mask = op(
                "Greater",
                [means, constant("threshold", 4, np.float32)],
                dtype=ir.DataType.BOOL,
                shape=["num_images"],
            )

            def branch(source, name):
                output = value(name, ir.DataType.BOOL, ["num_images"])
                return ir.Graph(
                    [],
                    [output],
                    nodes=[ir.Node("", "Identity", [source], outputs=[output])],
                    name=name,
                    opset_imports={"": 18},
                )

            # This is an arithmetic dynamic-K contract, not temporal packing or
            # EVS mathematics: images retain every row; videos retain means > 4.
            mask = op(
                "If",
                [is_video],
                name="retention_mask",
                dtype=ir.DataType.BOOL,
                shape=["num_images"],
                attrs=[
                    ir.AttrGraph("then_branch", branch(video_mask, "video_retention")),
                    ir.AttrGraph("else_branch", branch(image_mask, "image_retention")),
                ],
            )
            nonzero = op("NonZero", [mask], dtype=ir.DataType.INT64, shape=[1, "kept_tokens"])
            kept = op(
                "Squeeze",
                [nonzero, constant("index_axis", [0])],
                name="kept_indices",
                dtype=ir.DataType.INT64,
                shape=["kept_tokens"],
            )
            features.name = "dense_features"
            selected = op("Gather", [features, kept], name="image_features", shape=["kept_tokens", 2])
            outputs = [selected, mask, kept]
    else:
        ids = value("input_ids", ir.DataType.INT32, ["batch_size", "sequence_length"])
        features = value("image_features", ir.DataType.FLOAT, ["num_image_tokens", 2])
        inputs = [ids, features]
        floats = op("Cast", [ids], shape=["batch_size", "sequence_length"], attrs=[ir.AttrInt64("to", 1)])
        rows = op(
            "Unsqueeze",
            [floats, constant("embedding_axis", [2])],
            shape=["batch_size", "sequence_length", 1],
        )
        text = op("Concat", [rows, rows], shape=["batch_size", "sequence_length", 2], attrs=[ir.AttrInt64("axis", 2)])
        mask = op(
            "Equal",
            [ids, constant("image_token", 18, np.int32)],
            dtype=ir.DataType.BOOL,
            shape=["batch_size", "sequence_length"],
        )
        flags = op(
            "Cast",
            [mask],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
            attrs=[ir.AttrInt64("to", 7)],
        )
        indices = op(
            "CumSum",
            [flags, constant("sequence_axis", 1)],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
        )
        # Row zero is safe for text-only decode with no new media. Image positions
        # index rows 1..N in the prepared prompt's flat projected-feature stream.
        padded = op(
            "Concat",
            [constant("empty_feature", [[0, 0]], np.float32), features],
            shape=["padded_image_tokens", 2],
            attrs=[ir.AttrInt64("axis", 0)],
        )
        selected = op("Gather", [padded, indices], shape=["batch_size", "sequence_length", 2])
        mask = op(
            "Unsqueeze",
            [mask, constant("mask_axis", [2])],
            dtype=ir.DataType.BOOL,
            shape=["batch_size", "sequence_length", 1],
        )
        outputs = [
            op("Where", [mask, selected, text], name="inputs_embeds", shape=["batch_size", "sequence_length", 2])
        ]
    role = "vision" if vision else "embedding"
    graph = ir.Graph(inputs, outputs, nodes=nodes, initializers=initializers, opset_imports={"": 18}, name=role)
    ir.save(ir.Model(graph, ir_version=9), directory / f"{role}.onnx")
    onnx.checker.check_model(directory / f"{role}.onnx", full_check=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "models" / "cache-dispatch" / "multimodal"
    create_model(root, recurrent=True, model_type="nemotron_h_omni", multimodal=True)
    create_auxiliary(root, vision=True)
    create_auxiliary(root, vision=False)
    retained = root.with_name("multimodal-retention")
    create_model(retained, recurrent=True, model_type="nemotron_h_omni", multimodal=True)
    create_auxiliary(retained, vision=True, retention=True)
    create_auxiliary(retained, vision=False)
    config_path = retained / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["vision"]["inputs"]["is_video"] = "is_video"
    config["model"]["vision"]["outputs"].update(retention_mask="retention_mask", kept_indices="kept_indices")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
