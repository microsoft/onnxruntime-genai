# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Create executed sparse-cache fixtures for the legacy Generator dispatch tests.

These are arithmetic state machines, not miniature neural networks: every forward
appends real KV rows and advances fixed convolution/recurrent tensors. Logits depend
on all carried states, so stale, missing or double-bound caches change generation.

Regenerate with numpy, onnx and onnx-ir installed:
    python test/python/create/create_cache_dispatch_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir


def create_model(
    directory: Path, *, recurrent: bool, model_type: str, multimodal: bool = False, bf16: bool = False
) -> None:
    nodes: list[ir.Node] = []
    inputs: list[ir.Value] = []
    initializers: list[ir.Value] = []
    outputs: list[ir.Value] = []

    def value(name, dtype, shape):
        return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

    def constant(name, data, dtype=np.float32):
        tensor = ir.tensor(np.asarray(data, dtype=dtype), name=name)
        result = ir.Value(
            name=name,
            type=ir.TensorType(tensor.dtype),
            shape=tensor.shape,
            const_value=tensor,
        )
        initializers.append(result)
        return result

    def op(kind, args, *, name=None, dtype=ir.DataType.FLOAT, shape=(), attrs=()):
        result = value(name or f"value_{len(nodes)}", dtype, shape)
        nodes.append(ir.Node("", kind, args, attrs, outputs=[result]))
        return result

    if multimodal:
        embeds = value("inputs_embeds", ir.DataType.FLOAT, ["batch_size", "sequence_length", 2])
        position_ids = value("position_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        mask = value("attention_mask", ir.DataType.INT64, ["batch_size", "total_sequence_length"])
        inputs.extend([embeds, position_ids, mask])
        tokens = op(
            "Gather",
            [embeds, constant("embedding_channel", 0, np.int64)],
            shape=["batch_size", "sequence_length"],
            attrs=[ir.AttrInt64("axis", 2)],
        )
        positions = op("Cast", [position_ids], shape=["batch_size", "sequence_length"], attrs=[ir.AttrInt64("to", 1)])
        tokens = op("Add", [tokens, positions], shape=["batch_size", "sequence_length"])
        shape_source = position_ids
    else:
        input_ids = value("input_ids", ir.DataType.INT32, ["batch_size", "sequence_length"])
        inputs.append(input_ids)
        tokens = op(
            "Cast",
            [input_ids],
            shape=["batch_size", "sequence_length"],
            attrs=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
        )
        shape_source = input_ids
    token_sum = op("ReduceSum", [tokens], attrs=[ir.AttrInt64("keepdims", 0)])
    if multimodal:
        mask_sum = op("ReduceSum", [mask], dtype=ir.DataType.INT64, attrs=[ir.AttrInt64("keepdims", 0)])
        mask_sum = op("Cast", [mask_sum], attrs=[ir.AttrInt64("to", 1)])
        token_sum = op("Add", [token_sum, mask_sum])
    state_sum = constant("zero", 0)
    fixed_indices = [0, 3] if recurrent else [0, 2]
    attention_index = 2 if recurrent else 1

    for slot, layer in enumerate(fixed_indices):
        kinds = [("conv_state", [2, 3], 1 + 2 * slot)]
        if recurrent:
            kinds.append(("ssm_state" if multimodal else "recurrent_state", [2, 2, 2], 2 + 2 * slot))
        for kind, row_shape, scale in kinds:
            shape = ["batch_size", *row_shape]
            reduced = bf16 and kind == "conv_state"
            state_dtype = ir.DataType.BFLOAT16 if reduced else ir.DataType.FLOAT
            past = value(f"past_key_values.{layer}.{kind}", state_dtype, shape)
            inputs.append(past)
            increment = op("Mul", [token_sum, constant(f"scale_{layer}_{kind}", scale)])
            if reduced:
                # Exercise reduced-precision state storage independently of
                # arithmetic kernels: accumulate in FP32, then commit BF16 state.
                accumulator = op("Cast", [past], shape=shape, attrs=[ir.AttrInt64("to", 1)])
                update = op("Add", [accumulator, increment], shape=shape)
                present = op(
                    "Cast",
                    [update],
                    name=f"present.{layer}.{kind}",
                    dtype=state_dtype,
                    shape=shape,
                    attrs=[ir.AttrInt64("to", int(state_dtype))],
                )
            else:
                present = op("Add", [past, increment], name=f"present.{layer}.{kind}", dtype=state_dtype, shape=shape)
            outputs.append(present)
            statistics = op("Cast", [present], shape=shape, attrs=[ir.AttrInt64("to", 1)]) if reduced else present
            # Every element in a state has the same value. Its mean is independent
            # of state geometry, making the scalar oracle simply sum(scale)*sum(tokens).
            mean = op("ReduceMean", [statistics], attrs=[ir.AttrInt64("keepdims", 0)])
            state_sum = op("Add", [state_sum, mean])

    rows = op(
        "Unsqueeze",
        [tokens, constant("kv_axes", [1, 3], np.int64)],
        shape=["batch_size", 1, "sequence_length", 1],
    )
    for kind, scale in [("key", 1), ("value", 2)]:
        past = value(
            f"past_key_values.{attention_index}.{kind}",
            ir.DataType.BFLOAT16 if bf16 else ir.DataType.FLOAT,
            ["batch_size", 1, "past_sequence_length", 1],
        )
        inputs.append(past)
        new_rows = op(
            "Mul",
            [rows, constant(f"kv_scale_{kind}", scale)],
            shape=["batch_size", 1, "sequence_length", 1],
        )
        if bf16:
            new_rows = op(
                "Cast",
                [new_rows],
                dtype=ir.DataType.BFLOAT16,
                shape=["batch_size", 1, "sequence_length", 1],
                attrs=[ir.AttrInt64("to", int(ir.DataType.BFLOAT16))],
            )
        present = op(
            "Concat",
            [past, new_rows],
            name=f"present.{attention_index}.{kind}",
            dtype=ir.DataType.BFLOAT16 if bf16 else ir.DataType.FLOAT,
            shape=["batch_size", 1, "total_sequence_length", 1],
            attrs=[ir.AttrInt64("axis", 2)],
        )
        outputs.append(present)
        statistics = (
            op("Cast", [present], shape=["batch_size", 1, "total_sequence_length", 1], attrs=[ir.AttrInt64("to", 1)])
            if bf16
            else present
        )
        total = op("ReduceSum", [statistics], attrs=[ir.AttrInt64("keepdims", 0)])
        state_sum = op("Add", [state_sum, total])

    # Never emit EOS (15): target = 1 + (sum(fixed state means) + sum(KV)) % 14.
    score = op("Cast", [state_sum], dtype=ir.DataType.INT64, attrs=[ir.AttrInt64("to", 7)])
    score = op("Mod", [score, constant("modulus", 14, np.int64)], dtype=ir.DataType.INT64)
    score = op("Add", [score, constant("one", 1, np.int64)], dtype=ir.DataType.INT64)
    match = op(
        "Equal",
        [constant("vocabulary", np.arange(16), np.int64), score],
        dtype=ir.DataType.BOOL,
        shape=[16],
    )
    logits = op("Cast", [match], shape=[16], attrs=[ir.AttrInt64("to", 1)])
    token_shape = op("Shape", [shape_source], dtype=ir.DataType.INT64, shape=[2])
    logits_shape = op(
        "Concat",
        [token_shape, constant("vocab_shape", [16], np.int64)],
        dtype=ir.DataType.INT64,
        shape=[3],
        attrs=[ir.AttrInt64("axis", 0)],
    )
    outputs.insert(
        0,
        op(
            "Expand",
            [logits, logits_shape],
            name="logits",
            shape=["batch_size", "sequence_length", 16],
        ),
    )
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 18},
        name="cache_dispatch",
    )
    model = ir.Model(graph, ir_version=9, producer_name="onnxruntime-genai-cache-dispatch-test")
    directory.mkdir(parents=True, exist_ok=True)
    ir.save(model, directory / "decoder.onnx")
    onnx.checker.check_model(directory / "decoder.onnx", full_check=True)
    config = {
        "model": {
            "type": model_type,
            "vocab_size": 16,
            "context_length": 32,
            "eos_token_id": 15,
            "pad_token_id": 0,
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 1,
                "num_hidden_layers": 4 if recurrent else 3,
                "layer_types": (
                    ["mamba", "moe", "attention", "mamba"] if recurrent else ["conv", "full_attention", "conv"]
                ),
                "inputs": {"input_ids": "input_ids"},
                "outputs": {"logits": "logits"},
            },
        },
        "search": {
            "max_length": 16,
            "num_beams": 1,
            "do_sample": False,
            "past_present_share_buffer": False,
        },
    }
    if multimodal:
        config["model"]["image_token_id"] = 18
        config["model"]["embedding"] = {
            "filename": "embedding.onnx",
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }
        config["model"]["vision"] = {
            "filename": "vision.onnx",
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }
        decoder = config["model"]["decoder"]
        decoder["inputs"] = {
            "inputs_embeds": "inputs_embeds",
            "attention_mask": "attention_mask",
            "position_ids": "position_ids",
            "past_recurrent_names": "past_key_values.%d.ssm_state",
        }
        decoder["outputs"]["present_recurrent_names"] = "present.%d.ssm_state"
    (directory / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "models" / "cache-dispatch"
    create_model(root / "recurrent", recurrent=True, model_type="decoder")
    create_model(root / "conv", recurrent=False, model_type="decoder")
    create_model(root / "lfm2", recurrent=False, model_type="lfm2")
    create_model(root / "recurrent-bf16", recurrent=True, model_type="decoder", bf16=True)
