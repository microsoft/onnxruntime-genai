# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Create the deterministic hybrid model used by the fixed-state-pool tests.

The graph only has to *load* as a decoder-only ``Generators::Model`` on CPU.
The ``FixedStatePool`` C++ unit tests read the fixed convolution and recurrent
state-group metadata from the model's session and allocate their own tensors,
so the ONNX graph is never executed. Following the same "contents don't matter"
approach as ``create_dummy_engine_model.py``, the graph declares the decoder
inputs/outputs plus two fixed state groups and produces every present output
with an ``Identity`` (or a zero-filled initializer for logits).

Two fixed groups are declared so the tests can exercise manifest ordering and
per-group geometry:

    convolution: layer_ids [0, 3], state shape [batch, 2, 3]
    recurrent:   layer_ids [2, 5], state shape [batch, 2, 2, 2]

Both declare a dynamic (symbolic) batch axis 0, so the pool derives per-request
row geometry from axes >= 1 and admits any batch size up to capacity.
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB_SIZE = 16
NUM_LAYERS = 6
CONTEXT_LENGTH = 128
EOS_TOKEN_ID = 1

# Fixed state groups: (name, layer_ids, non-batch dims).
CONV_LAYERS = [0, 3]
RECURRENT_LAYERS = [2, 5]
CONV_ROW = [2, 3]
RECURRENT_ROW = [2, 2, 2]

STATE_UPDATE_CAPACITY = 3


def _zeros(name, shape, dtype=np.float32):
    tensor = numpy_helper.from_array(np.zeros(shape, dtype=dtype))
    tensor.name = name
    return tensor


def create_decoder(output_dir):
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("state_update_capture_count", TensorProto.INT32, ["batch_size"]),
        helper.make_tensor_value_info("state_update_active", TensorProto.INT32, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB_SIZE]),
    ]
    initializers = [
        _zeros("logits", (2, 2, VOCAB_SIZE)),
    ]
    nodes = []

    def add_fixed_group(prefix, layer_ids, row_dims):
        for layer in layer_ids:
            in_name = f"past_{prefix}.{layer}"
            out_name = f"present_{prefix}.{layer}"
            shape = ["batch_size", *row_dims]
            inputs.append(helper.make_tensor_value_info(in_name, TensorProto.FLOAT, shape))
            outputs.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, shape))
            # Produce the present output with an Identity so it inherits the dynamic batch axis.
            nodes.append(helper.make_node("Identity", [in_name], [out_name]))
            if prefix == "conv":
                update_name = f"state_update.{layer}.conv_value"
                outputs.append(
                    helper.make_tensor_value_info(
                        update_name,
                        TensorProto.FLOAT,
                        ["batch_size", STATE_UPDATE_CAPACITY, row_dims[0]],
                    )
                )
                nodes.append(helper.make_node("Transpose", [in_name], [update_name], perm=[0, 2, 1]))
                continue

            decay_name = f"state_update.{layer}.recurrent_decay"
            key_name = f"state_update.{layer}.recurrent_key"
            delta_name = f"state_update.{layer}.recurrent_delta"
            capsule_name = f"state_update.{layer}.recurrent_capsule"
            capsule_width = STATE_UPDATE_CAPACITY * (row_dims[0] + row_dims[2] + row_dims[0] * row_dims[1])
            outputs.append(
                helper.make_tensor_value_info(capsule_name, TensorProto.FLOAT, ["batch_size", capsule_width])
            )
            decay_base = f"{decay_name}/base"
            key_base = f"{key_name}/base"
            delta_base = f"{delta_name}/base"
            decay_step = f"{decay_name}/step"
            key_step = f"{key_name}/step"
            delta_step = f"{delta_name}/step"
            nodes.extend(
                [
                    helper.make_node("ReduceSum", [in_name, "gdn_decay_axes"], [decay_base], keepdims=0),
                    helper.make_node("Unsqueeze", [decay_base, "state_update_axis"], [decay_step]),
                    helper.make_node("Concat", [decay_step] * STATE_UPDATE_CAPACITY, [decay_name], axis=1),
                    helper.make_node("ReduceSum", [in_name, "gdn_key_axes"], [key_base], keepdims=0),
                    helper.make_node("Unsqueeze", [key_base, "gdn_key_unsqueeze_axes"], [key_step]),
                    helper.make_node("Concat", [key_step] * STATE_UPDATE_CAPACITY, [key_name], axis=1),
                    helper.make_node("ReduceSum", [in_name, "gdn_delta_axes"], [delta_base], keepdims=0),
                    helper.make_node("Unsqueeze", [delta_base, "state_update_axis"], [delta_step]),
                    helper.make_node("Concat", [delta_step] * STATE_UPDATE_CAPACITY, [delta_name], axis=1),
                    helper.make_node("Flatten", [decay_name], [f"{decay_name}/flat"], axis=1),
                    helper.make_node("Flatten", [key_name], [f"{key_name}/flat"], axis=1),
                    helper.make_node("Flatten", [delta_name], [f"{delta_name}/flat"], axis=1),
                    helper.make_node(
                        "Concat",
                        [f"{decay_name}/flat", f"{key_name}/flat", f"{delta_name}/flat"],
                        [capsule_name],
                        axis=1,
                    ),
                ]
            )

    initializers.extend(
        [
            numpy_helper.from_array(np.array([1], dtype=np.int64), "state_update_axis"),
            numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "gdn_decay_axes"),
            numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "gdn_key_axes"),
            numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "gdn_key_unsqueeze_axes"),
            numpy_helper.from_array(np.array([3], dtype=np.int64), "gdn_delta_axes"),
        ]
    )

    add_fixed_group("conv", CONV_LAYERS, CONV_ROW)
    add_fixed_group("recurrent", RECURRENT_LAYERS, RECURRENT_ROW)

    graph = helper.make_graph(
        nodes=nodes,
        name="synthetic_hybrid_decoder",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
        value_info=[],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
    )
    onnx.checker.check_model(model)
    path = os.path.join(output_dir, "decoder.onnx")
    onnx.save_model(model, path)
    print(f"  Saved decoder -> {path}")


def create_config(output_dir):
    config = {
        "model": {
            "type": "gpt2",
            "bos_token_id": 0,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "decoder.onnx",
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 1,
                "hidden_size": 1,
                "num_hidden_layers": NUM_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                },
                "outputs": {
                    "logits": "logits",
                },
                "state_groups": [
                    {
                        "kind": "fixed",
                        "layer_ids": CONV_LAYERS,
                        "bindings": {
                            "state": {"input": "past_conv.%d", "output": "present_conv.%d"},
                        },
                        "state_update": {
                            "kind": "causal_conv",
                            "capacity": STATE_UPDATE_CAPACITY,
                            "capture_count": "state_update_capture_count",
                            "active": "state_update_active",
                            "value": "state_update.%d.conv_value",
                        },
                    },
                    {
                        "kind": "fixed",
                        "layer_ids": RECURRENT_LAYERS,
                        "bindings": {
                            "state": {"input": "past_recurrent.%d", "output": "present_recurrent.%d"},
                        },
                        "state_update": {
                            "kind": "gated_delta_net",
                            "capacity": STATE_UPDATE_CAPACITY,
                            "capture_count": "state_update_capture_count",
                            "active": "state_update_active",
                            "capsule": "state_update.%d.recurrent_capsule",
                            "key_head_count": 1,
                        },
                    },
                ],
            },
        },
        "search": {"max_length": CONTEXT_LENGTH, "do_sample": False},
    }
    path = os.path.join(output_dir, "genai_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "engine", "synthetic-hybrid"),
    )
    args = parser.parse_args()
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating synthetic hybrid engine test model in {output_dir}")
    create_decoder(output_dir)
    create_config(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
