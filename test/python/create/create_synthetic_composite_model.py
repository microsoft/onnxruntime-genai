# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Create the deterministic composite model used by the Engine composite tests.

This model declares one ``paged_kv`` decoder state group and two ``fixed``
decoder state groups, plus ``engine.dynamic_batching``. It is the fixture for
the C++ composite-transaction tests (``engine_step_tests.cpp``): those tests
wire the real ``PagedCacheManager`` (so a real ``PagedKeyValueCache`` and real
``FixedStatePool`` are constructed from this model's session) but drive it with a
``RecordingModelExecutor`` that fabricates logits, so the ONNX graph is never
executed.

The first fixed convolution state increments on every run and feeds back into
the next-token score; the remaining fixed outputs are ``Identity`` pass-throughs.
The graph also consumes packed absolute ``position_ids``. The paged portion reuses the executable graph from
``create_synthetic_paged_model.py`` so the paged cache and manifest validate
identically to the paged fixture. ``HybridDecoderIO`` binds the fixed tensors
alongside the packed inputs, making the graph useful for both ownership tests
with a recording executor and end-to-end dynamic Engine tests.

Layer assignment (disjoint cover of 6 layers):

    fixed convolution: layer_ids [0, 3], state shape [batch, 2, 3]
    paged_kv:          layer_ids [1, 4]
    fixed recurrent:   layer_ids [2, 5], state shape [batch, 2, 2, 2]

Both fixed groups declare a dynamic (symbolic) batch axis 0, so the pool admits
any batch size up to capacity.
"""

import argparse
import json
import os

import numpy as np
import onnx
from create_synthetic_paged_model import (
    BLOCK_SIZE,
    CONTEXT_LENGTH,
    EOS_TOKEN_ID,
    MAX_BATCH_SIZE,
    NUM_BLOCKS,
    NUM_LAYERS,
    PAGED_LAYERS,
    VOCAB_SIZE,
)
from create_synthetic_paged_model import (
    _decoder_graph as _paged_decoder_graph,
)
from onnx import TensorProto, helper, numpy_helper

CONV_LAYERS = [0, 3]
RECURRENT_LAYERS = [2, 5]
CONV_ROW = [2, 3]
RECURRENT_ROW = [2, 2, 2]
STATE_UPDATE_CAPACITY = 3


def _const(name, array):
    tensor = numpy_helper.from_array(np.asarray(array))
    tensor.name = name
    return tensor


def _fixed_group(prefix, layer_ids, row_dims):
    inputs = []
    outputs = []
    nodes = []
    for layer in layer_ids:
        input_name = f"past_{prefix}.{layer}"
        output_name = f"present_{prefix}.{layer}"
        shape = ["batch_size", *row_dims]
        inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape))
        outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
        if prefix == "conv" and layer == 0:
            nodes.append(helper.make_node("Add", [input_name, "state_one"], [output_name]))
        else:
            nodes.append(helper.make_node("Identity", [input_name], [output_name]))

        if prefix == "conv":
            update_name = f"state_update.{layer}.conv_value"
            outputs.append(
                helper.make_tensor_value_info(
                    update_name,
                    TensorProto.FLOAT,
                    ["batch_size", STATE_UPDATE_CAPACITY, row_dims[0]],
                )
            )
            nodes.append(helper.make_node("Transpose", [input_name], [update_name], perm=[0, 2, 1]))
            continue

        decay_name = f"state_update.{layer}.recurrent_decay"
        key_name = f"state_update.{layer}.recurrent_key"
        delta_name = f"state_update.{layer}.recurrent_delta"
        capsule_name = f"state_update.{layer}.recurrent_capsule"
        capsule_width = STATE_UPDATE_CAPACITY * (row_dims[0] + row_dims[2] + row_dims[0] * row_dims[1])
        outputs.append(helper.make_tensor_value_info(capsule_name, TensorProto.FLOAT, ["batch_size", capsule_width]))
        decay_base = f"{decay_name}/base"
        key_base = f"{key_name}/base"
        delta_base = f"{delta_name}/base"
        decay_step = f"{decay_name}/step"
        key_step = f"{key_name}/step"
        delta_step = f"{delta_name}/step"
        nodes.extend(
            [
                helper.make_node("ReduceSum", [input_name, "gdn_decay_axes"], [decay_base], keepdims=0),
                helper.make_node("Unsqueeze", [decay_base, "state_update_axis"], [decay_step]),
                helper.make_node("Concat", [decay_step] * STATE_UPDATE_CAPACITY, [decay_name], axis=1),
                helper.make_node("ReduceSum", [input_name, "gdn_key_axes"], [key_base], keepdims=0),
                helper.make_node("Unsqueeze", [key_base, "gdn_key_unsqueeze_axes"], [key_step]),
                helper.make_node("Concat", [key_step] * STATE_UPDATE_CAPACITY, [key_name], axis=1),
                helper.make_node("ReduceSum", [input_name, "gdn_delta_axes"], [delta_base], keepdims=0),
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
    return inputs, outputs, nodes


def _decoder_graph():
    graph = _paged_decoder_graph()
    graph.name = "synthetic_composite_decoder"
    graph.input.extend(
        [
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, [3, "num_tokens"]
            ),
            helper.make_tensor_value_info(
                "state_update_capture_count", TensorProto.INT32, ["batch_size"]
            ),
        ]
    )
    graph.initializer.extend(
        [
            _const("state_one", np.asarray(1.0, dtype=np.float32)),
            _const("c2", np.asarray(2, dtype=np.int64)),
            _const("axes12", np.asarray([1, 2], dtype=np.int64)),
            _const("state_update_axis", np.asarray([1], dtype=np.int64)),
            _const("gdn_decay_axes", np.asarray([2, 3], dtype=np.int64)),
            _const("gdn_key_axes", np.asarray([1, 2], dtype=np.int64)),
            _const("gdn_key_unsqueeze_axes", np.asarray([1, 2], dtype=np.int64)),
            _const("gdn_delta_axes", np.asarray([3], dtype=np.int64)),
        ]
    )

    conv_inputs, conv_outputs, conv_nodes = _fixed_group("conv", CONV_LAYERS, CONV_ROW)
    recurrent_inputs, recurrent_outputs, recurrent_nodes = _fixed_group("recurrent", RECURRENT_LAYERS, RECURRENT_ROW)
    graph.input.extend([*conv_inputs, *recurrent_inputs])
    graph.output.extend([*conv_outputs, *recurrent_outputs])

    nodes = list(graph.node)
    score_index = next(index for index, node in enumerate(nodes) if "score_f" in node.output)
    nodes[score_index].output[0] = "base_score_f"
    current_length_index = next(
        index for index, node in enumerate(nodes) if "current_length" in node.output
    )
    nodes[current_length_index].input[0] = "text_position_ids"
    position_node = helper.make_node(
        "Gather", ["position_ids", "c2"], ["text_position_ids"], axis=0
    )
    fixed_bias_nodes = [
        helper.make_node("ReduceSum", ["past_conv.0", "axes12"], ["fixed_state_sum"], keepdims=0),
        helper.make_node("Gather", ["fixed_state_sum", "row_id"], ["fixed_state_bias"], axis=0),
        helper.make_node("Add", ["base_score_f", "fixed_state_bias"], ["score_f"]),
    ]
    graph.ClearField("node")
    graph.node.extend(
        [
            *nodes[:current_length_index],
            position_node,
            *nodes[current_length_index : score_index + 1],
            *fixed_bias_nodes,
            *nodes[score_index + 1 :],
            *conv_nodes,
            *recurrent_nodes,
        ]
    )
    return graph


def create_decoder(output_dir):
    model = helper.make_model(
        _decoder_graph(),
        opset_imports=[helper.make_operatorsetid("", 17)],
        ir_version=9,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, os.path.join(output_dir, "decoder.onnx"))


def create_config(output_dir):
    state_groups = [
        {
            "kind": "fixed",
            "layer_ids": CONV_LAYERS,
            "bindings": {"state": {"input": "past_conv.%d", "output": "present_conv.%d"}},
            "state_update": {
                "kind": "causal_conv",
                "capacity": STATE_UPDATE_CAPACITY,
                "capture_count": "state_update_capture_count",
                "value": "state_update.%d.conv_value",
            },
        },
        {
            "kind": "paged_kv",
            "layer_ids": PAGED_LAYERS,
            "bindings": {
                "key": {
                    "input": "past_key_values.%d.key",
                    "output": "present.%d.key",
                },
                "value": {
                    "input": "past_key_values.%d.value",
                    "output": "present.%d.value",
                },
            },
        },
        {
            "kind": "fixed",
            "layer_ids": RECURRENT_LAYERS,
            "bindings": {
                "state": {
                    "input": "past_recurrent.%d",
                    "output": "present_recurrent.%d",
                }
            },
            "state_update": {
                "kind": "gated_delta_net",
                "capacity": STATE_UPDATE_CAPACITY,
                "capture_count": "state_update_capture_count",
                "capsule": "state_update.%d.recurrent_capsule",
                "key_head_count": 1,
            },
        },
    ]
    config = {
        "model": {
            "type": "decoder",
            "bos_token_id": 0,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "decoder": {
                "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
                "filename": "decoder.onnx",
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 1,
                "hidden_size": 1,
                "num_hidden_layers": NUM_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "block_table": "block_table",
                    "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                    "past_sequence_lengths": "past_sequence_lengths",
                    "attention_metadata": "attention_metadata",
                    "position_ids": "position_ids",
                    "past_key_names": "legacy_past.%d.key",
                    "past_value_names": "legacy_past.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "legacy_present.%d.key",
                    "present_value_names": "legacy_present.%d.value",
                },
                "state_groups": state_groups,
            },
        },
        "search": {"max_length": CONTEXT_LENGTH, "do_sample": False},
        "engine": {
            "dynamic_batching": {
                "block_size": BLOCK_SIZE,
                "num_blocks": NUM_BLOCKS,
                "max_batch_size": MAX_BATCH_SIZE,
            }
        },
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as file:
        json.dump(config, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "engine", "synthetic-composite"),
    )
    output_dir = os.path.normpath(parser.parse_args().output_dir)
    os.makedirs(output_dir, exist_ok=True)
    create_decoder(output_dir)
    create_config(output_dir)


if __name__ == "__main__":
    main()
