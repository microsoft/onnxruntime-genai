# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Creates the I/O-only encoder/decoder graphs for the Marian batch test fixtures.

Inputs carry fixed dimensions: that is what makes the fixtures reject a wrongly
sized batch/beam tensor, so each configuration needs its own fixture. Outputs
are filled by ConstantOfShape to keep the committed files small.

Prompt length matters: it must be chosen so that sizing the prompt buffer by
batch*beam rather than batch_size yields a different sequence length, or the
fixture cannot detect that regression. Length 2 works for batch 2 / beams 2
(correct width 3, incorrect width 4); length 1 does not (both give 2).

Usage:
    python create_marian_batch_models.py --output-dir ../models/marian-batch
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HIDDEN_SIZE = 512
VOCAB_SIZE = 32001
# Marian's SSRU decoder carries three recurrent state slots.
RNN_STATE_SLOTS = 3


def make_graph(
    name,
    inputs,
    outputs,
    filename,
    logits_from_attention_mask=False,
    logits_from_rnn_state=False,
    logit_token_id=None,
):
    """Builds a graph whose outputs are zero tensors of a fixed shape."""
    initializers = []
    nodes = []
    for out_name, dtype, shape in outputs:
        if out_name == "logits" and logits_from_attention_mask:
            sum_axes = numpy_helper.from_array(np.array([1], dtype=np.int64))
            sum_axes.name = "attention_mask_sum_axes"
            unsqueeze_axes = numpy_helper.from_array(np.array([1], dtype=np.int64))
            unsqueeze_axes.name = "attention_mask_unsqueeze_axes"
            batch_indices = numpy_helper.from_array(np.arange(shape[0], dtype=np.int64).reshape(shape[0], 1))
            batch_indices.name = "batch_indices"
            logits_shape = numpy_helper.from_array(np.array(shape, dtype=np.int64))
            logits_shape.name = "logits_shape"
            updates = numpy_helper.from_array(np.ones(shape[0], dtype=np.float32))
            updates.name = "selected_logit_updates"
            initializers.extend([sum_axes, unsqueeze_axes, batch_indices, logits_shape, updates])
            nodes.extend(
                [
                    helper.make_node(
                        "ReduceSum",
                        inputs=["encoder_attention_mask", sum_axes.name],
                        outputs=["unmasked_token_counts"],
                        keepdims=0,
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=["unmasked_token_counts", unsqueeze_axes.name],
                        outputs=["unmasked_token_counts_2d"],
                    ),
                    helper.make_node(
                        "Cast",
                        inputs=["unmasked_token_counts_2d"],
                        outputs=["unmasked_token_counts_int64"],
                        to=TensorProto.INT64,
                    ),
                    helper.make_node(
                        "Concat",
                        inputs=[batch_indices.name, "unmasked_token_counts_int64"],
                        outputs=["selected_logit_indices"],
                        axis=1,
                    ),
                    helper.make_node(
                        "ConstantOfShape",
                        inputs=[logits_shape.name],
                        outputs=["zero_logits"],
                        value=numpy_helper.from_array(np.zeros(1, dtype=np.float32)),
                    ),
                    helper.make_node(
                        "ScatterND",
                        inputs=["zero_logits", "selected_logit_indices", updates.name],
                        outputs=[out_name],
                    ),
                ]
            )
            continue

        if out_name == "logits" and logits_from_rnn_state:
            sum_axes = numpy_helper.from_array(np.array([0, 2], dtype=np.int64))
            sum_axes.name = "rnn_state_sum_axes"
            one = numpy_helper.from_array(np.array(1.0, dtype=np.float32))
            one.name = "token_offset"
            unsqueeze_axes = numpy_helper.from_array(np.array([1], dtype=np.int64))
            unsqueeze_axes.name = "token_unsqueeze_axes"
            batch_indices = numpy_helper.from_array(np.arange(shape[0], dtype=np.int64).reshape(shape[0], 1))
            batch_indices.name = "batch_indices"
            logits_shape = numpy_helper.from_array(np.array(shape, dtype=np.int64))
            logits_shape.name = "logits_shape"
            base_indices = numpy_helper.from_array(
                np.array([[batch, token] for batch in range(shape[0]) for token in (1, 2)], dtype=np.int64)
            )
            base_indices.name = "base_logit_indices"
            base_updates = numpy_helper.from_array(np.ones(shape[0] * 2, dtype=np.float32))
            base_updates.name = "base_logit_updates"
            initializers.extend(
                [sum_axes, one, unsqueeze_axes, batch_indices, logits_shape, base_indices, base_updates]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "ReduceSum",
                        inputs=["rnn_states_prev", sum_axes.name],
                        outputs=["rnn_state_sums"],
                        keepdims=0,
                    ),
                    helper.make_node(
                        "Add",
                        inputs=["rnn_state_sums", one.name],
                        outputs=["selected_token_ids_float"],
                    ),
                    helper.make_node(
                        "Cast",
                        inputs=["selected_token_ids_float"],
                        outputs=["selected_token_ids"],
                        to=TensorProto.INT64,
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=["selected_token_ids", unsqueeze_axes.name],
                        outputs=["selected_token_ids_2d"],
                    ),
                    helper.make_node(
                        "Concat",
                        inputs=[batch_indices.name, "selected_token_ids_2d"],
                        outputs=["selected_logit_indices"],
                        axis=1,
                    ),
                    helper.make_node(
                        "ConstantOfShape",
                        inputs=[logits_shape.name],
                        outputs=["zero_logits"],
                        value=numpy_helper.from_array(np.zeros(1, dtype=np.float32)),
                    ),
                    helper.make_node(
                        "ScatterND",
                        inputs=["zero_logits", base_indices.name, base_updates.name],
                        outputs=["base_logits"],
                    ),
                    helper.make_node(
                        "ScatterND",
                        inputs=["zero_logits", "selected_logit_indices", "rnn_state_sums"],
                        outputs=["state_logits"],
                    ),
                    helper.make_node(
                        "Add",
                        inputs=["base_logits", "state_logits"],
                        outputs=[out_name],
                    ),
                ]
            )
            continue

        if out_name == "rnn_states" and logits_from_rnn_state:
            shape_tensor = numpy_helper.from_array(np.array(shape, dtype=np.int64))
            shape_tensor.name = "rnn_states_shape"
            indices = numpy_helper.from_array(np.array([[0, beam, 0] for beam in range(shape[1])], dtype=np.int64))
            indices.name = "rnn_state_marker_indices"
            updates = numpy_helper.from_array(np.arange(1, shape[1] + 1, dtype=np.float32))
            updates.name = "rnn_state_marker_values"
            initializers.extend([shape_tensor, indices, updates])
            nodes.extend(
                [
                    helper.make_node(
                        "ConstantOfShape",
                        inputs=[shape_tensor.name],
                        outputs=["zero_rnn_states"],
                        value=numpy_helper.from_array(np.zeros(1, dtype=np.float32)),
                    ),
                    helper.make_node(
                        "ScatterND",
                        inputs=["zero_rnn_states", indices.name, updates.name],
                        outputs=[out_name],
                    ),
                ]
            )
            continue

        if out_name == "logits" and logit_token_id is not None:
            shape_tensor = numpy_helper.from_array(np.array(shape, dtype=np.int64))
            shape_tensor.name = "logits_shape"
            indices = numpy_helper.from_array(
                np.array([[batch, logit_token_id] for batch in range(shape[0])], dtype=np.int64)
            )
            indices.name = "selected_logit_indices"
            updates = numpy_helper.from_array(np.ones(shape[0], dtype=np.float32))
            updates.name = "selected_logit_updates"
            initializers.extend([shape_tensor, indices, updates])
            nodes.extend(
                [
                    helper.make_node(
                        "ConstantOfShape",
                        inputs=[shape_tensor.name],
                        outputs=["zero_logits"],
                        value=numpy_helper.from_array(np.zeros(1, dtype=np.float32)),
                    ),
                    helper.make_node(
                        "ScatterND",
                        inputs=["zero_logits", indices.name, updates.name],
                        outputs=[out_name],
                    ),
                ]
            )
            continue

        np_dtype = {
            TensorProto.FLOAT: np.float32,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
        }[dtype]
        shape_tensor = numpy_helper.from_array(np.array(shape, dtype=np.int64))
        shape_tensor.name = f"{out_name}_shape"
        initializers.append(shape_tensor)
        nodes.append(
            helper.make_node(
                "ConstantOfShape",
                inputs=[shape_tensor.name],
                outputs=[out_name],
                value=numpy_helper.from_array(np.zeros(1, dtype=np_dtype)),
            )
        )

    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
        graph=helper.make_graph(
            name=name,
            inputs=[helper.make_tensor_value_info(n, d, s) for n, d, s in inputs],
            outputs=[helper.make_tensor_value_info(n, d, s) for n, d, s in outputs],
            initializer=initializers,
            value_info=[],
            nodes=nodes,
        ),
    )
    onnx.save_model(model, filename)
    print(f"wrote {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=1)
    parser.add_argument("--logits-from-attention-mask", action="store_true")
    parser.add_argument("--logits-from-rnn-state", action="store_true")
    parser.add_argument("--logit-token-id", type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # DefaultInputIDs expands the prompt over beams, so every graph tensor is
    # batch*beam wide. MarianState::Run appends eos, widening the sequence by 1.
    batch_beam = args.batch_size * args.num_beams
    seq = args.prompt_length + 1

    make_graph(
        "encoder",
        inputs=[
            # int64 input_ids exercises the cast path in MarianInputIDs::Update.
            ("input_ids", TensorProto.INT64, [batch_beam, seq]),
            ("attention_mask", TensorProto.INT32, [batch_beam, seq]),
        ],
        outputs=[("encoder_outputs", TensorProto.FLOAT, [batch_beam, seq, HIDDEN_SIZE])],
        filename=os.path.join(args.output_dir, "encoder.onnx"),
    )

    make_graph(
        "decoder",
        inputs=[
            ("input_ids", TensorProto.INT64, [batch_beam]),
            (
                "encoder_hidden_states",
                TensorProto.FLOAT,
                [batch_beam, seq, HIDDEN_SIZE],
            ),
            ("encoder_attention_mask", TensorProto.INT32, [batch_beam, seq]),
            (
                "rnn_states_prev",
                TensorProto.FLOAT,
                [RNN_STATE_SLOTS, batch_beam, HIDDEN_SIZE],
            ),
            # MarianState binds one int64 element here, independent of batch and beams.
            ("past_key_values_length", TensorProto.INT64, [1]),
        ],
        outputs=[
            ("logits", TensorProto.FLOAT, [batch_beam, VOCAB_SIZE]),
            (
                "rnn_states",
                TensorProto.FLOAT,
                [RNN_STATE_SLOTS, batch_beam, HIDDEN_SIZE],
            ),
        ],
        filename=os.path.join(args.output_dir, "decoder.onnx"),
        logits_from_attention_mask=args.logits_from_attention_mask,
        logits_from_rnn_state=args.logits_from_rnn_state,
        logit_token_id=args.logit_token_id,
    )


if __name__ == "__main__":
    main()
