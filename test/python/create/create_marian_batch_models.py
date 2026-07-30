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


def make_graph(name, inputs, outputs, filename):
    """Builds a graph whose outputs are zero tensors of a fixed shape."""
    initializers = []
    nodes = []
    for out_name, dtype, shape in outputs:
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
        outputs=[
            ("encoder_outputs", TensorProto.FLOAT, [batch_beam, seq, HIDDEN_SIZE])
        ],
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
            # Scalar int64: MarianState writes it via GetTensorMutableData<int64_t>.
            ("past_key_values_length", TensorProto.INT64, []),
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
    )


if __name__ == "__main__":
    main()
