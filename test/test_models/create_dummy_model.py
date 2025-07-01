# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Create dummy ONNX models that contain only inputs and outputs.
This is helpful for creating ONNX models to run simple API tests (e.g. pre-processing)
where the contents of the ONNX models don't matter.

Example usage:

Phi vision:
1) python create_dummy_model.py \
    --inputs "pixel_values; TensorProto.FLOAT; ['num_images', 'max_num_crops', 3, 'height', 'width']" "image_sizes; TensorProto.INT64; ['num_images', 2]" \
    --outputs "image_features; TensorProto.FLOAT; ['num_image_tokens', 3072]" \
    --filename "dummy_vision.onnx"
2) python create_dummy_model.py \
    --inputs "input_ids; TensorProto.INT64; ['batch_size', 'sequence_length']" "image_features; TensorProto.FLOAT; ['num_image_tokens', 3072]" \
    --outputs "inputs_embeds; TensorProto.FLOAT; ['batch_size', 'sequence_length', 3072]" \
    --filename "dummy_embedding.onnx"
3) python create_dummy_model.py \
    --inputs "inputs_embeds; TensorProto.FLOAT; ['batch_size', 'sequence_length', 3072]" "attention_mask; TensorProto.INT64; ['batch_size', 'total_sequence_length']" "past_key_values.0.key; TensorProto.FLOAT; ['batch_size', 32, 'past_sequence_length', 96]" "past_key_values.0.value; TensorProto.FLOAT; ['batch_size', 32, 'past_sequence_length', 96]" \
    --outputs "logits; TensorProto.FLOAT; ['batch_size', 'sequence_length', 32064]" "present.0.key; TensorProto.FLOAT; ['batch_size', 32, 'total_sequence_length', 96]" "present.0.value; TensorProto.FLOAT; ['batch_size', 32, 'total_sequence_length', 96]" \
    --filename "dummy_text.onnx"

Phi multi-modal:
4) python create_dummy_model.py \
    --inputs "pixel_values; TensorProto.FLOAT; ['num_images', 'max_num_crops', 3, 'height', 'width']" "attention_mask; TensorProto.FLOAT; ['num_images', 'max_num_crops', 32, 32]" "image_sizes; TensorProto.INT64; ['num_images', 2]" \
    --outputs "image_features; TensorProto.FLOAT; ['num_image_tokens', 3072]" \
    --filename "dummy_vision.onnx"
5) python create_dummy_model.py \
    --inputs "audio_embeds; TensorProto.FLOAT; ['num_audios', 'num_frames', 80]" "audio_attention_mask; TensorProto.BOOL; ['num_audios', 'num_frames'] "audio_sizes; TensorProto.INT64; ['num_audios']" "audio_projection_mode; TensorProto.INT64; [1]" \
    --outputs "audio_features; TensorProto.FLOAT; ['num_audio_tokens', 3072]" \
    --filename "dummy_speech.onnx"
6) python create_dummy_model.py \
    --inputs "input_ids; TensorProto.INT64; ['batch_size', 'sequence_length']" "image_features; TensorProto.FLOAT; ['num_image_tokens', 3072]" "audio_features; TensorProto.FLOAT; ['num_audio_tokens', 3072]" \
    --outputs "inputs_embeds; TensorProto.FLOAT; ['batch_size', 'sequence_length', 3072]" \
    --filename "dummy_embedding.onnx"
7) python create_dummy_model.py \
    --inputs "inputs_embeds; TensorProto.FLOAT; ['batch_size', 'sequence_length', 3072]" "attention_mask; TensorProto.INT64; ['batch_size', 'total_sequence_length']" "past_key_values.0.key; TensorProto.FLOAT; ['batch_size', 8, 'past_sequence_length', 128]" "past_key_values.0.value; TensorProto.FLOAT; ['batch_size', 8, 'past_sequence_length', 128]" \
    --outputs "logits; TensorProto.FLOAT; ['batch_size', 'sequence_length', 200064]" "present.0.key; TensorProto.FLOAT; ['batch_size', 8, 'total_sequence_length', 128]" "present.0.value; TensorProto.FLOAT; ['batch_size', 8, 'total_sequence_length', 128]" \
    --filename "dummy_text.onnx"

Whisper:
8) python create_dummy_model.py \
    --inputs "audio_features; TensorProto.FLOAT; ['batch_size', 80, 3000]" \
    --outputs "encoder_hidden_states; TensorProto.FLOAT; ['batch_size', 1500, 1280]" "present_key_cross_0; TensorProto.FLOAT; ['batch_size', 6, 1500, 64]" \
    --filename "dummy_encoder.onnx"
9) python create_dummy_model.py \
    --inputs "input_ids; TensorProto.INT32; ['batch_size', 'sequence_length']" "past_key_self_0; TensorProto.FLOAT; ['batch_size', 6, 'past_sequence_length', 64]" "past_value_self_0; TensorProto.FLOAT; ['batch_size', 6, 'past_sequence_length', 64]" "past_key_cross_0; TensorProto.FLOAT; ['batch_size', 6, 1500, 64]" "past_value_cross_0; TensorProto.FLOAT; ['batch_size', 6, 1500, 64]" \
    --outputs "logits; TensorProto.FLOAT; ['batch_size', 'sequence_length', 51865]" "present_key_self_0; TensorProto.FLOAT; ['batch_size', 6, 'total_sequence_length', 64]" "present_value_self_0; TensorProto.FLOAT; ['batch_size', 6, 'total_sequence_length', 64]" "output_cross_qk_0; TensorProto.FLOAT; ['batch_size', 6, 'sequence_length', 1500]" \
    --filename "dummy_decoder.onnx"
"""

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--inputs",
        metavar="(NAME; DTYPE; SHAPE)",
        nargs='+',
        help="Inputs of the form '(input_name; input_dtype; input_shape)' for model"
    )
    parser.add_argument(
        "-o",
        "--outputs",
        metavar="(NAME; DTYPE; SHAPE)",
        nargs='+',
        help="Outputs of the form '(output_name; output_dtype; output_shape)' for model"
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Filename to save dummy model as",
    )

    args = parser.parse_args()
    return args

def parse_args(input_or_output):
    list_of_inputs_or_outputs = []
    for input_str in input_or_output:
        input_or_output_to_add = input_str.split("; ")
        input_or_output_to_add = [elm.strip() for elm in input_or_output_to_add]
        list_of_inputs_or_outputs.append(input_or_output_to_add)
    return list_of_inputs_or_outputs

def get_input_or_output_value_infos(input_or_outputs):
    value_infos = []
    for input_or_output in input_or_outputs:
        print(input_or_output)
        name, dtype, shape = input_or_output[0], eval(input_or_output[1]), eval(input_or_output[2])
        value_info = helper.make_tensor_value_info(name, dtype, shape)
        value_infos.append(value_info)
    return value_infos

def get_dummy_tensor_shape(shape):
    np_shape = ()
    for dim in shape:
        if type(dim) == str:
            np_shape += (2,)
        elif type(dim) == int:
            np_shape += (dim,)
        else:
            raise NotImplementedError(f"Unknown dim type: {type(dim)}")
    return np_shape

def get_output_initializers(outputs):
    initializers = []
    for output in outputs:
        name, dtype, shape = output[0], eval(output[1]), eval(output[2])
        np_shape = get_dummy_tensor_shape(shape)
        np_dtype = to_numpy_dtype[dtype]
        tensor = numpy_helper.from_array(np.zeros(np_shape, dtype=np_dtype))
        tensor.name = name
        initializers.append(tensor)
    return initializers

def main():
    args = get_args()
    args.inputs = parse_args(args.inputs)
    args.outputs = parse_args(args.outputs)

    # Create dummy model
    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid('', 14)],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
        graph=helper.make_graph(
            name="main_graph",
            inputs=get_input_or_output_value_infos(args.inputs),
            outputs=get_input_or_output_value_infos(args.outputs),
            initializer=get_output_initializers(args.outputs),
            value_info=[],
            nodes=[],
        )
    )
    onnx.save_model(
        model,
        args.filename,
    )

if __name__ == "__main__":
    # Map TensorProto dtypes to NumPy dtypes
    to_numpy_dtype = {
        TensorProto.INT8: np.uint8,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.FLOAT16: np.float16,
        TensorProto.FLOAT: np.float32,
    }
    main()
