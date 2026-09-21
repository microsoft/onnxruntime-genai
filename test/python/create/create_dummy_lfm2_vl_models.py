#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Generate tiny ONNX models for LFM2-VL pipeline testing.

Writes the three models the lfm2_vl pipeline runs, with the input/output contract of the published
LFM2-VL / LFM2.5-VL exports, plus the genai_config.json that wires them together:

  dummy_vision.onnx     pixel_values [N, P, 768], pixel_attention_mask [N, P] (int64),
                        spatial_shapes [N, 2] (int64) -> image_features [T, hidden]
                        T is sum over images of ceil(rows / 2) * ceil(cols / 2), so the feature count
                        matches the <image> placeholders the processor writes.
  dummy_embedding.onnx  input_ids, image_features -> inputs_embeds (token lookup, features ignored)
  dummy_text.onnx       LFM2 decoder, layers [conv, full_attention, conv, full_attention].
                        Each logit depends on the current token and the two before it through a
                        causal convolution over past.%d.conv, so generation only matches between
                        runs when the conv state is carried correctly (e.g. across prefill chunks).

The tokenizer files and processor_config.json in the output directory are checked in separately:
tokenizer.json and tokenizer_config.json come from LiquidAI/LFM2.5-VL-1.6B.

Usage:
    python create_dummy_lfm2_vl_models.py --output test/models/lfm2-vl
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HIDDEN_SIZE = 32
NUM_KV_HEADS = 2
HEAD_SIZE = 16
CONV_CACHE_SIZE = 2
LAYER_TYPES = ["conv", "full_attention", "conv", "full_attention"]
# The tokenizer has 65536 entries. Logits are computed over the first LOGIT_VOCAB ids and padded with a
# large negative value, which keeps the checked-in weights small.
VOCAB_SIZE = 65536
LOGIT_VOCAB = 1024
PATCH_DIM = 16 * 16 * 3
OPSET = 17
IR_VERSION = 8


def _const(name, values, dtype=TensorProto.INT64, dims=None):
    values = list(values)
    return helper.make_node(
        "Constant",
        [],
        [name],
        value=helper.make_tensor(name, dtype, [len(values)] if dims is None else dims, values),
    )


def _save(graph, output_path):
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION,
        producer_name="onnxruntime-genai-test",
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def create_dummy_vision_model(output_path: str):
    """pixel_values, pixel_attention_mask, spatial_shapes -> image_features [num_image_tokens, hidden]"""
    inputs = [
        helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["num_images", "num_patches", PATCH_DIM]),
        helper.make_tensor_value_info("pixel_attention_mask", TensorProto.INT64, ["num_images", "num_patches"]),
        helper.make_tensor_value_info("spatial_shapes", TensorProto.INT64, ["num_images", 2]),
    ]
    outputs = [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_image_tokens", HIDDEN_SIZE])]

    nodes = [
        _const("one", [1]),
        _const("two", [2]),
        _const("hidden_dim", [HIDDEN_SIZE]),
        # tokens per image = ceil(rows / 2) * ceil(cols / 2), the projector's 2x2 pixel unshuffle
        helper.make_node("Add", ["spatial_shapes", "one"], ["shapes_plus_one"]),
        helper.make_node("Div", ["shapes_plus_one", "two"], ["downsampled"]),
        helper.make_node("ReduceProd", ["downsampled"], ["tokens_per_image"], axes=[1], keepdims=0),
        helper.make_node("ReduceSum", ["tokens_per_image"], ["num_tokens"], keepdims=1),
        helper.make_node("Concat", ["num_tokens", "hidden_dim"], ["features_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["features_shape"],
            ["image_features"],
            value=helper.make_tensor("value", TensorProto.FLOAT, [1], [0.01]),
        ),
    ]
    _save(helper.make_graph(nodes, "vision", inputs, outputs), output_path)


def create_dummy_embedding_model(output_path: str, rng: np.random.Generator):
    """input_ids, image_features -> inputs_embeds"""
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_image_tokens", HIDDEN_SIZE]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE]
        )
    ]
    table = numpy_helper.from_array(
        rng.standard_normal((LOGIT_VOCAB, HIDDEN_SIZE)).astype(np.float32), name="embed_table"
    )
    nodes = [
        _const("table_rows", [LOGIT_VOCAB]),
        helper.make_node("Mod", ["input_ids", "table_rows"], ["row_ids"]),
        helper.make_node("Gather", ["embed_table", "row_ids"], ["inputs_embeds"], axis=0),
    ]
    _save(helper.make_graph(nodes, "embedding", inputs, outputs, initializer=[table]), output_path)


def create_dummy_decoder_model(output_path: str, rng: np.random.Generator, hidden_states_output: bool = False):
    """inputs_embeds, attention_mask, KV cache, conv state -> logits, present KV cache, present conv state

    With hidden_states_output the values the logits are projected from come out too, as a decoder built
    with include_hidden_states=true gives them.
    """
    conv_layers = [i for i, t in enumerate(LAYER_TYPES) if t == "conv"]
    attention_layers = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
    kv_shape_past = ["batch_size", NUM_KV_HEADS, "past_sequence_length", HEAD_SIZE]
    kv_shape_total = ["batch_size", NUM_KV_HEADS, "total_sequence_length", HEAD_SIZE]
    conv_shape = ["batch_size", HIDDEN_SIZE, CONV_CACHE_SIZE]

    inputs = [
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE]
        ),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB_SIZE])
    ]
    for i in attention_layers:
        for kind in ("key", "value"):
            inputs.append(
                helper.make_tensor_value_info(f"past_key_values.{i}.{kind}", TensorProto.FLOAT, kv_shape_past)
            )
            outputs.append(helper.make_tensor_value_info(f"present.{i}.{kind}", TensorProto.FLOAT, kv_shape_total))
    for i in conv_layers:
        inputs.append(helper.make_tensor_value_info(f"past.{i}.conv", TensorProto.FLOAT, conv_shape))
        outputs.append(helper.make_tensor_value_info(f"present.{i}.conv", TensorProto.FLOAT, conv_shape))

    initializers = [
        numpy_helper.from_array(np.ones((HIDDEN_SIZE, 1, CONV_CACHE_SIZE + 1), dtype=np.float32), name="conv_weight"),
        numpy_helper.from_array(
            rng.standard_normal((HIDDEN_SIZE, LOGIT_VOCAB)).astype(np.float32), name="lm_head_weight"
        ),
    ]
    nodes = [
        _const("kv_shape", [0, 0, NUM_KV_HEADS, HEAD_SIZE]),
        _const("conv_keep_start", [-CONV_CACHE_SIZE]),
        _const("conv_keep_end", [np.iinfo(np.int64).max]),
        _const("sequence_axis", [2]),
        _const("logit_pads", [0, 0, 0, 0, 0, VOCAB_SIZE - LOGIT_VOCAB]),
        _const("logit_pad_value", [-1.0e4], dtype=TensorProto.FLOAT, dims=[]),
        # [B, S, H] -> [B, H, S] so the conv state [B, H, conv_cache_size] can be prepended in time
        helper.make_node("Transpose", ["inputs_embeds"], ["embeds_bhs"], perm=[0, 2, 1]),
    ]

    # Only the first conv layer feeds the logits; the others just roll their state forward.
    for i in conv_layers:
        nodes.append(helper.make_node("Concat", [f"past.{i}.conv", "embeds_bhs"], [f"conv_window_{i}"], axis=2))
        nodes.append(
            helper.make_node(
                "Slice",
                [f"conv_window_{i}", "conv_keep_start", "conv_keep_end", "sequence_axis"],
                [f"present.{i}.conv"],
            )
        )
    first_conv = conv_layers[0]
    nodes += [
        # Causal window sum: output t = x[t - 2] + x[t - 1] + x[t], with the earlier inputs taken from the state.
        helper.make_node("Conv", [f"conv_window_{first_conv}", "conv_weight"], ["mixed_bhs"], group=HIDDEN_SIZE),
        helper.make_node("Transpose", ["mixed_bhs"], ["mixed_bsh"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["mixed_bsh", "lm_head_weight"], ["logits_small"]),
        helper.make_node("Pad", ["logits_small", "logit_pads", "logit_pad_value"], ["logits"], mode="constant"),
        helper.make_node("Reshape", ["inputs_embeds", "kv_shape"], ["kv_bshd"]),
        helper.make_node("Transpose", ["kv_bshd"], ["kv_new"], perm=[0, 2, 1, 3]),
    ]
    for i in attention_layers:
        for kind in ("key", "value"):
            nodes.append(
                helper.make_node("Concat", [f"past_key_values.{i}.{kind}", "kv_new"], [f"present.{i}.{kind}"], axis=2)
            )

    if hidden_states_output:
        outputs.append(
            helper.make_tensor_value_info(
                "hidden_states", TensorProto.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE]
            )
        )
        nodes.append(helper.make_node("Identity", ["mixed_bsh"], ["hidden_states"]))

    _save(helper.make_graph(nodes, "decoder", inputs, outputs, initializer=initializers), output_path)


def create_genai_config(output_path: str):
    session_options = {"log_id": "onnxruntime-genai", "provider_options": []}
    config = {
        "model": {
            "bos_token_id": 1,
            "context_length": 4096,
            "decoder": {
                "session_options": session_options,
                "filename": "dummy_text.onnx",
                "head_size": HEAD_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                    "past_conv_names": "past.%d.conv",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                    "present_conv_names": "present.%d.conv",
                },
                "num_attention_heads": NUM_KV_HEADS,
                "num_hidden_layers": len(LAYER_TYPES),
                "num_key_value_heads": NUM_KV_HEADS,
                "layer_types": LAYER_TYPES,
                "conv_cache_size": CONV_CACHE_SIZE,
            },
            "eos_token_id": 7,
            "pad_token_id": 0,
            "type": "lfm2_vl",
            "vocab_size": VOCAB_SIZE,
            "embedding": {
                "session_options": session_options,
                "filename": "dummy_embedding.onnx",
                "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
            },
            "vision": {
                "session_options": session_options,
                "filename": "dummy_vision.onnx",
                "config_filename": "processor_config.json",
                "patch_size": 16,
                "spatial_merge_size": 2,
                "max_num_patches": 1024,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "attention_mask": "pixel_attention_mask",
                    "image_sizes": "spatial_shapes",
                },
                "outputs": {"image_features": "image_features"},
            },
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": 4096,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate tiny ONNX models for LFM2-VL pipeline testing")
    parser.add_argument("--output", type=str, default="test/models/lfm2-vl", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(42)
    create_dummy_vision_model(os.path.join(args.output, "dummy_vision.onnx"))
    create_dummy_embedding_model(os.path.join(args.output, "dummy_embedding.onnx"), rng)
    create_dummy_decoder_model(os.path.join(args.output, "dummy_text.onnx"), rng)
    create_genai_config(os.path.join(args.output, "genai_config.json"))
    print(f"Wrote dummy LFM2-VL models to {args.output}")


if __name__ == "__main__":
    main()
