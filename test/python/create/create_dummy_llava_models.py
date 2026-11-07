#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Generate a tiny "llava" model for testing the fixed-resolution vision -> embedding -> decoder pipeline.

  vision.onnx     pixel_values [N, 3, 8, 8] -> image_features [N, 4, 16]
                  Each 4x4 patch of the image is projected to one feature vector, so the features
                  depend on the image content and are emitted in row-major patch order. With
                  --rank2 the features are flattened to [N * 4, 16] and genai_config.json sets
                  num_visual_tokens, which can't be read from that shape.
  embedding.onnx  input_ids [B, L], image_features [N, T, 16] -> inputs_embeds [B, L, 16]
                  Token lookup, with the rows of image_features written, in order, at the positions
                  holding the image token (the LLaVA masked scatter).
  decoder.onnx    inputs_embeds, attention_mask, past_key_values.0.{key,value} -> logits, present.0.*
                  Each input embedding is multiplied by its 1-based absolute position and stored,
                  weighted, in the KV cache. The logits at each position are a projection of the sum
                  of all weighted embeddings so far, so every generated token depends on every image
                  feature, on the position it was placed at and on the cache being carried
                  correctly. Special tokens are never generated.

The tokenizer knows the ten digits and the special tokens <unk>, <s>, </s> and <image>.

Usage:
    python create_dummy_llava_models.py --output test/models/llava
    python create_dummy_llava_models.py --output test/models/llava-rank2 --rank2
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IMAGE_SIZE = 8
PATCH_SIZE = 4
CHANNELS = 3
TOKENS_PER_IMAGE = (IMAGE_SIZE // PATCH_SIZE) ** 2
HIDDEN_SIZE = 16
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<image>"]
IMAGE_TOKEN_ID = SPECIAL_TOKENS.index("<image>")
VOCAB = SPECIAL_TOKENS + [str(d) for d in range(10)]
VOCAB_SIZE = len(VOCAB)
OPSET = 17
IR_VERSION = 8


def _const(name, values, dtype=TensorProto.INT64):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name, dtype, [len(values)], list(values)))


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _features_shape(rank2):
    return ["num_image_tokens", HIDDEN_SIZE] if rank2 else ["num_images", TOKENS_PER_IMAGE, HIDDEN_SIZE]


def make_vision(path, rng, rank2):
    grid = IMAGE_SIZE // PATCH_SIZE
    patch_dim = CHANNELS * PATCH_SIZE * PATCH_SIZE
    weight = (rng.standard_normal((patch_dim, HIDDEN_SIZE)) / np.sqrt(patch_dim)).astype(np.float32)
    nodes = [
        _const("patch_shape", [-1, CHANNELS, grid, PATCH_SIZE, grid, PATCH_SIZE]),
        _const("tokens_shape", [-1, TOKENS_PER_IMAGE, patch_dim]),
        helper.make_node("Reshape", ["pixel_values", "patch_shape"], ["patches"]),
        helper.make_node("Transpose", ["patches"], ["patches_t"], perm=[0, 2, 4, 1, 3, 5]),
        helper.make_node("Reshape", ["patches_t", "tokens_shape"], ["tokens"]),
    ]
    if rank2:
        nodes += [
            helper.make_node("MatMul", ["tokens", "vision_weight"], ["batched_features"]),
            _const("features_shape", [-1, HIDDEN_SIZE]),
            helper.make_node("Reshape", ["batched_features", "features_shape"], ["image_features"]),
        ]
    else:
        nodes.append(helper.make_node("MatMul", ["tokens", "vision_weight"], ["image_features"]))
    graph = helper.make_graph(
        nodes,
        "vision",
        [
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, ["num_images", CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, _features_shape(rank2))],
        [numpy_helper.from_array(weight, "vision_weight")],
    )
    _save(graph, path)


def make_embedding(path, rng, rank2):
    table = rng.standard_normal((VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32)
    nodes = [
        helper.make_node("Gather", ["embedding_table", "input_ids"], ["text_embeds"]),
        helper.make_node("Equal", ["input_ids", "image_token_id"], ["is_image"]),
        _const("flat", [-1]),
        _const("flat_features", [-1, HIDDEN_SIZE]),
        helper.make_node("Reshape", ["is_image", "flat"], ["is_image_flat"]),
        helper.make_node("Cast", ["is_image_flat"], ["is_image_int"], to=TensorProto.INT64),
        helper.make_node("CumSum", ["is_image_int", "zero"], ["image_rank"]),
        helper.make_node("Sub", ["image_rank", "one"], ["feature_index"]),
        helper.make_node("Reshape", ["image_features", "flat_features"], ["features"]),
        # A zero row after the features keeps the Gather in range when there are none (text-only
        # prompts and decode steps); positions that are not image tokens point at it.
        helper.make_node("Concat", ["features", "zero_row"], ["features_padded"], axis=0),
        helper.make_node("Shape", ["features"], ["features_shape"]),
        helper.make_node("Gather", ["features_shape", "zero"], ["num_features"], axis=0),
        helper.make_node("Where", ["is_image_flat", "feature_index", "num_features"], ["safe_index"]),
        helper.make_node("Gather", ["features_padded", "safe_index"], ["picked"], axis=0),
        helper.make_node("Shape", ["text_embeds"], ["embeds_shape"]),
        helper.make_node("Reshape", ["picked", "embeds_shape"], ["image_embeds"]),
        helper.make_node("Unsqueeze", ["is_image", "last_axis"], ["is_image_3d"]),
        helper.make_node("Where", ["is_image_3d", "image_embeds", "text_embeds"], ["inputs_embeds"]),
    ]
    initializers = [
        numpy_helper.from_array(table, "embedding_table"),
        numpy_helper.from_array(np.array(IMAGE_TOKEN_ID, dtype=np.int64), "image_token_id"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_axis"),
        numpy_helper.from_array(np.zeros((1, HIDDEN_SIZE), dtype=np.float32), "zero_row"),
    ]
    graph = helper.make_graph(
        nodes,
        "embedding",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
            helper.make_tensor_value_info(
                "image_features",
                TensorProto.FLOAT,
                ["num_image_tokens", HIDDEN_SIZE] if rank2 else ["num_images", "num_image_tokens", HIDDEN_SIZE],
            ),
        ],
        [
            helper.make_tensor_value_info(
                "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE]
            )
        ],
        initializers,
    )
    _save(graph, path)


def make_decoder(path, rng):
    weight = rng.standard_normal((HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32)
    kv_shape = ["batch_size", 1, "past_sequence_length", HIDDEN_SIZE]
    present_shape = ["batch_size", 1, "total_sequence_length", HIDDEN_SIZE]
    nodes = [
        # positions = past_sequence_length + [1, ..., sequence_length], shaped [1, L, 1]
        helper.make_node("Shape", ["past_key_values.0.key"], ["past_shape"]),
        helper.make_node("Gather", ["past_shape", "two"], ["past_length"], axis=0),
        helper.make_node("Shape", ["inputs_embeds"], ["embeds_shape"]),
        helper.make_node("Gather", ["embeds_shape", "one"], ["sequence_length"], axis=0),
        helper.make_node("Add", ["past_length", "one"], ["first_position"]),
        helper.make_node("Add", ["first_position", "sequence_length"], ["end_position"]),
        helper.make_node("Range", ["first_position", "end_position", "one"], ["positions"]),
        helper.make_node("Cast", ["positions"], ["positions_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["positions_float", "position_axes"], ["position_weights"]),
        helper.make_node("Mul", ["inputs_embeds", "position_weights"], ["weighted_embeds"]),
        helper.make_node("Unsqueeze", ["weighted_embeds", "head_axis"], ["embeds_4d"]),
        helper.make_node("Concat", ["past_key_values.0.key", "embeds_4d"], ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.value", "embeds_4d"], ["present.0.value"], axis=2),
        helper.make_node("ReduceSum", ["past_key_values.0.key", "sequence_axis"], ["past_sum"], keepdims=0),
        helper.make_node("CumSum", ["weighted_embeds", "one"], ["running_sum"]),
        helper.make_node("Add", ["running_sum", "past_sum"], ["state"]),
        helper.make_node("Sin", ["state"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "lm_head"], ["digit_logits"]),
        helper.make_node("Add", ["digit_logits", "lm_bias"], ["logits"]),
    ]
    # Only digits are ever generated, so a run always lasts until max_length.
    bias = np.zeros(VOCAB_SIZE, dtype=np.float32)
    bias[: len(SPECIAL_TOKENS)] = -1e4
    initializers = [
        numpy_helper.from_array(weight, "lm_head"),
        numpy_helper.from_array(bias, "lm_bias"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "head_axis"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "sequence_axis"),
        numpy_helper.from_array(np.array([0, 2], dtype=np.int64), "position_axes"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
        numpy_helper.from_array(np.array(2, dtype=np.int64), "two"),
    ]
    graph = helper.make_graph(
        nodes,
        "decoder",
        [
            helper.make_tensor_value_info(
                "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE]
            ),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
            helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, kv_shape),
            helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, kv_shape),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB_SIZE]),
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, present_shape),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, present_shape),
        ],
        initializers,
    )
    _save(graph, path)


def make_tokenizer(output):
    added = [
        {
            "id": i,
            "content": t,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for i, t in enumerate(SPECIAL_TOKENS)
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {t: i for i, t in enumerate(VOCAB)},
            "merges": [],
        },
    }
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": 256,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(os.path.join(output, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2)
        f.write("\n")
    with open(os.path.join(output, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
        f.write("\n")


def make_configs(output, rank2):
    genai_config = {
        "model": {
            "type": "llava",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "image_token_id": IMAGE_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
            "context_length": 256,
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": HIDDEN_SIZE,
                "head_size": HIDDEN_SIZE,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "session_options": {"provider_options": []},
            },
            "embedding": {
                "filename": "embedding.onnx",
                "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
            },
            "vision": {
                "filename": "vision.onnx",
                "config_filename": "processor_config.json",
                "inputs": {"pixel_values": "pixel_values"},
                "outputs": {"image_features": "image_features"},
            },
        },
        "search": {
            "do_sample": False,
            "max_length": 256,
            "num_beams": 1,
            "past_present_share_buffer": False,
            "top_k": 1,
        },
    }

    if rank2:
        genai_config["model"]["vision"]["num_visual_tokens"] = TOKENS_PER_IMAGE

    def op(name, type_, attrs):
        return {"operation": {"name": name, "type": type_, "attrs": attrs}}

    processor_config = {
        "processor": {
            "name": "llava_image_processor",
            "transforms": [
                op("decode_image", "DecodeImage", {"color_space": "RGB"}),
                op(
                    "resize",
                    "Resize",
                    {"interpolation": "LINEAR", "width": IMAGE_SIZE, "height": IMAGE_SIZE, "keep_aspect_ratio": 0},
                ),
                op("rescale", "Rescale", {"rescale_factor": 1.0 / 255.0}),
                op("normalize", "Normalize", {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}),
                op("to_channel_first", "Permute3D", {"dims": [2, 0, 1]}),
            ],
        }
    }
    for name, config in (("genai_config.json", genai_config), ("processor_config.json", processor_config)):
        with open(os.path.join(output, name), "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--rank2", action="store_true", help="Emit image features as [total_tokens, hidden] instead of rank 3"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(0)
    make_vision(os.path.join(args.output, "vision.onnx"), rng, args.rank2)
    make_embedding(os.path.join(args.output, "embedding.onnx"), rng, args.rank2)
    make_decoder(os.path.join(args.output, "decoder.onnx"), rng)
    make_tokenizer(args.output)
    make_configs(args.output, args.rank2)


if __name__ == "__main__":
    main()
