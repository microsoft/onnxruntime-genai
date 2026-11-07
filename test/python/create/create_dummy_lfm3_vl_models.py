#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Generate a tiny LFM3-VL pipeline for testing.

Writes the three models the lfm3_vl pipeline runs, with the input/output contract of the model builder's LFM3-VL
exports, the genai_config.json and processor_config.json the builder writes for LiquidAI/LFM3-VL-2B, and a
tokenizer made here:

  dummy_vision.onnx     pixel_values [N, 1536], image_grid_thw [1, 3] (int64) -> image_features [N / 4, hidden]
                        One image per run. Each feature projects the mean of 4 consecutive rows, so the features
                        depend on the pixels, and the Reshape fails when the grid and the row count disagree.
  dummy_embedding.onnx  input_ids, image_features -> inputs_embeds
                        Token lookup with image_features scattered over the positions holding <|image|>.
  dummy_text.onnx       The LFM2 decoder of the LFM2-VL fixture (see create_dummy_lfm2_vl_models.py): each logit
                        depends on the current input embedding and the two before it, so image features reach
                        only the logits at and just after their positions.
  tokenizer.json        Byte-level BPE with no merges: one token per byte, plus the chat and vision markers as
                        special tokens. It holds no vocabulary from any released model.

Usage:
    python create_dummy_lfm3_vl_models.py --output test/models/lfm3-vl
"""

import argparse
import json
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_dummy_lfm2_vl_models import (
    HIDDEN_SIZE,
    LOGIT_VOCAB,
    _const,
    _save,
    create_dummy_decoder_model,
    create_genai_config,
)

ROW_WIDTH = 3 * 2 * 16 * 16  # channels * temporal_patch_size * patch_size**2
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image|>",
    "<|video|>",
]
# Special tokens follow the 256 byte tokens
PAD_ID, BOS_ID, EOS_ID, IMAGE_TOKEN_ID = 256, 257, 259, 262


def create_dummy_vision_model(output_path: str, rng: np.random.Generator):
    """pixel_values, image_grid_thw -> image_features [num_image_tokens, hidden]"""
    inputs = [
        helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["num_patches", ROW_WIDTH]),
        helper.make_tensor_value_info("image_grid_thw", TensorProto.INT64, [1, 3]),
    ]
    outputs = [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_image_tokens", HIDDEN_SIZE])]
    projection = numpy_helper.from_array(
        rng.standard_normal((ROW_WIDTH, HIDDEN_SIZE)).astype(np.float32) * 0.2, name="projection"
    )
    nodes = [
        _const("grid_shape", [3]),
        _const("rows_index", [1]),
        _const("cols_index", [2]),
        _const("merge_size", [2]),
        _const("rows_per_token", [4]),
        _const("row_width", [ROW_WIDTH]),
        helper.make_node("Reshape", ["image_grid_thw", "grid_shape"], ["grid"]),
        helper.make_node("Gather", ["grid", "rows_index"], ["rows"]),
        helper.make_node("Gather", ["grid", "cols_index"], ["cols"]),
        helper.make_node("Div", ["rows", "merge_size"], ["block_rows"]),
        helper.make_node("Div", ["cols", "merge_size"], ["block_cols"]),
        helper.make_node("Mul", ["block_rows", "block_cols"], ["tokens"]),
        helper.make_node("Concat", ["tokens", "rows_per_token", "row_width"], ["blocks_shape"], axis=0),
        helper.make_node("Reshape", ["pixel_values", "blocks_shape"], ["blocks"]),
        helper.make_node("ReduceMean", ["blocks"], ["block_means"], axes=[1], keepdims=0),
        helper.make_node("MatMul", ["block_means", "projection"], ["image_features"]),
    ]
    _save(helper.make_graph(nodes, "vision", inputs, outputs, initializer=[projection]), output_path)


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
        _const("image_token_id", [IMAGE_TOKEN_ID]),
        helper.make_node("Mod", ["input_ids", "table_rows"], ["row_ids"]),
        helper.make_node("Gather", ["embed_table", "row_ids"], ["token_embeds"], axis=0),
        # A feature count that differs from the number of <|image|> tokens fails in ScatterND
        helper.make_node("Equal", ["input_ids", "image_token_id"], ["is_image"]),
        helper.make_node("NonZero", ["is_image"], ["image_positions_t"]),
        helper.make_node("Transpose", ["image_positions_t"], ["image_positions"], perm=[1, 0]),
        helper.make_node("ScatterND", ["token_embeds", "image_positions", "image_features"], ["inputs_embeds"]),
    ]
    _save(helper.make_graph(nodes, "embedding", inputs, outputs, initializer=[table]), output_path)


def create_tokenizer(output_dir: str):
    # GPT-2's byte-to-character table, so every byte is one token of the byte-level BPE
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    others = [byte for byte in range(256) if byte not in printable]
    characters = [chr(byte) for byte in printable] + [chr(256 + i) for i in range(len(others))]
    special = [
        {"id": 256 + i, "content": token, "single_word": False, "lstrip": False, "rstrip": False}
        | {"normalized": False, "special": True}
        for i, token in enumerate(SPECIAL_TOKENS)
    ]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": special,
        "normalizer": None,
        "pre_tokenizer": byte_level,
        "post_processor": None,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {character: i for i, character in enumerate(characters)},
            "merges": [],
        },
    }
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": SPECIAL_TOKENS[BOS_ID - 256],
        "eos_token": SPECIAL_TOKENS[EOS_ID - 256],
        "pad_token": SPECIAL_TOKENS[PAD_ID - 256],
        "add_bos_token": False,
        "model_max_length": 4096,
    }
    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
        f.write("\n")


def write_configs(output_dir: str):
    """genai_config.json and processor_config.json, with the sections the builder writes for LFM3-VL-2B."""
    config_path = os.path.join(output_dir, "genai_config.json")
    create_genai_config(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    model = config["model"]
    model.update(type="lfm3_vl", bos_token_id=BOS_ID, eos_token_id=EOS_ID, pad_token_id=PAD_ID)
    model["vision"] = {
        "filename": "dummy_vision.onnx",
        "config_filename": "processor_config.json",
        "patch_size": 16,
        "spatial_merge_size": 2,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    def step(name, kind, **attrs):
        return {"operation": {"name": name, "type": kind, "attrs": attrs}}

    transforms = [
        step("decode_image", "DecodeImage", color_space="RGB"),
        step(
            "resize",
            "Resize",
            interpolation="CUBIC",
            smart_resize=1,
            min_pixels=65536,
            max_pixels=16777216,
            patch_size=16,
            merge_size=2,
        ),
        step("rescale", "Rescale", rescale_factor=1 / 255),
        step("normalize", "Normalize", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        step("patch_image", "PatchImage", patch_size=16, temporal_patch_size=2, merge_size=2),
    ]
    with open(os.path.join(output_dir, "processor_config.json"), "w", encoding="utf-8") as f:
        json.dump({"processor": {"name": "lfm3_vl_image_processor", "transforms": transforms}}, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny LFM3-VL pipeline for testing")
    parser.add_argument("--output", type=str, default="test/models/lfm3-vl", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(42)
    create_dummy_vision_model(os.path.join(args.output, "dummy_vision.onnx"), rng)
    create_dummy_embedding_model(os.path.join(args.output, "dummy_embedding.onnx"), rng)
    create_dummy_decoder_model(os.path.join(args.output, "dummy_text.onnx"), rng)
    write_configs(args.output)
    create_tokenizer(args.output)
    print(f"Wrote dummy LFM3-VL models to {args.output}")


if __name__ == "__main__":
    main()
