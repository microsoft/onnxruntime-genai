# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tiny, causal models for testing multimodal continuation without real weights.

Vision produces one scalar per patch: sum(pixel channels) + 11. Text embeddings
are 2 * token_id + 3; image placeholders receive the corresponding vision scalar.
The decoder caches K = embedding + position and V = 2 * embedding + 3 * position
+ 1. Its logits depend on the sum of both caches and the causal current prefix.
For mRoPE, position = temporal + 2 * height + 4 * width; otherwise it is 7 * id.
All arithmetic is exactly representable for the short test conversations.

These models deliberately test state/I/O contracts, not real-model quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper, numpy_helper

VOCAB_SIZE = 32
EOS_TOKEN_ID = 31
IMAGE_TOKEN_ID = 29
VISION_START_TOKEN_ID = 28
VISION_END_TOKEN_ID = 27
QWEN_FAMILIES = ("qwen2_5_vl", "qwen3_vl", "fara")


def _tensor(name, data, dtype=np.int64):
    return numpy_helper.from_array(np.asarray(data, dtype=dtype), name)


def _info(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def _model(name, nodes, inputs, outputs, initializers):
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    return model


def _text_embedding_nodes():
    return [
        helper.make_node("Cast", ["input_ids"], ["float_ids"], to=T.FLOAT),
        helper.make_node("Mul", ["float_ids", "two"], ["scaled_ids"]),
        helper.make_node("Add", ["scaled_ids", "three"], ["text_values"]),
        helper.make_node("Unsqueeze", ["text_values", "axis_2"], ["text_embeds"]),
    ]


def make_embedding_model(family):
    nodes = _text_embedding_nodes()
    nodes.extend(
        [
            helper.make_node(
                "Less" if family == "phi3v" else "Equal",
                ["input_ids", "placeholder"],
                ["is_image"],
            ),
            helper.make_node("Cast", ["is_image"], ["image_mask"], to=T.INT64),
            helper.make_node("CumSum", ["image_mask", "sequence_axis"], ["image_indices"]),
            helper.make_node("Where", ["is_image", "image_indices", "zero_int"], ["safe_indices"]),
            # Index zero also makes empty image_features safe during text-only decode.
            helper.make_node("Concat", ["empty_feature", "image_features"], ["padded_features"], axis=0),
            helper.make_node("Gather", ["padded_features", "safe_indices"], ["image_embeds"], axis=0),
            helper.make_node("Unsqueeze", ["is_image", "axis_2"], ["image_mask_3d"]),
            helper.make_node("Where", ["image_mask_3d", "image_embeds", "text_embeds"], ["inputs_embeds"]),
        ]
    )
    return _model(
        "embedding",
        nodes,
        [_info("input_ids", T.INT32, ["batch", "seq"]), _info("image_features", T.FLOAT, ["patches", 1])],
        [_info("inputs_embeds", T.FLOAT, ["batch", "seq", 1])],
        [
            _tensor("two", 2, np.float32),
            _tensor("three", 3, np.float32),
            _tensor("axis_2", [2]),
            _tensor("sequence_axis", 1),
            _tensor("placeholder", 0 if family == "phi3v" else IMAGE_TOKEN_ID, np.int32),
            _tensor("zero_int", 0),
            _tensor("empty_feature", [[0]], np.float32),
        ],
    )


def make_vision_model(family, *, fail_on_negative_pixels=False):
    if family in QWEN_FAMILIES:
        shape, axes = ["patches", 3], [1]
        metadata = [_info("image_grid_thw", T.INT64, ["images", 3])]
    elif family == "mistral3":
        shape, axes = ["images", 3, 1, 1], [1, 2, 3]
        metadata = []
    else:
        shape, axes = ["images", 1, 3, 1, 1], [1, 2, 3, 4]
        metadata = [_info("image_sizes", T.INT64, ["images", 2])]
    nodes = [
        helper.make_node("ReduceSum", ["pixel_values", "pixel_axes"], ["channel_sum"], keepdims=0),
        helper.make_node("Unsqueeze", ["channel_sum", "axis_1"], ["patch_sum"]),
        helper.make_node(
            "Add", ["patch_sum", "eleven"], ["unguarded_features" if fail_on_negative_pixels else "image_features"]
        ),
    ]
    constants = [_tensor("pixel_axes", axes), _tensor("axis_1", [1]), _tensor("eleven", 11, np.float32)]
    if fail_on_negative_pixels:
        # Valid metadata reaches ONNX execution, where a negative pixel selects an invalid index.
        nodes.extend(
            [
                helper.make_node("ReduceMin", ["pixel_values"], ["smallest_pixel"], keepdims=0),
                helper.make_node("Less", ["smallest_pixel", "zero_float"], ["must_fail"]),
                helper.make_node("Cast", ["must_fail"], ["guard_index"], to=T.INT64),
                helper.make_node("Gather", ["guard_table", "guard_index"], ["guard_bias"]),
                helper.make_node("Add", ["unguarded_features", "guard_bias"], ["image_features"]),
            ]
        )
        constants.extend([_tensor("zero_float", 0, np.float32), _tensor("guard_table", [0], np.float32)])
    return _model(
        "vision",
        nodes,
        [_info("pixel_values", T.FLOAT, shape), *metadata],
        [_info("image_features", T.FLOAT, ["patches", 1])],
        constants,
    )


def make_decoder_model(family):
    qwen = family in QWEN_FAMILIES
    text_only = family == "llama"
    nodes = _text_embedding_nodes() if text_only else []
    if text_only:
        nodes.append(helper.make_node("Identity", ["text_embeds"], ["inputs_embeds"]))
    nodes.append(helper.make_node("Cast", ["position_ids"], ["float_positions"], to=T.FLOAT))
    if qwen:
        nodes.extend(
            [
                helper.make_node("Mul", ["float_positions", "position_weights"], ["weighted_positions"]),
                helper.make_node("ReduceSum", ["weighted_positions", "axis_0"], ["position_sum"], keepdims=0),
            ]
        )
    else:
        nodes.append(helper.make_node("Mul", ["float_positions", "seven"], ["position_sum"]))
    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["position_sum", "axis_2"], ["position"]),
            helper.make_node("Add", ["inputs_embeds", "position"], ["current_key"]),
            helper.make_node("Mul", ["inputs_embeds", "two"], ["double_embed"]),
            helper.make_node("Mul", ["position", "three"], ["triple_position"]),
            helper.make_node("Add", ["double_embed", "triple_position"], ["value_without_bias"]),
            helper.make_node("Add", ["value_without_bias", "one"], ["current_value"]),
        ]
    )
    for kind in ("key", "value"):
        nodes.extend(
            [
                helper.make_node("Unsqueeze", [f"current_{kind}", "axis_1"], [f"cache_{kind}"]),
                helper.make_node(
                    "Concat", [f"past_key_values.0.{kind}", f"cache_{kind}"], [f"present.0.{kind}"], axis=2
                ),
                helper.make_node("ReduceSum", [f"past_key_values.0.{kind}", "axis_2"], [f"past_{kind}_4d"], keepdims=1),
                helper.make_node("Squeeze", [f"past_{kind}_4d", "axis_1"], [f"past_{kind}_sum"]),
                helper.make_node("CumSum", [f"current_{kind}", "sequence_axis"], [f"current_{kind}_sum"]),
                helper.make_node("Add", [f"past_{kind}_sum", f"current_{kind}_sum"], [f"total_{kind}"]),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Add", ["total_key", "total_value"], ["history"]),
            helper.make_node("Cast", ["history"], ["integer_history"], to=T.INT64),
            helper.make_node("Mod", ["integer_history", "sixteen"], ["target_mod"]),
            helper.make_node("Cast", ["target_mod"], ["float_target"], to=T.FLOAT),
            helper.make_node("Add", ["float_target", "two"], ["target"]),
            helper.make_node("Sub", ["vocabulary", "target"], ["distance"]),
            helper.make_node("Mul", ["distance", "distance"], ["square_distance"]),
            helper.make_node("Neg", ["square_distance"], ["scores"]),
            # Preserve the unmodded history in logits, so modulo collisions cannot hide cache loss.
            helper.make_node("Mul", ["history", "history_scale"], ["history_bias"]),
            helper.make_node("Add", ["scores", "history_bias"], ["logits"]),
        ]
    )
    inputs = [
        _info("input_ids", T.INT32, ["batch", "seq"])
        if text_only
        else _info("inputs_embeds", T.FLOAT, ["batch", "seq", 1]),
        _info("position_ids", T.INT64, [3, "batch", "seq"] if qwen else ["batch", "seq"]),
        _info("attention_mask", T.INT64, ["batch", "total_seq"]),
        *[_info(f"past_key_values.0.{kind}", T.FLOAT, ["batch", 1, "past_seq", 1]) for kind in ("key", "value")],
    ]
    constants = [
        _tensor("axis_1", [1]),
        _tensor("axis_2", [2]),
        _tensor("sequence_axis", 1),
        _tensor("one", 1, np.float32),
        _tensor("two", 2, np.float32),
        _tensor("three", 3, np.float32),
        _tensor("sixteen", 16),
        _tensor("vocabulary", np.arange(VOCAB_SIZE), np.float32),
        _tensor("history_scale", 1 / 1024, np.float32),
    ]
    constants.extend(
        [_tensor("axis_0", [0]), _tensor("position_weights", [[[1]], [[2]], [[4]]], np.float32)]
        if qwen
        else [_tensor("seven", 7, np.float32)]
    )
    return _model(
        "decoder",
        nodes,
        inputs,
        [
            _info("logits", T.FLOAT, ["batch", "seq", VOCAB_SIZE]),
            *[_info(f"present.0.{kind}", T.FLOAT, ["batch", 1, "total_seq", 1]) for kind in ("key", "value")],
        ],
        constants,
    )


def create_model(output_dir: Path, family: str = "phi3v", *, fail_on_negative_pixels: bool = False) -> Path:
    """Write test-time ONNX graphs and configuration beneath a pytest tmp_path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "type": family,
        "bos_token_id": 1,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": 0,
        "image_token_id": IMAGE_TOKEN_ID,
        "video_token_id": 30,
        "vision_start_token_id": VISION_START_TOKEN_ID,
        "vocab_size": VOCAB_SIZE,
        "context_length": 256,
        "decoder": {
            "filename": "decoder.onnx",
            "hidden_size": 1,
            "head_size": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "inputs": {
                "input_ids": "input_ids",
                "inputs_embeds": "inputs_embeds",
                "position_ids": "position_ids",
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
    }
    graphs = {"decoder.onnx": make_decoder_model(family)}
    if family != "llama":
        model["embedding"] = {
            "filename": "embedding.onnx",
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }
        model["vision"] = {
            "filename": "vision.onnx",
            "patch_size": 1,
            "spatial_merge_size": 1,
            "inputs": {
                "pixel_values": "pixel_values",
                "image_sizes": "image_sizes",
                "image_grid_thw": "image_grid_thw",
            },
            "outputs": {"image_features": "image_features"},
        }
        graphs.update(
            {
                "embedding.onnx": make_embedding_model(family),
                "vision.onnx": make_vision_model(family, fail_on_negative_pixels=fail_on_negative_pixels),
            }
        )
    for filename, graph in graphs.items():
        onnx.save(graph, output_dir / filename)
    config = {
        "model": model,
        "search": {"max_length": 192, "do_sample": False, "past_present_share_buffer": False},
    }
    (output_dir / "genai_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_dir
