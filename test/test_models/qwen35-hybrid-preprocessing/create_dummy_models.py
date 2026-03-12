#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Generate dummy ONNX models for Qwen3.5 hybrid model testing.

Creates minimal ONNX models (decoder, embedding, vision) with the correct
input/output signatures for a hybrid model with both KV cache and recurrent
state tensors. These models produce dummy outputs but have the correct
shapes for testing the ort-genai runtime's auto-discovery and state management.

Usage:
    python create_qwen35_dummy_models.py --output test/test_models/qwen35-hybrid-preprocessing
"""

import argparse
import json
import os
import shutil

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    print("onnx package required: pip install onnx")
    exit(1)


def create_dummy_embedding_model(output_path: str, hidden_size: int = 1024, vocab_size: int = 248320):
    """Create dummy embedding model: input_ids, image_features -> inputs_embeds"""
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence_len"])
    image_features = helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_image_tokens", hidden_size])
    inputs_embeds = helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch", "sequence_len", hidden_size])

    # Create a simple graph that outputs zeros of the right shape
    # Shape -> ConstantOfShape to produce zeros
    shape_node = helper.make_node("Shape", ["input_ids"], ["ids_shape"])
    # We need [batch, seq_len, hidden_size] output
    hidden_const = helper.make_node("Constant", [], ["hidden_dim"],
                                     value=helper.make_tensor("hidden_dim", TensorProto.INT64, [1], [hidden_size]))
    concat_node = helper.make_node("Concat", ["ids_shape", "hidden_dim"], ["embed_shape"], axis=0)
    zero_val = helper.make_node("Constant", [], ["zero_val"],
                                 value=helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0]))
    cos_node = helper.make_node("ConstantOfShape", ["embed_shape"], ["inputs_embeds"],
                                 value=helper.make_tensor("val", TensorProto.FLOAT, [1], [0.01]))

    graph = helper.make_graph(
        [shape_node, hidden_const, concat_node, zero_val, cos_node],
        "embedding",
        [input_ids, image_features],
        [inputs_embeds],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def create_dummy_vision_model(output_path: str, hidden_size: int = 1024):
    """Create dummy vision model: pixel_values, image_grid_thw -> image_features"""
    pixel_values = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["total_patches", 1536])
    image_grid_thw = helper.make_tensor_value_info("image_grid_thw", TensorProto.INT64, ["num_images", 3])
    image_features = helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [None, hidden_size])

    # Simple: output zeros with shape [total_patches/4, hidden_size]
    # (spatial_merge_size=2 -> merge_sq=4)
    shape_node = helper.make_node("Shape", ["pixel_values"], ["pv_shape"])
    gather_node = helper.make_node("Gather", ["pv_shape", "zero_idx"], ["num_patches"], axis=0)
    zero_idx_const = helper.make_node("Constant", [], ["zero_idx"],
                                       value=helper.make_tensor("zero_idx", TensorProto.INT64, [], [0]))
    four_const = helper.make_node("Constant", [], ["four"],
                                    value=helper.make_tensor("four", TensorProto.INT64, [], [4]))
    div_node = helper.make_node("Div", ["num_patches", "four"], ["num_feats"])
    hidden_const = helper.make_node("Constant", [], ["hidden_dim"],
                                     value=helper.make_tensor("hidden_dim", TensorProto.INT64, [1], [hidden_size]))
    reshape_feats = helper.make_node("Reshape", ["num_feats", "one_shape"], ["num_feats_1d"])
    one_shape_const = helper.make_node("Constant", [], ["one_shape"],
                                        value=helper.make_tensor("one_shape", TensorProto.INT64, [1], [1]))
    concat_node = helper.make_node("Concat", ["num_feats_1d", "hidden_dim"], ["feat_shape"], axis=0)
    cos_node = helper.make_node("ConstantOfShape", ["feat_shape"], ["image_features"],
                                 value=helper.make_tensor("val", TensorProto.FLOAT, [1], [0.01]))

    graph = helper.make_graph(
        [zero_idx_const, shape_node, gather_node, four_const, div_node,
         hidden_const, one_shape_const, reshape_feats, concat_node, cos_node],
        "vision",
        [pixel_values, image_grid_thw],
        [image_features],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def create_dummy_decoder_model(
    output_path: str,
    num_layers: int = 4,
    kv_layers: list = None,
    hidden_size: int = 1024,
    num_kv_heads: int = 2,
    head_size: int = 256,
    num_linear_heads: int = 16,
    linear_head_dim: int = 128,
    conv_dim: int = 6144,
    conv_kernel: int = 3,
    vocab_size: int = 248320,
):
    """
    Create dummy decoder model with hybrid KV cache + recurrent state inputs.

    For layers in kv_layers: KV cache inputs (past_key_values.%d.key/value)
    For other layers: recurrent state inputs (past_key_values.%d.conv_state/recurrent_state)
    """
    if kv_layers is None:
        kv_layers = [num_layers - 1]  # Last layer is full attention by default

    inputs = []
    outputs = []

    # Standard inputs
    inputs_embeds = helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch", "sequence_len", hidden_size])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "past_seq_len_plus_seq_len"])
    position_ids = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [3, "batch", "sequence_len"])
    inputs.extend([inputs_embeds, attention_mask, position_ids])

    # Per-layer inputs/outputs
    for layer_idx in range(num_layers):
        if layer_idx in kv_layers:
            # KV cache layer
            inputs.append(helper.make_tensor_value_info(
                f"past_key_values.{layer_idx}.key", TensorProto.FLOAT,
                ["batch", num_kv_heads, "past_sequence_len", head_size]))
            inputs.append(helper.make_tensor_value_info(
                f"past_key_values.{layer_idx}.value", TensorProto.FLOAT,
                ["batch", num_kv_heads, "past_sequence_len", head_size]))
            outputs.append(helper.make_tensor_value_info(
                f"present.{layer_idx}.key", TensorProto.FLOAT,
                ["batch", num_kv_heads, "total_sequence_len", head_size]))
            outputs.append(helper.make_tensor_value_info(
                f"present.{layer_idx}.value", TensorProto.FLOAT,
                ["batch", num_kv_heads, "total_sequence_len", head_size]))
        else:
            # Recurrent state layer
            inputs.append(helper.make_tensor_value_info(
                f"past_key_values.{layer_idx}.conv_state", TensorProto.FLOAT,
                ["batch", conv_dim, conv_kernel - 1]))
            inputs.append(helper.make_tensor_value_info(
                f"past_key_values.{layer_idx}.recurrent_state", TensorProto.FLOAT,
                ["batch", num_linear_heads, linear_head_dim, linear_head_dim]))
            outputs.append(helper.make_tensor_value_info(
                f"present.{layer_idx}.conv_state", TensorProto.FLOAT,
                ["batch", conv_dim, conv_kernel - 1]))
            outputs.append(helper.make_tensor_value_info(
                f"present.{layer_idx}.recurrent_state", TensorProto.FLOAT,
                ["batch", num_linear_heads, linear_head_dim, linear_head_dim]))

    # Logits output
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None, vocab_size])
    outputs.insert(0, logits)

    # Create a minimal graph: Identity pass-through for state tensors, zeros for logits
    nodes = []

    # Logits: zeros from inputs_embeds shape
    shape_node = helper.make_node("Shape", ["inputs_embeds"], ["embed_shape"])
    nodes.append(shape_node)

    gather_batch = helper.make_node("Gather", ["embed_shape", "idx_0"], ["batch_dim"], axis=0)
    gather_seq = helper.make_node("Gather", ["embed_shape", "idx_1"], ["seq_dim"], axis=0)
    idx_0_const = helper.make_node("Constant", [], ["idx_0"],
                                    value=helper.make_tensor("idx_0", TensorProto.INT64, [], [0]))
    idx_1_const = helper.make_node("Constant", [], ["idx_1"],
                                    value=helper.make_tensor("idx_1", TensorProto.INT64, [], [1]))
    vocab_const = helper.make_node("Constant", [], ["vocab_dim"],
                                    value=helper.make_tensor("vocab_dim", TensorProto.INT64, [1], [vocab_size]))
    nodes.extend([idx_0_const, idx_1_const, gather_batch, gather_seq, vocab_const])

    reshape_batch = helper.make_node("Reshape", ["batch_dim", "one_shape"], ["batch_1d"])
    reshape_seq = helper.make_node("Reshape", ["seq_dim", "one_shape"], ["seq_1d"])
    one_shape_const = helper.make_node("Constant", [], ["one_shape"],
                                        value=helper.make_tensor("one_shape", TensorProto.INT64, [1], [1]))
    concat_logits_shape = helper.make_node("Concat", ["batch_1d", "seq_1d", "vocab_dim"], ["logits_shape"], axis=0)
    logits_node = helper.make_node("ConstantOfShape", ["logits_shape"], ["logits"],
                                    value=helper.make_tensor("val", TensorProto.FLOAT, [1], [0.0]))
    nodes.extend([one_shape_const, reshape_batch, reshape_seq, concat_logits_shape, logits_node])

    # Identity for all state tensors
    for layer_idx in range(num_layers):
        if layer_idx in kv_layers:
            nodes.append(helper.make_node("Identity",
                [f"past_key_values.{layer_idx}.key"], [f"present.{layer_idx}.key"]))
            nodes.append(helper.make_node("Identity",
                [f"past_key_values.{layer_idx}.value"], [f"present.{layer_idx}.value"]))
        else:
            nodes.append(helper.make_node("Identity",
                [f"past_key_values.{layer_idx}.conv_state"], [f"present.{layer_idx}.conv_state"]))
            nodes.append(helper.make_node("Identity",
                [f"past_key_values.{layer_idx}.recurrent_state"], [f"present.{layer_idx}.recurrent_state"]))

    graph = helper.make_graph(nodes, "decoder", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def create_genai_config(output_path: str, num_kv_layers: int, kv_layers: list):
    """Create genai_config.json for the dummy hybrid model."""
    config = {
        "model": {
            "bos_token_id": 151643,
            "context_length": 4096,
            "decoder": {
                "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
                "filename": "dummy_text.onnx",
                "head_size": 256,
                "hidden_size": 1024,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "num_attention_heads": 8,
                "num_hidden_layers": num_kv_layers,
                "num_key_value_heads": 2,
            },
            "embedding": {
                "filename": "dummy_embedding.onnx",
                "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
            },
            "vision": {
                "filename": "dummy_vision.onnx",
                "spatial_merge_size": 2,
                "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
                "outputs": {"image_features": "image_features"},
            },
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "image_token_id": 151655,
            "vision_start_token_id": 151652,
            "type": "qwen3_5",
            "vocab_size": 248320,
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


def main():
    parser = argparse.ArgumentParser(description="Generate dummy ONNX models for Qwen3.5 hybrid model testing")
    parser.add_argument("--output", type=str, default="test/test_models/qwen35-hybrid-preprocessing",
                        help="Output directory for the dummy models")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Configuration: 4 layers, layer 3 is full attention, layers 0,1,2 are recurrent
    num_layers = 4
    kv_layers = [3]  # Only last layer has KV cache
    num_kv_layers = len(kv_layers)

    print(f"Creating dummy hybrid model in {output_dir}")
    print(f"  {num_layers} total layers, KV at {kv_layers}, recurrent at {[i for i in range(num_layers) if i not in kv_layers]}")

    create_dummy_embedding_model(os.path.join(output_dir, "dummy_embedding.onnx"))
    print("  Created dummy_embedding.onnx")

    create_dummy_vision_model(os.path.join(output_dir, "dummy_vision.onnx"))
    print("  Created dummy_vision.onnx")

    create_dummy_decoder_model(
        os.path.join(output_dir, "dummy_text.onnx"),
        num_layers=num_layers,
        kv_layers=kv_layers,
    )
    print("  Created dummy_text.onnx")

    create_genai_config(os.path.join(output_dir, "genai_config.json"), num_kv_layers, kv_layers)
    print("  Created genai_config.json")

    # Copy tokenizer files from qwen3-vl test model if available
    src_dir = os.path.join(os.path.dirname(output_dir), "qwen3-vl-vision-preprocessing")
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "processor_config.json"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")

    print("Done!")


if __name__ == "__main__":
    main()
