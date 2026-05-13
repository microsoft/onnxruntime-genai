#!/usr/bin/env python3
"""Create a tiny model with GQA for testing CUDA graph capture.

Generates a qwen2-type LLM (decoder-only) that supports
past_present_share_buffer and CUDA graph capture.

The decoder model is run through ORT's transformer optimizer to ensure proper
CUDA EP partitioning for graph capture compatibility.

Usage:
  python create_gqa_model.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.transformers.optimizer import optimize_model

# Tiny model config
VOCAB_SIZE = 1000
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = HIDDEN_SIZE // NUM_HEADS  # 16
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 128
MAX_SEQ_LEN = 128


def rand_init(name, shape, dtype=np.float16, scale=0.02):
    np.random.seed(hash(name) % (2**31))
    return numpy_helper.from_array((np.random.randn(*shape) * scale).astype(dtype), name=name)


def zeros_init(name, shape, dtype=np.float16):
    return numpy_helper.from_array(np.zeros(shape, dtype=dtype), name=name)


def create_decoder(output_dir):
    """Create and optimize a GQA decoder model with input_ids."""
    onnx_dtype = TensorProto.FLOAT16
    ms = "com.microsoft"
    nodes, inits = [], []

    # Embedding table
    inits.append(rand_init("embed_tokens.weight", [VOCAB_SIZE, HIDDEN_SIZE]))

    graph_inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "seq_len"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_seq_len"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "seq_len"]),
    ]
    graph_outputs = []

    for i in range(NUM_LAYERS):
        graph_inputs.append(helper.make_tensor_value_info(
            f"past_key_values.{i}.key", onnx_dtype, ["batch_size", NUM_KV_HEADS, "past_seq_len", HEAD_SIZE]))
        graph_inputs.append(helper.make_tensor_value_info(
            f"past_key_values.{i}.value", onnx_dtype, ["batch_size", NUM_KV_HEADS, "past_seq_len", HEAD_SIZE]))

    # Derive seqlens_k and total_sequence_length from attention_mask
    # These are small CPU-side computations that ORT handles outside graph capture
    nodes.append(helper.make_node("Cast", ["attention_mask"], ["attn_mask_i32"], to=TensorProto.INT32))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "reduce_axes"))
    nodes.append(helper.make_node("ReduceSum", ["attn_mask_i32", "reduce_axes"], ["seqlens_k_p1"],
                                  keepdims=0, noop_with_empty_axes=0))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int32), "const_one"))
    nodes.append(helper.make_node("Sub", ["seqlens_k_p1", "const_one"], ["seqlens_k"]))
    nodes.append(helper.make_node("Shape", ["attention_mask"], ["mask_shape"]))
    inits.append(numpy_helper.from_array(np.array(1, dtype=np.int64), "gather_idx"))
    nodes.append(helper.make_node("Gather", ["mask_shape", "gather_idx"], ["total_sl_i64"], axis=0))
    nodes.append(helper.make_node("Cast", ["total_sl_i64"], ["total_sl"], to=TensorProto.INT32))

    # Embedding lookup: input_ids -> inputs_embeds
    nodes.append(helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["inputs_embeds"], axis=0))

    prev = "inputs_embeds"

    for li in range(NUM_LAYERS):
        p = f"l{li}"

        # RMSNorm (using SimplifiedLayerNormalization which is in com.microsoft)
        # Use standard LayerNormalization instead for compatibility
        inits.append(rand_init(f"{p}.ln1.w", [HIDDEN_SIZE]))
        inits.append(zeros_init(f"{p}.ln1.b", [HIDDEN_SIZE]))
        nodes.append(helper.make_node("LayerNormalization",
                                      [prev, f"{p}.ln1.w", f"{p}.ln1.b"], [f"{p}.ln1"],
                                      epsilon=1e-5, axis=-1))

        # Q, K, V projections (separate, not packed)
        inits.append(rand_init(f"{p}.q.w", [HIDDEN_SIZE, NUM_HEADS * HEAD_SIZE]))
        inits.append(rand_init(f"{p}.k.w", [HIDDEN_SIZE, NUM_KV_HEADS * HEAD_SIZE]))
        inits.append(rand_init(f"{p}.v.w", [HIDDEN_SIZE, NUM_KV_HEADS * HEAD_SIZE]))
        nodes.append(helper.make_node("MatMul", [f"{p}.ln1", f"{p}.q.w"], [f"{p}.q"]))
        nodes.append(helper.make_node("MatMul", [f"{p}.ln1", f"{p}.k.w"], [f"{p}.k"]))
        nodes.append(helper.make_node("MatMul", [f"{p}.ln1", f"{p}.v.w"], [f"{p}.v"]))

        # GQA with separate Q, K, V
        nodes.append(helper.make_node("GroupQueryAttention",
            [f"{p}.q", f"{p}.k", f"{p}.v",
             f"past_key_values.{li}.key", f"past_key_values.{li}.value",
             "seqlens_k", "total_sl",
             "", "",  # cos/sin cache
             ],
            [f"{p}.attn", f"present.{li}.key", f"present.{li}.value"],
            domain=ms, name=f"GQA_{li}",
            num_heads=NUM_HEADS, kv_num_heads=NUM_KV_HEADS))

        # Output projection + residual
        inits.append(rand_init(f"{p}.o.w", [NUM_HEADS * HEAD_SIZE, HIDDEN_SIZE]))
        nodes.append(helper.make_node("MatMul", [f"{p}.attn", f"{p}.o.w"], [f"{p}.o"]))
        nodes.append(helper.make_node("Add", [prev, f"{p}.o"], [f"{p}.r1"]))

        # Post-attn norm
        inits.append(rand_init(f"{p}.ln2.w", [HIDDEN_SIZE]))
        inits.append(zeros_init(f"{p}.ln2.b", [HIDDEN_SIZE]))
        nodes.append(helper.make_node("LayerNormalization",
                                      [f"{p}.r1", f"{p}.ln2.w", f"{p}.ln2.b"], [f"{p}.ln2"],
                                      epsilon=1e-5, axis=-1))

        # MLP
        inits.append(rand_init(f"{p}.up.w", [HIDDEN_SIZE, INTERMEDIATE_SIZE]))
        inits.append(rand_init(f"{p}.dn.w", [INTERMEDIATE_SIZE, HIDDEN_SIZE]))
        nodes.append(helper.make_node("MatMul", [f"{p}.ln2", f"{p}.up.w"], [f"{p}.up"]))
        nodes.append(helper.make_node("Relu", [f"{p}.up"], [f"{p}.act"]))
        nodes.append(helper.make_node("MatMul", [f"{p}.act", f"{p}.dn.w"], [f"{p}.dn"]))

        prev = f"{p}.r2"
        nodes.append(helper.make_node("Add", [f"{p}.r1", f"{p}.dn"], [prev]))

        for kv in ("key", "value"):
            graph_outputs.append(helper.make_tensor_value_info(
                f"present.{li}.{kv}", onnx_dtype,
                ["batch_size", NUM_KV_HEADS, "total_seq_len", HEAD_SIZE]))

    # Final norm + LM head
    inits.append(rand_init("norm.w", [HIDDEN_SIZE]))
    inits.append(zeros_init("norm.b", [HIDDEN_SIZE]))
    nodes.append(helper.make_node("LayerNormalization",
                                  [prev, "norm.w", "norm.b"], ["final_ln"],
                                  epsilon=1e-5, axis=-1))
    inits.append(rand_init("lm_head.w", [HIDDEN_SIZE, VOCAB_SIZE]))
    nodes.append(helper.make_node("MatMul", ["final_ln", "lm_head.w"], ["logits"]))
    graph_outputs.insert(0, helper.make_tensor_value_info(
        "logits", onnx_dtype, ["batch_size", "seq_len", VOCAB_SIZE]))

    graph = helper.make_graph(nodes, "decoder", graph_inputs, graph_outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid(ms, 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    # Save unoptimized first
    raw_path = os.path.join(output_dir, "text_raw.onnx")
    onnx.save(model, raw_path)

    # Optimize with ORT transformer optimizer for proper CUDA EP placement
    opt = optimize_model(raw_path, model_type="gpt2", opt_level=0,
                         num_heads=NUM_HEADS, hidden_size=HIDDEN_SIZE)
    final_path = os.path.join(output_dir, "decoder.onnx")
    opt.save_model_to_file(final_path)
    os.remove(raw_path)
    print(f"  Saved decoder -> {final_path}")


def create_config(output_dir):
    config = {
        "model": {
            "type": "qwen2",
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE, "context_length": MAX_SEQ_LEN,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [{"cuda": {"enable_cuda_graph": "1"}}]
                },
                "filename": "decoder.onnx",
                "num_attention_heads": NUM_HEADS, "num_key_value_heads": NUM_KV_HEADS,
                "head_size": HEAD_SIZE, "hidden_size": HIDDEN_SIZE,
                "num_hidden_layers": NUM_LAYERS,
                "inputs": {
                    "input_ids": "input_ids", "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value"
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value"
                }
            }
        },
        "search": {"past_present_share_buffer": True}
    }
    path = os.path.join(output_dir, "genai_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Saved config -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__),
                                             "test_models", "hf-internal-testing", "tiny-qwen35-cuda"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Creating tiny GQA test model in {args.output_dir}")
    create_decoder(args.output_dir)
    create_config(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
