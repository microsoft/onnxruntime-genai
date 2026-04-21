# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Create a minimal 1-layer GQA decoder model for testing GenAI graph capture + RewindTo.

Uses GroupQueryAttention contrib op (what model builder produces for WebGPU).
All ops are WebGPU-EP compatible.
"""

import json
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def create_model(output_dir: str):
    vocab_size = 256
    hidden_size = 64
    num_heads = 2
    num_kv_heads = 2
    head_size = hidden_size // num_heads  # 32
    num_layers = 1

    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(42)
    initializers = []

    def make_weight(name, shape, dtype=np.float16):
        data = rng.randn(*shape).astype(dtype) * 0.02
        t = numpy_helper.from_array(data, name=name)
        initializers.append(t)
        return name

    # Inputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"])

    past_keys, past_values, present_keys, present_values = [], [], [], []
    for i in range(num_layers):
        past_keys.append(helper.make_tensor_value_info(
            f"past_key_values.{i}.key", TensorProto.FLOAT16,
            ["batch_size", num_kv_heads, "past_sequence_length", head_size]))
        past_values.append(helper.make_tensor_value_info(
            f"past_key_values.{i}.value", TensorProto.FLOAT16,
            ["batch_size", num_kv_heads, "past_sequence_length", head_size]))
        present_keys.append(helper.make_tensor_value_info(
            f"present.{i}.key", TensorProto.FLOAT16,
            ["batch_size", num_kv_heads, "total_sequence_length", head_size]))
        present_values.append(helper.make_tensor_value_info(
            f"present.{i}.value", TensorProto.FLOAT16,
            ["batch_size", num_kv_heads, "total_sequence_length", head_size]))

    logits_output = helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["batch_size", "sequence_length", vocab_size])

    # Weights
    embed_w = make_weight("embed.weight", [vocab_size, hidden_size])
    q_w = make_weight("attn.q.weight", [hidden_size, num_heads * head_size])
    k_w = make_weight("attn.k.weight", [hidden_size, num_kv_heads * head_size])
    v_w = make_weight("attn.v.weight", [hidden_size, num_kv_heads * head_size])
    o_w = make_weight("attn.o.weight", [num_heads * head_size, hidden_size])
    lm_head_w = make_weight("lm_head.weight", [hidden_size, vocab_size])

    nodes = []

    # 1. Embedding
    nodes.append(helper.make_node("Gather", [embed_w, "input_ids"], ["embed_out"], axis=0))

    # 2.5 Compute seqlens_k and total_sequence_length from attention_mask
    # seqlens_k = ReduceSum(attention_mask, axis=1) - 1  [shape: batch_size]
    # total_sequence_length = ReduceMax(ReduceSum(attention_mask, axis=1))  [shape: scalar]
    # First cast attention_mask from int64 to int32 for GQA compatibility
    nodes.append(helper.make_node("Cast", ["attention_mask"], ["mask_i32"], to=TensorProto.INT32))
    reduce_axis = numpy_helper.from_array(np.array([1], dtype=np.int64), "reduce_axis_1")
    initializers.append(reduce_axis)
    nodes.append(helper.make_node("ReduceSum", ["mask_i32", "reduce_axis_1"], ["sum_mask"], keepdims=0))
    one_const = numpy_helper.from_array(np.array([1], dtype=np.int32), "one_i32")
    initializers.append(one_const)
    nodes.append(helper.make_node("Sub", ["sum_mask", "one_i32"], ["seqlens_k"]))
    reduce_axis_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "reduce_axis_0")
    initializers.append(reduce_axis_0)
    nodes.append(helper.make_node("ReduceMax", ["sum_mask", "reduce_axis_0"], ["total_seq_len"], keepdims=0))

    # 3. Q, K, V projections
    nodes.append(helper.make_node("MatMul", ["embed_out", q_w], ["q_proj"]))
    nodes.append(helper.make_node("MatMul", ["embed_out", k_w], ["k_proj"]))
    nodes.append(helper.make_node("MatMul", ["embed_out", v_w], ["v_proj"]))

    # 3. GroupQueryAttention (com.microsoft domain)
    # GQA requires min 7 inputs. Uses separate Q, K, V (not fused QKV).
    # Input schema: query, key, value, past_key, past_value, seqlens_k, total_seq_len, [cos], [sin], ...
    nodes.append(helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "q_proj",                    # [0] query
            "k_proj",                    # [1] key
            "v_proj",                    # [2] value
            "past_key_values.0.key",     # [3] past_key
            "past_key_values.0.value",   # [4] past_value
            "seqlens_k",                 # [5] seqlens_k
            "total_seq_len",             # [6] total_sequence_length
        ],
        outputs=[
            "attn_out",
            "present.0.key",
            "present.0.value",
        ],
        name="gqa_0",
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=num_kv_heads,
    ))

    # 4. Output projection + residual
    nodes.append(helper.make_node("MatMul", ["attn_out", o_w], ["o_proj"]))
    nodes.append(helper.make_node("Add", ["embed_out", "o_proj"], ["residual"]))

    # 5. LM head
    nodes.append(helper.make_node("MatMul", ["residual", lm_head_w], ["logits"]))

    graph = helper.make_graph(
        nodes,
        "tiny_gqa_model",
        inputs=[input_ids, attention_mask] + past_keys + past_values,
        outputs=[logits_output] + present_keys + present_values,
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = 10

    model_path = os.path.join(output_dir, "model.onnx")
    onnx.save(model, model_path)
    print(f"Saved model to {model_path} ({os.path.getsize(model_path)} bytes)")

    # genai_config.json
    config = {
        "model": {
            "bos_token_id": 0,
            "context_length": 2048,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [
                        {
                            "webgpu": {
                                "enableGraphCapture": "1",
                                "validationMode": "disabled"
                            }
                        }
                    ]
                },
                "filename": "model.onnx",
                "head_size": head_size,
                "hidden_size": hidden_size,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value"
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value"
                },
                "num_attention_heads": num_heads,
                "num_hidden_layers": num_layers,
                "num_key_value_heads": num_kv_heads
            },
            "eos_token_id": [1],
            "pad_token_id": 0,
            "type": "phi3",
            "vocab_size": vocab_size
        },
        "search": {
            "do_sample": False,
            "max_length": 128,
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": True,
            "top_k": 1,
            "top_p": 1.0
        }
    }

    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Minimal tokenizer
    vocab = {chr(i) if 32 <= i < 127 else f"<{i}>": i for i in range(vocab_size)}
    with open(os.path.join(output_dir, "tokenizer.json"), "w") as f:
        json.dump({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
                {"id": 1, "content": "<eos>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            ]
        }, f)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"model_type": "phi3", "bos_token": "<pad>", "eos_token": "<eos>", "pad_token": "<pad>"}, f)

    print(f"Model ready: {output_dir}")
    print(f"  1 layer, hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_size={head_size}, vocab={vocab_size}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", default=os.path.join(".", "tiny-graph-capture"))
    create_model(p.parse_args().output)
