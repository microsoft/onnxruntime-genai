"""
Create a tiny functional ONNX decoder model for LFM2 testing.

LFM2 is a hybrid model with interleaved conv and attention layers.
This creates a model with 4 layers: [conv, attention, conv, attention].

The model is functional (not just I/O stubs): it uses a simple MatMul
to project input_ids embeddings to logits, and passes through KV cache
and conv state. This allows end-to-end generation testing.
"""

import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    # Model parameters
    num_heads = 4
    head_size = 8
    hidden_size = 32
    vocab_size = 1000
    conv_cache_size = 4

    # --- Inputs ---
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"])
    position_ids = helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "sequence_length"])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"])

    # KV cache inputs (attention layers 1, 3)
    past_kv_inputs = []
    for layer_idx in [1, 3]:
        past_kv_inputs.append(helper.make_tensor_value_info(
            "past_key_values.%d.key" % layer_idx, TensorProto.FLOAT,
            ["batch_size", num_heads, "past_sequence_length", head_size]))
        past_kv_inputs.append(helper.make_tensor_value_info(
            "past_key_values.%d.value" % layer_idx, TensorProto.FLOAT,
            ["batch_size", num_heads, "past_sequence_length", head_size]))

    # Conv state inputs (conv layers 0, 2)
    conv_inputs = []
    for layer_idx in [0, 2]:
        conv_inputs.append(helper.make_tensor_value_info(
            "past_conv.%d" % layer_idx, TensorProto.FLOAT,
            ["batch_size", hidden_size, conv_cache_size]))

    all_inputs = [input_ids, position_ids, attention_mask] + past_kv_inputs + conv_inputs

    # --- Outputs ---
    logits_output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", vocab_size])

    # KV cache outputs (attention layers 1, 3)
    present_kv_outputs = []
    for layer_idx in [1, 3]:
        present_kv_outputs.append(helper.make_tensor_value_info(
            "present.%d.key" % layer_idx, TensorProto.FLOAT,
            ["batch_size", num_heads, "total_sequence_length", head_size]))
        present_kv_outputs.append(helper.make_tensor_value_info(
            "present.%d.value" % layer_idx, TensorProto.FLOAT,
            ["batch_size", num_heads, "total_sequence_length", head_size]))

    # Conv state outputs (conv layers 0, 2)
    conv_outputs = []
    for layer_idx in [0, 2]:
        conv_outputs.append(helper.make_tensor_value_info(
            "present_conv.%d" % layer_idx, TensorProto.FLOAT,
            ["batch_size", hidden_size, conv_cache_size]))

    all_outputs = [logits_output] + present_kv_outputs + conv_outputs

    # --- Graph nodes ---
    nodes = []
    initializers = []

    # Embedding table: [vocab_size, hidden_size]
    np.random.seed(42)
    embed_weight = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    embed_init = numpy_helper.from_array(embed_weight, name="embed_weight")
    initializers.append(embed_init)

    # Gather to look up embeddings: [batch_size, sequence_length, hidden_size]
    nodes.append(helper.make_node("Gather", inputs=["embed_weight", "input_ids"], outputs=["embeddings"], axis=0))

    # Linear projection to logits: [batch_size, sequence_length, vocab_size]
    lm_head_weight = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02
    lm_head_init = numpy_helper.from_array(lm_head_weight, name="lm_head_weight")
    initializers.append(lm_head_init)

    nodes.append(helper.make_node("MatMul", inputs=["embeddings", "lm_head_weight"], outputs=["logits"]))

    # KV cache: concatenate past with new dummy values along sequence dimension
    # For each attention layer, create zero-filled new KV entries and concat with past
    for layer_idx in [1, 3]:
        for kv_type in ["key", "value"]:
            past_name = "past_key_values.%d.%s" % (layer_idx, kv_type)
            present_name = "present.%d.%s" % (layer_idx, kv_type)

            # Reshape embeddings [B, S, H] -> [B, S, num_heads, head_size]
            reshape_4d_name = "reshape_4d_%d_%s" % (layer_idx, kv_type)
            shape_4d = numpy_helper.from_array(
                np.array([0, 0, num_heads, head_size], dtype=np.int64),
                name="shape_4d_%d_%s" % (layer_idx, kv_type))
            initializers.append(shape_4d)
            nodes.append(helper.make_node("Reshape", inputs=["embeddings", shape_4d.name],
                                          outputs=[reshape_4d_name], allowzero=0))

            # Transpose [B, S, heads, head_size] -> [B, heads, S, head_size]
            transpose_name = "transpose_%d_%s" % (layer_idx, kv_type)
            nodes.append(helper.make_node("Transpose", inputs=[reshape_4d_name],
                                          outputs=[transpose_name], perm=[0, 2, 1, 3]))

            # Concat past and new along sequence dimension (axis=2)
            nodes.append(helper.make_node("Concat", inputs=[past_name, transpose_name],
                                          outputs=[present_name], axis=2))

    # Conv state: just pass through (identity)
    for layer_idx in [0, 2]:
        nodes.append(helper.make_node("Identity",
                                      inputs=["past_conv.%d" % layer_idx],
                                      outputs=["present_conv.%d" % layer_idx]))

    # --- Build model ---
    graph = helper.make_graph(
        name="main_graph",
        inputs=all_inputs,
        outputs=all_outputs,
        initializer=initializers,
        nodes=nodes,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
        producer_name="onnxruntime-genai-test",
        producer_version="0.0.0",
    )

    onnx.checker.check_model(model)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoder.onnx")
    onnx.save_model(model, output_path)
    print("Created %s" % output_path)


if __name__ == "__main__":
    main()
