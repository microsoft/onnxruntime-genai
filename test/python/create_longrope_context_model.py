# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create a tiny observable LongRoPE cache fixture, without downloaded weights."""

import json
from pathlib import Path

from onnx import TensorProto, helper, save


def create_model(
    output_dir, declare_context=True, runtime_indices=False, record_chunk_sizes=False, reverse_logits=False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, ["batch", 1, "past", 1]),
        helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, ["batch", 1, "past", 1]),
    ]
    if declare_context:
        inputs.insert(3, helper.make_tensor_value_info("rope_context_length", TensorProto.INT64, []))
    if runtime_indices:
        inputs.append(helper.make_tensor_value_info("last_token_indices", TensorProto.INT32, ["batch", 1, 1]))
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1 if runtime_indices else "sequence", 3]),
        helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, ["batch", 1, "total", 1]),
        helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, ["batch", 1, "total", 1]),
    ]
    constants = [
        helper.make_tensor("boundary", TensorProto.INT64, [], [8]),
        helper.make_tensor("short_factor", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("long_factor", TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("cache_axes", TensorProto.INT64, [2], [1, 3]),
        helper.make_tensor("reduce_axes", TensorProto.INT64, [3], [1, 2, 3]),
        helper.make_tensor("logits_axis", TensorProto.INT64, [1], [2]),
        helper.make_tensor("sequence_axis", TensorProto.INT64, [], [1]),
    ]
    nodes = (
        []
        if declare_context
        else [
            helper.make_node("Shape", ["attention_mask"], ["attention_shape"]),
            helper.make_node("Gather", ["attention_shape", "sequence_axis"], ["rope_context_length"], axis=0),
        ]
    )
    if record_chunk_sizes:
        nodes += [
            helper.make_node("Shape", ["input_ids"], ["token_shape"]),
            helper.make_node("Gather", ["token_shape", "sequence_axis"], ["chunk_size"], axis=0),
            helper.make_node("Cast", ["chunk_size"], ["chunk_size_float"], to=TensorProto.FLOAT),
            helper.make_node("Expand", ["chunk_size_float", "token_shape"], ["token_chunk_sizes"]),
            helper.make_node("Unsqueeze", ["token_chunk_sizes", "cache_axes"], ["new_chunk_sizes"]),
        ]
    nodes += [
        helper.make_node("Greater", ["rope_context_length", "boundary"], ["long_mode"]),
        helper.make_node("Where", ["long_mode", "long_factor", "short_factor"], ["factor"]),
        helper.make_node("Cast", ["input_ids"], ["tokens"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["tokens", "factor"], ["scaled_tokens"]),
        helper.make_node("Unsqueeze", ["scaled_tokens", "cache_axes"], ["new_keys"]),
        helper.make_node("Concat", ["past_key_values.0.key", "new_keys"], ["present.0.key"], axis=2),
        helper.make_node(
            "Concat",
            ["past_key_values.0.value", "new_chunk_sizes" if record_chunk_sizes else "new_keys"],
            ["present.0.value"],
            axis=2,
        ),
        helper.make_node("ReduceSum", ["past_key_values.0.key", "reduce_axes"], ["past_sum"], keepdims=1),
        helper.make_node("Squeeze", ["past_sum", "cache_axes"], ["past_sum_column"]),
        helper.make_node("CumSum", ["scaled_tokens", "sequence_axis"], ["token_sums"]),
        helper.make_node("Add", ["past_sum_column", "token_sums"], ["sum"]),
        helper.make_node("Neg", ["sum"], ["negative"]),
        helper.make_node("Mul", ["sum", "zero"], ["zeros"]),
        helper.make_node("Unsqueeze", ["sum", "logits_axis"], ["positive_logit"]),
        helper.make_node("Unsqueeze", ["negative", "logits_axis"], ["negative_logit"]),
        helper.make_node("Unsqueeze", ["zeros", "logits_axis"], ["zero_logit"]),
        helper.make_node(
            "Concat",
            ["negative_logit", "positive_logit", "zero_logit"]
            if reverse_logits
            else ["positive_logit", "negative_logit", "zero_logit"],
            ["all_logits" if runtime_indices else "logits"],
            axis=2,
        ),
    ]
    if runtime_indices:
        constants.append(helper.make_tensor("indices_shape", TensorProto.INT64, [3], [1, 1, 3]))
        nodes += [
            helper.make_node("Expand", ["last_token_indices", "indices_shape"], ["expanded_indices"]),
            helper.make_node("GatherElements", ["all_logits", "expanded_indices"], ["logits"], axis=1),
        ]
    graph = helper.make_graph(nodes, "longrope-context", inputs, outputs, initializer=constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    if declare_context:
        helper.set_model_props(
            model,
            {
                "com.microsoft.genai.rope_context_length_input": "rope_context_length",
                "com.microsoft.genai.original_context_length": "8",
            },
        )
    save(model, output_dir / "model.onnx")
    config = {
        "model": {
            "type": "llama",
            "context_length": 32,
            "vocab_size": 3,
            "pad_token_id": 2,
            "eos_token_id": 2,
            "decoder": {
                "filename": "model.onnx",
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 1,
                "hidden_size": 1,
                "session_options": {"intra_op_num_threads": 1},
            },
        },
        "search": {"max_length": 32, "past_present_share_buffer": False, "do_sample": False},
    }
    (output_dir / "genai_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    create_model(Path(__file__).parents[1] / "models" / "longrope-context")
    create_model(Path(__file__).parents[1] / "models" / "longrope-context-without-metadata", declare_context=False)
    create_model(Path(__file__).parents[1] / "models" / "longrope-context-runtime-indices", runtime_indices=True)
    create_model(Path(__file__).parents[1] / "models" / "longrope-context-chunk-sizes", record_chunk_sizes=True)
    create_model(
        Path(__file__).parents[1] / "models" / "longrope-context-runtime-indices-chunk-sizes",
        runtime_indices=True,
        record_chunk_sizes=True,
    )
    create_model(
        Path(__file__).parents[1] / "models" / "longrope-context-legacy",
        declare_context=False,
        reverse_logits=True,
    )
    create_model(
        Path(__file__).parents[1] / "models" / "longrope-context-legacy-runtime-indices",
        declare_context=False,
        runtime_indices=True,
        reverse_logits=True,
    )
