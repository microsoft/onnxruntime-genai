# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Real conditional/cache fixtures for the optional decoder cache-phase input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from create_cache_dispatch_models import create_model
from create_multimodal_cache_model import create_auxiliary


def value(name, dtype, shape):
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def create(directory: Path, *, multimodal=False, alias="has_previous_state", rank=0, embedding_failure=False, bf16=False):
    create_model(
        directory, recurrent=True, model_type="nemotron_h_omni" if multimodal else "decoder", multimodal=multimodal,
        bf16=bf16,
    )
    if multimodal:
        create_auxiliary(directory, vision=True, retention=True)
        create_auxiliary(directory, vision=False)
        embedding = ir.load(directory / "embedding.onnx")
        features = next(x for x in embedding.graph.inputs if x.name == "image_features")
        features.dtype = ir.DataType.BFLOAT16
        converted = value("float_features", ir.DataType.FLOAT, ["num_image_tokens", 2])
        original_nodes = list(embedding.graph)
        for node in original_nodes:
            for i, inp in enumerate(node.inputs):
                if inp is features:
                    node.replace_input_with(i, converted)
        embedding.graph.insert_before(
            original_nodes[0], ir.Node("", "Cast", [features], [ir.AttrInt64("to", 1)], outputs=[converted])
        )
        if embedding_failure:
            graph = embedding.graph

            def constant(name, data, dtype=np.int64):
                tensor = ir.tensor(np.asarray(data, dtype=dtype), name=name)
                result = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=tensor.shape, const_value=tensor)
                graph.initializers[name] = result
                return result

            ids = next(x for x in graph.inputs if x.name == "input_ids")
            matches = value("failure_matches", ir.DataType.BOOL, ["batch_size", "sequence_length"])
            flags = value("failure_flags", ir.DataType.INT64, ["batch_size", "sequence_length"])
            found = value("failure_found", ir.DataType.INT64, [])
            fail = value("embedding_fail", ir.DataType.BOOL, [])
            shape = value("embedding_check_shape", ir.DataType.INT64, [1])
            checked = value("embedding_check", ir.DataType.FLOAT, [1])
            graph.append(ir.Node("", "Equal", [ids, constant("failure_token", 13, np.int32)], outputs=[matches]))
            graph.append(ir.Node("", "Cast", [matches], [ir.AttrInt64("to", 7)], outputs=[flags]))
            graph.append(ir.Node("", "ReduceMax", [flags], [ir.AttrInt64("keepdims", 0)], outputs=[found]))
            graph.append(ir.Node("", "Equal", [found, constant("failure_one", 1)], outputs=[fail]))
            graph.append(
                ir.Node("", "Where", [fail, constant("bad_shape", [-2]), constant("good_shape", [1])], outputs=[shape])
            )
            graph.append(
                ir.Node(
                    "",
                    "Reshape",
                    [constant("embedding_zero", [0], np.float32), shape],
                    outputs=[checked],
                    name="embedding_failure_reshape",
                )
            )
            original_output = graph.outputs[0]
            result = value(original_output.name, original_output.dtype, list(original_output.shape))
            original_output.name = "unchecked_inputs_embeds"
            graph.append(ir.Node("", "Add", [original_output, checked], outputs=[result]))
            graph.outputs[0] = result
        if bf16:
            # Match the public text pipeline: INT32 host IDs -> INT64 embedding IDs,
            # with BF16 features and embedding storage remaining inside C++.
            embedding.graph.inputs[0].dtype = ir.DataType.INT64
            for name in ("image_token", "failure_token"):
                if name in embedding.graph.initializers:
                    token = embedding.graph.initializers[name]
                    token.dtype = ir.DataType.INT64
                    token.const_value = ir.tensor(np.asarray(token.const_value.numpy(), dtype=np.int64))
            output = embedding.graph.outputs[0]
            output.name = "float_inputs_embeds"
            reduced = value("inputs_embeds", ir.DataType.BFLOAT16, list(output.shape))
            embedding.graph.append(
                ir.Node("", "Cast", [output], [ir.AttrInt64("to", 16)], outputs=[reduced])
            )
            embedding.graph.outputs[0] = reduced
        ir.save(embedding, directory / "embedding.onnx")
        onnx.checker.check_model(directory / "embedding.onnx", full_check=True)
    model = ir.load(directory / "decoder.onnx")
    graph = model.graph
    phase = value(alias, ir.DataType.BOOL, [] if rank == 0 else [1])
    graph.inputs.append(phase)
    original_nodes = list(graph)
    for past in list(graph.inputs):
        if not past.name.startswith("past_key_values."):
            continue
        shape = value(f"{past.name}_shape", ir.DataType.INT64, [len(past.shape)])
        selected = value(f"{past.name}_selected", past.dtype, list(past.shape))
        then_out = value(f"{past.name}_then", past.dtype, list(past.shape))
        else_out = value(f"{past.name}_else", past.dtype, list(past.shape))
        then = ir.Graph([], [then_out], nodes=[ir.Node("", "Identity", [past], outputs=[then_out])], name="cached")
        zero = value(f"{past.name}_zero", ir.DataType.FLOAT, list(past.shape))
        fresh_nodes = (
            [ir.Node("", "ConstantOfShape", [shape], outputs=[zero]),
             ir.Node("", "Cast", [zero], [ir.AttrInt64("to", int(past.dtype))], outputs=[else_out])]
            if past.dtype != ir.DataType.FLOAT
            else [ir.Node("", "ConstantOfShape", [shape], outputs=[else_out])]
        )
        otherwise = ir.Graph(
            [], [else_out], nodes=fresh_nodes, name="fresh",
        )
        # False does not read dirty cache data; true consumes the preceding outputs.
        prefix = [
            ir.Node("", "Shape", [past], outputs=[shape]),
            ir.Node(
                "",
                "If",
                [phase],
                [ir.AttrGraph("then_branch", then), ir.AttrGraph("else_branch", otherwise)],
                outputs=[selected],
            ),
        ]
        for node in original_nodes:
            for i, inp in enumerate(node.inputs):
                if inp is past:
                    node.replace_input_with(i, selected)
        graph.insert_before(original_nodes[0], prefix)
    observed = value("observed_phase", ir.DataType.BOOL, [] if rank == 0 else [1])
    graph.append(ir.Node("", "Identity", [phase], outputs=[observed]))
    graph.outputs.append(observed)
    if not multimodal:
        offset = value("cache_offset", ir.DataType.INT64, [])
        observed_offset = value("observed_offset", ir.DataType.INT64, [])
        graph.inputs.append(offset)
        graph.append(ir.Node("", "Identity", [offset], outputs=[observed_offset]))
        graph.outputs.append(observed_offset)

        def constant(name, data):
            tensor = ir.tensor(np.asarray(data, dtype=np.int64), name=name)
            result = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=tensor.shape, const_value=tensor)
            graph.initializers[name] = result
            return result

        failure = value("fail_requested", ir.DataType.BOOL, [])
        reshape_shape = value("execution_shape", ir.DataType.INT64, [1])
        checked = value("execution_check", ir.DataType.INT64, [1])
        graph.append(ir.Node("", "Equal", [offset, constant("failure_offset", -1)], outputs=[failure]))
        graph.append(
            ir.Node(
                "",
                "Where",
                [failure, constant("invalid_shape", [-2]), constant("valid_shape", [1])],
                outputs=[reshape_shape],
            )
        )
        graph.append(ir.Node("", "Reshape", [constant("execution_data", [0]), reshape_shape], outputs=[checked]))
        graph.outputs.append(checked)
    ir.save(model, directory / "decoder.onnx")
    onnx.checker.check_model(directory / "decoder.onnx", full_check=True)
    config = json.loads((directory / "genai_config.json").read_text())
    config["model"]["decoder"]["inputs"]["has_previous_state"] = alias
    (directory / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if bf16 and multimodal:
        vocab = {f"token{i}": i for i in range(16)}
        vocab["a"] = vocab.pop("token2")
        vocab["b"] = vocab.pop("token3")
        tokenizer = {
            "version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [], "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE", "dropout": None, "unk_token": "token0",
                "continuing_subword_prefix": "", "end_of_word_suffix": "",
                "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                "vocab": vocab, "merges": [],
            },
        }
        (directory / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2) + "\n")
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}) + "\n"
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "models" / "cache-dispatch"
    create(root / "phase")
    create(root / "phase-alias", alias="cache_is_valid")
    create(root / "phase-multimodal", multimodal=True)
    create(root / "phase-multimodal-bf16", multimodal=True, bf16=True)
    create(root / "phase-multimodal-bf16-embedding-failure", multimodal=True, bf16=True, embedding_failure=True)
    create(root / "phase-rank", rank=1)
    create(root / "phase-multimodal-embedding-failure", multimodal=True, embedding_failure=True)
