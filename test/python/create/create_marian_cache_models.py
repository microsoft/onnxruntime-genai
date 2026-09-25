# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Create small, value-sensitive Marian source-projection cache fixtures."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PREFIX = "cached_source_projection_"
HIDDEN = 4
VOCAB = 128


def save_graph(directory, stage, inputs, outputs, nodes, constants):
    graph = helper.make_graph(
        nodes,
        stage,
        [helper.make_tensor_value_info(*value) for value in inputs],
        [helper.make_tensor_value_info(*value) for value in outputs],
        [
            numpy_helper.from_array(np.asarray(value), name)
            for name, value in constants.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.checker.check_model(model)
    onnx.save(model, directory / f"{stage}.onnx")


def create_model(
    directory,
    *,
    count=6,
    dtype=TensorProto.FLOAT,
    width=HIDDEN,
    fixed=False,
    error=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    batch, source = (2, 3) if fixed else ("batch", "source")
    shape = [batch, source, width]
    hidden_shape = [batch, source, HIDDEN]
    float_type = TensorProto.FLOAT
    constants = {
        "unsqueeze_axis": np.array([2], np.int64),
        "hidden_repeats": np.array([1, 1, HIDDEN], np.int64),
    }
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["float_ids"], to=float_type),
        helper.make_node(
            "Unsqueeze", ["float_ids", "unsqueeze_axis"], ["expanded_ids"]
        ),
        helper.make_node(
            "Tile", ["expanded_ids", "hidden_repeats"], ["encoder_outputs"]
        ),
    ]
    inputs = [
        ("input_ids", TensorProto.INT32, [batch, source]),
        ("attention_mask", TensorProto.INT32, [batch, source]),
    ]
    outputs = [("encoder_outputs", float_type, hidden_shape)]
    for i in range(count):
        name = PREFIX + str(i)
        if error == "missing-output" and i == 0:
            continue
        if error in ("wrong-sessions", "encoder-input") and i == 0:
            inputs.append((name, dtype, shape))
            if error == "encoder-input":
                outputs.append((name, dtype, shape))
            continue
        constants[f"offset_{i}"] = np.array(i, np.float32)
        constants[f"repeats_{i}"] = np.array([1, 1, width], np.int64)
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["expanded_ids", f"offset_{i}"], [f"offset_ids_{i}"]
                ),
                helper.make_node(
                    "Tile", [f"offset_ids_{i}", f"repeats_{i}"], [f"projection_{i}"]
                ),
                helper.make_node("Cast", [f"projection_{i}"], [name], to=dtype),
            ]
        )
        outputs.append((name, dtype, shape))
    save_graph(directory, "encoder", inputs, outputs, nodes, constants)

    inputs = [
        ("input_ids", TensorProto.INT32, [batch]),
        ("encoder_hidden_states", float_type, hidden_shape),
        ("encoder_attention_mask", TensorProto.INT32, [batch, source]),
        ("rnn_states_prev", float_type, [3, batch, HIDDEN]),
        ("past_key_values_length", TensorProto.INT64, [1]),
    ]
    outputs = [
        ("logits", float_type, [batch, VOCAB]),
        ("rnn_states", float_type, [3, batch, HIDDEN]),
    ]
    constants = {
        "reduce_axes": np.array([1, 2], np.int64),
        "modulus": np.array(100, np.int64),
        "one": np.array(1, np.int64),
        "vocab": np.array(VOCAB, np.int64),
        "one_hot_values": np.array([0, 10], np.float32),
    }
    nodes = [helper.make_node("Identity", ["rnn_states_prev"], ["rnn_states"])]
    sums = []
    for i in range(count or 6):
        name = PREFIX + str(i)
        cache_shape = copy.copy(shape)
        cache_type = dtype
        if i == 0:
            if error == "missing-input":
                continue
            if error == "wrong-sessions":
                nodes.append(
                    helper.make_node("Identity", ["encoder_hidden_states"], [name])
                )
                outputs.append((name, float_type, hidden_shape))
                continue
            if error == "type":
                cache_type = TensorProto.FLOAT16
            elif error == "batch":
                cache_shape[0] = 1
            elif error == "source":
                cache_shape[1] = 2
            elif error == "width":
                cache_shape[2] = 2
            elif error == "rank":
                cache_shape = [batch, 12]
            elif error == "symbols":
                cache_shape[1] = "other_source"
            elif error == "static-dynamic":
                cache_shape[1] = 3
        if count:
            inputs.append((name, cache_type, cache_shape))
            if error == "decoder-output" and i == 0:
                outputs.append((name, cache_type, cache_shape))
        else:
            constants[f"offset_{i}"] = np.array(i, np.float32)
            nodes.append(
                helper.make_node(
                    "Add", ["encoder_hidden_states", f"offset_{i}"], [name]
                )
            )
        constants[f"weight_{i}"] = np.array(i + 1, np.float32)
        axes = "reduce_axes"
        if len(cache_shape) == 2:
            axes = "rank_two_axes"
            constants[axes] = np.array([1], np.int64)
        nodes.extend(
            [
                helper.make_node("Cast", [name], [f"float_cache_{i}"], to=float_type),
                helper.make_node(
                    "ReduceSum", [f"float_cache_{i}", axes], [f"sum_{i}"], keepdims=0
                ),
                helper.make_node(
                    "Mul", [f"sum_{i}", f"weight_{i}"], [f"weighted_sum_{i}"]
                ),
            ]
        )
        sums.append(f"weighted_sum_{i}")
    nodes.extend(
        [
            helper.make_node("Sum", sums, ["cache_sum"]),
            helper.make_node(
                "Cast", ["cache_sum"], ["cache_sum_int"], to=TensorProto.INT64
            ),
            helper.make_node(
                "Add", ["cache_sum_int", "past_key_values_length"], ["step_sum"]
            ),
            helper.make_node("Mod", ["step_sum", "modulus"], ["token_mod"]),
            helper.make_node("Add", ["token_mod", "one"], ["token"]),
            helper.make_node(
                "OneHot", ["token", "vocab", "one_hot_values"], ["logits"], axis=-1
            ),
        ]
    )
    save_graph(directory, "decoder", inputs, outputs, nodes, constants)
    config = {
        "model": {
            "type": "marian-ssru",
            "bos_token_id": 127,
            "eos_token_id": 0,
            "pad_token_id": 127,
            "vocab_size": VOCAB,
            "context_length": 64,
            "encoder": {
                "filename": "encoder.onnx",
                "hidden_size": HIDDEN,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                },
                "outputs": {"encoder_outputs": "encoder_outputs"},
            },
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": HIDDEN,
                "session_options": {"intra_op_num_threads": 1},
                "inputs": {
                    "input_ids": "input_ids",
                    "encoder_hidden_states": "encoder_hidden_states",
                    "encoder_attention_mask": "encoder_attention_mask",
                    "rnn_states_prev": "rnn_states_prev",
                    "past_key_values_length": "past_key_values_length",
                },
                "outputs": {"logits": "logits", "rnn_states": "rnn_states"},
            },
        },
        "search": {"max_length": 12, "num_beams": 1, "do_sample": False, "top_k": 1},
    }
    (directory / "genai_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    root = parser.parse_args().output_dir
    create_model(root / "dynamic")
    create_model(root / "static", fixed=True)
    create_model(root / "fp16", count=3, dtype=TensorProto.FLOAT16, width=2)
    create_model(root / "unsupported-type", dtype=TensorProto.INT32)
    create_model(root / "uncached", count=0)
    for error in (
        "missing-output",
        "missing-input",
        "wrong-sessions",
        "encoder-input",
        "decoder-output",
        "type",
        "batch",
        "source",
        "width",
        "rank",
        "symbols",
        "static-dynamic",
    ):
        create_model(
            root / error,
            error=error,
            fixed=error in ("batch", "source", "width", "rank"),
        )


if __name__ == "__main__":
    main()
