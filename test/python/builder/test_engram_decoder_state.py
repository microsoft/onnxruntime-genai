# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from transformers import LlamaConfig

from models.builders.base import Model
from models.builders.qwen3_8 import Qwen4ExpTextModel


@pytest.mark.parametrize("history", [1, 2, 4])
@pytest.mark.parametrize("lengths", [[0, 0], [1, 3], [7, 2], [0, 5]])
def test_paged_engram_token_snapshots_without_decoder_lookup(history, lengths):
    model = object.__new__(Qwen4ExpTextModel)
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
    )
    Model.__init__(model, config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", "", {})
    capacity = 7
    model.use_paged_attention = True
    model.context_length_attrs["state_update_capacity"] = capacity
    model.ple_attrs = {"ngram_size": history + 1, "token_pad_id": 0}
    model.input_names.update(
        {
            "input_ids": "input_ids",
            "cumulative_sequence_lengths": "cu",
            "past.ple_tokens": {0: "past"},
            "state_update.capture_count": "capture",
        }
    )
    model.output_names.update(
        {
            "present.ple_tokens": {0: "present"},
            "state_update.ple_tokens": {0: "snapshots"},
        }
    )
    for name, dtype, shape in (
        ("input_ids", ir.DataType.INT64, ["num_tokens"]),
        ("cu", ir.DataType.INT32, [3]),
        ("past", ir.DataType.INT64, [2, history]),
        ("capture", ir.DataType.INT32, [2]),
        ("present", ir.DataType.INT64, [2, history]),
    ):
        model.graph.inputs.append(model.make_value(name, dtype, shape))
    model.make_ple_token_state(0)
    model.graph.outputs.extend(
        [
            model.make_value("present", ir.DataType.INT64, [2, history]),
            model.make_value("snapshots", ir.DataType.INT64, [2, capacity, history]),
        ]
    )
    assert not any(
        node.op_type in {"NGramHashMapping", "VarlenNGramHashMapping", "GatherBlockQuantized"} for node in model.graph
    )
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    tokens = np.arange(sum(lengths), dtype=np.int64) % 4
    past = np.arange(2 * history, dtype=np.int64).reshape(2, history)
    cu = np.asarray([0, lengths[0], sum(lengths)], dtype=np.int32)
    for capture in ([7, 7], [0, 1], [-1, 20]):
        standalone_history = np.stack(
            [np.concatenate([past[b], tokens[cu[b] : cu[b + 1]]])[-history:] for b in range(2)]
        )
        present, snapshots = session.run(
            None,
            {
                "input_ids": tokens,
                "cu": cu,
                "past": past,
                "capture": np.asarray(capture, dtype=np.int32),
                "present": standalone_history,
            },
        )
        np.testing.assert_array_equal(present, standalone_history)
        expected = np.zeros((2, capacity, history), dtype=np.int64)
        for batch, length in enumerate(lengths):
            combined = np.concatenate([past[batch], tokens[cu[batch] : cu[batch + 1]]])
            for step in range(min(max(capture[batch], 0), length, capacity)):
                expected[batch, step] = combined[step + 1 : step + 1 + history]
        np.testing.assert_array_equal(snapshots, expected)
