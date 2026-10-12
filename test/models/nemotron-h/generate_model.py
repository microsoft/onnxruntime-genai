# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate weight-free hybrid/MTP fixtures, not a miniature pretrained Nemotron.

The target's causal scalar recurrence depends on both its BF16 convolution
carry and FP32 SSM carry. KV lengths are checked against an independent SSM
token counter. Sparse attention KV lives at layer 3; layers 1 and 2 have no
cache. Post-normalization hidden states condition the independent draft.
The draft deliberately mispredicts when its input token is 3, exercising rollback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: list[ir.Node] = []
        self.initializers: list[ir.Value] = []

    def constant(self, value: object, dtype: object = np.int64) -> ir.Value:
        tensor = ir.Tensor(np.asarray(value, dtype=dtype))
        result = ir.Value(
            name=f"constant_{len(self.initializers)}",
            type=ir.TensorType(tensor.dtype),
            shape=ir.Shape(tensor.shape),
            const_value=tensor,
        )
        self.initializers.append(result)
        return result

    def op(self, name: str, *inputs: ir.Value, **attributes: int | float) -> ir.Value:
        attrs = {
            key: ir.AttrFloat32(key, value) if isinstance(value, float) else ir.AttrInt64(key, value)
            for key, value in attributes.items()
        }
        result = ir.Value(name=f"value_{len(self.nodes)}")
        self.nodes.append(ir.Node("", name, inputs, attrs, outputs=[result]))
        return result

    def slice(self, value: ir.Value, start: int, end: int, axis: int) -> ir.Value:
        return self.op(
            "Slice",
            value,
            self.constant([start]),
            self.constant([end]),
            self.constant([axis]),
        )

    def unsqueeze(self, value: ir.Value, axes: list[int]) -> ir.Value:
        return self.op("Unsqueeze", value, self.constant(axes))

    def squeeze(self, value: ir.Value, axes: list[int]) -> ir.Value:
        return self.op("Squeeze", value, self.constant(axes))

    def floating(self, value: ir.Value) -> ir.Value:
        return self.op("Cast", value, to=int(ir.DataType.FLOAT))

    def hidden(self, accumulator: ir.Value, tokens: ir.Value, positions: ir.Value) -> ir.Value:
        zero = self.op("Mul", accumulator, self.constant(0.0, np.float32))
        one = self.op("Add", zero, self.constant(1.0, np.float32))
        hidden = self.op(
            "Concat",
            *(self.unsqueeze(value, [2]) for value in (accumulator, one, tokens, positions)),
            axis=2,
        )
        variance = self.op(
            "ReduceMean",
            self.op("Mul", hidden, hidden),
            self.constant([-1]),
            keepdims=1,
        )
        denominator = self.op("Sqrt", self.op("Add", variance, self.constant(1e-5, np.float32)))
        return self.op("Cast", self.op("Div", hidden, denominator), to=int(ir.DataType.BFLOAT16))

    def logits(self, accumulator: ir.Value) -> ir.Value:
        indices = self.op(
            "Cast",
            self.op("Add", accumulator, self.constant(1.0, np.float32)),
            to=int(ir.DataType.INT64),
        )
        logits = self.op(
            "OneHot",
            indices,
            self.constant(16),
            self.constant([0.0, 1.0], np.float32),
            axis=-1,
        )
        return self.op("Cast", logits, to=int(ir.DataType.BFLOAT16))

    def cache(self, tokens: ir.Value, past: ir.Value) -> ir.Value:
        current = self.unsqueeze(self.op("Cast", tokens, to=int(ir.DataType.BFLOAT16)), [1, 3])
        current = self.op("Concat", current, current, axis=3)
        return self.op("Concat", past, current, axis=2)

    def save(self, path: Path, inputs: list[ir.Value], outputs: list[ir.Value]) -> None:
        graph = ir.Graph(
            inputs,
            outputs,
            nodes=self.nodes,
            initializers=self.initializers,
            opset_imports={"": 24},
            name="synthetic_nemotron_h_state_contract",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ir.save(ir.Model(graph, ir_version=10, producer_name="onnxruntime-genai-test"), path)


def value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def output(result: ir.Value, name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    result.name = name
    result.type = ir.TensorType(dtype)
    result.shape = ir.Shape(shape)
    return result


def generate_target(path: Path) -> None:
    graph = GraphBuilder()
    ids = value("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    mask = value("attention_mask", ir.DataType.INT64, ["batch", "total_sequence"])
    conv = value("past_key_values.0.conv_state", ir.DataType.BFLOAT16, ["batch", 4, 3])
    ssm = value("past_key_values.0.ssm_state", ir.DataType.FLOAT, ["batch", 1, 2, 2])
    key = value("past_key_values.3.key", ir.DataType.BFLOAT16, ["batch", 1, "past", 2])
    cache_value = value("past_key_values.3.value", ir.DataType.BFLOAT16, ["batch", 1, "past", 2])
    tokens = graph.floating(ids)
    previous = graph.squeeze(graph.slice(graph.slice(graph.floating(conv), 0, 1, 1), -1, 2**63 - 1, 2), [1])
    shifted = graph.slice(graph.op("Concat", previous, tokens, axis=1), 0, -1, 1)
    initial = graph.slice(graph.op("Reshape", ssm, graph.constant([0, -1])), 0, 1, 1)
    initial_position = graph.slice(graph.op("Reshape", ssm, graph.constant([0, -1])), 1, 2, 1)
    ones = graph.op("Add", graph.op("Mul", tokens, graph.constant(0.0, np.float32)), graph.constant(1.0, np.float32))
    positions = graph.op("Add", initial_position, graph.op("CumSum", ones, graph.constant(1)))
    cache_matches = graph.op(
        "And",
        graph.op("Equal", graph.floating(graph.op("Shape", key, start=2, end=3)), initial_position),
        graph.op("Equal", graph.floating(graph.op("Shape", cache_value, start=2, end=3)), initial_position),
    )
    accumulator = graph.op(
        "Mod",
        graph.op(
            "Add",
            initial,
            graph.op("CumSum", graph.op("Add", tokens, shifted), graph.constant(1)),
        ),
        graph.constant(7.0, np.float32),
        fmod=1,
    )
    current_conv = graph.unsqueeze(graph.op("Cast", tokens, to=int(ir.DataType.BFLOAT16)), [1])
    current_conv = graph.op("Concat", current_conv, current_conv, current_conv, current_conv, axis=1)
    present_conv = graph.slice(graph.op("Concat", conv, current_conv, axis=2), -3, 2**63 - 1, 2)
    final = graph.slice(accumulator, -1, 2**63 - 1, 1)
    final_position = graph.slice(positions, -1, 2**63 - 1, 1)
    zero = graph.op("Mul", final, graph.constant(0.0, np.float32))
    present_ssm = graph.op(
        "Reshape", graph.op("Concat", final, final_position, zero, zero, axis=1), graph.constant([0, 1, 2, 2])
    )
    # Poison logits when KV rollback/reset disagrees with the independently carried count.
    prediction = graph.op(
        "Where", cache_matches, accumulator, graph.op("Add", accumulator, graph.constant(8.0, np.float32))
    )
    graph.save(
        path,
        [ids, mask, conv, ssm, key, cache_value],
        [
            output(graph.logits(prediction), "logits", ir.DataType.BFLOAT16, ["batch", "sequence", 16]),
            output(
                graph.hidden(accumulator, tokens, positions), "mtp_seed", ir.DataType.BFLOAT16, ["batch", "sequence", 4]
            ),
            output(present_conv, "present.0.conv_state", ir.DataType.BFLOAT16, ["batch", 4, 3]),
            output(present_ssm, "present.0.ssm_state", ir.DataType.FLOAT, ["batch", 1, 2, 2]),
            output(graph.cache(tokens, key), "present.3.key", ir.DataType.BFLOAT16, ["batch", 1, "total_sequence", 2]),
            output(
                graph.cache(tokens, cache_value),
                "present.3.value",
                ir.DataType.BFLOAT16,
                ["batch", 1, "total_sequence", 2],
            ),
        ],
    )


def generate_head(path: Path) -> None:
    graph = GraphBuilder()
    ids = value("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    hidden = value("hidden_states", ir.DataType.BFLOAT16, ["batch", "sequence", 4])
    mask = value("attention_mask", ir.DataType.INT64, ["batch", "total_sequence"])
    key = value("past_key_values.0.key", ir.DataType.BFLOAT16, ["batch", 1, "past", 2])
    cache_value = value("past_key_values.0.value", ir.DataType.BFLOAT16, ["batch", 1, "past", 2])
    tokens = graph.floating(ids)
    source = graph.floating(hidden)
    divisor = graph.squeeze(graph.slice(source, 1, 2, 2), [2])
    initial = graph.op("Round", graph.op("Div", graph.squeeze(graph.slice(source, 0, 1, 2), [2]), divisor))
    previous_hidden = graph.op("Round", graph.op("Div", graph.squeeze(graph.slice(source, 2, 3, 2), [2]), divisor))
    previous_position = graph.op("Round", graph.op("Div", graph.squeeze(graph.slice(source, 3, 4, 2), [2]), divisor))
    positions = graph.op("Add", previous_position, graph.constant(1.0, np.float32))
    expected_past = graph.op("Sub", graph.slice(previous_position, 0, 1, 1), graph.constant(1.0, np.float32))
    cache_matches = graph.op(
        "And",
        graph.op("Equal", graph.floating(graph.op("Shape", key, start=2, end=3)), expected_past),
        graph.op("Equal", graph.floating(graph.op("Shape", cache_value, start=2, end=3)), expected_past),
    )
    past_tokens = graph.squeeze(graph.slice(graph.floating(key), 0, 1, 3), [1, 3])
    sentinel = graph.slice(previous_hidden, 0, 1, 1)
    previous = graph.slice(graph.op("Concat", sentinel, past_tokens, axis=1), -1, 2**63 - 1, 1)
    shifted = graph.slice(graph.op("Concat", previous, tokens, axis=1), 0, -1, 1)
    accumulator = graph.op(
        "Mod", graph.op("Add", initial, graph.op("Add", tokens, shifted)), graph.constant(7.0, np.float32), fmod=1
    )
    # Periodic errors force rollback even when token 3 only appears as a bonus token.
    wrong = graph.floating(
        graph.op(
            "Or",
            graph.op("Equal", ids, graph.constant(3)),
            graph.op(
                "Equal",
                graph.op("Mod", previous_position, graph.constant(5.0, np.float32), fmod=1),
                graph.constant(0.0, np.float32),
            ),
        )
    )
    prediction = graph.op("Mod", graph.op("Add", accumulator, wrong), graph.constant(7.0, np.float32), fmod=1)
    prediction = graph.op(
        "Where", cache_matches, prediction, graph.op("Add", prediction, graph.constant(8.0, np.float32))
    )
    graph.save(
        path,
        [ids, hidden, mask, key, cache_value],
        [
            output(graph.logits(prediction), "logits", ir.DataType.BFLOAT16, ["batch", "sequence", 16]),
            output(
                graph.hidden(accumulator, tokens, positions),
                "mtp_hidden",
                ir.DataType.BFLOAT16,
                ["batch", "sequence", 4],
            ),
            output(graph.cache(tokens, key), "present.0.key", ir.DataType.BFLOAT16, ["batch", 1, "total_sequence", 2]),
            output(
                graph.cache(tokens, cache_value),
                "present.0.value",
                ir.DataType.BFLOAT16,
                ["batch", 1, "total_sequence", 2],
            ),
        ],
    )


def configuration(head: bool) -> dict:
    decoder = {
        "filename": "model.onnx",
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_size": 2,
        "hidden_size": 4,
        "num_hidden_layers": 1 if head else 4,
        "inputs": {"past_recurrent_names": "past_key_values.%d.ssm_state"},
        "outputs": {
            "hidden_states": "mtp_hidden" if head else "mtp_seed",
            "present_recurrent_names": "present.%d.ssm_state",
        },
    }
    if head:
        decoder["inputs"]["hidden_states"] = "hidden_states"
    model = {
        "type": "nemotron_h",
        "vocab_size": 16,
        "context_length": 64,
        "bos_token_id": 1,
        "eos_token_id": 15,
        "pad_token_id": 0,
        "decoder": decoder,
    }
    if not head:
        model["mtp"] = {
            "prefill_head": True,
            "filename": "../mtp/model.onnx",
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "head_size": 2,
            "main_hidden_states": "mtp_seed",
            "outputs": {"hidden_states": "mtp_hidden"},
        }
    return {"model": model, "search": {"past_present_share_buffer": False}}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    generate_target(root / "decoder" / "model.onnx")
    generate_head(root / "mtp" / "model.onnx")
    for name in ("decoder", "mtp"):
        (root / name / "genai_config.json").write_text(
            json.dumps(configuration(name == "mtp"), indent=4) + "\n", encoding="utf-8"
        )
