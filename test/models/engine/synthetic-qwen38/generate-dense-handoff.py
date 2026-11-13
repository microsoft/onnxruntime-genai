# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate a supported folded package with independently checked dense handoff.

The folded target validates positions, pooled keys, pending rows and lengths,
including the separately prefilled head KV. Corruption becomes a nonzero device
validation status, not merely a cold/warm parity failure.

Run with --check to also exercise that oracle using the installed CPU ORT.
The complete folded graph requires CUDA ORT with SpeculativePrefixCommit.
"""

import argparse
import copy
import importlib
import json
import struct
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).parent


def value(name, data_type, shape):
    return helper.make_tensor_value_info(name, data_type, shape)


inputs = [
    value("input_ids", TensorProto.INT64, [1, "sequence"]),
    value("position_ids", TensorProto.INT64, [1, "sequence"]),
    value("past_position_ids", TensorProto.INT64, [1, "past"]),
    value("past_key_values.0.index_key", TensorProto.BFLOAT16, [1, "past", 128]),
]
outputs = [
    value("present_position_ids", TensorProto.INT64, [1, "total"]),
    value("present.0.index_key", TensorProto.BFLOAT16, [1, "total", 128]),
    value("logits", TensorProto.FLOAT, [1, "sequence", 128]),
    value("hidden_states", TensorProto.BFLOAT16, [1, "sequence", 1]),
]
initializers = [
    helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
    helper.make_tensor("one", TensorProto.INT64, [1], [1]),
    helper.make_tensor("ten", TensorProto.INT64, [1], [10]),
    helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
    helper.make_tensor("raw_basis", TensorProto.FLOAT, [128], [128.0] + [0.0] * 127),
    helper.make_tensor("vocab_ids", TensorProto.FLOAT, [128], list(range(128))),
]
nodes = [
    helper.make_node("Concat", ["past_position_ids", "position_ids"], ["present_position_ids"], axis=1),
    helper.make_node("Mul", ["input_ids", "zero"], ["zero_tokens"]),
    helper.make_node("Add", ["zero_tokens", "one"], ["ones"]),
    helper.make_node("Cast", ["ones"], ["float_ones"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["float_ones", "axis2"], ["rows"]),
    helper.make_node("Mul", ["rows", "raw_basis"], ["raw_keys"]),
    helper.make_node("Cast", ["raw_keys"], ["raw_bf16"], to=TensorProto.BFLOAT16),
    helper.make_node("Concat", ["past_key_values.0.index_key", "raw_bf16"], ["present.0.index_key"], axis=1),
    helper.make_node("Add", ["position_ids", "ten"], ["predictions"]),
    helper.make_node("Cast", ["predictions"], ["float_predictions"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["float_predictions", "axis2"], ["hidden_float"]),
    helper.make_node("Cast", ["hidden_float"], ["hidden_states"], to=TensorProto.BFLOAT16),
    helper.make_node("Sub", ["vocab_ids", "hidden_float"], ["distance"]),
    helper.make_node("Mul", ["distance", "distance"], ["distance_squared"]),
    helper.make_node("Neg", ["distance_squared"], ["logits"]),
]
graph = helper.make_graph(nodes, "synthetic-qwen38-dense", inputs, outputs, initializers)
dense = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(dense)
onnx.save(dense, ROOT / "dense-prefill.onnx")


def save_graph(name, nodes, inputs, outputs, initializers, microsoft=False, status_labels=None):
    imports = [helper.make_opsetid("", 17)]
    if microsoft:
        imports.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, initializers),
        opset_imports=imports,
        ir_version=8,
    )
    helper.set_model_props(model, {"experiment.validation_status_names": json.dumps(status_labels or ["handoff"])})
    onnx.checker.check_model(model)
    onnx.save(model, ROOT / f"{name}.onnx")
    return model


state = (
    ("key", TensorProto.BFLOAT16, [1, 2, 32, 256]),
    ("value", TensorProto.BFLOAT16, [1, 2, 32, 256]),
    ("index_key", TensorProto.FLOAT, [1, 8, 128]),
    ("index_buffer", TensorProto.BFLOAT16, [1, 4, 128]),
    ("index_lengths", TensorProto.INT32, [1, 2]),
)

# The KV-only head consumes dense hidden rows and produces nonzero KV so its
# separate prefill and strided handoff are observable by the folded validator.
head_kv_inputs = [value("hidden_states", TensorProto.BFLOAT16, [1, "sequence", 1])]
head_kv_outputs = [
    value("key", TensorProto.BFLOAT16, [1, "sequence", 512]),
    value("value", TensorProto.BFLOAT16, [1, "sequence", 512]),
    value("index_key", TensorProto.BFLOAT16, [1, "sequence", 128]),
]
save_graph(
    "head-kv",
    [
        helper.make_node("Cast", ["hidden_states"], ["hidden_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["hidden_float", "float_zero"], ["zero_rows"]),
        helper.make_node("Add", ["zero_rows", "kv_ones"], ["key_float"]),
        helper.make_node("Add", ["key_float", "kv_ones"], ["value_float"]),
        helper.make_node("Add", ["zero_rows", "raw_basis"], ["raw_float"]),
        helper.make_node("Cast", ["key_float"], ["key"], to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["value_float"], ["value"], to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["raw_float"], ["index_key"], to=TensorProto.BFLOAT16),
    ],
    head_kv_inputs,
    head_kv_outputs,
    [
        helper.make_tensor("float_zero", TensorProto.FLOAT, [1], [0.0]),
        helper.make_tensor("kv_ones", TensorProto.FLOAT, [512], [1.0] * 512),
        helper.make_tensor("raw_basis", TensorProto.FLOAT, [128], [128.0] + [0.0] * 127),
    ],
)

head_inputs = [
    value("input_ids", TensorProto.INT64, [1, 2]),
    value("hidden_states", TensorProto.BFLOAT16, [1, 2, 1]),
]
head_outputs = [
    value("next_token", TensorProto.INT64, [1, 2, 1]),
    value("hidden_states_out", TensorProto.BFLOAT16, [1, 2, 1]),
    value("logits", TensorProto.FLOAT, [1, 2, 128]),
    value("validation.aggregate", TensorProto.INT32, [1]),
]
head_nodes = [
    helper.make_node("Mul", ["input_ids", "zero"], ["zero_tokens"]),
    helper.make_node("Unsqueeze", ["zero_tokens", "axis2"], ["next_token"]),
    helper.make_node("Cast", ["next_token"], ["hidden_float"], to=TensorProto.FLOAT),
    helper.make_node("Cast", ["hidden_float"], ["hidden_states_out"], to=TensorProto.BFLOAT16),
    helper.make_node("Add", ["hidden_float", "vocab_zero"], ["logits"]),
    helper.make_node("Gather", ["zero_tokens", "zero"], ["zero_status"], axis=1),
    helper.make_node("Reshape", ["zero_status", "one"], ["status64"]),
    helper.make_node("Cast", ["status64"], ["validation.aggregate"], to=TensorProto.INT32),
]
for tail, dtype, shape in state:
    source = f"past_key_values.0.{tail}"
    head_inputs.append(value(source, dtype, shape))
    for prefix in ("present_mid", "present") if tail in ("index_buffer", "index_lengths") else ("present",):
        destination = f"{prefix}.0.{tail}"
        head_outputs.append(value(destination, dtype, shape))
        head_nodes.append(helper.make_node("Identity", [source], [destination]))
save_graph(
    "head-r2",
    head_nodes,
    head_inputs,
    head_outputs,
    [
        helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
        helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("vocab_zero", TensorProto.FLOAT, [128], [0.0] * 128),
    ],
)

inputs = [
    value("input_ids", TensorProto.INT64, [1, 3]),
    value("ple_embeddings.0", TensorProto.BFLOAT16, [1, 3, 1]),
    value("past_position_ids", TensorProto.INT64, [1, 32]),
]
outputs = [
    value("next_tokens", TensorProto.INT64, [1, 3]),
    value("logits", TensorProto.FLOAT, [1, 3, 128]),
    value("hidden_states", TensorProto.BFLOAT16, [1, 3, 1]),
    value("validation.aggregate", TensorProto.INT32, [1]),
    value("present_position_ids", TensorProto.INT64, [1, 32]),
]
nodes = [helper.make_node("Identity", ["past_position_ids"], ["present_position_ids"])]
initializers = []


def constant(name, dtype, shape, values):
    initializers.append(helper.make_tensor(name, dtype, shape, values))
    return name


def op(kind, args, name, **attrs):
    nodes.append(helper.make_node(kind, args, [name], **attrs))
    return name


zero64 = constant("zero64", TensorProto.INT64, [1], [0])
one64 = constant("one64", TensorProto.INT64, [1], [1])
axis2 = constant("axis2", TensorProto.INT64, [1], [2])
one = constant("one", TensorProto.FLOAT, [1], [1.0])
zero = constant("zero", TensorProto.FLOAT, [1], [0.0])
four = constant("four", TensorProto.FLOAT, [1], [4.0])
tolerance = constant("tolerance", TensorProto.FLOAT, [1], [0.02])
length64 = op("Gather", ["fold.ctl", one64], "length64", axis=0)
length = op("Cast", [length64], "length", to=TensorProto.FLOAT)
errors = []
error_labels = []


def check(actual, expected, tag):
    actual_float = op("Cast", [actual], f"{tag}_actual", to=TensorProto.FLOAT)
    delta = op("Sub", [actual_float, expected], f"{tag}_delta")
    maximum = op("ReduceMax", [op("Abs", [delta], f"{tag}_abs")], f"{tag}_max", keepdims=0)
    errors.append(op("Greater", [maximum, tolerance], f"{tag}_bad"))
    error_labels.append(tag)


positions = constant("positions", TensorProto.FLOAT, [1, 32], list(range(32)))
# Public BeginDecode prepares the three verify positions before running the graph.
prepared_length = op("Add", [length, constant("verify_rows", TensorProto.FLOAT, [1], [3.0])], "prepared_length")
mask = op("Less", [positions, prepared_length], "position_mask")
expected = op("Where", [mask, positions, zero], "expected_positions")
check("past_position_ids", expected, "positions")

for group in ("target", "head"):
    prefix = "" if group == "target" else "head."
    rows = length if group == "target" else op("Sub", [length, one], "head_rows")
    blocks = op("Floor", [op("Div", [rows, four], f"{group}_quarters")], f"{group}_blocks")
    pending = op("Sub", [rows, op("Mul", [blocks, four], f"{group}_complete")], f"{group}_pending")
    expected_lengths = op("Concat", [blocks, pending], f"{group}_lengths", axis=0)
    for tail, dtype, shape in state:
        if group == "target" and tail in ("key", "value"):
            continue
        source = f"{prefix}past_key_values.0.{tail}"
        inputs.append(value(source, dtype, shape))
        if tail in ("index_buffer", "index_lengths"):
            variants = ("present_mid", "present_mid2", "present")
        else:
            variants = ("present",)
        for variant in variants:
            destination = f"{prefix}{variant}.0.{tail}"
            outputs.append(value(destination, dtype, shape))
            op("Identity", [source], destination)
        tag = f"{group}_{tail}"
        if tail == "index_lengths":
            expected = expected_lengths
        else:
            if tail in ("key", "value"):
                indices = constant(f"{tag}_rows", TensorProto.FLOAT, [1, 1, 32, 1], list(range(32)))
                basis = constant(
                    f"{tag}_basis", TensorProto.FLOAT, [1, 2, 1, 256], [1.0 if tail == "key" else 2.0] * 512
                )
                count = rows
            else:
                capacity = 8 if tail == "index_key" else 4
                indices = constant(f"{tag}_rows", TensorProto.FLOAT, [1, capacity, 1], list(range(capacity)))
                basis = constant(
                    f"{tag}_basis",
                    TensorProto.FLOAT,
                    [1, 1, 128],
                    [128.0**0.5 if tail == "index_key" else 128.0] + [0.0] * 127,
                )
                count = blocks if tail == "index_key" else pending
            active = op("Cast", [op("Less", [indices, count], f"{tag}_mask")], f"{tag}_active", to=TensorProto.FLOAT)
            expected = op("Mul", [active, basis], f"{tag}_expected")
        check(source, expected, tag)

for name, shape in (("ctl", [16]), ("ctx_commit", [2]), ("ctx_a", [6]), ("ptrs", [2])):
    inputs.append(value(f"fold.{name}", TensorProto.INT64, shape))
for name, dtype, shape in (
    ("tail_done", TensorProto.INT32, [1]),
    ("ctl_out", TensorProto.INT64, [16]),
    ("ctx_commit_out", TensorProto.INT64, [2]),
):
    outputs.append(value(f"fold.{name}", dtype, shape))

# Native warmup also runs the target in muted prompt mode, without dense handoff.
mute = op("Gather", ["fold.ctl", constant("three64", TensorProto.INT64, [1], [3])], "mute", axis=0)
live = op("Equal", [mute, zero64], "live")
statuses = [
    op("Cast", [op("And", [live, error], f"{tag}_live_bad")], f"{tag}_status", to=TensorProto.INT32)
    for tag, error in zip(error_labels, errors, strict=True)
]
op("Concat", statuses, "validation.aggregate", axis=0)
for output in outputs:
    if output.name == "validation.aggregate":
        output.type.tensor_type.shape.dim[0].dim_value = len(errors)
predictions = op("Add", [length64, constant("offsets", TensorProto.INT64, [1, 3], [20, 21, 22])], "predictions")
op("Add", [op("Mul", ["input_ids", zero64], "zero_tokens"), predictions], "next_tokens")
hidden = op("Unsqueeze", [op("Cast", ["next_tokens"], "float_tokens", to=TensorProto.FLOAT), axis2], "hidden_float")
op("Cast", [hidden], "hidden_states", to=TensorProto.BFLOAT16)
vocab = constant("vocab", TensorProto.FLOAT, [128], list(range(128)))
distance = op("Sub", [vocab, hidden], "distance")
op("Neg", [op("Mul", [distance, distance], "squared")], "logits")
nodes.append(
    helper.make_node(
        "SpeculativePrefixCommit",
        ["next_tokens", "input_ids", "validation.aggregate", "fold.ctl", "fold.ctx_commit", "fold.ctx_a", "fold.ptrs"],
        ["fold.tail_done", "fold.ctl_out", "fold.ctx_commit_out"],
        domain="com.microsoft",
        head_rows=3,
    )
)
target = save_graph("decoder-handoff", nodes, inputs, outputs, initializers, microsoft=True, status_labels=error_labels)

data = bytearray()
index = ["qwen38_qsa_consts 1", "file qsa.data", "epsilon 0.000001", "width 2"]
for name, bits in (("cos", [0x3F80] * 64), ("sin", [0] * 64), ("knorm.0", [0x3F80] * 128)):
    payload = struct.pack(f"<{len(bits)}H", *bits)
    for group in ("target", "head"):
        index.append(f"tensor {group} {name} {len(data)} {len(payload)}")
    data.extend(payload)
(ROOT / "qsa.data").write_bytes(data)
(ROOT / "qsa.txt").write_bytes(("\n".join(index) + "\n").encode("utf-8"))


def check_oracle():
    """Run the actual target validator on CPU, omitting only the CUDA ring writer."""
    np = importlib.import_module("numpy")
    ort = importlib.import_module("onnxruntime")

    oracle = copy.deepcopy(target)
    del oracle.graph.node[-1]  # SpeculativePrefixCommit publishes to a CUDA-mapped ring.
    del oracle.graph.output[:]
    oracle.graph.output.extend(
        [
            value("validation.aggregate", TensorProto.INT32, [len(errors)]),
            value("next_tokens", TensorProto.INT64, [1, 3]),
        ]
    )
    onnx.checker.check_model(oracle)
    session = ort.InferenceSession(oracle.SerializeToString(), providers=["CPUExecutionProvider"])

    # BF16 input arrays are supplied as raw OrtValues because NumPy has no BF16 dtype.
    def run(feeds, graph_session=session, outputs=None):
        binding = graph_session.io_binding()
        keepalive = []
        for info in graph_session.get_inputs():
            array = np.ascontiguousarray(feeds[info.name])
            keepalive.append(array)
            if info.type == "tensor(bfloat16)":
                tensor = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(array, TensorProto.BFLOAT16)
                keepalive.append(tensor)
                binding.bind_ortvalue_input(info.name, tensor)
            else:
                binding.bind_cpu_input(info.name, array)
        for name in outputs or [info.name for info in graph_session.get_outputs()]:
            binding.bind_output(name)
        graph_session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()

    def bf16(array):
        return (np.asarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def float_outputs(model):
        # Leave BF16 computation and inputs intact; make outputs readable by NumPy.
        model = copy.deepcopy(model)
        for output in model.graph.output:
            if output.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                internal = output.name + "_bf16"
                for node in model.graph.node:
                    for i, name in enumerate(node.output):
                        if name == output.name:
                            node.output[i] = internal
                    for i, name in enumerate(node.input):
                        if name == output.name:
                            node.input[i] = internal
                model.graph.node.append(helper.make_node("Cast", [internal], [output.name], to=TensorProto.FLOAT))
                output.type.tensor_type.elem_type = TensorProto.FLOAT
        onnx.checker.check_model(model)
        return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    dense_session = float_outputs(dense)
    head_kv_session = float_outputs(onnx.load(ROOT / "head-kv.onnx"))
    head_session = float_outputs(onnx.load(ROOT / "head-r2.onnx"))
    for length in (3, 4, 5, 8, 9, 10):
        positions = np.zeros((1, 0), dtype=np.int64)
        raw_keys = np.zeros((1, 0, 128), dtype=np.float32)
        hidden_rows = []
        for start in range(0, length, 4):
            stop = min(start + 4, length)
            positions, raw_keys, logits, hidden = run(
                {
                    "input_ids": np.zeros((1, stop - start), dtype=np.int64),
                    "position_ids": np.arange(start, stop, dtype=np.int64).reshape(1, -1),
                    "past_position_ids": positions,
                    "past_key_values.0.index_key": bf16(raw_keys),
                },
                dense_session,
            )
            hidden_rows.append(hidden)
            assert logits[0, -1].argmax() == stop + 9
        assert positions.tolist() == [list(range(length))]
        expected_raw = np.zeros((1, length, 128), dtype=np.float32)
        expected_raw[:, :, 0] = 128.0
        np.testing.assert_array_equal(raw_keys, expected_raw)
        hidden = np.concatenate(hidden_rows, axis=1)
        key, val, raw = run({"hidden_states": bf16(hidden[:, :-1])}, head_kv_session)
        np.testing.assert_array_equal(key, np.ones((1, length - 1, 512), dtype=np.float32))
        np.testing.assert_array_equal(val, np.full((1, length - 1, 512), 2.0, dtype=np.float32))
        np.testing.assert_array_equal(raw, expected_raw[:, :-1])

        feeds = {}
        for info in session.get_inputs():
            dtype = {
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(float)": np.float32,
                "tensor(bfloat16)": np.uint16,
            }[info.type]
            feeds[info.name] = np.zeros(info.shape, dtype=dtype)
        feeds["fold.ctl"][1] = length
        feeds["past_position_ids"][0, : length + 3] = np.arange(length + 3)
        for group, rows in (("", length), ("head.", length - 1)):
            base = f"{group}past_key_values.0."
            blocks, pending = divmod(rows, 4)
            feeds[base + "index_key"][0, :blocks, 0] = 128.0**0.5
            feeds[base + "index_buffer"][0, :pending, 0] = bf16(128.0)
            feeds[base + "index_lengths"][0] = [blocks, pending]
            if group:
                feeds[base + "key"][:, :, :rows, :] = bf16(1.0)
                feeds[base + "value"][:, :, :rows, :] = bf16(2.0)
        status, predictions = run(feeds)
        assert status.tolist() == [0] * len(errors), (length, status)
        assert predictions.tolist() == [[length + 20, length + 21, length + 22]]
        for name, array in feeds.items():
            if name == "past_position_ids" or name.startswith(("past_key_values.", "head.past_key_values.")):
                original = array.copy()
                array.flat[0] += 1 if array.dtype != np.uint16 else 0x3F80
                assert sum(run(feeds)[0]) == 1, (length, name)
                feeds[name] = original
        head_feeds = {"input_ids": np.zeros((1, 2), dtype=np.int64), "hidden_states": bf16(hidden[:, -2:])}
        for tail, _, _ in state:
            head_feeds[f"past_key_values.0.{tail}"] = feeds[f"head.past_key_values.0.{tail}"]
        draft, status = run(head_feeds, head_session, ["next_token", "validation.aggregate"])
        assert draft.tolist() == [[[0], [0]]]
        assert status.tolist() == [0]
    print("Dense/head sessions and handoff oracle passed at 3/4/5/8/9/10 rows, including individually corrupted state.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="also validate the handoff oracle with CPU ORT")
    if parser.parse_args().check:
        check_oracle()
