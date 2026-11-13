# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prefill_converter():
    spec = importlib.util.spec_from_file_location(
        "nemotron_bolt_prefill", ROOT / "examples" / "python" / "nemotron-bolt-prefill.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_r8_package(tmp_path):
    source = tmp_path / "packed-r8"
    source.mkdir()
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["hidden"], to=int(TensorProto.BFLOAT16)),
        helper.make_node("Reshape", ["hidden", "shape_3d"], ["hidden_3d"]),
        helper.make_node("Identity", ["hidden_3d"], ["tap0"]),
        helper.make_node("Identity", ["hidden_3d"], ["tap1"]),
        helper.make_node("Concat", ["tap0", "tap1"], ["aux_cat"], axis=-1),
        helper.make_node("Identity", ["aux_cat"], ["aux_hidden_states"]),
        helper.make_node(
            "MatMulPacked",
            ["hidden_3d", "lm_w"],
            ["lm_out"],
            domain="com.microsoft",
            K=32,
            N=64,
            packed_type=8,
        ),
        helper.make_node("Identity", ["lm_out"], ["logits"]),
        helper.make_node("Cast", ["lm_out"], ["lm_f32"], to=int(TensorProto.FLOAT)),
        helper.make_node("ArgMax", ["lm_f32"], ["pred_ids"], axis=-1, keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic_bolt_r8",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8])],
        [
            helper.make_tensor_value_info("logits", TensorProto.BFLOAT16, [1, 8, 64]),
            helper.make_tensor_value_info("pred_ids", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("aux_hidden_states", TensorProto.BFLOAT16, [1, 8, 64]),
        ],
        initializer=[
            helper.make_tensor("shape_3d", TensorProto.INT64, [3], [1, -1, 32]),
            helper.make_tensor("lm_w", TensorProto.UINT8, [64, 34], [0] * (64 * 34)),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save(model, str(source / "model.onnx"))
    (source / "weights.bin").write_bytes(bytes(range(128)))
    config = {
        "model": {"type": "nemotron_bolt", "decoder": {"filename": "model.onnx"}},
        "speculative": {"max_draft_tokens": 7},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    return source


def test_prefill_converter_preserves_speculation_and_adds_row_selectors(prefill_converter, synthetic_r8_package):
    output = synthetic_r8_package.parent / "prefill-pkg"
    prefill_converter.convert(synthetic_r8_package, output)

    assert (output / "weights.bin").samefile(synthetic_r8_package / "weights.bin")
    out_cfg = json.loads((output / "genai_config.json").read_text())
    assert out_cfg["speculative"]["max_draft_tokens"] == 7

    model = onnx.load(str(output / "model.onnx"))
    in_shapes = {
        vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in model.graph.input
    }
    out_shapes = {
        vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in model.graph.output
    }
    assert in_shapes["input_ids"] == [1, "target_rows"]
    assert in_shapes["logit_rows"] == ["logit_row_count"]
    assert in_shapes["aux_rows"] == ["aux_row_count"]
    assert out_shapes["logits"] == [1, "logit_row_count", 64]
    assert out_shapes["pred_ids"] == [1, "logit_row_count"]
    assert out_shapes["aux_hidden_states"] == [1, "aux_row_count", 64]

    gather_names = [n.name for n in model.graph.node if n.op_type == "Gather"]
    assert gather_names == ["Gather_AuxRows_0", "Gather_AuxRows_1", "Gather_LogitRows"]
