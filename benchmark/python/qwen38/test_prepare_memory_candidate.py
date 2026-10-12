# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import sys

import onnx
import pytest
from onnx import TensorProto, helper
from prepare_memory_candidate import main, transform_graph


def graph(indexers):
    nodes = [
        helper.make_node(
            "PackedSparseAttentionIndexer", [], [], name=f"indexer{i}", compress_ratio=4, state_capacity=65536
        )
        for i in range(indexers)
    ]
    nodes.append(helper.make_node("MatMul", ["hidden_states", "weight"], ["logits"], name="/lm_head/MatMul"))
    inputs = [
        helper.make_tensor_value_info(f"past.{i}.indexer_key", TensorProto.FLOAT16, [1, 65536, 128])
        for i in range(indexers)
    ]
    outputs = [
        helper.make_tensor_value_info(f"present.{i}.indexer_key", TensorProto.FLOAT16, [1, 65536, 128])
        for i in range(indexers)
    ]
    outputs.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["rows", 16]))
    weight = helper.make_tensor("weight", TensorProto.FLOAT16, [1], [1])
    return helper.make_model(helper.make_graph(nodes, "fixture", inputs, outputs, [weight]))


@pytest.mark.parametrize("indexers", [1, 12])
def test_graph_preserves_weights_and_selects_logits(indexers):
    model = graph(indexers)
    weight = model.graph.initializer[0].SerializeToString()
    transform_graph(model, indexers)
    assert model.graph.initializer[0].SerializeToString() == weight
    assert model.graph.node[-2].op_type == "Gather"
    assert model.graph.node[-1].input[0] == model.graph.node[-2].output[0]
    assert model.graph.input[-1].name == "logits_indices"
    assert model.graph.output[-1].type.tensor_type.shape.dim[0].dim_param == "num_logits"
    assert all(value.type.tensor_type.shape.dim[1].dim_value == 2560 for value in model.graph.input[:-1])
    with pytest.raises(ValueError, match="already selects logits"):
        transform_graph(model, indexers)


def test_rejects_unexpected_indexer_count():
    with pytest.raises(ValueError, match="Expected 12 indexers"):
        transform_graph(graph(1), 12)


def test_cli_copies_weights_and_persists_config(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "candidate"
    source.mkdir()
    for name, count in (("model.onnx", 12), ("mtp.onnx", 1)):
        onnx.save_model(graph(count), source / name)
    (source / "model.onnx.data").write_bytes(b"unchanged external weight fixture")
    config = {
        "model": {
            "decoder": {
                "filename": "model.onnx",
                "inputs": {},
                "session_options": {"provider_options": [{"cuda": {}}]},
            },
            "mtp": {"filename": "mtp.onnx", "inputs": {}, "session_options": {}},
        },
        "search": {},
        "engine": {"dynamic_batching": {}},
        "speculative": {},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["prepare", "--source", str(source), "--destination", str(target)])
    main()
    output = json.loads((target / "genai_config.json").read_text())
    assert output["model"]["context_length"] == 10240
    assert output["engine"]["dynamic_batching"]["max_scheduled_tokens"] == 4096
    assert output["engine"]["dynamic_batching"]["num_blocks"] == 40
    assert output["speculative"]["max_draft_tokens"] == 2
    assert (target / "model.onnx.data").read_bytes() == (source / "model.onnx.data").read_bytes()
    assert (target / "model.onnx.data").stat().st_ino != (source / "model.onnx.data").stat().st_ino
    assert json.loads((source / "genai_config.json").read_text()) == config
    with pytest.raises(SystemExit):
        main()


def test_rejects_destination_inside_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--source",
            str(tmp_path),
            "--destination",
            str(tmp_path / "candidate"),
        ],
    )
    with pytest.raises(SystemExit):
        main()
