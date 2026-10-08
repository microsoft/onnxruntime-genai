import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from models.builders.mtp import MTPModel


@pytest.fixture
def mtp_graph(tmp_path):
    indices = "/model/layers.0/attn/PackedSparseAttentionIndexer/output_0"
    counts = "/model/layers.0/attn/PackedSparseAttentionIndexer/output_1"
    nodes = [
        helper.make_node(
            "MatMul",
            ["hidden", "indexer.weight"],
            ["index_qk"],
            name="/model/layers.0/attn/indexer/index_qk_proj/MatMul",
        ),
        helper.make_node(
            "Split",
            ["index_qk", "split_sizes"],
            ["index_query", "index_key"],
            name="/model/layers.0/attn/PackedSparseAttentionIndexer/Split",
            axis=1,
        ),
        helper.make_node(
            "PackedSparseAttentionIndexer",
            ["index_query", "index_key", "norm", "norm", "hidden", "hidden", "hidden", "past_sequence_lengths"],
            [indices, counts, "present.indexer"],
            name="/model/layers.0/attn/PackedSparseAttentionIndexer",
            domain="com.microsoft",
            policy_mode="qsa",
            token_budget=4,
            compress_ratio=2,
        ),
        helper.make_node(
            "SparsePagedAttention",
            ["hidden", "expert.weight", "", "", "", "", "", "", "", indices, counts],
            ["attention_out", "present.key", "present.value"],
            name="/model/layers.0/attn/SparsePagedAttention",
            domain="com.microsoft",
        ),
        helper.make_node("Identity", ["attention_out"], ["logits"], name="logits"),
        helper.make_node("Identity", ["attention_out"], ["hidden_states_out"], name="feedback"),
    ]
    inputs = [
        helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["num_tokens", 2]),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["num_tokens"]),
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["num_tokens", 2])
        for name in ["logits", "hidden_states_out", "present.key", "present.value", "present.indexer"]
    ]
    tensors = [
        numpy_helper.from_array(np.ones((2, 6), dtype=np.float32), "indexer.weight"),
        numpy_helper.from_array(np.array([4, 2], dtype=np.int64), "split_sizes"),
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "norm"),
        numpy_helper.from_array(np.array([[0x12, 0xAB], [0xFF, 0x00]], dtype=np.uint8), "expert.weight"),
    ]
    graph = helper.make_graph(nodes, "mtp", inputs, outputs, tensors)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save_model(
        model,
        tmp_path / "mtp.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="mtp.onnx.data",
        size_threshold=0,
    )
    return tmp_path


@pytest.mark.parametrize("policy", ["csa", "invalid"])
def test_rejects_non_qsa_policy(mtp_graph, policy):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    indexer = next(node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer")
    next(attribute for attribute in indexer.attribute if attribute.name == "policy_mode").s = policy.encode()
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    with pytest.raises(ValueError, match="QSA raw-token"):
        MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)


@pytest.mark.parametrize("draft_count", range(1, 8))
def test_single_model_selection_io(mtp_graph, draft_count):
    source = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    weight_bytes = (mtp_graph / "mtp.onnx.data").read_bytes()
    metadata = MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", draft_count)
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    onnx.checker.check_model(str(mtp_graph / "mtp.onnx"), check_custom_domain=False)
    assert metadata["enabled"] is True
    assert set(metadata) == {"enabled", "base_capacity", "max_draft_tokens", "indices_output", "counts_output"}
    assert {path.name for path in mtp_graph.glob("*.onnx")} == {"mtp.onnx"}
    assert not any(node.op_type == "If" for node in model.graph.node)
    merge = next(node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexerMerge")
    assert merge.name.endswith("/IndexerMerge")
    assert list(merge.output) == [merge.name + suffix for suffix in ("/output_0", "/output_1", "/status")]
    assert {attribute.name for attribute in merge.attribute} == {"policy_mode", "max_output_entries"}
    indexer = next(node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer")
    assert list(indexer.input[18:22]) == ["indexshare.mode", *merge.output]
    assert indexer.input[:2] == ["index_qk", ""]
    assert len([node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer"]) == 1
    assert len([node for node in model.graph.node if node.op_type == "SparsePagedAttention"]) == 1
    assert not any(node.op_type == "Split" for node in model.graph.node)
    projection = next(node for node in model.graph.node if node.name.endswith("index_qk_proj/MatMul"))
    gather = next(node for node in model.graph.node if node.name.endswith("GatherProjectionRows"))
    assert projection.input[0] == gather.output[0]
    assert list(gather.input) == ["hidden", "indexshare.projection_rows"]
    assert "indexshare.base_row_indices" in {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    assert {"logits", "indexshare.0.status", metadata["indices_output"]} <= output_names
    assert not any(name.startswith("indexshare.decode.") for name in output_names)
    assert (mtp_graph / "mtp.onnx.data").read_bytes() == weight_bytes
    originals = {value.name: value.SerializeToString() for value in source.graph.initializer}
    for value in model.graph.initializer:
        if value.name in originals:
            assert value.SerializeToString() == originals[value.name]


def test_rejects_already_converted_graph(mtp_graph):
    MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    source_bytes = (mtp_graph / "mtp.onnx").read_bytes()
    with pytest.raises(ValueError, match="already been converted"):
        MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 3)
    assert (mtp_graph / "mtp.onnx").read_bytes() == source_bytes


def test_rejects_projection_with_other_consumers(mtp_graph):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    model.graph.node.append(helper.make_node("Identity", ["index_qk"], ["other_projection_use"]))
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    source_bytes = (mtp_graph / "mtp.onnx").read_bytes()
    with pytest.raises(ValueError, match="projection with other consumers"):
        MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    assert (mtp_graph / "mtp.onnx").read_bytes() == source_bytes


def test_side_by_side_package_does_not_touch_source(mtp_graph):
    tool_path = Path(__file__).resolve().parents[3] / "tools/export_indexshare_mtp.py"
    spec = importlib.util.spec_from_file_location("export_indexshare_mtp", tool_path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source_config = {"model": {"mtp": {"filename": "mtp.onnx"}}}
    (mtp_graph / "genai_config.json").write_text(json.dumps(source_config))
    source_bytes = {path.name: path.read_bytes() for path in mtp_graph.iterdir()}
    output = mtp_graph.parent / "indexshare"
    metadata = tool.export_package(mtp_graph, output, 3)
    assert metadata["enabled"] is True
    assert not (output / "mtp.onnx.data").is_symlink()
    assert (output / "mtp.onnx.data").samefile(mtp_graph / "mtp.onnx.data")
    source = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    expert = next(value for value in source.graph.initializer if value.name == "expert.weight")
    probe = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", [expert.name], ["native_bytes"])],
            "external_data_probe",
            [],
            [helper.make_tensor_value_info("native_bytes", TensorProto.UINT8, [2, 2])],
            [expert],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    probe.ir_version = 10
    onnx.save_model(probe, output / "probe.onnx")
    session = ort.InferenceSession(str(output / "probe.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {})[0], [[0x12, 0xAB], [0xFF, 0x00]])
    assert json.loads((output / "genai_config.json").read_text())["model"]["mtp"]["index_share"] == metadata
    for path in mtp_graph.iterdir():
        assert path.read_bytes() == source_bytes[path.name]
    with pytest.raises(ValueError, match="must not exist"):
        tool.export_package(mtp_graph, output, 3)
    source_config["model"]["mtp"]["filename"] = "../mtp.onnx"
    (mtp_graph / "genai_config.json").write_text(json.dumps(source_config))
    rejected_output = mtp_graph.parent / "rejected"
    with pytest.raises(ValueError, match="top-level"):
        tool.export_package(mtp_graph, rejected_output, 3)
    assert not rejected_output.exists()
