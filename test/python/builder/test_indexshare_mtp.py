import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

from models.builders.mtp import MTPModel
from models.builders.qwen3_8 import Qwen4ExpModel


@pytest.fixture
def mtp_graph(tmp_path):
    indices = "/model/layers.0/attn/PackedSparseAttentionIndexer/output_0"
    counts = "/model/layers.0/attn/PackedSparseAttentionIndexer/output_1"
    nodes = [
        helper.make_node(
            "Constant", [], ["split_sizes"], name="split_sizes",
            value=numpy_helper.from_array(np.array([4, 2], dtype=np.int64)),
        ),
        helper.make_node(
            "Constant", [], ["constant_output"], name="constant_output",
            value=numpy_helper.from_array(np.ones((1, 2), dtype=np.float32)),
        ),
        helper.make_node(
            "Constant", [], ["unused_constant"], name="unused_constant",
            value=numpy_helper.from_array(np.ones(2, dtype=np.float32)),
        ),
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
    outputs.append(helper.make_tensor_value_info("constant_output", TensorProto.FLOAT, [1, 2]))
    tensors = [
        numpy_helper.from_array(np.ones((2, 6), dtype=np.float32), "indexer.weight"),
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "unused.initializer"),
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
@pytest.mark.parametrize("quantized_projection", [False, True])
def test_single_model_selection_io(mtp_graph, draft_count, quantized_projection):
    if quantized_projection:
        quantizer = MatMulNBitsQuantizer(
            onnx.load(mtp_graph / "mtp.onnx"), bits=4, block_size=16, is_symmetric=True,
            op_types_to_quantize=("MatMul",),
        )
        quantizer.process()
        onnx.save_model(quantizer.model.model, mtp_graph / "mtp.onnx")
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
    gather = next(node for node in model.graph.node if node.name.endswith("GatherProjectionRows"))
    projection = next(node for node in model.graph.node if gather.output[0] in node.input)
    assert projection.op_type == ("MatMulNBits" if quantized_projection else "MatMul")
    assert projection.input[0] == gather.output[0]
    assert list(gather.input) == ["hidden", "indexshare.projection_rows"]
    assert "indexshare.base_row_indices" in {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    assert {"logits", "indexshare.status", "indexshare.present_indices", "indexshare.present_counts", "constant_output"} <= output_names
    assert {"indexshare.past_indices", "indexshare.past_counts"} <= {value.name for value in model.graph.input}
    assert list(merge.input[:2]) == ["indexshare.past_indices", "indexshare.past_counts"]
    assert metadata["indices_output"] == "indexshare.present_indices"
    assert metadata["counts_output"] == "indexshare.present_counts"
    assert list(indexer.output[:2]) == ["indexshare.present_indices", "indexshare.present_counts"]
    assert indexer.output[7] == "indexshare.status"
    attention = next(node for node in model.graph.node if node.op_type == "SparsePagedAttention")
    assert list(attention.input[9:11]) == list(indexer.output[:2])
    assert not any(node.op_type == "Identity" and set(node.output).intersection(indexer.output[:2]) for node in model.graph.node)
    unused = {"split_sizes", "unused_constant", "unused.initializer"}
    assert not any(unused.intersection(node.output) for node in model.graph.node)
    assert not any(value.name in unused for value in model.graph.initializer)
    for public_name in ("indexshare.present_indices", "indexshare.present_counts"):
        value = next(value for value in model.graph.output if value.name == public_name)
        assert value.type.tensor_type.elem_type == TensorProto.INT32
        assert value.type.tensor_type.shape.dim[0].dim_param == "num_tokens"
        if public_name == "indexshare.present_indices":
            assert value.type.tensor_type.shape.dim[1].dim_value == metadata["base_capacity"] + draft_count - 1
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


@pytest.mark.parametrize("reserved_name", [
    "indexshare.past_indices", "indexshare.past_counts",
    "indexshare.present_indices", "indexshare.present_counts", "indexshare.status",
])
def test_rejects_indexshare_name_collision(mtp_graph, reserved_name):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    model.graph.node.append(helper.make_node("Identity", ["hidden"], [reserved_name]))
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    source_bytes = (mtp_graph / "mtp.onnx").read_bytes()
    with pytest.raises(ValueError, match="input/output names must be unused"):
        MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    assert (mtp_graph / "mtp.onnx").read_bytes() == source_bytes


def test_rejects_projection_with_other_consumers(mtp_graph):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    model.graph.node.append(helper.make_node("Identity", ["index_qk"], ["other_projection_use"]))
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    source_bytes = (mtp_graph / "mtp.onnx").read_bytes()
    with pytest.raises(ValueError, match="projection with other consumers"):
        MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    assert (mtp_graph / "mtp.onnx").read_bytes() == source_bytes


@pytest.mark.parametrize("ep,paged", [("cuda", True), ("cuda", False), ("cpu", True), ("webgpu", True), ("webgpu", False)])
@pytest.mark.parametrize("cuda_graph", [None, "0", "1"])
def test_qwen_indexshare_config_and_session_options(mtp_graph, ep, paged, cuda_graph):
    metadata = MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    options = {"provider_options": [{ep: {}}]}
    capture_option = "enableGraphCapture" if ep == "webgpu" else "enable_cuda_graph"
    if cuda_graph is not None:
        options["provider_options"][0][ep][capture_option] = cuda_graph
    if ep == "webgpu":
        options["provider_options"][0][ep]["validationMode"] = "disabled"
    config_path = mtp_graph / "genai_config.json"
    config_path.write_text(json.dumps({"model": {"decoder": {"session_options": options}, "eos_token_id": [11, 12]}}))
    model = object.__new__(Qwen4ExpModel)
    model.decoder = SimpleNamespace(ep=ep, use_paged_attention=paged, num_kv_heads=2, head_size=256)
    model.mtp_attrs = {"shared_initializers": [], "index_share": metadata}

    model.add_mtp_to_genai_config(str(mtp_graph))

    config = json.loads(config_path.read_text())
    mtp = config["model"]["mtp"]
    assert "index_share" not in mtp
    assert mtp["base_capacity"] == metadata["base_capacity"]
    assert "max_draft_tokens" not in mtp
    assert config["speculative"]["max_draft_tokens"] == 7
    assert mtp["inputs"]["past_indices"] == "indexshare.past_indices"
    assert mtp["inputs"]["past_counts"] == "indexshare.past_counts"
    assert mtp["outputs"]["present_indices"] == "indexshare.present_indices"
    assert mtp["outputs"]["present_counts"] == "indexshare.present_counts"
    assert '"eos_token_id": [11, 12]' in config_path.read_text()
    expected_options = {
        "ep.cuda.fpa_intb_gemm": "1",
        "ep.cuda.qmoe_skip_nvfp4_gemv_profiling": "1",
        "ep.cuda.sparse_paged_attention_grouped": "1",
        "ep.cuda.sparse_paged_attention_grouped_decode": "0",
        "ep.cuda.sparse_paged_attention_grouped_decode_splits": "0",
        "ep.cuda.sparse_paged_attention_grouped_tile_size": "8",
        "ep.cuda.sparse_paged_attention_grouped_vectorized": "1",
        "ep.cuda.sparse_paged_attention_warp_reduction": "0",
        "session.use_device_allocator_for_initializers": "1",
    }
    for section in ("decoder", "mtp"):
        session_options = config["model"][section]["session_options"]
        if ep == "cuda" and paged:
            assert all(session_options.get(name) == value for name, value in expected_options.items())
            if cuda_graph is not None:
                provider = next(provider["cuda"] for provider in session_options["provider_options"] if "cuda" in provider)
                assert provider["enable_cuda_graph"] == cuda_graph
            else:
                assert not any("enable_cuda_graph" in provider.get("cuda", {}) for provider in session_options.get("provider_options", []))
        else:
            assert "ep.cuda.sparse_paged_attention_grouped" not in session_options
        if ep == "webgpu" and paged:
            provider = next(provider[ep] for provider in session_options["provider_options"] if ep in provider)
            assert provider["validationMode"] == "disabled"
            assert provider.get(capture_option) == cuda_graph
            assert not any(name.startswith("ep.cuda.") for name in session_options)


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
    config = json.loads((output / "genai_config.json").read_text())
    mtp_config = config["model"]["mtp"]
    assert "index_share" not in mtp_config
    assert mtp_config["base_capacity"] == metadata["base_capacity"]
    assert "max_draft_tokens" not in mtp_config
    assert config["speculative"]["max_draft_tokens"] == 3
    assert mtp_config["inputs"]["past_indices"] == "indexshare.past_indices"
    assert mtp_config["inputs"]["past_counts"] == "indexshare.past_counts"
    assert mtp_config["outputs"]["present_indices"] == "indexshare.present_indices"
    assert mtp_config["outputs"]["present_counts"] == "indexshare.present_counts"
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
