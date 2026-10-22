import json
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
            "Constant",
            [],
            ["split_sizes"],
            name="split_sizes",
            value=numpy_helper.from_array(np.array([4, 2], dtype=np.int64)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["constant_output"],
            name="constant_output",
            value=numpy_helper.from_array(np.ones((1, 2), dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["unused_constant"],
            name="unused_constant",
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
            onnx.load(mtp_graph / "mtp.onnx"),
            bits=4,
            block_size=16,
            is_symmetric=True,
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
    assert set(metadata) == {
        "enabled",
        "base_capacity",
        "max_draft_tokens",
        "indices_output",
        "counts_output",
        "status_output",
    }
    assert {path.name for path in mtp_graph.glob("*.onnx")} == {"mtp.onnx"}
    assert not any(node.op_type == "If" for node in model.graph.node)
    merge = next(node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexerMerge")
    assert merge.name.endswith("/IndexerMerge")
    assert list(merge.output) == [merge.name + suffix for suffix in ("/output_0", "/output_1", "/status")]
    assert {attribute.name for attribute in merge.attribute} == {"policy_mode", "max_output_entries"}
    indexer = next(node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer")
    assert list(indexer.input[18:22]) == ["mode", *merge.output]
    assert indexer.input[:2] == ["index_qk", ""]
    assert len([node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer"]) == 1
    assert len([node for node in model.graph.node if node.op_type == "SparsePagedAttention"]) == 1
    assert not any(node.op_type == "Split" for node in model.graph.node)
    gather = next(node for node in model.graph.node if node.name.endswith("GatherProjectionRows"))
    projection = next(node for node in model.graph.node if gather.output[0] in node.input)
    assert projection.op_type == ("MatMulNBits" if quantized_projection else "MatMul")
    assert projection.input[0] == gather.output[0]
    assert list(gather.input) == ["hidden", "projection_rows"]
    assert "base_row_indices" in {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    assert {"logits", "indexer_status", "present_indices", "present_counts", "constant_output"} <= output_names
    assert {"past_indices", "past_counts"} <= {value.name for value in model.graph.input}
    assert list(merge.input[:2]) == ["past_indices", "past_counts"]
    assert metadata["indices_output"] == "present_indices"
    assert metadata["counts_output"] == "present_counts"
    assert list(indexer.output[:2]) == ["present_indices", "present_counts"]
    assert indexer.output[7] == "indexer_status"
    assert metadata["status_output"] == "indexer_status"
    attention = next(node for node in model.graph.node if node.op_type == "SparsePagedAttention")
    assert list(attention.input[9:11]) == list(indexer.output[:2])
    assert not any(
        node.op_type == "Identity" and set(node.output).intersection(indexer.output[:2]) for node in model.graph.node
    )
    unused = {"split_sizes", "unused_constant", "unused.initializer"}
    assert not any(unused.intersection(node.output) for node in model.graph.node)
    assert not any(value.name in unused for value in model.graph.initializer)
    for public_name in ("present_indices", "present_counts"):
        value = next(value for value in model.graph.output if value.name == public_name)
        assert value.type.tensor_type.elem_type == TensorProto.INT32
        assert value.type.tensor_type.shape.dim[0].dim_param == "num_tokens"
        if public_name == "present_indices":
            assert value.type.tensor_type.shape.dim[1].dim_value == metadata["base_capacity"] + draft_count - 1
    graph_names = output_names | {value.name for value in model.graph.input}
    graph_names.update(name for node in model.graph.node for name in (*node.input, *node.output))
    assert not any(name.startswith("indexshare.") for name in graph_names)
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


@pytest.mark.parametrize(
    "reserved_name",
    [
        "past_indices",
        "past_counts",
        "present_indices",
        "present_counts",
        "indexer_status",
        "mode",
        "projection_rows",
        "base_row_indices",
        "range_starts",
        "range_ends",
        "/model/layers.0/attn/indexshare/GatherProjectionRows/output_0",
        "/model/layers.0/attn/indexshare/IndexerMerge/output_0",
        "/model/layers.0/attn/indexshare/IndexerMerge/output_1",
        "/model/layers.0/attn/indexshare/IndexerMerge/status",
    ],
)
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


@pytest.mark.parametrize("exposed_value", ["index_qk", "index_query", "index_key"])
def test_rejects_projection_or_split_exposed_as_graph_output(mtp_graph, exposed_value):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    model.graph.output.append(helper.make_tensor_value_info(exposed_value, TensorProto.FLOAT, ["num_tokens", 2]))
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    source_bytes = (mtp_graph / "mtp.onnx").read_bytes()
    with pytest.raises(ValueError, match="other consumers"):
        MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    assert (mtp_graph / "mtp.onnx").read_bytes() == source_bytes


def test_indexshare_preserves_metadata_and_live_value_info(mtp_graph):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    model.doc_string = "model documentation"
    model.producer_name = "test producer"
    helper.set_model_props(model, {"model_key": "model_value"})
    model.graph.doc_string = "graph documentation"
    model.graph.metadata_props.add(key="graph_key", value="graph_value")
    indexer = next(node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer")
    indexer.doc_string = "indexer documentation"
    indexer.metadata_props.add(key="node_key", value="node_value")
    untouched = helper.make_tensor_value_info("attention_out", TensorProto.FLOAT, ["num_tokens", 2])
    untouched.doc_string = "keep this intermediate"
    untouched.metadata_props.add(key="value_key", value="value_value")
    projected = helper.make_tensor_value_info("index_qk", TensorProto.FLOAT, ["num_tokens", 6])
    projected.doc_string = "projection documentation"
    projected.type.tensor_type.shape.dim[1].denotation = "FEATURE"
    selected = helper.make_tensor_value_info(indexer.output[0], TensorProto.INT32, ["num_tokens", 5])
    selected.doc_string = "selection documentation"
    selected.type.denotation = "indices"
    selected.type.tensor_type.shape.dim[1].denotation = "SELECTION"
    selected.metadata_props.add(key="selection_key", value="selection_value")
    model.graph.value_info.extend([untouched, projected, selected])
    model.graph.input[0].doc_string = "hidden documentation"
    model.graph.output[0].doc_string = "logits documentation"
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    external_refs = {
        tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer
        if tensor.name != "unused.initializer"
    }
    external_bytes = (mtp_graph / "mtp.onnx.data").read_bytes()

    MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)

    result = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    assert result.doc_string == model.doc_string
    assert result.producer_name == model.producer_name
    assert result.metadata_props == model.metadata_props
    assert result.graph.doc_string == model.graph.doc_string
    assert result.graph.metadata_props == model.graph.metadata_props
    result_indexer = next(node for node in result.graph.node if node.op_type == indexer.op_type)
    assert result_indexer.doc_string == indexer.doc_string
    assert result_indexer.metadata_props == indexer.metadata_props
    value_info = {value.name: value for value in result.graph.value_info}
    assert value_info["attention_out"].SerializeToString() == untouched.SerializeToString()
    assert value_info["index_qk"].doc_string == projected.doc_string
    assert value_info["index_qk"].type.tensor_type.shape.dim[0].dim_param == "indexer_projection_rows"
    assert value_info["index_qk"].type.tensor_type.shape.dim[1].denotation == "FEATURE"
    present_indices = next(value for value in result.graph.output if value.name == "present_indices")
    assert present_indices.doc_string == selected.doc_string
    assert present_indices.metadata_props == selected.metadata_props
    assert present_indices.type.tensor_type.shape.dim[1].dim_value == 11
    assert present_indices.type.denotation == "indices"
    assert present_indices.type.tensor_type.shape.dim[1].denotation == "SELECTION"
    assert result.graph.input[0].SerializeToString() == model.graph.input[0].SerializeToString()
    assert result.graph.output[0].SerializeToString() == model.graph.output[0].SerializeToString()
    assert {tensor.name: tensor.SerializeToString() for tensor in result.graph.initializer} == external_refs
    assert (mtp_graph / "mtp.onnx.data").read_bytes() == external_bytes


@pytest.mark.parametrize("projection_row_count", [0, 2])
@pytest.mark.parametrize("quantized_projection", [False, True])
@pytest.mark.parametrize("with_split", [False, True])
def test_gated_projection_value_info_has_independent_row_count(
    mtp_graph, projection_row_count, quantized_projection, with_split
):
    source = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    source.graph.value_info.append(
        helper.make_tensor_value_info("index_qk", TensorProto.FLOAT, ["num_tokens", 6])
    )
    split = next(node for node in source.graph.node if node.op_type == "Split")
    if with_split:
        source.graph.value_info.extend(
            [
                helper.make_tensor_value_info("index_query", TensorProto.FLOAT, ["num_tokens", 4]),
                helper.make_tensor_value_info("index_key", TensorProto.FLOAT, ["num_tokens", 2]),
            ]
        )
    else:
        indexer = next(node for node in source.graph.node if node.op_type == "PackedSparseAttentionIndexer")
        indexer.input[:2] = ["index_qk", ""]
        source.graph.node.remove(split)
    onnx.save_model(source, mtp_graph / "mtp.onnx")
    if quantized_projection:
        quantizer = MatMulNBitsQuantizer(
            onnx.load(mtp_graph / "mtp.onnx"),
            bits=4,
            block_size=16,
            is_symmetric=True,
            op_types_to_quantize=("MatMul",),
        )
        quantizer.process()
        onnx.save_model(quantizer.model.model, mtp_graph / "mtp.onnx")

    MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)

    result = onnx.load(mtp_graph / "mtp.onnx")
    value_info = {value.name: value for value in result.graph.value_info}
    gather = next(node for node in result.graph.node if node.name.endswith("/GatherProjectionRows"))
    projection = next(node for node in result.graph.node if node.op_type in {"MatMul", "MatMulNBits"})
    for name, width in ((gather.output[0], 2), (projection.output[0], 6)):
        shape = value_info[name].type.tensor_type.shape
        assert shape.dim[0].dim_param == "indexer_projection_rows"
        assert shape.dim[1].dim_value == width
    assert "index_query" not in value_info
    assert "index_key" not in value_info
    assert not any(node.op_type == "Split" for node in result.graph.node)
    inputs = {value.name: value for value in result.graph.input}
    assert inputs["hidden"].type.tensor_type.shape.dim[0].dim_param == "num_tokens"
    assert inputs["projection_rows"].type.tensor_type.shape.dim[0].dim_param == "indexer_projection_rows"
    outputs = {value.name: value for value in result.graph.output}
    assert outputs["present_indices"].type.tensor_type.shape.dim[0].dim_param == "num_tokens"

    # Execute the gated projection alone: reuse can project zero rows while hidden has tokens.
    prefix = helper.make_model(
        helper.make_graph(
            [gather, projection],
            "gated_projection",
            [inputs["hidden"], inputs["projection_rows"]],
            [value_info[projection.output[0]]],
            [tensor for tensor in result.graph.initializer if tensor.name in projection.input],
            value_info=[value_info[gather.output[0]]],
        ),
        opset_imports=list(result.opset_import),
        ir_version=10,
    )
    session = ort.InferenceSession(prefix.SerializeToString(), providers=["CPUExecutionProvider"])
    projected = session.run(
        None,
        {
            "hidden": np.ones((3, 2), dtype=np.float32),
            "projection_rows": np.arange(projection_row_count, dtype=np.int64),
        },
    )[0]
    assert projected.shape == (projection_row_count, 6)


def test_indexshare_does_not_require_node_names(mtp_graph):
    model = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    for node in model.graph.node:
        node.ClearField("name")
    onnx.save_model(model, mtp_graph / "mtp.onnx")
    MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    result = onnx.load(mtp_graph / "mtp.onnx", load_external_data=False)
    assert not any(node.op_type == "Split" for node in result.graph.node)
    assert any(node.op_type == "Gather" for node in result.graph.node)
    indexer = next(node for node in result.graph.node if node.op_type == "PackedSparseAttentionIndexer")
    attention = next(node for node in result.graph.node if node.op_type == "SparsePagedAttention")
    assert list(attention.input[9:11]) == list(indexer.output[:2]) == ["present_indices", "present_counts"]


@pytest.mark.parametrize("budget", [8, 16])
def test_qwen_indexshare_rejects_budget_beyond_exported_capacity(mtp_graph, budget):
    metadata = MTPModel().export_indexshare_graphs(str(mtp_graph), "mtp.onnx", 7)
    config_path = mtp_graph / "genai_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"decoder": {}},
                "speculative": {"max_draft_tokens": budget},
            }
        )
    )
    model = object.__new__(Qwen4ExpModel)
    model.decoder = SimpleNamespace(ep="cuda", use_paged_attention=True, num_kv_heads=2, head_size=256)
    model.mtp = SimpleNamespace(use_paged_attention=True, prune_lm_head=False)
    model.mtp_attrs = {"shared_initializers": [], "index_share": metadata}
    with pytest.raises(ValueError, match="exceeds the exported IndexShare capacity"):
        model.add_mtp_to_genai_config(str(mtp_graph))


@pytest.mark.parametrize(
    "ep,paged", [("cuda", True), ("cuda", False), ("cpu", True), ("webgpu", True), ("webgpu", False)]
)
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
    model.mtp = SimpleNamespace(use_paged_attention=paged, prune_lm_head=False)
    model.mtp_attrs = {"shared_initializers": [], "index_share": metadata}

    model.add_mtp_to_genai_config(str(mtp_graph))

    config = json.loads(config_path.read_text())
    mtp = config["model"]["mtp"]
    assert "index_share" not in mtp
    assert mtp["base_capacity"] == metadata["base_capacity"]
    assert "max_draft_tokens" not in mtp
    assert config["speculative"]["max_draft_tokens"] == 7
    assert mtp["inputs"]["past_indices"] == "past_indices"
    assert mtp["inputs"]["past_counts"] == "past_counts"
    assert mtp["outputs"]["present_indices"] == "present_indices"
    assert mtp["outputs"]["present_counts"] == "present_counts"
    assert mtp["outputs"]["indexer_status"] == "indexer_status"
    assert '"eos_token_id": [11, 12]' in config_path.read_text()
    expected_options = {
        "ep.cuda.fpa_intb_gemm": "1",
        "ep.cuda.qmoe_skip_nvfp4_gemv_profiling": "1",
        "ep.cuda.sparse_paged_attention_grouped": "1",
        "ep.cuda.sparse_paged_attention_grouped_tile_size": "8",
        "ep.cuda.sparse_paged_attention_grouped_vectorized": "1",
        "session.use_device_allocator_for_initializers": "1",
    }
    for section in ("decoder", "mtp"):
        session_options = config["model"][section]["session_options"]
        assert not {
            "ep.cuda.sparse_paged_attention_grouped_decode",
            "ep.cuda.sparse_paged_attention_grouped_decode_splits",
            "ep.cuda.sparse_paged_attention_warp_reduction",
        }.intersection(session_options)
        if ep == "cuda" and paged:
            assert all(session_options.get(name) == value for name, value in expected_options.items())
            if cuda_graph is not None:
                provider = next(
                    provider["cuda"] for provider in session_options["provider_options"] if "cuda" in provider
                )
                assert provider["enable_cuda_graph"] == cuda_graph
            else:
                assert not any(
                    "enable_cuda_graph" in provider.get("cuda", {})
                    for provider in session_options.get("provider_options", [])
                )
        else:
            assert "ep.cuda.sparse_paged_attention_grouped" not in session_options
        if ep == "webgpu" and paged:
            provider = next(provider[ep] for provider in session_options["provider_options"] if ep in provider)
            assert provider["validationMode"] == "disabled"
            assert provider.get(capture_option) == cuda_graph
            assert not any(name.startswith("ep.cuda.") for name in session_options)
