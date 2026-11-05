# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from test_qwen38_contrib_ops import emitted_nodes, make_attention, make_sparse_model
from transformers import LlamaConfig

from models.builders.base import Model
from models.builders.mtp import MTPModel
from models.builders.qwen3_8 import Qwen4ExpModel, Qwen4ExpMTPTextModel


def make_native_model(draft_count):
    model = make_sparse_model(paged=True)
    model.hidden_size = 16
    model.input_types = {}
    model.input_shapes = {}
    model.output_types = {}
    model.output_shapes = {}
    Qwen4ExpMTPTextModel.make_index_share_init(model, {"max_draft_tokens": draft_count})
    return model


def make_metadata(draft_count):
    return make_native_model(draft_count).sparse_indexer_attrs["index_share"]


@pytest.mark.parametrize("draft_count", range(1, 8))
@pytest.mark.parametrize("capture", [False, True])
def test_indexshare_is_emitted_with_the_initial_mtp_graph(draft_count, capture):
    model = make_native_model(draft_count)
    if capture:
        model.context_length_attrs = {"state_update_capacity": 7}
        model.input_names.update({"state_update.capture_count": "capture_count", "state_update.active": "active"})
        model.output_names["state_update.indexer"] = {3: "indexer_update"}

    model.make_qwen_sparse_attention(3, make_attention(), "hidden")

    indexer = next(kwargs for op, kwargs in emitted_nodes(model) if op == "PackedSparseAttentionIndexer")
    capacity = 35 + draft_count - 1
    assert indexer["inputs"][:2] == ["index_qk", ""]
    assert indexer["inputs"][18] == "mode"
    assert indexer["outputs"][:2] == ["present_indices", "present_counts"]
    assert indexer["outputs"][7] == "indexer_status"
    assert indexer["max_output_entries"] == capacity
    merge = next(kwargs for op, kwargs in emitted_nodes(model) if op == "PackedSparseAttentionIndexerMerge")
    assert merge["max_output_entries"] == capacity
    assert merge["policy_mode"] == "append_range"
    assert not any(method == "make_split" for method, *_ in model.calls)
    assert model.input_shapes["projection_rows"] == ["indexer_projection_rows"]
    assert model.output_shapes["present_indices"] == ["num_tokens", capacity]
    assert model.sparse_indexer_attrs["index_share"]["max_draft_tokens"] == draft_count
    assert not hasattr(MTPModel, "export_indexshare_graphs")
    assert not hasattr(MTPModel, "export_packed_indexshare_graph")


@pytest.mark.parametrize("draft_count", [0, 8, 16])
def test_native_mtp_rejects_unsupported_indexshare_capacity(draft_count):
    with pytest.raises(ValueError, match="between one and seven"):
        make_native_model(draft_count)


@pytest.mark.parametrize("projection_rows", [0, 2])
@pytest.mark.parametrize("bits", [0, 4])
def test_native_indexer_projection_accepts_independent_and_zero_row_counts(projection_rows, bits):
    class Projection(Model):
        def __init__(self):
            config = LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=32,
                max_position_embeddings=128,
                architectures=["LlamaForCausalLM"],
            )
            super().__init__(config, ir.DataType.FLOAT, ir.DataType.INT4 if bits else ir.DataType.FLOAT, "cpu", "", {})
            self.make_graph("native_index_projection")
            self.model.graph.inputs.extend(
                [
                    self.make_value("hidden", ir.DataType.FLOAT, ["num_tokens", 16]),
                    self.make_value("projection_rows", ir.DataType.INT64, ["indexer_projection_rows"]),
                ]
            )
            self.make_gather(
                "/GatherProjectionRows",
                ["hidden", "projection_rows"],
                ir.DataType.FLOAT,
                ["indexer_projection_rows", 16],
                axis=0,
            )
            weight = SimpleNamespace(weight=torch.ones(6, 16), bias=None)
            projection = self.make_matmul(
                weight,
                "/indexer/MatMul",
                "/GatherProjectionRows/output_0",
                output_shape=["indexer_projection_rows", 6],
            )
            self.model.graph.outputs.append(self.values[f"{projection}/output_0"])

    builder = Projection()
    model = builder.to_nbits() if bits else builder.model
    session = ort.InferenceSession(ir.to_proto(model).SerializeToString(), providers=["CPUExecutionProvider"])
    result = session.run(
        None,
        {
            "hidden": np.ones((3, 16), np.float32),
            "projection_rows": np.arange(projection_rows, dtype=np.int64),
        },
    )[0]
    assert result.shape == (projection_rows, 6)
    np.testing.assert_allclose(result, np.full((projection_rows, 6), 16, np.float32))


@pytest.mark.parametrize("budget", [8, 16])
def test_qwen_indexshare_rejects_budget_beyond_exported_capacity(tmp_path, budget):
    metadata = make_metadata(7)
    config_path = tmp_path / "genai_config.json"
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
        model.add_mtp_to_genai_config(str(tmp_path))


@pytest.mark.parametrize(
    "ep,paged", [("cuda", True), ("cuda", False), ("cpu", True), ("webgpu", True), ("webgpu", False)]
)
@pytest.mark.parametrize("cuda_graph", [None, "0", "1"])
def test_qwen_indexshare_config_and_session_options(tmp_path, ep, paged, cuda_graph):
    metadata = make_metadata(7)
    options = {"provider_options": [{ep: {}}]}
    capture_option = "enableGraphCapture" if ep == "webgpu" else "enable_cuda_graph"
    if cuda_graph is not None:
        options["provider_options"][0][ep][capture_option] = cuda_graph
    if ep == "webgpu":
        options["provider_options"][0][ep]["validationMode"] = "disabled"
    config_path = tmp_path / "genai_config.json"
    config_path.write_text(json.dumps({"model": {"decoder": {"session_options": options}, "eos_token_id": [11, 12]}}))
    model = object.__new__(Qwen4ExpModel)
    model.decoder = SimpleNamespace(ep=ep, use_paged_attention=paged, num_kv_heads=2, head_size=256)
    model.mtp = SimpleNamespace(use_paged_attention=paged, prune_lm_head=False)
    model.mtp_attrs = {"shared_initializers": [], "index_share": metadata}

    model.add_mtp_to_genai_config(str(tmp_path))

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
