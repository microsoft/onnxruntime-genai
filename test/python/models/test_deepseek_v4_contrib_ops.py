# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import onnx_ir as ir
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.deepseek import DeepSeekV4Model  # noqa: E402


def _linear(in_features: int, out_features: int):
    return SimpleNamespace(weight=SimpleNamespace(data=torch.randn(out_features, in_features)))


def _norm(width: int):
    return SimpleNamespace(weight=SimpleNamespace(data=torch.ones(width)))


def _compressor(hidden_size: int, head_size: int, rate: int, *, with_indexer: bool):
    compressor = SimpleNamespace(
        kv_proj=_linear(hidden_size, head_size * (2 if with_indexer else 1)),
        gate_proj=_linear(hidden_size, head_size * (2 if with_indexer else 1)),
        position_bias=SimpleNamespace(data=torch.randn(rate, head_size * (2 if with_indexer else 1))),
        kv_norm=_norm(head_size),
    )
    if with_indexer:
        index_head_size = 4
        compressor.indexer = SimpleNamespace(
            kv_proj=_linear(hidden_size, 2 * index_head_size),
            gate_proj=_linear(hidden_size, 2 * index_head_size),
            position_bias=SimpleNamespace(data=torch.randn(rate, 2 * index_head_size)),
            kv_norm=_norm(index_head_size),
            q_b_proj=_linear(3, 2 * index_head_size),
            scorer=SimpleNamespace(weights_proj=_linear(hidden_size, 2)),
        )
    return compressor


def test_fp8_checkpoint_is_dequantized_by_transformers(monkeypatch):
    calls = {}
    loaded_model = object()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            return loaded_model

    monkeypatch.setattr("models.builders.deepseek.AutoModelForCausalLM", FakeAutoModel)

    model = object.__new__(DeepSeekV4Model)
    model.dequantize_fp8 = True
    model.model_name_or_path = "deepseek-ai/DeepSeek-V4-Flash-0731"
    model.cache_dir = "/tmp/cache"
    model.hf_token = None
    model.hf_remote = False
    model.num_layers = 2
    model.extra_options = {"num_hidden_layers": 2}

    assert model.load_weights("unused") is loaded_model
    assert calls["model_name"] == model.model_name_or_path
    assert calls["kwargs"]["dtype"] == "auto"
    assert calls["kwargs"]["num_hidden_layers"] == 2
    assert calls["kwargs"]["quantization_config"].dequantize is True


def _make_builder(layer_types: list[str]) -> DeepSeekV4Model:
    model = object.__new__(DeepSeekV4Model)
    model.graph = ir.Graph(
        inputs=(), outputs=(), nodes=(), opset_imports={"": 21, "com.microsoft": 1}, name="main_graph"
    )
    model.model = ir.Model(model.graph, ir_version=10, producer_name="onnxruntime-genai")
    model.values = {}
    model.node_names = set()
    model.io_dtype = ir.DataType.FLOAT
    model.onnx_dtype = ir.DataType.FLOAT
    model.hidden_size = 8
    model.num_attn_heads = 2
    model.head_size = 6
    model.qk_rope_head_dim = 2
    model.compress_rates = {"heavily_compressed_attention": 4, "compressed_sparse_attention": 2}
    model.compress_rope_theta = 160000.0
    model.index_n_heads = 2
    model.index_head_dim = 4
    model.index_topk = 3
    model.rms_norm_epsilon = 1e-6
    model.hc_mult = 2
    model.hc_eps = 1e-6
    model.hc_sinkhorn_iters = 3
    model.layer_types = layer_types
    model.num_layers = len(layer_types)
    model.rope_attrs = {"cache_length": 16}
    model.input_names = {"position_ids": "position_ids"}
    model.input_types = {"position_ids": ir.DataType.INT64}
    model.input_shapes = {"position_ids": ["batch_size", "sequence_length"]}
    model.output_names = {}
    model.output_types = {}
    model.output_shapes = {}
    model.compression_state_names = []
    model.initialize_compression_states()
    model.make_value("hidden", ir.DataType.FLOAT, ["batch_size", "sequence_length", 8])
    model.make_value("q_residual", ir.DataType.FLOAT, ["batch_size", "sequence_length", 3])
    model.make_value("hc_streams", ir.DataType.FLOAT, ["batch_size", "sequence_length", 2, 8])
    return model


def test_emits_compression_contrib_contracts_and_state_config():
    model = _make_builder(["heavily_compressed_attention", "compressed_sparse_attention"])
    hca = SimpleNamespace(compressor=_compressor(8, 6, 4, with_indexer=False))
    csa = SimpleNamespace(compressor=_compressor(8, 6, 2, with_indexer=True))

    model.make_compressor(0, hca, "hidden", "q_residual")
    model.make_compressor(1, csa, "hidden", "q_residual")

    proto = ir.to_proto(model.model)
    contracts = {
        node.op_type: (len(node.input), len(node.output)) for node in proto.graph.node if node.domain == "com.microsoft"
    }
    assert contracts["HeavilyCompressedAttention"] == (11, 5)
    assert contracts["CompressedSparseAttention"] == (13, 6)
    assert contracts["LightningIndexer"] == (16, 6)
    assert len(model.compression_state_names) == 13

    config = {"model": {"decoder": {"inputs": {}, "outputs": {}}}}
    model.update_genai_config(config)
    assert config["model"]["decoder"]["inputs"]["past_state_names"] == [
        names[0] for names in model.compression_state_names
    ]
    assert config["model"]["decoder"]["outputs"]["present_state_names"] == [
        names[1] for names in model.compression_state_names
    ]


def test_emits_compressed_attention_contract():
    model = _make_builder(["sliding_attention"])
    model.make_value("query", ir.DataType.FLOAT, ["batch_size", 2, "sequence_length", 6])
    model.make_value("local_kv", ir.DataType.FLOAT, ["batch_size", 1, "total_sequence_length", 6])
    model.make_value("sinks", ir.DataType.FLOAT, [2])

    model.make_compressed_attention(0, "query", "local_kv", "", "", "", "sinks")

    proto = ir.to_proto(model.model)
    node = next(node for node in proto.graph.node if node.op_type == "CompressedAttention")
    assert (len(node.input), len(node.output)) == (6, 1)


def test_emits_hyper_connection_and_head_contracts():
    model = _make_builder(["sliding_attention"])
    connection = SimpleNamespace(
        fn=SimpleNamespace(data=torch.randn(8, 16)),
        base=SimpleNamespace(data=torch.randn(8)),
        scale=SimpleNamespace(data=torch.randn(3)),
    )
    head = SimpleNamespace(
        hc_fn=SimpleNamespace(data=torch.randn(2, 16)),
        hc_base=SimpleNamespace(data=torch.randn(2)),
        hc_scale=SimpleNamespace(data=torch.randn(1)),
    )

    model.make_hyper_connection(0, "attn", connection, "hc_streams")
    model.make_hyper_connection_mix(
        0,
        "ffn",
        connection,
        _norm(8),
        "hidden",
        "hc_streams",
        "post",
        "comb",
    )
    model.make_hc_head(head, "hc_streams")

    proto = ir.to_proto(model.model)
    contracts = {
        node.op_type: (len(node.input), len(node.output)) for node in proto.graph.node if node.domain == "com.microsoft"
    }
    assert contracts["HyperConnection"] == (4, 3)
    assert contracts["HyperConnectionMix"] == (8, 4)
    assert contracts["HyperHead"] == (4, 1)
    mix_weight = next(
        initializer
        for initializer in proto.graph.initializer
        if initializer.name == "model.layers.0.ffn_hc.fn"
    )
    assert list(mix_weight.dims) == [16, 8]


def test_casts_hyper_connection_state_between_activation_and_mix_types():
    model = _make_builder(["sliding_attention"])
    model.io_dtype = ir.DataType.FLOAT16
    model.make_value("post_fp16", ir.DataType.FLOAT16, ["batch_size", "sequence_length", 2])
    model.make_value("post_fp32", ir.DataType.FLOAT, ["batch_size", "sequence_length", 2])

    fp32_state = model.cast_hyper_mix_state(
        "post_fp16", ["batch_size", "sequence_length", 2], ir.DataType.FLOAT
    )
    fp16_state = model.cast_hyper_mix_state(
        "post_fp32", ["batch_size", "sequence_length", 2], ir.DataType.FLOAT16
    )

    proto = ir.to_proto(model.model)
    casts = {node.output[0]: node for node in proto.graph.node if node.op_type == "Cast"}
    assert fp32_state in casts
    assert fp16_state in casts
