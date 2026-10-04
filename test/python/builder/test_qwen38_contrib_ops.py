# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from types import MethodType, SimpleNamespace
import json

import onnx
import onnx_ir as ir
import pytest
import torch

from models.builders.base import Model
from models.builders.qwen3_5 import Qwen35MoETextModel
from models.builders.qwen3_8 import (
    Qwen4ExpEmbeddingModel,
    Qwen4ExpEngramModel,
    Qwen4ExpModel,
    Qwen4ExpMTPTextModel,
    Qwen4ExpTextModel,
)


def test_composite_float_checkpoint_loader_and_sparse_bindings(monkeypatch, tmp_path):
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config={
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "mtp_num_hidden_layers": 0,
        },
    )
    original_layer_types = list(config.text_config.layer_types)
    model = Qwen4ExpModel(
        config, ir.DataType.FLOAT16, ir.DataType.INT4, "cuda", str(tmp_path), {"exclude_mtp": True}
    )
    loaded = object()
    monkeypatch.setattr(Qwen4ExpForConditionalGeneration, "from_pretrained", lambda *args, **kwargs: loaded)
    assert model.decoder.load_weights("unused") is loaded
    assert set(model.decoder.input_names["past_key_values.key"]) == {3}
    assert set(model.decoder.input_names["past.indexer"]) == {3}
    assert set(model.decoder.output_names["present.key"]) == {3}
    assert config.text_config.layer_types == original_layer_types


def test_paged_mtp_config_preserves_engine_state_bindings(tmp_path):
    from transformers import Qwen4ExpConfig

    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config={
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "mtp_num_hidden_layers": 1,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 256,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_budget": 2048,
        },
    )
    model = Qwen4ExpModel(
        config,
        ir.DataType.FLOAT16,
        ir.DataType.INT4,
        "cuda",
        str(tmp_path),
        {"use_paged_attention": True, "state_update_capacity": 4},
    )
    groups = model.decoder.make_decoder_state_groups({}, {})
    path = tmp_path / "genai_config.json"
    path.write_text(json.dumps({
        "model": {"decoder": {"outputs": {}, "state_groups": groups}},
        "engine": {"dynamic_batching": {}},
    }))
    model.add_mtp_to_genai_config(str(tmp_path))
    result = json.loads(path.read_text())
    assert result["model"]["mtp"]["enabled"] is True
    assert result["engine"]["dynamic_batching"]["prefix_caching"] is False
    assert result["model"]["decoder"]["state_groups"] == groups
    assert result["model"]["mtp"]["inputs"]["past_indexer_names"] == "past.%d.indexer_key"
    assert model.decoder.input_shapes["input_ids"] == ["num_tokens"]
    assert model.decoder.input_shapes["inputs_embeds"] == ["num_tokens", model.decoder.hidden_size]
    assert model.mtp.input_shapes["hidden_states"] == ["num_tokens", model.decoder.hc_hidden_size]


def record_calls(model, method_names):
    model.calls = []

    def make_recorder(method_name):
        def record(self, *args, **kwargs):
            self.calls.append((method_name, args, kwargs))
            if method_name in {"make_matmul", "make_matmul_nbits"}:
                return args[1]

        return record

    for method_name in method_names:
        setattr(model, method_name, MethodType(make_recorder(method_name), model))


def test_engine_embedding_has_only_packed_ids_and_gather():
    model = Qwen4ExpEmbeddingModel(
        SimpleNamespace(), torch.zeros(8, 4), ir.DataType.FLOAT16, use_paged_attention=True
    )
    assert [value.name for value in model.graph.inputs] == ["input_ids"]
    assert [str(dim) for dim in model.graph.inputs[0].shape] == ["num_tokens"]
    assert [str(dim) for dim in model.graph.outputs[0].shape] == ["num_tokens", "4"]
    assert [node.op_type for node in model.graph] == ["Gather"]


def make_sparse_model(paged):
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = paged
    model.hidden_rows_dim = "num_tokens"
    model.io_dtype = ir.DataType.FLOAT16
    model.num_attn_heads = 8
    model.num_kv_heads = 2
    model.head_size = 16
    model.indexer_num_heads = 4
    model.indexer_kv_heads = 1
    model.indexer_head_dim = 16
    model.indexer_budget = 32
    model.indexer_compress_ratio = 4
    model.indexer_state_capacity = 32
    model.fixed_indexer_cache = not paged
    model.layernorm_attrs = {"epsilon": 1e-6}
    model.rope_attrs = {"interleaved": 0, "cast": {"use_fp32": True}}
    model.attention_attrs = {
        "q_path": "query",
        "k_path": "key",
        "v_path": "value",
        "scale": 0.25,
        "softcap": None,
        "qk_norm_epsilon": 1e-6,
    }
    model.mask_attrs = {"seqlens_k": "seqlens", "total_seq_len": "total_length"}
    model.input_names = {
        "attention_mask": "attention_mask",
        "position_ids": "position_ids",
        "past_sequence_length": "past_sequence_length",
        "past.indexer": {3: "past.3.indexer_key"},
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "block_table": "block_table",
        "attention_metadata": "attention_metadata",
        "past.indexer_kv_buffer": {3: "past.3.indexer_kv_buffer"},
        "past.indexer_state_lengths": {3: "past.3.indexer_state_lengths"},
    }
    model.output_names = {
        "present.indexer": {3: "present.3.indexer_key"},
        "present.indexer_kv_buffer": {3: "present.3.indexer_kv_buffer"},
        "present.indexer_state_lengths": {3: "present.3.indexer_state_lengths"},
    }
    record_calls(
        model,
        [
            "make_initializer",
            "make_matmul",
            "make_split",
            "make_reshape",
            "make_cast",
            "make_gather",
            "make_node",
            "make_value",
        ],
    )

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        if self.use_paged_attention:
            self.attention_attrs.update(q_path="query", k_path="key", v_path="value")
        else:
            self.attention_attrs.update(q_path="packed_qkv", k_path="", v_path="")
        if kwargs.get("indexer_proj") is not None:
            self.attention_attrs["indexer_qk_path"] = "index_qk"

    model.make_attention_input_proj = MethodType(make_attention_input_proj, model)
    model.get_qk_norm_weight_names = MethodType(lambda self, layer_id: ("q_norm", "k_norm"), model)
    model.make_rotary_embedding_caches = MethodType(lambda self: ("cos_cache", "sin_cache"), model)
    model.make_key_value_cache_names = MethodType(
        lambda self, layer_id: ("past_key", "past_value", "present_key", "present_value"), model
    )
    model.make_selected_counts = MethodType(
        lambda self, layer_id, selected, capacity, packed: "selected_counts", model
    )
    model.make_attention_output_proj = MethodType(lambda self, layer_id, attention, root_input: None, model)
    return model


def make_attention():
    norm = SimpleNamespace(weight=torch.zeros(16))
    indexer = SimpleNamespace(
        index_qk_proj=SimpleNamespace(),
        q_layernorm=norm,
        k_layernorm=norm,
    )
    return SimpleNamespace(q_norm=norm, k_norm=norm, indexer=indexer)


def test_dense_cuda_indexer_cache_shapes_are_symbolic(monkeypatch):
    def initialize_parent(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.io_dtype = io_dtype
        self.ep = ep
        self.use_paged_attention = False
        self.hidden_size = 16
        self.layer_types = ["qwen_sparse_attention"]
        self.input_names = {"position_ids": "position_ids"}
        self.input_types = {}
        self.input_shapes = {}
        self.output_names = {}
        self.output_types = {}
        self.output_shapes = {}
        self.rope_attrs = {"cast": {"use_fp32": True}}
        self.model = SimpleNamespace(metadata_props={})

    monkeypatch.setattr(Qwen35MoETextModel, "__init__", initialize_parent)
    config = SimpleNamespace(
        hc_count=1,
        ple_layer_ids=[],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=2,
        eos_token_id=0,
        heads_per_ngram=1,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=32,
        indexer_compress_ratio=4,
        output_gate_type=None,
        hidden_act="silu",
    )

    model = Qwen4ExpTextModel(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {})

    assert model.fixed_indexer_cache
    assert model.input_shapes["past.indexer"] == ["batch_size", "past_sequence_length", 16]
    assert model.output_shapes["present.indexer"] == ["batch_size", "total_sequence_length", 16]
    assert model.input_shapes["past_sequence_length"] == [1]


def test_paged_indexer_state_shapes_are_fixed_capacity(monkeypatch):
    def initialize_parent(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.io_dtype = io_dtype
        self.ep = ep
        self.use_paged_attention = True
        self.hidden_rows_dim = "num_tokens"
        self.context_length = 128
        self.hidden_size = 16
        self.layer_types = ["qwen_sparse_attention"]
        self.input_names = {}
        self.input_types = {}
        self.input_shapes = {"attention_metadata": [1]}
        self.output_names = {}
        self.output_types = {}
        self.output_shapes = {}
        self.rope_attrs = {"cast": {"use_fp32": True}}
        self.model = SimpleNamespace(metadata_props={})

    monkeypatch.setattr(Qwen35MoETextModel, "__init__", initialize_parent)
    config = SimpleNamespace(
        hc_count=1,
        ple_layer_ids=[],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=2,
        eos_token_id=0,
        heads_per_ngram=1,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=32,
        indexer_compress_ratio=4,
        output_gate_type=None,
        hidden_act="silu",
    )

    model = Qwen4ExpTextModel(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {})

    assert not model.fixed_indexer_cache
    assert "position_ids" not in model.input_names
    assert model.input_shapes["attention_metadata"] == [5]
    assert model.indexer_state_capacity == 32
    assert model.input_shapes["past.indexer"] == ["batch_size", 32, 16]
    assert model.input_shapes["past.indexer_kv_buffer"] == ["batch_size", 7, 16]
    assert model.input_shapes["past.indexer_state_lengths"] == ["batch_size", 2]
    assert model.output_shapes["present.indexer"] == model.input_shapes["past.indexer"]
    assert model.output_shapes["present.indexer_kv_buffer"] == model.input_shapes["past.indexer_kv_buffer"]
    assert model.output_shapes["present.indexer_state_lengths"] == model.input_shapes["past.indexer_state_lengths"]
    assert "qwen4_exp.selected_index_names" not in model.model.metadata_props
    assert "qwen4_exp.selected_count_names" not in model.model.metadata_props

    config.ple_layer_ids = [2]
    external_model = Qwen4ExpTextModel(
        config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None,
        {"external_engram": True, "state_update_capacity": 7},
    )
    assert "state_update.ple_tokens" in external_model.output_names
    assert external_model.output_names["state_update.ple_conv_value"] == {1: "state_update.1.ple_conv_value"}


def test_qwen38_ple_conv_state_layout_tracks_packed_execution(monkeypatch):
    def initialize_parent(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.io_dtype = io_dtype
        self.ep = ep
        self.use_paged_attention = extra_options["use_paged_attention"]
        self.hidden_rows_dim = "num_tokens"
        self.context_length = 128
        self.hidden_size = 16
        self.layer_types = ["linear_attention"]
        self.input_names = {}
        self.input_types = {}
        self.input_shapes = {}
        self.output_names = {}
        self.output_types = {}
        self.output_shapes = {}
        self.rope_attrs = {"cast": {"use_fp32": True}}
        self.model = SimpleNamespace(metadata_props={})

    monkeypatch.setattr(Qwen35MoETextModel, "__init__", initialize_parent)
    config = SimpleNamespace(
        hc_count=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=4,
        ngram_size=3,
        eos_token_id=1,
        heads_per_ngram=2,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=32,
        indexer_compress_ratio=4,
        output_gate_type=None,
        hidden_act="silu",
    )

    dense = Qwen4ExpTextModel(
        config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {"use_paged_attention": False}
    )
    packed = Qwen4ExpTextModel(
        config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {"use_paged_attention": True}
    )

    assert dense.input_shapes["past.ple_conv"] == ["batch_size", 9, 32]
    assert packed.input_shapes["past.ple_conv"] == ["batch_size", 32, 9]
    assert dense.output_shapes["present.ple_conv"] == dense.input_shapes["past.ple_conv"]
    assert packed.output_shapes["present.ple_conv"] == packed.input_shapes["past.ple_conv"]


def test_qwen38_gated_delta_net_expansion_emits_linear_attention():
    model = object.__new__(Qwen4ExpTextModel)
    model.linear_num_key_heads = 2
    model.linear_num_value_heads = 3
    model.input_names = {"past.recurrent": {4: "past"}}
    model.output_names = {"present.recurrent": {4: "present"}}
    record_calls(model, ["make_linear_attention"])
    model.make_gated_delta_net_gates_expansion = MethodType(
        lambda self, *args: ("q", "k", "v", "decay", "beta"), model
    )

    output = model.make_gated_delta_net_expansion(4, SimpleNamespace(), "conv", "b", "a")

    assert output == "/model/layers.4/linear_attn/LinearAttention/output_0"
    call = model.calls[0]
    assert call[0] == "make_linear_attention"
    assert call[2] == {
        "q_path": "q",
        "k_path": "k",
        "v_path": "v",
        "past_recurrent_state": "past",
        "present_recurrent_state": "present",
        "decay": "decay",
        "beta": "beta",
        "q_num_heads": 2,
        "kv_num_heads": 3,
        "update_rule": "gated_delta",
        "scale": 1.0,
    }


def test_qwen38_dense_linear_attention_layer_always_emits_gated_delta_net():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.linear_conv_dim = 8
    model.input_names = {"past.conv": {2: "past.conv"}}
    model.output_names = {"present.conv": {2: "present.conv"}}
    calls = []
    model.make_linear_attention_input_proj = MethodType(
        lambda self, *args: ("z", "b", "a", "conv_input", "conv_weight"), model
    )
    model.make_initializer = MethodType(lambda self, *args, **kwargs: None, model)
    model.make_causal_conv_with_state = MethodType(lambda self, *args, **kwargs: None, model)
    model.make_transpose = MethodType(lambda self, *args, **kwargs: None, model)
    model.make_gated_delta_net_layer = MethodType(
        lambda self, *args: calls.append(("gated_delta_net", args)) or "gdn_output", model
    )
    model.make_linear_attention_output_proj = MethodType(
        lambda self, *args: calls.append(("output_proj", args)), model
    )
    model.make_linear_attention = MethodType(
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected expansion")), model
    )
    attention = SimpleNamespace()

    model.make_qwen_gated_delta_net(2, attention, "hidden_states")

    assert calls == [
        (
            "gated_delta_net",
            (
                2,
                attention,
                "/model/layers.2/linear_attn/conv_out/Transpose/output_0",
                "b",
                "a",
            ),
        ),
        ("output_proj", (2, attention, "gdn_output", "z")),
    ]


def test_dense_gated_delta_net_emits_compact_state_update():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    record_calls(model, ["make_cast", "make_node", "make_value"])

    model.make_gated_delta_net(
        "/gdn",
        q_path="qkv",
        k_path="",
        v_path="",
        decay="decay",
        beta="beta",
        gate_shape=["batch_size", "sequence_length", 48],
        initial_state="past",
        final_state="present",
        state_shape=["batch_size", 48, 128, 128],
        output_shape=["batch_size", "sequence_length", 48, 128],
        a_log="a_log",
        dt_bias="dt_bias",
        state_update_capacity=1,
        state_update_capture_count="capture_count",
        state_update_active="capture_active",
        state_update_capsule="state_update.0.recurrent_capsule",
        state_update_capsule_shape=["batch_size", 7216],
    )

    node_call = next(call for call in model.calls if call[0] == "make_node")
    assert node_call[1][0] == "GatedDeltaNet"
    assert node_call[2]["inputs"][9:] == ["capture_count", "capture_active"]
    assert node_call[2]["outputs"] == [
        "/gdn/output_0",
        "present",
        "state_update.0.recurrent_capsule",
    ]
    assert node_call[2]["state_update_capacity"] == 1


def test_qwen_attention_packs_gated_qkv_before_splitting():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.q_size = 128
    model.kv_size = 32
    model.num_attn_heads = 4
    model.head_size = 32
    model.attention_attrs = {
        "use_matmul_in_attn": False,
        "use_packed_matmul": True,
        "q_norm": True,
        "k_norm": True,
    }
    projection = lambda output_size: SimpleNamespace(  # noqa: E731
        weight=torch.zeros(output_size, 64, dtype=torch.float16), bias=None
    )
    attention = SimpleNamespace(q_proj=projection(256), k_proj=projection(32), v_proj=projection(32))
    record_calls(model, ["make_attention_unpacked", "make_split", "make_reshape"])

    def make_packed_matmul(self, q_proj, k_proj, v_proj, basename, root_input):
        self.calls.append(("make_packed_matmul", (q_proj, k_proj, v_proj, basename, root_input), {}))
        return basename

    model.make_packed_matmul = MethodType(make_packed_matmul, model)

    model.make_attention_input_proj(3, attention, "hidden_states")

    packed = next(call for call in model.calls if call[0] == "make_packed_matmul")
    assert packed[1][4] == "hidden_states"
    qkv_split = next(call for call in model.calls if call[0] == "make_split")
    assert qkv_split[2]["inputs"][1] == "/model/constants/INT64/[256, 32, 32]"
    assert qkv_split[2]["shapes"] == [
        ["batch_size", "sequence_length", 256],
        ["batch_size", "sequence_length", 32],
        ["batch_size", "sequence_length", 32],
    ]
    assert model.q_size == 128
    assert model.attention_attrs["q_path"] == "/model/layers.3/attn/q_proj/Reshape/output_0"
    assert model.attention_attrs["k_path"] == "/model/layers.3/attn/qkv_proj/Split/output_1"
    assert model.attention_attrs["v_path"] == "/model/layers.3/attn/qkv_proj/Split/output_2"


def test_qwen_linear_attention_packs_qkv_and_z():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.make_hidden_state_shape = MethodType(
        lambda self, last_dim: ["batch_size", "sequence_length", last_dim], model
    )
    record_calls(model, ["make_matmul", "make_split"])
    attention = SimpleNamespace(
        in_proj_qkv=SimpleNamespace(weight=torch.ones((10, 8))),
        in_proj_z=SimpleNamespace(weight=torch.full((6, 8), 2.0)),
    )

    qkv_name, z_name = model.make_linear_attention_qkv_z_proj(3, attention, "hidden_states")

    matmul = next(call for call in model.calls if call[0] == "make_matmul")
    assert matmul[1][0].weight.shape == (16, 8)
    torch.testing.assert_close(matmul[1][0].weight[:10], attention.in_proj_qkv.weight)
    torch.testing.assert_close(matmul[1][0].weight[10:], attention.in_proj_z.weight)
    assert matmul[1][1:] == ("/model/layers.3/linear_attn/qkv_z_proj/MatMul", "hidden_states")
    split = next(call for call in model.calls if call[0] == "make_split")
    assert split[1][1][1] == "/model/constants/INT64/[10, 6]"
    assert split[1][2] == [
        "/model/layers.3/linear_attn/qkv_proj/MatMul/output_0",
        "/model/layers.3/linear_attn/z_proj/MatMul/output_0",
    ]
    assert qkv_name == "/model/layers.3/linear_attn/qkv_proj/MatMul"
    assert z_name == "/model/layers.3/linear_attn/z_proj/MatMul"


def test_qwen_linear_attention_packs_prequantized_qkv_and_z():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.make_hidden_state_shape = MethodType(
        lambda self, last_dim: ["batch_size", "sequence_length", last_dim], model
    )
    record_calls(model, ["make_matmul_nbits", "make_split"])

    def projection(out_features, value):
        return SimpleNamespace(
            qweight=torch.full((out_features, 2, 2), value, dtype=torch.int32),
            scales=torch.full((out_features, 2), float(value)),
            qzeros=torch.full((out_features, 2), value, dtype=torch.int32),
            g_idx=None,
            in_features=8,
            out_features=out_features,
            bits=4,
            group_size=4,
        )

    attention = SimpleNamespace(in_proj_qkv=projection(10, 1), in_proj_z=projection(6, 2))
    model.make_linear_attention_qkv_z_proj(3, attention, "hidden_states")

    matmul = next(call for call in model.calls if call[0] == "make_matmul_nbits")
    packed = matmul[1][0]
    assert packed.qweight.shape == (16, 2, 2)
    assert packed.scales.shape == (16, 2)
    assert packed.qzeros.shape == (16, 2)
    assert packed.out_features == 16
    assert torch.all(packed.qweight[:10] == 1)
    assert torch.all(packed.qweight[10:] == 2)
    assert matmul[1][1:] == ("/model/layers.3/linear_attn/qkv_z_proj/MatMul", "hidden_states")


def test_qwen_linear_attention_packs_a_and_b():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.make_hidden_state_shape = MethodType(
        lambda self, last_dim: ["batch_size", "sequence_length", last_dim], model
    )
    record_calls(model, ["make_matmul", "make_split", "exclude_node_from_quantization"])
    attention = SimpleNamespace(
        in_proj_a=SimpleNamespace(weight=torch.ones((3, 8))),
        in_proj_b=SimpleNamespace(weight=torch.full((5, 8), 2.0)),
    )

    b_name, a_name = model.make_linear_attention_a_b_proj(3, attention, "hidden_states")

    matmul = next(call for call in model.calls if call[0] == "make_matmul")
    assert matmul[1][0].weight.shape == (8, 8)
    torch.testing.assert_close(matmul[1][0].weight[:3], attention.in_proj_a.weight)
    torch.testing.assert_close(matmul[1][0].weight[3:], attention.in_proj_b.weight)
    assert matmul[1][1:] == ("/model/layers.3/linear_attn/a_b_proj/MatMul", "hidden_states")
    split = next(call for call in model.calls if call[0] == "make_split")
    assert split[1][1][1] == "/model/constants/INT64/[3, 5]"
    assert split[1][2] == [
        "/model/layers.3/linear_attn/a_proj/MatMul/output_0",
        "/model/layers.3/linear_attn/b_proj/MatMul/output_0",
    ]
    assert ("exclude_node_from_quantization", ("/model/layers.3/linear_attn/a_b_proj/MatMul",), {}) in model.calls
    assert b_name == "/model/layers.3/linear_attn/b_proj/MatMul"
    assert a_name == "/model/layers.3/linear_attn/a_proj/MatMul"


def test_qwen_attention_emits_packed_qkv_with_separate_gate_and_indexer_qk():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.q_size = 128
    model.kv_size = 32
    model.num_attn_heads = 4
    model.head_size = 32
    model.attention_attrs = {"use_packed_matmul": True}
    projection = lambda output_size, offset=0: SimpleNamespace(  # noqa: E731
        weight=torch.arange(offset, offset + output_size, dtype=torch.float16).unsqueeze(1).expand(-1, 64),
        bias=None,
        out_features=output_size,
    )
    attention = SimpleNamespace(q_proj=projection(256), k_proj=projection(32, 300), v_proj=projection(32, 400))
    indexer_proj = projection(40, 500)
    record_calls(model, ["make_matmul", "make_split"])

    def make_packed_matmul_class(self, *projections):
        self.calls.append(("make_packed_matmul_class", projections, {}))
        return SimpleNamespace()

    model.make_packed_matmul_class = MethodType(make_packed_matmul_class, model)

    model.make_attention_input_proj(3, attention, "hidden_states", indexer_proj=indexer_proj)

    packed = next(call for call in model.calls if call[0] == "make_packed_matmul_class")
    q_proj, k_proj, v_proj = packed[1]
    assert (k_proj, v_proj) == (attention.k_proj, attention.v_proj)
    expected_q_rows = torch.cat([torch.arange(head * 64, head * 64 + 32) for head in range(4)])
    torch.testing.assert_close(q_proj.weight[:, 0], expected_q_rows.to(torch.float16))

    matmuls = [call for call in model.calls if call[0] == "make_matmul"]
    assert [call[1][1] for call in matmuls] == [
        "/model/layers.3/attn/qkv_proj/MatMul",
        "/model/layers.3/attn/gate_proj/MatMul",
        "/model/layers.3/attn/indexer/index_qk_proj/MatMul",
    ]
    expected_gate_rows = expected_q_rows + 32
    torch.testing.assert_close(matmuls[1][1][0].weight[:, 0], expected_gate_rows.to(torch.float16))
    assert not any(call[0] == "make_split" for call in model.calls)
    assert model.attention_attrs["q_path"] == "/model/layers.3/attn/qkv_proj/MatMul/output_0"
    assert model.attention_attrs["k_path"] == ""
    assert model.attention_attrs["v_path"] == ""
    assert model.attention_attrs["gate_path"] == "/model/layers.3/attn/gate_proj/MatMul/output_0"
    assert model.attention_attrs["indexer_qk_path"] == (
        "/model/layers.3/attn/indexer/index_qk_proj/MatMul/output_0"
    )


def test_packed_matmul_class_concatenates_four_int4_projections():
    model = object.__new__(Qwen4ExpTextModel)
    model.onnx_dtype = ir.DataType.INT4

    def projection(output_size, value):
        return SimpleNamespace(
            qweight=torch.full((output_size, 2, 16), value, dtype=torch.uint8),
            scales=torch.full((output_size, 2), float(value), dtype=torch.float16),
            qzeros=torch.zeros((output_size, 2), dtype=torch.uint8),
            g_idx=None,
            in_features=64,
            out_features=output_size,
            bits=4,
            group_size=32,
        )

    projections = [projection(size, index + 1) for index, size in enumerate((8, 2, 2, 4))]

    packed = model.make_packed_matmul_class(*projections)

    assert packed.qweight.shape == (16, 2, 16)
    assert packed.scales.shape == (16, 2)
    assert packed.qzeros.shape == (16, 2)
    assert packed.in_features == 64
    assert packed.out_features == 16
    assert packed.bits == 4
    assert packed.group_size == 32
    assert [packed.qweight[offset, 0, 0].item() for offset in (0, 8, 10, 12)] == [1, 2, 3, 4]


def test_qwen_attention_deinterleaves_prequantized_q_and_gate_rows():
    model = object.__new__(Qwen4ExpTextModel)
    projection = SimpleNamespace(
        qweight=torch.arange(8).reshape(8, 1, 1),
        scales=torch.arange(8).reshape(8, 1),
        qzeros=torch.arange(8).reshape(8, 1),
        g_idx=None,
        in_features=4,
        out_features=8,
        bits=4,
        group_size=4,
    )

    query = model.select_projection_outputs(projection, torch.tensor([0, 1, 4, 5]))
    gate = model.select_projection_outputs(projection, torch.tensor([2, 3, 6, 7]))

    assert query.out_features == gate.out_features == 4
    assert query.qweight.flatten().tolist() == [0, 1, 4, 5]
    assert query.scales.flatten().tolist() == [0, 1, 4, 5]
    assert query.qzeros.flatten().tolist() == [0, 1, 4, 5]
    assert gate.qweight.flatten().tolist() == [2, 3, 6, 7]
    assert gate.scales.flatten().tolist() == [2, 3, 6, 7]
    assert gate.qzeros.flatten().tolist() == [2, 3, 6, 7]


def test_qwen_attention_projection_selection_does_not_mutate_torch_module():
    model = object.__new__(Qwen4ExpTextModel)
    projection = torch.nn.Linear(4, 8, bias=False)
    projection.weight.data.copy_(torch.arange(32).reshape(8, 4))
    original = projection.weight.detach().clone()

    query = model.select_projection_outputs(projection, torch.tensor([0, 1, 4, 5]))
    gate = model.select_projection_outputs(projection, torch.tensor([2, 3, 6, 7]))

    assert torch.equal(projection.weight, original)
    assert torch.equal(query.weight, original[[0, 1, 4, 5]])
    assert torch.equal(gate.weight, original[[2, 3, 6, 7]])


def test_qwen_moe_emits_separate_shared_gate_up_and_router_matmuls():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.hidden_size = 64
    model.intermediate_size = 256
    model.shared_expert_intermediate_size = 16
    model.moe_attrs = {"num_experts": 32}
    model.mlp_attrs = {}
    model.make_hidden_state_shape = MethodType(
        lambda self, last_dim: ["batch_size", "sequence_length", last_dim], model
    )
    projection = lambda output_size: SimpleNamespace(out_features=output_size, bias=None)  # noqa: E731
    shared_expert = SimpleNamespace(
        gate_proj=projection(16),
        up_proj=projection(16),
        down_proj=projection(64),
    )
    moe = SimpleNamespace(shared_expert=shared_expert, gate=projection(32))
    record_calls(
        model,
        ["make_split", "make_reshape", "make_activation", "make_mul", "make_matmul", "make_sigmoid"],
    )

    model.make_moe_router(3, moe, "hidden_states")
    output, gate = model.make_shared_expert(3, shared_expert, projection(1), "hidden_states")

    matmuls = [call[1][1] for call in model.calls if call[0] == "make_matmul"]
    assert matmuls == [
        "/model/layers.3/moe/router/MatMul",
        "/model/layers.3/mlp/gate_proj/MatMul",
        "/model/layers.3/mlp/up_proj/MatMul",
        "/model/layers.3/mlp/down_proj/MatMul",
        "/model/layers.3/shared_expert_gate/MatMul",
    ]
    assert not any(call[0] == "make_split" for call in model.calls)
    router_reshape = next(call for call in model.calls if call[0] == "make_reshape")
    assert router_reshape[1][1][0] == "/model/layers.3/moe/router/MatMul/output_0"
    mul = next(call for call in model.calls if call[0] == "make_mul")
    assert mul[1][1][1] == "/model/layers.3/mlp/up_proj/MatMul/output_0"
    assert output == "/model/layers.3/mlp/down_proj/MatMul/output_0"
    assert gate == "/model/layers.3/shared_expert_gate/Sigmoid/output_0"


def emitted_nodes(model):
    return [(args[0], kwargs) for method, args, kwargs in model.calls if method == "make_node"]


def test_mtp_residual_linear_shared_preserves_hyper_connection_shape():
    model = object.__new__(Qwen4ExpMTPTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.use_paged_attention = False
    model.hidden_size = 8
    model.hc_count = 4
    model.hc_hidden_size = 32
    model.input_names = {"input_ids": "input_ids", "hidden_states": "hidden_states"}
    model.mtp_weights = SimpleNamespace(
        embedding=SimpleNamespace(weight=torch.ones((16, 8))),
        fc_embedding=SimpleNamespace(weight=torch.ones((8, 8))),
        fc_hidden=SimpleNamespace(weight=torch.ones((8, 8))),
        pre_fc_norm_embedding=SimpleNamespace(weight=torch.ones(8)),
        pre_fc_norm_hidden=SimpleNamespace(weight=torch.ones(32)),
    )
    record_calls(
        model,
        ["make_initializer", "make_node", "make_value", "make_reshape", "make_matmul", "make_unsqueeze", "make_add"],
    )
    model.make_hidden_state_shape = MethodType(
        lambda self, last_dim=None: ["batch_size", "sequence_length", last_dim or self.hidden_size], model
    )
    model.make_offset_rmsnorm = MethodType(lambda self, *args: "embedding_norm", model)
    model.make_branchwise_rms_norm = MethodType(lambda self, *args: "hidden_norm", model)

    output = model.make_mtp_input_projection()

    hidden_projection = next(
        call for call in model.calls if call[0] == "make_matmul" and call[1][1].endswith("fc_hidden/MatMul")
    )
    assert hidden_projection[2]["output_shape"] == ["batch_size", "sequence_length", 4, 8]
    fusion = next(call for call in model.calls if call[0] == "make_add")
    assert fusion[1][3] == ["batch_size", "sequence_length", 4, 8]
    assert output == "/model/mtp/input_fusion/Reshape/output_0"


def test_paged_branchwise_norm_emits_fused_op():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = True
    model.io_dtype = ir.DataType.FLOAT16
    model.hc_count = 2
    model.layernorm_attrs = {"epsilon": 1e-6}
    record_calls(model, ["make_initializer", "make_node", "make_value"])

    model.make_branchwise_rms_norm("/norm", "hidden_states", SimpleNamespace(weight=torch.zeros(16)), 8)

    initializers = [call for call in model.calls if call[0] == "make_initializer"]
    assert initializers[0][1][0].shape == (16,)
    assert torch.equal(initializers[0][1][0], torch.ones(16))
    norm = emitted_nodes(model)[0]
    assert norm[0] == "BranchwiseRMSNorm"
    assert norm[1]["inputs"] == ["hidden_states", "norm.weight"]
    assert norm[1]["domain"] == "com.microsoft"
    assert norm[1]["num_branches"] == 2
    assert norm[1]["epsilon"] == 1e-6


def test_dense_branchwise_norm_emits_fused_op():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.BFLOAT16
    model.hc_count = 4
    model.layernorm_attrs = {"epsilon": 1e-6}
    record_calls(model, ["make_initializer", "make_node", "make_value"])

    output = model.make_branchwise_rms_norm(
        "/norm", "hidden_states", SimpleNamespace(weight=torch.zeros(32)), 8
    )

    assert output == "/norm/output_0"
    norm = emitted_nodes(model)[0]
    assert norm[0] == "BranchwiseRMSNorm"
    assert norm[1]["num_branches"] == 4


def test_hyper_connection_emits_fused_ops():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.hc_count = 4
    model.hidden_size = 8
    model.hc_hidden_size = 32
    record_calls(
        model,
        [
            "make_matmul",
            "make_div",
            "make_sigmoid",
            "make_mul",
            "make_reshape",
            "make_split",
            "make_node",
            "make_value",
        ],
    )
    model.make_branchwise_rms_norm = MethodType(lambda self, *args: "normalized", model)
    weights = SimpleNamespace(
        input_mix_weight_down=SimpleNamespace(out_features=4, weight=torch.ones((4, 8))),
        input_mix_weight_up=SimpleNamespace(),
        block_inject_weight=SimpleNamespace(out_features=4, weight=torch.full((4, 8), 2.0)),
        hc_norm=SimpleNamespace(),
    )

    model.make_hyper_connection_mix(0, weights, "hidden_states", "attn")

    nodes = emitted_nodes(model)
    assert [op_type for op_type, _ in nodes] == ["ScaledSiLU", "HyperConnectionPreMix"]
    assert nodes[0][1]["inputs"] == ["/model/layers.0/attn_hyper_connection/input_mix_weight_down/MatMul/output_0"]
    assert nodes[0][1]["alpha"] == 0.25
    assert nodes[1][1]["inputs"] == [
        "normalized",
        "/model/layers.0/attn_hyper_connection/input_mix_weight_up/Sigmoid/output_0",
    ]
    assert nodes[1][1]["num_branches"] == 4
    assert nodes[1][1]["reduction_scale"] == 0.25
    matmuls = [call for call in model.calls if call[0] == "make_matmul"]
    assert matmuls[0][1][0].weight.shape == (8, 8)
    torch.testing.assert_close(matmuls[0][1][0].weight[:4], weights.input_mix_weight_down.weight)
    torch.testing.assert_close(matmuls[0][1][0].weight[4:], weights.block_inject_weight.weight)
    assert matmuls[0][1][1:] == (
        "/model/layers.0/attn_hyper_connection/input_mix_down_block_inject/MatMul",
        "normalized",
    )
    assert matmuls[1][1][2] == "/model/layers.0/attn_hyper_connection/input_mix_weight_down/SiLU/output_0"
    split = next(call for call in model.calls if call[0] == "make_split")
    assert split[1][1][1] == "/model/constants/INT64/[4, 4]"
    assert split[1][2] == [
        "/model/layers.0/attn_hyper_connection/input_mix_weight_down/MatMul/output_0",
        "/model/layers.0/attn_hyper_connection/block_inject_weight/MatMul/output_0",
    ]


def test_hyper_connection_injection_emits_fused_post_mix():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.hc_count = 4
    model.hidden_size = 8
    model.hc_hidden_size = 32
    record_calls(model, ["make_node", "make_value"])

    output = model.make_hyper_connection_injection(
        0, "block_output", "hyper_input", "injection_weights", "attn"
    )

    assert output == "/model/layers.0/attn_hyper_connection/injection/output_0"
    post_mix = emitted_nodes(model)[0]
    assert post_mix[0] == "HyperConnectionPostMix"
    assert post_mix[1]["inputs"] == ["hyper_input", "block_output", "injection_weights"]
    assert post_mix[1]["domain"] == "com.microsoft"
    assert post_mix[1]["num_branches"] == 4


def test_qwen_hyper_connection_expansions_emit_standard_onnx():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.FLOAT16
    model.hc_count = 4
    model.hidden_size = 8
    model.layernorm_attrs = {"epsilon": 1e-6}
    record_calls(
        model,
        [
            "make_add",
            "make_div",
            "make_initializer",
            "make_mul",
            "make_node",
            "make_reduce_mean",
            "make_reshape",
            "make_sigmoid",
            "make_unsqueeze",
            "make_value",
        ],
    )

    model.make_branchwise_rms_norm_expansion(
        "/norm", "streams", SimpleNamespace(weight=torch.zeros(32)), 8
    )
    model.make_scaled_silu_expansion("/silu", "down", ["batch_size", "sequence_length", 4], 0.25)
    model.make_hyper_connection_pre_mix_expansion(
        "/pre", "normalized", "pre_mix", ["batch_size", "sequence_length"], 8
    )
    model.make_hyper_connection_post_mix_expansion(
        "/post", "streams", "block_output", "post_mix", ["batch_size", "sequence_length"], 8
    )

    standard_nodes = [op_type for op_type, _ in emitted_nodes(model)]
    assert standard_nodes == ["SimplifiedLayerNormalization"]
    assert not {
        "BranchwiseRMSNorm",
        "ScaledSiLU",
        "HyperConnectionPreMix",
        "HyperConnectionPostMix",
    }.intersection(standard_nodes)
    assert [call[0] for call in model.calls].count("make_reshape") == 5
    assert [call[0] for call in model.calls].count("make_mul") == 4
    assert [call[0] for call in model.calls].count("make_reduce_mean") == 1
    assert [call[0] for call in model.calls].count("make_add") == 1


def test_gated_rms_norm_emits_configured_activation():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.output_gate_type = "sigmoid"
    record_calls(model, ["make_node", "make_value"])

    model.make_gated_rms_norm(
        "/gated_norm",
        "hidden_states",
        "norm_scale",
        "output_gate",
        ["num_tokens", 16],
        epsilon=1e-6,
    )

    gated_norm = emitted_nodes(model)[0]
    assert gated_norm[0] == "GatedRMSNorm"
    assert gated_norm[1]["inputs"] == ["hidden_states", "norm_scale", "output_gate"]
    assert gated_norm[1]["epsilon"] == 1e-6
    assert gated_norm[1]["activation"] == "sigmoid"


def test_dense_qwen_sparse_attention_emits_indexer_and_dynamic_executor():
    model = make_sparse_model(paged=False)

    model.make_qwen_sparse_attention(3, make_attention(), "hidden_states")

    nodes = emitted_nodes(model)
    op_types = [op_type for op_type, _ in nodes]
    assert "QwenSparseAttention" not in op_types
    assert op_types[-2:] == ["SparseAttentionIndexer", "DynamicSparseAttention"]

    indexer = nodes[-2][1]
    assert indexer["inputs"] == [
        "index_qk",
        "",
        "model.layers.3.attn.indexer.q_norm.weight",
        "model.layers.3.attn.indexer.k_norm.weight",
        "cos_cache",
        "sin_cache",
        "",
        "past.3.indexer_key",
        "",
        "",
        "",
        "",
        "past_sequence_length",
    ]
    assert indexer["outputs"] == [
        "/model/layers.3/attn/SparseAttentionIndexer/output_0",
        "present.3.indexer_key",
    ]
    assert indexer["policy_mode"] == "qsa"
    assert indexer["token_budget"] == 32
    assert indexer["compress_ratio"] == 4
    assert not any(op_type == "SimplifiedLayerNormalization" for op_type, _ in nodes)

    attention = nodes[-1][1]
    assert attention["inputs"][:3] == ["packed_qkv", "", ""]
    casts = [call for call in model.calls if call[0] == "make_cast"]
    assert not casts
    assert attention["inputs"][7:11] == [
        "/model/layers.3/attn/SparseAttentionIndexer/Flatten/output_0",
        "/model/layers.3/attn/SparseAttentionIndexer/CountsFlatten/output_0",
        "seqlens/output_0",
        "total_length/output_0",
    ]
    assert attention["inputs"][11:13] == ["cos_cache", "sin_cache"]
    assert attention["inputs"][13] == "/model/layers.3/attn/DynamicSparseAttention/position_ids/Gather/output_0"
    position_gather = next(
        call
        for call in model.calls
        if call[0] == "make_gather" and call[1][0].endswith("DynamicSparseAttention/position_ids/Gather")
    )
    assert position_gather[1][1] == ["position_ids", "/model/constants/INT64/0"]
    assert position_gather[1][3] == ["batch_size", "sequence_length"]
    assert position_gather[2]["axis"] == 0
    assert attention["outputs"] == [
        "/model/layers.3/attn/DynamicSparseAttention/output_0",
        "present_key",
        "present_value",
    ]
    assert attention["is_causal"] == 1
    assert attention["attention_mode"] == "selected_only"
    assert attention["selected_kv_source"] == "main"


def test_paged_qwen_sparse_attention_emits_packed_indexer_and_shared_attention_schema():
    model = make_sparse_model(paged=True)

    model.make_qwen_sparse_attention(3, make_attention(), "hidden_states")

    nodes = emitted_nodes(model)
    assert [op_type for op_type, _ in nodes] == ["PackedSparseAttentionIndexer", "SparsePagedAttention"]
    indexer = nodes[0][1]
    assert indexer["inputs"][:8] == [
        "/model/layers.3/attn/PackedSparseAttentionIndexer/query",
        "/model/layers.3/attn/PackedSparseAttentionIndexer/key",
        "model.layers.3.attn.indexer.q_norm.weight",
        "model.layers.3.attn.indexer.k_norm.weight",
        "cos_cache",
        "sin_cache",
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
    ]
    assert indexer["inputs"][12:] == [
        "past.3.indexer_key",
        "past.3.indexer_kv_buffer",
        "",
        "past.3.indexer_state_lengths",
    ]
    assert indexer["outputs"] == [
        "/model/layers.3/attn/PackedSparseAttentionIndexer/output_0",
        "/model/layers.3/attn/PackedSparseAttentionIndexer/output_1",
        "present.3.indexer_key",
        "present.3.indexer_kv_buffer",
        "",
        "present.3.indexer_state_lengths",
    ]
    assert indexer["state_capacity"] == 32
    assert indexer["token_budget"] == 32
    assert indexer["compress_ratio"] == 4

    split = next(call for call in model.calls if call[0] == "make_split")
    assert split[1][2] == [
        "/model/layers.3/attn/PackedSparseAttentionIndexer/query",
        "/model/layers.3/attn/PackedSparseAttentionIndexer/key",
    ]
    assert split[1][4] == [["num_tokens", 64], ["num_tokens", 16]]

    attention = nodes[1][1]
    assert len(attention["inputs"]) == 22
    assert attention["inputs"][5:11] == [
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
        "block_table",
        "",
        "/model/layers.3/attn/PackedSparseAttentionIndexer/output_0",
        "/model/layers.3/attn/PackedSparseAttentionIndexer/output_1",
    ]
    assert attention["inputs"][14:19] == ["cos_cache", "sin_cache", "", "q_norm", "k_norm"]
    assert attention["inputs"][21] == "attention_metadata"
    assert attention["is_causal"] == 1
    assert "causal" not in attention


def test_paged_qwen_sparse_attention_appends_compact_update_abi():
    model = make_sparse_model(paged=True)
    model.context_length_attrs = {"state_update_capacity": 7}
    model.input_names.update(
        {
            "state_update.capture_count": "state_update_capture_count",
            "state_update.active": "state_update_active",
        }
    )
    model.output_names["state_update.indexer"] = {3: "state_update.3.indexer"}

    model.make_qwen_sparse_attention(3, make_attention(), "hidden_states")

    indexer = emitted_nodes(model)[0][1]
    assert indexer["inputs"][16:] == ["state_update_capture_count", "state_update_active"]
    assert indexer["outputs"][6] == "state_update.3.indexer"
    assert indexer["state_update_capacity"] == 7


def test_qwen_attention_gate_uses_resolved_attention_output():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.num_attn_heads = 8
    model.head_size = 16
    model.attention_attrs = {
        "gate_path": "/model/layers.3/attn/gate/Reshape/output_0",
        "o_path": "/model/layers.3/attn/DynamicSparseAttention/output_0",
    }
    model.layernorm_attrs = {"skip_input": ""}
    model.make_hidden_state_shape = MethodType(
        lambda self, last_dim: ["batch_size", "sequence_length", last_dim], model
    )
    record_calls(model, ["make_sigmoid", "make_mul", "make_matmul"])
    attention = SimpleNamespace(o_proj=SimpleNamespace(bias=None))

    model.make_attention_output_proj(3, attention, "hidden_states")

    gate = next(call for call in model.calls if call[0] == "make_mul")
    assert gate[1][1][0] == "/model/layers.3/attn/DynamicSparseAttention/output_0"


def make_ple_model(paged, fp8_embedding=False):
    model = object.__new__(Qwen4ExpTextModel)
    model.filename = "model.onnx"
    model.use_paged_attention = paged
    model.io_dtype = ir.DataType.FLOAT16
    model.hidden_size = 8
    model.hc_count = 2
    model.hc_hidden_size = 16
    model.ple_embed_dim = 8
    model.ngram_size = 3
    model.heads_per_ngram = 2
    model.ple_conv_kernel_size = 4
    model.ple_conv_dilation = 3
    model.layernorm_attrs = {"epsilon": 1e-5}
    model.values = {}
    model.input_names = {
        "input_ids": "input_ids",
        "past.ple_tokens": {1: "past.1.ple_tokens"},
        "past.ple_conv": {1: "past.1.ple_conv"},
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
    }
    model.output_names = {
        "present.ple_tokens": {1: "present.1.ple_tokens"},
        "present.ple_conv": {1: "present.1.ple_conv"},
    }
    record_calls(
        model,
        [
            "make_initializer",
            "make_node",
            "make_value",
            "make_gather",
            "make_unsqueeze",
            "make_reshape",
            "make_matmul",
            "make_add",
        ],
    )
    table = SimpleNamespace(weight=torch.zeros((32, 2), dtype=torch.float16))
    if fp8_embedding:
        table.weight = torch.zeros((32, 2), dtype=torch.float8_e4m3fn)
        table.weight_scale = torch.ones(1)
    embedding = SimpleNamespace(
        layer_multipliers=torch.arange(3, dtype=torch.int64),
        ngram_heads_vocab_sizes=torch.full((4,), 7, dtype=torch.int64),
        ngram_heads_offsets=torch.arange(4, dtype=torch.int64),
        eos_token_id=1,
        ngram_embedding=table,
    )
    norm = SimpleNamespace(weight=torch.zeros(16))
    ple = SimpleNamespace(
        ple_embedding=embedding,
        key_proj=SimpleNamespace(),
        value_proj=SimpleNamespace(),
        norm_key=norm,
        norm_query=norm,
        norm_conv=norm,
        conv1d=SimpleNamespace(weight=torch.zeros((16, 1, 4))),
    )
    return model, ple


def test_dense_ple_emits_verified_contrib_schemas():
    model, ple = make_ple_model(paged=False)

    model.make_ple(1, ple, "hidden_states")

    assert model.external_data_files == {
        "model.ple.ngram_embedding.weight": "engram.onnx.data"
    }

    nodes = emitted_nodes(model)
    ngram = next(kwargs for op_type, kwargs in nodes if op_type == "NGramHashMapping")
    assert ngram["inputs"] == [
        "input_ids",
        "model.layers.1.ple.layer_multipliers",
        "model.layers.1.ple.head_vocab_sizes",
        "past.1.ple_tokens",
        "model.layers.1.ple.head_offsets",
        "model.layers.1.ple.eos_token_id",
    ]
    assert ngram["max_ngram_size"] == 3
    assert ngram["n_head_per_ngram"] == 2
    assert ngram["pad_id"] == 1

    gather = next(kwargs for op_type, kwargs in nodes if op_type == "GatherBlockQuantized")
    assert gather["inputs"] == [
        "model.ple.ngram_embedding.weight",
        "/model/layers.1/ple/NGramHashMapping/output_0",
        "model.ple.ngram_embedding.weight_scale",
    ]
    assert gather["domain"] == "com.microsoft"
    assert gather["metadata_props"] == {"layer_ann": "cpu_embedding"}
    assert not any(op_type == "GatherND" for op_type, _ in nodes)

    embedding_initializers = {
        args[1]: args[0] for method, args, _ in model.calls if method == "make_initializer"
    }
    assert embedding_initializers["model.ple.ngram_embedding.weight"].dtype == torch.float8_e4m3fn
    assert embedding_initializers["model.ple.ngram_embedding.weight_scale"].shape == (1, 1)

    gate = next(kwargs for op_type, kwargs in nodes if op_type == "EngramGate")
    assert len(gate["inputs"]) == 6
    assert gate["outputs"] == [
        "/model/layers.1/ple/EngramGate/output_0",
        "/model/layers.1/ple/EngramGate/output_1",
    ]
    assert "hc_count" not in gate

    conv = next(kwargs for op_type, kwargs in nodes if op_type == "CausalConvWithState")
    assert conv["inputs"] == [
        "/model/layers.1/ple/EngramGate/FlattenNormed/output_0",
        "model.layers.1.ple.conv1d.weight",
        "",
        "past.1.ple_conv",
    ]
    assert conv["dilation"] == 3
    assert conv["channels_last"] == 1


def test_paged_fp8_ple_uses_varlen_hash_and_quantized_gather():
    model, ple = make_ple_model(paged=True, fp8_embedding=True)

    model.make_ple(1, ple, "hidden_states")

    nodes = emitted_nodes(model)
    ngram = next(kwargs for op_type, kwargs in nodes if op_type == "VarlenNGramHashMapping")
    assert ngram["inputs"][3] == "cumulative_sequence_lengths"
    assert ngram["inputs"][4:] == [
        "past.1.ple_tokens",
        "model.layers.1.ple.head_offsets",
        "model.layers.1.ple.eos_token_id",
    ]
    gather = next(kwargs for op_type, kwargs in nodes if op_type == "GatherBlockQuantized")
    assert gather["domain"] == "com.microsoft"
    assert gather["quantize_axis"] == 1
    assert gather["block_size"] == 0
    assert gather["inputs"][:3] == [
        "model.ple.ngram_embedding.weight",
        "/model/layers.1/ple/VarlenNGramHashMapping/output_0",
        "model.ple.ngram_embedding.weight_scale",
    ]
    assert gather["metadata_props"] == {"layer_ann": "cpu_embedding"}
    assert model.external_data_files == {
        "model.ple.ngram_embedding.weight": "engram.onnx.data"
    }
    conv = next(kwargs for op_type, kwargs in nodes if op_type == "VarlenCausalConvWithState")
    assert conv["inputs"] == [
        "/model/layers.1/ple/EngramGate/FlattenNormed/output_0",
        "model.layers.1.ple.conv1d.weight",
        "cumulative_sequence_lengths",
        "",
        "past.1.ple_conv",
    ]
    assert conv["outputs"] == [
        "/model/layers.1/ple/VarlenCausalConvWithState/output_0",
        "present.1.ple_conv",
    ]
    assert conv["dilation"] == 3
    assert not any(op_type == "CausalConvWithState" for op_type, _ in nodes)


def test_paged_ple_appends_compact_token_and_conv_updates():
    model, ple = make_ple_model(paged=True)
    model.context_length_attrs = {"state_update_capacity": 7}
    model.input_names.update(
        {
            "state_update.capture_count": "state_update_capture_count",
            "state_update.active": "state_update_active",
        }
    )
    model.output_names.update(
        {
            "state_update.ple_tokens": {1: "state_update.1.ple_tokens"},
            "state_update.ple_conv_value": {1: "state_update.1.ple_conv_value"},
        }
    )
    model.output_shapes = {"state_update.ple_conv_value": ["batch_size", 7, 16]}

    model.make_ple(1, ple, "hidden_states")

    nodes = emitted_nodes(model)
    ngram = next(kwargs for op_type, kwargs in nodes if op_type == "VarlenNGramHashMapping")
    assert ngram["inputs"][9] == "state_update_capture_count"
    assert ngram["outputs"][3] == "state_update.1.ple_tokens"
    assert ngram["state_update_capacity"] == 7
    conv = next(kwargs for op_type, kwargs in nodes if op_type == "VarlenCausalConvWithState")
    assert conv["inputs"][5:] == ["state_update_capture_count", "state_update_active"]
    assert conv["outputs"][2] == "state_update.1.ple_conv_value"
    assert conv["state_update_capacity"] == 7


def test_external_engram_captures_identity_token_state():
    model, ple = make_ple_model(paged=True)
    model.external_engram = True
    model.context_length_attrs = {"state_update_capacity": 4}
    model.input_names.update({
        "engram_embeddings": "engram_embeddings",
        "state_update.capture_count": "state_update_capture_count",
        "state_update.active": "state_update_active",
    })
    model.output_names.update({
        "state_update.ple_tokens": {1: "state_update.1.ple_tokens"},
        "state_update.ple_conv_value": {1: "state_update.1.ple_conv_value"},
    })
    model.output_shapes = {"state_update.ple_conv_value": ["batch_size", 4, 16]}
    model.make_ple(1, ple, "hidden_states")
    nodes = emitted_nodes(model)
    identity = next(kwargs for op_type, kwargs in nodes if op_type == "Identity")
    assert identity["inputs"] == ["past.1.ple_tokens"]
    assert identity["outputs"] == ["present.1.ple_tokens"]
    expand = next(kwargs for op_type, kwargs in nodes if op_type == "Expand")
    assert expand["outputs"] == ["state_update.1.ple_tokens"]
    assert not any(op_type == "VarlenNGramHashMapping" for op_type, _ in nodes)


def test_paged_ple_emits_packed_engram_gate_shapes():
    model, ple = make_ple_model(paged=True)

    model.make_ple(1, ple, "hidden_states")

    gate = next(kwargs for op_type, kwargs in emitted_nodes(model) if op_type == "EngramGate")
    assert gate["inputs"][:3] == [
        "/model/layers.1/ple/key_proj/Reshape/output_0",
        "/model/layers.1/ple/query/Reshape/output_0",
        "/model/layers.1/ple/value_proj/MatMul/output_0",
    ]

    reshapes = {args[0]: args for method, args, _ in model.calls if method == "make_reshape"}
    assert reshapes["/model/layers.1/ple/key_proj/Reshape"][1][1] == "/model/constants/INT64/[-1, 2, 8]"
    assert reshapes["/model/layers.1/ple/query/Reshape"][1][1] == "/model/constants/INT64/[-1, 2, 8]"
    assert reshapes["/model/layers.1/ple/key_proj/Reshape"][3] == ["num_tokens", 2, 8]
    assert reshapes["/model/layers.1/ple/query/Reshape"][3] == ["num_tokens", 2, 8]
    assert reshapes["/model/layers.1/ple/EngramGate/Flatten"][1][1] == "/model/constants/INT64/[-1, 16]"
    assert reshapes["/model/layers.1/ple/EngramGate/FlattenNormed"][1][1] == "/model/constants/INT64/[-1, 16]"

    value_shapes = {
        args[0]: args[2]
        for method, args, _ in model.calls
        if method == "make_value" and len(args) >= 3
    }
    assert value_shapes["/model/layers.1/ple/EngramGate/output_0"] == ["num_tokens", 2, 8]
    assert value_shapes["/model/layers.1/ple/EngramGate/output_1"] == ["num_tokens", 2, 8]


def test_ple_reuses_model_level_embedding_initializers():
    model, ple = make_ple_model(paged=True, fp8_embedding=True)

    model.make_ple(1, ple, "hidden_states")
    model.values.update({
        "model.ple.ngram_embedding.weight": object(),
        "model.ple.ngram_embedding.weight_scale": object(),
    })
    model.input_names["past.ple_tokens"][2] = "past.2.ple_tokens"
    model.input_names["past.ple_conv"][2] = "past.2.ple_conv"
    model.output_names["present.ple_tokens"][2] = "present.2.ple_tokens"
    model.output_names["present.ple_conv"][2] = "present.2.ple_conv"
    model.make_ple(2, ple, "hidden_states")

    shared_initializers = [
        call[1][1]
        for call in model.calls
        if call[0] == "make_initializer" and call[1][1].startswith("model.ple.ngram_embedding")
    ]
    assert shared_initializers == [
        "model.ple.ngram_embedding.weight",
        "model.ple.ngram_embedding.weight_scale",
    ]


def test_qwen4_exp_engram_model_extracts_cpu_lookup_graph(tmp_path):
    embedding = SimpleNamespace(
        layer_multipliers=torch.tensor([0, 1, 2], dtype=torch.int64),
        ngram_heads_vocab_sizes=torch.tensor([7, 7, 7, 7], dtype=torch.int64),
        ngram_heads_offsets=torch.tensor([0, 7, 14, 21], dtype=torch.int64),
        eos_token_id=0,
        ngram_embedding=torch.nn.Embedding(28, 8),
    )
    ple = SimpleNamespace(ple_embedding=embedding)
    config = SimpleNamespace(ngram_size=3, heads_per_ngram=2, ple_embed_dim=8)

    model = Qwen4ExpEngramModel(ple, config, ir.DataType.FLOAT16)
    model.save_model(tmp_path)

    assert [value.name for value in model.graph.inputs] == ["input_ids", "past_ple_tokens"]
    assert [value.name for value in model.graph.outputs] == [
        "engram_embeddings",
        "present_ple_tokens",
    ]
    assert [node.op_type for node in model.graph] == [
        "NGramHashMapping",
        "GatherBlockQuantized",
        "Constant",
        "Reshape",
        "Identity",
    ]
    assert (tmp_path / "engram.onnx").exists()
    assert (tmp_path / "engram.onnx.data").exists()
    exported = onnx.load(tmp_path / "engram.onnx", load_external_data=False)
    table = next(initializer for initializer in exported.graph.initializer
                 if initializer.name == "model.ple.ngram_embedding.weight")
    assert next(entry.value for entry in table.external_data if entry.key == "location") == "engram.onnx.data"


@pytest.mark.parametrize("text_only", [False, True])
def test_qwen4_exp_models_share_engram_data(tmp_path, monkeypatch, text_only):
    embedding = SimpleNamespace(
        layer_multipliers=torch.tensor([0, 1, 2], dtype=torch.int64),
        ngram_heads_vocab_sizes=torch.tensor([7, 7, 7, 7], dtype=torch.int64),
        ngram_heads_offsets=torch.tensor([0, 7, 14, 21], dtype=torch.int64),
        eos_token_id=0,
        ngram_embedding=torch.nn.Embedding(28, 8),
    )
    ple = SimpleNamespace(ple_embedding=embedding)
    text_config = SimpleNamespace(ngram_size=3, heads_per_ngram=2, ple_embed_dim=8)
    table_name = "model.ple.ngram_embedding.weight"
    table = Qwen4ExpEngramModel(ple, text_config, ir.DataType.FLOAT16).graph.initializers[table_name].const_value

    def component(filename):
        builder = Model.__new__(Model)
        builder.model = ir.Model(ir.Graph(
            inputs=(), outputs=(), nodes=(),
            initializers=[ir.Value(name=table_name, const_value=table)],
            opset_imports={"": 22},
        ), ir_version=10, producer_name="onnxruntime-genai")
        builder.filename = filename
        builder.cache_dir = str(tmp_path / "cache")
        builder.quant_type = None
        builder.onnx_dtype = ir.DataType.FLOAT16
        builder.external_data_files = {table_name: "engram.onnx.data"}
        return builder

    wrapper = object.__new__(Qwen4ExpModel)
    wrapper.text_only = text_only
    wrapper.input_path = "unused"
    wrapper.config = SimpleNamespace(text_config=text_config, vision_config=SimpleNamespace(out_hidden_size=8))
    wrapper.decoder = component("model.onnx")
    wrapper.decoder.io_dtype = ir.DataType.FLOAT16
    wrapper.decoder.use_paged_attention = False
    wrapper.decoder.ple_layer_ids = {1}
    wrapper.decoder.hidden_size = 8
    language_model = SimpleNamespace(
        layers=[None, SimpleNamespace(ple=ple)],
        embed_tokens=SimpleNamespace(weight=torch.zeros((2, 8))),
    )
    wrapper.decoder.load_weights = lambda _: SimpleNamespace(model=SimpleNamespace(language_model=language_model, visual=None))
    wrapper.mtp = component("mtp.onnx")
    wrapper.mtp_attrs = {}
    wrapper.share_initializers = lambda *args: []
    monkeypatch.setattr("models.builders.qwen3_8.Qwen4ExpEmbeddingModel", lambda *args, **kwargs: SimpleNamespace(save_model=lambda _: None))
    monkeypatch.setattr("models.builders.qwen3_8.Qwen4ExpVisionModel", lambda *args: SimpleNamespace(save_model=lambda _: None))

    wrapper.save_model(tmp_path)

    filenames = ("model.onnx", "mtp.onnx") if text_only else ("engram.onnx", "model.onnx", "mtp.onnx")
    exported = [onnx.load(tmp_path / filename, load_external_data=False) for filename in filenames]
    tables = [next(value for value in model.graph.initializer if value.name == table_name) for model in exported]
    offsets = [next(entry.value for entry in value.external_data if entry.key == "offset") for value in tables]
    assert offsets == [offsets[0]] * len(tables)
    data_file = "engram.onnx.data"
    assert all(next(entry.value for entry in value.external_data if entry.key == "location") == data_file
               for value in tables)
    loaded = [onnx.load(tmp_path / filename, load_external_data=True) for filename in filenames]
    table_values = [onnx.numpy_helper.to_array(next(value for value in model.graph.initializer
                                                    if value.name == table_name)) for model in loaded]
    assert all((value == table_values[0]).all() for value in table_values[1:])


def test_qwen38_config_assigns_embedding_annotation_to_cpu():
    model = object.__new__(Qwen4ExpTextModel)
    model.ep = "cuda"
    model.ple_token_pad_id = 248044
    model.fixed_indexer_cache = True
    model.input_names = {"past_sequence_length": "past_sequence_length"}
    genai_config = {"model": {"decoder": {"inputs": {}, "outputs": {}, "session_options": {}}}}

    model.update_genai_config(genai_config)

    assert genai_config["model"]["decoder"]["inputs"]["past_sequence_length"] == "past_sequence_length"
    assert genai_config["model"]["decoder"]["session_options"]["session.layer_assignment_settings"] == (
        "cpu(=cpu_embedding)"
    )


def test_qwen38_external_engram_keeps_decoder_cuda_only():
    model = object.__new__(Qwen4ExpTextModel)
    model.ep = "cuda"
    model.external_engram = True
    model.ple_token_pad_id = 248044
    model.fixed_indexer_cache = True
    model.input_names = {"past_sequence_length": "past_sequence_length"}
    genai_config = {"model": {"decoder": {"inputs": {}, "outputs": {}, "session_options": {}}}}

    model.update_genai_config(genai_config)

    assert "session.layer_assignment_settings" not in genai_config["model"]["decoder"]["session_options"]
