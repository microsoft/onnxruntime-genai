# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from types import MethodType, SimpleNamespace

import onnx_ir as ir
import torch

from models.builders.qwen import Qwen4ExpTextModel


def record_calls(model, method_names):
    model.calls = []

    def make_recorder(method_name):
        def record(self, *args, **kwargs):
            self.calls.append((method_name, args, kwargs))
            if method_name == "make_matmul":
                return args[1]

        return record

    for method_name in method_names:
        setattr(model, method_name, MethodType(make_recorder(method_name), model))


def make_sparse_model(paged):
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = paged
    model.io_dtype = ir.DataType.FLOAT16
    model.num_attn_heads = 8
    model.num_kv_heads = 2
    model.head_size = 16
    model.indexer_num_heads = 4
    model.indexer_kv_heads = 1
    model.indexer_head_dim = 16
    model.indexer_budget = 32
    model.indexer_compress_ratio = 4
    model.layernorm_attrs = {"epsilon": 1e-6}
    model.rope_attrs = {"interleaved": 0}
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
        "position_ids": "position_ids",
        "past.indexer": {3: "past.3.indexer_key"},
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "block_table": "block_table",
        "attention_metadata": "attention_metadata",
        "sparse_attention.selected_indices": {3: "sparse_attention.3.selected_indices"},
        "sparse_attention.selected_counts": {3: "sparse_attention.3.selected_counts"},
    }
    model.output_names = {"present.indexer": {3: "present.3.indexer_key"}}
    record_calls(
        model,
        [
            "make_initializer",
            "make_matmul",
            "make_split",
            "make_reshape",
            "make_node",
            "make_value",
        ],
    )

    def make_attention_input_proj(self, layer_id, attention, root_input):
        self.attention_attrs.update(q_path="query", k_path="key", v_path="value")

    model.make_attention_input_proj = MethodType(make_attention_input_proj, model)
    model.get_qk_norm_weight_names = MethodType(lambda self, layer_id: ("q_norm", "k_norm"), model)
    model.make_rotary_embedding_caches = MethodType(lambda self: ("cos_cache", "sin_cache"), model)
    model.make_key_value_cache_names = MethodType(
        lambda self, layer_id: ("past_key", "past_value", "present_key", "present_value"), model
    )
    model.make_qsa_rotary_caches = MethodType(
        lambda self, layer_id, root_input, cos, sin: ("index_cos", "index_sin"), model
    )
    model.make_qsa_visibility_mask = MethodType(lambda self, layer_id, root_input: "visibility_mask", model)
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


def emitted_nodes(model):
    return [(args[0], kwargs) for method, args, kwargs in model.calls if method == "make_node"]


def test_paged_branchwise_norm_uses_packed_shapes():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = True
    model.io_dtype = ir.DataType.FLOAT16
    model.hc_count = 2
    model.layernorm_attrs = {"epsilon": 1e-6}
    record_calls(
        model,
        [
            "make_reshape",
            "make_initializer",
            "make_node",
            "make_value",
            "make_mul",
        ],
    )

    model.make_branchwise_rms_norm("/norm", "hidden_states", SimpleNamespace(weight=torch.zeros(16)), 8)

    reshapes = [args for method, args, _ in model.calls if method == "make_reshape"]
    assert reshapes[0][1][1] == "/model/constants/INT64/[-1, 2, 8]"
    assert reshapes[0][3] == ["num_tokens", 2, 8]
    assert reshapes[1][1][1] == "/model/constants/INT64/[-1, 16]"
    assert reshapes[1][3] == ["num_tokens", 16]

    initializers = [call for call in model.calls if call[0] == "make_initializer"]
    assert initializers[0][1][0].shape == (8,)
    assert torch.equal(initializers[0][1][0], torch.ones(8))
    assert initializers[1][1][0].shape == (16,)
    assert torch.equal(initializers[1][1][0], torch.ones(16))

    layer_norm = next(
        call for call in model.calls if call[0] == "make_node" and call[1][0] == "SimplifiedLayerNormalization"
    )
    assert layer_norm[1][0] == "SimplifiedLayerNormalization"
    assert layer_norm[2]["inputs"] == ["/norm/CastInput/output_0", "norm.norm_scale"]
    assert "domain" not in layer_norm[2]
    assert layer_norm[2]["axis"] == -1
    assert layer_norm[2]["epsilon"] == 1e-6


def test_dense_branchwise_norm_preserves_batch_and_sequence_shapes():
    model = object.__new__(Qwen4ExpTextModel)
    model.use_paged_attention = False
    model.io_dtype = ir.DataType.BFLOAT16
    model.hc_count = 4
    model.layernorm_attrs = {"epsilon": 1e-6}
    record_calls(
        model,
        ["make_reshape", "make_initializer", "make_node", "make_value", "make_mul"],
    )

    output = model.make_branchwise_rms_norm(
        "/norm", "hidden_states", SimpleNamespace(weight=torch.zeros(32)), 8
    )

    reshapes = [args for method, args, _ in model.calls if method == "make_reshape"]
    assert reshapes[0][1][1] == "/model/constants/INT64/[0, 0, 4, 8]"
    assert reshapes[0][3] == ["batch_size", "sequence_length", 4, 8]
    assert reshapes[1][1][1] == "/model/constants/INT64/[0, 0, 32]"
    assert reshapes[1][3] == ["batch_size", "sequence_length", 32]
    assert output == "/norm/CastOutput/output_0"


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
        "/model/layers.3/attn/indexer/query/SimplifiedLayerNormalization/output_0",
        "/model/layers.3/attn/indexer/key",
        "model.layers.3.attn.indexer.k_norm.weight",
        "index_cos",
        "index_sin",
        "visibility_mask",
        "past.3.indexer_key",
    ]
    assert indexer["outputs"] == [
        "/model/layers.3/attn/SparseAttentionIndexer/output_0",
        "present.3.indexer_key",
    ]
    assert indexer["policy_mode"] == "qsa"
    assert indexer["token_budget"] == 32
    assert indexer["compress_ratio"] == 4

    attention = nodes[-1][1]
    assert attention["inputs"][7:11] == [
        "/model/layers.3/attn/SparseAttentionIndexer/Flatten/output_0",
        "/model/layers.3/attn/SparseAttentionIndexer/CountsFlatten/output_0",
        "seqlens/output_0",
        "total_length/output_0",
    ]
    assert attention["outputs"] == [
        "/model/layers.3/attn/DynamicSparseAttention/output_0",
        "present_key",
        "present_value",
    ]
    assert attention["is_causal"] == 1
    assert attention["attention_mode"] == "selected_only"
    assert attention["selected_kv_source"] == "main"


def test_paged_qwen_sparse_attention_emits_shared_webgpu_schema():
    model = make_sparse_model(paged=True)

    model.make_qwen_sparse_attention(3, make_attention(), "hidden_states")

    nodes = emitted_nodes(model)
    assert [op_type for op_type, _ in nodes] == ["SparsePagedAttention"]
    attention = nodes[0][1]
    assert len(attention["inputs"]) == 22
    assert attention["inputs"][5:11] == [
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
        "block_table",
        "",
        "sparse_attention.3.selected_indices",
        "sparse_attention.3.selected_counts",
    ]
    assert attention["inputs"][14:19] == ["cos_cache", "sin_cache", "", "q_norm", "k_norm"]
    assert attention["inputs"][21] == "attention_metadata"
    assert attention["is_causal"] == 1
    assert "causal" not in attention


def make_ple_model(paged, fp8_embedding=False):
    model = object.__new__(Qwen4ExpTextModel)
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
