# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Graph-shape tests for the Qwen-3.8 Flash Next (``Qwen4Exp``) builder.

These are static tests: no checkpoint is downloaded and no ONNX model is run.  A partially
initialized builder is created with ``object.__new__`` and only ``make_node`` / ``make_value`` /
``make_initializer`` / ``make_matmul`` are stubbed, so every intermediate helper
(``make_reshape``, ``make_mul``, ``make_reduce_mean``, ...) executes for real and the recorded
node list is the graph the builder would actually emit.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace

import onnx_ir as ir
import pytest
import torch

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))

sys.modules.setdefault("models", types.ModuleType("models"))
_builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
_builders_package.__path__ = [str(BUILDERS_DIR)]


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


base_module = _load_builder_module("base")
_load_builder_module("quant_config")
qwen_module = _load_builder_module("qwen")
qwen4exp = _load_builder_module("qwen4exp")

Qwen4ExpTextModel = qwen4exp.Qwen4ExpTextModel

HIDDEN = 32
HC_COUNT = 4
HC_LOWRANK = 16
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
INDEXER_HEADS = 2
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 8
NGRAM_SIZE = 4
HEADS_PER_NGRAM = 2
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
PLE_EMBED_DIM = NGRAM_HEADS * 6
PLE_CONV_KERNEL = 3
VOCAB = 1024


def _make_recording_model(use_paged_attention=False, layer_types=None, ple_layers=(0,)):
    """A ``Qwen4ExpTextModel`` whose leaf emitters record instead of building an IR graph."""
    model = object.__new__(Qwen4ExpTextModel)

    model.nodes = []
    model.values = []
    model.initializers = []

    def make_node(op_type, inputs, outputs, *, name, domain="", **kwargs):
        model.nodes.append(
            SimpleNamespace(op_type=op_type, inputs=list(inputs), outputs=list(outputs), name=name, domain=domain, attrs=kwargs)
        )

    def make_value(name, dtype=None, shape=None):
        model.values.append(SimpleNamespace(name=name, dtype=dtype, shape=shape))

    def make_initializer(tensor, name, to=None):
        model.initializers.append(SimpleNamespace(name=name, tensor=tensor, to=to))

    def make_matmul(matmul_module, basename, root_input, **kwargs):
        model.nodes.append(
            SimpleNamespace(
                op_type="MatMul",
                inputs=[root_input, f"{basename}/weight"],
                outputs=[f"{basename}/output_0"],
                name=basename,
                domain="",
                attrs={},
            )
        )
        return basename

    model.make_node = MethodType(lambda self, *a, **k: make_node(*a, **k), model)
    model.make_value = MethodType(lambda self, *a, **k: make_value(*a, **k), model)
    model.make_initializer = MethodType(lambda self, *a, **k: make_initializer(*a, **k), model)
    model.make_matmul = MethodType(lambda self, *a, **k: make_matmul(*a, **k), model)

    model.io_dtype = ir.DataType.FLOAT16
    model.onnx_dtype = ir.DataType.FLOAT16
    model.ep = "cuda"
    model.hidden_size = HIDDEN
    model.hc_count = HC_COUNT
    model.hc_lowrank = HC_LOWRANK
    model.hc_hidden_size = HC_COUNT * HIDDEN
    model.head_size = HEAD_DIM
    model.num_attn_heads = NUM_HEADS
    model.num_kv_heads = NUM_KV_HEADS
    model.num_layers = 4
    model.vocab_size = VOCAB
    model.use_paged_attention = use_paged_attention
    model.layer_types = list(layer_types or ["full_attention"] * 4)
    model.layernorm_attrs = {"epsilon": 1e-6, "add_offset": 1, "root_input": "inputs_embeds", "skip_input": ""}
    model.attention_attrs = {"scale": HEAD_DIM**-0.5, "softcap": 0.0, "op_type": "GroupQueryAttention"}
    model.mask_attrs = {"seqlens_k": "seqlens_k", "total_seq_len": "total_seq_len", "mask_name": ""}

    # QSA indexer
    model.indexer_n_heads = INDEXER_HEADS
    model.indexer_kv_heads = INDEXER_KV_HEADS
    model.indexer_head_dim = INDEXER_HEAD_DIM
    model.indexer_budget = 512
    model.indexer_compress_ratio = 16

    # PLE / n-gram
    model.ple_layer_index = {layer_id: idx for idx, layer_id in enumerate(ple_layers)}
    model.ple_embed_dim = PLE_EMBED_DIM
    model.ple_conv_kernel_size = PLE_CONV_KERNEL
    model.ple_conv_state_len = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE
    model.ngram_size = NGRAM_SIZE
    model.heads_per_ngram = HEADS_PER_NGRAM
    model.ngram_heads = NGRAM_HEADS
    model.ngram_vocab_size_base = 1000
    model.ngram_seed = 1234
    model.ngram_eos_token_id = 7

    model.input_names = {
        "input_ids": "input_ids",
        "block_table": "block_table",
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "attention_metadata": "attention_metadata",
    }
    model.output_names = {"logits": "logits", "hidden_states": "hidden_states"}
    model.input_types = {}
    model.input_shapes = {}
    model.output_types = {}
    model.output_shapes = {}
    for layer_id, layer_type in enumerate(model.layer_types):
        if layer_type == "full_attention":
            model.input_names[f"past_state.{layer_id}.indexer_key"] = f"past_key_values.{layer_id}.indexer_key"
            model.output_names[f"present_state.{layer_id}.indexer_key"] = f"present.{layer_id}.indexer_key"
    for layer_id in ple_layers:
        model.input_names[f"past_state.{layer_id}.ple_conv"] = f"past_key_values.{layer_id}.ple_conv_state"
        model.output_names[f"present_state.{layer_id}.ple_conv"] = f"present.{layer_id}.ple_conv_state"
        model.input_names[f"past_state.{layer_id}.ple_tokens"] = f"past_key_values.{layer_id}.ple_tokens"
        model.output_names[f"present_state.{layer_id}.ple_tokens"] = f"present.{layer_id}.ple_tokens"
    return model


def _gated_residual_module():
    wide = HC_COUNT * HIDDEN
    return SimpleNamespace(
        hc_norm=SimpleNamespace(weight=torch.zeros(wide)),
        input_mix_weight_down=SimpleNamespace(weight=torch.zeros(HC_LOWRANK, wide)),
        input_mix_weight_up=SimpleNamespace(weight=torch.zeros(wide, HC_LOWRANK)),
        block_inject_weight=SimpleNamespace(weight=torch.zeros(HC_COUNT, wide)),
    )


def _ple_module():
    wide = HC_COUNT * HIDDEN
    return SimpleNamespace(
        ple_embedding=SimpleNamespace(ngram_embedding=SimpleNamespace(weight=torch.zeros(4096, PLE_EMBED_DIM // NGRAM_HEADS))),
        key_proj=SimpleNamespace(weight=torch.zeros(wide, PLE_EMBED_DIM)),
        value_proj=SimpleNamespace(weight=torch.zeros(HIDDEN, PLE_EMBED_DIM)),
        norm_key=SimpleNamespace(weight=torch.zeros(wide)),
        norm_query=SimpleNamespace(weight=torch.zeros(wide)),
        norm_conv=SimpleNamespace(weight=torch.zeros(wide)),
        conv1d=SimpleNamespace(weight=torch.zeros(wide, 1, PLE_CONV_KERNEL)),
    )


def _indexer_module():
    qk = (INDEXER_HEADS + INDEXER_KV_HEADS) * INDEXER_HEAD_DIM
    return SimpleNamespace(
        index_qk_proj=SimpleNamespace(weight=torch.zeros(qk, HIDDEN)),
        q_layernorm=SimpleNamespace(weight=torch.zeros(INDEXER_HEAD_DIM)),
        k_layernorm=SimpleNamespace(weight=torch.zeros(INDEXER_HEAD_DIM)),
    )


def _named(model):
    return {node.name: node for node in model.nodes}


def _ops(model):
    return [node.op_type for node in model.nodes]


#####################################################################################
# Hyper-connections
#####################################################################################


def test_gated_residual_emits_grouped_rmsnorm_over_hidden_size_not_stream_width():
    model = _make_recording_model()
    model._make_gated_residual("/hc", _gated_residual_module(), "stream")

    norm = _named(model)["/hc/hc_norm/norm/SimplifiedLayerNormalization"]
    assert norm.op_type == "SimplifiedLayerNormalization"
    assert norm.domain is None
    assert norm.attrs["axis"] == -1
    assert norm.attrs["stash_type"] == 1
    assert norm.attrs["epsilon"] == 1e-6

    # The normalization group is `hidden_size`, so the input is first reshaped to
    # [B, S, hc_count, hidden_size] and the learned scale is applied afterwards.
    reshape = _named(model)["/hc/hc_norm/group/Reshape"]
    assert reshape.inputs[1] == f"/model/constants/INT64/[0, 0, {HC_COUNT}, {HIDDEN}]"
    ones = next(init for init in model.initializers if init.name == "/hc/hc_norm/norm/ones")
    assert tuple(ones.tensor.shape) == (HIDDEN,)
    assert torch.equal(ones.tensor, torch.ones(HIDDEN))


def test_gated_residual_bakes_the_plus_one_offset_into_the_norm_scale():
    model = _make_recording_model()
    module = _gated_residual_module()
    module.hc_norm.weight = torch.full((HC_COUNT * HIDDEN,), 0.25)
    model._make_gated_residual("/hc", module, "stream")

    scale = next(init for init in model.initializers if init.name == "/hc/hc_norm/norm/weight")
    assert torch.allclose(scale.tensor, torch.full((HC_COUNT * HIDDEN,), 1.25))


def test_gated_residual_scales_both_low_rank_projections_by_stream_count():
    model = _make_recording_model()
    model._make_gated_residual("/hc", _gated_residual_module(), "stream")

    nodes = _named(model)
    for div_name in ("/hc/input_mix_weight_down/Div", "/hc/block_inject_weight/Div"):
        assert nodes[div_name].op_type == "Div"
        assert nodes[div_name].inputs[1] == f"/model/constants/FLOAT16/{float(HC_COUNT)}"


def test_gated_residual_mixes_streams_with_reduce_mean_over_the_stream_axis():
    model = _make_recording_model()
    mixed, hyper_input, injection = model._make_gated_residual("/hc", _gated_residual_module(), "stream")

    reduce_mean = _named(model)["/hc/mix/ReduceMean"]
    assert reduce_mean.op_type == "ReduceMean"
    assert reduce_mean.inputs[1] == "/model/constants/INT64/[-2]", "must average over hc_count, not hidden_size"
    assert reduce_mean.attrs["keepdims"] == 0

    assert mixed == "/hc/mix/ReduceMean/output_0"
    assert hyper_input == "stream", "the un-normalized stream is the residual carrier"
    assert injection == "/hc/block_inject_weight/Mul/output_0"

    mixed_value = next(value for value in model.values if value.name == mixed)
    assert mixed_value.shape == ["batch_size", "sequence_length", HIDDEN]


def test_gated_residual_injection_weight_is_two_times_sigmoid():
    model = _make_recording_model()
    model._make_gated_residual("/hc", _gated_residual_module(), "stream")

    nodes = _named(model)
    assert nodes["/hc/block_inject_weight/Sigmoid"].op_type == "Sigmoid"
    mul = nodes["/hc/block_inject_weight/Mul"]
    assert mul.inputs == ["/hc/block_inject_weight/Sigmoid/output_0", "/model/constants/FLOAT16/2.0"]


def test_final_mixer_skips_the_injection_branch():
    model = _make_recording_model()
    mixed, hyper_input, injection = model._make_gated_residual(
        "/mixer", _gated_residual_module(), "stream", use_combine=False
    )

    assert (hyper_input, injection) == (None, None)
    assert mixed == "/mixer/mix/ReduceMean/output_0"
    assert "/mixer/block_inject_weight/MatMul" not in _named(model)


def test_injection_broadcasts_block_output_across_streams_then_flattens():
    model = _make_recording_model()
    output = model._make_injection("/hc/inject", "stream", "block_out", "inject_w")

    nodes = _named(model)
    assert nodes["/hc/inject/block/Unsqueeze"].inputs == ["block_out", "/model/constants/INT64/[-2]"]
    assert nodes["/hc/inject/weights/Unsqueeze"].inputs == ["inject_w", "/model/constants/INT64/[-1]"]
    assert nodes["/hc/inject/Mul"].inputs == [
        "/hc/inject/block/Unsqueeze/output_0",
        "/hc/inject/weights/Unsqueeze/output_0",
    ]
    assert nodes["/hc/inject/Reshape"].inputs[1] == f"/model/constants/INT64/[0, 0, {HC_COUNT * HIDDEN}]"
    # The residual add uses the pre-norm stream, never the normalized one.
    assert nodes["/hc/inject/Add"].inputs == ["stream", "/hc/inject/Reshape/output_0"]
    assert output == "/hc/inject/Add/output_0"


#####################################################################################
# PLE / n-gram
#####################################################################################


def test_ple_emits_ngram_hash_mapping_v2_with_the_documented_contract():
    model = _make_recording_model()
    model._make_ple_layer(0, _ple_module(), "stream")

    node = _named(model)["/model/layers.0/ple/NGramHashMapping"]
    assert node.domain == "com.microsoft"
    assert node.attrs == {
        "ngram_size": NGRAM_SIZE,
        "heads_per_ngram": HEADS_PER_NGRAM,
        "eos_token_id": 7,
        "version": 2,
    }
    assert node.inputs == [
        "input_ids",
        "model.layers.0.ple.layer_multipliers",
        "model.layers.0.ple.ngram_heads_vocab_sizes",
        "model.layers.0.ple.ngram_heads_offsets",
        "past_key_values.0.ple_tokens",
    ]
    assert node.outputs == ["/model/layers.0/ple/NGramHashMapping/output_0", "present.0.ple_tokens"]


def test_ple_hash_constants_are_int64_and_sized_by_the_ngram_geometry():
    model = _make_recording_model()
    model._make_ple_layer(0, _ple_module(), "stream")

    inits = {init.name: init.tensor for init in model.initializers}
    assert inits["model.layers.0.ple.layer_multipliers"].dtype == torch.int64
    assert tuple(inits["model.layers.0.ple.layer_multipliers"].shape) == (NGRAM_SIZE,)
    assert tuple(inits["model.layers.0.ple.ngram_heads_vocab_sizes"].shape) == (NGRAM_HEADS,)
    assert tuple(inits["model.layers.0.ple.ngram_heads_offsets"].shape) == (NGRAM_HEADS,)

    sizes, offsets, _ = qwen4exp.ngram_head_vocab_layout(1000, NGRAM_HEADS, 0)
    assert inits["model.layers.0.ple.ngram_heads_vocab_sizes"].tolist() == sizes
    assert inits["model.layers.0.ple.ngram_heads_offsets"].tolist() == offsets


def test_ple_hash_constants_are_distinct_per_ple_layer_index():
    model = _make_recording_model(ple_layers=(0, 2))
    model._make_ple_layer(0, _ple_module(), "stream")
    model._make_ple_layer(2, _ple_module(), "stream")

    inits = {init.name: init.tensor for init in model.initializers}
    assert inits["model.layers.0.ple.layer_multipliers"].tolist() != inits["model.layers.2.ple.layer_multipliers"].tolist()
    assert inits["model.layers.0.ple.ngram_heads_offsets"].tolist() != inits["model.layers.2.ple.ngram_heads_offsets"].tolist()


def test_ple_emits_engram_gate_with_key_query_value_order():
    model = _make_recording_model()
    model._make_ple_layer(0, _ple_module(), "stream")

    node = _named(model)["/model/layers.0/ple/EngramGate"]
    assert node.domain == "com.microsoft"
    assert node.attrs == {"num_streams": HC_COUNT, "hidden_size": HIDDEN, "epsilon": 1e-6}
    key, query, value = node.inputs
    assert key.startswith("/model/layers.0/ple/norm_key/")
    assert query.startswith("/model/layers.0/ple/norm_query/")
    assert value == "/model/layers.0/ple/value_proj/MatMul/output_0"


def test_ple_query_norm_reads_the_residual_stream_and_key_norm_reads_the_embedding():
    model = _make_recording_model()
    model._make_ple_layer(0, _ple_module(), "stream")

    nodes = _named(model)
    assert nodes["/model/layers.0/ple/norm_query/group/Reshape"].inputs[0] == "stream"
    assert (
        nodes["/model/layers.0/ple/norm_key/group/Reshape"].inputs[0]
        == "/model/layers.0/ple/key_proj/MatMul/output_0"
    )


def test_ple_short_conv_is_dilated_depthwise_and_carries_state():
    model = _make_recording_model()
    model._make_ple_layer(0, _ple_module(), "stream")

    node = _named(model)["/model/layers.0/ple/conv1d/ShortConvWithState"]
    assert node.domain == "com.microsoft"
    assert node.attrs == {
        "dilation": NGRAM_SIZE,
        "activation": "silu",
        "group": HC_COUNT * HIDDEN,
    }, "the receptive field is dilated by ngram_size and the conv is fully depthwise"
    assert node.inputs == [
        "/model/layers.0/ple/conv1d/in/Transpose/output_0",
        "model.layers.0.ple.conv1d.weight",
        "past_key_values.0.ple_conv_state",
    ]
    assert node.outputs == ["/model/layers.0/ple/conv1d/ShortConvWithState/output_0", "present.0.ple_conv_state"]

    # NCL in, NCL out: the conv is wrapped in a transpose pair.
    assert _named(model)["/model/layers.0/ple/conv1d/in/Transpose"].attrs["perm"] == [0, 2, 1]
    assert _named(model)["/model/layers.0/ple/conv1d/out/Transpose"].attrs["perm"] == [0, 2, 1]


def test_ple_output_adds_the_conv_branch_to_the_ungated_engram_value():
    model = _make_recording_model()
    output = model._make_ple_layer(0, _ple_module(), "stream")

    add = _named(model)["/model/layers.0/ple/Add"]
    assert add.inputs == [
        "/model/layers.0/ple/EngramGate/output_0",
        "/model/layers.0/ple/conv1d/out/Transpose/output_0",
    ], "the residual is the gate output, not the normalized conv input"
    assert output == "/model/layers.0/ple/Add/output_0"


def test_ple_token_state_is_one_shorter_than_the_ngram_order():
    model = _make_recording_model()
    model._make_ple_layer(0, _ple_module(), "stream")

    present = next(value for value in model.values if value.name == "present.0.ple_tokens")
    assert present.dtype == ir.DataType.INT32
    assert present.shape == ["batch_size", NGRAM_SIZE - 1]


#####################################################################################
# QSA attention
#####################################################################################


def test_indexer_splits_a_single_projection_into_query_and_key():
    model = _make_recording_model()
    q_normed, k_normed = model._make_qsa_indexer(1, _indexer_module(), "hidden")

    split = _named(model)["/model/layers.1/attn/indexer/Split"]
    assert split.op_type == "Split"
    assert split.attrs["axis"] == -1
    assert split.inputs[1] == (
        f"/model/constants/INT64/[{INDEXER_HEADS * INDEXER_HEAD_DIM}, {INDEXER_KV_HEADS * INDEXER_HEAD_DIM}]"
    )
    assert q_normed.startswith("/model/layers.1/attn/indexer/q_layernorm/")
    assert k_normed.startswith("/model/layers.1/attn/indexer/k_layernorm/")


def test_indexer_norms_are_per_head_over_indexer_head_dim():
    model = _make_recording_model()
    model._make_qsa_indexer(1, _indexer_module(), "hidden")

    nodes = _named(model)
    assert nodes["/model/layers.1/attn/indexer/q_layernorm/Reshape"].inputs[1] == (
        f"/model/constants/INT64/[0, 0, {INDEXER_HEADS}, {INDEXER_HEAD_DIM}]"
    )
    assert nodes["/model/layers.1/attn/indexer/k_layernorm/Reshape"].inputs[1] == (
        f"/model/constants/INT64/[0, 0, {INDEXER_KV_HEADS}, {INDEXER_HEAD_DIM}]"
    )
    norm = nodes["/model/layers.1/attn/indexer/q_layernorm/SimplifiedLayerNormalization"]
    assert norm.attrs["axis"] == -1 and norm.attrs["epsilon"] == 1e-6


def test_indexer_norm_weights_get_the_plus_one_offset():
    model = _make_recording_model()
    indexer = _indexer_module()
    indexer.q_layernorm.weight = torch.full((INDEXER_HEAD_DIM,), 0.5)
    model._make_qsa_indexer(1, indexer, "hidden")

    weight = next(init for init in model.initializers if init.name == "/model/layers.1/attn/indexer/q_layernorm/weight")
    assert torch.allclose(weight.tensor, torch.full((INDEXER_HEAD_DIM,), 1.5))


def test_non_paged_sparse_attention_input_and_output_order():
    model = _make_recording_model(use_paged_attention=False)
    model._make_qsa_attention_op("/attn", 0, "q", "k", "v", "iq", "ik")

    node = _named(model)["/attn"]
    assert node.op_type == "QwenSparseAttention"
    assert node.domain == "com.microsoft"
    assert node.inputs == [
        "q",
        "k",
        "v",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "seqlens_k/output_0",
        "total_seq_len/output_0",
        "",
        "",
        "iq",
        "ik",
        "past_key_values.0.indexer_key",
    ]
    assert node.outputs == ["/attn/output_0", "present.0.key", "present.0.value", "present.0.indexer_key"]


def test_paged_sparse_attention_input_and_output_order():
    model = _make_recording_model(use_paged_attention=True)
    model._make_qsa_attention_op("/attn", 0, "q", "k", "v", "iq", "ik")

    node = _named(model)["/attn"]
    assert node.op_type == "SparsePagedAttention"
    assert node.inputs == [
        "q",
        "k",
        "v",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
        "block_table",
        "attention_metadata",
        "",
        "",
        "iq",
        "ik",
        "past_key_values.0.indexer_key",
    ]
    # Paged attention updates the KV cache in place, so only the indexer key is a real output.
    assert node.outputs == ["/attn/output_0", "present.0.indexer_key"]


@pytest.mark.parametrize("use_paged_attention", [False, True])
def test_sparse_attention_carries_the_full_indexer_geometry(use_paged_attention):
    model = _make_recording_model(use_paged_attention=use_paged_attention)
    model._make_qsa_attention_op("/attn", 0, "q", "k", "v", "iq", "ik")

    attrs = _named(model)["/attn"].attrs
    assert attrs["num_heads"] == NUM_HEADS
    assert attrs["kv_num_heads"] == NUM_KV_HEADS
    assert attrs["do_rotary"] == 0, "mRoPE is applied in the graph, not inside the kernel"
    assert attrs["indexer_num_heads"] == INDEXER_HEADS
    assert attrs["indexer_kv_num_heads"] == INDEXER_KV_HEADS
    assert attrs["indexer_head_size"] == INDEXER_HEAD_DIM
    assert attrs["indexer_token_budget"] == 512
    assert attrs["indexer_compress_ratio"] == 16


def test_attention_op_falls_back_to_the_dense_path_without_an_indexer():
    model = _make_recording_model(use_paged_attention=False)
    model._pending_indexer = None
    calls = []
    model.make_group_query_attention = MethodType(lambda self, name, **kw: calls.append((name, kw)), model)

    model.make_attention_op("/attn", layer_id=0, q_path="q", k_path="k", v_path="v")

    assert len(calls) == 1, "layers without a QSA indexer must keep using GroupQueryAttention"
    assert not any(node.op_type.endswith("SparseAttention") for node in model.nodes)


def test_attention_op_uses_the_sparse_op_when_the_indexer_is_pending():
    model = _make_recording_model(use_paged_attention=False)
    model._pending_indexer = ("iq", "ik")

    model.make_attention_op("/attn", layer_id=0, q_path="q", k_path="k", v_path="v")

    assert _ops(model) == ["QwenSparseAttention"]


#####################################################################################
# Sequence dimension
#####################################################################################


@pytest.mark.parametrize("use_paged_attention, expected", [(False, "sequence_length"), (True, "num_tokens")])
def test_sequence_dim_follows_the_packing_mode(use_paged_attention, expected):
    model = _make_recording_model(use_paged_attention=use_paged_attention)
    assert model.sequence_dim_name() == expected


#####################################################################################
# Checkpoint compatibility shims
#####################################################################################


def _resolve(layer_types, num_layers=4):
    """Run `_resolve_layer_types` against a config pair shaped like a real checkpoint."""
    text_config = SimpleNamespace(layer_types=list(layer_types), num_hidden_layers=num_layers)
    config = SimpleNamespace(text_config=text_config, layer_types=list(layer_types))
    model = object.__new__(Qwen4ExpTextModel)
    resolved = Qwen4ExpTextModel._resolve_layer_types(model, config, num_layers)
    return resolved, config, text_config


def test_qwen_sparse_attention_layers_are_normalized_to_full_attention():
    """`Qwen4ExpTextConfig` renames full attention to `qwen_sparse_attention`; the shared
    Qwen3.5 resolver only accepts the canonical names, so the alias has to be rewritten."""
    resolved, _, _ = _resolve(["linear_attention", "qwen_sparse_attention"] * 2)

    assert resolved == ["linear_attention", "full_attention", "linear_attention", "full_attention"]


def test_normalization_rewrites_both_config_objects_in_place():
    # The base builder re-reads `layer_types` off whichever config it flattened, so leaving a
    # stale alias behind on either object resurfaces as an unsupported-layer error later on.
    _, config, text_config = _resolve(["qwen_sparse_attention", "linear_attention"], num_layers=2)

    assert config.layer_types == ["full_attention", "linear_attention"]
    assert text_config.layer_types == ["full_attention", "linear_attention"]


def test_already_canonical_layer_types_survive_normalization():
    resolved, _, _ = _resolve(["full_attention", "linear_attention"], num_layers=2)

    assert resolved == ["full_attention", "linear_attention"]


def test_packed_experts_are_wrapped_so_the_nvfp4_probe_can_iterate():
    """`Qwen35MoeTextModel.make_moe` probes `next(iter(mlp.experts), None)` for NVFP4 weights.
    Qwen4Exp stores experts as packed 3-D tensors on a non-iterable module, so the view has to
    make iteration empty rather than raising."""
    packed = SimpleNamespace(gate_up_proj=object(), down_proj=object())
    mlp = SimpleNamespace(experts=packed, gate=object(), shared_expert=object())

    view = qwen4exp._PackedMoeView(mlp)

    assert next(iter(view.experts), None) is None
    assert list(view.experts) == []
    assert view.experts.gate_up_proj is packed.gate_up_proj
    assert view.experts.down_proj is packed.down_proj
    assert view.gate is mlp.gate and view.shared_expert is mlp.shared_expert
