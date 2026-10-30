# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for the Qwen3.8 Flash ``merge_shared_input_projections`` builder option.

The tests export a small synthetic checkpoint (see ``_qwen38_tiny``) through the real builder entry points, once
with and once without the option, and compare the two graphs. The merged projections are checked structurally
(operator counts, concatenated weights, unchanged graph interface) and numerically: the sub-graph between each
shared activation and the projection outputs is run for both exports and must agree.

The export-based tests need a transformers release that provides ``qwen4_exp`` and names the sparse-attention
layer type ``indexed_attention``; they are skipped otherwise. The tests that build graph fragments from mocks
always run.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.utils import Extractor

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))
sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

import _qwen38_tiny as tiny
import builder

from models.builders.qwen3_8 import Qwen4ExpModel, Qwen4ExpTextModel

NUM_LINEAR_LAYERS = tiny.TINY["num_layers"] - 1
QSA_LAYER = tiny.TINY["num_layers"] - 1
NUM_LAYERS = tiny.TINY["num_layers"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    path = tiny.write_checkpoint(tmp_path_factory.mktemp("qwen38_checkpoint"))
    layer_types = builder.AutoConfig.from_pretrained(str(path)).text_config.layer_types
    if "indexed_attention" not in layer_types:
        pytest.skip(f"The installed transformers names the sparse-attention layer type {layer_types[-1]!r}.")
    return path


@pytest.fixture(scope="module")
def exports(tmp_path_factory, checkpoint):
    """Export every variant once; the exports are read-only for the tests."""
    root = tmp_path_factory.mktemp("qwen38_exports")
    results = {}

    def export(name, precision="bf16", extra_options=(), **options):
        out = root / name
        extra = [
            "text_only=true",
            "exclude_mtp=true",
            *(f"{key}={str(value).lower()}" for key, value in options.items()),
            *extra_options,
        ]
        parsed = builder.parse_extra_options(
            "", str(checkpoint), str(out), precision, "cuda", str(out / "cache"), extra
        )
        builder.create_model("", str(checkpoint), str(out), precision, "cuda", str(out / "cache"), **parsed)
        results[name] = out / "model.onnx"

    for paged in (False, True):
        suffix = "paged" if paged else "dense"
        export(f"default_{suffix}", use_paged_attention=paged)
        export(f"merged_{suffix}", use_paged_attention=paged, merge_shared_input_projections=True)
    # Excluding a Q/K/V projection from quantization is what makes the packed Q/K/V of the sparse layer unavailable.
    export(
        "merged_unpacked_qkv",
        use_paged_attention=True,
        merge_shared_input_projections=True,
        extra_options=[f"nodes_to_exclude=/model/layers.{QSA_LAYER}/attn/q_proj/MatMul"],
    )
    export("default_fp32", precision="fp32")
    export("merged_fp32", precision="fp32", merge_shared_input_projections=True)
    return results


def load(path):
    return onnx.load(str(path))


def op_counts(model):
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def initializers(model):
    return {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}


@pytest.mark.parametrize("layout", ["dense", "paged"])
def test_default_export_already_has_the_exact_graph_hygiene(exports, layout):
    model = load(exports[f"default_{layout}"])
    values = initializers(model)

    gated_delta_nets = [node for node in model.graph.node if node.op_type == "GatedDeltaNet"]
    assert len(gated_delta_nets) == NUM_LINEAR_LAYERS
    for node in gated_delta_nets:
        attributes = {attribute.name: attribute for attribute in node.attribute}
        assert attributes["qk_l2_norm"].i == 1
        # a_log and dt_bias are float32 initializers consumed by the op directly, so no Cast or Softplus is emitted
        a_log, dt_bias = node.input[7], node.input[8]
        assert values[a_log].dtype == np.float32
        assert values[dt_bias].dtype == np.float32

    # Nothing is left to constant-fold: every Cast and Add has a computed input.
    produced = {output for node in model.graph.node for output in node.output if node.op_type != "Constant"}
    for node in model.graph.node:
        if node.op_type in {"Cast", "Add"}:
            assert any(name in produced for name in node.input), node.name

    transposes = [node for node in model.graph.node if node.op_type == "Transpose"]
    # Only the dense layout transposes, around the channels-first convolution; the recurrent state never does.
    assert len(transposes) == (2 * NUM_LINEAR_LAYERS if layout == "dense" else 0)


@pytest.mark.parametrize("layout", ["dense", "paged"])
def test_default_export_has_no_merged_projection_nodes(exports, layout):
    names = [node.name for node in load(exports[f"default_{layout}"]).graph.node]
    assert not [name for name in names if "qkv_z_a_b_proj" in name or "qkv_gate_indexer_proj" in name]
    assert not [name for name in names if "router_shared_gate_up_proj" in name]


@pytest.mark.parametrize("layout", ["dense", "paged"])
def test_merged_export_changes_only_the_projection_nodes(exports, layout):
    default = load(exports[f"default_{layout}"])
    merged = load(exports[f"merged_{layout}"])

    default_counts, merged_counts = op_counts(default), op_counts(merged)
    # GatedDeltaNet layers 2 -> 1 MatMul, the sparse-attention layer 3 -> 1, and every MoE layer 2 -> 1.
    expected_matmul_delta = -(NUM_LINEAR_LAYERS + 2 + NUM_LAYERS)
    # GatedDeltaNet 2 -> 1 Split, sparse attention 0 -> 1, MoE router/shared gate-up Splits are exchanged 1 for 1.
    expected_split_delta = -NUM_LINEAR_LAYERS + 1
    for op_type in set(default_counts) | set(merged_counts):
        delta = merged_counts.get(op_type, 0) - default_counts.get(op_type, 0)
        expected = {"MatMul": expected_matmul_delta, "Split": expected_split_delta}.get(op_type, 0)
        if op_type == "Constant":
            continue
        assert delta == expected, op_type

    assert [value.name for value in merged.graph.input] == [value.name for value in default.graph.input]
    assert [value.name for value in merged.graph.output] == [value.name for value in default.graph.output]
    # Every tensor name an unmerged export produces survives, except the three wide MatMul outputs that were
    # only ever intermediates of a former pair of projections.
    produced = lambda model: {  # noqa: E731
        name for node in model.graph.node if node.op_type != "Constant" for name in node.output if name
    }
    assert produced(default) - produced(merged) == {
        f"/model/layers.{layer_id}/{name}/output_0"
        for layer_id in range(NUM_LAYERS)
        for name in ("linear_attn/qkv_z_proj/MatMul", "linear_attn/a_b_proj/MatMul", "mlp/gate_up_proj/MatMul")
        if not (layer_id == QSA_LAYER and name.startswith("linear_attn"))
    }


@pytest.mark.parametrize("layout", ["dense", "paged"])
def test_merged_graph_has_no_dangling_inputs(exports, layout):
    model = load(exports[f"merged_{layout}"])
    onnx.checker.check_model(model, check_custom_domain=False)
    available = {value.name for value in model.graph.input} | {tensor.name for tensor in model.graph.initializer}
    for node in model.graph.node:
        for name in node.input:
            assert not name or name in available, f"{node.name} reads undefined {name}"
        available.update(node.output)


def matmul_weights(model):
    """Weight of every MatMul node keyed by node name; initializer names differ between the two exports."""
    values = initializers(model)
    return {
        node.name: values[node.input[1]]
        for node in model.graph.node
        if node.op_type == "MatMul" and node.input[1] in values
    }


def test_merged_weights_are_the_concatenated_default_weights(exports):
    default = matmul_weights(load(exports["default_paged"]))
    merged = matmul_weights(load(exports["merged_paged"]))

    for layer_id in range(NUM_LINEAR_LAYERS):
        prefix = f"/model/layers.{layer_id}/linear_attn"
        expected = np.concatenate(
            [default[f"{prefix}/qkv_z_proj/MatMul"], default[f"{prefix}/a_b_proj/MatMul"]], axis=1
        )
        np.testing.assert_array_equal(merged[f"{prefix}/qkv_z_a_b_proj/MatMul"], expected)

    prefix = f"/model/layers.{QSA_LAYER}/attn"
    expected = np.concatenate(
        [
            default[f"{prefix}/qkv_proj/MatMul"],
            default[f"{prefix}/gate_proj/MatMul"],
            default[f"{prefix}/indexer/index_qk_proj/MatMul"],
        ],
        axis=1,
    )
    np.testing.assert_array_equal(merged[f"{prefix}/qkv_gate_indexer_proj/MatMul"], expected)

    for layer_id in range(NUM_LAYERS):
        expected = np.concatenate(
            [
                default[f"/model/layers.{layer_id}/moe/router/MatMul"],
                default[f"/model/layers.{layer_id}/mlp/gate_up_proj/MatMul"],
            ],
            axis=1,
        )
        np.testing.assert_array_equal(
            merged[f"/model/layers.{layer_id}/moe/router_shared_gate_up_proj/MatMul"], expected
        )

    # The shared expert's one-wide output gate keeps its own MatMul.
    assert "/model/layers.0/shared_expert_gate/MatMul" in merged


def group_boundaries(layer_id):
    """(shared activation, projection MatMul names) for each group of the tiny model's layer."""
    attn_input = f"/model/layers.{layer_id}/attn_hyper_connection/mixed/Mean/output_0"
    mlp_input = f"/model/layers.{layer_id}/mlp_hyper_connection/mixed/Mean/output_0"
    if layer_id == QSA_LAYER:
        attn = [f"attn/{name}" for name in ("qkv_proj", "gate_proj", "indexer/index_qk_proj")]
    else:
        attn = [f"linear_attn/{name}_proj" for name in ("qkv", "z", "a", "b")]
    moe = ["moe/router", "mlp/gate_proj", "mlp/up_proj"]
    outputs = lambda names: [f"/model/layers.{layer_id}/{name}/MatMul/output_0" for name in names]  # noqa: E731
    return [(attn_input, outputs(attn)), (mlp_input, outputs(moe))]


def extract(path, inputs, outputs, wrap_dtype=None):
    """Sub-model from ``inputs`` to ``outputs``; optionally float32-wrapped around a reduced-precision core."""
    sub = Extractor(load(path)).extract_model(inputs, outputs)
    if wrap_dtype is None:
        return sub
    graph = sub.graph
    inner = [output.name for output in graph.output]
    for node in graph.node:
        node.output[:] = [name + "_core" if name in inner else name for name in node.output]
    nodes = [helper.make_node("Cast", ["x"], [inputs[0]], to=wrap_dtype)]
    nodes += list(graph.node)
    nodes += [helper.make_node("Cast", [name + "_core"], [name], to=TensorProto.FLOAT) for name in inner]
    wrapped = helper.make_graph(
        nodes,
        "wrapped",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in inner],
        list(graph.initializer),
    )
    return helper.make_model(wrapped, opset_imports=sub.opset_import)


def run_groups(default_path, merged_path, provider, wrap_dtype=None, token_counts=(1, 5, 33)):
    rng = np.random.default_rng(0)
    for layer_id in range(NUM_LAYERS):
        for shared_input, outputs in group_boundaries(layer_id):
            sessions = [
                ort.InferenceSession(
                    extract(path, [shared_input], outputs, wrap_dtype).SerializeToString(), providers=[provider]
                )
                for path in (default_path, merged_path)
            ]
            input_name = sessions[0].get_inputs()[0].name
            rank = len(sessions[0].get_inputs()[0].shape)
            for tokens in token_counts:
                shape = (tokens, tiny.TINY["hidden"]) if rank == 2 else (1, tokens, tiny.TINY["hidden"])
                x = rng.standard_normal(shape).astype(np.float32)
                expected = sessions[0].run(None, {input_name: x})
                actual = sessions[1].run(None, {input_name: x})
                assert len(expected) == len(actual) == len(outputs)
                yield layer_id, outputs, expected, actual


def test_merged_projections_match_default_on_cpu_fp32(exports):
    checked = 0
    for _, _, expected, actual in run_groups(exports["default_fp32"], exports["merged_fp32"], "CPUExecutionProvider"):
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_allclose(right, left, rtol=1e-5, atol=1e-6)
            checked += 1
    assert checked


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="Requires CUDA")
def test_merged_projections_match_default_on_cuda_bf16(exports):
    checked = 0
    for _, _, expected, actual in run_groups(
        exports["default_paged"], exports["merged_paged"], "CUDAExecutionProvider", wrap_dtype=TensorProto.BFLOAT16
    ):
        for left, right in zip(expected, actual, strict=True):
            # Merging changes the GEMM shape, not the dot products; allow one bf16 rounding step.
            np.testing.assert_allclose(right, left, rtol=2**-7, atol=2**-7)
            checked += 1
    assert checked


def make_projection(rows, columns=16, **attributes):
    return SimpleNamespace(weight=torch.ones(rows, columns), **attributes)


def make_group_model():
    model = object.__new__(Qwen4ExpTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.onnx_dtype = ir.DataType.FLOAT16
    model.hidden_rows_dim = "num_tokens"
    model.calls = []

    def record(method_name):
        def call(self, *args, **kwargs):
            self.calls.append((method_name, args, kwargs))
            return args[1]

        return call

    model.make_matmul = MethodType(record("make_matmul"), model)
    model.make_split = MethodType(record("make_split"), model)
    model.make_hidden_state_shape = MethodType(lambda self, last_dim=None, **_: ["num_tokens", last_dim], model)
    return model


def test_merged_projection_emits_one_matmul_and_one_split_with_the_original_output_names():
    model = make_group_model()

    model.make_merged_projections("/g", "x", [make_projection(8), make_projection(24)], ["/a", "/b"])

    (matmul, matmul_args, _), (split, split_args, _) = model.calls
    assert matmul == "make_matmul"
    assert matmul_args[0].weight.shape == (32, 16)
    assert matmul_args[1:] == ("/g/MatMul", "x")
    assert split == "make_split"
    assert split_args[0] == "/g/Split"
    assert split_args[1] == ["/g/MatMul/output_0", "/model/constants/INT64/[8, 24]"]
    assert split_args[2] == ["/a/output_0", "/b/output_0"]


@pytest.mark.parametrize(
    ("second", "message"),
    [
        (SimpleNamespace(qweight=torch.ones(8, 4), weight=torch.ones(8, 16)), "dense, bias-free"),
        (make_projection(8, bias=torch.ones(8)), "dense, bias-free"),
        (make_projection(12), "multiple of 8"),
        (make_projection(8, exclude_from_quantization=True), "quantization exclusion"),
    ],
)
def test_merged_projection_rejects_what_it_cannot_preserve(second, message):
    model = make_group_model()
    with pytest.raises(ValueError, match=message):
        model.make_merged_projections("/g", "x", [make_projection(8), second], ["/a", "/b"])
    assert not model.calls


def test_zero_bias_is_dropped_like_the_unmerged_path():
    model = make_group_model()
    model.make_merged_projections(
        "/g", "x", [make_projection(8, bias=torch.zeros(8)), make_projection(8)], ["/a", "/b"]
    )
    assert [call[0] for call in model.calls] == ["make_matmul", "make_split"]


def make_sparse_group_model():
    model = make_group_model()
    model.merge_shared_input_projections = True
    model.attention_attrs = {"use_packed_matmul": True}
    model.num_attn_heads = 2
    model.head_size = 4
    model.q_size = 8
    model.is_qkv_projection_packable = MethodType(lambda self, layer_id, attention: True, model)

    def select_projection_outputs(self, projection, rows):
        bias = getattr(projection, "bias", None)
        return make_projection(rows.numel(), bias=None if bias is None else bias[rows])

    model.select_projection_outputs = MethodType(select_projection_outputs, model)
    model.make_packed_matmul_class = MethodType(
        lambda self, *projections: make_projection(sum(projection.weight.shape[0] for projection in projections)),
        model,
    )
    return model


def make_sparse_attention(**biases):
    # The query projection carries the interleaved output gate, so it has twice the query rows.
    return SimpleNamespace(
        q_proj=make_projection(16, bias=biases.get("q")),
        k_proj=make_projection(8, bias=biases.get("k")),
        v_proj=make_projection(8, bias=biases.get("v")),
    )


def test_merged_sparse_projection_emits_one_group_for_qkv_gate_and_indexer():
    model = make_sparse_group_model()

    model.make_attention_input_proj(3, make_sparse_attention(), "x", indexer_proj=make_projection(8))

    (matmul, matmul_args, _), (split, split_args, _) = model.calls
    assert matmul == "make_matmul"
    # Packed Q/K/V (8 + 8 + 8 rows), the output gate (8) and the indexer (8).
    assert matmul_args[0].weight.shape == (24 + 8 + 8, 16)
    assert matmul_args[1:] == ("/model/layers.3/attn/qkv_gate_indexer_proj/MatMul", "x")
    assert split == "make_split"
    assert split_args[1][1] == "/model/constants/INT64/[24, 8, 8]"
    assert split_args[2] == [
        "/model/layers.3/attn/qkv_proj/MatMul/output_0",
        "/model/layers.3/attn/gate_proj/MatMul/output_0",
        "/model/layers.3/attn/indexer/index_qk_proj/MatMul/output_0",
    ]
    assert model.attention_attrs["q_path"] == "/model/layers.3/attn/qkv_proj/MatMul/output_0"
    assert model.attention_attrs["gate_path"] == "/model/layers.3/attn/gate_proj/MatMul/output_0"
    assert model.attention_attrs["indexer_qk_path"] == "/model/layers.3/attn/indexer/index_qk_proj/MatMul/output_0"


def test_merged_sparse_projection_rejects_an_indexer_that_reads_gathered_rows():
    model = make_sparse_group_model()
    with pytest.raises(ValueError, match="gathered rows"):
        model.make_attention_input_proj(
            3, make_sparse_attention(), "x", indexer_proj=make_projection(8), indexer_root_input="gathered_rows"
        )
    assert not model.calls


@pytest.mark.parametrize("projection", ["q", "k", "v"])
def test_merged_sparse_projection_rejects_qkv_bias(projection):
    model = make_sparse_group_model()
    attention = make_sparse_attention(**{projection: torch.ones(16 if projection == "q" else 8)})
    with pytest.raises(ValueError, match="bias-free Q/K/V"):
        model.make_attention_input_proj(3, attention, "x", indexer_proj=make_projection(8))
    assert not model.calls


def test_option_requires_a_floating_point_export(checkpoint):
    parsed_config = builder.AutoConfig.from_pretrained(str(checkpoint))
    for dtype in (ir.DataType.INT4, ir.DataType.UINT8):
        with pytest.raises(ValueError, match="fp32, fp16 or bf16"):
            Qwen4ExpTextModel(
                parsed_config, ir.DataType.BFLOAT16, dtype, "cuda", None, {"merge_shared_input_projections": True}
            )


def test_mtp_head_is_exported_without_merged_projections(monkeypatch):
    captured = {}

    class RecordingMtp:
        def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options, **kwargs):
            captured.update(extra_options)

    monkeypatch.setattr(importlib.import_module("models.builders.qwen3_8.model"), "Qwen4ExpMTPTextModel", RecordingMtp)
    model = object.__new__(Qwen4ExpModel)
    model.mtp_attrs = {}
    model.decoder = SimpleNamespace(quant_type=None)
    model.resolve_mtp_model_config = MethodType(lambda self, extra_options: None, model)

    model.make_mtp_model(
        None, ir.DataType.BFLOAT16, ir.DataType.BFLOAT16, "cuda", None, {"merge_shared_input_projections": True}
    )

    assert "merge_shared_input_projections" not in captured
    assert captured["text_only"] is True


def test_unpacked_qkv_keeps_the_sparse_attention_projections_separate(exports):
    names = [node.name for node in load(exports["merged_unpacked_qkv"]).graph.node]
    assert not [name for name in names if "qkv_gate_indexer_proj" in name]
    assert f"/model/layers.{QSA_LAYER}/attn/q_proj/MatMul" in names
    assert [name for name in names if "qkv_z_a_b_proj" in name]
    assert [name for name in names if "router_shared_gate_up_proj" in name]
