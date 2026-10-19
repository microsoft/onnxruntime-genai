# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import SimpleNamespace

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from _builder_test_utils import load_builder_module
from transformers import LlamaConfig

base = load_builder_module("base")
qwen = load_builder_module("qwen3_8")
graph_module = load_builder_module("qwen3_8_graph")


@pytest.mark.parametrize("reason", ["zero_points", "g_idx", "shared", "fp32"])
@pytest.mark.parametrize("bits", [4, 8])
def test_matmul_prepack_preserves_unsupported_payloads(monkeypatch, reason, bits):
    model = base.Model.__new__(base.Model)
    model.ep = "cuda"
    model.io_dtype = ir.DataType.FLOAT if reason == "fp32" else ir.DataType.FLOAT16
    model.matmul_attrs = {"weights_prepacked": 1}
    model.quant_attrs = {"is_symmetric": True}
    weight = np.arange(64 * 32 * bits // 8, dtype=np.uint8).reshape(64, 1, 32 * bits // 8)
    inputs = ["x", "weight", "scales", "", ""]
    initializers = [onnx.numpy_helper.from_array(weight, "weight")]
    if reason == "zero_points":
        inputs[3] = "zp"
        initializers.append(onnx.numpy_helper.from_array(np.zeros((64, 1), np.uint8), "zp"))
    elif reason == "g_idx":
        inputs[4] = "g_idx"
    nodes = [
        onnx.helper.make_node(
            "MatMulNBits", inputs, ["y"], domain="com.microsoft", bits=bits, block_size=32, K=32, N=64
        )
    ]
    if reason == "shared":
        nodes.extend(
            [
                onnx.helper.make_node("Reshape", ["weight", "shape"], ["embedding_weights"]),
                onnx.helper.make_node(
                    "GatherBlockQuantized",
                    ["embedding_weights", "ids", "scales"],
                    ["embedding"],
                    domain="com.microsoft",
                ),
            ]
        )
    proto = onnx.helper.make_model(onnx.helper.make_graph(nodes, "packing", [], [], initializers))
    original = proto.SerializeToString()

    def reject_prepack(*args):
        pytest.fail("Unsupported weight buffers must never be repacked")

    monkeypatch.setattr(base.CudaQuantizer, "prepack_matmulnbits_weight", reject_prepack)
    model.prepack_matmulnbits_weights(proto)
    assert proto.SerializeToString() == original


@pytest.mark.parametrize("frames", [1, 2])
def test_vision_attention_never_mixes_frames(frames):
    model = qwen.Qwen4ExpVisionModel.__new__(qwen.Qwen4ExpVisionModel)
    graph_module.Qwen4ExpGraphModel.__init__(model, ir.DataType.FLOAT, "vision.onnx", "vision_test")
    model.hidden_size = model.head_size = 4
    model.num_heads = 1
    for name, shape, dtype in (
        ("x", ["num_patches", 4], ir.DataType.FLOAT),
        ("cos", ["num_patches", 4], ir.DataType.FLOAT),
        ("sin", ["num_patches", 4], ir.DataType.FLOAT),
        ("/vision/positions/Range/output_0", ["num_patches"], ir.DataType.INT64),
    ):
        model.graph.inputs.append(model.make_value(name, dtype, shape))
    model.make_initializer(np.asarray(4, np.int64), "/vision/grid/frame_size/Mul/output_0")
    qkv = torch.nn.Linear(4, 12, bias=False)
    qkv.weight.data.zero_()
    qkv.weight.data[8:] = torch.eye(4)
    proj = torch.nn.Linear(4, 4, bias=False)
    proj.weight.data.copy_(torch.eye(4))
    output = model.make_attention(0, SimpleNamespace(qkv=qkv, proj=proj, scaling=0.5), "x", "cos", "sin")
    model.graph.outputs.append(model.values[output])
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    values = np.repeat(np.arange(frames, dtype=np.float32) * 2 + 1, 4)[:, None].repeat(4, axis=1)
    result = session.run(
        None,
        {
            "x": values,
            "cos": np.ones_like(values),
            "sin": np.zeros_like(values),
            "/vision/positions/Range/output_0": np.arange(frames * 4, dtype=np.int64),
        },
    )[0]
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize("ep", ["cpu", "dml", "trt-rtx"])
def test_dense_qwen35_keeps_linear_attention_fallback(ep):
    model = load_builder_module("qwen3_5").Qwen35TextModel.__new__(load_builder_module("qwen3_5").Qwen35TextModel)
    model.ep = ep
    model.use_paged_attention = False
    model.context_length_attrs = {"state_update_capacity": 0}
    model.validate_gated_delta_net_options(0, ep)
    calls = []
    model.make_gated_delta_net_expansion = lambda *args: calls.append(args) or "linear_attention"
    assert model.make_gated_delta_net_layer(0, "weights", "conv", "b", "a") == "linear_attention"
    assert calls == [(0, "weights", "conv", "b", "a")]


@pytest.mark.parametrize("window,capacity", [(20, 0), (3, 4)])
def test_recurrent_window_rejects_overflow_or_conflicting_capacity(window, capacity):
    model = load_builder_module("qwen3_5").Qwen35TextModel.__new__(load_builder_module("qwen3_5").Qwen35TextModel)
    model.use_paged_attention = True
    model.context_length_attrs = {"state_update_capacity": capacity}
    with pytest.raises(ValueError, match="must not exceed|must equal"):
        model.validate_gated_delta_net_options(window, "cuda")


def test_qwen38_mtp_requantization_is_rejected_before_loading():
    from loaders.mtp import Qwen4ExpMTPModel  # noqa: PLC0415

    with pytest.raises(ValueError, match="checkpoint_policy=requantize"):
        Qwen4ExpMTPModel.from_modelopt(object(), None, False)


@pytest.mark.parametrize("dense", [True, False])
def test_modelopt_gate_exclusions_are_specific_to_model_family(dense):
    from models.loaders.modelopt import ModeloptModel  # noqa: PLC0415
    from models.loaders.qwen3_8 import Qwen38ModeloptModel  # noqa: PLC0415

    for loader, expected in ((ModeloptModel, True), (Qwen38ModeloptModel, False)):
        model = loader.__new__(loader)
        model.make_dense_linear_module = lambda _: SimpleNamespace()
        model.make_tensor_module = lambda _: SimpleNamespace()
        gate = model.make_gate_module("layer.mlp.gate", dense)
        assert gate.exclude_from_quantization is expected


def test_multimodal_qwen38_fails_before_emitting_incomplete_rotary_contract():
    model = qwen.Qwen4ExpModel.__new__(qwen.Qwen4ExpModel)
    model.text_only = False
    with pytest.raises(ValueError, match="historical T/H/W rotary positions"):
        model.make_model("unused")


def test_qwen38_qkv_fusion_honors_projection_packability(monkeypatch):
    model = qwen.Qwen4ExpTextModel.__new__(qwen.Qwen4ExpTextModel)
    model.attention_attrs = {"use_packed_matmul": True}
    calls = []
    model.is_qkv_projection_packable = lambda *args: False
    parent = load_builder_module("qwen3_5").Qwen35MoETextModel
    monkeypatch.setattr(parent, "make_attention_input_proj", lambda self, *args, **kwargs: calls.append((args, kwargs)))
    attention = object()
    model.make_attention_input_proj(2, attention, "hidden", indexer_proj=object())
    assert calls == [((2, attention, "hidden"), {})]


def test_explicit_embedding_precision_is_not_silently_ignored():
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
    )
    model = base.Model(
        config,
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT16,
        "cpu",
        "",
        {"embedding_quant_type": "int4", "op_types_to_quantize": ("MatMul", "Gather")},
    )
    with pytest.raises(ValueError, match="int4 model precision"):
        model.make_embedding_lookup(torch.ones(32, 8), "/model/embed_tokens", None)


def test_int8_embedding_request_does_not_reuse_int4_lm_head():
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
        tie_word_embeddings=True,
    )
    model = base.Model(
        config,
        ir.DataType.FLOAT16,
        ir.DataType.INT4,
        "cpu",
        "",
        {
            "embedding_quant_type": "int8",
            "shared_embeddings": True,
            "op_types_to_quantize": ("MatMul", "Gather"),
        },
    )
    model.make_embedding_lookup(torch.ones(32, 32), "/model/embed_tokens", None)
    gather = next(node for node in model.model.graph if node.op_type == "GatherBlockQuantized")
    assert gather.attributes["bits"].as_int() == 8
    assert gather.inputs[0].name == "model.embed_tokens.weight_Q8"


def test_dense_engram_clears_padding_before_convolution_and_residual():
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
    )
    model = qwen.Qwen4ExpTextModel.__new__(qwen.Qwen4ExpTextModel)
    base.Model.__init__(model, config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", "", {})
    model.hc_count, model.hc_hidden_size, model.ple_embed_dim = 2, 16, 8
    model.ngram_size, model.heads_per_ngram = 2, 1
    model.ple_conv_kernel_size, model.ple_conv_dilation = 2, 2
    model.external_engram = True
    model.input_names.update(
        {
            "engram_embeddings": "engram_embeddings",
            "past.ple_tokens": {0: "past_tokens"},
            "past.ple_conv": {0: "past_conv"},
        }
    )
    model.output_names.update({"present.ple_tokens": {0: "present_tokens"}, "present.ple_conv": {0: "present_conv"}})
    for name, shape, dtype in (
        ("input_ids", [1, 2], ir.DataType.INT64),
        ("attention_mask", [1, 2], ir.DataType.INT64),
        ("root", [1, 2, 16], ir.DataType.FLOAT),
        ("engram_embeddings", [1, 2, 8], ir.DataType.FLOAT),
        ("past_tokens", [1, 1], ir.DataType.INT64),
        ("past_conv", [1, 2, 16], ir.DataType.FLOAT),
    ):
        model.model.graph.inputs.append(model.make_value(name, dtype, shape))
    projection = torch.nn.Linear(8, 16, bias=False)
    projection.weight.data.fill_(0.125)
    value_projection = torch.nn.Linear(8, 8, bias=False)
    value_projection.weight.data.fill_(0.125)
    norm = SimpleNamespace(weight=torch.zeros(16))
    ple = SimpleNamespace(
        ple_embedding=SimpleNamespace(),
        key_proj=projection,
        value_proj=value_projection,
        norm_key=norm,
        norm_query=norm,
        norm_conv=norm,
        conv1d=SimpleNamespace(weight=torch.ones(16, 1, 2)),
    )
    output = model.make_ple(0, ple, "root")
    model.model.graph.outputs.extend(
        [
            model.values[output],
            model.make_value("present_conv", ir.DataType.FLOAT, [1, 2, 16]),
        ]
    )
    model.model.graph.sort()
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids": np.ones((1, 2), np.int64),
        "attention_mask": np.array([[0, 1]], np.int64),
        "root": np.ones((1, 2, 16), np.float32),
        "past_tokens": np.zeros((1, 1), np.int64),
        "past_conv": np.zeros((1, 2, 16), np.float32),
        "engram_embeddings": np.ones((1, 2, 8), np.float32),
    }
    first = session.run(None, feeds)
    feeds["engram_embeddings"][:, 0] = 100
    second = session.run(None, feeds)
    for actual, expected in zip(second, first, strict=True):
        np.testing.assert_array_equal(actual, expected)
