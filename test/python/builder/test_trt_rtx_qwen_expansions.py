# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from models.builders.base import Model
from models.builders.qwen3_8_text import Qwen4ExpTextModel


def _model(ep, dtype, model_class=Model):
    model = model_class.__new__(model_class)
    model.ep = ep
    model.io_dtype = dtype
    model.values = {}
    model.node_names = set()
    model.use_paged_attention = False
    model.hidden_size = 24
    model.head_size = 12
    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21, "com.microsoft": 1})
    model.model = ir.Model(graph, ir_version=10)
    model.make_ep_expansions_init()
    return model


def _input(model, name, dtype, shape):
    model.model.graph.inputs.append(model.make_value(name, dtype, shape))


def _run(model, outputs, feeds, model_path):
    model.model.graph.outputs.extend(model.values[name] for name in outputs)
    # Match full exports, which externalize even small initializers. Shape constants must
    # remain readable by ONNX shape inference after this round trip.
    ir.save(model.model, model_path, external_data=f"{model_path.name}.data", size_threshold_bytes=0)
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as error:
        if model.ep != "trt-rtx" and "is not a registered function/op" in str(error):
            pytest.skip("Installed ONNX Runtime does not include the fused reference operator")
        raise
    return session.run(None, feeds)


@pytest.mark.parametrize("ep", ["dml", "trt-rtx"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("paged", [False, True])
def test_standard_hyper_connections_match_reference(ep, dtype, paged, tmp_path):
    model = _model(ep, dtype, Qwen4ExpTextModel)
    model.use_paged_attention = paged
    model.hidden_size = 8
    model.hyper_connection_attrs = {"num_branches": 4, "hidden_size": 32}
    model.layernorm_attrs = {"epsilon": 1e-6}
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float16
    token_shape = [6] if paged else [2, 3]
    rng = np.random.default_rng(42)
    feeds = {
        "streams": rng.normal(size=(*token_shape, 32)).astype(np_dtype),
        "pre": rng.normal(size=(*token_shape, 32)).astype(np_dtype),
        "post": rng.normal(size=(*token_shape, 4)).astype(np_dtype),
        "block": rng.normal(size=(*token_shape, 8)).astype(np_dtype),
        "down": rng.normal(size=(*token_shape, 4)).astype(np_dtype),
    }
    for name, value in feeds.items():
        _input(model, name, dtype, list(value.shape))
    weight = torch.from_numpy(rng.normal(size=32).astype(np.float32))
    norm = model.make_branchwise_rms_norm("/norm", "streams", SimpleNamespace(weight=weight), 8)
    silu = model.make_scaled_silu("/silu", "down", [*token_shape, 4])
    pre = model.make_hyper_connection_pre_mix("/pre_mix", "streams", "pre", token_shape)
    post = model.make_hyper_connection_post_mix(
        "/post_mix", "streams", "block", "post", token_shape, output_name="mixed"
    )
    reductions = [node.attributes["keepdims"].value for node in model.model.graph if node.op_type == "ReduceMean"]
    assert reductions == [1, 0]
    assert all(type(value) is int for value in reductions)
    results = _run(model, [norm, silu, pre, post], feeds, tmp_path / "hyper.onnx")

    grouped = feeds["streams"].astype(np.float32).reshape(*token_shape, 4, 8)
    inverse_rms = 1.0 / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)
    normalized = (grouped * inverse_rms).reshape(*token_shape, 32).astype(np_dtype)
    scaled = (feeds["down"] / 4).astype(np_dtype)
    expected = [
        normalized * (weight.numpy() + 1.0).astype(np_dtype),
        scaled / (1.0 + np.exp(-scaled.astype(np.float32))),
        np.mean((feeds["streams"] * feeds["pre"]).reshape(*token_shape, 4, 8), axis=-2),
        feeds["streams"] + (feeds["block"][..., None, :] * feeds["post"][..., None]).reshape(*token_shape, 32),
    ]
    for actual, reference in zip(results, expected, strict=False):
        np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-3)
    assert all(node.domain in ("", None) for node in model.model.graph)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    "op", ["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "SkipLayerNormalization"]
)
def test_trt_rtx_layernorm_matches_fused_reference(dtype, paged, op, tmp_path):
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float16
    shape = [6, 24] if paged else [2, 3, 24]
    rng = np.random.default_rng(44)
    feeds = {name: rng.normal(size=shape).astype(np_dtype) for name in ("root", "skip")}
    feeds.update({name: rng.normal(size=24).astype(np_dtype) for name in ("weight", "bias")})
    skip = op.startswith("Skip")
    inputs = ["root", "skip", "weight"] if skip else ["root", "weight"]
    if "Simplified" not in op:
        inputs.append("bias")
    outputs = ["normalized", "", "", "residual"] if skip else ["normalized"]
    results = []
    for ep in ("cuda", "trt-rtx"):
        model = _model(ep, dtype)
        model.use_paged_attention = paged
        model.layernorm_attrs = {"epsilon": 1e-6}
        for name in inputs:
            _input(model, name, dtype, list(feeds[name].shape))
        kwargs = {"epsilon": 1e-6}
        if not skip:
            kwargs.update(axis=-1, stash_type=1)
        model.make_layernorm_subgraph(
            "/norm", op_type=op, inputs=inputs, outputs=outputs, skip=skip, new_io_dtype=dtype, **kwargs
        )
        for name in ("normalized", "residual") if skip else ("normalized",):
            model.make_value(name, dtype, shape)
        results.append(
            _run(
                model,
                ["normalized", "residual"] if skip else ["normalized"],
                {name: feeds[name] for name in inputs},
                tmp_path / f"{ep}.onnx",
            )
        )
    for fused, expanded in zip(*results, strict=True):
        np.testing.assert_allclose(expanded, fused, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("op", ["GatedAdd", "LinearAttentionGate", "GatedRMSNorm"])
def test_trt_rtx_gates_match_fused_reference(dtype, op, tmp_path):
    rng = np.random.default_rng(41)
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float16
    shape = ["batch_size", "sequence_length", 24]
    feeds = {name: rng.normal(size=(2, 3, 24)).astype(np_dtype) for name in ("x", "gate", "shared")}
    feeds["bias"] = rng.normal(size=24).astype(np.float32)
    feeds["decay_scale"] = -np.exp(rng.normal(size=24)).astype(np.float32)
    feeds["scale"] = rng.normal(size=6).astype(np_dtype)
    if op == "GatedAdd":
        feeds["gate"] = rng.normal(size=(2, 3, 1)).astype(np_dtype)
    outputs = ["result/output_0", "result/output_1"] if op == "LinearAttentionGate" else ["result/output_0"]
    results = []
    for ep in ("cuda", "trt-rtx"):
        model = _model(ep, dtype)
        if op == "GatedAdd":
            names = ["x", "shared", "gate"]
            for name in names:
                _input(model, name, dtype, [*shape[:-1], 1] if name == "gate" else shape)
            model.make_gated_add("result", *names, shape)
        elif op == "LinearAttentionGate":
            names = ["x", "bias", "decay_scale", "gate"]
            # Include extreme values to exercise stable softplus in the unfused graph.
            feeds["x"][0, 0, :2] = [-80, 80]
            for name in names:
                _input(
                    model,
                    name,
                    ir.DataType.FLOAT if name in ("bias", "decay_scale") else dtype,
                    [24] if name in ("bias", "decay_scale") else shape,
                )
            model.make_linear_attention_gate("result", *names, shape)
        else:
            names = ["x", "scale", "gate"]
            for name in names:
                _input(model, name, dtype, [6] if name == "scale" else shape)
            model.make_gated_rms_norm("result", *names, shape, epsilon=1e-6)
        if ep == "trt-rtx":
            assert op not in {node.op_type for node in model.model.graph}
        else:
            assert op in {node.op_type for node in model.model.graph}
        results.append(_run(model, outputs, {name: feeds[name] for name in names}, tmp_path / f"{ep}.onnx"))
    for actual, reference in zip(results[1], results[0], strict=True):
        np.testing.assert_allclose(actual, reference, rtol=5e-3 if np_dtype == np.float16 else 2e-5, atol=2e-3)


@pytest.mark.parametrize("layout", [0, 1])
@pytest.mark.parametrize("interleaved", [0, 1])
@pytest.mark.parametrize(
    "rotary_dim, sections",
    [
        (8, [2, 1, 1]),
        (12, [4, 1, 1]),
        (8, [1, 1, 2]),
        (8, [1, 2, 1]),
        (8, [0, 2, 2]),
        (8, [2, 0, 2]),
        (8, [2, 2, 0]),
        (8, [0, 0, 4]),
    ],
)
@pytest.mark.parametrize("sequence_length", [1, 3])
def test_trt_rtx_mrope_matches_fused_reference(layout, interleaved, rotary_dim, sections, sequence_length, tmp_path):
    rng = np.random.default_rng(52)
    half = rotary_dim // 2
    angles = rng.normal(size=(16, half)).astype(np.float32)
    feeds = {
        "x": rng.normal(size=(2, sequence_length, 24)).astype(np.float32),
        # Distinct T/H/W positions ensure this cannot accidentally become text-only RoPE.
        "positions": rng.integers(0, 16, size=(3, 2, sequence_length), dtype=np.int64),
        "cos": np.cos(angles),
        "sin": np.sin(angles),
    }
    results = []
    for ep in ("cuda", "trt-rtx"):
        model = _model(ep, ir.DataType.FLOAT)
        model.rope_attrs = {
            "interleaved": interleaved,
            "rotary_embedding_dim": rotary_dim,
            "mrope_section": sections,
            "mrope_layout": layout,
        }
        _input(model, "x", ir.DataType.FLOAT, ["batch_size", "sequence_length", 24])
        _input(model, "positions", ir.DataType.INT64, [3, "batch_size", "sequence_length"])
        for name in ("cos", "sin"):
            _input(model, name, ir.DataType.FLOAT, [16, half])
        model.make_mrotary_embedding(
            "rotary",
            "x",
            "output",
            position_ids="positions",
            cos_cache_name="cos",
            sin_cache_name="sin",
            num_heads=2,
            dtype=ir.DataType.FLOAT,
        )
        ops = {node.op_type for node in model.model.graph}
        assert ("MRotaryEmbedding" in ops) == (ep == "cuda")
        results.append(_run(model, ["output"], feeds, tmp_path / f"{ep}.onnx"))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(0, 3, 24), (2, 0, 24)])
def test_trt_rtx_gated_rms_norm_empty_dimensions(shape, tmp_path):
    model = _model("trt-rtx", ir.DataType.FLOAT)
    symbolic_shape = ["batch_size", "sequence_length", 24]
    _input(model, "x", ir.DataType.FLOAT, symbolic_shape)
    _input(model, "gate", ir.DataType.FLOAT, symbolic_shape)
    _input(model, "scale", ir.DataType.FLOAT, [6])
    model.make_gated_rms_norm("result", "x", "scale", "gate", symbolic_shape)

    (result,) = _run(
        model,
        ["result/output_0"],
        {"x": np.empty(shape, np.float32), "gate": np.empty(shape, np.float32), "scale": np.ones(6, np.float32)},
        tmp_path / "empty.onnx",
    )

    assert result.shape == shape


@pytest.mark.parametrize(
    "rotary_dim, sections, layout",
    [
        (8, [-1, 2, 3], 0),
        (8, [-1, 2, 3], 1),
        (8, [2, -1, 3], 1),
        (8, [2, 3, -1], 1),
        (7, [1, 1, 1], 1),
        (14, [3, 2, 2], 1),
        (-2, [0, 0, -1], 1),
        (8, [1, 1], 1),
        (8, [1, 1, 1], 1),
        (8, [2, 1, 1], 2),
    ],
)
def test_trt_rtx_mrope_rejects_invalid_attributes(rotary_dim, sections, layout):
    model = _model("trt-rtx", ir.DataType.FLOAT)
    model.rope_attrs = {"rotary_embedding_dim": rotary_dim, "mrope_section": sections, "mrope_layout": layout}

    with pytest.raises(ValueError, match="TRT-RTX MRoPE"):
        model.get_mrope_owners(rotary_dim)
