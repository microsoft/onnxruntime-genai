# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from models.builders.base import Model


def _model(ep, dtype):
    model = Model.__new__(Model)
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
