# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM2-MoE model builder.

The LFM2-MoE router selects experts by top-k of ``sigmoid(logits) + expert_bias`` but mixes them
with ``sigmoid_i / (sum_selected + 1e-6)``. The fused MoE/QMoE op only knows softmax-over-top-k
routing, so the builder performs the selection in the graph, feeds the op ``log(sigmoid)`` at the
selected experts and a sentinel elsewhere, and scales the op output by
``sum_selected / (sum_selected + 1e-6)``. These tests pin that graph and execute it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parent))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
lfm2_module = _load_builder_module("lfm2")
ir = base_module.ir
Model = base_module.Model
LFM2Model = lfm2_module.LFM2Model
LFM2MoEModel = lfm2_module.LFM2MoEModel


def _moe_attrs(num_experts, top_k, use_expert_bias, routed_scaling_factor):
    return {
        "op_type": "MoE",
        "num_experts": num_experts,
        "top_k": top_k,
        "activation_alpha": 1.0,
        "activation_beta": 0.0,
        "activation_type": "swiglu",
        "normalize_routing_weights": True,
        "swiglu_fusion": 1,
        "swiglu_limit": None,
        "use_sparse_mixer": False,
        "router_sentinel": -10000.0,
        "num_dense_layers": 0,
        "use_expert_bias": use_expert_bias,
        "routed_scaling_factor": routed_scaling_factor,
        "zero_point_names": {},
        "global_scale_names": {},
    }


def _recording_model(io_dtype, *, num_experts=8, top_k=2, use_expert_bias=True, routed_scaling_factor=1.0):
    """LFM2MoEModel whose graph emitters record instead of building an onnx_ir graph."""
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    model.io_dtype = io_dtype
    model.ep = "cpu"
    model.hidden_size = 64
    model.moe_intermediate_size = 32
    model.use_paged_attention = False
    model.moe_attrs = _moe_attrs(num_experts, top_k, use_expert_bias, routed_scaling_factor)
    model.layernorm_attrs = {}
    model.quant_attrs = {"nodes_to_exclude": []}
    model.nodes = []
    model.initializers = {}
    model.make_node = lambda op_type, inputs, outputs, name, domain="", **attrs: model.nodes.append(
        types.SimpleNamespace(op_type=op_type, inputs=list(inputs), outputs=list(outputs), name=name, attrs=attrs)
    )
    model.make_value = lambda name, dtype=None, shape=None: None
    model.make_initializer = lambda tensor, name, to=None, raw=False: model.initializers.__setitem__(name, (tensor, to))
    model.make_matmul = lambda matmul, basename, root_input, **kwargs: (
        model.make_node("MatMul", [root_input, f"{basename}.weight"], [f"{basename}/output_0"], name=basename),
        basename,
    )[1]
    return model


def _moe_module(num_experts=8, hidden=64, inter=32):
    gate = types.SimpleNamespace(weight=torch.randn(num_experts, hidden))
    experts = types.SimpleNamespace(
        gate_up_proj=torch.randn(num_experts, 2 * inter, hidden),
        down_proj=torch.randn(num_experts, hidden, inter),
    )
    return types.SimpleNamespace(gate=gate, experts=experts, expert_bias=torch.randn(num_experts))


def test_lfm2_moe_routes_dense_and_moe_layers(monkeypatch):
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    model.moe_attrs = {"num_dense_layers": 2}
    calls = []
    monkeypatch.setattr(model, "make_mlp", lambda layer_id, mlp, root_input: calls.append(("mlp", layer_id, mlp)))
    monkeypatch.setattr(model, "make_moe", lambda layer_id, moe, root_input: calls.append(("moe", layer_id, moe)))

    dense_ffn = types.SimpleNamespace(w1="w1", w2="w2", w3="w3")
    moe_ffn = object()
    model.make_feed_forward(1, types.SimpleNamespace(feed_forward=dense_ffn), "x")
    model.make_feed_forward(2, types.SimpleNamespace(feed_forward=moe_ffn), "x")

    assert calls == [("mlp", 1, dense_ffn), ("moe", 2, moe_ffn)]
    # The dense path aliases the HF w1/w3/w2 names onto the base-class gate/up/down names.
    assert (dense_ffn.gate_proj, dense_ffn.up_proj, dense_ffn.down_proj) == ("w1", "w3", "w2")


def test_lfm2_feed_forward_accepts_both_loader_layouts(monkeypatch):
    """Hugging Face layers expose `feed_forward`; the quantized-checkpoint IR exposes `mlp`."""
    model = LFM2Model.__new__(LFM2Model)
    seen = []
    monkeypatch.setattr(model, "make_mlp", lambda layer_id, mlp, root_input: seen.append(mlp))

    hf_ffn = types.SimpleNamespace(w1="w1", w2="w2", w3="w3")
    model.make_feed_forward(0, types.SimpleNamespace(feed_forward=hf_ffn), "x")
    assert (hf_ffn.gate_proj, hf_ffn.up_proj, hf_ffn.down_proj) == ("w1", "w3", "w2")

    # The quantized IR already carries gate/up/down names and has no w1/w3/w2 to alias.
    quantized_ffn = types.SimpleNamespace(gate_proj="g", up_proj="u", down_proj="d")
    model.make_feed_forward(1, types.SimpleNamespace(mlp=quantized_ffn), "x")

    assert seen == [hf_ffn, quantized_ffn]


def test_lfm2_moe_feed_forward_accepts_quantized_layer_layout(monkeypatch):
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    model.moe_attrs = {"num_dense_layers": 0}
    moe = object()
    seen = []
    monkeypatch.setattr(model, "make_moe", lambda layer_id, module, root_input: seen.append(module))
    model.make_feed_forward(3, types.SimpleNamespace(mlp=moe), "x")
    assert seen == [moe]


def test_lfm2_dense_model_still_owns_the_mlp(monkeypatch):
    model = LFM2Model.__new__(LFM2Model)
    calls = []
    monkeypatch.setattr(model, "make_mlp", lambda layer_id, mlp, root_input: calls.append((layer_id, mlp)))
    ffn = types.SimpleNamespace(w1="w1", w2="w2", w3="w3")
    model.make_feed_forward(0, types.SimpleNamespace(feed_forward=ffn), "x")
    assert calls == [(0, ffn)]


@pytest.mark.parametrize("io_dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
def test_lfm2_moe_router_selects_with_bias_and_mixes_without(io_dtype, routed_scaling_factor):
    model = _recording_model(io_dtype, routed_scaling_factor=routed_scaling_factor)
    moe = _moe_module()
    router_probs, output_scale = model.make_moe_router(3, moe, "hidden")

    ops = [n.op_type for n in model.nodes]
    casts = ["Cast"] if io_dtype != ir.DataType.FLOAT else []
    scaling = ["Mul"] if routed_scaling_factor != 1.0 else []
    assert ops == [
        "MatMul",
        "Reshape",
        *casts,
        # sigmoid spelled out: ORT's Sigmoid kernel loses all relative precision below ~1e-6
        "Neg",
        "Exp",
        "Add",
        "Reciprocal",
        "Add",
        "TopK",
        "GatherElements",
        # output scale: sum_selected / (sum_selected + 1e-6) [* routed_scaling_factor], reshaped to 3D
        "ReduceSum",
        "Add",
        "Div",
        *scaling,
        "Shape",
        "Slice",
        "Concat",
        "Reshape",
        *casts,
        # masked router scores
        "Clip",
        "Log",
        "Shape",
        "ConstantOfShape",
        "ScatterElements",
        *casts,
    ]
    r = "/model/layers.3/moe/router"
    by_name = {n.name[len(r) + 1 :]: n for n in model.nodes}

    # Scores: 1 / (1 + exp(-logits)) in fp32.
    logits = f"{r}/Cast/output_0" if casts else f"{r}/Reshape/output_0"
    assert by_name["sigmoid/Neg"].inputs == [logits]
    assert by_name["sigmoid/Exp"].inputs == [f"{r}/sigmoid/Neg/output_0"]
    assert by_name["sigmoid/Add"].inputs == [f"{r}/sigmoid/Exp/output_0", "/model/constants/FLOAT/1.0"]
    assert by_name["sigmoid/Reciprocal"].inputs == [f"{r}/sigmoid/Add/output_0"]
    scores = f"{r}/sigmoid/Reciprocal/output_0"

    # Selection: TopK over sigmoid + expert_bias (fp32).
    assert by_name["Add"].inputs == [scores, "model.layers.3.moe.expert_bias"]
    assert model.initializers["model.layers.3.moe.expert_bias"][1] == ir.DataType.FLOAT
    assert by_name["TopK"].inputs == [f"{r}/Add/output_0", "/model/constants/INT64/[2]"]
    assert by_name["TopK"].attrs == {"axis": -1, "largest": True}
    indices = f"{r}/TopK/output_1"

    # The unbiased selected scores feed both the mass factor and the mask.
    assert by_name["GatherElements"].inputs == [scores, indices]
    assert by_name["GatherElements"].attrs == {"axis": 1}
    selected = f"{r}/GatherElements/output_0"

    # Mass factor: the sum is taken before the clamp so flushed-to-zero scores give a zero factor.
    assert by_name["scale/ReduceSum"].inputs == [selected, "/model/constants/INT64/[-1]"]
    assert by_name["scale/ReduceSum"].attrs == {"keepdims": True}
    assert by_name["scale/Add"].inputs == [f"{r}/scale/ReduceSum/output_0", "/model/constants/FLOAT/1e-06"]
    assert by_name["scale/Div"].inputs == [f"{r}/scale/ReduceSum/output_0", f"{r}/scale/Add/output_0"]
    scale = f"{r}/scale/Div/output_0"
    if scaling:
        assert by_name["scale/Mul"].inputs == [scale, f"/model/constants/FLOAT/{routed_scaling_factor}"]
        scale = f"{r}/scale/Mul/output_0"
    assert by_name["scale/Shape"].inputs == ["hidden"]
    assert by_name["scale/Slice"].inputs == [
        f"{r}/scale/Shape/output_0",
        "/model/constants/INT64/[0]",
        "/model/constants/INT64/[-1]",
    ]
    assert by_name["scale/Concat"].inputs == [f"{r}/scale/Slice/output_0", "/model/constants/INT64/[1]"]
    assert by_name["scale/Reshape"].inputs == [scale, f"{r}/scale/Concat/output_0"]
    expected_scale = f"{r}/scale/Cast/output_0" if casts else f"{r}/scale/Reshape/output_0"
    assert output_scale == expected_scale

    # Mask: clamp, log, scatter over a sentinel row. The clamp keeps a flushed-to-zero sigmoid from
    # producing -inf below the sentinel, which would let the op's own top-k pick another expert.
    assert by_name["Clip"].inputs == [selected, "/model/constants/FLOAT/1e-30", ""]
    assert by_name["Log"].inputs == [f"{r}/Clip/output_0"]
    sentinel = by_name["ConstantOfShape"].attrs["value"]
    assert sentinel.dtype == ir.DataType.FLOAT
    router_sentinel = model.moe_attrs["router_sentinel"]
    assert sentinel.numpy().tolist() == [router_sentinel]
    assert router_sentinel < -1000 and torch.finfo(torch.float16).min < router_sentinel
    assert torch.log(torch.tensor(1e-30)).item() > router_sentinel
    assert by_name["ScatterElements"].inputs == [f"{r}/ConstantOfShape/output_0", indices, f"{r}/Log/output_0"]
    assert by_name["ScatterElements"].attrs == {"axis": 1}

    expected_probs = f"{r}/Cast_1/output_0" if casts else f"{r}/ScatterElements/output_0"
    assert router_probs == expected_probs
    if casts:
        assert by_name["Cast"].attrs == {"to": ir.DataType.FLOAT}
        assert by_name["scale/Cast"].attrs == {"to": io_dtype}
        assert by_name["Cast_1"].attrs == {"to": io_dtype}


def test_lfm2_moe_threads_router_outputs_to_subgraph(monkeypatch):
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    calls = []
    monkeypatch.setattr(model, "make_moe_preprocessing", lambda *args: calls.append(("pre", args)))
    monkeypatch.setattr(model, "make_moe_router", lambda *args: (calls.append(("router", args)), ("probs", "scale"))[1])
    monkeypatch.setattr(model, "make_moe_subgraph", lambda *args: calls.append(("subgraph", args)))
    moe = object()
    model.make_moe(5, moe, "hidden")
    assert calls == [
        ("pre", (5, moe, "hidden")),
        ("router", (5, moe, "hidden")),
        ("subgraph", (5, moe, "hidden", "probs", "scale")),
    ]


def test_lfm2_moe_router_without_expert_bias_selects_on_sigmoid():
    model = _recording_model(ir.DataType.FLOAT, use_expert_bias=False)
    model.make_moe_router(3, _moe_module(), "hidden")

    ops = [n.op_type for n in model.nodes]
    assert ops.count("Add") == 2  # the sigmoid's 1 + exp(-x) and the mass factor's epsilon, no bias add
    topk = next(n for n in model.nodes if n.op_type == "TopK")
    assert topk.inputs[0] == "/model/layers.3/moe/router/sigmoid/Reciprocal/output_0"
    assert "model.layers.3.moe.expert_bias" not in model.initializers


def test_lfm2_moe_preprocessing_interleaves_gate_and_up(monkeypatch):
    model = _recording_model(ir.DataType.FLOAT16, num_experts=2)
    moe = _moe_module(num_experts=2, hidden=4, inter=3)
    captured = {}
    monkeypatch.setattr(
        model,
        "make_moe_expert_initializers",
        lambda layer_id, experts, gate_up_weight=None, down_weight=None: captured.update(
            layer_id=layer_id, experts=experts, gate_up=gate_up_weight, down=down_weight
        ),
    )
    model.make_moe_preprocessing(2, moe, "hidden")

    raw = moe.experts.gate_up_proj
    gate, up = raw[:, :3, :], raw[:, 3:, :]
    assert captured["experts"] is moe.experts
    assert captured["gate_up"].shape == raw.shape
    torch.testing.assert_close(captured["gate_up"][:, 0::2, :], gate)
    torch.testing.assert_close(captured["gate_up"][:, 1::2, :], up)
    assert captured["down"] is moe.experts.down_proj

    gate_up_bias, _ = model.initializers["model.layers.2.moe.experts.gate_up_proj.bias"]
    down_bias, _ = model.initializers["model.layers.2.moe.experts.down_proj.bias"]
    assert gate_up_bias.shape == (2, 2 * model.moe_intermediate_size) and not gate_up_bias.any()
    assert down_bias.shape == (2, model.hidden_size) and not down_bias.any()

    # The router MatMul is excluded from int4 quantization so rounding cannot flip an expert choice.
    assert moe.gate.exclude_from_quantization is True


@pytest.mark.parametrize("op_type", ["MoE", "QMoE"])
def test_make_moe_expert_names_follows_op_type(op_type):
    model = Model.__new__(Model)
    model.moe_attrs = {"op_type": op_type}
    weight = "qweight" if op_type == "QMoE" else "weight"
    assert model.make_moe_expert_names(7) == {
        "gate_up_weight": f"model.layers.7.moe.experts.gate_up_proj.{weight}",
        "gate_up_scales": "model.layers.7.moe.experts.gate_up_proj.scales",
        "gate_up_bias": "model.layers.7.moe.experts.gate_up_proj.bias",
        "down_weight": f"model.layers.7.moe.experts.down_proj.{weight}",
        "down_scales": "model.layers.7.moe.experts.down_proj.scales",
        "down_bias": "model.layers.7.moe.experts.down_proj.bias",
    }


def test_make_moe_router_shape_is_per_token():
    model = Model.__new__(Model)
    model.moe_attrs = {"num_experts": 8}
    assert model.make_moe_router_shape() == ["batch_size * sequence_length", 8]
    assert model.make_moe_router_shape(last_dim=2) == ["batch_size * sequence_length", 2]


@pytest.mark.parametrize("op_type", ["MoE", "QMoE"])
def test_lfm2_moe_subgraph_feeds_masked_router_probs_and_scales_the_output(monkeypatch, op_type):
    model = _recording_model(ir.DataType.FLOAT16)
    model.moe_attrs["op_type"] = op_type
    captured = {}
    monkeypatch.setattr(model, "make_moe_op", lambda name, **kwargs: captured.update(name=name, **kwargs))

    model.make_moe_subgraph(4, _moe_module(), "hidden", "masked_probs", "output_scale")

    weight = "qweight" if op_type == "QMoE" else "weight"
    scales = ".scales" if op_type == "QMoE" else ""
    assert captured["name"] == f"/model/layers.4/moe/{op_type}"
    assert captured["root_input"] == "hidden"
    assert captured["router_probs"] == "masked_probs"
    assert captured["weight1"] == f"model.layers.4.moe.experts.gate_up_proj.{weight}"
    assert captured["weight2"] == f"model.layers.4.moe.experts.down_proj.{weight}"
    assert captured["scales1"] == (f"model.layers.4.moe.experts.gate_up_proj{scales}" if scales else "")
    assert captured["scales2"] == (f"model.layers.4.moe.experts.down_proj{scales}" if scales else "")
    assert captured["bias1"] == "model.layers.4.moe.experts.gate_up_proj.bias"
    assert captured["bias2"] == "model.layers.4.moe.experts.down_proj.bias"

    (mul,) = model.nodes
    assert mul.op_type == "Mul"
    assert mul.inputs == [f"/model/layers.4/moe/{op_type}/output_0", "output_scale"]
    assert model.layernorm_attrs["skip_input"] == "/model/layers.4/moe/Mul/output_0"


def test_lfm2_moe_subgraph_requires_router_outputs(monkeypatch):
    model = _recording_model(ir.DataType.FLOAT16)
    monkeypatch.setattr(model, "make_moe_op", lambda name, **kwargs: None)
    with pytest.raises(ValueError, match="make_moe_router"):
        model.make_moe_subgraph(4, _moe_module(), "hidden")
    with pytest.raises(ValueError, match="make_moe_router"):
        model.make_moe_subgraph(4, _moe_module(), "hidden", "masked_probs")


def _executable_model(num_experts, top_k, hidden, inter, routed_scaling_factor=1.0):
    """LFM2MoEModel with the real emitters, writing into an onnx_ir graph that ORT can execute."""
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    model.io_dtype = ir.DataType.FLOAT
    model.onnx_dtype = ir.DataType.FLOAT
    model.ep = "cpu"
    model.hidden_size = hidden
    model.moe_intermediate_size = inter
    model.use_paged_attention = False
    model.moe_attrs = _moe_attrs(num_experts, top_k, True, routed_scaling_factor)
    model.layernorm_attrs = {}
    model.quant_attrs = {"nodes_to_exclude": []}
    model.values = {}
    model.node_names = set()
    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21, "com.microsoft": 1}, name="lfm2_moe")
    model.model = ir.Model(graph, ir_version=10)
    graph.inputs.append(model.make_value("hidden", ir.DataType.FLOAT, ["batch_size", "sequence_length", hidden]))
    return model, graph


def _hf_reference(hidden_states, moe, top_k, routed_scaling_factor):
    """Mirror of transformers' Lfm2MoeSparseMoeBlock (route_tokens_to_experts + experts)."""
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = x @ moe.gate.weight.T
    scores = torch.sigmoid(logits)
    _, selected = torch.topk(scores + moe.expert_bias, k=top_k, dim=-1)
    weights = torch.gather(scores, 1, selected)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
    out = torch.zeros_like(x)
    inter = moe.experts.down_proj.shape[-1]
    for token in range(x.shape[0]):
        for slot in range(top_k):
            expert = selected[token, slot]
            gate_up = moe.experts.gate_up_proj[expert] @ x[token]
            act = torch.nn.functional.silu(gate_up[:inter]) * gate_up[inter:]
            out[token] += weights[token, slot] * (moe.experts.down_proj[expert] @ act)
    return (out * routed_scaling_factor).reshape(hidden_states.shape)


@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
def test_lfm2_moe_executed_graph_matches_hf_routing(tmp_path, routed_scaling_factor):
    """Run the emitted MoE layer in ORT and compare with HF's routing math, including the cases where
    the fused op's forced sum-to-one normalization differs from HF's `sum + 1e-6`."""
    num_experts, top_k, hidden, inter = 8, 4, 32, 64
    generator = torch.Generator().manual_seed(7)
    model, graph = _executable_model(num_experts, top_k, hidden, inter, routed_scaling_factor)

    # Token t reads router logits from column t of the gate weight (one-hot hidden states below).
    gate_weight = torch.zeros(num_experts, hidden)
    gate_weight[:, 0] = torch.tensor([1.5, -0.5, 0.3, 2.0, -1.0, 0.8, -2.0, 0.1])  # ordinary scores
    gate_weight[:, 1] = torch.tensor([-16.0, -16.0, -16.0, -16.0, -10.0, -11.0, -12.0, -13.0])  # tiny selected
    gate_weight[:, 2] = torch.tensor([-110.0, -111.0, -112.0, -113.0, -10.0, -11.0, -12.0, -13.0])  # flushed
    expert_bias = torch.tensor([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    moe = types.SimpleNamespace(
        gate=types.SimpleNamespace(weight=gate_weight),
        experts=types.SimpleNamespace(
            gate_up_proj=torch.randn(num_experts, 2 * inter, hidden, generator=generator) / hidden**0.5,
            down_proj=torch.randn(num_experts, hidden, inter, generator=generator) / inter**0.5,
        ),
        expert_bias=expert_bias,
    )
    model.make_moe(0, moe, "hidden")
    graph.outputs.append(model.values[model.layernorm_attrs["skip_input"]])
    model_path = tmp_path / "lfm2_moe.onnx"
    ir.save(model.model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    hidden_states = torch.zeros(1, 3, hidden)
    for token in range(3):
        hidden_states[0, token, token] = 1.0
    (actual,) = session.run(None, {"hidden": hidden_states.numpy()})
    expected = _hf_reference(hidden_states, moe, top_k, routed_scaling_factor).numpy()

    # Token 1: the selected sigmoids sum to ~4.5e-7, so HF's weights sum to ~0.31 and not 1.
    # Token 2: the selected sigmoids flush to 0 in fp32, so HF returns exactly zero.
    assert 0.2 < np.linalg.norm(expected[0, 1]) / np.linalg.norm(actual[0, 1] / 0.31) < 1.6
    assert np.all(expected[0, 2] == 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    assert np.all(actual[0, 2] == 0)


def _stub_base_init(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
    self.moe_attrs = {}
    self.ep = ep


def test_lfm2_moe_rejects_unnormalized_topk(monkeypatch):
    monkeypatch.setattr(LFM2Model, "__init__", _stub_base_init)
    config = types.SimpleNamespace(norm_topk_prob=False)
    with pytest.raises(NotImplementedError, match="norm_topk_prob"):
        LFM2MoEModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})


def test_lfm2_dense_intermediate_size_honors_block_auto_adjust():
    # Dense LFM2 keeps transformers' FFN width adjustment and does not default the flag away.
    model = LFM2Model.__new__(LFM2Model)
    config = types.SimpleNamespace(
        intermediate_size=12288,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
    )
    model.make_intermediate_size_init(config)
    assert model.intermediate_size == 8192

    with pytest.raises(AttributeError, match="block_auto_adjust_ff_dim"):
        model.make_intermediate_size_init(types.SimpleNamespace(intermediate_size=12288))


def test_lfm2_moe_intermediate_size_uses_config_value():
    # Lfm2MoeConfig has no block_auto_adjust_ff_dim; the dense layers use intermediate_size as is.
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    model.make_intermediate_size_init(types.SimpleNamespace(intermediate_size=7168))
    assert model.intermediate_size == 7168


def test_lfm2_moe_init_configures_fused_swiglu(monkeypatch):
    monkeypatch.setattr(LFM2Model, "__init__", _stub_base_init)
    config = types.SimpleNamespace(
        norm_topk_prob=True,
        num_dense_layers=2,
        moe_intermediate_size=1792,
        use_expert_bias=True,
        routed_scaling_factor=1.0,
    )
    model = LFM2MoEModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})
    assert model.moe_attrs["activation_type"] == "swiglu"
    assert model.moe_attrs["swiglu_fusion"] == 1
    assert model.moe_attrs["normalize_routing_weights"] is True
    assert model.moe_attrs["router_sentinel"] == -10000.0
    assert model.moe_attrs["num_dense_layers"] == 2
    assert model.moe_attrs["use_expert_bias"] is True
    assert model.moe_attrs["routed_scaling_factor"] == 1.0
    assert model.moe_intermediate_size == 1792


@pytest.mark.parametrize(
    "ep,swiglu_limit,expected",
    [
        ("cpu", None, None),
        ("trt-rtx", None, float("inf")),
        ("trt-rtx", 7.0, 7.0),
    ],
)
def test_make_moe_init_fills_swiglu_limit_for_trt_rtx(ep, swiglu_limit, expected):
    # TRT-RTX requires swiglu_limit on QMoE for every MoE model, so the base init supplies +inf.
    model = Model.__new__(Model)
    model.ep = ep
    model.moe_attrs = {"swiglu_limit": swiglu_limit}
    model.quant_config = types.SimpleNamespace(moe=types.SimpleNamespace(type="int4", weights_prepacked=-1))
    model.make_moe_init()
    assert model.moe_attrs["swiglu_limit"] == expected


@pytest.mark.parametrize(
    "moe_type,expected",
    [
        ("int4", ("QMoE", "int", 4)),
        ("int8", ("QMoE", "int", 8)),
        ("mxfp4", ("QMoE", "fp4", 4)),
        ("nvfp4", ("QMoE", "nvfp4", 4)),
        ("none", ("MoE", "int", 0)),
    ],
)
def test_make_moe_init_selects_qmoe_for_quantized_experts(moe_type, expected):
    # Regression: these keys are what make_moe_op / make_moe_expert_initializers read.
    model = Model.__new__(Model)
    model.ep = "cpu"
    model.moe_attrs = {"swiglu_limit": None}
    model.quant_config = types.SimpleNamespace(moe=types.SimpleNamespace(type=moe_type, weights_prepacked=-1))
    model.make_moe_init()
    assert (
        model.moe_attrs["op_type"],
        model.moe_attrs["quant_type"],
        model.moe_attrs["expert_weight_bits"],
    ) == expected
