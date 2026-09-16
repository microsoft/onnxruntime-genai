# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM2-MoE model builder.

The LFM2-MoE router selects experts by top-k of ``sigmoid(logits) + expert_bias`` but mixes them
with the unbiased ``sigmoid(logits)`` (renormalized). The fused MoE/QMoE op only knows
softmax-over-top-k routing, so the builder performs the selection in the graph and feeds the op
``log(sigmoid)`` at the selected experts and a sentinel elsewhere. These tests pin that graph.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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


def _recording_model(io_dtype, *, num_experts=8, top_k=2, use_expert_bias=True, routed_scaling_factor=1.0):
    """LFM2MoEModel whose graph emitters record instead of building an onnx_ir graph."""
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    model.io_dtype = io_dtype
    model.ep = "cpu"
    model.hidden_size = 64
    model.moe_intermediate_size = 32
    model.use_expert_bias = use_expert_bias
    model.routed_scaling_factor = routed_scaling_factor
    model.moe_attrs = {
        "op_type": "MoE",
        "num_experts": num_experts,
        "top_k": top_k,
        "router_sentinel": -10000.0,
    }
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
    model.num_dense_layers = 2
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
    model.num_dense_layers = 0
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
def test_lfm2_moe_router_selects_with_bias_and_mixes_without(io_dtype):
    model = _recording_model(io_dtype)
    moe = _moe_module()
    router_probs = model.make_moe_router(3, moe, "hidden")

    ops = [n.op_type for n in model.nodes]
    casts = ["Cast"] if io_dtype != ir.DataType.FLOAT else []
    assert ops == [
        "MatMul",
        "Reshape",
        *casts,
        "Sigmoid",
        "Add",
        "TopK",
        "GatherElements",
        "Clip",
        "Log",
        "Shape",
        "ConstantOfShape",
        "ScatterElements",
        *casts,
    ]
    by_name = {n.name.rsplit("/", 1)[-1]: n for n in model.nodes}
    r = "/model/layers.3/moe/router"

    # Selection: TopK over sigmoid + expert_bias (fp32).
    assert by_name["Add"].inputs == [f"{r}/Sigmoid/output_0", "model.layers.3.moe.expert_bias"]
    assert model.initializers["model.layers.3.moe.expert_bias"][1] == ir.DataType.FLOAT
    assert by_name["TopK"].inputs == [f"{r}/Add/output_0", "/model/constants/INT64/[2]"]
    assert by_name["TopK"].attrs == {"axis": -1, "largest": True}
    indices = f"{r}/TopK/output_1"

    # Mixing: sigmoid (no bias) gathered at the selected experts, clamped, logged, scattered over a
    # sentinel row. Gather runs before Log so Log only touches top_k entries, and the clamp keeps a
    # flushed-to-zero sigmoid from producing -inf below the sentinel.
    assert by_name["GatherElements"].inputs == [f"{r}/Sigmoid/output_0", indices]
    assert by_name["GatherElements"].attrs == {"axis": 1}
    assert by_name["Clip"].inputs == [f"{r}/GatherElements/output_0", "/model/constants/FLOAT/1e-30", ""]
    assert by_name["Log"].inputs == [f"{r}/Clip/output_0"]
    sentinel = by_name["ConstantOfShape"].attrs["value"]
    assert sentinel.dtype == ir.DataType.FLOAT
    router_sentinel = model.moe_attrs["router_sentinel"]
    assert sentinel.numpy().tolist() == [router_sentinel]
    assert router_sentinel < -1000 and torch.finfo(torch.float16).min < router_sentinel
    assert by_name["ScatterElements"].inputs == [
        f"{r}/ConstantOfShape/output_0",
        indices,
        f"{r}/Log/output_0",
    ]
    assert by_name["ScatterElements"].attrs == {"axis": 1}

    expected_probs = f"{r}/Cast_1/output_0" if casts else f"{r}/ScatterElements/output_0"
    assert router_probs == expected_probs
    if casts:
        assert by_name["Cast"].attrs == {"to": ir.DataType.FLOAT}
        assert by_name["Cast_1"].attrs == {"to": io_dtype}
    # The clamp floor is well above the sentinel, so a clamped score can never lose the op's own top-k.
    assert torch.log(torch.tensor(1e-30)).item() > model.moe_attrs["router_sentinel"]


def test_lfm2_moe_threads_router_probs_from_router_to_subgraph(monkeypatch):
    model = LFM2MoEModel.__new__(LFM2MoEModel)
    calls = []
    monkeypatch.setattr(model, "make_moe_preprocessing", lambda *args: calls.append(("pre", args)))
    monkeypatch.setattr(model, "make_moe_router", lambda *args: (calls.append(("router", args)), "probs")[1])
    monkeypatch.setattr(model, "make_moe_subgraph", lambda *args: calls.append(("subgraph", args)))
    moe = object()
    model.make_moe(5, moe, "hidden")
    assert calls == [
        ("pre", (5, moe, "hidden")),
        ("router", (5, moe, "hidden")),
        ("subgraph", (5, moe, "hidden", "probs")),
    ]


def test_lfm2_moe_router_without_expert_bias_selects_on_sigmoid():
    model = _recording_model(ir.DataType.FLOAT, use_expert_bias=False)
    model.make_moe_router(3, _moe_module(), "hidden")

    ops = [n.op_type for n in model.nodes]
    assert "Add" not in ops
    topk = next(n for n in model.nodes if n.op_type == "TopK")
    assert topk.inputs[0] == "/model/layers.3/moe/router/Sigmoid/output_0"
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


@pytest.mark.parametrize("op_type", ["MoE", "QMoE"])
def test_lfm2_moe_subgraph_feeds_masked_router_probs(monkeypatch, op_type):
    model = _recording_model(ir.DataType.FLOAT16)
    model.moe_attrs["op_type"] = op_type
    captured = {}
    monkeypatch.setattr(model, "make_moe_op", lambda name, **kwargs: captured.update(name=name, **kwargs))

    model.make_moe_subgraph(4, _moe_module(), "hidden", "masked_probs")

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
    assert model.nodes == []
    assert model.layernorm_attrs["skip_input"] == f"/model/layers.4/moe/{op_type}/output_0"


def test_lfm2_moe_subgraph_requires_router_probs(monkeypatch):
    model = _recording_model(ir.DataType.FLOAT16)
    monkeypatch.setattr(model, "make_moe_op", lambda name, **kwargs: None)
    with pytest.raises(ValueError, match="make_moe_router"):
        model.make_moe_subgraph(4, _moe_module(), "hidden")


def test_lfm2_moe_subgraph_applies_routed_scaling_factor(monkeypatch):
    model = _recording_model(ir.DataType.FLOAT16, routed_scaling_factor=2.5)
    monkeypatch.setattr(model, "make_moe_op", lambda name, **kwargs: None)
    monkeypatch.setattr(model, "make_hidden_state_shape", lambda **kwargs: ["batch_size", "sequence_length", 64])

    model.make_moe_subgraph(4, _moe_module(), "hidden", "masked_probs")

    (mul,) = model.nodes
    assert mul.op_type == "Mul"
    assert mul.inputs == ["/model/layers.4/moe/MoE/output_0", "/model/constants/FLOAT16/2.5"]
    assert model.layernorm_attrs["skip_input"] == "/model/layers.4/moe/Mul/output_0"


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
    assert (model.num_dense_layers, model.moe_intermediate_size) == (2, 1792)


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
