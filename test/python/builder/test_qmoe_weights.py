# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for CUDA QMoE expert-weight quantization in the model builder.

These guard the INT4 QMoE encoding fix for Qwen3.5/3.6 MoE models. The CUDA path
must:
  1. dispatch to the CUTLASS-prepack quantizer for ``weights_prepacked`` -1/1
     (both mean the op reads prepacked weights) and to the raw MatMulNBits
     quantizer for ``weights_prepacked`` 0,
  2. only do so on the CUDA EP,
  3. reject unsupported block sizes with a real exception (not a ``python -O``
     strippable ``assert``), and
  4. restore per-channel quantization when ``qmoe_block_size <= 0``, and
  5. keep the SIGNED blockwise scales (taking ``abs()`` reintroduces the
     garbage-output bug).

The CPU and WebGPU paths share the raw MatMulNBits encoding and must keep the
``[N, K/pack]`` storage contract the QMoE op validates, for INT4 and INT8.
WebGPU additionally requires whole blocks; INT4 requires even K.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch
from _builder_test_utils import BUILDERS_DIR, load_builder_module

base_module = load_builder_module("base")
Model = base_module.Model
cuda_quantizer_module = sys.modules["quantization.cuda_quantizer"]
qmoe_symmetric_per_channel_quantize = cuda_quantizer_module.CudaQuantizer.qmoe_symmetric_per_channel_quantize
quant_config_module = sys.modules["quantization.quant_config"]
MoEConfig = quant_config_module.MoEConfig
gptoss_module = load_builder_module("gptoss")
GPTOSSModel = gptoss_module.GPTOSSModel
phi_module = load_builder_module("phi")
Phi3MoELongRoPEModel = phi_module.Phi3MoELongRoPEModel


def test_base_moe_orchestrates_model_hooks(monkeypatch):
    model = Model.__new__(Model)
    calls = []
    moe = object()

    monkeypatch.setattr(model, "make_moe_preprocessing", lambda *args: calls.append(("preprocess", args)))
    monkeypatch.setattr(model, "make_moe_router", lambda *args: calls.append(("router", args)), raising=False)
    monkeypatch.setattr(model, "make_moe_subgraph", lambda *args: calls.append(("subgraph", args)), raising=False)

    model.make_moe(3, moe, "hidden_states")

    expected_args = (3, moe, "hidden_states")
    assert calls == [
        ("preprocess", expected_args),
        ("router", expected_args),
        ("subgraph", expected_args),
    ]


def test_make_initializer_preserves_raw_fp8_bytes():
    if not hasattr(base_module.ir.DataType, "FLOAT8E8M0"):
        pytest.skip("Installed onnx_ir does not support FLOAT8E8M0")

    class InitializerGraph:
        def register_initializer(self, value):
            self.value = value

    model = Model.__new__(Model)
    model.values = {}
    model.model = types.SimpleNamespace(graph=InitializerGraph())
    encoded = torch.tensor([0x7F, 0x80], dtype=torch.uint8)

    for dtype in (base_module.ir.DataType.FLOAT8E8M0, base_module.ir.DataType.FLOAT8E4M3FN):
        model.make_initializer(encoded, dtype.name, to=dtype, raw=True)
        proto = base_module.ir.serde.serialize_tensor(model.model.graph.value.const_value)
        assert proto.data_type == dtype.value
        assert proto.raw_data == bytes([0x7F, 0x80])


def test_phi_moe_uses_base_layer_route():
    model = Phi3MoELongRoPEModel.__new__(Phi3MoELongRoPEModel)
    moe = object()
    layer = types.SimpleNamespace(block_sparse_moe=moe)

    assert model.get_moe_module(2, layer) is moe

    assert "make_layer" not in Phi3MoELongRoPEModel.__dict__


def test_phi_moe_quantizes_each_projection_with_its_effective_bits():
    model = Phi3MoELongRoPEModel.__new__(Phi3MoELongRoPEModel)
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.moe_attrs = {
        "num_experts": 1,
        "fc1_expert_weight_bits": 2,
        "fc2_expert_weight_bits": 4,
        "fc3_expert_weight_bits": 2,
    }
    calls = []
    model.make_qmoe_weights = lambda weight, bits: (
        calls.append(bits) or torch.zeros(1, dtype=torch.uint8),
        torch.zeros(1),
    )
    model.make_initializer = lambda *args, **kwargs: None
    expert = types.SimpleNamespace(
        w1=types.SimpleNamespace(weight=torch.zeros(4, 4)),
        w2=types.SimpleNamespace(weight=torch.zeros(4, 4)),
        w3=types.SimpleNamespace(weight=torch.zeros(4, 4)),
    )

    model.make_moe_preprocessing(0, types.SimpleNamespace(experts=[expert]), "root")

    assert calls == [2, 4, 2]


def test_gptoss_moe_uses_base_layer_route():
    model = GPTOSSModel.__new__(GPTOSSModel)
    moe = object()
    layer = types.SimpleNamespace(mlp=moe)

    assert model.get_moe_module(2, layer) is moe
    assert "make_layer" not in GPTOSSModel.__dict__


def test_gptoss_moe_dispatches_fused_and_decomposed_paths(monkeypatch):
    model = GPTOSSModel.__new__(GPTOSSModel)
    calls = []
    moe = object()

    monkeypatch.setattr(model, "make_moe_preprocessing", lambda *args: calls.append("preprocess"))
    monkeypatch.setattr(model, "make_moe_router", lambda *args: calls.append("router"))
    monkeypatch.setattr(model, "make_moe_subgraph", lambda *args: calls.append("subgraph"))
    monkeypatch.setattr(model, "make_moe_decomposed", lambda *args: calls.append("decomposed"))

    model.ep = "cuda"
    model.make_moe(2, moe, "hidden_states")
    assert calls == ["preprocess", "router", "subgraph"]

    calls.clear()
    model.ep = "dml"
    model.make_moe(2, moe, "hidden_states")
    assert calls == ["decomposed"]


def _load_builder_cli_module(monkeypatch):
    builders_module = types.ModuleType("builders")
    for class_name in (
        "ChatGLMModel",
        "ErnieModel",
        "Gemma2Model",
        "Gemma3Model",
        "GemmaModel",
        "GPTOSSModel",
        "GraniteMoEHybridModel",
        "GraniteModel",
        "HunyuanDenseV1Model",
        "InternLM2Model",
        "LFM2AudioModel",
        "LFM2Model",
        "LFM2MoEModel",
        "LlamaModel",
        "Mistral3TextModel",
        "MistralModel",
        "Model",
        "NemotronModel",
        "OLMoModel",
        "Phi3MiniLongRoPEModel",
        "Phi3MiniModel",
        "Phi3MoELongRoPEModel",
        "Phi3SmallLongRoPEModel",
        "Phi3SmallModel",
        "Phi3VModel",
        "Phi4MMModel",
        "PhiModel",
        "Qwen25VLTextModel",
        "Qwen35MoETextModel",
        "Qwen35TextModel",
        "Qwen3Model",
        "Qwen3VLTextModel",
        "QwenModel",
        "SmolLM3Model",
        "VideoChatFlashQwenModel",
        "WhisperModel",
    ):
        setattr(builders_module, class_name, type(class_name, (), {}))
    # Submodule imports (e.g. `from quantization import ...`) must resolve to the
    # real, dependency-free modules rather than the class stubs above.
    builders_module.__path__ = [str(BUILDERS_DIR)]
    monkeypatch.setitem(sys.modules, "builders", builders_module)

    module_name = "builder_cli_under_test"
    spec = importlib.util.spec_from_file_location(module_name, BUILDERS_DIR.parent / "builder.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _parse_extra_options(builder, extra_options, precision="int4", execution_provider="cuda", quantization_config=None):
    # Parser validation in these tests should not depend on remote/model IO.
    builder.get_hf_details = lambda *args, **kwargs: {
        "hf_config": types.SimpleNamespace(
            tie_word_embeddings=True,
            quantization_config=quantization_config or {},
        )
    }
    return builder.parse_extra_options(
        "dummy-model",
        "",
        "",
        precision,
        execution_provider,
        "",
        extra_options,
    )


def test_moe_quant_type_mxfp4_is_accepted(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    options = _parse_extra_options(builder, ["moe_quant_type=mxfp4"], "int4", "cuda")
    assert options["moe_quant_type"] == "mxfp4"


def test_moe_quant_type_rejects_invalid_value(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    with pytest.raises(ValueError, match="moe_quant_type must be one of"):
        _parse_extra_options(builder, ["moe_quant_type=fp8"], "int4", "cuda")


def test_use_8bits_moe_maps_to_moe_quant_type(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    options = _parse_extra_options(builder, ["use_8bits_moe=true"], "int4", "cuda")
    assert options["moe_quant_type"] == "int8"


def test_moe_quant_type_mxfp4_requires_qmoe_precision(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    with pytest.raises(ValueError, match="moe_quant_type=mxfp4 requires building with precision=int4"):
        _parse_extra_options(builder, ["moe_quant_type=mxfp4"], "fp16", "cuda")


def test_mixed_width_qmoe_options_are_accepted(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    options = _parse_extra_options(
        builder,
        ["moe_quant_type=int4", "qmoe_fc1_type=int2", "qmoe_fc2_type=int4", "qmoe_block_size=64"],
        "int4",
        "cuda",
    )

    assert options["qmoe_fc1_type"] == "int2"
    assert options["qmoe_fc2_type"] == "int4"
    assert options["qmoe_block_size"] == "64"


@pytest.mark.parametrize(
    "execution_provider,block_size,error",
    [("cpu", 64, "only supported on the CUDA EP"), ("cuda", 32, "require qmoe_block_size=64 or 128")],
)
def test_mixed_width_qmoe_options_validate_target(monkeypatch, execution_provider, block_size, error):
    builder = _load_builder_cli_module(monkeypatch)
    with pytest.raises(ValueError, match=error):
        _parse_extra_options(
            builder,
            ["qmoe_fc1_type=int2", f"qmoe_block_size={block_size}"],
            "int4",
            execution_provider,
        )


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp32"])
def test_moe_quant_type_nvfp4_accepts_floating_graph_precision(monkeypatch, precision):
    builder = _load_builder_cli_module(monkeypatch)
    options = _parse_extra_options(builder, ["moe_quant_type=nvfp4"], precision, "cuda")

    assert options["moe_quant_type"] == "nvfp4"


def test_modelopt_selects_native_quantization_from_metadata(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    options = _parse_extra_options(
        builder,
        [],
        quantization_config={"quant_method": "modelopt", "kv_cache_quant_algo": "FP8"},
    )

    assert options["moe_quant_type"] == "nvfp4"
    assert "kv_cache_quant_scheme" not in options


def test_modelopt_rejects_non_cuda_execution_provider(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)

    with pytest.raises(ValueError, match="only supported on the CUDA EP"):
        _parse_extra_options(builder, [], execution_provider="cpu", quantization_config={"quant_method": "modelopt"})


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp32", "int4"])
def test_modelopt_native_quantization_is_independent_of_graph_precision(monkeypatch, precision):
    builder = _load_builder_cli_module(monkeypatch)
    options = _parse_extra_options(builder, [], precision=precision, quantization_config={"quant_method": "modelopt"})

    assert options["moe_quant_type"] == "nvfp4"


def test_base_rejects_packed_expert_quant_type_mismatch():
    model = Model.__new__(Model)
    model.moe_attrs = {"op_type": "QMoE", "quant_type": "fp4"}
    experts = types.SimpleNamespace(quant_type="int")

    with pytest.raises(ValueError, match="Checkpoint experts use int, but QMoE is configured for fp4"):
        model.make_moe_expert_initializers(0, experts)


@pytest.mark.parametrize(
    "fc1_type,fc2_type,expected",
    [
        ("int2", "int4", (2, 4, 2)),
        ("int4", "int2", (4, 2, 4)),
        ("int2", "int2", (2, 2, 2)),
    ],
)
def test_make_moe_init_configures_mixed_width_cuda(fc1_type, fc2_type, expected):
    model = Model.__new__(Model)
    model.ep = "cuda"
    model.moe_attrs = {"swiglu_limit": None}
    model.quant_config = types.SimpleNamespace(
        moe=MoEConfig(type="int4", fc1_type=fc1_type, fc2_type=fc2_type, block_size=64)
    )

    model.make_moe_init()

    assert (
        model.moe_attrs["fc1_expert_weight_bits"],
        model.moe_attrs["fc2_expert_weight_bits"],
        model.moe_attrs["fc3_expert_weight_bits"],
    ) == expected
    assert model.moe_attrs["weights_prepacked"] == 0


def test_mixed_width_expert_initializers_use_projection_bits():
    model = Model.__new__(Model)
    model.ep = "cuda"
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.moe_attrs = {
        "op_type": "QMoE",
        "quant_type": "int",
        "num_experts": 1,
        "expert_weight_bits": 4,
        "fc1_expert_weight_bits": 2,
        "fc2_expert_weight_bits": 4,
        "weights_prepacked": 0,
    }
    model.quant_attrs = {"qmoe_block_size": 64}
    initializers = {}
    model.make_initializer = lambda tensor, name, **kwargs: initializers.setdefault(name, tensor)
    experts = types.SimpleNamespace()

    model.make_moe_expert_initializers(
        0,
        experts,
        torch.zeros(1, 8, 64),
        torch.zeros(1, 64, 8),
    )

    assert initializers["model.layers.0.moe.experts.gate_up_proj.qweight"].shape == (1, 8, 16)
    assert initializers["model.layers.0.moe.experts.down_proj.qweight"].shape == (1, 64, 4)
    assert initializers["model.layers.0.moe.experts.gate_up_proj.scales"].shape == (1, 8, 1)
    assert initializers["model.layers.0.moe.experts.down_proj.scales"].shape == (1, 64, 1)


def test_qmoe_node_emits_mixed_width_attributes():
    model = Model.__new__(Model)
    model.ep = "cuda"
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.moe_attrs = {
        "activation_alpha": 1.0,
        "activation_beta": 0.0,
        "activation_type": "swiglu",
        "expert_weight_bits": 4,
        "fc1_expert_weight_bits": 2,
        "fc2_expert_weight_bits": 4,
        "fc3_expert_weight_bits": 2,
        "normalize_routing_weights": True,
        "quant_type": "int",
        "swiglu_fusion": 1,
        "swiglu_limit": None,
        "top_k": 4,
        "use_sparse_mixer": False,
        "weights_prepacked": 0,
        "block_size": 64,
    }
    captured = {}
    model.make_node = lambda op_type, **kwargs: captured.update(op_type=op_type, **kwargs)
    model.make_value = lambda *args, **kwargs: None
    model.make_hidden_state_shape = lambda: ["tokens", 64]

    model.make_qmoe_op(
        "/moe",
        root_input="input",
        router_probs="router",
        weight1="fc1",
        scales1="fc1_scales",
        weight2="fc2",
        scales2="fc2_scales",
    )

    assert captured["fc1_expert_weight_bits"] == 2
    assert captured["fc2_expert_weight_bits"] == 4
    assert captured["fc3_expert_weight_bits"] == 2
    assert captured["weights_prepacked"] == 0


def test_base_emits_declared_mxfp4_scale_format_and_globals():
    model = Model.__new__(Model)
    model.io_dtype = base_module.ir.DataType.FLOAT16
    model.moe_attrs = {
        "op_type": "QMoE",
        "quant_type": "fp4",
        "global_scale_names": {},
        "zero_point_names": {},
    }
    experts = types.SimpleNamespace(
        quant_type="fp4",
        block_size=32,
        scale_dtype=base_module.ir.DataType.FLOAT8E8M0,
        scales_raw=True,
        weights_prepacked=None,
        gate_up_qweight=torch.zeros(1, 32, 2, dtype=torch.uint8),
        gate_up_scales=torch.zeros(1, 4, 1, dtype=torch.uint8),
        gate_up_zero_points=None,
        gate_up_global_scales=torch.ones(1),
        down_qweight=torch.zeros(1, 32, 1, dtype=torch.uint8),
        down_scales=torch.zeros(1, 2, 1, dtype=torch.uint8),
        down_zero_points=None,
        down_global_scales=torch.ones(1),
    )
    initializers = {}
    model.make_initializer = lambda tensor, name, **kwargs: initializers.setdefault(name, kwargs)

    model.make_moe_expert_initializers(2, experts)

    scale_name = "model.layers.2.moe.experts.gate_up_proj.scales"
    assert initializers[scale_name] == {"to": base_module.ir.DataType.FLOAT8E8M0, "raw": True}
    assert model.moe_attrs["block_size"] == 32
    assert model.moe_attrs["global_scale_names"][2] == (
        "model.layers.2.moe.experts.gate_up_proj.global_scales",
        "model.layers.2.moe.experts.down_proj.global_scales",
    )


def test_gptoss_fp4_delegates_normalized_experts_to_base():
    model = GPTOSSModel.__new__(GPTOSSModel)
    model.moe_attrs = {"op_type": "QMoE", "quant_type": "fp4"}
    model.io_dtype = base_module.ir.DataType.FLOAT16
    packed_experts = object()
    calls = []
    model.load_mxfp4_experts = lambda layer_id: packed_experts
    model.make_moe_expert_initializers = lambda *args: calls.append(args)
    model.make_initializer = lambda *args, **kwargs: None
    experts = types.SimpleNamespace(
        gate_up_proj_bias=torch.ones(2, 4),
        down_proj_bias=torch.ones(2, 2),
    )

    model.make_moe_preprocessing(3, types.SimpleNamespace(experts=experts), "root")

    assert calls == [(3, packed_experts)]


def test_gptoss_integer_qmoe_decodes_mxfp4_checkpoint_experts():
    model = GPTOSSModel.__new__(GPTOSSModel)
    model.moe_attrs = {"op_type": "QMoE", "quant_type": "int"}
    model.io_dtype = base_module.ir.DataType.FLOAT16
    gate_up = torch.ones(2, 4, 8)
    down = torch.ones(2, 8, 2)
    calls = []
    model.load_dense_mxfp4_experts = lambda layer_id: (gate_up, down)
    model.make_moe_expert_initializers = lambda *args: calls.append(args)
    model.make_initializer = lambda *args, **kwargs: None
    experts = types.SimpleNamespace(
        gate_up_proj=torch.empty(2, 8, 4),
        gate_up_proj_bias=torch.ones(2, 4),
        down_proj_bias=torch.ones(2, 8),
    )

    model.make_moe_preprocessing(3, types.SimpleNamespace(experts=experts), "root")

    assert calls == [(3, experts, gate_up, down)]


def test_gptoss_delegates_packed_checkpoint_experts_to_base():
    model = GPTOSSModel.__new__(GPTOSSModel)
    model.moe_attrs = {"op_type": "QMoE", "quant_type": "int"}
    model.io_dtype = base_module.ir.DataType.FLOAT16
    calls = []
    experts = types.SimpleNamespace(
        quant_type="int",
        gate_up_bias=torch.ones(2, 4),
        down_bias=torch.ones(2, 2),
    )
    model.make_moe_expert_initializers = lambda *args: calls.append(args)
    model.make_initializer = lambda *args, **kwargs: None

    model.make_moe_preprocessing(3, types.SimpleNamespace(experts=experts), "root")

    assert calls == [(3, experts)]


class _FakeMoEModel:
    """Minimal stand-in exposing ``make_qmoe_weights`` and recording which
    quantization path it dispatched to."""

    make_qmoe_weights = Model.make_qmoe_weights

    def __init__(self, ep, block_size, weights_prepacked):
        self.ep = ep
        self.qmoe_block_size = block_size
        self.moe_attrs = {"expert_weight_bits": 4, "weights_prepacked": weights_prepacked}
        self.quant_attrs = {"qmoe_block_size": block_size}
        self.calls = []

    def _cutlass_prepacked_blockwise_quantize(self, weights, bits=None):
        self.calls.append(("cutlass", self.qmoe_block_size))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _matmulnbits_blockwise_quantize(self, weights, bits=None):
        self.calls.append(("matmulnbits", self.qmoe_block_size))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _symmetric_blockwise_quantize(self, weights, block_size):
        self.calls.append(("symmetric", block_size))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _cuda_per_channel_quantize(self, weights, prepack):
        self.calls.append(("cuda_per_channel", prepack))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _symmetric_per_channel_quantize(self, weights):
        self.calls.append(("per_channel",))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)


class _RealMoEModel:
    make_qmoe_weights = Model.make_qmoe_weights
    _symmetric_per_channel_quantize = Model._symmetric_per_channel_quantize
    _cuda_per_channel_quantize = Model._cuda_per_channel_quantize

    def __init__(self, ep, block_size, weights_prepacked, bits=4, extra_options=None):
        self.ep = ep
        self.qmoe_block_size = block_size
        self.moe_attrs = {"expert_weight_bits": bits, "weights_prepacked": weights_prepacked}
        self.quant_attrs = {"qmoe_block_size": block_size}
        self.extra_options = extra_options or {}


_W = torch.zeros(8, 128)  # dummy expert weight [N, K]


@pytest.mark.parametrize("weights_prepacked", [-1, 1])
def test_cuda_prepacked_path_for_auto_and_one(weights_prepacked):
    """-1 (auto=prepacked) and 1 (explicitly prepacked) both produce
    CUTLASS-prepacked weights so the emitted layout matches the op attribute."""
    model = _FakeMoEModel("cuda", 128, weights_prepacked)
    model.make_qmoe_weights(_W)
    assert model.calls == [("cutlass", 128)]


def test_cuda_raw_path_for_zero():
    """weights_prepacked=0 ships raw weights for the runtime PrePack hook."""
    model = _FakeMoEModel("cuda", 128, 0)
    model.make_qmoe_weights(_W)
    assert model.calls == [("matmulnbits", 128)]


@pytest.mark.parametrize("ep", ["cpu", "webgpu"])
@pytest.mark.parametrize("weights_prepacked", [-1, 0, 1])
def test_cpu_and_webgpu_use_signed_scale_blockwise_quantizer(ep, weights_prepacked):
    """CPU and WebGPU ship raw MatMulNBits-convention blockwise weights (signed
    block scales) regardless of the CUDA-only weights_prepacked knob, and never
    the CUTLASS-prepacked encoding."""
    model = _FakeMoEModel(ep, 128, weights_prepacked)
    model.make_qmoe_weights(_W)
    assert model.calls == [("matmulnbits", 128)]
    assert model.moe_attrs["block_size"] == 128


@pytest.mark.parametrize("weights_prepacked", [-1, 0, 1])
def test_trt_rtx_keeps_the_original_symmetric_blockwise_encoding(weights_prepacked):
    """The signed-scale grid has not been measured on the TRT-RTX kernel, so that
    EP stays on the encoding it shipped with."""
    model = _FakeMoEModel("trt-rtx", 128, weights_prepacked)
    model.make_qmoe_weights(_W)
    assert model.calls == [("symmetric", 128)]
    assert model.moe_attrs["block_size"] == 128


@pytest.mark.parametrize("ep", ["cpu", "webgpu", "cuda"])
def test_matmulnbits_blockwise_paths_validate_block_size(ep):
    """Every EP on the MatMulNBits grid shares the CUDA block-size constraint."""
    model = _FakeMoEModel(ep, 16, -1)
    with pytest.raises(ValueError, match="block_size 32, 64, or 128"):
        model.make_qmoe_weights(_W)
    assert model.calls == []


@pytest.mark.parametrize("ep,weights_prepacked", [("cpu", -1), ("cuda", 0)])
@pytest.mark.parametrize("bits,k,expected_columns", [(2, 40, 10), (4, 40, 20), (8, 40, 40), (8, 33, 33)])
def test_raw_blockwise_storage_drops_the_block_padding(ep, weights_prepacked, bits, k, expected_columns):
    """The MatMulNBits quantizer pads K up to whole blocks; the QMoE op validates raw storage as
    [E, N, K/pack], so the padding must not reach the initializer."""
    model = _RealMoEModel(ep, 32, weights_prepacked, bits=bits)
    model._matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize.__get__(model)
    torch.manual_seed(0)
    weights = torch.randn(3, k)
    qweight, scales = model.make_qmoe_weights(weights)
    assert tuple(qweight.shape) == (3, expected_columns)
    assert tuple(scales.shape) == (3, 2)
    padded, _ = cuda_quantizer_module.CudaQuantizer.qmoe_blockwise_quantize(weights, bits, 32)
    assert tuple(padded.shape) == (3, 2 * (32 // (8 // bits)))
    assert torch.equal(qweight, padded[:, :expected_columns])


@pytest.mark.parametrize("ep,weights_prepacked", [("cpu", -1), ("webgpu", -1), ("cuda", 0)])
def test_raw_blockwise_int4_rejects_odd_input_dimension(ep, weights_prepacked):
    model = _RealMoEModel(ep, 32, weights_prepacked)
    model._matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize.__get__(model)
    with pytest.raises(RuntimeError, match=r"INT4 QMoE requires expert input dimension K \(33\) to be divisible by 2"):
        model.make_qmoe_weights(torch.zeros(4, 33))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_webgpu_blockwise_rejects_partial_blocks(bits, block_size):
    model = _RealMoEModel("webgpu", block_size, -1, bits=bits)
    model._matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize.__get__(model)
    with pytest.raises(RuntimeError, match=rf"WebGPU QMoE.*K \(40\).*qmoe_block_size \({block_size}\)"):
        model.make_qmoe_weights(torch.zeros(4, 40))


@pytest.mark.parametrize("ep,weights_prepacked", [("cpu", -1), ("webgpu", -1)])
def test_non_cuda_blockwise_int8_uses_offset_128_storage(ep, weights_prepacked):
    """INT8 QMoE bytes are q + 128 (the CPU kernel dequantizes with default_zp_8bit = 128), on the
    same signed-scale grid as INT4: the block's max-magnitude element maps exactly to qmin."""
    model = _RealMoEModel(ep, 32, weights_prepacked, bits=8)
    model._matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize.__get__(model)
    weights = torch.zeros(1, 32)
    weights[0, 0] = 8.0  # positive extreme of block 0
    weights[0, 1] = -4.0
    weights[0, 2] = 0.0625
    qweight, scales = model.make_qmoe_weights(weights)
    assert qweight.dtype == torch.uint8 and tuple(qweight.shape) == (1, 32)
    assert scales.shape == (1, 1) and scales[0, 0] == -0.0625  # signed: 8.0 / qmin(-128)
    assert qweight[0, 0].item() == 0  # 8.0 -> qmin (-128) + 128, exact
    assert qweight[0, 1].item() == 64 + 128  # -4.0 / -0.0625 = 64
    assert qweight[0, 2].item() == 127  # 0.0625 / -0.0625 = -1
    assert qweight[0, 3].item() == 128  # zero -> the zero point
    dequantized = (qweight[0].to(torch.float32) - 128.0) * scales[0, 0].to(torch.float32)
    assert torch.equal(dequantized, weights[0])


def test_cuda_raw_blockwise_int2_packs_four_codes_per_byte():
    model = _RealMoEModel("cuda", 64, 0, bits=2)
    model._matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize.__get__(model)
    weights = torch.zeros(1, 64)
    weights[0, :4] = torch.tensor([2.0, 1.0, 0.0, -1.0])

    qweight, scales = model.make_qmoe_weights(weights)

    assert tuple(qweight.shape) == (1, 16)
    assert scales[0, 0] == -1.0
    assert qweight[0, 0].item() == 0b11100100


@pytest.mark.parametrize("ep", ["cpu", "webgpu"])
def test_non_cuda_blockwise_scales_are_signed_and_do_not_clip_the_extreme(ep):
    """The signed-scale grid maps each block's max-magnitude element exactly to
    qmin, so a positive extreme is no longer clipped to 7/8 of its value. This is
    also the grid the CPU QMoE MLAS Q4 fast path re-quantizes to, keeping that path
    lossless."""
    model = _RealMoEModel(ep, 32, -1, bits=4)
    model._matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize.__get__(model)
    weights = torch.zeros(1, 32)
    weights[0, 0] = 8.0  # positive extreme of block 0
    weights[0, 1] = -4.0
    qweight, scales = model.make_qmoe_weights(weights)
    assert scales.shape == (1, 1) and scales[0, 0] == -1.0  # signed: 8.0 / qmin(-8)
    q = qweight[0, 0].item()
    assert (q & 0xF) == 0  # 8.0 -> qmin (-8) + 8 == 0, exact
    assert (q >> 4) == 12  # -4.0 / -1.0 = 4 -> 4 + 8


@pytest.mark.parametrize(
    "weights_prepacked,expected_prepack",
    [(-1, True), (0, False), (1, True)],
)
@pytest.mark.parametrize("block_size", [-1, 0])
def test_cuda_per_channel_path_for_non_positive_block_size(block_size, weights_prepacked, expected_prepack):
    """qmoe_block_size <= 0 means per-channel QMoE scales and must not emit a
    block_size attribute."""
    model = _FakeMoEModel("cuda", block_size, weights_prepacked)
    model.moe_attrs["block_size"] = 128
    model.make_qmoe_weights(_W)
    assert model.calls == [("cuda_per_channel", expected_prepack)]
    assert "block_size" not in model.moe_attrs


def test_non_cuda_per_channel_path_for_non_positive_block_size():
    model = _FakeMoEModel("cpu", 0, -1)
    model.moe_attrs["block_size"] = 128
    model.make_qmoe_weights(_W)
    assert model.calls == [("per_channel",)]
    assert "block_size" not in model.moe_attrs


def test_per_channel_int4_uses_qmoe_full_range_unsigned_offset_storage_by_default():
    model = _RealMoEModel("cpu", 0, -1, bits=4)
    weights = torch.tensor([[-8.0, -7.0, 0.0, 7.0]], dtype=torch.float32)

    qweight, scales = model._symmetric_per_channel_quantize(weights)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, torch.tensor([[0x10, 0xF8]], dtype=torch.uint8))


def test_per_channel_int8_uses_qmoe_full_range_unsigned_offset_storage_by_default():
    model = _RealMoEModel("cpu", 0, -1, bits=8)
    weights = torch.tensor([[-128.0, -127.0, 0.0, 127.0]], dtype=torch.float32)

    qweight, scales = model._symmetric_per_channel_quantize(weights)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, torch.tensor([[0x00, 0x01, 0x80, 0xFF]], dtype=torch.uint8))


@pytest.mark.parametrize(
    "bits,weights,expected_qweight",
    [
        (
            4,
            torch.tensor([[-8.0, -7.0, 0.0, 7.0]], dtype=torch.float32),
            torch.tensor([[0x10, 0xF8]], dtype=torch.uint8),
        ),
        (
            8,
            torch.tensor([[-128.0, -127.0, 0.0, 127.0]], dtype=torch.float32),
            torch.tensor([[0x00, 0x01, 0x80, 0xFF]], dtype=torch.uint8),
        ),
    ],
)
def test_per_channel_can_emit_full_range_unsigned_offset_storage(bits, weights, expected_qweight):
    qweight, scales = qmoe_symmetric_per_channel_quantize(weights, bits)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, expected_qweight)


@pytest.mark.parametrize(
    "bits,weights,expected_qweight",
    [
        (
            4,
            torch.tensor([[-7.0, -3.0, 0.0, 7.0]], dtype=torch.float32),
            torch.tensor([[0x51, 0xF8]], dtype=torch.uint8),
        ),
        (
            8,
            torch.tensor([[-127.0, -1.0, 0.0, 127.0]], dtype=torch.float32),
            torch.tensor([[0x01, 0x7F, 0x80, 0xFF]], dtype=torch.uint8),
        ),
    ],
)
def test_per_channel_can_emit_legacy_15_bucket_unsigned_offset_storage(bits, weights, expected_qweight):
    qweight, scales = qmoe_symmetric_per_channel_quantize(weights, bits, unsigned_full_range=False)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, expected_qweight)


@pytest.mark.parametrize("weights_prepacked", [-1, 0, 1])
@pytest.mark.parametrize("bad_block", [16, 256])
def test_cuda_rejects_unsupported_block_size(weights_prepacked, bad_block):
    """Unsupported block sizes must raise a real exception (not an assert that
    ``python -O`` would strip)."""
    model = _FakeMoEModel("cuda", bad_block, weights_prepacked)
    with pytest.raises(ValueError, match="block_size 32, 64, or 128"):
        model.make_qmoe_weights(_W)


def _ort_cuda_available():
    try:
        import onnxruntime as ort  # noqa: PLC0415
        from onnxruntime.capi import _pybind_state as _pyb  # noqa: PLC0415

        return (
            hasattr(_pyb, "quantize_matmul_4bits")
            and hasattr(_pyb, "pack_weights_for_cuda_mixed_gemm")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        )
    except Exception:
        return False


def _ort_qmoe_pack_available():
    try:
        from onnxruntime.capi import _pybind_state as _pyb  # noqa: PLC0415

        return hasattr(_pyb, "pack_weights_for_cuda_mixed_gemm")
    except Exception:
        return False


@pytest.mark.skipif(
    not _ort_qmoe_pack_available(),
    reason="onnxruntime QMoE pack pybind not available",
)
@pytest.mark.parametrize(
    "bits,raw_shape,packed_shape",
    [(4, (64, 64), (128, 32)), (8, (64, 128), (128, 64))],
)
def test_cuda_per_channel_quantization_shapes(bits, raw_shape, packed_shape):
    torch.manual_seed(0)
    weights = torch.randn(64, 128) * 0.05

    raw_model = _RealMoEModel("cuda", 0, 0, bits=bits)
    raw_qweight, raw_scales = raw_model.make_qmoe_weights(weights)
    assert raw_qweight.dtype == torch.uint8
    assert tuple(raw_qweight.shape) == raw_shape
    assert tuple(raw_scales.shape) == (64,)
    assert (raw_scales > 0).all()
    assert "block_size" not in raw_model.moe_attrs

    prepacked_model = _RealMoEModel("cuda", 0, -1, bits=bits)
    prepacked_qweight, prepacked_scales = prepacked_model.make_qmoe_weights(weights)
    assert prepacked_qweight.dtype == torch.uint8
    assert tuple(prepacked_qweight.shape) == packed_shape
    assert tuple(prepacked_scales.shape) == (64,)
    assert (prepacked_scales > 0).all()
    assert "block_size" not in prepacked_model.moe_attrs


@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_per_channel_quantization_uses_qmoe_symmetric_storage(bits):
    torch.manual_seed(0)
    n = 17
    k = 16
    weights = torch.randn(n, k, dtype=torch.float32) * 0.05
    model = _RealMoEModel("cuda", 0, 0, bits=bits)

    qweight, scales = model.make_qmoe_weights(weights)

    if bits == 4:
        qmin, qmax, scale_divisor, zero_point = -8, 7, 8, 8
    else:
        qmin, qmax, scale_divisor, zero_point = -128, 127, 128, 128

    expected_scales = torch.clamp(
        weights.abs().amax(dim=1, keepdim=True) / float(scale_divisor), min=torch.finfo(torch.float32).eps
    )
    quantized = torch.clamp(torch.round(weights / expected_scales), qmin, qmax).to(torch.int16)
    quantized = (quantized + zero_point).to(torch.uint8)
    if bits == 4:
        expected_qweight = quantized[:, 0::2] | (quantized[:, 1::2] << 4)
    else:
        expected_qweight = quantized

    assert torch.equal(qweight, expected_qweight)
    assert torch.allclose(scales.float(), expected_scales.squeeze(-1), atol=1e-6, rtol=1e-3)
    assert quantized.min() == 0
    assert quantized.max() == (15 if bits == 4 else 255)


def test_cuda_raw_per_channel_quantization_does_not_require_qmoe_pack_pybind(monkeypatch):
    torch.manual_seed(0)
    weights = torch.randn(17, 16, dtype=torch.float32) * 0.05
    model = _RealMoEModel("cuda", 0, 0, bits=4)

    qweight, scales = model.make_qmoe_weights(weights)

    assert qweight.dtype == torch.uint8
    assert tuple(qweight.shape) == (17, 8)
    assert tuple(scales.shape) == (17,)
    assert "block_size" not in model.moe_attrs


@pytest.mark.skipif(not _ort_cuda_available(), reason="onnxruntime CUDA pybind not available")
def test_cutlass_prepacked_scales_preserve_mlas_sign():
    model = _FakeMoEModel("cuda", 128, -1)
    torch.manual_seed(0)
    weights = torch.randn(256, 256) * 0.05  # [N, K]
    qweight, scales = Model._cutlass_prepacked_blockwise_quantize(model, weights)

    blocked = weights.reshape(256, 2, 128)
    argmax = blocked.abs().argmax(dim=2, keepdim=True)
    expected_scales = blocked.gather(2, argmax).squeeze(2) / -8.0

    assert qweight.dtype == torch.uint8
    assert tuple(qweight.shape) == (256, 128)  # [K, N/2] for INT4
    assert tuple(scales.shape) == (256, 2)  # [N, K/block]
    assert (scales < 0).any() and (scales > 0).any()
    assert torch.equal(scales, expected_scales)
