# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Small, offline regressions for Bailing checkpoint loading and graph export."""

import builtins
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from onnxruntime.capi._pybind_state import get_all_operator_schema
from safetensors.torch import save_file
from transformers import PretrainedConfig

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

import builder
from builders import bailing
from builders.bailing import BailingMoeV3Model
from loaders import bailing as bailing_loader
from loaders.bailing import BailingModel
from loaders.base import QuantizedDecoderLayer, QuantizedExperts, TensorModule


def _model(
    tmp_path,
    *,
    dtype=ir.DataType.FLOAT,
    precision=None,
    ep="cpu",
    quant_method=None,
    layer_group_size=4,
    config_options=None,
    **options,
):
    config = PretrainedConfig(
        architectures=["BailingMoeV3ForCausalLM"],
        _name_or_path="test/bailing",
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        head_dim=64,
        rope_theta=10000.0,
    )
    if layer_group_size is not None:
        config.layer_group_size = layer_group_size
    for name, value in (config_options or {}).items():
        setattr(config, name, value)
    if quant_method:
        config.quantization_config = {"quant_method": quant_method}
    return BailingMoeV3Model(config, dtype, precision or dtype, ep, str(tmp_path / "cache"), options)


def _checkpoint(tmp_path, out_features=16, in_features=128):
    scales = torch.arange(
        1, ((out_features + 127) // 128) * ((in_features + 127) // 128) + 1, dtype=torch.float32
    ).reshape((out_features + 127) // 128, (in_features + 127) // 128)
    save_file(
        {
            "model.word_embeddings.weight": torch.ones(32, 128, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
            "projection.weight": torch.ones(out_features, in_features).to(torch.float8_e4m3fn),
            "projection.weight_scale_inv": scales,
        },
        tmp_path / "model.safetensors",
    )
    return scales


def _layer_checkpoint(tmp_path, model, weight_format="bf16"):
    tensors = {
        "model.word_embeddings.weight": torch.ones(model.vocab_size, model.hidden_size, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(model.hidden_size, dtype=torch.bfloat16),
    }

    def linear(prefix, out_features, in_features):
        if weight_format == "int4":
            tensors[f"{prefix}.weight_packed"] = torch.full(
                (out_features, in_features // 8), 0x76543210, dtype=torch.int32
            )
            tensors[f"{prefix}.weight_scale"] = torch.ones(out_features, in_features // 32)
        else:
            weight = torch.ones(out_features, in_features)
            tensors[f"{prefix}.weight"] = weight.to(torch.float8_e4m3fn if weight_format == "fp8" else torch.bfloat16)
            if weight_format == "fp8":
                tensors[f"{prefix}.weight_scale_inv"] = torch.ones(
                    (out_features + 127) // 128, (in_features + 127) // 128
                )

    def mlp(prefix, intermediate):
        linear(f"{prefix}.gate_proj", intermediate, model.hidden_size)
        linear(f"{prefix}.up_proj", intermediate, model.hidden_size)
        linear(f"{prefix}.down_proj", model.hidden_size, intermediate)

    linear("lm_head", model.vocab_size, model.hidden_size)
    for layer_id in range(model.num_layers):
        prefix = f"model.layers.{layer_id}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{prefix}.{name}.weight"] = torch.ones(model.hidden_size, dtype=torch.bfloat16)
        attention = f"{prefix}.attention"
        if model.layer_types[layer_id] == "full_attention":
            mla = model.mla_attrs
            linear(f"{attention}.q_a_proj", mla["q_lora_rank"], model.hidden_size)
            linear(f"{attention}.q_b_proj", model.num_attn_heads * model.head_size, mla["q_lora_rank"])
            linear(f"{attention}.kv_a_proj_with_mqa", mla["kv_lora_rank"] + mla["qk_rope_head_dim"], model.hidden_size)
            linear(
                f"{attention}.kv_b_proj",
                model.num_attn_heads * (mla["qk_nope_head_dim"] + mla["v_head_dim"]),
                mla["kv_lora_rank"],
            )
            linear(f"{attention}.g_proj", model.num_attn_heads, model.hidden_size)
            linear(f"{attention}.dense", model.hidden_size, model.num_attn_heads * mla["v_head_dim"])
            for name, rank in (("q_a_layernorm", mla["q_lora_rank"]), ("kv_a_layernorm", mla["kv_lora_rank"])):
                tensors[f"{attention}.{name}.weight"] = torch.ones(rank, dtype=torch.bfloat16)
        else:
            dim = model.linear_value_dim
            for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                linear(f"{attention}.{name}", dim, model.hidden_size)
            linear(f"{attention}.b_proj", model.linear_num_value_heads, model.hidden_size)
            linear(f"{attention}.o_proj", model.hidden_size, dim)
            for name in ("q", "k", "v"):
                tensors[f"{attention}.{name}_conv1d.weight"] = torch.ones(dim, 1, model.linear_conv_kernel_dim)
            tensors[f"{attention}.A_log"] = torch.zeros(model.linear_num_value_heads)
            tensors[f"{attention}.dt_bias"] = torch.zeros(dim)
            tensors[f"{attention}.o_norm.weight"] = torch.ones(model.linear_value_head_dim)
        mlp_prefix = f"{prefix}.mlp"
        if layer_id < model.moe_attrs["first_k_dense_replace"]:
            mlp(mlp_prefix, model.moe_attrs["dense_intermediate_size"])
        else:
            linear(f"{mlp_prefix}.gate", model.moe_attrs["num_experts"], model.hidden_size)
            tensors[f"{mlp_prefix}.gate.expert_bias"] = torch.zeros(model.moe_attrs["num_experts"])
            mlp(f"{mlp_prefix}.shared_experts", model.moe_attrs["shared_expert_intermediate_size"])
            for expert_id in range(model.moe_attrs["num_experts"]):
                mlp(f"{mlp_prefix}.experts.{expert_id}", model.intermediate_size)
    save_file(tensors, tmp_path / "model.safetensors")
    return tensors


def _mla_weights(model, *, sparse=False):
    attrs = model.mla_attrs
    shapes = {
        "q_a_proj": (attrs["q_lora_rank"], model.hidden_size),
        "q_b_proj": (model.num_attn_heads * model.head_size, attrs["q_lora_rank"]),
        "kv_a_proj_with_mqa": (attrs["kv_lora_rank"] + attrs["qk_rope_head_dim"], model.hidden_size),
        "kv_b_proj": (
            model.num_attn_heads * (attrs["qk_nope_head_dim"] + attrs["v_head_dim"]),
            attrs["kv_lora_rank"],
        ),
        "g_proj": (model.num_attn_heads, model.hidden_size),
        "dense": (model.hidden_size, model.num_attn_heads * attrs["v_head_dim"]),
    }
    generator = torch.Generator().manual_seed(23)
    attention = SimpleNamespace(
        **{name: TensorModule(weight=torch.randn(shape, generator=generator) * 0.1) for name, shape in shapes.items()}
    )
    if sparse:
        # ARM64's native FP16 MatMul rounds each FMA to FP16, whereas the
        # reference accumulates in FP32. Four power-of-two coefficients per
        # row bound that drift while still mixing distinct input channels.
        # Keep dense random weights for the full CPU/CUDA attention test.
        for name, (out_features, in_features) in shapes.items():
            rows = torch.arange(out_features)[:, None]
            columns = (7 * rows + torch.tensor([0, 5, 13, 23])) % in_features
            weight = torch.zeros(out_features, in_features)
            weight[rows, columns] = torch.tensor([0.125, -0.25, 0.5, -0.125])
            getattr(attention, name).weight = weight
    attention.q_a_layernorm = TensorModule(weight=torch.linspace(0.5, 1.5, attrs["q_lora_rank"]))
    attention.kv_a_layernorm = TensorModule(weight=torch.linspace(0.5, 1.5, attrs["kv_lora_rank"]))
    return attention


def _mla_reference(model, attention, hidden, positions=None, attn_output=None):
    dtype = torch.float16 if model.io_dtype == ir.DataType.FLOAT16 else torch.float32
    attrs = model.mla_attrs
    token_shape = hidden.shape[:-1]
    hidden = torch.from_numpy(hidden).float()

    def rounded(value):
        return value.to(dtype).float()

    def linear(name, value):
        return rounded(value @ getattr(attention, name).weight.to(dtype).float().T)

    def normalized(value, scale):
        return rounded(
            value
            / torch.sqrt(value.square().mean(dim=-1, keepdim=True) + model.layernorm_attrs["epsilon"])
            * scale.to(dtype).float()
        )

    def rotary(value):
        frequencies = torch.from_numpy(positions).float()[..., None] / (
            model.rope_attrs["theta"]
            ** (torch.arange(0, attrs["qk_rope_head_dim"], 2).float() / attrs["qk_rope_head_dim"])
        )
        cos, sin = frequencies.cos()[..., None, :], frequencies.sin()[..., None, :]
        if model.rope_attrs["interleaved"]:
            output = torch.empty_like(value)
            output[..., 0::2] = value[..., 0::2] * cos - value[..., 1::2] * sin
            output[..., 1::2] = value[..., 1::2] * cos + value[..., 0::2] * sin
        else:
            left, right = value.chunk(2, dim=-1)
            output = torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)
        return rounded(output)

    if attn_output is not None:
        values = torch.from_numpy(attn_output).float().reshape(*token_shape, model.num_attn_heads, model.head_size)
        values = values[..., : attrs["v_head_dim"]]
        gates = rounded(linear("g_proj", hidden).sigmoid())
        gated = rounded(values * gates[..., None]).reshape(*token_shape, model.num_attn_heads * attrs["v_head_dim"])
        return linear("dense", gated).to(dtype).numpy()
    query = linear("q_b_proj", normalized(linear("q_a_proj", hidden), attention.q_a_layernorm.weight))
    query = query.reshape(*token_shape, model.num_attn_heads, model.head_size)
    q_nope, q_rope = query.split([attrs["qk_nope_head_dim"], attrs["qk_rope_head_dim"]], dim=-1)
    query = torch.cat([q_nope, rotary(q_rope)], dim=-1)
    compressed, k_rope = linear("kv_a_proj_with_mqa", hidden).split(
        [attrs["kv_lora_rank"], attrs["qk_rope_head_dim"]], dim=-1
    )
    kv = linear("kv_b_proj", normalized(compressed, attention.kv_a_layernorm.weight))
    kv = kv.reshape(*token_shape, model.num_attn_heads, attrs["qk_nope_head_dim"] + attrs["v_head_dim"])
    k_nope, values = kv.split([attrs["qk_nope_head_dim"], attrs["v_head_dim"]], dim=-1)
    k_rope = rotary(k_rope[..., None, :]).expand(*token_shape, model.num_attn_heads, attrs["qk_rope_head_dim"])
    key = torch.cat([k_nope, k_rope], dim=-1)
    value = torch.cat([values, torch.zeros_like(k_rope)], dim=-1)
    return tuple(
        tensor.reshape(*token_shape, model.num_attn_heads * model.head_size).to(dtype).numpy()
        for tensor in (query, key, value)
    )


@pytest.mark.parametrize("trust", [False, True])
def test_config_failure_preserves_remote_code_policy(tmp_path, monkeypatch, trust):
    load_config = Mock(side_effect=ValueError("custom configuration requires remote code"))
    tokenizer = Mock()
    monkeypatch.setattr(builder.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(builder.LFM2AudioModel, "load_config", Mock(return_value=None))
    options = {"hf_remote": trust}

    with pytest.raises(ValueError, match="requires remote code"):
        builder.get_hf_details("test/bailing", "", str(tmp_path), options)

    load_config.assert_called_once_with("test/bailing", token=True, trust_remote_code=trust, cache_dir=str(tmp_path))
    tokenizer.assert_not_called()
    assert options["hf_remote"] is trust


def test_explicit_remote_code_opt_in_reaches_tokenizer(tmp_path, monkeypatch):
    config = PretrainedConfig(architectures=["BailingMoeV3ForCausalLM"])
    monkeypatch.setattr(builder.AutoConfig, "from_pretrained", Mock(return_value=config))
    tokenizer = Mock(return_value=Mock(bos_token_id=1, eos_token_id=2, pad_token_id=0))
    tokenizer.return_value.get_vocab.return_value = {}
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", tokenizer)

    builder.get_hf_details("test/bailing", "", str(tmp_path), {"hf_remote": True})

    assert tokenizer.call_args.kwargs["trust_remote_code"] is True


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_cpu_graph_declares_float_recurrent_state(tmp_path, dtype):
    model = _model(tmp_path, dtype=dtype)
    model.make_inputs_and_outputs()

    states = [value for value in [*model.model.graph.inputs, *model.model.graph.outputs] if ".recurrent" in value.name]
    assert len(states) == 6
    assert all(value.dtype == ir.DataType.FLOAT for value in states)
    assert all(list(value.shape)[1:] == [2, 64, 64] for value in states)
    conv_states = [value for value in model.model.graph.inputs if ".conv" in value.name]
    assert all(list(value.shape)[1:] == [384, 3] for value in conv_states)


@pytest.mark.parametrize("ep", ["cuda", "webgpu", "cpu"])
@pytest.mark.parametrize("options", [{"state_update_capacity": 1}, {"state_window": 2}])
def test_compact_capture_rejected_for_per_key_decay(tmp_path, ep, options):
    with pytest.raises(ValueError, match="capture and replay do not support per-key decay"):
        _model(tmp_path, dtype=ir.DataType.FLOAT16, ep=ep, use_paged_attention=ep != "cpu", **options)


def test_streaming_loader_rejects_adapters_before_loading(tmp_path, monkeypatch):
    model = _model(tmp_path, adapter_path="test/adapter")
    download = Mock()
    monkeypatch.setattr(bailing_loader, "snapshot_download", download)

    with pytest.raises(ValueError, match="does not support adapter_path"):
        model.load_weights("")

    download.assert_not_called()


@pytest.mark.parametrize("remote", [False, True])
def test_streaming_loader_resolves_checkpoint_source(tmp_path, monkeypatch, remote):
    _checkpoint(tmp_path)
    model = _model(tmp_path, hf_token=False, base_revision="test-revision")
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(bailing_loader, "snapshot_download", download)

    weights = model.load_weights("" if remote else str(tmp_path))
    try:
        assert weights.input_path == str(tmp_path)
        assert weights.embedding.weight.shape == (32, 128)
    finally:
        weights.close()
    if remote:
        download.assert_called_once_with(
            repo_id="test/bailing",
            cache_dir=model.cache_dir,
            token=False,
            revision="test-revision",
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    else:
        download.assert_not_called()


@pytest.mark.parametrize("quant_method", ["fp8", "compressed-tensors"])
def test_dequantized_source_is_quantized_to_int4_qdq(tmp_path, quant_method):
    _checkpoint(tmp_path)
    model = _model(tmp_path, precision=ir.DataType.INT4, quant_method=quant_method, use_qdq=True, block_size=32)
    model.model.graph.inputs.append(model.make_value("input", ir.DataType.FLOAT, [1, 1, 128]))
    weights = model.load_weights(str(tmp_path))
    try:
        projection = weights.make_linear_module("projection")
        assert projection.weight.dtype == torch.bfloat16
        model.make_matmul(projection, "/projection/MatMul", "input")
    finally:
        weights.close()
    model.model.graph.outputs.append(model.values["/projection/MatMul/output_0"])
    out_dir = tmp_path / "export"
    out_dir.mkdir()

    model.save_model(str(out_dir))

    exported = onnx.load(out_dir / "model.onnx")
    assert any(node.op_type == "DequantizeLinear" for node in exported.graph.node)
    assert any(
        tensor.data_type in {onnx.TensorProto.INT4, onnx.TensorProto.UINT4} for tensor in exported.graph.initializer
    )


@pytest.mark.parametrize("layout", ["scalar", "channel", "row", "column"])
def test_fp8_per_channel_layouts_keep_one_input_block(tmp_path, layout):
    model = _model(tmp_path)
    weight = torch.ones(16, 4096).to(torch.float8_e4m3fn)
    scales = torch.arange(1, 17, dtype=torch.float32)
    layouts = {"scalar": torch.tensor(2.0), "channel": scales, "row": scales[None, :], "column": scales[:, None]}

    model.make_matmul_block_quantized_fp8_weight("/projection", "input", weight, layouts[layout])

    node = model.values["/projection/output_0"].producer()
    assert node.attributes["block_size"].as_int() == 4096
    expected = np.full((16, 1), 2.0) if layout == "scalar" else scales.numpy()[:, None]
    np.testing.assert_array_equal(node.inputs[2].const_value.numpy(), expected)


def test_block_fp8_metadata_and_partial_blocks_survive_concatenation(tmp_path):
    scales = _checkpoint(tmp_path, out_features=130, in_features=257)
    model = _model(tmp_path, quant_method="fp8")
    weights = model.load_weights(str(tmp_path))
    try:
        projection = weights.make_linear_module("projection")
        fused = weights.concat_linears([projection, projection])
        model.make_matmul(fused, "/projection", "input")
    finally:
        weights.close()

    node = model.values["/projection/output_0"].producer()
    assert node.attributes["block_size"].as_int() == 128
    normalized = scales.repeat_interleave(128, dim=0)[:130].numpy()
    np.testing.assert_array_equal(node.inputs[2].const_value.numpy(), np.concatenate([normalized, normalized]))


@pytest.mark.parametrize("activation", [None, "silu", "sigmoid"])
def test_trt_rms_norm_expansion_matches_gate_math(tmp_path, activation):
    model = _model(tmp_path, ep="trt-rtx")
    shape = [1, 2, 128]
    for name in ("input", "gate"):
        model.model.graph.inputs.append(model.make_value(name, ir.DataType.FLOAT, shape))
    scale = torch.linspace(0.5, 1.5, 64)
    model.make_initializer(scale, "scale")
    model.make_gated_rms_norm("/norm", "input", "scale", "gate", shape, activation=activation)
    model.model.graph.outputs.append(model.values["/norm/output_0"])
    model.model.graph.sort()
    inputs = np.linspace(-2, 2, 256, dtype=np.float32).reshape(shape)
    gate = np.linspace(-3, 3, 256, dtype=np.float32).reshape(shape)

    actual = ReferenceEvaluator(ir.serde.serialize_model(model.model)).run(None, {"input": inputs, "gate": gate})[0]

    grouped = inputs.reshape(1, 2, 2, 64)
    normalized = (grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-5) * scale.numpy()).reshape(shape)
    expected = normalized / (1 + np.exp(-gate))
    if activation != "sigmoid":
        expected *= gate
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("num_layers", "group_size", "full_layers"),
    [(6, 4, [3, 4, 5]), (3, 4, [0, 1, 2]), (5, 4, [3, 4]), (4, 4, [3]), (3, None, [0, 1, 2])],
)
def test_layer_schedule_matches_hf_incomplete_group(tmp_path, num_layers, group_size, full_layers):
    model = _model(tmp_path, layer_group_size=group_size, config_options={"num_hidden_layers": num_layers})

    assert [i for i, kind in enumerate(model.layer_types) if kind == "full_attention"] == full_layers
    assert model.layer_types[-1] == "full_attention"


@pytest.mark.parametrize("group_size", [0, -1, True, 2.5, "4", [1, 1, 0, 0]])
def test_unsupported_layer_group_size_is_rejected(tmp_path, group_size):
    with pytest.raises(ValueError, match="layer_group_size must be a positive integer"):
        _model(tmp_path, layer_group_size=group_size)


@pytest.mark.parametrize("group_topk", [0, -1, True, 1.5, "2"])
def test_invalid_group_topk_is_rejected(tmp_path, group_topk):
    with pytest.raises(ValueError, match="group_topk must be a positive integer"):
        _model(tmp_path, config_options={"group_topk": group_topk})


def test_router_defaults_match_sglang(tmp_path):
    model = _model(tmp_path)

    assert model.moe_attrs["normalize_routing_weights"] is False
    assert model.moe_attrs["routed_scaling_factor"] == 1.0
    assert model.moe_attrs["group_topk"] == 2
    assert model.kda_attrs["gate_lower_bound"] is None


@pytest.mark.parametrize("lower_bound", [None, 0, 1, float("nan"), float("-inf"), True, "-5"])
def test_safe_gate_requires_negative_finite_bound(tmp_path, lower_bound):
    with pytest.raises(ValueError, match="requires a finite negative kda_lower_bound"):
        _model(tmp_path, config_options={"kda_safe_gate": True, "kda_lower_bound": lower_bound})


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_mla_rope_matches_hf_layout(tmp_path, interleaved, packed):
    model = _model(
        tmp_path,
        dtype=ir.DataType.FLOAT16 if packed else ir.DataType.FLOAT,
        ep="cuda" if packed else "cpu",
        use_paged_attention=packed,
        config_options={"rope_interleave": interleaved},
    )
    heads, rotary_dim = 2, model.mla_attrs["qk_rope_head_dim"]
    token_shape = [4] if packed else [2, 2]
    shape = [*token_shape, heads * rotary_dim]
    model.model.graph.inputs.append(model.make_value("input", model.io_dtype, shape))
    positions = np.array([0, 1, 3, 7], dtype=np.int64).reshape(token_shape)
    model.model.graph.inputs.append(model.make_value(model.input_names["position_ids"], ir.DataType.INT64, token_shape))
    output = model.make_mla_rope("/rope", "input", heads, token_shape)
    model.model.graph.outputs.append(model.values[output])
    model.model.graph.sort()
    inputs = (
        np.linspace(-2, 2, np.prod(shape), dtype=np.float32).reshape(shape).astype(np.float16 if packed else np.float32)
    )
    session = ort.InferenceSession(
        ir.serde.serialize_model(model.model).SerializeToString(), providers=["CPUExecutionProvider"]
    )

    actual = session.run(None, {"input": inputs, model.input_names["position_ids"]: positions})[0]

    values = inputs.astype(np.float32).reshape(*token_shape, heads, rotary_dim)
    frequencies = positions[..., None] / (
        model.rope_attrs["theta"] ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )
    cos, sin = np.cos(frequencies)[..., None, :], np.sin(frequencies)[..., None, :]
    if interleaved:
        expected = np.empty_like(values)
        expected[..., 0::2] = values[..., 0::2] * cos - values[..., 1::2] * sin
        expected[..., 1::2] = values[..., 1::2] * cos + values[..., 0::2] * sin
    else:
        left, right = np.split(values, 2, axis=-1)
        expected = np.concatenate([left * cos - right * sin, right * cos + left * sin], axis=-1)
    np.testing.assert_allclose(
        actual, expected.reshape(shape), rtol=2e-3 if packed else 1e-5, atol=2e-3 if packed else 1e-6
    )


def register_cuda_plugin_ep():
    """Registers the standalone CUDA plugin EP when the core onnxruntime wheel does not bundle CUDA."""
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return
    try:
        cuda_ep = importlib.import_module("onnxruntime_ep_cuda")
    except ImportError:
        return
    if not any(device.ep_name == cuda_ep.get_ep_name() for device in ort.get_ep_devices()):
        ort.register_execution_provider_library(cuda_ep.get_ep_name(), cuda_ep.get_library_path())


def prepare_cuda_provider():
    """Makes CUDAExecutionProvider usable from an onnxruntime-gpu wheel or from the standalone plugin EP."""
    if hasattr(ort, "preload_dlls"):
        # onnxruntime-gpu does not search the NVIDIA pip packages for its CUDA libraries on its own.
        ort.preload_dlls()
    register_cuda_plugin_ep()


def kernel_session(model, provider, required_op):
    def unavailable(reason):
        if os.environ.get("ORTGENAI_REQUIRE_BAILING_KERNELS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    schemas = {schema.name: schema for schema in get_all_operator_schema() if schema.domain == "com.microsoft"}
    if required_op not in schemas or (
        required_op == "GatedDeltaNet" and "gate_lower_bound" not in schemas[required_op].attributes
    ):
        unavailable(f"Requires an ONNX Runtime build with {required_op} from ORT PR #33817")
    if provider == "CUDAExecutionProvider":
        try:
            prepare_cuda_provider()
        except Exception as exc:
            unavailable(f"Preparing the CUDA provider failed: {exc}")
    if provider not in ort.get_available_providers():
        unavailable(f"{provider} is unavailable")
    options = ort.SessionOptions()
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    model.model.graph.sort()
    return ort.InferenceSession(
        ir.serde.serialize_model(model.model).SerializeToString(), sess_options=options, providers=[provider]
    )


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("group_topk", [1, 2, 5])
def test_router_weights_match_sglang_grouped_topk(tmp_path, provider, packed, normalize, group_topk):
    dtype = np.float16 if packed or provider == "CUDAExecutionProvider" else np.float32
    model = _model(
        tmp_path,
        dtype=ir.DataType.FLOAT16 if dtype == np.float16 else ir.DataType.FLOAT,
        ep="cuda" if packed or provider == "CUDAExecutionProvider" else "cpu",
        use_paged_attention=packed,
        config_options={
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "group_topk": group_topk,
            "norm_topk_prob": normalize,
            "routed_scaling_factor": 2.5,
        },
    )
    token_shape = [4] if packed else [2, 2]
    hidden = np.zeros((*token_shape, model.hidden_size), dtype=dtype)
    hidden.reshape(4, -1)[np.arange(4), np.arange(4)] = 1
    logits = torch.tensor(
        [
            [-3, 1, -2, 0.25, 0.5, -1.5, 3, -0.75],
            [2, -2, 0.5, 1, 0.75, -3, -0.25, 1.5],
            [-1, 3, 2, -4, 2.5, 0, -0.5, -2.5],
            [1, 0.5, 2, 3, -1, -0.5, -2, 3.5],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.1, -0.15, 0.2, -0.05, -0.1, 0.15, -0.2, 0.05])
    gate = torch.zeros(8, model.hidden_size)
    gate[:, :4] = logits.T
    moe = SimpleNamespace(gate=TensorModule(weight=gate), expert_bias=bias)
    model.model.graph.inputs.append(model.make_value("hidden", model.io_dtype, list(hidden.shape)))
    routed, scale = model.make_moe_router(1, moe, "hidden")
    model.make_softmax("/weights/Softmax", routed, model.io_dtype, [4, 8])
    if not normalize and not packed:
        model.make_reshape("/scale/Reshape", [scale, "/model/constants/INT64/[-1, 1]"], model.io_dtype, [4, 1])
        scale = "/scale/Reshape/output_0"
    model.make_mul("/weights/Mul", ["/weights/Softmax/output_0", scale], model.io_dtype, [4, 8])
    model.model.graph.outputs.append(model.values["/weights/Mul/output_0"])

    actual = kernel_session(model, provider, "MoEGroupRouter").run(None, {"hidden": hidden})[0]

    scores = torch.sigmoid(logits).numpy()
    biased = scores + bias.numpy()
    expected = np.zeros((4, 8), dtype=np.float32)
    for row in range(4):
        grouped = biased[row].reshape(2, 4)
        group_scores = np.sort(grouped, axis=-1)[:, -min(group_topk, 4) :].sum(axis=-1)
        group = int(np.argmax(group_scores))
        candidates = np.arange(group * 4, (group + 1) * 4)
        selected = candidates[np.argsort(-biased[row, candidates])[:2]]
        weights = scores[row, selected]
        if normalize:
            weights = weights / weights.sum()
        expected[row, selected] = weights * 2.5
    assert model.moe_attrs["normalize_routing_weights"] is normalize
    np.testing.assert_allclose(actual, expected, rtol=3e-3 if dtype == np.float16 else 1e-5, atol=1e-3)


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
@pytest.mark.parametrize("safe_gate, lower_bound", [(False, None), (False, -0.25), (True, -0.25), (True, -5.0)])
def test_kda_output_and_state_match_recurrent_reference(tmp_path, provider, safe_gate, lower_bound):
    dtype = np.float16 if provider == "CUDAExecutionProvider" else np.float32
    model = _model(
        tmp_path,
        dtype=ir.DataType.FLOAT16 if dtype == np.float16 else ir.DataType.FLOAT,
        ep="cuda" if provider == "CUDAExecutionProvider" else "cpu",
        config_options={"kda_safe_gate": safe_gate, "kda_lower_bound": lower_bound},
    )
    model.make_inputs_and_outputs()
    state_name = model.input_names["past.recurrent"][0]
    state_value = model.values[state_name]
    model.model.graph.inputs.clear()
    model.model.graph.outputs.clear()
    model.model.graph.inputs.append(state_value)
    batch, length, heads, dim = 2, 3, model.linear_num_value_heads, model.linear_key_head_dim
    for name, shape in (
        ("conv", [batch, length, 3 * heads * dim]),
        ("a", [batch, length, heads * dim]),
        ("b", [batch, length, heads]),
    ):
        model.model.graph.inputs.append(model.make_value(name, model.io_dtype, shape))
    a_log = torch.tensor([-0.5, 0.25])
    dt_bias = torch.linspace(-1, 1, heads * dim).reshape(heads, dim)
    attention = SimpleNamespace(A_log=a_log, dt_bias=dt_bias)
    output = model.make_gated_delta_net_layer(0, attention, "conv", "b", "a")
    present = model.output_names["present.recurrent"][0]
    model.model.graph.outputs.extend([model.values[output], model.values[present]])
    rng = np.random.default_rng(7)
    conv = rng.normal(size=(batch, length, 3 * heads * dim)).astype(dtype)
    decay = (
        np.linspace(-8, 8, batch * length * heads * dim, dtype=np.float32)
        .reshape(batch, length, heads * dim)
        .astype(dtype)
    )
    beta = rng.normal(size=(batch, length, heads)).astype(dtype)
    state_dtype = np.float32 if state_value.dtype == ir.DataType.FLOAT else dtype
    initial = rng.normal(scale=0.1, size=(batch, heads, dim, dim)).astype(state_dtype)

    actual, actual_state = kernel_session(model, provider, "GatedDeltaNet").run(
        None, {"conv": conv, "a": decay, "b": beta, state_name: initial}
    )

    q, k, v = [value.reshape(batch, length, heads, dim) for value in np.split(conv.astype(np.float32), 3, axis=-1)]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    raw_gate = decay.astype(np.float32).reshape(batch, length, heads, dim) + dt_bias.numpy()
    if safe_gate:
        gate = lower_bound * torch.sigmoid(torch.from_numpy(raw_gate * np.exp(a_log.numpy())[:, None])).numpy()
    else:
        gate = -np.exp(a_log.numpy())[:, None] * np.logaddexp(0, raw_gate)
    beta = torch.sigmoid(torch.from_numpy(beta.astype(np.float32))).numpy()
    state = initial.astype(np.float32).copy()
    expected = np.empty((batch, length, heads, dim), dtype=np.float32)
    for t in range(length):
        state *= np.exp(gate[:, t, :, None, :])
        delta = v[:, t] - np.einsum("bhvk,bhk->bhv", state, k[:, t])
        state += delta[..., None] * beta[:, t, :, None, None] * k[:, t, :, None, :]
        expected[:, t] = np.einsum("bhvk,bhk->bhv", state, q[:, t]) / np.sqrt(dim)
    tolerance = 5e-3 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(actual, expected.reshape(batch, length, heads * dim), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_state, state, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT])
@pytest.mark.parametrize("ep", ["cpu", "cuda"])
def test_explicit_precision_matches_policy_and_saved_tensor_types(tmp_path, dtype, ep):
    model = _model(tmp_path, dtype=dtype, ep=ep, config_options={"torch_dtype": torch.bfloat16})
    model.model.graph.inputs.append(model.make_value("hidden", dtype, [1, 1, model.hidden_size]))
    model.make_matmul(TensorModule(weight=torch.ones(16, model.hidden_size)), "/projection", "hidden")
    model.model.graph.outputs.append(model.values["/projection/output_0"])
    output = tmp_path / "export"
    output.mkdir()

    model.save_model(str(output))

    expected_policy = {ir.DataType.FLOAT16: "fp16", ir.DataType.BFLOAT16: "bf16", ir.DataType.FLOAT: "fp32"}[dtype]
    assert model.io_dtype == model.onnx_dtype == dtype
    assert model.quant_config.io_dtype == expected_policy
    exported = ir.load(output / "model.onnx")
    assert exported.graph.inputs[0].dtype == exported.graph.outputs[0].dtype == dtype
    weight = next(node for node in exported.graph if node.op_type == "MatMul").inputs[1]
    assert weight.dtype == dtype


@pytest.mark.parametrize("io_dtype", [None, "fp16", "bf16"])
def test_inferred_precision_uses_the_existing_structured_policy(tmp_path, io_dtype):
    quant_data = {"weights": {"type": "int4"}, "moe": {"type": "int4"}}
    if io_dtype is not None:
        quant_data["io_dtype"] = io_dtype
    policy = builder.normalize_builder_config(None, "cuda", target_options={"quant_config": quant_data})
    resolved_io, resolved_weights = policy.extra_options["_quant_config"].to_onnx_dtypes()
    model = _model(
        tmp_path,
        dtype=resolved_io,
        precision=resolved_weights,
        ep="cuda",
        config_options={"torch_dtype": torch.bfloat16},
        **policy.extra_options,
    )
    model.model.graph.inputs.append(model.make_value("hidden", resolved_io, [1, 1, model.hidden_size]))
    model.make_matmul(TensorModule(weight=torch.ones(16, model.hidden_size)), "/projection", "hidden")

    assert model.io_dtype == resolved_io
    assert model.quant_config.io_dtype == (io_dtype or "fp16")
    assert model.values["/projection/output_0"].dtype == resolved_io


@pytest.mark.parametrize("precision", [None, "fp32", "fp16", "bf16"])
def test_bf16_prefill_precision_is_added_by_config_hook(tmp_path, precision, monkeypatch):
    options = {} if precision is None else {"gated_delta_net_bf16_precision": precision}
    model = _model(tmp_path, dtype=ir.DataType.BFLOAT16, ep="cuda", use_paged_attention=True, **options)
    model.make_inputs_and_outputs()
    monkeypatch.setattr(model, "load_generation_config", Mock(side_effect=OSError("offline")))
    monkeypatch.setattr(model, "resolve_special_token_ids", Mock(return_value=(0, 1, 0)))
    open_file = Mock(wraps=builtins.open)
    monkeypatch.setattr(builtins, "open", open_file)

    model.make_genai_config(model.config, {}, str(tmp_path))

    config_path = tmp_path / "genai_config.json"
    writes = [call for call in open_file.call_args_list if call.args[0] == str(config_path)]
    assert len(writes) == 1
    assert writes[0].args[1] == "w"
    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["session_options"]["ep.cuda.gated_delta_net_bf16_precision"] == (
        precision or "bf16"
    )


@pytest.mark.parametrize("precision", [None, True, 0, "", "fast", "FP32", ["bf16"]])
def test_invalid_bf16_prefill_precision_is_rejected(tmp_path, precision):
    with pytest.raises(ValueError, match="must be one of fp32, fp16, bf16"):
        _model(tmp_path, gated_delta_net_bf16_precision=precision)


@pytest.mark.parametrize(
    ("ep", "dtype", "paged"),
    [("cpu", ir.DataType.FLOAT, False), ("cuda", ir.DataType.BFLOAT16, False), ("cuda", ir.DataType.FLOAT16, True)],
)
def test_bf16_prefill_precision_rejects_inapplicable_exports(tmp_path, ep, dtype, paged):
    with pytest.raises(ValueError, match="only to CUDA BF16 paged-attention exports"):
        _model(tmp_path, dtype=dtype, ep=ep, use_paged_attention=paged, gated_delta_net_bf16_precision="bf16")


@pytest.mark.parametrize(
    ("ep", "dtype", "paged"),
    [("cpu", ir.DataType.FLOAT, False), ("cuda", ir.DataType.BFLOAT16, False), ("cuda", ir.DataType.FLOAT16, True)],
)
def test_bf16_prefill_default_does_not_change_other_exports(tmp_path, ep, dtype, paged):
    model = _model(tmp_path, dtype=dtype, ep=ep, use_paged_attention=paged)
    config = {"model": {"decoder": {"session_options": {}}}}

    model.update_genai_config(config)

    assert "ep.cuda.gated_delta_net_bf16_precision" not in config["model"]["decoder"]["session_options"]


@pytest.mark.parametrize(
    ("weight_format", "requantize"), [("bf16", False), ("fp8", False), ("fp8", True), ("int4", True)]
)
def test_loader_exposes_normalized_layers_and_preserves_weight_formats(tmp_path, weight_format, requantize):
    model = _model(
        tmp_path,
        precision=ir.DataType.INT4 if requantize else ir.DataType.FLOAT,
        quant_method="fp8" if weight_format == "fp8" else None,
        config_options={"num_experts": 2},
    )
    _layer_checkpoint(tmp_path, model, weight_format)
    weights = model.load_weights(str(tmp_path))

    modules = list(weights.modules())

    layers = [module for module in modules if model.is_layer(module)]
    assert len(layers) == model.num_layers
    assert all(isinstance(layer, QuantizedDecoderLayer) for layer in layers)
    assert layers[0].self_attn is None
    assert layers[-1].linear_attn is None
    assert layers[0].linear_attn.in_proj_qkv.weight.shape == (3 * model.linear_value_dim, model.hidden_size)
    assert layers[0].linear_attn.in_proj_b.exclude_from_quantization
    assert layers[1].mlp.gate.exclude_from_quantization
    assert weights.handles == {}
    assert weights.handle_keys == {}
    if weight_format == "fp8" and not requantize:
        assert layers[0].linear_attn.in_proj_qkv.quant_type == "fp8"
        assert not weights.lm_head.can_reuse_as_embedding
        experts = layers[1].mlp.experts
        assert isinstance(experts, QuantizedExperts)
        assert experts.gate_up_qweight.dtype == torch.float8_e4m3fn
        assert experts.gate_up_scales.shape == (2, 2, 1)
    else:
        assert layers[0].linear_attn.in_proj_qkv.weight.dtype == torch.bfloat16
        assert layers[1].mlp.experts.gate_up_proj.dtype == torch.bfloat16


def test_streaming_loader_does_not_read_layers_beyond_export_limit(tmp_path):
    model = _model(tmp_path, num_hidden_layers=1, config_options={"num_experts": 2})
    _layer_checkpoint(tmp_path, model)
    weights = model.load_weights(str(tmp_path))

    modules = list(weights.modules())

    assert len(modules) == 4
    assert isinstance(modules[1], QuantizedDecoderLayer)
    assert weights.handles == {}


def test_streaming_loader_closes_after_partial_layer_failure(tmp_path):
    model = _model(tmp_path, quant_method="fp8", config_options={"num_experts": 2})
    tensors = _layer_checkpoint(tmp_path, model, "fp8")
    del tensors["model.layers.1.mlp.experts.1.gate_proj.weight_scale_inv"]
    save_file(tensors, tmp_path / "model.safetensors")
    weights = model.load_weights(str(tmp_path))
    modules = weights.modules()
    next(modules)
    first_layer = next(modules)

    with pytest.raises(ValueError, match="weight_scale_inv.*must have shape"):
        list(modules)

    assert weights.handles == {}
    assert weights.handle_keys == {}
    assert first_layer.linear_attn.in_proj_qkv.weight.shape == (384, 128)


def test_packed_int4_loader_unpacks_signed_nibbles_and_group_scales(tmp_path):
    packed = np.array([[0x76543210, 0xFEDCBA98]], dtype=np.uint32).view(np.int32)
    scales = torch.tensor([[0.5, 2.0]])
    save_file(
        {
            "model.word_embeddings.weight": torch.ones(32, 128),
            "model.norm.weight": torch.ones(128),
            "projection.weight_packed": torch.from_numpy(packed),
            "projection.weight_scale": scales,
        },
        tmp_path / "model.safetensors",
    )
    model = _model(tmp_path)
    weights = BailingModel(str(tmp_path), model.config)
    try:
        projection = weights.make_linear_module("projection")
    finally:
        weights.close()

    expected = (torch.arange(16).reshape(1, 16) - 8) * scales.repeat_interleave(8, dim=1)
    torch.testing.assert_close(projection.weight, expected.to(torch.bfloat16))


@pytest.mark.parametrize("scale_shape", [None, (1,), (2, 2)])
def test_loader_rejects_missing_or_malformed_fp8_block_scales(tmp_path, scale_shape):
    tensors = {
        "model.word_embeddings.weight": torch.ones(32, 128),
        "model.norm.weight": torch.ones(128),
        "projection.weight": torch.ones(16, 128).to(torch.float8_e4m3fn),
    }
    if scale_shape is not None:
        tensors["projection.weight_scale_inv"] = torch.ones(scale_shape)
    save_file(tensors, tmp_path / "model.safetensors")
    weights = _model(tmp_path).load_weights(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="weight_scale_inv.*must have shape"):
            weights.make_linear_module("projection")
    finally:
        weights.close()


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("missing", ["schema", "provider"])
def test_ci_requires_kernel_support_instead_of_skipping(tmp_path, monkeypatch, required, missing):
    monkeypatch.setenv("ORTGENAI_REQUIRE_BAILING_KERNELS", "1" if required else "0")
    schemas = [] if missing == "schema" else [SimpleNamespace(name="MoEGroupRouter", domain="com.microsoft")]
    monkeypatch.setattr(sys.modules[__name__], "get_all_operator_schema", lambda: schemas)
    monkeypatch.setattr(sys.modules[__name__], "prepare_cuda_provider", lambda: None)
    monkeypatch.setattr(ort, "get_available_providers", list)
    exception = pytest.fail.Exception if required else pytest.skip.Exception

    with pytest.raises(exception, match="ORT PR #33817" if missing == "schema" else "is unavailable"):
        kernel_session(_model(tmp_path), "CUDAExecutionProvider", "MoEGroupRouter")


@pytest.mark.parametrize("required", [False, True])
def test_cuda_provider_preparation_failure_is_reported(tmp_path, monkeypatch, required):
    monkeypatch.setenv("ORTGENAI_REQUIRE_BAILING_KERNELS", "1" if required else "0")
    schemas = [SimpleNamespace(name="MoEGroupRouter", domain="com.microsoft")]
    monkeypatch.setattr(sys.modules[__name__], "get_all_operator_schema", lambda: schemas)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda: None, raising=False)
    plugin = SimpleNamespace(get_ep_name=lambda: "CUDAExecutionProvider", get_library_path=lambda: "libcuda.so")
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_cuda", plugin)
    monkeypatch.setattr(ort, "get_ep_devices", list)

    def fail_to_register(name, path):
        raise RuntimeError("libcudart.so.12 not found")

    monkeypatch.setattr(ort, "register_execution_provider_library", fail_to_register)
    exception = pytest.fail.Exception if required else pytest.skip.Exception

    with pytest.raises(exception, match="Preparing the CUDA provider failed: libcudart.so.12 not found"):
        kernel_session(_model(tmp_path), "CUDAExecutionProvider", "MoEGroupRouter")


@pytest.mark.parametrize("has_preload_dlls", [False, True])
def test_cuda_libraries_are_preloaded_before_plugin_registration(monkeypatch, has_preload_dlls):
    calls = []
    monkeypatch.setattr(sys.modules[__name__], "register_cuda_plugin_ep", lambda: calls.append("register"))
    if has_preload_dlls:
        monkeypatch.setattr(ort, "preload_dlls", lambda: calls.append("preload"), raising=False)
    else:
        monkeypatch.delattr(ort, "preload_dlls", raising=False)

    prepare_cuda_provider()

    assert calls == (["preload", "register"] if has_preload_dlls else ["register"])


def test_cuda_plugin_ep_is_registered_once_when_core_wheel_lacks_cuda(monkeypatch):
    registered = []
    plugin = SimpleNamespace(get_ep_name=lambda: "CUDAExecutionProvider", get_library_path=lambda: "libcuda.so")
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_cuda", plugin)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [SimpleNamespace(ep_name=name) for name, _ in registered])
    monkeypatch.setattr(ort, "register_execution_provider_library", lambda name, path: registered.append((name, path)))

    register_cuda_plugin_ep()
    register_cuda_plugin_ep()

    assert registered == [("CUDAExecutionProvider", "libcuda.so")]


def test_cuda_plugin_ep_is_not_registered_when_core_wheel_bundles_cuda(monkeypatch):
    plugin = SimpleNamespace(get_ep_name=lambda: "CUDAExecutionProvider", get_library_path=lambda: "libcuda.so")
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_cuda", plugin)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "register_execution_provider_library", Mock(side_effect=AssertionError))

    register_cuda_plugin_ep()


def test_tokenizer_regex_is_local_and_preserves_supported_patterns(tmp_path, monkeypatch):
    tokenizer = {
        "pre_tokenizer": {
            "pretokenizers": [
                {"pattern": {"Regex": r"\p{N}++"}},
                {"pattern": {"Regex": r"\w+"}},
            ]
        }
    }
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(tokenizer))
    monkeypatch.setattr(bailing.Qwen35TextModel, "save_processing", Mock())

    _model(tmp_path).save_processing("test/bailing", {}, str(tmp_path))

    patterns = json.loads(tokenizer_path.read_text())["pre_tokenizer"]["pretokenizers"]
    assert "++" not in patterns[0]["pattern"]["Regex"]
    assert patterns[1]["pattern"]["Regex"] == r"\w+"


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_mla_query_key_value_helpers_match_projection_reference(tmp_path, packed, interleaved):
    model = _model(
        tmp_path,
        dtype=ir.DataType.FLOAT16 if packed else ir.DataType.FLOAT,
        ep="cuda" if packed else "cpu",
        use_paged_attention=packed,
        config_options={"q_lora_rank": 32, "kv_lora_rank": 32, "rope_interleave": interleaved},
    )
    attention = _mla_weights(model, sparse=True)
    token_shape = (4,) if packed else (2, 2)
    dtype = np.float16 if packed else np.float32
    hidden = (
        np.linspace(-1, 1, np.prod(token_shape) * model.hidden_size, dtype=np.float32)
        .reshape(*token_shape, model.hidden_size)
        .astype(dtype)
    )
    positions = np.array([0, 1, 3, 7], dtype=np.int64).reshape(token_shape)
    model.model.graph.inputs.append(model.make_value("hidden", model.io_dtype, list(hidden.shape)))
    position_name = model.input_names["position_ids"]
    model.model.graph.inputs.append(model.make_value(position_name, ir.DataType.INT64, list(token_shape)))
    query = model.make_mla_query_projection(3, attention, "hidden")
    key, value = model.make_mla_key_value_projection(3, attention, "hidden")
    model.model.graph.outputs.extend(model.values[name] for name in (query, key, value))
    model.model.graph.sort()
    session = ort.InferenceSession(
        ir.serde.serialize_model(model.model).SerializeToString(), providers=["CPUExecutionProvider"]
    )

    actual = session.run(None, {"hidden": hidden, position_name: positions})
    expected = _mla_reference(model, attention, hidden, positions)

    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=5e-3 if packed else 1e-5, atol=2e-3 if packed else 1e-6)


@pytest.mark.parametrize("packed", [False, True])
def test_mla_output_gating_helper_matches_reference(tmp_path, packed):
    model = _model(
        tmp_path,
        dtype=ir.DataType.FLOAT16 if packed else ir.DataType.FLOAT,
        ep="cuda" if packed else "cpu",
        use_paged_attention=packed,
        config_options={"q_lora_rank": 32, "kv_lora_rank": 32},
    )
    attention = _mla_weights(model, sparse=True)
    token_shape = (4,) if packed else (2, 2)
    dtype = np.float16 if packed else np.float32
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(*token_shape, model.hidden_size)).astype(dtype)
    attn_output = rng.normal(size=(*token_shape, model.num_attn_heads * model.head_size)).astype(dtype)
    for name, data in (("hidden", hidden), ("attention", attn_output)):
        model.model.graph.inputs.append(model.make_value(name, model.io_dtype, list(data.shape)))
    model.make_mla_output_projection(3, attention, "hidden", "attention")
    model.model.graph.outputs.append(model.values[model.layernorm_attrs["skip_input"]])
    model.model.graph.sort()
    session = ort.InferenceSession(
        ir.serde.serialize_model(model.model).SerializeToString(), providers=["CPUExecutionProvider"]
    )

    actual = session.run(None, {"hidden": hidden, "attention": attn_output})[0]
    expected = _mla_reference(model, attention, hidden, attn_output=attn_output)

    np.testing.assert_allclose(actual, expected, rtol=5e-3 if packed else 1e-5, atol=2e-3 if packed else 1e-6)


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
@pytest.mark.parametrize("interleaved", [False, True])
def test_full_mla_attention_matches_causal_reference(tmp_path, provider, interleaved):
    cuda = provider == "CUDAExecutionProvider"
    model = _model(
        tmp_path,
        dtype=ir.DataType.FLOAT16 if cuda else ir.DataType.FLOAT,
        ep="cuda" if cuda else "cpu",
        config_options={"q_lora_rank": 32, "kv_lora_rank": 32, "rope_interleave": interleaved},
    )
    attention = _mla_weights(model)
    model.make_inputs_and_outputs()
    model.model.graph.inputs.clear()
    model.model.graph.outputs.clear()
    batch, length = 2, 3
    dtype = np.float16 if cuda else np.float32
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(batch, length, model.hidden_size)).astype(dtype)
    positions = np.tile(np.arange(length, dtype=np.int64), (batch, 1))
    past_k, past_v, present_k, present_v = model.make_key_value_cache_names(3)
    position_name = model.input_names["position_ids"]
    model.model.graph.inputs.append(model.make_value("hidden", model.io_dtype, list(hidden.shape)))
    model.model.graph.inputs.extend(model.values[name] for name in (position_name, past_k, past_v))
    model.mask_attrs["seqlens_k"] = "/seqlens"
    model.mask_attrs["total_seq_len"] = "/total"
    model.model.graph.inputs.extend(
        [
            model.make_value("/seqlens/output_0", ir.DataType.INT32, [batch]),
            model.make_value("/total/output_0", ir.DataType.INT32, [1]),
        ]
    )
    model.make_attention(3, attention, "hidden")
    output = model.layernorm_attrs["skip_input"]
    model.model.graph.outputs.extend(model.values[name] for name in (output, present_k, present_v))
    inputs = {
        "hidden": hidden,
        position_name: positions,
        past_k: np.empty((batch, model.num_attn_heads, 0, model.head_size), dtype=dtype),
        past_v: np.empty((batch, model.num_attn_heads, 0, model.head_size), dtype=dtype),
        "/seqlens/output_0": np.full(batch, length - 1, dtype=np.int32),
        "/total/output_0": np.array([length], dtype=np.int32),
    }

    actual, actual_key, actual_value = kernel_session(model, provider, "GroupQueryAttention").run(None, inputs)

    query, key, value = _mla_reference(model, attention, hidden, positions)
    tensors = [
        torch.from_numpy(data.astype(np.float32))
        .reshape(batch, length, model.num_attn_heads, model.head_size)
        .transpose(1, 2)
        for data in (query, key, value)
    ]
    query, key, value = tensors
    scores = query @ key.transpose(-1, -2) * model.attention_attrs["scale"]
    scores.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), float("-inf"))
    attended = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, length, -1).numpy().astype(dtype)
    expected = _mla_reference(model, attention, hidden, attn_output=attended)
    tolerance = 5e-3 if cuda else 1e-5
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_key, key.numpy(), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_value, value.numpy(), rtol=tolerance, atol=tolerance)


def test_bailing_prefill_precision_cli_help_is_renderable(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["builder", "--help"])

    with pytest.raises(SystemExit) as result:
        builder.get_args()

    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "gated_delta_net_bf16_precision" in help_text
    assert "gated_delta_net_bf16_precision=fp32" in help_text
    assert "N % 64" in help_text
