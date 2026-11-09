# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Small, offline regressions for Bailing checkpoint loading and graph export."""

import sys
from pathlib import Path
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
from builders.bailing import BailingMoeV3Model, _BailingWeights


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
    monkeypatch.setattr(bailing, "snapshot_download", download)

    with pytest.raises(ValueError, match="does not support adapter_path"):
        model.load_weights("")

    download.assert_not_called()


@pytest.mark.parametrize("remote", [False, True])
def test_streaming_loader_resolves_checkpoint_source(tmp_path, monkeypatch, remote):
    _checkpoint(tmp_path)
    model = _model(tmp_path, hf_token=False, base_revision="test-revision")
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(bailing, "snapshot_download", download)

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
        projection = weights.make_linear("projection")
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
        projection = weights.make_linear("projection")
        fused = _BailingWeights.concat_linears([projection, projection])
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
    assert model.kda_lower_bound is None


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
    heads, rotary_dim = 2, model.qk_rope_head_dim
    token_shape = [4] if packed else [2, 2]
    shape = [*token_shape, heads * rotary_dim]
    model.model.graph.inputs.append(model.make_value("input", model.io_dtype, shape))
    positions = np.array([0, 1, 3, 7], dtype=np.int64).reshape(token_shape)
    model.model.graph.inputs.append(model.make_value(model.input_names["position_ids"], ir.DataType.INT64, token_shape))
    output = model._make_mla_rope("/rope", "input", heads, token_shape)
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


def kernel_session(model, provider, required_op):
    schemas = {schema.name: schema for schema in get_all_operator_schema() if schema.domain == "com.microsoft"}
    if required_op not in schemas or (
        required_op == "GatedDeltaNet" and "gate_lower_bound" not in schemas[required_op].attributes
    ):
        pytest.skip(f"Requires an ONNX Runtime build with {required_op} from ORT PR #33817")
    if provider not in ort.get_available_providers():
        pytest.skip(f"{provider} is unavailable")
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
    moe = bailing.SimpleNamespace(gate=bailing.TensorModule(weight=gate), expert_bias=bias)
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
    attention = bailing.SimpleNamespace(A_log=a_log, dt_bias=dt_bias)
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
