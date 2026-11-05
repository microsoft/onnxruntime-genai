# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Small, offline regressions for Bailing checkpoint loading and graph export."""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from safetensors.torch import save_file
from transformers import PretrainedConfig

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

import builder
from builders import bailing
from builders.bailing import BailingMoeV3Model, _BailingWeights


def _model(tmp_path, *, dtype=ir.DataType.FLOAT, precision=None, ep="cpu", quant_method=None, **options):
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
