# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""The MTP model resolves quantization independently from the main model."""

import json
import sys
import types
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch
from loaders.qwen import QwenMTPModel

from models.builders.qwen import Qwen35Model, Qwen35MoEModel


def _resolve(extra_options, main_onnx_dtype=ir.DataType.INT4):
    """Run the MTP model config resolution on a bare stub of the builder."""
    model = object.__new__(Qwen35MoEModel)
    model.mtp_attrs = {
        "onnx_dtype": main_onnx_dtype,
        "io_dtype": ir.DataType.FLOAT16,
        "extra_options": dict(extra_options),
        "ep": "cuda",
    }
    model.resolve_mtp_model_config(extra_options)
    return model


class FakeComponent:
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config
        self.extra_options = extra_options
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = 32
        self.hf_token = True
        self.hf_remote = False
        self.context_length = 128
        self.exclude_embeds = False
        self.model_type = "qwen3_5_moe"


def test_composite_without_mtp_creates_only_decoder(monkeypatch):
    monkeypatch.setitem(Qwen35MoEModel.__init__.__globals__, "Qwen35MoETextModel", FakeComponent)
    monkeypatch.setitem(Qwen35MoEModel.__init__.__globals__, "Qwen35MTPModel", FakeComponent)

    model = Qwen35MoEModel(SimpleNamespace(), ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cpu", None, {})

    assert isinstance(model.decoder, FakeComponent)
    assert model.mtp is None
    assert "include_hidden_states" not in model.decoder.extra_options


def test_composite_with_mtp_creates_separate_components(monkeypatch):
    monkeypatch.setitem(Qwen35MoEModel.__init__.__globals__, "Qwen35MoETextModel", FakeComponent)
    monkeypatch.setitem(Qwen35MoEModel.__init__.__globals__, "Qwen35MTPModel", FakeComponent)
    config = SimpleNamespace(mtp_num_hidden_layers=1)

    model = Qwen35MoEModel(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cpu", None, {})

    assert isinstance(model.decoder, FakeComponent)
    assert isinstance(model.mtp, FakeComponent)
    assert model.decoder is not model.mtp
    assert model.decoder.extra_options["include_hidden_states"] is True
    assert "include_hidden_states" not in model.mtp.extra_options
    assert model.mtp.extra_options["filename"] == "mtp.onnx"


def test_dense_composite_with_mtp_uses_dense_components(monkeypatch):
    monkeypatch.setitem(Qwen35Model.__init__.__globals__, "Qwen35TextModel", FakeComponent)
    monkeypatch.setitem(Qwen35Model.__init__.__globals__, "Qwen35DenseMTPModel", FakeComponent)
    config = SimpleNamespace(mtp_num_hidden_layers=1)

    model = Qwen35Model(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cpu", None, {})

    assert isinstance(model.decoder, FakeComponent)
    assert isinstance(model.mtp, FakeComponent)
    assert model.decoder.extra_options["include_hidden_states"] is True
    assert model.mtp.extra_options["filename"] == "mtp.onnx"


def test_declared_mtp_layers_include_hidden_states():
    original = {}
    model = object.__new__(Qwen35MoEModel)
    config = SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=1))

    options = model.make_mtp_init(config, original)

    assert options["include_hidden_states"] is True
    assert original == {}
    assert model.mtp_attrs["build"] is True


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(mtp_num_hidden_layers=0),
        SimpleNamespace(mtp_num_hidden_layers=None),
        SimpleNamespace(text_config=SimpleNamespace()),
        SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=0)),
        SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=None)),
    ],
)
def test_model_without_mtp_layers_keeps_original_options(config):
    model = object.__new__(Qwen35MoEModel)
    original = {}

    options = model.make_mtp_init(config, original)

    assert options == original
    assert options is not original
    assert model.mtp_attrs["build"] is False
    assert "include_hidden_states" not in options


def test_top_level_mtp_declaration_is_used_when_text_config_omits_it():
    model = object.__new__(Qwen35MoEModel)
    config = SimpleNamespace(text_config=SimpleNamespace(), mtp_num_hidden_layers=1)

    options = model.make_mtp_init(config, {})

    assert options["include_hidden_states"] is True
    assert model.mtp_attrs["build"] is True


@pytest.mark.parametrize("option", ["exclude_lm_head", "prune_lm_head"])
def test_mtp_export_rejects_incompatible_lm_head_options(option):
    model = object.__new__(Qwen35MoEModel)
    config = SimpleNamespace(mtp_num_hidden_layers=1)

    with pytest.raises(ValueError, match=option):
        model.make_mtp_init(config, {option: True})


def test_mtp_drops_main_model_kv_scales_without_mtp_section(tmp_path):
    scales = tmp_path / "kv_scales.json"
    scales.write_text(json.dumps({"scales": {"k_scales": [1.0], "v_scales": [1.0]}}))
    options = {"kv_cache_quant_scheme": "fp8_per_tensor", "kv_cache_scale_file": str(scales)}

    Qwen35MoEModel.drop_unusable_mtp_kv_scales(object(), options)

    assert "kv_cache_quant_scheme" not in options
    assert "kv_cache_scale_file" not in options


def test_mtp_keeps_explicit_mtp_kv_scales(tmp_path):
    scales = tmp_path / "kv_scales.json"
    scales.write_text(json.dumps({"mtp": {"scales": {"k_scales": [1.0], "v_scales": [1.0]}}}))
    options = {"kv_cache_quant_scheme": "fp8_per_tensor", "kv_cache_scale_file": str(scales)}

    Qwen35MoEModel.drop_unusable_mtp_kv_scales(object(), options)

    assert options["kv_cache_quant_scheme"] == "fp8_per_tensor"
    assert options["kv_cache_scale_file"] == str(scales)


def test_no_mtp_config_inherits_main_model_settings():
    options = {"moe_quant_type": "nvfp4", "block_size": 64}
    model = _resolve(options)

    assert model.mtp_attrs["onnx_dtype"] == ir.DataType.INT4
    assert model.mtp_attrs["extra_options"] == options


def test_mtp_quant_config_json_configures_targets_independently():
    model = _resolve(
        {
            "mtp_quant_config": json.dumps(
                {
                    "io_dtype": "bf16",
                    "weights": {"type": "int4", "symmetric": False},
                    "moe": {"type": "none"},
                }
            )
        },
        main_onnx_dtype=ir.DataType.INT8,
    )

    assert model.mtp_attrs["io_dtype"] == ir.DataType.BFLOAT16
    assert model.mtp_attrs["onnx_dtype"] == ir.DataType.UINT4
    quant_config = model.mtp_attrs["extra_options"]["_quant_config"]
    assert quant_config.weights.type == "int4"
    assert quant_config.moe.type == "none"


def test_mtp_quant_config_can_keep_the_entire_head_fp16():
    model = _resolve(
        {
            "mtp_quant_config": json.dumps(
                {
                    "io_dtype": "fp16",
                    "weights": {"type": "none"},
                    "moe": {"type": "none"},
                }
            )
        }
    )

    assert model.mtp_attrs["io_dtype"] == ir.DataType.FLOAT16
    assert model.mtp_attrs["onnx_dtype"] == ir.DataType.FLOAT16
    quant_config = model.mtp_attrs["extra_options"]["_quant_config"]
    assert quant_config.weights.type == "none"
    assert quant_config.moe.type == "none"


def test_mtp_dense_fp4_requires_a_supported_dense_format():
    with pytest.raises(ValueError, match=r"select mxfp4/nvfp4 independently through moe\.type"):
        _resolve({"mtp_quant_config": '{"weights": {"type": "nvfp4", "block_size": 16}}'})


def test_modelopt_mtp_loader_consumes_parsed_modules():
    layer = SimpleNamespace()
    embedding = torch.ones((2, 2), dtype=torch.bfloat16)
    embedding_module = SimpleNamespace(weight=embedding)
    lm_head = SimpleNamespace(weight_scale_2=torch.tensor(0.5))
    fc = SimpleNamespace(weight_scale_2=torch.tensor(0.25))
    parsed = SimpleNamespace(
        embedding=embedding_module,
        lm_head=lm_head,
        mtp=SimpleNamespace(
            fc=fc,
            pre_fc_norm_embedding=SimpleNamespace(weight=torch.ones(2)),
            pre_fc_norm_hidden=SimpleNamespace(weight=torch.ones(2)),
            norm=SimpleNamespace(weight=torch.ones(2)),
            layers=[layer],
        ),
    )
    mtp = QwenMTPModel.from_modelopt(parsed, layer_config=None, preserve_quantization=True)

    assert mtp.embedding is embedding_module
    assert mtp.lm_head is lm_head
    assert mtp.fc is fc
    assert mtp.layers == [layer]


def test_compressed_tensors_mtp_loader_consumes_parsed_modules():
    layer = SimpleNamespace()
    parsed = SimpleNamespace(
        embedding=SimpleNamespace(weight=torch.ones((2, 2), dtype=torch.bfloat16)),
        lm_head=SimpleNamespace(),
        mtp=SimpleNamespace(
            fc=SimpleNamespace(),
            pre_fc_norm_embedding=SimpleNamespace(),
            pre_fc_norm_hidden=SimpleNamespace(),
            norm=SimpleNamespace(),
            layers=[layer],
        ),
    )

    mtp = QwenMTPModel.from_pretrained(
        "compressed-tensors",
        "checkpoint",
        "checkpoint",
        layer_config=None,
        preserve_quantization=True,
        load_quantized_model=lambda _: parsed,
        is_moe=False,
    )

    assert mtp.embedding is parsed.embedding
    assert mtp.layers == [layer]


def test_dense_mtp_state_uses_dense_decoder_layer(monkeypatch):
    class FakeDenseDecoderLayer:
        def __init__(self, config, layer_idx):
            self.config = config
            self.layer_idx = layer_idx

        def load_state_dict(self, state, strict):
            self.state = state
            return [], []

        def eval(self):
            return self

    module_name = "transformers.models.qwen3_5.modeling_qwen3_5"
    modeling_module = types.ModuleType(module_name)
    modeling_module.Qwen3_5DecoderLayer = FakeDenseDecoderLayer
    monkeypatch.setitem(sys.modules, module_name, modeling_module)
    mtp_state = {
        "mtp.fc.weight": torch.ones((2, 4)),
        "mtp.pre_fc_norm_embedding.weight": torch.ones(2),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(2),
        "mtp.norm.weight": torch.ones(2),
        "mtp.layers.0.marker": torch.tensor(1.0),
    }

    mtp = QwenMTPModel.from_state(
        mtp_state,
        torch.ones((4, 2)),
        torch.ones((4, 2)),
        layer_config=SimpleNamespace(),
        is_moe=False,
    )

    assert isinstance(mtp.layers[0], FakeDenseDecoderLayer)
    assert mtp.layers[0].state == {"marker": mtp_state["mtp.layers.0.marker"]}
