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
from loaders.base import QuantizedExperts, TensorModule
from loaders.modelopt import ModeloptModel
from loaders.qwen import QwenMTPModel
from quantization import QuantConfig
from safetensors.torch import save_file

from models.builders.base import Model
from models.builders.qwen import Qwen35DenseMTPModel, Qwen35Model, Qwen35MoEModel


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


def test_dense_config_only_composite_does_not_require_decoder_token_ids(monkeypatch):
    monkeypatch.setitem(Qwen35Model.__init__.__globals__, "Qwen35TextModel", FakeComponent)

    model = Qwen35Model(
        SimpleNamespace(),
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT16,
        "cpu",
        None,
        {"config_only": True},
    )

    assert model.decoder.extra_options["config_only"] is True


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


def test_inherited_target_config_drops_overrides_and_preserves_native_mtp():
    target = QuantConfig.from_dict(
        {
            "io_dtype": "bf16",
            "checkpoint_policy": "requantize",
            "weights": {"type": "int8", "overrides": [{"match": {"name": "/lm_head/MatMul"}, "exclude": True}]},
        }
    )
    model = _resolve({"quant_config": target})
    inherited = model.mtp_attrs["extra_options"]["quant_config"]
    assert inherited.checkpoint_policy == "preserve"
    assert inherited.weights.type == "int8"
    assert inherited.weights.overrides == []
    assert target.weights.overrides
    assert model.mtp_attrs["io_dtype"] == ir.DataType.BFLOAT16


def test_mtp_explicit_checkpoint_policy_overrides_legacy_default():
    model = _resolve(
        {"mtp_quant_config": {"checkpoint_policy": "preserve", "weights": {"type": "none"}, "moe": {"type": "none"}}}
    )
    assert model.mtp_attrs["extra_options"]["_quant_config"].checkpoint_policy == "preserve"
    legacy = _resolve({"mtp_quant_config": {"weights": {"type": "int4"}}})
    assert legacy.mtp_attrs["extra_options"]["_quant_config"].checkpoint_policy == "requantize"


@pytest.mark.parametrize("policy", ["preserve", "requantize"])
def test_native_loader_applies_target_checkpoint_policy(policy):
    loader = object.__new__(ModeloptModel)
    loader.quant_type = "modelopt"
    loader.quant_attrs = {
        "export_config": QuantConfig.from_dict({"checkpoint_policy": policy}),
        "checkpoint_scope": "target",
    }
    module = TensorModule(weight=torch.ones((2, 4), dtype=torch.float8_e4m3fn))
    module.quant_type = "fp8"
    module.weight_scale = torch.tensor(0.5)
    result = loader.apply_checkpoint_policy(module, "model.language_model.layers.0.self_attn.q_proj")
    if policy == "preserve":
        assert result is module
    else:
        assert result.quant_type == "none"
        torch.testing.assert_close(result.weight, torch.full((2, 4), 0.5, dtype=torch.bfloat16))
    assert loader.apply_checkpoint_policy(module, "mtp.layers.0.self_attn.q_proj") is module


def test_native_loader_rejects_explicit_conflicting_format():
    loader = object.__new__(ModeloptModel)
    loader.quant_attrs = {"export_config": QuantConfig.from_dict({"weights": {"type": "int8"}})}
    module = TensorModule()
    module.quant_type = "fp8"
    with pytest.raises(ValueError, match="checkpoint_policy=preserve"):
        loader.apply_checkpoint_policy(module, "lm_head")


def test_native_experts_requantize_to_dense_loader_representation():
    loader = object.__new__(ModeloptModel)
    loader.quant_type = "modelopt"
    loader.num_experts = 1
    loader.quant_attrs = {
        "export_config": QuantConfig.from_dict({"checkpoint_policy": "requantize", "moe": {"type": "int4"}})
    }
    prefix = "model.language_model.layers.0"
    tensors = {
        f"{prefix}.self_attn.{projection}.weight": torch.ones((16, 16), dtype=torch.bfloat16)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    tensors[f"{prefix}.mlp.gate.weight"] = torch.ones((1, 16))
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"{prefix}.mlp.experts.0.{projection}"
        tensors[f"{name}.weight"] = torch.full((16, 8), 0x22, dtype=torch.uint8)
        tensors[f"{name}.weight_scale"] = torch.ones((16, 1), dtype=torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale_2"] = torch.tensor(0.5)
    loader.get_tensor = tensors.get
    layer = loader.make_layer(0)
    torch.testing.assert_close(layer.mlp.experts.gate_up_proj, torch.full((1, 32, 16), 0.5, dtype=torch.bfloat16))
    torch.testing.assert_close(layer.mlp.experts.down_proj, torch.full((1, 16, 16), 0.5, dtype=torch.bfloat16))


def test_inherited_mtp_preserves_native_expert_format_over_integer_default():
    model = Model.__new__(Model)
    model.quant_config = QuantConfig.from_dict({})
    model.moe_attrs = {"op_type": "QMoE", "quant_type": "int", "expert_weight_bits": 4}
    model.io_dtype = ir.DataType.BFLOAT16
    model.make_initializer = lambda *args, **kwargs: None
    experts = QuantizedExperts()
    experts.quant_type = "nvfp4"
    experts.block_size = 16
    model.make_moe_expert_initializers(0, experts)
    assert model.moe_attrs["quant_type"] == "nvfp4"
    assert model.moe_attrs["block_size"] == 16


@pytest.mark.parametrize(
    "settings,conflicts",
    [
        ({}, False),
        ({"block_size": 16}, False),
        ({"block_size": 32}, True),
        ({"weights_prepacked": -1}, False),
        ({"weights_prepacked": 1}, False),
        ({"weights_prepacked": 0}, True),
    ],
)
def test_native_experts_validate_explicit_preserved_layout(settings, conflicts):
    model = Model.__new__(Model)
    model.quant_config = QuantConfig.from_dict({"moe": settings})
    model.moe_attrs = {"op_type": "QMoE", "quant_type": "int", "expert_weight_bits": 4}
    model.io_dtype = ir.DataType.BFLOAT16
    model.make_initializer = lambda *args, **kwargs: None
    experts = QuantizedExperts()
    experts.quant_type = "nvfp4"
    experts.block_size = 16
    experts.weights_prepacked = 1

    if conflicts:
        with pytest.raises(ValueError, match="checkpoint_policy=preserve cannot apply moe"):
            model.make_moe_expert_initializers(0, experts)
    else:
        model.make_moe_expert_initializers(0, experts)
        assert model.moe_attrs["block_size"] == 16
        assert model.moe_attrs["weights_prepacked"] == 1


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


@pytest.mark.parametrize("shared_embeddings", [False, True])
def test_mtp_quant_config_preserves_shared_embeddings_option(shared_embeddings):
    model = _resolve(
        {
            "shared_embeddings": shared_embeddings,
            "mtp_quant_config": '{"weights": {"type": "int4"}, "moe": {"type": "none"}}',
        }
    )

    assert model.mtp_attrs["extra_options"]["shared_embeddings"] is shared_embeddings


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
    lm_head = SimpleNamespace(weight=torch.ones((2, 2)), weight_scale_2=torch.tensor(0.5))
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
        lm_head=SimpleNamespace(weight=torch.ones((2, 2))),
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


@pytest.mark.parametrize("preserve_quantization", [False, True])
def test_modelopt_mtp_loader_uses_tied_embedding_when_lm_head_is_absent(monkeypatch, preserve_quantization):
    embedding = SimpleNamespace(weight=torch.ones((4, 2), dtype=torch.bfloat16))
    parsed = SimpleNamespace(
        embedding=embedding,
        lm_head=SimpleNamespace(weight=None),
        mtp=SimpleNamespace(
            state={},
            fc=SimpleNamespace(),
            pre_fc_norm_embedding=SimpleNamespace(),
            pre_fc_norm_hidden=SimpleNamespace(),
            norm=SimpleNamespace(),
            layers=[SimpleNamespace()],
        ),
        dequantize_state=lambda state: state,
        dequantize_tensor=lambda *args: pytest.fail("A missing tied LM head must not be dequantized"),
    )
    captured = {}

    def capture_from_state(cls, mtp_state, embed_weight, lm_head_weight, layer_config, is_moe):
        captured["embed_weight"] = embed_weight
        captured["lm_head_weight"] = lm_head_weight
        return SimpleNamespace()

    monkeypatch.setattr(QwenMTPModel, "from_state", classmethod(capture_from_state))
    mtp = QwenMTPModel.from_modelopt(
        parsed,
        layer_config=SimpleNamespace(tie_word_embeddings=True),
        preserve_quantization=preserve_quantization,
        is_moe=False,
    )

    if preserve_quantization:
        assert mtp.lm_head is embedding
    else:
        assert captured["embed_weight"] is embedding.weight
        assert captured["lm_head_weight"] is embedding.weight


def test_modelopt_mtp_loader_rejects_missing_untied_lm_head():
    parsed = SimpleNamespace(
        embedding=SimpleNamespace(weight=torch.ones((4, 2))),
        lm_head=SimpleNamespace(weight=None),
        mtp=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="does not tie word embeddings"):
        QwenMTPModel.from_modelopt(
            parsed,
            layer_config=SimpleNamespace(tie_word_embeddings=False),
            preserve_quantization=True,
            is_moe=False,
        )


def test_modelopt_loader_materializes_tied_lm_head_for_main_decoder(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 0,
                    "num_experts": 0,
                    "tie_word_embeddings": True,
                }
            }
        )
    )
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.ones((4, 2), dtype=torch.bfloat16),
            "model.language_model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )

    model = ModeloptModel(
        "compressed-tensors",
        str(tmp_path),
        quant_attrs={},
        q_size=2,
        kv_size=2,
        intermediate_size=4,
        num_layers=0,
    )

    assert model.lm_head is not model.embedding
    assert model.lm_head.weight is model.embedding.weight
    assert model.modules()[0] is model.embedding
    assert model.modules()[-1] is model.lm_head


def test_remote_mtp_loader_resolves_hugging_face_snapshot(monkeypatch, tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    calls = {}

    huggingface_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, cache_dir, token):
        calls.update(repo_id=repo_id, cache_dir=cache_dir, token=token)
        return str(snapshot_dir)

    huggingface_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    def capture_from_safetensors(cls, model_dir, layer_config, is_moe):
        calls.update(model_dir=model_dir, layer_config=layer_config, is_moe=is_moe)
        return "mtp-weights"

    monkeypatch.setattr(QwenMTPModel, "from_safetensors", classmethod(capture_from_safetensors))

    result = QwenMTPModel.from_pretrained(
        None,
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-2B",
        layer_config="config",
        is_moe=False,
        cache_dir="model-cache",
        token="hf-token",
    )

    assert result == "mtp-weights"
    assert calls == {
        "repo_id": "Qwen/Qwen3.5-2B",
        "cache_dir": "model-cache",
        "token": "hf-token",
        "model_dir": str(snapshot_dir),
        "layer_config": "config",
        "is_moe": False,
    }


def test_local_mtp_loader_does_not_resolve_hugging_face_snapshot(monkeypatch, tmp_path):
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = lambda *args, **kwargs: pytest.fail(
        "snapshot_download must not be called for a local model"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    monkeypatch.setattr(QwenMTPModel, "from_safetensors", classmethod(lambda cls, *args: args[0]))

    assert (
        QwenMTPModel.from_pretrained(
            None,
            tmp_path,
            tmp_path,
            layer_config="config",
            cache_dir="unused",
            token="unused",
        )
        == tmp_path
    )


def test_safetensors_mtp_loader_uses_keys_api(monkeypatch, tmp_path):
    import safetensors.torch as safetensors_torch  # noqa: PLC0415

    tensors = {
        "model.embed_tokens.weight": torch.ones((4, 2)),
        "lm_head.weight": torch.ones((4, 2)),
        "mtp.fc.weight": torch.ones((2, 4)),
    }

    class KeysOnlySafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            return False

        def keys(self):
            return tensors.keys()

        def get_tensor(self, key):
            return tensors[key]

    def open_safetensors(shard, framework):
        assert shard == str(tmp_path / "model.safetensors")
        assert framework == "pt"
        return KeysOnlySafeOpen()

    captured = {}

    def capture_from_state(cls, mtp_state, embed_weight, lm_head_weight, layer_config, is_moe):
        captured["mtp_state"] = mtp_state
        captured["embed_weight"] = embed_weight
        captured["lm_head_weight"] = lm_head_weight
        captured["layer_config"] = layer_config
        captured["is_moe"] = is_moe
        return SimpleNamespace()

    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(safetensors_torch, "safe_open", open_safetensors)
    monkeypatch.setattr(QwenMTPModel, "from_state", classmethod(capture_from_state))

    QwenMTPModel.from_safetensors(tmp_path, layer_config="config", is_moe=False)

    assert captured["mtp_state"]["mtp.fc.weight"] is tensors["mtp.fc.weight"]
    assert captured["embed_weight"] is tensors["model.embed_tokens.weight"]
    assert captured["lm_head_weight"] is tensors["lm_head.weight"]
    assert captured["layer_config"] == "config"
    assert captured["is_moe"] is False


def test_safetensors_mtp_loader_uses_tied_embedding_when_lm_head_is_absent(monkeypatch, tmp_path):
    import safetensors.torch as safetensors_torch  # noqa: PLC0415

    tensors = {
        "model.embed_tokens.weight": torch.ones((4, 2)),
        "mtp.fc.weight": torch.ones((2, 4)),
    }

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            return False

        def keys(self):
            return tensors.keys()

        def get_tensor(self, key):
            return tensors[key]

    captured = {}

    def capture_from_state(cls, mtp_state, embed_weight, lm_head_weight, layer_config, is_moe):
        captured["embed_weight"] = embed_weight
        captured["lm_head_weight"] = lm_head_weight
        return SimpleNamespace()

    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(safetensors_torch, "safe_open", lambda *args, **kwargs: FakeSafeOpen())
    monkeypatch.setattr(QwenMTPModel, "from_state", classmethod(capture_from_state))

    QwenMTPModel.from_safetensors(
        tmp_path,
        layer_config=SimpleNamespace(tie_word_embeddings=True),
        is_moe=False,
    )

    assert captured["embed_weight"] is tensors["model.embed_tokens.weight"]
    assert captured["lm_head_weight"] is tensors["model.embed_tokens.weight"]


def _make_minimal_mtp_embedding_model(*, tied_quantized=False, tied_unquantized=False, can_reuse_lm_head=True):
    model = Qwen35DenseMTPModel.__new__(Qwen35DenseMTPModel)
    model.tied_quantized_embeddings = tied_quantized
    model.tied_unquantized_embeddings = tied_unquantized
    model.hidden_size = 64
    model.vocab_size = 32000
    model.io_dtype = ir.DataType.FLOAT16
    model.input_names = {"input_ids": "input_ids"}
    model.quant_attrs = {"matmul_block_size": 32}
    model.mtp_weights = SimpleNamespace(
        embedding=SimpleNamespace(weight=object()),
        lm_head=SimpleNamespace(can_reuse_as_embedding=can_reuse_lm_head),
    )
    model._initializer_calls = []
    model._reshape_calls = []
    model._transpose_calls = []
    model._node_calls = []

    def make_initializer(tensor, name, to=None):
        model._initializer_calls.append((tensor, name, to))

    def make_reshape(name, inputs, dtype, shape):
        model._reshape_calls.append((name, inputs, dtype, shape))

    def make_transpose(name, root_input, dtype, shape, perm):
        model._transpose_calls.append((name, root_input, dtype, shape, perm))

    def make_node(op_type, inputs, outputs, name, **kwargs):
        model._node_calls.append((op_type, inputs, outputs, name, kwargs))

    model.make_initializer = make_initializer
    model.make_reshape = make_reshape
    model.make_transpose = make_transpose
    model.make_node = make_node
    model.make_tied_quantized_embedding_input_names = lambda: (
        8,
        "lm_head.MatMul.weight_Q8G32",
        "lm_head.MatMul.weight_scale",
        None,
    )
    return model


def test_mtp_quantized_shared_embedding_reuses_lm_head_initializers():
    model = _make_minimal_mtp_embedding_model(tied_quantized=True)

    output = model.make_mtp_embedding("/model/mtp")

    assert output == "/model/mtp/embed_tokens/GatherBlockQuantized/output_0"
    assert model._initializer_calls == []
    assert model._reshape_calls[0][1][0] == "lm_head.MatMul.weight_Q8G32"
    op_type, inputs, _, _, attributes = model._node_calls[0]
    assert op_type == "GatherBlockQuantized"
    assert inputs == [
        "/model/mtp/embed_tokens/Reshape/output_0",
        "input_ids",
        "lm_head.MatMul.weight_scale",
    ]
    assert attributes["bits"] == 8


def test_mtp_unquantized_shared_embedding_reuses_lm_head_initializer():
    model = _make_minimal_mtp_embedding_model(tied_unquantized=True)

    output = model.make_mtp_embedding("/model/mtp")

    assert output == "/model/mtp/embed_tokens/Gather/output_0"
    assert model._initializer_calls == []
    assert model._transpose_calls[0][1] == "lm_head.MatMul.weight"
    assert model._node_calls[0][1][0] == "/model/mtp/embed_tokens/Transpose/output_0"


def test_mtp_unshared_embedding_keeps_separate_initializer():
    model = _make_minimal_mtp_embedding_model()

    output = model.make_mtp_embedding("/model/mtp")

    assert output == "/model/mtp/embed_tokens/Gather/output_0"
    assert len(model._initializer_calls) == 1
    assert model._initializer_calls[0][1] == "model.embed_tokens.weight"
    assert model._reshape_calls == []
    assert model._transpose_calls == []


@pytest.mark.parametrize("tied_quantized, tied_unquantized", [(True, False), (False, True)])
def test_mtp_incompatible_lm_head_keeps_separate_embedding(tied_quantized, tied_unquantized):
    model = _make_minimal_mtp_embedding_model(
        tied_quantized=tied_quantized,
        tied_unquantized=tied_unquantized,
        can_reuse_lm_head=False,
    )

    output = model.make_mtp_embedding("/model/mtp")

    assert output == "/model/mtp/embed_tokens/Gather/output_0"
    assert len(model._initializer_calls) == 1
    assert model._initializer_calls[0][1] == "model.embed_tokens.weight"
    assert model._reshape_calls == []
    assert model._transpose_calls == []


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
