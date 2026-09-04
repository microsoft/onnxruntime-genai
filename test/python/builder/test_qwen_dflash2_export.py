# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib
import json
import os
import types

import onnx_ir as ir
import pytest
import torch

from models.builders.dflash2 import DFlash2Builder
from models.builders.mtp import MTPModel
from models.builders.qwen import Qwen35MoEModel

TARGET_LAYER_IDS = [1, 11, 21]


def _draft_checkpoint(tmp_path, target_layer_ids=TARGET_LAYER_IDS):
    draft_dir = tmp_path / "dflash2_draft"
    draft_dir.mkdir()
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "rope_parameters": {"rope_theta": 10000.0},
        "dflash_config": {
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 2,
            "mask_token_id": 31,
            "target_layer_ids": target_layer_ids,
            "block_size": 5,
        },
    }
    (draft_dir / "config.json").write_text(json.dumps(config))
    return str(draft_dir)


def _composite(aux_layers=TARGET_LAYER_IDS, use_paged_attention=True):
    model = object.__new__(Qwen35MoEModel)
    model.dflash2 = None
    model.dflash2_shared_initializers = []
    model.decoder = types.SimpleNamespace(
        use_paged_attention=use_paged_attention,
        aux_hidden_state_layers=list(aux_layers),
        num_kv_heads=2,
        head_size=128,
        filename="model.onnx",
        attention_attrs={"paged_block_size": 256},
        context_length=32768,
        original_context_length=131072,
    )
    return model


def test_absent_option_builds_no_drafter(tmp_path):
    model = _composite()

    model.make_dflash2_init(io_dtype=None, extra_options={})
    model.make_dflash2_model(str(tmp_path))

    assert model.dflash2 is None


def test_drafter_requires_paged_attention(tmp_path):
    model = _composite(use_paged_attention=False)

    with pytest.raises(ValueError, match="use_paged_attention"):
        model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})


# The drafter reads the target's residual streams by position, so a tap set that does not match
# the checkpoint's target_layer_ids silently feeds it the wrong tensors.
@pytest.mark.parametrize("aux_layers", [[1, 11], [1, 11, 22], [21, 11, 1], []])
def test_mismatched_tap_layers_are_rejected(tmp_path, aux_layers):
    model = _composite(aux_layers=aux_layers)

    with pytest.raises(ValueError, match="aux_hidden_state_layers"):
        model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})


def test_matching_tap_layers_are_accepted(tmp_path):
    model = _composite()

    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})

    assert model.dflash2_attrs["num_draft_tokens"] is None


def test_draft_token_count_can_be_overridden(tmp_path):
    model = _composite()

    model.make_dflash2_init(
        io_dtype=None,
        extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_num_draft_tokens": "4"},
    )

    assert model.dflash2_attrs["num_draft_tokens"] == 4


def test_draft_token_count_cannot_exceed_checkpoint_limit(tmp_path):
    model = _composite()

    with pytest.raises(ValueError, match=r"checkpoint limit \(4\)"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_num_draft_tokens": "5"},
        )


@pytest.mark.parametrize("num_draft_tokens", ["0", "-1"])
def test_draft_token_count_must_be_positive(tmp_path, num_draft_tokens):
    model = _composite()

    with pytest.raises(ValueError, match="positive integer"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={
                "dflash2_path": _draft_checkpoint(tmp_path),
                "dflash2_num_draft_tokens": num_draft_tokens,
            },
        )


def test_genai_config_gains_the_drafter_and_the_target_tap(tmp_path):
    config_path = tmp_path / "genai_config.json"
    config_path.write_text(json.dumps({"model": {"decoder": {}}}))
    model = _composite()
    model.dflash2 = types.SimpleNamespace(genai_config_section=lambda: {"filename": "dflash2.onnx"})

    model.add_dflash2_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["outputs"]["aux_hidden_states"] == "aux_hidden_states"
    assert config["model"]["dflash2"]["filename"] == "dflash2.onnx"
    assert config["model"]["dflash2"]["aux_hidden_state_layers"] == TARGET_LAYER_IDS


def test_shared_initializers_are_recorded_once_on_both_sides(tmp_path):
    config_path = tmp_path / "genai_config.json"
    shared = {"name": "model.embed_tokens.weight", "filename": "model.onnx.data"}
    config_path.write_text(json.dumps({"model": {"decoder": {"shared_initializers": [shared]}}}))
    model = _composite()
    model.dflash2 = types.SimpleNamespace(genai_config_section=lambda: {"filename": "dflash2.onnx"})
    model.dflash2_shared_initializers = [shared]

    model.add_dflash2_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["shared_initializers"] == [shared]
    assert config["model"]["dflash2"]["shared_initializers"] == [shared]


def test_builder_exposes_the_api_the_composite_drives():
    assert all(hasattr(DFlash2Builder, name) for name in ("make_model", "save_model", "genai_config_section"))


def test_duplicate_node_names_are_rejected():
    builder = object.__new__(DFlash2Builder)
    builder.node_names = {"duplicate"}

    with pytest.raises(ValueError, match="duplicate node name duplicate"):
        builder.make_node("Identity", [], [], name="duplicate")


def test_kv_cache_uses_configured_paged_block_size(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=512,
        max_position_embeddings=128,
    )

    builder.declare_io()

    assert builder.values["past_key_values.0.key"].shape[1] == 512


def test_non_fp8_lm_head_preserves_target_layout_and_dtype(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
    )
    builder.weights = {"lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size))}

    output = builder.make_lm_head("hidden_states", "num_sample")

    initializer = builder.graph.initializers["lm_head.MatMul.weight"].const_value
    assert tuple(initializer.shape) == (builder.hidden_size, builder.vocab_size)
    assert initializer.dtype == ir.DataType.FLOAT16
    assert builder.values[output].dtype == ir.DataType.FLOAT16


@pytest.mark.parametrize("scale_shape", [(), (1,), (1, 32), (32, 1)])
def test_fp8_lm_head_normalizes_supported_scale_layouts(tmp_path, scale_shape):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
    )
    builder.weights = {
        "lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size), dtype=torch.float8_e4m3fn),
        "lm_head.weight_scale": torch.ones(scale_shape),
    }

    builder.make_lm_head("hidden_states", "num_sample")

    scale = builder.graph.initializers["lm_head.MatMul.fp8_weight_scale"].const_value
    assert tuple(scale.shape) == (builder.vocab_size, 1)


def test_unsupported_rope_type_is_rejected(tmp_path):
    draft_dir = _draft_checkpoint(tmp_path)
    config_path = tmp_path / "dflash2_draft" / "config.json"
    config = json.loads(config_path.read_text())
    config["rope_parameters"]["rope_type"] = "longrope"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="does not support the 'longrope' RoPE type"):
        DFlash2Builder(draft_dir, str(tmp_path), ir.DataType.FLOAT16, 256, 128)


def test_drafter_uses_target_context_length(tmp_path, monkeypatch):
    captured = {}

    class StubDFlash2Builder:
        def __init__(self, _draft_dir, _target_dir, _io_dtype, _paged_block_size, max_position, **_kwargs):
            captured["max_position"] = max_position

        def make_model(self):
            pass

    dflash2_module = importlib.import_module("models.builders.dflash2")
    monkeypatch.setattr(dflash2_module, "DFlash2Builder", StubDFlash2Builder)
    model = _composite()
    model.dflash2_path = _draft_checkpoint(tmp_path)
    model.dflash2_attrs = {"io_dtype": None, "num_draft_tokens": None}

    model.make_dflash2_model(str(tmp_path))

    assert captured["max_position"] == model.decoder.context_length


def test_failed_save_preserves_existing_dflash2_files(tmp_path, monkeypatch):
    model_path = tmp_path / "dflash2.onnx"
    data_path = tmp_path / "dflash2.onnx.data"
    model_path.write_bytes(b"old model")
    data_path.write_bytes(b"old data")
    builder = object.__new__(DFlash2Builder)
    builder.filename = "dflash2.onnx"
    # save_model stamps build metadata on the model, so the stub has to accept attributes.
    builder.model = types.SimpleNamespace()

    def fail_save(_model, staged_path, **kwargs):
        with open(staged_path, "wb") as staged_model:
            staged_model.write(b"partial model")
        with open(os.path.join(os.path.dirname(staged_path), kwargs["external_data"]), "wb") as staged_data:
            staged_data.write(b"partial data")
        raise OSError("injected save failure")

    monkeypatch.setattr(ir, "save", fail_save)

    with pytest.raises(OSError, match="injected save failure"):
        builder.save_model(tmp_path)

    assert model_path.read_bytes() == b"old model"
    assert data_path.read_bytes() == b"old data"


@pytest.fixture
def mtp_init(monkeypatch):
    """Drive Qwen35MoEModel.make_mtp_init with the base seeding stubbed out."""

    def run(extra_options, num_mtp_layers=1):
        model = object.__new__(Qwen35MoEModel)
        model.mtp_attrs = {}
        # MTPModel.make_mtp_init only seeds mtp_attrs and hands back the decoder's options.
        monkeypatch.setattr(MTPModel, "make_mtp_init", lambda self, _c, opts: dict(opts))
        config = types.SimpleNamespace(mtp_num_hidden_layers=num_mtp_layers)
        decoder_options = Qwen35MoEModel.make_mtp_init(model, config, extra_options)
        return model.mtp_attrs["build"], decoder_options

    return run


def test_a_checkpoint_with_an_mtp_head_builds_one_by_default(mtp_init):
    build, decoder_options = mtp_init({})

    assert build is True
    assert decoder_options["include_hidden_states"] is True


# The Engine drives one drafter per model, so the block drafter replaces the MTP head. Building
# both would emit a ~916 MB mtp.onnx that nothing ever runs.
def test_a_block_drafter_suppresses_the_mtp_head(tmp_path, mtp_init):
    build, decoder_options = mtp_init({"dflash2_path": _draft_checkpoint(tmp_path)})

    assert build is False
    assert "include_hidden_states" not in decoder_options


def test_a_checkpoint_without_an_mtp_head_is_unaffected(tmp_path, mtp_init):
    build, _ = mtp_init({"dflash2_path": _draft_checkpoint(tmp_path)}, num_mtp_layers=0)

    assert build is False
