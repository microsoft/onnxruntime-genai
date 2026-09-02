# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib
import json
import types

import onnx_ir as ir
import pytest
import torch

from models.builders.dspark import DSparkBuilder
from models.builders.mtp import MTPModel
from models.builders.qwen import Qwen35MoEModel

# SpecForge names the layers whose *output* it taps, and the builder names the residual stream
# *entering* a layer, so the tap set is each target layer id plus one.
TARGET_LAYER_IDS = [0, 10, 20]
AUX_LAYERS = [1, 11, 21]


def _draft_checkpoint(tmp_path, name="dspark_draft", target_layer_ids=TARGET_LAYER_IDS):
    draft_dir = tmp_path / name
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
        "block_size": 5,
        "markov_rank": 4,
        "dflash_config": {
            "mask_token_id": 31,
            "target_layer_ids": target_layer_ids,
        },
    }
    (draft_dir / "config.json").write_text(json.dumps(config))
    return str(draft_dir)


def _composite(aux_layers=AUX_LAYERS, use_paged_attention=True, dflash2_path=None):
    model = object.__new__(Qwen35MoEModel)
    model.dspark = None
    model.dspark_shared_initializers = []
    model.dflash2_path = dflash2_path
    model.decoder = types.SimpleNamespace(
        use_paged_attention=use_paged_attention,
        aux_hidden_state_layers=list(aux_layers),
        filename="model.onnx",
        attention_attrs={"paged_block_size": 256},
        context_length=32768,
        original_context_length=131072,
    )
    return model


def test_absent_option_builds_no_drafter(tmp_path):
    model = _composite()

    model.make_dspark_init(io_dtype=None, extra_options={})
    model.make_dspark_model(str(tmp_path))

    assert model.dspark is None


def test_drafter_requires_paged_attention(tmp_path):
    model = _composite(use_paged_attention=False)

    with pytest.raises(ValueError, match="use_paged_attention"):
        model.make_dspark_init(io_dtype=None, extra_options={"dspark_path": _draft_checkpoint(tmp_path)})


def test_two_block_drafters_cannot_be_exported_together(tmp_path):
    model = _composite(dflash2_path=_draft_checkpoint(tmp_path, name="dflash2_draft"))

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.make_dspark_init(io_dtype=None, extra_options={"dspark_path": _draft_checkpoint(tmp_path)})


# Passing SpecForge's own target_layer_ids straight through is the off-by-one that leaves
# acceptance pinned at exactly 1.0, so it has to be rejected rather than tolerated.
@pytest.mark.parametrize("aux_layers", [TARGET_LAYER_IDS, [1, 11], [2, 11, 21], [21, 11, 1], []])
def test_mismatched_tap_layers_are_rejected(tmp_path, aux_layers):
    model = _composite(aux_layers=aux_layers)

    with pytest.raises(ValueError, match="aux_hidden_state_layers"):
        model.make_dspark_init(io_dtype=None, extra_options={"dspark_path": _draft_checkpoint(tmp_path)})


def test_tap_layers_one_past_each_target_layer_are_accepted(tmp_path):
    model = _composite()

    model.make_dspark_init(io_dtype=None, extra_options={"dspark_path": _draft_checkpoint(tmp_path)})

    assert model.dspark_attrs["num_draft_tokens"] is None
    assert model.dspark_attrs["top_k"] == 16


def test_lattice_width_and_draft_count_can_be_overridden(tmp_path):
    model = _composite()

    model.make_dspark_init(
        io_dtype=None,
        extra_options={
            "dspark_path": _draft_checkpoint(tmp_path),
            "dspark_num_draft_tokens": "6",
            "dspark_top_k": "8",
        },
    )

    assert model.dspark_attrs["num_draft_tokens"] == 6
    assert model.dspark_attrs["top_k"] == 8


@pytest.mark.parametrize("num_draft_tokens", ["0", "-1", "invalid"])
def test_draft_token_count_must_be_positive(tmp_path, num_draft_tokens):
    model = _composite()

    with pytest.raises(ValueError, match="dspark_num_draft_tokens must be a positive integer"):
        model.make_dspark_init(
            io_dtype=None,
            extra_options={
                "dspark_path": _draft_checkpoint(tmp_path),
                "dspark_num_draft_tokens": num_draft_tokens,
            },
        )


@pytest.mark.parametrize("top_k", ["0", "-1", "invalid"])
def test_lattice_width_must_be_positive(tmp_path, top_k):
    model = _composite()

    with pytest.raises(ValueError, match="dspark_top_k must be a positive integer"):
        model.make_dspark_init(
            io_dtype=None,
            extra_options={"dspark_path": _draft_checkpoint(tmp_path), "dspark_top_k": top_k},
        )


def test_lattice_width_cannot_exceed_vocabulary(tmp_path):
    model = _composite()

    with pytest.raises(ValueError, match="dspark_top_k must not exceed the drafter vocabulary size"):
        model.make_dspark_init(
            io_dtype=None,
            extra_options={"dspark_path": _draft_checkpoint(tmp_path), "dspark_top_k": "33"},
        )


def test_duplicate_node_names_are_rejected():
    builder = object.__new__(DSparkBuilder)
    builder.node_names = {"duplicate"}

    with pytest.raises(ValueError, match="duplicate node name duplicate"):
        builder.make_node("Identity", [], [], name="duplicate")


def test_kv_cache_uses_configured_paged_block_size(tmp_path):
    builder = DSparkBuilder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=512,
        max_position_embeddings=128,
    )

    builder.declare_io()

    assert builder.values["past_key_values.0.key"].shape[1] == 512


def test_non_fp8_lm_head_does_not_require_a_scale(tmp_path):
    builder = DSparkBuilder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
    )
    builder.weights = {"lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size), dtype=torch.float16)}

    output = builder.make_lm_head("hidden_states")

    assert output == "/lm_head/MatMul/output_0"


def test_fp8_lm_head_requires_a_scale(tmp_path):
    builder = DSparkBuilder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
    )
    builder.weights = {
        "lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size), dtype=torch.float8_e4m3fn)
    }

    with pytest.raises(ValueError, match="FP8 LM head weight is missing 'lm_head.weight_scale'"):
        builder.make_lm_head("hidden_states")


def test_drafter_uses_original_context_length(tmp_path, monkeypatch):
    captured = {}

    class StubDSparkBuilder:
        def __init__(self, _draft_dir, _target_dir, _io_dtype, _paged_block_size, max_position, **_kwargs):
            captured["max_position"] = max_position

        def make_model(self):
            pass

    dspark_module = importlib.import_module("models.builders.dspark")
    monkeypatch.setattr(dspark_module, "DSparkBuilder", StubDSparkBuilder)
    model = _composite()
    model.dspark_path = _draft_checkpoint(tmp_path)
    model.dspark_attrs = {"io_dtype": None, "num_draft_tokens": None, "top_k": 16}

    model.make_dspark_model(str(tmp_path))

    assert captured["max_position"] == model.decoder.original_context_length


def test_genai_config_gains_the_drafter_and_the_target_tap(tmp_path):
    config_path = tmp_path / "genai_config.json"
    config_path.write_text(json.dumps({"model": {"decoder": {}}}))
    model = _composite()
    model.dspark = types.SimpleNamespace(genai_config_section=lambda: {"filename": "dspark.onnx"})

    model.add_dspark_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["outputs"]["aux_hidden_states"] == "aux_hidden_states"
    assert config["model"]["dspark"]["filename"] == "dspark.onnx"
    assert config["model"]["dspark"]["aux_hidden_state_layers"] == AUX_LAYERS


def test_shared_initializers_are_recorded_once_on_both_sides(tmp_path):
    config_path = tmp_path / "genai_config.json"
    shared = {"name": "model.embed_tokens.weight", "filename": "model.onnx.data"}
    config_path.write_text(json.dumps({"model": {"decoder": {"shared_initializers": [shared]}}}))
    model = _composite()
    model.dspark = types.SimpleNamespace(genai_config_section=lambda: {"filename": "dspark.onnx"})
    model.dspark_shared_initializers = [shared]

    model.add_dspark_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["shared_initializers"] == [shared]
    assert config["model"]["dspark"]["shared_initializers"] == [shared]


def test_builder_exposes_the_api_the_composite_drives():
    assert all(hasattr(DSparkBuilder, name) for name in ("make_model", "save_model", "genai_config_section"))


@pytest.fixture
def mtp_init(monkeypatch):
    """Drive Qwen35MoEModel.make_mtp_init with the base seeding stubbed out."""

    def run(extra_options, num_mtp_layers=1):
        model = object.__new__(Qwen35MoEModel)
        model.mtp_attrs = {}
        monkeypatch.setattr(MTPModel, "make_mtp_init", lambda self, _c, opts: dict(opts))
        config = types.SimpleNamespace(mtp_num_hidden_layers=num_mtp_layers)
        Qwen35MoEModel.make_mtp_init(model, config, extra_options)
        return model.mtp_attrs["build"]

    return run


# The Engine drives one drafter per model, so DSpark replaces the MTP head just as DFlash 2 does.
def test_a_dspark_drafter_suppresses_the_mtp_head(tmp_path, mtp_init):
    assert mtp_init({"dspark_path": _draft_checkpoint(tmp_path)}) is False


def test_an_mtp_head_is_still_built_without_a_block_drafter(mtp_init):
    assert mtp_init({}) is True
