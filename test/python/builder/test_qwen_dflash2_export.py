# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import types

import pytest
from models.builders.dflash2 import DFlash2Builder
from models.builders.mtp import MTPModel
from models.builders.qwen import Qwen35MoEModel

TARGET_LAYER_IDS = [1, 11, 21]


def _draft_checkpoint(tmp_path, target_layer_ids=TARGET_LAYER_IDS):
    draft_dir = tmp_path / "dflash2_draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps({"dflash_config": {"target_layer_ids": target_layer_ids}}))
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
        extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_num_draft_tokens": "5"},
    )

    assert model.dflash2_attrs["num_draft_tokens"] == 5


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
