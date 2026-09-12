# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib
import json
import os
import types

import onnx_ir as ir
import pytest
import torch

from models.builders.base import Model
from models.builders.dflash2 import DFlash2Builder
from models.builders.mtp import MTPModel
from models.builders.qwen import Qwen35MoEModel

TARGET_LAYER_IDS = [1, 11, 21]
AUX_LAYERS = [layer_id + 1 for layer_id in TARGET_LAYER_IDS]


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


def _composite(aux_layers=AUX_LAYERS, use_paged_attention=True):
    model = object.__new__(Qwen35MoEModel)
    model.dflash2 = None
    model.dflash2_shared_initializers = []
    model.decoder = types.SimpleNamespace(
        use_paged_attention=use_paged_attention,
        aux_hidden_state_layers=list(aux_layers),
        num_kv_heads=2,
        head_size=128,
        num_layers=32,
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


# SpecForge target_layer_ids name layer outputs, while aux_hidden_state_layers names the residual
# entering a layer. Passing the checkpoint IDs through unchanged silently selects the prior outputs.
@pytest.mark.parametrize("aux_layers", [TARGET_LAYER_IDS, [2, 12], [2, 12, 23], [22, 12, 2], []])
def test_mismatched_tap_layers_are_rejected(tmp_path, aux_layers):
    model = _composite(aux_layers=aux_layers)

    with pytest.raises(ValueError, match="aux_hidden_state_layers"):
        model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})


def test_matching_tap_layers_are_accepted(tmp_path):
    model = _composite()

    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})

    assert model.dflash2_attrs["num_draft_tokens"] is None


def test_checkpoint_must_define_target_layers(tmp_path):
    model = _composite(aux_layers=[])

    with pytest.raises(ValueError, match="at least one target_layer_ids entry"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={"dflash2_path": _draft_checkpoint(tmp_path, target_layer_ids=[])},
        )


def test_checkpoint_cannot_target_an_unexposable_layer(tmp_path):
    model = _composite(aux_layers=[32])

    with pytest.raises(ValueError, match=r"target_layer_ids must lie in \[0, 31\)"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={"dflash2_path": _draft_checkpoint(tmp_path, target_layer_ids=[31])},
        )


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
    assert config["model"]["dflash2"]["aux_hidden_state_layers"] == AUX_LAYERS


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


def _quant_composite(
    weight_name="lm_head.MatMul.weight_Q4",
    scales_name="lm_head.MatMul.weight_scales",
    zero_point_name="",
    exclude_lm_head=False,
    quantized_lm_head=None,
    onnx_dtype=ir.DataType.INT4,
    last_matmul_type=None,
):
    model = _composite()
    model.decoder.exclude_lm_head = exclude_lm_head
    model.decoder.onnx_dtype = onnx_dtype
    model.decoder.quantization_algo = "default"
    model.decoder.matmul_mixed_precision = (
        {"last_matmul": last_matmul_type} if last_matmul_type is not None else {}
    )
    model.decoder.quant_attrs = {
        "matmul_block_size": 32,
        "is_symmetric": True,
        "op_types_to_quantize": ["MatMul"],
        "nodes_to_exclude": [],
    }
    model.decoder.matmul_attrs = {"weights_prepacked": 1}
    if quantized_lm_head is None:
        model.decoder.is_lm_head_quantized = types.MethodType(Model.is_lm_head_quantized, model.decoder)
    else:
        model.decoder.is_lm_head_quantized = lambda: quantized_lm_head
    if weight_name is None:
        model.decoder.make_tied_quantized_embedding_input_names = types.MethodType(
            Model.make_tied_quantized_embedding_input_names, model.decoder
        )
    else:
        model.decoder.make_tied_quantized_embedding_input_names = lambda: (
            4,
            weight_name,
            scales_name,
            zero_point_name,
        )
    return model


@pytest.mark.parametrize("precision", ["fp16", "int3", "INT4BIT", ""])
def test_precision_option_is_rejected_when_unknown(tmp_path, precision):
    model = _composite()

    with pytest.raises(ValueError, match="dflash2_precision"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_precision": precision},
        )


def test_precision_defaults_to_dense_bf16(tmp_path):
    model = _composite()

    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})

    assert model.dflash2_attrs["precision"] == "bf16"
    assert model.block_drafter_quant("bf16") is None


def test_quantized_drafter_reuses_the_targets_lm_head_names():
    quant = _quant_composite().block_drafter_quant("int4")

    assert quant["bits"] == 4
    assert quant["block_size"] == 32
    assert quant["prepack"] == 1
    # Folding onto the target's copy only works if the drafter quantizes its head identically.
    assert quant["lm_head"] == {"bits": 4, "block_size": 32, "prepack": 1}


@pytest.mark.parametrize(
    "onnx_dtype,last_matmul_type,expected_bits",
    [
        (ir.DataType.INT4, None, 4),
        (ir.DataType.INT4, "int8", 8),
        (ir.DataType.INT8, None, 8),
    ],
)
def test_drafter_resolves_the_actual_target_lm_head_bit_width(onnx_dtype, last_matmul_type, expected_bits):
    model = _quant_composite(
        weight_name=None,
        onnx_dtype=onnx_dtype,
        last_matmul_type=last_matmul_type,
    )

    head_bits, *_ = model.decoder.make_tied_quantized_embedding_input_names()
    quant = model.block_drafter_quant("int4")

    assert head_bits == expected_bits
    if expected_bits == 4:
        assert quant["lm_head"]["bits"] == expected_bits
    else:
        # The block-drafter quantizer cannot reproduce the target's Q8G initializer layout.
        assert quant["lm_head"] is None


def test_dense_target_keeps_the_drafter_lm_head_dense():
    model = _quant_composite(weight_name=None, onnx_dtype=ir.DataType.FLOAT16)

    assert model.block_drafter_quant("int4")["lm_head"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_name": "lm_head.MatMul.weight_Q4G32", "scales_name": "lm_head.MatMul.weight_scale"},
        {"zero_point_name": "lm_head.MatMul.weight_zp"},
        {"weight_name": "lm_head.MatMul.weight", "scales_name": ""},
        {"exclude_lm_head": True},
        {"quantized_lm_head": False},
    ],
)
def test_unshareable_target_head_leaves_the_drafter_head_dense(kwargs):
    quant = _quant_composite(**kwargs).block_drafter_quant("int4")

    assert quant["bits"] == 4
    assert quant["lm_head"] is None


def test_quantized_body_emits_matmulnbits_without_transposing(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 4, "block_size": 8, "prepack": 0, "lm_head": None},
    )

    builder.matmul("/probe/MatMul", "hidden_states", torch.ones((16, 8)), 8, 16, "num_block")

    node = next(node for node in builder.graph if node.name == "/probe/MatMul")
    assert node.op_type == "MatMulNBits"
    assert node.domain == "com.microsoft"
    assert node.attributes["K"].value == 8
    assert node.attributes["N"].value == 16
    # MatMulNBits takes [N, K], so the dense path's transpose must not be applied.
    assert tuple(builder.graph.initializers["probe.MatMul.weight_Q4"].const_value.shape) == (16, 1, 4)


# The prepacked fpA_intB kernel takes FP16 activations only, so the bf16 body must ship the
# plain blockwise layout even though the target it drafts for is prepacked.
def test_bf16_body_never_prepacks_even_when_the_target_does(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 4, "block_size": 8, "prepack": 1, "lm_head": None},
    )

    builder.matmul("/probe/MatMul", "hidden_states", torch.ones((16, 8)), 8, 16, "num_block")

    node = next(node for node in builder.graph if node.name == "/probe/MatMul")
    assert builder.io_dtype == ir.DataType.BFLOAT16
    assert "weight_prepacked" not in node.attributes


def test_quantized_lm_head_matches_the_targets_initializer_names(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 4, "block_size": 8, "prepack": 0, "lm_head": {"bits": 4, "block_size": 8, "prepack": 0}},
    )
    builder.weights = {"lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size))}

    output = builder.make_lm_head("hidden_states", "num_sample")

    node = next(node for node in builder.graph if node.name == "/lm_head/MatMul")
    assert node.op_type == "MatMulNBits"
    assert [value.name for value in node.inputs][1:] == [
        "lm_head.MatMul.weight_Q4",
        "lm_head.MatMul.weight_scales",
    ]
    # Scales ride at the target's dtype, not the drafter's bf16 body dtype, or they cannot fold.
    assert builder.graph.initializers["lm_head.MatMul.weight_scales"].const_value.dtype == ir.DataType.FLOAT16
    assert builder.values[output].dtype == ir.DataType.FLOAT16


@pytest.mark.parametrize(
    "bits,hidden_size,vocab_size,prepack,external_dtype",
    [
        (4, 32, 32, 1, ir.DataType.FLOAT16),
        (8, 32, 33, 1, ir.DataType.FLOAT16),
        (4, 33, 64, 1, ir.DataType.FLOAT16),
        (4, 32, 64, 2, ir.DataType.FLOAT16),
        (4, 32, 64, 1, ir.DataType.BFLOAT16),
    ],
)
def test_ineligible_lm_head_keeps_blockwise_layout(tmp_path, bits, hidden_size, vocab_size, prepack, external_dtype):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        external_dtype,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={
            "bits": bits,
            "block_size": 32,
            "prepack": prepack,
            "lm_head": {"bits": bits, "block_size": 32, "prepack": prepack},
        },
    )
    builder.hidden_size = hidden_size
    builder.vocab_size = vocab_size
    builder.weights = {"lm_head.weight": torch.ones((vocab_size, hidden_size))}

    builder.make_lm_head("hidden_states", "num_sample")

    node = next(node for node in builder.graph if node.name == "/lm_head/MatMul")
    assert node.op_type == "MatMulNBits"
    assert "weight_prepacked" not in node.attributes
    weight = builder.graph.initializers[f"lm_head.MatMul.weight_Q{bits}"].const_value
    assert tuple(weight.shape) == (vocab_size, (hidden_size + 31) // 32, 32 * bits // 8)


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


def test_five_uniformly_windowed_layers_accept_total_layer_count(tmp_path):
    draft_dir = _draft_checkpoint(tmp_path)
    config_path = tmp_path / "dflash2_draft" / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        num_hidden_layers=5,
        use_sliding_window=True,
        sliding_window=2048,
        max_window_layers=5,
        layer_types=["sliding_attention"] * 5,
    )
    config_path.write_text(json.dumps(config))

    builder = DFlash2Builder(draft_dir, str(tmp_path), ir.DataType.FLOAT16, 256, 128)

    assert builder.sliding_window == 2048


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
    model.dflash2_attrs = {"io_dtype": None, "num_draft_tokens": None, "precision": "bf16"}

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


# The MTP workflow needs per-token logits from the main LM head, so a deployment that
# prunes the head has to be able to drop the MTP head at build time rather than by
# editing genai_config.json afterwards.
def test_exclude_mtp_suppresses_the_head(mtp_init):
    build, decoder_options = mtp_init({"exclude_mtp": True})

    assert build is False
    assert "include_hidden_states" not in decoder_options


def test_exclude_mtp_admits_a_pruned_lm_head(mtp_init):
    build, _ = mtp_init({"exclude_mtp": True, "prune_lm_head": True})

    assert build is False


def test_a_pruned_lm_head_is_still_rejected_while_the_mtp_head_is_built(mtp_init):
    with pytest.raises(ValueError, match="prune_lm_head"):
        mtp_init({"prune_lm_head": True})
