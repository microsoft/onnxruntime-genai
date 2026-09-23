# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib
import json
import os
import types

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from quantization import QuantConfig

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
        exclude_embeds=False,
        exclude_lm_head=False,
        is_lm_head_quantized=lambda: False,
        onnx_dtype=ir.DataType.FLOAT16,
        quantization_algo="default",
        tied_quantized_embeddings=False,
        quant_attrs={
            "op_types_to_quantize": ("MatMul",),
            "nodes_to_exclude": [],
            "is_symmetric": True,
            "matmul_block_size": 32,
            "use_qdq": False,
        },
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
    model.dflash2 = types.SimpleNamespace(
        genai_config_section=lambda: {
            "filename": "dflash2.onnx",
            "session_options": {"ep.cuda.fpa_intb_gemm": "0"},
        }
    )

    model.add_dflash2_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["outputs"]["aux_hidden_states"] == "aux_hidden_states"
    assert config["model"]["dflash2"]["filename"] == "dflash2.onnx"
    assert config["model"]["dflash2"]["session_options"]["ep.cuda.fpa_intb_gemm"] == "0"
    assert config["model"]["dflash2"]["aux_hidden_state_layers"] == AUX_LAYERS


def test_dflash2_config_disables_inherited_fpa_intb_selection(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
    )

    section = builder.genai_config_section()

    assert section["session_options"] == {"ep.cuda.fpa_intb_gemm": "0"}


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


def test_target_mlp_gate_up_fusion_preserves_projection_order():
    model = object.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.intermediate_size = 2
    model.hidden_size = 3
    model.mlp_attrs = {"output_0": ""}
    calls = {}

    def make_matmul(self, projection, basename, root_input):
        calls.setdefault("matmuls", []).append((basename, root_input, projection.weight.clone()))
        return basename

    def make_split(self, name, inputs, outputs, dtypes, shapes, axis=-1, num_outputs=None):
        calls["split"] = (name, inputs, outputs, dtypes, shapes, axis, num_outputs)

    model.make_matmul = types.MethodType(make_matmul, model)
    model.make_split = types.MethodType(make_split, model)
    model.make_activation = types.MethodType(lambda self, layer_id, root_input: f"/act/{layer_id}", model)
    model.make_mul = types.MethodType(lambda self, *args, **kwargs: None, model)
    model.make_hidden_state_shape = types.MethodType(lambda self, last_dim: ["tokens", last_dim], model)

    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.full((2, 3), 1.0), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.full((2, 3), 2.0), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )

    model.make_mlp_proj_fused(4, mlp, "residual")

    fused_name, fused_input, fused_weight = calls["matmuls"][0]
    assert fused_name == "/model/layers.4/mlp/gate_up_proj/MatMul"
    assert fused_input == "residual"
    torch.testing.assert_close(fused_weight[:2], mlp.gate_proj.weight)
    torch.testing.assert_close(fused_weight[2:], mlp.up_proj.weight)
    assert calls["split"][2] == [
        "/model/layers.4/mlp/gate_proj/MatMul/output_0",
        "/model/layers.4/mlp/up_proj/MatMul/output_0",
    ]
    assert calls["split"][-1] == 2
    assert model.mlp_attrs["output_0"] == "/model/layers.4/mlp/down_proj/MatMul/output_0"


def test_target_mlp_gate_up_fusion_zero_fills_a_missing_bias():
    model = object.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.intermediate_size = 2
    model.hidden_size = 3
    model.mlp_attrs = {"output_0": ""}
    calls = {}

    def make_matmul(self, projection, basename, root_input):
        calls.setdefault("projections", []).append(projection)
        return basename

    model.make_matmul = types.MethodType(make_matmul, model)
    model.make_split = types.MethodType(lambda self, *args, **kwargs: None, model)
    model.make_add_bias = types.MethodType(lambda self, *args, **kwargs: None, model)
    model.make_activation = types.MethodType(lambda self, layer_id, root_input: f"/act/{layer_id}", model)
    model.make_mul = types.MethodType(lambda self, *args, **kwargs: None, model)
    model.make_hidden_state_shape = types.MethodType(lambda self, last_dim: ["tokens", last_dim], model)

    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=torch.ones(2)),
        up_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )

    model.make_mlp_proj_fused(0, mlp, "residual")

    torch.testing.assert_close(calls["projections"][0].bias, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_target_mlp_gate_up_fusion_rejects_projection_shape_mismatch():
    model = object.__new__(Model)
    model.intermediate_size = 2
    model.hidden_size = 3
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones((3, 3)), bias=None),
    )

    with pytest.raises(ValueError, match="gate/up weights with shape"):
        model.make_mlp_proj_fused(0, mlp, "residual")


def test_target_mlp_gate_up_fusion_maps_matching_exclusions():
    model = object.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.intermediate_size = 2
    model.hidden_size = 3
    model.mlp_attrs = {"output_0": ""}
    model.quant_attrs = {
        "nodes_to_exclude": [
            "/model/layers.0/mlp/gate_proj/MatMul",
            "/model/layers.0/mlp/up_proj/MatMul",
        ]
    }
    model.make_matmul = types.MethodType(
        lambda self, projection, basename, root_input: (
            self.exclude_node_from_quantization(basename)
            if getattr(projection, "exclude_from_quantization", False)
            else None
        )
        or basename,
        model,
    )
    model.make_split = types.MethodType(lambda self, *args, **kwargs: None, model)
    model.make_activation = types.MethodType(lambda self, layer_id, root_input: f"/act/{layer_id}", model)
    model.make_mul = types.MethodType(lambda self, *args, **kwargs: None, model)
    model.make_hidden_state_shape = types.MethodType(lambda self, last_dim: ["tokens", last_dim], model)
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )

    model.make_mlp_proj_fused(0, mlp, "residual")

    assert "/model/layers.0/mlp/gate_up_proj/MatMul" in model.quant_attrs["nodes_to_exclude"]


def test_target_mlp_gate_up_fusion_rejects_one_sided_exclusion():
    model = object.__new__(Model)
    model.intermediate_size = 2
    model.hidden_size = 3
    model.quant_attrs = {"nodes_to_exclude": ["/model/layers.0/mlp/gate_proj/MatMul"]}
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
    )

    with pytest.raises(ValueError, match="applies to only one"):
        model.make_mlp_proj_fused(0, mlp, "residual")


@pytest.mark.parametrize("quant_type,bits", [("int4", 4), ("int8", 8)])
def test_exact_name_quantization_override_is_forwarded_to_quantizer(quant_type, bits):
    model = object.__new__(Model)
    model.quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(
            method="default",
            overrides=[
                types.SimpleNamespace(
                    match={"name": "/model/layers.0/mlp/down_proj/MatMul"},
                    type=quant_type,
                    exclude=False,
                )
            ],
        )
    )
    model.quant_type = None
    model.quant_attrs = {}

    model.make_quant_init(types.SimpleNamespace())

    assert model.int4_customized_weight_config == {"/model/layers.0/mlp/down_proj/MatMul": {"bits": bits}}


@pytest.mark.parametrize("match", [{"name": "/model/layers.0/mlp/down_proj/MatMul"}, {"preset": "last_matmul"}])
def test_int8_quantization_override_rejects_qdq_format(match):
    model = object.__new__(Model)
    model.quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(
            method="default",
            overrides=[
                types.SimpleNamespace(
                    match=match,
                    type="int8",
                    exclude=False,
                )
            ],
        )
    )
    model.quant_type = None
    model.quant_attrs = {"use_qdq": True}

    with pytest.raises(NotImplementedError, match="INT8 weight overrides are not supported with QDQ"):
        model.make_quant_init(types.SimpleNamespace())


def _exact_override_model(tmp_path, *, constant_weight):
    node_name = "/probe/MatMul"
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
    )
    if constant_weight:
        builder.matmul(node_name, "hidden_states", torch.ones((16, 32)), 32, 16, "num_block")
    else:
        builder.make_value("dynamic_weight", ir.DataType.FLOAT16, [32, 16])
        builder.make_node("MatMul", ["hidden_states", "dynamic_weight"], ["output"], name=node_name)

    override = types.SimpleNamespace(match={"name": node_name}, type="int8", exclude=False)
    model = object.__new__(Model)
    model.model = builder.model
    model.exact_quant_override_names = {node_name}
    model.exact_quant_overrides = {node_name: override}
    model.quantization_algo = "default"
    model.int4_customized_weight_config = {node_name: {"bits": 8}}
    model.quant_attrs = {
        "accuracy_level": 0,
        "bits": 4,
        "is_symmetric": True,
        "matmul_block_size": 16,
        "nodes_to_exclude": [],
        "op_types_to_quantize": ("MatMul",),
        "use_qdq": False,
    }
    model.matmul_attrs = {"weights_prepacked": 0}
    model.ep = "cpu"
    return model, node_name


def test_exact_name_override_rejects_dynamic_weight(tmp_path):
    model, _ = _exact_override_model(tmp_path, constant_weight=False)

    with pytest.raises(ValueError, match="require a constant weight initializer"):
        model.to_nbits()


def test_exact_name_override_is_verified_after_quantization(tmp_path):
    model, node_name = _exact_override_model(tmp_path, constant_weight=True)

    quantized = model.to_nbits()
    node = next(node for node in quantized.graph if node.name == f"{node_name}_Q8")

    assert node.op_type == "MatMulNBits"
    assert node.attributes["bits"].value == 8


@pytest.mark.parametrize("preset", ["last_matmul", "mixed_layers", "linear_attn"])
@pytest.mark.parametrize("quant_type,bits", [("int4", 4), ("int8", 8), (" INT8 ", 8)])
def test_preset_quantization_override_initializes_the_node_map(preset, quant_type, bits):
    model = object.__new__(Model)
    model.quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(
            method="default",
            overrides=[types.SimpleNamespace(match={"preset": preset}, type=quant_type, exclude=False)],
        )
    )
    model.quant_type = None
    model.quant_attrs = {"nodes_to_exclude": []}
    model.num_layers = 1
    model.layer_types = ["linear_attention"]
    model.mlp_attrs = {}

    model.make_quant_init(types.SimpleNamespace())

    assert model.int4_customized_weight_config
    assert all(config == {"bits": bits} for config in model.int4_customized_weight_config.values())
    if preset == "last_matmul":
        assert model.int4_customized_weight_config == {"/lm_head/MatMul": {"bits": bits}}


@pytest.mark.parametrize(
    "match",
    [{"preset": preset} for preset in ("last_matmul", "mixed_layers", "linear_attn")] + [{"name": "/lm_head/MatMul"}],
)
@pytest.mark.parametrize("quant_type", ["none", "fp16", "fp32", "bf16", "mxfp4", "nvfp4", "uint4", "uint8"])
def test_weight_overrides_reject_formats_not_representable_by_bit_width(match, quant_type):
    model = object.__new__(Model)
    model.quant_config = QuantConfig.from_dict(
        {"weights": {"type": "int4", "overrides": [{"match": match, "type": quant_type}]}}
    )
    model.quant_type = None
    model.quant_attrs = {}

    with pytest.raises(ValueError, match="weight overrides currently support only int4 or int8"):
        model.make_quant_init(types.SimpleNamespace())


def test_legacy_exclusion_wins_over_mixed_precision_preset():
    quant_config = QuantConfig.from_extra_options(
        {"matmul_mixed_precision": "last_matmul:int8", "nodes_to_exclude": ["/lm_head/MatMul"]},
        precision="int4",
    )
    model = object.__new__(Model)
    model.quant_config = quant_config
    model.quant_type = None
    model.quant_attrs = {"nodes_to_exclude": []}

    model.make_quant_init(types.SimpleNamespace())

    assert model.int4_customized_weight_config == {}
    assert model.quant_attrs["nodes_to_exclude"] == ["/lm_head/MatMul"]


@pytest.mark.parametrize("exclude_first", [False, True])
def test_exact_quantization_rules_use_first_match(exclude_first):
    node_name = "/lm_head/MatMul"
    typed = types.SimpleNamespace(match={"name": node_name}, type="int8", exclude=False)
    excluded = types.SimpleNamespace(match={"name": node_name}, type=None, exclude=True)
    model = object.__new__(Model)
    model.quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(
            method="default",
            overrides=[excluded, typed] if exclude_first else [typed, excluded],
        )
    )
    model.quant_type = None
    model.quant_attrs = {"nodes_to_exclude": [node_name]}

    model.make_quant_init(types.SimpleNamespace())

    if exclude_first:
        assert model.int4_customized_weight_config == {}
        assert model.quant_attrs["nodes_to_exclude"] == [node_name]
    else:
        assert model.int4_customized_weight_config == {node_name: {"bits": 8}}
        assert model.quant_attrs["nodes_to_exclude"] == []


def test_unsupported_typed_quantization_match_is_rejected():
    model = object.__new__(Model)
    model.quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(
            method="default",
            overrides=[types.SimpleNamespace(match={"name_regex": ".*down_proj.*"}, type="int8", exclude=False)],
        )
    )
    model.quant_type = None

    with pytest.raises(ValueError, match="only a preset or an exact node name"):
        model.make_quant_init(types.SimpleNamespace())


def test_int8_embedding_override_fails_until_export_is_supported():
    model = object.__new__(Model)
    model.quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(
            method="default",
            overrides=[
                types.SimpleNamespace(
                    match={"name": "/model/embed_tokens/Gather"},
                    type="int8",
                    exclude=False,
                )
            ],
        )
    )
    model.quant_type = None

    with pytest.raises(NotImplementedError, match="INT8 embedding export is not supported"):
        model.make_quant_init(types.SimpleNamespace())


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
    io_dtype=ir.DataType.FLOAT16,
    ep="cuda",
    use_qdq=False,
):
    model = _composite()
    model.decoder.exclude_lm_head = exclude_lm_head
    model.decoder.onnx_dtype = onnx_dtype
    model.decoder.io_dtype = io_dtype
    model.decoder.ep = ep
    model.decoder.quantization_algo = "default"
    model.decoder.matmul_mixed_precision = {"last_matmul": last_matmul_type} if last_matmul_type is not None else {}
    model.decoder.quant_attrs = {
        "matmul_block_size": 32,
        "is_symmetric": True,
        "op_types_to_quantize": ["MatMul"],
        "nodes_to_exclude": [],
        "use_qdq": use_qdq,
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


def test_precision_defaults_to_dense_bf16_with_adopted_target_head(tmp_path, monkeypatch):
    captured = {}

    class StubDFlash2Builder:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        def make_model(self):
            pass

    dflash2_module = importlib.import_module("models.builders.dflash2")
    monkeypatch.setattr(dflash2_module, "DFlash2Builder", StubDFlash2Builder)
    model = _quant_composite()

    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})
    model.make_dflash2_model(str(tmp_path))

    assert model.dflash2_attrs["precision"] == "bf16"
    assert captured["quant"] is None
    assert captured["lm_head_quant"] == {
        "bits": 4,
        "block_size": 32,
        "prepack": 1,
        "adopt_target": True,
    }


def test_quantized_drafter_reuses_the_targets_lm_head_names():
    model = _quant_composite()
    quant = model.block_drafter_quant("int4")

    assert quant["bits"] == 4
    assert quant["block_size"] == 32
    assert quant["prepack"] == 1
    # Matching metadata lets the drafter adopt the target's exact quantized head during save.
    assert model.block_drafter_lm_head_quant() == {
        "bits": 4,
        "block_size": 32,
        "prepack": 1,
        "adopt_target": True,
    }


def test_structured_drafter_quantization_does_not_inherit_target_layout():
    quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(block_size=128),
        format=types.SimpleNamespace(matmulnbits_weights_prepacked=0),
    )

    quant = _quant_composite().block_drafter_quant("int4", quant_config)

    assert quant["block_size"] == 128
    assert quant["prepack"] == 0


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
    head_quant = model.block_drafter_lm_head_quant()

    assert head_bits == expected_bits
    if expected_bits == 4:
        assert head_quant["bits"] == expected_bits
    else:
        # The block-drafter quantizer cannot reproduce the target's Q8G initializer layout.
        assert head_quant is None


def test_dense_target_keeps_the_drafter_lm_head_dense():
    model = _quant_composite(weight_name=None, onnx_dtype=ir.DataType.FLOAT16)

    assert model.block_drafter_lm_head_quant() is None


# use_qdq makes the target write DequantizeLinear/MatMul over `*.weight_DQ_Q4`, so there is no
# MatMulNBits to adopt and asking for one would abort the export.
def test_qdq_target_keeps_the_drafter_lm_head_dense():
    model = _quant_composite(use_qdq=True)

    assert model.block_drafter_lm_head_quant() is None


def test_prepacked_bf16_target_keeps_a_private_raw_quantized_drafter_head():
    model = _quant_composite(io_dtype=ir.DataType.BFLOAT16)

    assert model.block_drafter_lm_head_quant() == {
        "bits": 4,
        "block_size": 32,
        "prepack": 0,
        "adopt_target": False,
    }


def test_non_cuda_target_ignores_requested_prepack_for_shared_drafter_head():
    model = _quant_composite(ep="webgpu")

    assert model.block_drafter_lm_head_quant() == {
        "bits": 4,
        "block_size": 32,
        "prepack": 0,
        "adopt_target": True,
    }


def _embed_quant_composite(**overrides):
    model = _quant_composite()
    model.decoder.quant_attrs["op_types_to_quantize"] = ["MatMul", "Gather"]
    for name, value in overrides.items():
        setattr(model.decoder, name, value)
    return model


def test_quantized_target_table_is_adopted_by_the_drafter():
    assert _embed_quant_composite().block_drafter_embed_quant() == {"bits": 4, "block_size": 32}


@pytest.mark.parametrize(
    "overrides",
    [
        # A dense target has no `model.embed_tokens.weight_Q4` to adopt.
        {"onnx_dtype": ir.DataType.FLOAT16},
        {"exclude_embeds": True},
        # `rtn`/`k_quant` name the table differently.
        {"quantization_algo": "k_quant"},
        # shared_embeddings gathers from a reshape of the LM head weight instead of its own table.
        {"tied_quantized_embeddings": True},
    ],
)
def test_an_unadoptable_target_table_leaves_the_drafter_embedding_dense(overrides):
    assert _embed_quant_composite(**overrides).block_drafter_embed_quant() is None


def test_private_quantized_drafter_head_is_not_shared(tmp_path):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant={"bits": 4, "block_size": 32, "prepack": 0, "adopt_target": False},
        adopt_target_tensors=lambda _target_model_path: None,
        save_model=lambda _output_dir: None,
    )
    captured = {}

    def share_initializers(*args, **kwargs):
        captured.update(kwargs)
        return []

    model.share_initializers = share_initializers

    model.save_dflash2_model(str(tmp_path))

    assert captured["adopt_source_initializers"] == frozenset()
    assert captured["required_source_initializers"] == frozenset()
    assert captured["excluded_source_initializers"] == {
        "lm_head.MatMul.weight_Q4",
        "lm_head.MatMul.weight_scales",
    }


def test_off_policy_keeps_embedding_and_head_private(tmp_path):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant=None,
        save_model=lambda _output_dir: None,
    )
    model.dflash2_shared_weight_policies = {"embedding": "off", "lm_head": "off"}
    captured = {}

    def share_initializers(*args, **kwargs):
        captured.update(kwargs)
        return []

    model.share_initializers = share_initializers

    model.save_dflash2_model(str(tmp_path))

    assert captured["excluded_source_initializers"] == {
        "model.embed_tokens.weight",
        "lm_head.MatMul.weight",
    }


def test_off_policy_uses_emitted_fp8_head_inventory(tmp_path):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant=None,
        graph=types.SimpleNamespace(
            initializers={
                "model.embed_tokens.weight": object(),
                "lm_head.MatMul.fp8_weight": object(),
                "lm_head.MatMul.fp8_weight_scale": object(),
            }
        ),
        save_model=lambda _output_dir: None,
    )
    model.dflash2_shared_weight_policies = {"embedding": "off", "lm_head": "off"}
    captured = {}

    def share_initializers(*args, **kwargs):
        captured.update(kwargs)
        return []

    model.share_initializers = share_initializers

    model.save_dflash2_model(str(tmp_path))

    assert captured["excluded_source_initializers"] == {
        "model.embed_tokens.weight",
        "lm_head.MatMul.fp8_weight",
        "lm_head.MatMul.fp8_weight_scale",
    }


def test_off_policy_suppresses_unshared_adopted_head_warning(tmp_path, capsys):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant={"bits": 4, "block_size": 32, "prepack": 0, "adopt_target": True},
        save_model=lambda _output_dir: None,
    )
    model.dflash2_shared_weight_policies = {"embedding": "auto", "lm_head": "off"}
    model.share_initializers = lambda *args, **kwargs: []

    model.save_dflash2_model(str(tmp_path))

    assert "may no longer agree" not in capsys.readouterr().out


def test_required_policy_requests_exact_target_adoption(tmp_path):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant=None,
        save_model=lambda _output_dir: None,
    )
    model.dflash2_shared_weight_policies = {"embedding": "required", "lm_head": "required"}
    required = {"model.embed_tokens.weight", "lm_head.MatMul.weight"}

    def share_initializers(*args, **kwargs):
        assert kwargs["adopt_source_initializers"] == required
        assert kwargs["required_source_initializers"] == required
        return [{"name": name} for name in required]

    model.share_initializers = share_initializers

    model.save_dflash2_model(str(tmp_path))


def test_required_policy_rejects_a_private_quantized_head(tmp_path):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant={"bits": 4, "block_size": 32, "prepack": 0, "adopt_target": False},
        save_model=lambda _output_dir: None,
    )
    model.dflash2_shared_weight_policies = {"embedding": "auto", "lm_head": "required"}

    with pytest.raises(ValueError, match="required LM-head sharing is incompatible"):
        model.save_dflash2_model(str(tmp_path))


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
    model = _quant_composite(**kwargs)
    quant = model.block_drafter_quant("int4")

    assert quant["bits"] == 4
    assert model.block_drafter_lm_head_quant() is None


def test_quantized_body_emits_matmulnbits_without_transposing(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 4, "block_size": 8, "prepack": 0},
    )

    builder.matmul("/probe/MatMul", "hidden_states", torch.ones((16, 8)), 8, 16, "num_block")

    node = next(node for node in builder.graph if node.name == "/probe/MatMul")
    assert node.op_type == "MatMulNBits"
    assert node.domain == "com.microsoft"
    assert node.attributes["K"].value == 8
    assert node.attributes["N"].value == 16
    # MatMulNBits takes [N, K], so the dense path's transpose must not be applied.
    assert tuple(builder.graph.initializers["probe.MatMul.weight_Q4"].const_value.shape) == (16, 1, 4)


def test_quantized_body_uses_ort_tie_breaking(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 4, "block_size": 32, "prepack": 0},
    )
    values = torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125] * 4])

    builder.matmul("/probe/MatMul", "hidden_states", values, 32, 1, "num_block")

    qweight = np.asarray(builder.graph.initializers["probe.MatMul.weight_Q4"].const_value)
    scales = np.asarray(builder.graph.initializers["probe.MatMul.weight_scales"].const_value)
    np.testing.assert_array_equal(qweight.reshape(-1), [15, 76, 106, 121] * 4)
    np.testing.assert_array_equal(scales, [[0.125]])


# The prepacked fpA_intB kernel takes FP16 activations only, so the bf16 body must ship the
# plain blockwise layout even though the target it drafts for is prepacked.
def test_bf16_body_never_prepacks_even_when_the_target_does(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 4, "block_size": 8, "prepack": 1},
    )

    builder.matmul("/probe/MatMul", "hidden_states", torch.ones((16, 8)), 8, 16, "num_block")

    node = next(node for node in builder.graph if node.name == "/probe/MatMul")
    assert builder.io_dtype == ir.DataType.BFLOAT16
    assert "weight_prepacked" not in node.attributes


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("fuse_gate_up", [False, True])
def test_mlp_gate_up_fusion_preserves_weight_rows(tmp_path, bits, fuse_gate_up):
    quant = {"bits": bits, "block_size": 8, "prepack": 0} if bits else None
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.BFLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant=quant,
        fuse_gate_up=fuse_gate_up,
    )
    generator = torch.Generator().manual_seed(42)
    weights = {
        f"layers.0.mlp.{projection}.weight": torch.randn(shape, generator=generator).to(torch.bfloat16)
        for projection, shape in (
            ("gate_proj", (16, 8)),
            ("up_proj", (16, 8)),
            ("down_proj", (8, 16)),
        )
    }

    builder._make_mlp(0, "hidden_states", weights, "num_block")

    projections = [node for node in builder.graph if node.op_type in ("MatMul", "MatMulNBits")]
    assert len(projections) == (2 if fuse_gate_up else 3)
    if not fuse_gate_up:
        assert not any(node.op_type == "Split" for node in builder.graph)
        assert projections[0].name == "/dflash2/layers.0/mlp/gate_proj/MatMul"
        assert projections[1].name == "/dflash2/layers.0/mlp/up_proj/MatMul"
        return

    projection = next(node for node in builder.graph if node.name == "/dflash2/layers.0/mlp/gate_up_proj/MatMul")
    assert projection.op_type == ("MatMulNBits" if bits else "MatMul")
    assert "weight_prepacked" not in projection.attributes
    split = next(node for node in builder.graph if node.op_type == "Split")
    assert split.inputs[0] is projection.outputs[0]
    assert split.attributes["axis"].value == -1
    assert all(output.shape == ir.Shape(["num_block", 16]) for output in split.outputs)

    reference = DFlash2Builder(
        builder.draft_dir,
        str(tmp_path),
        ir.DataType.BFLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant=quant,
    )
    for name in ("gate_proj", "up_proj"):
        reference.matmul(f"/{name}/MatMul", "hidden_states", weights[f"layers.0.mlp.{name}.weight"], 8, 16, "num_block")
    reference_nodes = list(reference.graph)
    for input_index in range(1, len(projection.inputs)):
        combined = projection.inputs[input_index].const_value.numpy()
        separate = [node.inputs[input_index].const_value.numpy() for node in reference_nodes]
        axis = 0 if bits else 1
        np.testing.assert_array_equal(combined, np.concatenate(separate, axis=axis))


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_mlp_gate_up_fusion_execution_matches_unfused(tmp_path, bits):
    draft_dir = _draft_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(123)
    weights = {
        f"layers.0.mlp.{projection}.weight": torch.randn(shape, generator=generator) / shape[1] ** 0.5
        for projection, shape in (
            ("gate_proj", (64, 32)),
            ("up_proj", (64, 32)),
            ("down_proj", (32, 64)),
        )
    }
    sessions = []
    for fused in (False, True):
        builder = DFlash2Builder(
            draft_dir,
            str(tmp_path),
            ir.DataType.FLOAT,
            paged_block_size=256,
            max_position_embeddings=128,
            quant={"bits": bits, "block_size": 32, "prepack": 0} if bits else None,
            fuse_gate_up=fused,
        )
        builder.io_dtype = ir.DataType.FLOAT
        builder.hidden_size = 32
        builder.intermediate_size = 64
        builder.graph.inputs.append(builder.make_value("hidden_states", ir.DataType.FLOAT, ["num_block", 32]))
        output = builder._make_mlp(0, "hidden_states", weights, "num_block")
        builder.graph.outputs.append(builder.values[output])
        model = ir.serde.serialize_model(builder.model)
        onnx.checker.check_model(model)
        sessions.append(ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]))

    for rows in (1, 8, 16, 32):
        inputs = {"hidden_states": torch.randn((rows, 32), generator=generator).numpy()}
        expected = sessions[0].run(None, inputs)[0]
        actual = sessions[1].run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _quantized_head_builder(tmp_path, bits=4, block_size=8):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={
            "bits": bits,
            "block_size": block_size,
            "prepack": 0,
        },
        lm_head_quant={"bits": bits, "block_size": block_size, "prepack": 0, "adopt_target": True},
    )
    builder.weights = {"lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size))}
    return builder


def _save_target(out_dir, node, initializers, hidden_size, vocab_size):
    graph = onnx.helper.make_graph(
        [node],
        "target",
        [onnx.helper.make_tensor_value_info("hidden_states", onnx.TensorProto.FLOAT16, ["rows", hidden_size])],
        [onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT16, ["rows", vocab_size])],
        [onnx.numpy_helper.from_array(array, name) for name, array in initializers.items()],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)],
    )
    path = os.path.join(out_dir, "model.onnx")
    onnx.save_model(model, path, save_as_external_data=True, location="model.onnx.data", size_threshold=0)
    return path


def _save_quantized_target(out_dir, builder, bits=4, block_size=8, weight_prepacked=2):
    columns = builder.hidden_size // block_size
    qweight = np.arange(builder.vocab_size * columns * block_size * bits // 8, dtype=np.uint8).reshape(
        builder.vocab_size, columns, block_size * bits // 8
    )
    scales = np.arange(builder.vocab_size * columns, dtype=np.float16).reshape(builder.vocab_size, columns)
    initializers = {f"lm_head.MatMul.weight_Q{bits}": qweight, "lm_head.MatMul.weight_scales": scales}
    node = onnx.helper.make_node(
        "MatMulNBits",
        ["hidden_states", *initializers],
        ["logits"],
        # The quantizer renames the target's node, so the drafter cannot find it by name.
        name="/lm_head/MatMul_Q4",
        domain="com.microsoft",
        bits=bits,
        block_size=block_size,
        K=builder.hidden_size,
        N=builder.vocab_size,
        weight_prepacked=weight_prepacked,
    )
    path = _save_target(out_dir, node, initializers, builder.hidden_size, builder.vocab_size)
    return path, qweight, scales


def test_quantized_lm_head_matches_the_targets_initializer_names(tmp_path):
    builder = _quantized_head_builder(tmp_path)

    output = builder.make_lm_head("hidden_states", "num_sample")

    node = next(node for node in builder.graph if node.name == "/lm_head/MatMul")
    assert node.op_type == "MatMulNBits"
    assert [value.name for value in node.inputs][1:] == [
        "lm_head.MatMul.weight_Q4",
        "lm_head.MatMul.weight_scales",
    ]
    # The weights are the target's, so nothing is quantized or registered until adoption.
    assert "lm_head.MatMul.weight_Q4" not in builder.graph.initializers
    assert builder.values[output].dtype == ir.DataType.FLOAT16


def test_quantized_lm_head_adopts_the_targets_bytes_and_attributes(tmp_path):
    builder = _quantized_head_builder(tmp_path)
    builder.make_lm_head("hidden_states", "num_sample")
    target_path, qweight, scales = _save_quantized_target(tmp_path, builder)

    builder.adopt_target_tensors(target_path)

    initializers = builder.graph.initializers
    np.testing.assert_array_equal(initializers["lm_head.MatMul.weight_Q4"].const_value.numpy(), qweight)
    np.testing.assert_array_equal(initializers["lm_head.MatMul.weight_scales"].const_value.numpy(), scales)
    node = next(node for node in builder.graph if node.name == "/lm_head/MatMul")
    # The target's layout decision comes across with its bytes rather than being recomputed.
    assert node.attributes["weight_prepacked"].value == 2
    assert node.attributes["K"].value == builder.hidden_size
    assert node.attributes["N"].value == builder.vocab_size
    assert builder.lm_head_adoption is None


def test_adopted_lm_head_survives_a_round_trip_to_disk(tmp_path):
    builder = _quantized_head_builder(tmp_path)
    builder.make_lm_head("hidden_states", "num_sample")
    builder.graph.outputs.append(builder.values[builder.out("/lm_head/MatMul")])
    builder.graph.inputs.append(builder.make_value("hidden_states", ir.DataType.FLOAT16, ["rows", builder.hidden_size]))
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_path, qweight, _ = _save_quantized_target(target_dir, builder)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    builder.adopt_target_tensors(target_path)
    builder.save_model(str(out_dir))

    saved = onnx.load(str(out_dir / builder.filename))
    initializer = next(init for init in saved.graph.initializer if init.name == "lm_head.MatMul.weight_Q4")
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(initializer, base_dir=str(out_dir)), qweight)


def test_saving_before_adoption_is_rejected(tmp_path):
    builder = _quantized_head_builder(tmp_path)
    builder.make_lm_head("hidden_states", "num_sample")

    with pytest.raises(ValueError, match="adopt_target_tensors"):
        builder.save_model(str(tmp_path))


def _quantized_embedding_target(out_dir, builder, bits=4, block_size=64):
    columns = builder.hidden_size // block_size
    qweight = ir.tensor(
        np.arange(builder.vocab_size * builder.hidden_size, dtype=np.uint8).reshape(
            builder.vocab_size, builder.hidden_size
        )
        % 16,
        dtype=ir.DataType.INT4,
        name=f"model.embed_tokens.weight_Q{bits}",
    )
    scales = np.arange(builder.vocab_size * columns, dtype=np.float16).reshape(builder.vocab_size, columns)
    node = onnx.helper.make_node(
        "GatherBlockQuantized",
        [f"model.embed_tokens.weight_Q{bits}", "input_ids", "model.embed_tokens.weight_scales"],
        ["embeddings"],
        # The quantizer renames the target's node, so the drafter cannot find it by name.
        name="/model/embed_tokens/Gather_Q4",
        domain="com.microsoft",
        block_size=block_size,
        gather_axis=0,
        quantize_axis=1,
    )
    graph = onnx.helper.make_graph(
        [node],
        "target",
        [onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["rows"])],
        [onnx.helper.make_tensor_value_info("embeddings", onnx.TensorProto.FLOAT16, ["rows", builder.hidden_size])],
        [
            ir.to_proto(qweight),
            onnx.numpy_helper.from_array(scales, "model.embed_tokens.weight_scales"),
        ],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)],
    )
    path = os.path.join(out_dir, "model.onnx")
    onnx.save_model(model, path, save_as_external_data=True, location="model.onnx.data", size_threshold=0)
    return path, qweight.numpy(), scales


def _quantized_embedding_builder(tmp_path, block_size=64):
    builder = _quantized_head_builder(tmp_path)
    builder.embed_quant = {"bits": 4, "block_size": block_size}
    builder.weights["embed_tokens.weight"] = torch.ones((builder.vocab_size, builder.hidden_size))
    return builder


def test_a_dense_target_leaves_the_drafter_embedding_dense(tmp_path):
    builder = _quantized_head_builder(tmp_path)
    builder.weights["embed_tokens.weight"] = torch.ones((builder.vocab_size, builder.hidden_size))

    builder.make_embedding("/dflash2/embed_tokens/Gather", "num_sample")

    node = next(node for node in builder.graph if node.name == "/dflash2/embed_tokens/Gather")
    assert node.op_type == "Gather"
    assert "model.embed_tokens.weight" in builder.graph.initializers
    assert builder.embed_adoption is None


def test_quantized_embedding_matches_the_targets_initializer_names(tmp_path):
    builder = _quantized_embedding_builder(tmp_path)

    output = builder.make_embedding("/dflash2/embed_tokens/Gather", "num_sample")

    node = next(node for node in builder.graph if node.name == "/dflash2/embed_tokens/Gather")
    assert node.op_type == "GatherBlockQuantized"
    assert [value.name for value in node.inputs] == [
        "model.embed_tokens.weight_Q4",
        "input_ids",
        "model.embed_tokens.weight_scales",
    ]
    # The table is the target's, so nothing is quantized or registered until adoption.
    assert "model.embed_tokens.weight_Q4" not in builder.graph.initializers
    assert builder.values[output].dtype == ir.DataType.FLOAT16


def test_quantized_embedding_adopts_the_targets_bytes_and_attributes(tmp_path):
    builder = _quantized_embedding_builder(tmp_path, block_size=8)
    builder.make_embedding("/dflash2/embed_tokens/Gather", "num_sample")
    target_path, qweight, scales = _quantized_embedding_target(tmp_path, builder, block_size=64)

    builder.adopt_target_tensors(target_path)

    initializers = builder.graph.initializers
    np.testing.assert_array_equal(initializers["model.embed_tokens.weight_Q4"].const_value.numpy(), qweight)
    np.testing.assert_array_equal(initializers["model.embed_tokens.weight_scales"].const_value.numpy(), scales)
    node = next(node for node in builder.graph if node.name == "/dflash2/embed_tokens/Gather")
    # The target's block size comes across with its bytes rather than being recomputed.
    assert node.attributes["block_size"].value == 64
    assert node.attributes["gather_axis"].value == 0
    assert builder.embed_adoption is None


def test_saving_before_embedding_adoption_is_rejected(tmp_path):
    builder = _quantized_embedding_builder(tmp_path)
    builder.make_embedding("/dflash2/embed_tokens/Gather", "num_sample")

    with pytest.raises(ValueError, match="adopt_target_tensors"):
        builder.save_model(str(tmp_path))


def test_a_target_embedding_the_drafter_cannot_adopt_is_rejected(tmp_path):
    builder = _quantized_embedding_builder(tmp_path)
    builder.make_embedding("/dflash2/embed_tokens/Gather", "num_sample")
    initializers = {"model.embed_tokens.weight": np.ones((builder.vocab_size, builder.hidden_size), dtype=np.float16)}
    node = onnx.helper.make_node(
        "Gather", ["model.embed_tokens.weight", "hidden_states"], ["logits"], name="/model/embed_tokens/Gather"
    )
    target_path = _save_target(tmp_path, node, initializers, builder.hidden_size, builder.vocab_size)

    with pytest.raises(ValueError, match="different input space"):
        builder.adopt_target_tensors(target_path)


def test_a_target_head_the_drafter_cannot_adopt_is_rejected(tmp_path):
    builder = _quantized_head_builder(tmp_path)
    builder.make_lm_head("hidden_states", "num_sample")
    initializers = {
        "lm_head.MatMul.fp8_weight": np.zeros((builder.vocab_size, builder.hidden_size), dtype=np.uint8),
        "lm_head.MatMul.fp8_weight_scale": np.ones((builder.vocab_size, 1), dtype=np.float32),
    }
    node = onnx.helper.make_node(
        "MatMulBlockQuantizedFp8Weight",
        ["hidden_states", *initializers],
        ["logits"],
        name="/lm_head/MatMul",
        domain="com.microsoft",
        block_size=builder.hidden_size,
    )
    target_path = _save_target(tmp_path, node, initializers, builder.hidden_size, builder.vocab_size)

    with pytest.raises(ValueError, match="reject nearly every draft"):
        builder.adopt_target_tensors(target_path)


# A prequantized head overrides `--precision`, and the drafter has to follow the target there:
# scoring drafts with a head the target does not verify with collapses acceptance.
def test_prequantized_fp8_target_head_overrides_the_requested_precision(tmp_path):
    builder = _quantized_head_builder(tmp_path)
    builder.weights = {
        "lm_head.weight": torch.ones((builder.vocab_size, builder.hidden_size), dtype=torch.float8_e4m3fn),
        "lm_head.weight_scale": torch.ones(()),
    }

    builder.make_lm_head("hidden_states", "num_sample")

    node = next(node for node in builder.graph if node.name == "/lm_head/MatMul")
    assert node.op_type == "MatMulBlockQuantizedFp8Weight"
    assert builder.lm_head_quant is None
    assert builder.lm_head_adoption is None


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


@pytest.mark.parametrize("fuse_gate_up", [False, True, "true", "false"])
def test_drafter_uses_target_context_length(tmp_path, monkeypatch, fuse_gate_up):
    captured = {}

    class StubDFlash2Builder:
        def __init__(self, _draft_dir, _target_dir, _io_dtype, _paged_block_size, max_position, **_kwargs):
            captured["max_position"] = max_position
            captured["fuse_gate_up"] = _kwargs["fuse_gate_up"]

        def make_model(self):
            pass

    dflash2_module = importlib.import_module("models.builders.dflash2")
    monkeypatch.setattr(dflash2_module, "DFlash2Builder", StubDFlash2Builder)
    model = _composite()
    model.make_dflash2_init(
        io_dtype=None,
        extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_fuse_gate_up": fuse_gate_up},
    )

    model.make_dflash2_model(str(tmp_path))

    assert captured["max_position"] == model.decoder.context_length
    assert captured["fuse_gate_up"] is (str(fuse_gate_up).lower() == "true")


def test_gate_up_fusion_defaults_off(tmp_path):
    model = _composite()
    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})

    assert model.dflash2_attrs["fuse_gate_up"] is False
    builder = DFlash2Builder(model.dflash2_path, str(tmp_path), ir.DataType.BFLOAT16, 256, 128)
    assert builder.mlp_attrs["fuse_gate_up"] is False


@pytest.mark.parametrize("value", ["yes", "", 1, None])
def test_gate_up_fusion_rejects_invalid_option(tmp_path, value):
    model = _composite()
    with pytest.raises(ValueError, match="dflash2_fuse_gate_up must be true or false"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_fuse_gate_up": value},
        )


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
