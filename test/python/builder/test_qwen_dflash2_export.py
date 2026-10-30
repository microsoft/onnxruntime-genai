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
import transformers
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    DefaultWeightOnlyQuantizer,
)
from safetensors.torch import save_file

from models.builders.base import Model
from models.builders.block_drafter import BlockDrafterBuilder
from models.builders.dflash2 import DFlash2Builder
from models.builders.mtp import MTPModel
from models.builders.qwen import Qwen35MoEModel
from models.quantization.quant_config import QuantConfig

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


def _composite(aux_layers=AUX_LAYERS, use_paged_attention=True, ep="cuda"):
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
        ep=ep,
        model=ir.Model(ir.Graph([], [], nodes=[]), ir_version=10),
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


@pytest.mark.parametrize(
    ("sliding_window", "prefix_caching"),
    [(0, False), (2048, True)],
)
def test_genai_config_disables_prefix_caching_only_for_full_attention_drafter(tmp_path, sliding_window, prefix_caching):
    config_path = tmp_path / "genai_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"decoder": {}},
                "engine": {"dynamic_batching": {}},
            }
        )
    )
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        genai_config_section=lambda: {
            "filename": "dflash2.onnx",
            "sliding_window": sliding_window,
        }
    )

    model.add_dflash2_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["engine"]["dynamic_batching"].get("prefix_caching", True) is prefix_caching


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_genai_config_preserves_explicit_prefix_caching_for_full_attention_drafter(tmp_path, prefix_caching):
    config_path = tmp_path / "genai_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"decoder": {}},
                "engine": {"dynamic_batching": {"prefix_caching": prefix_caching}},
            }
        )
    )
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        genai_config_section=lambda: {
            "filename": "dflash2.onnx",
            "sliding_window": 0,
        }
    )

    model.add_dflash2_to_genai_config(str(tmp_path))

    config = json.loads(config_path.read_text())
    assert config["engine"]["dynamic_batching"]["prefix_caching"] is prefix_caching


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


@pytest.mark.parametrize("provider", ["cpu", "cuda", "webgpu", "dml", "trt-rtx"])
def test_target_mlp_gate_up_fusion_preserves_projection_order(provider):
    model = _mlp_graph_model(provider=provider)
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.full((2, 3), 1.0), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.full((2, 3), 2.0), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )
    model.make_mlp(4, mlp, "residual")

    fused = next(node for node in model.graph if node.name == "/model/layers.4/mlp/gate_up_proj/MatMul")
    assert fused.inputs[0].name == "residual"
    np.testing.assert_array_equal(
        fused.inputs[1].const_value.numpy(), torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)).T
    )
    split = next(node for node in model.graph if node.op_type == "Split")
    assert [value.name for value in split.outputs] == [
        "/model/layers.4/mlp/gate_proj/MatMul/output_0",
        "/model/layers.4/mlp/up_proj/MatMul/output_0",
    ]
    assert model.mlp_attrs["output_0"] == "/model/layers.4/mlp/down_proj/MatMul/output_0"


def _mlp_graph_model(provider="cpu", dtype=ir.DataType.FLOAT):
    model = object.__new__(Model)
    model.ep = provider
    model.io_dtype = dtype
    model.onnx_dtype = dtype
    model.hidden_size = 3
    model.intermediate_size = 2
    model.use_paged_attention = False
    model.activation = "silu"
    model.mlp_attrs = {"use_proj": True, "output_0": ""}
    model.layernorm_attrs = {}
    model.quant_attrs = {"nodes_to_exclude": [], "accuracy_level": 4, "use_qdq": False}
    model.values = {}
    model.node_names = set()
    model.graph = ir.Graph([], [], nodes=[], opset_imports={"": 21, "com.microsoft": 1})
    model.model = ir.Model(model.graph, ir_version=10)
    model.graph.inputs.append(model.make_value("residual", dtype, [1, 2, 3]))
    return model


@pytest.mark.parametrize("incompatibility", ["adapter", "packed", "integer", "exclusion", "override", "precision"])
def test_automatic_gate_up_fusion_preserves_unsupported_projection_paths(monkeypatch, incompatibility):
    model = object.__new__(Model)
    model.mlp_attrs = {"use_proj": True}
    model.layernorm_attrs = {}
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones(2, 3), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones(2, 3), bias=None),
    )
    gate_name = "/model/layers.0/mlp/gate_proj/MatMul"
    if incompatibility == "adapter":
        mlp.gate_proj.base_layer = object()
    elif incompatibility == "packed":
        mlp.gate_proj.quant_type = "fp8"
    elif incompatibility == "integer":
        mlp.gate_proj.weight = torch.ones(2, 3, dtype=torch.int8)
    elif incompatibility == "exclusion":
        model.quant_attrs = {"nodes_to_exclude": [gate_name]}
    elif incompatibility == "override":
        model.exact_quant_override_names = {gate_name}
    elif incompatibility == "precision":
        model.int4_customized_weight_config = {gate_name: {"bits": 8}}
    calls = []

    def unfused(layer_id, weights, root_input):
        calls.append((layer_id, weights, root_input))
        model.mlp_attrs["output_0"] = "unfused_output"

    monkeypatch.setattr(model, "make_mlp_proj", unfused)
    model.make_mlp(0, mlp, "hidden")
    assert calls == [(0, mlp, "hidden")]
    assert model.layernorm_attrs["skip_input"] == "unfused_output"


def test_target_mlp_gate_up_graph_fusion_preserves_a_missing_bias():
    model = _mlp_graph_model()
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=torch.ones(2)),
        up_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )
    model.make_mlp(0, mlp, "residual")
    assert sum(node.op_type == "MatMul" for node in model.graph) == 2
    assert sum(node.op_type == "Add" for node in model.graph) == 1
    model.graph.outputs.append(model.values[model.mlp_attrs["output_0"]])
    hidden = torch.randn(1, 2, 3)
    expected = (
        torch.nn.functional.silu(hidden @ mlp.gate_proj.weight.T + 1) * (hidden @ mlp.up_proj.weight.T)
    ) @ mlp.down_proj.weight.T
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        session.run(None, {"residual": hidden.numpy()})[0], expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_target_mlp_gate_up_fusion_preserves_projection_shape_mismatch():
    model = _mlp_graph_model()
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones((3, 3)), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )
    model.make_mlp(0, mlp, "residual")
    assert not any(node.op_type == "Split" for node in model.graph)


def test_target_mlp_gate_up_fusion_maps_matching_exclusions():
    model = _mlp_graph_model()
    model.quant_attrs["nodes_to_exclude"] = [
        "/model/layers.0/mlp/gate_proj/MatMul",
        "/model/layers.0/mlp/up_proj/MatMul",
    ]
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )

    model.make_mlp(0, mlp, "residual")

    assert "/model/layers.0/mlp/gate_up_proj/MatMul" in model.quant_attrs["nodes_to_exclude"]


def test_target_mlp_gate_up_fusion_preserves_one_sided_exclusion():
    model = _mlp_graph_model()
    model.quant_attrs["nodes_to_exclude"] = ["/model/layers.0/mlp/gate_proj/MatMul"]
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones((2, 3)), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones((3, 2)), bias=None),
    )
    model.make_mlp(0, mlp, "residual")
    assert sum(node.op_type == "MatMul" for node in model.graph) == 3
    assert not any(node.op_type == "Split" for node in model.graph)


@pytest.mark.parametrize("bits,with_g_idx", [(4, False), (8, False), (4, True)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_prequantized_gate_up_graph_fusion_matches_separate_projections(bits, symmetric, with_g_idx):
    generator = torch.Generator().manual_seed(7)
    quantizer = DefaultWeightOnlyQuantizer(
        DefaultWeightOnlyQuantConfig(block_size=32, bits=bits, is_symmetric=symmetric)
    )

    def projection():
        weight = torch.randn(16, 64, generator=generator)
        qweight, scales, zeros = quantizer.qbits_block_quant(weight.T.numpy())
        if symmetric and with_g_idx:
            zeros.fill(0x88 if bits == 4 else 128)
        return types.SimpleNamespace(
            qweight=torch.from_numpy(qweight),
            scales=torch.from_numpy(scales),
            qzeros=torch.from_numpy(zeros) if not symmetric or with_g_idx else None,
            g_idx=torch.arange(64, dtype=torch.int32) // 32 if with_g_idx else None,
            bits=bits,
            group_size=32,
            in_features=64,
            out_features=16,
            bias=None,
        )

    mlp = types.SimpleNamespace(
        gate_proj=projection(),
        up_proj=projection(),
        down_proj=types.SimpleNamespace(weight=torch.randn(64, 16, generator=generator), bias=None),
    )
    hidden = torch.randn(1, 2, 64, generator=generator).numpy()
    outputs = []
    for fused in (False, True):
        model = _mlp_graph_model()
        model.onnx_dtype = ir.DataType.INT4 if bits == 4 else ir.DataType.INT8
        model.hidden_size = 64
        model.intermediate_size = 16
        model.values["residual"].shape = ir.Shape([1, 2, 64])
        if not fused:
            model.exact_quant_override_names = {"/model/layers.0/mlp/gate_proj/MatMul"}
        model.make_mlp(0, mlp, "residual")
        model.graph.outputs.append(model.values[model.mlp_attrs["output_0"]])
        nodes = [node for node in model.graph if node.op_type == "MatMulNBits"]
        assert len(nodes) == (1 if fused else 2)
        if fused:
            assert nodes[0].attributes["N"].as_int() == 32
        session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
        outputs.append(session.run(None, {"residual": hidden})[0])
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "policy,fuses",
    [
        ("matching", True),
        ("fused", True),
        ("different", False),
        ("exact", False),
        ("excluded", False),
        ("conflict", False),
    ],
)
def test_mlp_graph_fusion_preserves_precision_policy(policy, fuses):
    model = _mlp_graph_model()
    gate = "/model/layers.0/mlp/gate_proj/MatMul"
    up = "/model/layers.0/mlp/up_proj/MatMul"
    fused = "/model/layers.0/mlp/gate_up_proj/MatMul"
    model.int4_customized_weight_config = {gate: {"bits": 8}, up: {"bits": 8}}
    if policy == "different":
        model.int4_customized_weight_config[up] = {"bits": 4}
    elif policy == "exact":
        model.exact_quant_override_names = {gate}
    elif policy == "excluded":
        model.quant_attrs["nodes_to_exclude"] = [gate]
    elif policy == "conflict":
        model.int4_customized_weight_config[fused] = {"bits": 4}
    elif policy == "fused":
        model.int4_customized_weight_config = {fused: {"bits": 8}}
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones(2, 3), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.ones(2, 3), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones(3, 2), bias=None),
    )
    model.make_mlp(0, mlp, "residual")
    assert any(node.op_type == "Split" for node in model.graph) is fuses
    if fuses:
        assert model.int4_customized_weight_config[fused] == {"bits": 8}
    else:
        assert all(any(node.name == name for node in model.graph) for name in (gate, up))


@pytest.mark.parametrize("incompatibility", ["bits", "g_idx", "prepacked", "exact", "exclusion"])
def test_prequantized_gate_up_graph_fusion_preserves_unsupported_nodes(incompatibility):
    model = _mlp_graph_model()
    model.onnx_dtype = ir.DataType.INT4
    model.hidden_size = 64
    model.values["residual"].shape = ir.Shape([1, 2, 64])
    names = []
    for kind in ("gate", "up"):
        bits = 8 if incompatibility == "bits" and kind == "up" else 4
        projection = types.SimpleNamespace(
            qweight=torch.full((2, 2, 32 * bits // 8), 0x99, dtype=torch.uint8),
            scales=torch.ones(2, 2),
            qzeros=torch.full((2, (2 * bits + 7) // 8), 0x88, dtype=torch.uint8),
            g_idx=(torch.arange(64, dtype=torch.int32) // 32) if incompatibility == "g_idx" else None,
            bits=bits,
            group_size=32,
            in_features=64,
            out_features=2,
        )
        if incompatibility == "g_idx" and kind == "up":
            projection.g_idx = projection.g_idx.flip(0)
        name = model.make_matmul(projection, f"/model/layers.0/mlp/{kind}_proj/MatMul", "residual")
        names.append(name)
        if incompatibility == "prepacked":
            model.values[f"{name}/output_0"].producer().attributes["weight_prepacked"] = ir.AttrInt64(
                "weight_prepacked", 1
            )
    if incompatibility == "exact":
        model.exact_quant_override_names = {names[0]}
    elif incompatibility == "exclusion":
        model.quant_attrs["nodes_to_exclude"] = [names[0]]
    model.make_mlp_proj_fused(0, *names)
    assert [node.name for node in model.graph] == names


def test_mlp_graph_fusion_preserves_real_adapter_path():
    model = _mlp_graph_model()
    generator = torch.Generator().manual_seed(13)
    gate_weight = torch.randn(2, 3, generator=generator)
    lora_a = torch.randn(1, 3, generator=generator)
    lora_b = torch.randn(2, 1, generator=generator)
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(
            base_layer=types.SimpleNamespace(weight=gate_weight, bias=None),
            lora_A=types.SimpleNamespace(default=types.SimpleNamespace(weight=lora_a)),
            lora_B=types.SimpleNamespace(default=types.SimpleNamespace(weight=lora_b.clone())),
            scaling={"default": 0.5},
            bias=None,
        ),
        up_proj=types.SimpleNamespace(weight=torch.randn(2, 3, generator=generator), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.randn(3, 2, generator=generator), bias=None),
    )
    model.make_mlp(0, mlp, "residual")
    assert not any(node.op_type == "Split" for node in model.graph)
    model.graph.outputs.append(model.values[model.mlp_attrs["output_0"]])
    hidden = torch.randn(1, 2, 3, generator=generator)
    gate = hidden @ gate_weight.T + (hidden @ lora_a.T) @ lora_b.T * 0.5
    expected = (torch.nn.functional.silu(gate) * (hidden @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        session.run(None, {"residual": hidden.numpy()})[0], expected.numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_mlp_graph_fusion_preserves_initializer_dtype(dtype):
    model = _mlp_graph_model(dtype=dtype)
    mlp = types.SimpleNamespace(
        gate_proj=types.SimpleNamespace(weight=torch.ones(2, 3), bias=None),
        up_proj=types.SimpleNamespace(weight=torch.full((2, 3), 2.0), bias=None),
        down_proj=types.SimpleNamespace(weight=torch.ones(3, 2), bias=None),
    )
    model.make_mlp(0, mlp, "residual")
    fused = next(node for node in model.graph if node.name == "/model/layers.0/mlp/gate_up_proj/MatMul")
    assert fused.inputs[1].dtype == dtype
    proto = ir.to_proto(model.model)
    assert next(tensor for tensor in proto.graph.initializer if tensor.name == fused.inputs[1].name).data_type == dtype


@pytest.mark.parametrize("matching", [False, True])
def test_mlp_graph_fusion_preserves_real_per_node_quantization(matching):
    model = _mlp_graph_model()
    model.hidden_size = model.intermediate_size = 64
    model.values["residual"].shape = ir.Shape([1, 2, 64])
    model.onnx_dtype = ir.DataType.INT4
    model.quant_attrs.update(bits=4, matmul_block_size=32, is_symmetric=True, op_types_to_quantize=("MatMul",))
    model.matmul_attrs = {"weights_prepacked": 0}
    gate = "/model/layers.0/mlp/gate_proj/MatMul"
    up = "/model/layers.0/mlp/up_proj/MatMul"
    model.int4_customized_weight_config = {gate: {"bits": 8}}
    if matching:
        model.int4_customized_weight_config[up] = {"bits": 8}
    generator = torch.Generator().manual_seed(23)
    mlp = types.SimpleNamespace(
        **{
            f"{kind}_proj": types.SimpleNamespace(weight=torch.randn(64, 64, generator=generator), bias=None)
            for kind in ("gate", "up", "down")
        }
    )
    model.make_mlp(0, mlp, "residual")
    model.graph.outputs.append(model.values[model.mlp_attrs["output_0"]])
    quantized = model.to_nbits()
    nodes = {node.name: node for node in quantized.graph if node.op_type == "MatMulNBits"}
    expected = {"/model/layers.0/mlp/gate_up_proj/MatMul_Q8": 8} if matching else {f"{gate}_Q8": 8, f"{up}_Q4": 4}
    expected["/model/layers.0/mlp/down_proj/MatMul_Q4"] = 4
    assert {name: node.attributes["bits"].as_int() for name, node in nodes.items()} == expected
    session = ort.InferenceSession(ir.to_proto(quantized).SerializeToString(), providers=["CPUExecutionProvider"])
    result = session.run(None, {"residual": torch.randn(1, 2, 64, generator=generator).numpy()})[0]
    assert result.shape == (1, 2, 64)
    assert np.isfinite(result).all()


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


def test_legacy_exclusion_does_not_require_an_emitted_node(tmp_path):
    model, _ = _exact_override_model(tmp_path, constant_weight=True)
    missing_name = "/missing/MatMul"
    model.quant_config = QuantConfig.from_extra_options(
        {"nodes_to_exclude": [missing_name]}, precision="int4", execution_provider="cpu"
    )
    model.quant_type = None

    model.make_quant_init(types.SimpleNamespace())
    quantized = model.to_nbits()

    assert any(node.name == "/probe/MatMul_Q4" for node in quantized.graph)


def test_structured_exact_exclusion_requires_an_emitted_node(tmp_path):
    model, _ = _exact_override_model(tmp_path, constant_weight=True)
    missing_name = "/missing/MatMul"
    model.quant_config = QuantConfig.from_dict(
        {
            "weights": {
                "type": "int4",
                "overrides": [{"match": {"name": missing_name}, "exclude": True}],
            }
        }
    )
    model.quant_type = None

    model.make_quant_init(types.SimpleNamespace())
    with pytest.raises(ValueError, match="did not match an emitted node"):
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


def test_webgpu_drafter_omits_attention_metadata(tmp_path):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        ep="webgpu",
    )

    builder.declare_io()

    assert "attention_metadata" not in builder.values
    assert builder.genai_config_section()["inputs"]["attention_metadata"] == ""


def make_precision_checkpoint(tmp_path, num_layers):
    """Write a small checkpoint so graph tests exercise the real loader and quantizer."""
    draft_dir = _draft_checkpoint(tmp_path)
    config_path = os.path.join(draft_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config.update(hidden_size=32, intermediate_size=64, head_dim=16, num_hidden_layers=num_layers)
    with open(config_path, "w") as f:
        json.dump(config, f)
    generator = torch.Generator().manual_seed(7)

    def weight(*shape):
        return torch.randn(shape, generator=generator) * 0.01

    weights = {
        "fc.weight": weight(32, 96),
        "hidden_norm.weight": torch.ones(32),
        "norm.weight": torch.ones(32),
        "candidate_selector.hidden_projection.weight": weight(4, 32),
        "candidate_selector.predecessor_codebook": weight(32, 4),
        "candidate_selector.successor_codebook": weight(32, 4),
    }
    for layer in range(num_layers):
        prefix = f"layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{prefix}.{norm}.weight"] = torch.ones(32)
        for name, shape in (
            ("self_attn.q_proj", (32, 32)),
            ("self_attn.k_proj", (16, 32)),
            ("self_attn.v_proj", (16, 32)),
            ("self_attn.o_proj", (32, 32)),
            ("mlp.gate_proj", (64, 32)),
            ("mlp.up_proj", (64, 32)),
            ("mlp.down_proj", (32, 64)),
        ):
            weights[f"{prefix}.{name}.weight"] = weight(*shape)
        for norm in ("q_norm", "k_norm"):
            weights[f"{prefix}.self_attn.{norm}.weight"] = torch.ones(16)
        for conv in ("attention_conv", "mlp_conv"):
            weights[f"{prefix}.{conv}.kernel_projection.weight"] = weight(32, 32)
            weights[f"{prefix}.{conv}.base_kernel"] = weight(2, 2, 8, 4)
    save_file(weights, os.path.join(draft_dir, "model.safetensors"))
    save_file(
        {"model.embed_tokens.weight": weight(32, 32), "lm_head.weight": weight(32, 32)},
        str(tmp_path / "model.safetensors"),
    )
    return draft_dir


@pytest.mark.parametrize("ep", ["webgpu", "cuda"])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_dflash2_residual_precision_graph(tmp_path, ep, num_layers):
    builder = DFlash2Builder(
        make_precision_checkpoint(tmp_path, num_layers),
        str(tmp_path),
        ir.DataType.FLOAT16,
        256,
        128,
        ep=ep,
        quant={"bits": 4, "block_size": 32, "prepack": 0, "lm_head": None},
    )
    builder.make_model()
    model = ir.serde.serialize_model(builder.model)
    activation = ir.DataType.FLOAT16 if ep == "webgpu" else ir.DataType.BFLOAT16
    residual = ir.DataType.FLOAT if ep == "webgpu" else activation
    norms = [node for node in builder.graph if node.op_type == "SkipSimplifiedLayerNormalization"]
    assert len(norms) == 2 * num_layers
    for node in norms:
        assert all(value.dtype == residual for value in node.inputs)
        assert all(value.dtype == residual for value in node.outputs if value.name)
        assert builder.graph.initializers[node.name[1:].replace("/", ".") + ".weight"].dtype == activation
        if ep == "webgpu":
            cast = next(n for n in builder.graph if n.name == node.name + "/Cast")
            assert cast.inputs[0] is node.outputs[0]
            assert cast.outputs[0].dtype == activation
            assert node.inputs[1].producer().op_type == "Add"  # No narrowing between finish and norm.
            if node.name != "/dflash2/layers.0/post_attention_layernorm":
                parent = node.inputs[0].producer()
                assert parent.op_type == "SkipSimplifiedLayerNormalization"
                assert node.inputs[0] is parent.outputs[3]
        else:
            assert not any(n.name == node.name + "/Cast" for n in builder.graph)
    for node in builder.graph:
        if "/finish/term" in node.name or "/finish/sum" in node.name:
            assert node.outputs[0].dtype == residual
        if "/prepare/" in node.name and node.op_type in ("Mul", "Add"):
            assert node.outputs[0].dtype == activation
        if "/coef" in node.name and node.op_type == "Add":
            assert node.outputs[0].dtype == activation
        if node.op_type in ("MatMulNBits", "PagedAttention"):
            assert node.inputs[0].dtype == activation
            assert node.outputs[0].dtype == activation
        if node.op_type == "MatMulNBits":
            assert node.attributes["bits"].value == 4
    assert all(v.dtype == activation for v in builder.graph.inputs if v.name.startswith("past_key_values."))
    assert {v.name for v in builder.graph.outputs} == {
        "draft_candidate_ids",
        "draft_scores",
        *(f"present.{i}.{kind}" for i in range(num_layers) for kind in ("key", "value")),
    }
    if ep == "webgpu":
        reference = DFlash2Builder(
            builder.draft_dir,
            str(tmp_path),
            ir.DataType.FLOAT16,
            256,
            128,
            ep=ep,
            quant={"bits": 4, "block_size": 32, "prepack": 0, "lm_head": None},
        )
        reference.layernorm_attrs["residual_dtype"] = reference.io_dtype
        reference.make_model()
        original = ir.serde.serialize_model(reference.model)
        assert {v.name: v.SerializeToString() for v in model.graph.initializer} == {
            v.name: v.SerializeToString() for v in original.graph.initializer
        }
        assert list(model.graph.input) == list(original.graph.input)
        assert list(model.graph.output) == list(original.graph.output)


@pytest.mark.parametrize("ep,io_dtype", [("cpu", ir.DataType.FLOAT), ("webgpu", ir.DataType.FLOAT)])
def test_non_fp16_precision_policy_is_unchanged(tmp_path, ep, io_dtype):
    builder = DFlash2Builder(_draft_checkpoint(tmp_path), str(tmp_path), io_dtype, 256, 128, ep=ep)
    assert builder.io_dtype == builder.layernorm_attrs["residual_dtype"] == io_dtype


def test_block_drafter_supports_model_slice_and_split_signatures():
    builder = BlockDrafterBuilder()
    builder.graph.inputs.append(builder.make_value("input", ir.DataType.FLOAT, [2, 6]))
    builder.make_slice(
        "/slice",
        ["input", builder.const([1]), builder.const([5]), builder.const([1])],
        ir.DataType.FLOAT,
        [2, 4],
    )
    builder.make_split(
        "/split",
        ["/slice/output_0"],
        ["left", "right"],
        [ir.DataType.FLOAT, ir.DataType.FLOAT],
        [[2, 2], [2, 2]],
        axis=1,
        num_outputs=2,
    )
    builder.graph.outputs.extend([builder.values["left"], builder.values["right"]])
    model = ir.serde.serialize_model(builder.model)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    data = np.arange(12, dtype=np.float32).reshape(2, 6)
    left, right = session.run(None, {"input": data})
    np.testing.assert_array_equal(left, data[:, 1:3])
    np.testing.assert_array_equal(right, data[:, 3:5])


def test_shared_skip_norm_default_keeps_existing_dtype():
    builder = BlockDrafterBuilder("skip_norm", "test")
    builder.io_dtype = ir.DataType.BFLOAT16
    builder.hidden_size, builder.rms_eps = 8, 1e-6
    for name in ("root", "skip"):
        builder.graph.inputs.append(builder.make_value(name, builder.io_dtype, [5, 8]))
    output, residual = builder.skip_rms_norm("/norm", "root", "skip", torch.ones(8), 5)
    assert [node.op_type for node in builder.graph] == ["SkipSimplifiedLayerNormalization"]
    assert builder.values[output].dtype == builder.values[residual].dtype == ir.DataType.BFLOAT16


@pytest.mark.parametrize("with_conv", [False, True])
def test_fp32_finish_and_residual_norm_stay_finite(tmp_path, with_conv):
    builder = DFlash2Builder(_draft_checkpoint(tmp_path), str(tmp_path), ir.DataType.FLOAT16, 256, 128, ep="webgpu")
    rows = 10  # Two blocks, including a shift-mask reset.
    for name in ("root", "branch"):
        builder.graph.inputs.append(builder.make_value(name, ir.DataType.FLOAT16, [rows, 8]))
    branch = "branch"
    if with_conv:
        mask = builder._block_shift_mask(branch)
        deltas = [builder.make_initializer(torch.zeros(rows, 2, 1), f"delta{i}", to=builder.io_dtype) for i in range(2)]
        branch = builder._grouped_conv(
            "/conv/finish",
            branch,
            deltas,
            torch.stack((torch.full((2, 4), 7.0), torch.ones(2, 4))),
            mask,
            rows,
            dtype=builder.layernorm_attrs["residual_dtype"],
        )
    normalized, residual = builder.skip_rms_norm(
        "/norm", "root", branch, torch.ones(8), rows, dtype=builder.layernorm_attrs["residual_dtype"]
    )
    builder.graph.outputs.extend([builder.values[normalized], builder.values[residual]])
    model = ir.serde.serialize_model(builder.model)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    root = np.full((rows, 8), 55008, dtype=np.float16)
    values = np.tile(np.array([-15200, 100, 200, 300, 400, 500, 600, 700], dtype=np.float16), (rows, 1))
    if not with_conv:
        values.fill(40576)
    expected_branch = values.astype(np.float32)
    if with_conv:
        shifted = np.concatenate((expected_branch[:1], expected_branch[:-1]))
        mask = (np.arange(rows) % builder.block_size != 0).astype(np.float32)[:, None]
        expected_branch = expected_branch * 7 + shifted * mask
    expected_sum = root.astype(np.float32) + expected_branch
    expected = expected_sum / np.sqrt(np.mean(expected_sum**2, axis=-1, keepdims=True) + builder.rms_eps)
    assert np.abs(expected_branch if with_conv else expected_sum).max() > np.finfo(np.float16).max
    actual, actual_sum = session.run(None, {"root": root, "branch": values})
    assert np.isfinite(actual).all() and np.isfinite(actual_sum).all()
    np.testing.assert_allclose(actual_sum, expected_sum, rtol=0, atol=0)
    np.testing.assert_allclose(actual, expected.astype(np.float16), rtol=1e-3, atol=1e-3)


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
        model.decoder.make_tied_quantized_embedding_input_names = lambda _lm_head: (
            4,
            32,
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


def test_legacy_dflash2_rejects_int2_precision(tmp_path):
    model = _composite()

    with pytest.raises(ValueError, match="dflash2_precision"):
        model.make_dflash2_init(
            io_dtype=None,
            extra_options={"dflash2_path": _draft_checkpoint(tmp_path), "dflash2_precision": "int2"},
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


def test_structured_drafter_quantization_uses_target_hardware_layout():
    quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(block_size=128),
        format=types.SimpleNamespace(matmulnbits_weights_prepacked=0),
    )

    quant = _quant_composite().block_drafter_quant("int4", quant_config)

    assert quant["block_size"] == 128
    assert quant["prepack"] == 1


@pytest.mark.parametrize(
    "drafter_dtype,expected_dtype",
    [("fp16", ir.DataType.FLOAT16), ("bf16", ir.DataType.BFLOAT16)],
)
def test_structured_drafter_uses_its_own_io_dtype(tmp_path, drafter_dtype, expected_dtype):
    model = _composite()
    model.make_dflash2_init(
        io_dtype=ir.DataType.FLOAT16,
        extra_options={
            "dflash2_path": _draft_checkpoint(tmp_path),
            "_drafter_quant_config": QuantConfig.from_dict({"io_dtype": drafter_dtype}),
        },
    )

    assert model.dflash2_attrs["io_dtype"] == ir.DataType.FLOAT16
    assert model.dflash2_attrs["compute_dtype"] == expected_dtype


def test_legacy_drafter_keeps_bf16_body_with_fp16_target(tmp_path):
    model = _composite()
    draft_dir = _draft_checkpoint(tmp_path)
    model.make_dflash2_init(
        io_dtype=ir.DataType.FLOAT16,
        extra_options={"dflash2_path": draft_dir, "dflash2_precision": "int4"},
    )

    assert model.dflash2_attrs["compute_dtype"] == ir.DataType.BFLOAT16
    builder = DFlash2Builder(draft_dir, str(tmp_path), ir.DataType.FLOAT16, 256, 128)
    assert builder.io_dtype == ir.DataType.BFLOAT16
    assert builder.external_dtype == ir.DataType.FLOAT16


def test_int2_fpa_body_keeps_the_targets_int4_lm_head():
    model = _quant_composite()
    quant_config = types.SimpleNamespace(
        weights=types.SimpleNamespace(block_size=64),
        format=types.SimpleNamespace(matmulnbits_weights_prepacked=0),
    )

    quant = model.block_drafter_quant("int2", quant_config)

    assert quant == {
        "bits": 2,
        "block_size": 64,
        "prepack": 0,
    }
    assert model.block_drafter_lm_head_quant() == {
        "bits": 4,
        "block_size": 32,
        "prepack": 1,
        "adopt_target": True,
    }


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

    head_bits, *_ = model.decoder.make_tied_quantized_embedding_input_names(None)
    head_quant = model.block_drafter_lm_head_quant()

    assert head_bits == expected_bits
    if expected_bits == 4:
        assert head_quant["bits"] == expected_bits
    else:
        # Only a 4-bit target LM head is reused.
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


@pytest.mark.parametrize("io_dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_tied_prepacked_target_shares_its_raw_lm_head_with_the_drafter(io_dtype):
    # The prepack pass leaves an LM head the tied embedding also reads raw, so any drafter can adopt it.
    model = _quant_composite(io_dtype=io_dtype)
    model.decoder.tied_quantized_embeddings = True

    assert model.block_drafter_lm_head_quant() == {
        "bits": 4,
        "block_size": 32,
        "prepack": 0,
        "adopt_target": True,
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


def test_int8_target_table_leaves_the_drafter_embedding_dense():
    # embedding_quant_type=int8 writes model.embed_tokens.weight_Q8, not the weight_Q4 the drafter adopts.
    model = _embed_quant_composite()
    model.decoder.quant_attrs["embedding_bits"] = 8

    assert model.block_drafter_embed_quant() is None


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


def test_tied_embedding_rejects_conflicting_sharing_policies(tmp_path):
    model = _composite()
    model.dflash2 = types.SimpleNamespace(
        filename="dflash2.onnx",
        lm_head_quant=None,
        graph=types.SimpleNamespace(initializers={"lm_head.MatMul.weight": object()}),
        embedding_initializers=lambda: frozenset({"lm_head.MatMul.weight"}),
        save_model=lambda _output_dir: None,
    )
    model.dflash2_shared_weight_policies = {"embedding": "required", "lm_head": "off"}
    with pytest.raises(ValueError, match="tied embedding"):
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


@pytest.mark.parametrize("io_dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_quantized_body_uses_requested_dtype_and_sm80_prepack(tmp_path, io_dtype):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        io_dtype,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": 2, "block_size": 64, "prepack": 1},
        compute_dtype=io_dtype,
    )

    output = builder.matmul("/probe/MatMul", "hidden_states", torch.ones((128, 64)), 64, 128, "num_block")

    node = next(node for node in builder.graph if node.name == "/probe/MatMul")
    assert builder.io_dtype == io_dtype
    assert builder.values[output].dtype == io_dtype
    assert builder.graph.initializers["probe.MatMul.weight_scales"].dtype == io_dtype
    assert node.attributes["bits"].value == 2
    assert node.attributes["block_size"].value == 64
    assert node.attributes["weight_prepacked"].value == 1


@pytest.mark.parametrize(
    "bits,block_size,prepack,out_features,expected_prepack",
    [
        (2, 64, 1, 96, False),
        (2, 64, 1, 128, True),
        (2, 32, 1, 128, False),
        (2, 64, 2, 128, False),
        (4, 64, 1, 32, False),
        (4, 64, 1, 64, True),
        (4, 32, 2, 64, False),
        (8, 64, 1, 16, False),
        (8, 64, 1, 32, True),
    ],
)
def test_quantized_body_prepacking_requires_supported_shape(
    tmp_path, bits, block_size, prepack, out_features, expected_prepack
):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": bits, "block_size": block_size, "prepack": prepack},
    )

    builder.matmul("/probe/MatMul", "hidden_states", torch.ones((out_features, 64)), 64, out_features, "num_block")

    node = next(node for node in builder.graph if node.name == "/probe/MatMul")
    assert ("weight_prepacked" in node.attributes) is expected_prepack


@pytest.mark.parametrize("bits", [None, 2, 4, 8])
def test_mlp_gate_up_fusion_preserves_weight_rows(tmp_path, bits):
    quant = {"bits": bits, "block_size": 8, "prepack": 0} if bits else None
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.BFLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant=quant,
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
    assert len(projections) == 2

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


@pytest.mark.parametrize("bits", [None, 2, 4, 8])
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
        )
        builder.io_dtype = ir.DataType.FLOAT
        builder.hidden_size = 32
        builder.intermediate_size = 64
        builder.graph.inputs.append(builder.make_value("hidden_states", ir.DataType.FLOAT, ["num_block", 32]))
        if fused:
            output = builder._make_mlp(0, "hidden_states", weights, "num_block")
        else:
            gate, up = [
                builder.matmul(
                    f"/{projection}/MatMul",
                    "hidden_states",
                    weights[f"layers.0.mlp.{projection}.weight"],
                    32,
                    64,
                    "num_block",
                )
                for projection in ("gate_proj", "up_proj")
            ]
            sigmoid = builder.unary("Sigmoid", "/sigmoid", gate, ir.DataType.FLOAT, ["num_block", 64])
            silu = builder.binary("Mul", "/silu", gate, sigmoid, ir.DataType.FLOAT, ["num_block", 64])
            product = builder.binary("Mul", "/product", silu, up, ir.DataType.FLOAT, ["num_block", 64])
            output = builder.matmul(
                "/down/MatMul", product, weights["layers.0.mlp.down_proj.weight"], 64, 32, "num_block"
            )
        builder.graph.outputs.append(builder.values[output])
        model = ir.serde.serialize_model(builder.model)
        onnx.checker.check_model(model)
        sessions.append(ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]))

    for rows in (1, 8, 16, 32):
        inputs = {"hidden_states": torch.randn((rows, 32), generator=generator).numpy()}
        expected = sessions[0].run(None, inputs)[0]
        actual = sessions[1].run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _attention_weights(generator, hidden, q_dim, kv_dim):
    shapes = {
        "q_proj": (q_dim, hidden),
        "k_proj": (kv_dim, hidden),
        "v_proj": (kv_dim, hidden),
        "o_proj": (hidden, q_dim),
    }
    weights = {
        f"layers.0.self_attn.{name}.weight": torch.randn(shape, generator=generator) / shape[1] ** 0.5
        for name, shape in shapes.items()
    }
    weights["layers.0.self_attn.q_norm.weight"] = torch.ones(q_dim // 2)
    weights["layers.0.self_attn.k_norm.weight"] = torch.ones(q_dim // 2)
    return weights


@pytest.mark.parametrize("bits", [None, 4])
def test_qkv_fusion_feeds_packed_qkv_to_paged_attention(tmp_path, bits):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.BFLOAT16,
        paged_block_size=256,
        max_position_embeddings=128,
        quant={"bits": bits, "block_size": 8, "prepack": 0} if bits else None,
    )
    builder.weights = _attention_weights(torch.Generator().manual_seed(7), 8, 8, 4)

    builder._make_attention(0, "hidden_states", "ctx_n", "num_block")

    nodes = {node.name: node for node in builder.graph}
    projections = [node.name for node in builder.graph if node.op_type in ("MatMul", "MatMulNBits")]
    assert projections == ["/dflash2/layers.0/attn/qkv_proj/MatMul", "/dflash2/layers.0/attn/o_proj/MatMul"]
    rows = nodes["/dflash2/layers.0/attn/qkv_rows/Concat"]
    assert [value.name for value in rows.inputs] == ["hidden_states", "ctx_n"]
    assert rows.attributes["axis"].value == 0
    qkv = nodes["/dflash2/layers.0/attn/qkv_proj/MatMul"]
    assert qkv.inputs[0] is rows.outputs[0]
    assert qkv.outputs[0].shape == ir.Shape(["num_rows", 16])
    gather = nodes["/dflash2/layers.0/attn/qkv_gather"]
    assert [value.name for value in gather.inputs] == [qkv.outputs[0].name, "qkv_row_map"]
    attention = nodes["/dflash2/layers.0/attn/PagedAttention"]
    assert attention.inputs[0] is gather.outputs[0]
    assert all(value is None or value.name == "" for value in attention.inputs[1:3])
    assert not any(node.op_type == "Split" for node in builder.graph)


def test_qkv_fusion_drops_the_query_row_map_input(tmp_path):
    builder = DFlash2Builder(_draft_checkpoint(tmp_path), str(tmp_path), ir.DataType.BFLOAT16, 256, 128)

    builder.declare_io()

    assert "q_row_map" not in {value.name for value in builder.graph.inputs}
    assert "q_row_map" not in builder.genai_config_section()["inputs"]


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_qkv_fusion_matches_unfused_projections(tmp_path, bits):
    draft_dir = _draft_checkpoint(tmp_path)
    hidden, q_dim, kv_dim = 32, 32, 16
    weights = _attention_weights(torch.Generator().manual_seed(11), hidden, q_dim, kv_dim)
    sessions = []
    for fused in (False, True):
        builder = DFlash2Builder(
            draft_dir,
            str(tmp_path),
            ir.DataType.FLOAT,
            paged_block_size=256,
            max_position_embeddings=128,
            quant={"bits": bits, "block_size": 32, "prepack": 0} if bits else None,
        )
        builder.io_dtype = ir.DataType.FLOAT
        builder.hidden_size, builder.num_heads, builder.num_kv_heads, builder.head_size = hidden, 2, 1, 16
        builder.weights = weights
        for name, rows in (("hidden_states", "num_block"), ("ctx_n", "num_ctx")):
            builder.graph.inputs.append(builder.make_value(name, ir.DataType.FLOAT, [rows, hidden]))
        for name in ("qkv_row_map",) if fused else ("q_row_map", "qkv_row_map"):
            builder.graph.inputs.append(builder.make_value(name, ir.DataType.INT32, ["num_tokens"]))
        if fused:
            outputs = [builder.make_packed_qkv(0, "hidden_states", "ctx_n")]
        else:
            projections = [
                builder.matmul(
                    f"/reference/{proj}/MatMul",
                    "hidden_states" if proj == "q_proj" else "all_rows",
                    weights[f"layers.0.self_attn.{proj}.weight"],
                    hidden,
                    q_dim if proj == "q_proj" else kv_dim,
                    "num_block" if proj == "q_proj" else "num_rows",
                )
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
            builder.make_node("Concat", ["hidden_states", "ctx_n"], ["all_rows"], name="/reference/Concat", axis=0)
            outputs = [
                builder.binary(
                    "Gather",
                    f"/reference/{projection}/Gather",
                    root,
                    "q_row_map" if projection == "q_proj" else "qkv_row_map",
                    ir.DataType.FLOAT,
                    ["num_tokens", q_dim if projection == "q_proj" else kv_dim],
                )
                for projection, root in zip(("q_proj", "k_proj", "v_proj"), projections, strict=True)
            ]
        builder.graph.outputs.extend(builder.values[output] for output in outputs)
        builder.graph.sort()
        model = ir.serde.serialize_model(builder.model)
        onnx.checker.check_model(model)
        sessions.append(ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]))

    generator = torch.Generator().manual_seed(12)
    for num_block, num_ctx in ((8, 1), (8, 5), (16, 3)):
        # Context rows first, as the runtime packs each request; block rows index themselves.
        inputs = {
            "hidden_states": torch.randn((num_block, hidden), generator=generator).numpy(),
            "ctx_n": torch.randn((num_ctx, hidden), generator=generator).numpy(),
            "q_row_map": np.concatenate([np.zeros(num_ctx), np.arange(num_block)]).astype(np.int32),
            "qkv_row_map": np.concatenate([num_block + np.arange(num_ctx), np.arange(num_block)]).astype(np.int32),
        }
        q, k, v = sessions[0].run(None, inputs)
        (packed,) = sessions[1].run(None, {name: inputs[name] for name in ("hidden_states", "ctx_n", "qkv_row_map")})
        np.testing.assert_allclose(packed[num_ctx:, :q_dim], q[num_ctx:], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(packed[:, q_dim : q_dim + kv_dim], k, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(packed[:, q_dim + kv_dim :], v, rtol=1e-5, atol=1e-5)


def _quantized_head_builder(tmp_path, bits=4, block_size=8, io_dtype=ir.DataType.FLOAT16):
    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        io_dtype,
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


def test_bf16_drafter_adopts_an_unpacked_target_lm_head(tmp_path):
    builder = _quantized_head_builder(tmp_path, io_dtype=ir.DataType.BFLOAT16)
    builder.make_lm_head("hidden_states", "num_sample")
    target_path, qweight, _ = _save_quantized_target(tmp_path, builder, weight_prepacked=None)

    builder.adopt_target_tensors(target_path)

    adopted = builder.graph.initializers["lm_head.MatMul.weight_Q4"].const_value.numpy()
    np.testing.assert_array_equal(adopted, qweight)
    node = next(node for node in builder.graph if node.name == "/lm_head/MatMul")
    assert "weight_prepacked" not in node.attributes


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


@pytest.mark.parametrize("tied", [False, True])
def test_drafter_reuses_decoder_embedding_during_export(tmp_path, tied):
    decoder = object.__new__(Model)
    decoder.make_graph("decoder", opset=21)
    decoder.io_dtype = ir.DataType.FLOAT
    decoder.onnx_dtype = ir.DataType.FLOAT
    decoder.vocab_size, decoder.hidden_size = 32, 8
    decoder.use_paged_attention = True
    decoder.hidden_rows_dim = "num_tokens"
    decoder.input_names = {"input_ids": "input_ids"}
    decoder.quant_attrs = {}
    decoder.tied_quantized_embeddings = False
    decoder.tied_unquantized_embeddings = tied
    decoder.embed_attrs = {"scale": 1}
    decoder.layernorm_attrs = {"cast": {"use_fp32": False}}
    decoder.graph.inputs.append(decoder.make_value("input_ids", ir.DataType.INT64, ["num_tokens"]))
    table = torch.arange(256, dtype=torch.float32).reshape(32, 8) / 256
    if tied:
        decoder.make_initializer(table.T, "lm_head.MatMul.weight")
    decoder.make_embedding(table)
    decoder.values[decoder.layernorm_attrs["root_input"]].metadata_props["embedding_test"] = "decoder"
    decoder.graph.outputs.append(decoder.values[decoder.layernorm_attrs["root_input"]])

    builder = DFlash2Builder(
        _draft_checkpoint(tmp_path),
        str(tmp_path),
        ir.DataType.FLOAT,
        256,
        128,
        ep="cpu",
        target_model=decoder.model,
    )
    builder.graph.inputs.append(builder.make_value("input_ids", ir.DataType.INT64, ["num_block"]))
    output = builder.make_embedding("/dflash2/embed_tokens/Gather", "num_block")
    builder.graph.outputs.append(builder.values[output])
    builder.save_model(str(tmp_path))

    name = "lm_head.MatMul.weight" if tied else "model.embed_tokens.weight"
    assert builder.graph.initializers[name].const_value is decoder.graph.initializers[name].const_value
    assert builder.embedding_initializers() == frozenset({name})
    copied = builder.values[output]
    assert copied.metadata_props["embedding_test"] == "decoder"
    assert copied.shape == ir.Shape(["num_block", 8])
    saved = ir.load(tmp_path / "dflash2.onnx")
    assert {value.const_value.location for value in saved.graph.initializers.values()} == {"dflash2.onnx.data"}
    session = ort.InferenceSession(str(tmp_path / "dflash2.onnx"), providers=["CPUExecutionProvider"])
    ids = np.array([3, 0, 31, 3], dtype=np.int64)
    np.testing.assert_array_equal(session.run(None, {"input_ids": ids})[0], table.numpy()[ids])


@pytest.mark.parametrize("ep", ["cuda", "webgpu"])
def test_drafter_excludes_embedding_during_export(tmp_path, ep):
    builder = DFlash2Builder(
        make_precision_checkpoint(tmp_path, 1),
        str(tmp_path),
        ir.DataType.FLOAT16,
        256,
        128,
        ep=ep,
        exclude_embeds=True,
    )
    builder.make_model()
    builder.save_model(str(tmp_path))
    saved = ir.load(tmp_path / "dflash2.onnx")
    inputs = {value.name: value for value in saved.graph.inputs}
    assert inputs["inputs_embeds"].shape == ir.Shape(["num_block", 32])
    assert "input_ids" in inputs  # Selector scores still need the anchor token.
    assert ("attention_metadata" in inputs) == (ep == "cuda")
    assert not any(name.startswith("model.embed_tokens.") for name in saved.graph.initializers)
    assert builder.genai_config_section()["inputs"]["inputs_embeds"] == "inputs_embeds"
    assert {value.const_value.location for value in saved.graph.initializers.values()} == {"dflash2.onnx.data"}
    assert len(saved.graph) > 0


@pytest.mark.parametrize("ep", ["cuda", "webgpu"])
def test_complete_dflash_export_matches_embedding_inputs_on_cuda(tmp_path, ep):
    if "CUDAExecutionProvider" not in ort.get_available_providers() or not torch.cuda.is_available():
        pytest.skip("CUDA is required to execute the exported PagedAttention graph.")
    draft_dir = make_precision_checkpoint(tmp_path, 1)
    decoder = object.__new__(Model)
    decoder.make_graph("decoder", opset=21)
    decoder.io_dtype, decoder.onnx_dtype = ir.DataType.FLOAT16, ir.DataType.FLOAT16
    decoder.vocab_size, decoder.hidden_size = 32, 32
    decoder.use_paged_attention, decoder.hidden_rows_dim = True, "num_tokens"
    decoder.input_names, decoder.quant_attrs = {"input_ids": "input_ids"}, {}
    decoder.tied_quantized_embeddings = decoder.tied_unquantized_embeddings = False
    decoder.embed_attrs, decoder.layernorm_attrs = {"scale": 1}, {"cast": {"use_fp32": False}}
    decoder.graph.inputs.append(decoder.make_value("input_ids", ir.DataType.INT64, ["num_tokens"]))
    table = torch.arange(1024, dtype=torch.float16).reshape(32, 32) / 1024
    decoder.make_embedding(table)
    sessions = []
    for exclude_embeds in (False, True):
        builder = DFlash2Builder(
            draft_dir,
            str(tmp_path),
            ir.DataType.FLOAT16,
            256,
            128,
            ep=ep,
            compute_dtype=ir.DataType.FLOAT16,
            target_model=decoder.model,
            exclude_embeds=exclude_embeds,
            filename=f"dflash-{exclude_embeds}.onnx",
        )
        builder.make_model()
        builder.save_model(str(tmp_path))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sessions.append(
            ort.InferenceSession(str(tmp_path / builder.filename), options, providers=["CUDAExecutionProvider"])
        )
    rng = np.random.default_rng(42)
    ids = np.array([3, 31, 31, 31, 31], dtype=np.int64)
    feeds = {
        "input_ids": ids,
        "aux_hidden_states": rng.standard_normal((3, 96)).astype(np.float16) * 0.01,
        "qkv_row_map": np.array([5, 6, 7, 0, 1, 2, 3, 4], dtype=np.int32),
        "block_row_index": np.arange(3, 8, dtype=np.int32),
        "cumulative_sequence_lengths": np.array([0, 8], dtype=np.int32),
        "past_sequence_lengths": np.array([0], dtype=np.int32),
        "block_table": np.array([[0]], dtype=np.int32),
        "past_key_values.0.key": np.zeros((1, 256, 1, 16), dtype=np.float16),
        "past_key_values.0.value": np.zeros((1, 256, 1, 16), dtype=np.float16),
    }
    if ep == "cuda":
        feeds["attention_metadata"] = np.array([8, 8, 8], dtype=np.int32)

    def run(session, inputs):
        binding = session.io_binding()
        caches = {}
        for name, value in inputs.items():
            if name.startswith("past_key_values."):
                cache = ort.OrtValue.ortvalue_from_numpy(value.copy(), "cuda", 0)
                caches[name] = cache
                binding.bind_ortvalue_input(name, cache)
            else:
                binding.bind_cpu_input(name, value)
        for output in session.get_outputs():
            if output.name.startswith("present."):
                binding.bind_ortvalue_output(output.name, caches[output.name.replace("present.", "past_key_values.")])
            else:
                binding.bind_output(output.name, "cuda", 0)
        session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()

    expected = run(sessions[0], feeds)
    actual = run(sessions[1], {**feeds, "inputs_embeds": table.numpy()[ids]})
    for left, right in zip(expected, actual, strict=True):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("weight_dtype", [ir.DataType.FLOAT16, ir.DataType.INT4])
def test_pure_model_exports_excluded_dflash_embedding(tmp_path, monkeypatch, weight_dtype):
    config = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
        _name_or_path=str(tmp_path),
        eos_token_id=2,
    )
    weights = transformers.LlamaForCausalLM(config).eval()
    options = {
        "exclude_embeds": True,
        "use_paged_attention": True,
        "dflash2_path": _draft_checkpoint(tmp_path),
        "op_types_to_quantize": ("MatMul", "Gather"),
    }
    decoder = Model(config, ir.DataType.FLOAT16, weight_dtype, "cuda", str(tmp_path / "cache"), options)
    monkeypatch.setattr(decoder, "load_weights", lambda _path: weights)
    decoder.make_model(str(tmp_path))
    decoder.save_model(str(tmp_path))
    decoder.make_genai_config(config, {}, str(tmp_path))

    embedding_path = tmp_path / decoder.embedding_model.filename
    assert embedding_path.name == "embedding.onnx"
    embedding = ir.load(embedding_path)
    assert [value.name for value in embedding.graph.inputs] == ["input_ids"]
    assert [value.name for value in embedding.graph.outputs] == ["inputs_embeds"]
    assert embedding.graph.outputs[0].dtype == ir.DataType.FLOAT16
    assert embedding.graph.inputs[0].shape == ir.Shape(decoder.input_shapes["input_ids"])
    assert {value.const_value.location for value in embedding.graph.initializers.values()} == {"embedding.onnx.data"}
    assert not any(name.startswith("model.embed_tokens.") for name in decoder.graph.initializers)
    with open(tmp_path / "genai_config.json") as handle:
        generated = json.load(handle)
    assert generated["model"]["embedding"]["filename"] == "embedding.onnx"
    assert generated["model"]["embedding"]["inputs"]["input_ids"] == "input_ids"
    assert generated["model"]["embedding"]["outputs"]["inputs_embeds"] == "inputs_embeds"
    assert generated["model"]["decoder"]["inputs"]["inputs_embeds"] == "inputs_embeds"
    session = ort.InferenceSession(str(embedding_path), providers=["CPUExecutionProvider"])
    ids = np.array([3, 31, 0, 3], dtype=np.int64)
    actual = session.run(None, {"input_ids": ids})[0]
    expected = weights.model.embed_tokens.weight.detach().numpy()[ids].astype(np.float16)
    if weight_dtype == ir.DataType.FLOAT16:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert any(node.op_type == "GatherBlockQuantized" for node in embedding.graph)
        np.testing.assert_allclose(actual, expected, rtol=0.2, atol=0.005)


def test_embedding_adoption_uses_engine_graph_and_head_uses_decoder(tmp_path):
    builder = _quantized_embedding_builder(tmp_path, block_size=8)
    builder.make_embedding("/dflash2/embed_tokens/Gather", "num_block")
    builder.make_lm_head("hidden_states")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path, embedding_weight, embedding_scales = _quantized_embedding_target(source_dir, builder, block_size=8)
    embedding_path = tmp_path / "embedding.onnx"
    Model.save_onnx_model(ir.load(source_path), embedding_path)
    decoder_path, head_weight, head_scales = _save_quantized_target(tmp_path, builder)

    builder.adopt_target_tensors(decoder_path, embedding_model_path=embedding_path)

    for name, expected in (
        ("model.embed_tokens.weight_Q4", embedding_weight),
        ("model.embed_tokens.weight_scales", embedding_scales),
        ("lm_head.MatMul.weight_Q4", head_weight),
        ("lm_head.MatMul.weight_scales", head_scales),
    ):
        np.testing.assert_array_equal(builder.graph.initializers[name].const_value.numpy(), expected)
    assert not builder.pending_adoptions()


def test_composite_routes_adoption_to_named_engine_graph(tmp_path):
    model = _composite()
    model.decoder.exclude_embeds = True
    model.decoder.embedding_model = types.SimpleNamespace(filename="embedding.onnx")
    captured = {}
    drafter = types.SimpleNamespace(
        filename="dflash2.onnx",
        exclude_embeds=True,
        lm_head_quant=None,
        adopt_target_tensors=lambda source, **kwargs: captured.update(source=source, **kwargs),
        save_model=lambda _output: None,
    )
    model.share_initializers = lambda *args, **kwargs: []

    model.save_block_drafter_model(drafter, str(tmp_path), "DFlash 2")

    assert captured == {
        "source": str(tmp_path / "model.onnx"),
        "embedding_model_path": str(tmp_path / "embedding.onnx"),
    }


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


@pytest.mark.parametrize("conv_kernel_size", [0, 1, 3, 4])
def test_unsupported_conv_kernel_size_is_rejected(tmp_path, conv_kernel_size):
    draft_dir = _draft_checkpoint(tmp_path)
    config_path = tmp_path / "dflash2_draft" / "config.json"
    config = json.loads(config_path.read_text())
    config["dflash_config"]["conv_kernel_size"] = conv_kernel_size
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=rf"conv_kernel_size=2.*got {conv_kernel_size}"):
        DFlash2Builder(draft_dir, str(tmp_path), ir.DataType.FLOAT16, 256, 128)


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
            captured["ep"] = _kwargs["ep"]

        def make_model(self):
            pass

    dflash2_module = importlib.import_module("models.builders.dflash2")
    monkeypatch.setattr(dflash2_module, "DFlash2Builder", StubDFlash2Builder)
    model = _composite(ep="webgpu")
    model.make_dflash2_init(
        io_dtype=ir.DataType.FLOAT16,
        extra_options={"dflash2_path": _draft_checkpoint(tmp_path)},
    )

    model.make_dflash2_model(str(tmp_path))

    assert captured["max_position"] == model.decoder.context_length
    assert captured["ep"] == "webgpu"


def test_drafter_resolves_target_repository_to_local_snapshot(tmp_path, monkeypatch):
    captured = {}
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()

    class StubDFlash2Builder:
        def __init__(self, _draft_dir, target_dir, _io_dtype, _paged_block_size, _max_position, **_kwargs):
            captured["target_dir"] = target_dir

        def make_model(self):
            pass

    dflash2_module = importlib.import_module("models.builders.dflash2")
    monkeypatch.setattr(dflash2_module, "DFlash2Builder", StubDFlash2Builder)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, cache_dir, token: captured.update(repo_id=repo_id, cache_dir=cache_dir, token=token)
        or str(snapshot_dir),
    )
    model = _composite()
    model.decoder.model_name_or_path = "Qwen/Qwen3.8-27B"
    model.decoder.cache_dir = str(tmp_path / "cache")
    model.decoder.hf_token = False
    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})

    model.make_dflash2_model("Qwen/Qwen3.8-27B")

    assert captured == {
        "repo_id": "Qwen/Qwen3.8-27B",
        "cache_dir": str(tmp_path / "cache"),
        "token": False,
        "target_dir": str(snapshot_dir),
    }


def test_fusion_is_intrinsic_to_the_drafter(tmp_path):
    model = _composite()
    model.make_dflash2_init(io_dtype=None, extra_options={"dflash2_path": _draft_checkpoint(tmp_path)})

    builder = DFlash2Builder(model.dflash2_path, str(tmp_path), ir.DataType.BFLOAT16, 256, 128)
    assert not hasattr(builder, "attn_attrs")
    assert not builder.uses_q_row_map


def test_failed_save_preserves_existing_dflash2_files(tmp_path, monkeypatch):
    model_path = tmp_path / "dflash2.onnx"
    data_path = tmp_path / "dflash2.onnx.data"
    model_path.write_bytes(b"old model")
    data_path.write_bytes(b"old data")
    builder = object.__new__(DFlash2Builder)
    builder.filename = "dflash2.onnx"
    builder.model = ir.Model(ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22}), ir_version=10)

    def fail_save(_model, staged_path, **kwargs):
        with open(staged_path, "wb") as staged_model:
            staged_model.write(b"partial model")
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
