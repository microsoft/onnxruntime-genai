# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import safetensors.torch as safetensors_torch
import torch

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
sys.path.insert(0, str(MODELS_DIR))

from loaders.modelopt import ModeloptModel  # noqa: E402

from models.builders.dflash2 import DFlash2Builder  # noqa: E402
from models.builders.lilicorr import LiLiCorrBuilder  # noqa: E402


def pack_nvfp4(values, scale_f32=1.0, global_scale=1.0, uint8_scale=False):
    """Pack a ``[N, K]`` tensor of FP4-representable values into ModelOpt NVFP4 tensors."""
    lut = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7}
    out_features, in_features = values.shape
    assert in_features % 16 == 0
    unscaled = values / (scale_f32 * global_scale)
    codes = torch.zeros_like(unscaled, dtype=torch.uint8)
    for idx, mag in lut.items():
        if idx == 0.0:
            continue
        codes = torch.where(
            torch.isclose(unscaled.abs(), torch.tensor(idx)), torch.tensor(mag, dtype=torch.uint8), codes
        )
    codes = torch.where(unscaled < 0, codes | 0x8, codes)
    packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    scale_fp8 = torch.full((out_features, in_features // 16), scale_f32, dtype=torch.float32).to(torch.float8_e4m3fn)
    weight_scale = scale_fp8.view(torch.uint8) if uint8_scale else scale_fp8
    weight_scale_2 = torch.tensor(global_scale, dtype=torch.float32)
    return packed.contiguous(), weight_scale.contiguous(), weight_scale_2


def pack_q8_0(int8_weights, scales_fp16):
    """Pack ``int8 [N, K/32, 32]`` and ``float16 [N, K/32]`` into ggml Q8_0 ``uint8 [N, K/32, 34]`` blocks."""
    scale_bytes = scales_fp16.contiguous().unsqueeze(-1).view(torch.uint8)
    quant_bytes = int8_weights.contiguous().view(torch.uint8)
    return torch.cat([scale_bytes, quant_bytes], dim=-1).contiguous()


def write_lilicorr_checkpoints(tmp_path, uint8_scale=False, q8_lm_head=True, num_layers=1):
    draft_dir = tmp_path / ("draft_uint8" if uint8_scale else "draft_fp8")
    target_dir = tmp_path / ("target_q8" if q8_lm_head else "target_dense")
    draft_dir.mkdir(exist_ok=True)
    target_dir.mkdir(exist_ok=True)

    hidden_size = 32
    head_dim = 16
    num_heads = 2
    num_kv_heads = 1
    intermediate_size = 64
    vocab_size = 48
    block_size = 4
    target_layer_ids = [1, 3]
    aux_hidden_size = hidden_size * len(target_layer_ids)
    lh = 16
    l_heads = 2
    l_mlp = 32
    factor_dim = 8
    top_k = 4

    config = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "use_sliding_window": True,
        "sliding_window": 32,
        "block_size": block_size,
        "dflash_config": {
            "projector_type": "lilicorr",
            "lilicorr_enabled": True,
            "causal": False,
            "attention_sink_bias": True,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "lilicorr_candidate_topk": top_k,
            "mask_token_id": vocab_size - 1,
            "target_layer_ids": target_layer_ids,
            "lilicorr_hidden_size": lh,
            "lilicorr_num_layers": 1,
            "lilicorr_num_heads": l_heads,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": factor_dim,
            "lilicorr_vector_eps": 1e-6,
            "lilicorr_logit_scale": 4.0,
        },
    }
    (draft_dir / "config.json").write_text(json.dumps(config))

    fc_dense = torch.full((hidden_size, aux_hidden_size), 1.5, dtype=torch.float32)
    fc_w, fc_s, fc_s2 = pack_nvfp4(fc_dense, scale_f32=1.0, global_scale=1.0, uint8_scale=uint8_scale)
    q_dense = torch.full((num_heads * head_dim, hidden_size), -0.5, dtype=torch.float32)
    q_w, q_s, q_s2 = pack_nvfp4(q_dense, scale_f32=1.0, global_scale=0.5, uint8_scale=uint8_scale)

    draft_tensors = {
        "fc.weight": fc_w,
        "fc.weight_scale": fc_s,
        "fc.weight_scale_2": fc_s2,
        "hidden_norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "layers.0.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "layers.0.post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "layers.0.attention_conv.kernel_projection.weight": torch.full(
            (2 * 2 * (hidden_size // 16), hidden_size), 0.125, dtype=torch.bfloat16
        ),
        "layers.0.attention_conv.base_kernel": torch.ones((2, 2, hidden_size), dtype=torch.bfloat16),
        "layers.0.self_attn.q_proj.weight": q_w,
        "layers.0.self_attn.q_proj.weight_scale": q_s,
        "layers.0.self_attn.q_proj.weight_scale_2": q_s2,
        "layers.0.self_attn.k_proj.weight": torch.full(
            (num_kv_heads * head_dim, hidden_size), 0.25, dtype=torch.bfloat16
        ),
        "layers.0.self_attn.v_proj.weight": torch.full(
            (num_kv_heads * head_dim, hidden_size), 0.25, dtype=torch.bfloat16
        ),
        "layers.0.self_attn.o_proj.weight": torch.full((hidden_size, num_heads * head_dim), 0.25, dtype=torch.bfloat16),
        "layers.0.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
        "layers.0.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
        "layers.0.self_attn.attention_sink_bias": torch.zeros(num_heads, dtype=torch.bfloat16),
        "layers.0.mlp_conv.kernel_projection.weight": torch.full(
            (2 * 2 * (hidden_size // 16), hidden_size), 0.125, dtype=torch.bfloat16
        ),
        "layers.0.mlp_conv.base_kernel": torch.ones((2, 2, hidden_size), dtype=torch.bfloat16),
        "layers.0.mlp.gate_proj.weight": torch.full((intermediate_size, hidden_size), 0.25, dtype=torch.bfloat16),
        "layers.0.mlp.up_proj.weight": torch.full((intermediate_size, hidden_size), 0.25, dtype=torch.bfloat16),
        "layers.0.mlp.down_proj.weight": torch.full((hidden_size, intermediate_size), 0.25, dtype=torch.bfloat16),
        "lilicorr.token_proj.weight": torch.full((lh, hidden_size), 0.25, dtype=torch.bfloat16),
        "lilicorr.token_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.pass_hidden_proj.weight": torch.full((lh, hidden_size), 0.25, dtype=torch.bfloat16),
        "lilicorr.pass_hidden_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.feature_norm.weight": torch.ones(5, dtype=torch.bfloat16),
        "lilicorr.feature_norm.bias": torch.zeros(5, dtype=torch.bfloat16),
        "lilicorr.feature_mlp.up_proj.weight": torch.full((lh, 5), 0.25, dtype=torch.bfloat16),
        "lilicorr.feature_mlp.up_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.feature_mlp.down_proj.weight": torch.full((lh, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.feature_mlp.down_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.slot_embedding": torch.zeros((1, 1, block_size - 1, 1, lh), dtype=torch.bfloat16),
        "lilicorr.rank_embedding": torch.zeros((1, 1, 1, top_k, lh), dtype=torch.bfloat16),
        "lilicorr.relative_slot_bias": torch.zeros((l_heads, 2 * block_size - 1), dtype=torch.bfloat16),
        "lilicorr.same_slot_bias": torch.zeros((l_heads,), dtype=torch.bfloat16),
        "lilicorr.layers.0.attn_norm.weight": torch.ones(lh, dtype=torch.bfloat16),
        "lilicorr.layers.0.attn.in_proj_weight": torch.full((3 * lh, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.layers.0.attn.in_proj_bias": torch.zeros(3 * lh, dtype=torch.bfloat16),
        "lilicorr.layers.0.attn.out_proj.weight": torch.full((lh, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.layers.0.attn.out_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.layers.0.mlp_norm.weight": torch.ones(lh, dtype=torch.bfloat16),
        "lilicorr.layers.0.mlp.up_proj.weight": torch.full((l_mlp, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.layers.0.mlp.up_proj.bias": torch.zeros(l_mlp, dtype=torch.bfloat16),
        "lilicorr.layers.0.mlp.down_proj.weight": torch.full((lh, l_mlp), 0.25, dtype=torch.bfloat16),
        "lilicorr.layers.0.mlp.down_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.output_norm.weight": torch.ones(lh, dtype=torch.bfloat16),
        "lilicorr.context_proj.weight": torch.full((lh, hidden_size), 0.25, dtype=torch.bfloat16),
        "lilicorr.context_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.anchor_norm.weight": torch.ones(lh, dtype=torch.bfloat16),
        "lilicorr.factor_input_proj.weight": torch.full((lh, 3 * lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.factor_input_proj.bias": torch.zeros(lh, dtype=torch.bfloat16),
        "lilicorr.out_head.weight": torch.full((factor_dim, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.out_head.bias": torch.zeros(factor_dim, dtype=torch.bfloat16),
        "lilicorr.in_head.weight": torch.full((factor_dim, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.in_head.bias": torch.zeros(factor_dim, dtype=torch.bfloat16),
        "lilicorr.anchor_out_head.weight": torch.full((factor_dim, lh), 0.25, dtype=torch.bfloat16),
        "lilicorr.anchor_out_head.bias": torch.zeros(factor_dim, dtype=torch.bfloat16),
    }
    layer_weights = {name: tensor for name, tensor in draft_tensors.items() if name.startswith("layers.0.")}
    for layer_id in range(1, num_layers):
        for name, tensor in layer_weights.items():
            draft_tensors[name.replace("layers.0.", f"layers.{layer_id}.", 1)] = tensor.clone()
    safetensors_torch.save_file(draft_tensors, str(draft_dir / "model.safetensors"))

    embed_weight = torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size) * 0.01
    embed_weight = embed_weight.to(torch.bfloat16)
    int8_vals = (
        ((torch.arange(vocab_size * hidden_size) % 15) - 7).to(torch.int8).reshape(vocab_size, hidden_size // 32, 32)
    )
    # 0.5 * (1 + 2**-8) is exact in FP16 but not in BF16.
    scales_fp16 = torch.full((vocab_size, hidden_size // 32), 0.501953125, dtype=torch.float16)
    if q8_lm_head:
        lm_head_tensor = pack_q8_0(int8_vals, scales_fp16)
    else:
        lm_head_tensor = (
            (int8_vals.to(torch.float32) * scales_fp16.to(torch.float32).unsqueeze(-1))
            .reshape(vocab_size, hidden_size)
            .to(torch.bfloat16)
        )

    safetensors_torch.save_file(
        {
            "language_model.backbone.embeddings.weight": embed_weight,
            "language_model.lm_head.weight": lm_head_tensor,
        },
        str(target_dir / "model.safetensors"),
    )
    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.backbone.embeddings.weight": "model.safetensors",
                    "language_model.lm_head.weight": "model.safetensors",
                }
            }
        )
    )
    return str(draft_dir), str(target_dir)


def test_nvfp4_decoding_matches_for_uint8_and_float8_scales(tmp_path):
    draft_fp8, target_dir = write_lilicorr_checkpoints(tmp_path, uint8_scale=False)
    draft_u8, _ = write_lilicorr_checkpoints(tmp_path, uint8_scale=True)

    weights_fp8 = ModeloptModel.load_lilicorr_weights(draft_fp8, target_dir)
    weights_u8 = ModeloptModel.load_lilicorr_weights(draft_u8, target_dir)

    # A float8_e4m3fn scale of 1.0 has raw byte value 0x38 (56); viewing as float8_e4m3fn before converting to float32
    # must produce 1.5 * 1.0 = 1.5, not 1.5 * 56.0 = 84.0.
    torch.testing.assert_close(weights_u8["fc.weight"], torch.full_like(weights_u8["fc.weight"], 1.5))
    torch.testing.assert_close(
        weights_u8["layers.0.self_attn.q_proj.weight"],
        torch.full_like(weights_u8["layers.0.self_attn.q_proj.weight"], -0.5),
    )
    torch.testing.assert_close(weights_u8["fc.weight"], weights_fp8["fc.weight"])
    torch.testing.assert_close(
        weights_u8["layers.0.self_attn.q_proj.weight"],
        weights_fp8["layers.0.self_attn.q_proj.weight"],
    )

    builder_fp8 = LiLiCorrBuilder(draft_fp8, target_dir, paged_block_size=256)
    builder_fp8.make_model()
    builder_u8 = LiLiCorrBuilder(draft_u8, target_dir, paged_block_size=256)
    builder_u8.make_model()
    for init_name in ("dflash2.fc.MatMul.weight", "dflash2.layers.0.attn.qkv_proj.MatMul.weight"):
        np.testing.assert_array_equal(
            builder_u8.graph.initializers[init_name].const_value.numpy(),
            builder_fp8.graph.initializers[init_name].const_value.numpy(),
        )


def test_q8_0_unpacking_and_dequantization_in_loader(tmp_path):
    draft_dir, target_dir = write_lilicorr_checkpoints(tmp_path, uint8_scale=True, q8_lm_head=True)

    weights_int8 = ModeloptModel.load_lilicorr_weights(draft_dir, target_dir, dequantize_lm_head=False)
    weights_bf16 = ModeloptModel.load_lilicorr_weights(draft_dir, target_dir, dequantize_lm_head=True)

    assert "lm_head.qweight" in weights_int8
    assert "lm_head.scales" in weights_int8
    assert "lm_head.weight" not in weights_int8
    assert weights_int8["lm_head.qweight"].dtype == torch.uint8
    assert weights_int8["lm_head.scales"].dtype == torch.float16

    q, d = weights_int8["lm_head.qweight"], weights_int8["lm_head.scales"]
    reconstructed = (
        ((q.to(torch.float32) - 128.0) * d.to(torch.float32).unsqueeze(-1)).reshape(q.shape[0], -1).to(torch.bfloat16)
    )
    torch.testing.assert_close(weights_bf16["lm_head.weight"], reconstructed)


@pytest.mark.parametrize("precision,bits", [("bf16", None), ("int4", 4), ("int8", 8)])
def test_lilicorr_reuses_parent_initialization(tmp_path, monkeypatch, precision, bits):
    draft_dir, target_dir = write_lilicorr_checkpoints(tmp_path)
    parent_init = DFlash2Builder.__init__
    initialized = []

    def initialize(builder, *args, **kwargs):
        parent_init(builder, *args, **kwargs)
        initialized.append(builder)

    monkeypatch.setattr(DFlash2Builder, "__init__", initialize)
    builder = LiLiCorrBuilder(
        draft_dir,
        target_dir,
        paged_block_size=128,
        max_position_embeddings=32,
        filename="custom.onnx",
        num_draft_tokens=2,
        precision=precision,
    )
    assert initialized == [builder]
    assert builder.quant_bits == bits
    assert (
        builder.io_dtype == builder.external_dtype == builder.layernorm_attrs["residual_dtype"] == ir.DataType.BFLOAT16
    )
    assert builder.ep == "cuda"
    assert builder.target_model is None
    assert not builder.exclude_embeds
    assert builder.filename == "custom.onnx"
    assert builder.paged_block_size == 128
    assert builder.max_position == 32
    assert builder.block_size == 3
    assert builder.num_draft_tokens == 2
    assert builder.model.graph.opset_imports[""] == 21
    assert builder.model.graph.opset_imports["com.microsoft"] == 1


@pytest.mark.parametrize("lm_head_precision", ["int8", "bf16"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_lilicorr_builder_emits_valid_graph(tmp_path, lm_head_precision, num_layers):
    draft_dir, target_dir = write_lilicorr_checkpoints(
        tmp_path, uint8_scale=True, q8_lm_head=True, num_layers=num_layers
    )

    builder = LiLiCorrBuilder(
        draft_dir,
        target_dir,
        paged_block_size=256,
        lm_head_precision=lm_head_precision,
    )
    builder.make_model()

    out_dir = tmp_path / f"out_{lm_head_precision}"
    out_dir.mkdir()
    builder.save_model(str(out_dir))
    saved = onnx.load(str(out_dir / builder.filename), load_external_data=False)
    assert len(saved.graph.node) == len(builder.graph)

    input_names = [inp.name for inp in builder.graph.inputs]
    output_names = [out.name for out in builder.graph.outputs]
    assert "anchor_row_index" in input_names
    assert "q_row_map" not in input_names
    assert "attention_metadata" in input_names
    assert all(inp.uses() for inp in builder.graph.inputs)
    assert output_names[:2] == ["draft_candidate_ids", "draft_scores"]
    assert builder.layernorm_attrs["residual_dtype"] == ir.DataType.BFLOAT16

    nodes = {node.name: node for node in builder.graph}
    assert nodes["/lm_head/MatMul"].op_type == ("MatMulNBits" if lm_head_precision == "int8" else "MatMul")
    for layer_id in range(num_layers):
        prefix = f"/dflash2/layers.{layer_id}/attn"
        paged_attn = nodes[f"{prefix}/PagedAttention"]
        assert paged_attn.inputs[0].name == f"{prefix}/qkv_gather/output_0"
        assert not paged_attn.inputs[1].name
        assert not paged_attn.inputs[2].name
        assert paged_attn.inputs[11].name == f"dflash2.layers.{layer_id}.self_attn.sinks"
        assert paged_attn.inputs[16].name == "attention_metadata"
    assert builder.ctx_pre_norm == "/dflash2/fc/MatMul/output_0"

    # `in_vec` must be consumed by only one Slice so ORT's GatherSliceToSplitFusion does not fuse two slices into a Split.
    in_vec_name = "/lilicorr/in_norm/div/output_0"
    in_vec_slices = [node for node in builder.graph if node.op_type == "Slice" and node.inputs[0].name == in_vec_name]
    assert len(in_vec_slices) == 1

    section = builder.genai_config_section()
    assert section["inputs"]["anchor_row_index"] == "anchor_row_index"
    assert "q_row_map" not in section["inputs"]
    assert set(section["inputs"].values()) - {"past_key_values.%d.key", "past_key_values.%d.value"} <= set(input_names)
    assert section["aux_hidden_state_layers"] == [1, 3]


def test_q8_lm_head_keeps_fp16_scales_in_bf16_graph(tmp_path):
    draft_dir, target_dir = write_lilicorr_checkpoints(tmp_path, uint8_scale=True, q8_lm_head=True)
    builder = LiLiCorrBuilder(draft_dir, target_dir, paged_block_size=256, lm_head_precision="int8")
    builder.make_model()

    nodes = {node.name: node for node in builder.graph}
    scales = builder.graph.initializers["lm_head.MatMul.weight_scales"].const_value
    assert scales.dtype == ir.DataType.FLOAT16
    assert scales.numpy().flat[0] == np.float16(0.501953125)

    assert nodes["/lm_head/Cast"].attributes["to"].value == ir.DataType.FLOAT16
    assert nodes["/lm_head/MatMul"].outputs[0].dtype == ir.DataType.FLOAT16
    assert nodes["/lm_head/logits/Cast"].attributes["to"].value == ir.DataType.BFLOAT16


def test_lilicorr_stages_run_on_cpu(tmp_path):
    draft_dir, target_dir = write_lilicorr_checkpoints(tmp_path, uint8_scale=True, q8_lm_head=False)
    builder = LiLiCorrBuilder(draft_dir, target_dir, paged_block_size=256, lm_head_precision="bf16")
    builder.io_dtype = ir.DataType.FLOAT
    builder.external_dtype = ir.DataType.FLOAT
    builder.load_drafter_weights()
    builder.make_initializer(builder.weights["embed_tokens.weight"], "model.embed_tokens.weight", to=ir.DataType.FLOAT)

    builder.graph.inputs.append(
        builder.make_value("final_hidden", ir.DataType.FLOAT, ["num_block", builder.hidden_size])
    )
    builder.graph.inputs.append(builder.make_value("ctx_pre_norm", ir.DataType.FLOAT, ["num_ctx", builder.hidden_size]))
    builder.graph.inputs.append(builder.make_value("anchor_row_index", ir.DataType.INT32, ["batch_size"]))
    builder.ctx_pre_norm = "ctx_pre_norm"

    hsel, idx, logp, cand = builder.make_candidates("final_hidden")
    lattice = builder.make_lattice_inputs(hsel, idx, logp)
    hid4 = builder.make_lattice_transformer(lattice)
    anchor = builder.make_anchor_state()
    builder.make_lattice_scores(hid4, anchor, cand)
    builder.graph.outputs.append(builder.values["draft_candidate_ids"])
    builder.graph.outputs.append(builder.values["draft_scores"])

    proto = ir.serde.serialize_model(builder.model)
    session = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])

    batch_size = 2
    final_hidden = (
        np.random.default_rng(0)
        .standard_normal((batch_size * builder.block_size, builder.hidden_size))
        .astype(np.float32)
    )
    ctx_pre_norm = np.random.default_rng(1).standard_normal((5, builder.hidden_size)).astype(np.float32)
    anchor_row_index = np.array([2, -1], dtype=np.int32)

    candidate_ids, scores = session.run(
        None,
        {
            "final_hidden": final_hidden,
            "ctx_pre_norm": ctx_pre_norm,
            "anchor_row_index": anchor_row_index,
        },
    )
    assert candidate_ids.shape == (batch_size, builder.num_draft_tokens, builder.selector_top_k)
    assert candidate_ids.dtype == np.int32
    assert scores.shape == (batch_size, builder.num_draft_tokens, builder.selector_top_k, builder.selector_top_k)
    assert scores.dtype == np.float32
    # Slot 0 start scores are broadcast across the predecessor axis, so all rows in slot 0 are identical.
    np.testing.assert_allclose(scores[:, 0, 0, :], scores[:, 0, 1, :])
