# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Procedural ONNX graph and weight builder for Qwen3.8-Flash-Next RC5 (`qwen38_flash_graph`) + DFlash2.

Constructs `embedding.onnx`, `dflash2.onnx`, `model.onnx` (plus context-tier variants up to 256K),
and all external `.data` / `ple/` weight files directly from the two Hugging Face checkpoints
(`Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-DFlash2`) with zero dependency on any pre-exported
`.onnx` model.
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from rope import DEFAULT_ROPE, DRAFT_NAMES, TARGET_NAMES, RopeConfig, dflash_rope, target_rope

ALIGN = 4096
PLE_INT8_SCALE = 0.000703105776
STRP_MAGIC = 0x50525453

DEFAULT_KV_SCALE_FILE = Path(__file__).with_name("kv_scales_int8_per_channel.json")


def align_up(val: int, align: int = ALIGN) -> int:
    return (val + align - 1) // align * align


def set_ext_data(tensor: TensorProto, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> None:
    tensor.data_type = dtype
    del tensor.dims[:]
    tensor.dims.extend(dims)
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
        e = tensor.external_data.add()
        e.key, e.value = k, v


def make_ext_init(name: str, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> TensorProto:
    t = TensorProto()
    t.name = name
    set_ext_data(t, location, offset, length, dtype, dims)
    return t


def make_bf16_tensor(name: str, u16_arr: np.ndarray, dims: list[int]) -> TensorProto:
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.BFLOAT16
    t.dims.extend(dims)
    t.raw_data = np.ascontiguousarray(u16_arr, dtype=np.uint16).tobytes()
    return t


def load_calibrated_kv_scales(kv_scale_file: Path | None = None) -> dict[str, np.ndarray]:
    """Return `{'node_GuardedDSA_{4..15}_{k,v}_scale': float32[512]}` from `kv_scale_file` (or `DEFAULT_KV_SCALE_FILE`)."""
    scale_path = DEFAULT_KV_SCALE_FILE if kv_scale_file is None else Path(kv_scale_file)
    data = json.loads(scale_path.read_text(encoding="utf-8"))
    layer_ids = [int(x) for x in data["layer_ids"]]
    k_map = {lid: np.asarray(s, dtype=np.float32) for lid, s in zip(layer_ids, data["scales"]["k_scales"], strict=True)}
    v_map = {lid: np.asarray(s, dtype=np.float32) for lid, s in zip(layer_ids, data["scales"]["v_scales"], strict=True)}
    out: dict[str, np.ndarray] = {}
    for idx, lid in enumerate(range(3, 48, 4), start=4):
        out[f"node_GuardedDSA_{idx}_k_scale"] = k_map[lid]
        out[f"node_GuardedDSA_{idx}_v_scale"] = v_map[lid]
    return out


class BinaryBlobWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rel_name = path.name
        self.fh = open(path, "wb")  # noqa: SIM115
        self.pos = 0

    def write_aligned(self, raw: bytes, dtype: int, dims: list[int], align: int = ALIGN) -> dict:
        aligned = align_up(self.pos, align) if align > 1 else self.pos
        if aligned > self.pos:
            self.fh.write(b"\x00" * (aligned - self.pos))
            self.pos = aligned
        off = self.pos
        self.fh.write(raw)
        self.pos += len(raw)
        return {
            "location": self.rel_name,
            "offset": off,
            "length": len(raw),
            "dtype": dtype,
            "dims": list(dims),
        }

    def close(self) -> None:
        self.fh.close()


def layer_base_node_id(layer_idx: int) -> int:
    """Return the canonical ONNX node-ID base for `layer_idx` in `0..47`."""
    base = 87
    for i in range(1, layer_idx + 1):
        if i == 1:
            base += 149
        else:
            base += 147 if ((i - 1) % 4 == 3) else 109
    return base


def build_embedding_model_proto(vocab_size: int = 248320, hidden_size: int = 2560) -> onnx.ModelProto:
    """Procedurally construct `embedding.onnx` (per-row INT8 Gather + FP32 scale dequant -> BF16)."""
    i8_bytes = vocab_size * hidden_size
    s_bytes = vocab_size * 4
    inits = [
        make_ext_init(
            "model.embed_tokens.weight", "embedding.onnx.data", 0, i8_bytes, TensorProto.INT8, [vocab_size, hidden_size]
        ),
        make_ext_init(
            "model.embed_tokens.scales", "embedding.onnx.data", i8_bytes, s_bytes, TensorProto.FLOAT, [vocab_size, 1]
        ),
    ]
    nodes = [
        helper.make_node(
            "Gather", ["model.embed_tokens.weight", "input_ids"], ["embed_i8"], name="Gather_weight", axis=0
        ),
        helper.make_node("Cast", ["embed_i8"], ["embed_f32"], name="Cast_weight_f32", to=TensorProto.FLOAT),
        helper.make_node(
            "Gather", ["model.embed_tokens.scales", "input_ids"], ["scale_f32"], name="Gather_scale", axis=0
        ),
        helper.make_node("Mul", ["embed_f32", "scale_f32"], ["out_f32"], name="Mul_dequant"),
        helper.make_node("Cast", ["out_f32"], ["inputs_embeds"], name="Cast_out_bf16", to=TensorProto.BFLOAT16),
    ]
    graph = helper.make_graph(
        nodes,
        "int8_embedding_graph",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.BFLOAT16, ["num_tokens", hidden_size])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def build_dflash2_model_proto(
    tensor_locations: dict[str, dict],
    base_kernels_u16: dict[str, np.ndarray],
    *,
    block_size: int = 8,
    num_draft_tokens: int = 7,
    selector_top_k: int = 16,
    sliding_window: int = 2048,
) -> onnx.ModelProto:
    """Procedurally construct the 310-node `dflash2.onnx` block-diffusion speculative drafter graph."""
    eps = 9.999999974752427e-07

    def make_const(name: str, arr) -> onnx.NodeProto:
        return helper.make_node(
            "Constant",
            [],
            [name],
            name=f"{name}/Constant",
            value=numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name),
        )

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Cast",
            ["aux_hidden_states"],
            ["/dflash2/aux/Cast/output_0"],
            name="/dflash2/aux/Cast",
            to=TensorProto.BFLOAT16,
        ),
        helper.make_node(
            "MatMulNBits",
            ["/dflash2/aux/Cast/output_0", "dflash2.fc.MatMul.weight.qweight", "dflash2.fc.MatMul.weight.scales"],
            ["/dflash2/fc/MatMul/output_0"],
            name="/dflash2/fc/MatMul",
            domain="com.microsoft",
            K=12800,
            N=2560,
            accuracy_level=0,
            bits=4,
            block_size=32,
            weight_prepacked=1,
        ),
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["/dflash2/fc/MatMul/output_0", "dflash2.hidden_norm.weight"],
            ["/dflash2/hidden_norm/output_0"],
            name="/dflash2/hidden_norm",
            axis=-1,
            epsilon=eps,
            stash_type=1,
        ),
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["/dflash2/embed_tokens/Gather_i8/output_0"],
            name="/dflash2/embed_tokens/Gather_i8",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight.row_scales", "input_ids"],
            ["/dflash2/embed_tokens/Gather_scale/output_0"],
            name="/dflash2/embed_tokens/Gather_scale",
            axis=0,
        ),
        helper.make_node(
            "Cast",
            ["/dflash2/embed_tokens/Gather_i8/output_0"],
            ["/dflash2/embed_tokens/Cast_f32/output_0"],
            name="/dflash2/embed_tokens/Cast_f32",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            ["/dflash2/embed_tokens/Cast_f32/output_0", "/dflash2/embed_tokens/Gather_scale/output_0"],
            ["/dflash2/embed_tokens/Mul_dequant/output_0"],
            name="/dflash2/embed_tokens/Mul_dequant",
        ),
        helper.make_node(
            "Cast",
            ["/dflash2/embed_tokens/Mul_dequant/output_0"],
            ["/dflash2/embed_tokens/Cast/output_0"],
            name="/dflash2/embed_tokens/Cast",
            to=TensorProto.BFLOAT16,
        ),
        make_const("dflash2.const.0", 0),
        make_const("dflash2.const.1", 1),
        make_const("dflash2.const.2", block_size),
        make_const("dflash2.const.3", [-1, 1]),
    ]

    def emit_conv_prepare(lid: int, kind: str, norm_out: str) -> tuple[str, str, str]:
        pfx = f"/dflash2/layers.{lid}/{kind}"
        mm_out = f"{pfx}/kernel_projection/MatMul/output_0"
        r4d = f"{pfx}/kernel_projection/Reshape4D_out"
        add4d = f"{pfx}/kernel_projection/AddAll_out"
        masked4d = f"{pfx}/kernel_projection/MaskTap1_out"
        r3d = f"{pfx}/kernel_projection/Reshape3D_out"
        s0, s1, s2, s3 = [f"{pfx}/kernel_projection/SplitAll_{k}" for k in range(4)]
        c_p0 = f"{pfx}/prepare/coef0/Reshape/output_0"
        c_p1 = f"{pfx}/prepare/coef1/Reshape/output_0"
        c_f0 = f"{pfx}/finish/coef0/Reshape/output_0"
        c_f1 = f"{pfx}/finish/coef1/Reshape/output_0"
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [norm_out, f"dflash2.layers.{lid}.{kind}.kernel_projection.MatMul.weight"],
                    [mm_out],
                    name=f"{pfx}/kernel_projection/MatMul",
                ),
                helper.make_node(
                    "Reshape", [mm_out, "dflash2.shape_m1_4_160_1"], [r4d], name=f"{pfx}/kernel_projection/Reshape4D"
                ),
                helper.make_node(
                    "Add", [f"{pfx}/base_kernel_all_1x4x160x16", r4d], [add4d], name=f"{pfx}/kernel_projection/AddAll"
                ),
                helper.make_node(
                    "Mul",
                    [add4d, "dflash2.conv_tap_mask_1x8x4x1x1"],
                    [masked4d],
                    name=f"{pfx}/kernel_projection/MaskTap1",
                ),
                helper.make_node(
                    "Reshape", [masked4d, "dflash2.shape_m1_4_2560"], [r3d], name=f"{pfx}/kernel_projection/Reshape3D"
                ),
                helper.make_node(
                    "Split",
                    [r3d, "dflash2.split_4x1"],
                    [s0, s1, s2, s3],
                    name=f"{pfx}/kernel_projection/SplitAll",
                    axis=1,
                ),
                helper.make_node(
                    "Reshape", [s0, "dflash2.shape_m1_2560"], [c_p0], name=f"{pfx}/prepare/coef0/Reshape2D"
                ),
                helper.make_node(
                    "Reshape", [s1, "dflash2.shape_m1_2560"], [c_p1], name=f"{pfx}/prepare/coef1/Reshape2D"
                ),
                helper.make_node(
                    "Reshape", [s2, "dflash2.shape_m1_2560"], [c_f0], name=f"{pfx}/finish/coef0/Reshape2D"
                ),
                helper.make_node(
                    "Reshape", [s3, "dflash2.shape_m1_2560"], [c_f1], name=f"{pfx}/finish/coef1/Reshape2D"
                ),
            ]
        )
        if lid == 0 and kind == "attention_conv":
            nodes.extend(
                [
                    make_const("dflash2.const.4", [-1, 4, 160]),
                    make_const("dflash2.const.5", [-1, 160, 1]),
                    make_const("dflash2.const.6", [-1, 2560]),
                ]
            )
        nodes.append(
            helper.make_node(
                "Mul", [norm_out, c_p0], [f"{pfx}/prepare/term0/Mul/output_0"], name=f"{pfx}/prepare/term0/Mul"
            )
        )
        if lid == 0 and kind == "attention_conv":
            nodes.extend(
                [
                    make_const("dflash2.const.7", [0]),
                    make_const("dflash2.const.8", [1]),
                    make_const("dflash2.const.9", [-1]),
                ]
            )
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [norm_out, "dflash2.const.7", "dflash2.const.9", "dflash2.const.7"],
                    [f"{pfx}/prepare/shift1/Slice/output_0"],
                    name=f"{pfx}/prepare/shift1/Slice",
                ),
                helper.make_node(
                    "Concat",
                    ["dflash2.zero_row_1x2560", f"{pfx}/prepare/shift1/Slice/output_0"],
                    [f"{pfx}/prepare/shift1/Concat/output_0"],
                    name=f"{pfx}/prepare/shift1/Concat",
                    axis=0,
                ),
                helper.make_node(
                    "Mul",
                    [f"{pfx}/prepare/shift1/Concat/output_0", c_p1],
                    [f"{pfx}/prepare/term1/Mul/output_0"],
                    name=f"{pfx}/prepare/term1/Mul",
                ),
                helper.make_node(
                    "Add",
                    [f"{pfx}/prepare/term0/Mul/output_0", f"{pfx}/prepare/term1/Mul/output_0"],
                    [f"{pfx}/prepare/sum1/Add/output_0"],
                    name=f"{pfx}/prepare/sum1/Add",
                ),
            ]
        )
        return f"{pfx}/prepare/sum1/Add/output_0", c_f0, c_f1

    def emit_conv_finish(lid: int, kind: str, body_out: str, c_f0: str, c_f1: str) -> str:
        pfx = f"/dflash2/layers.{lid}/{kind}"
        nodes.extend(
            [
                helper.make_node(
                    "Mul", [body_out, c_f0], [f"{pfx}/finish/term0/Mul/output_0"], name=f"{pfx}/finish/term0/Mul"
                ),
                helper.make_node(
                    "Slice",
                    [body_out, "dflash2.const.7", "dflash2.const.9", "dflash2.const.7"],
                    [f"{pfx}/finish/shift1/Slice/output_0"],
                    name=f"{pfx}/finish/shift1/Slice",
                ),
                helper.make_node(
                    "Concat",
                    ["dflash2.zero_row_1x2560", f"{pfx}/finish/shift1/Slice/output_0"],
                    [f"{pfx}/finish/shift1/Concat/output_0"],
                    name=f"{pfx}/finish/shift1/Concat",
                    axis=0,
                ),
                helper.make_node(
                    "Mul",
                    [f"{pfx}/finish/shift1/Concat/output_0", c_f1],
                    [f"{pfx}/finish/term1/Mul/output_0"],
                    name=f"{pfx}/finish/term1/Mul",
                ),
                helper.make_node(
                    "Add",
                    [f"{pfx}/finish/term0/Mul/output_0", f"{pfx}/finish/term1/Mul/output_0"],
                    [f"{pfx}/finish/sum1/Add/output_0"],
                    name=f"{pfx}/finish/sum1/Add",
                ),
            ]
        )
        return f"{pfx}/finish/sum1/Add/output_0"

    prev_res = "/dflash2/embed_tokens/Cast/output_0"
    prev_mlp_sum = ""
    for lid in range(5):
        in_norm = f"/dflash2/layers.{lid}/input_layernorm/output_0"
        if lid == 0:
            nodes.append(
                helper.make_node(
                    "SimplifiedLayerNormalization",
                    [prev_res, "dflash2.layers.0.input_layernorm.weight"],
                    [in_norm],
                    name="/dflash2/layers.0/input_layernorm",
                    axis=-1,
                    epsilon=eps,
                    stash_type=1,
                )
            )
            attn_res_base = prev_res
        else:
            attn_res_base = f"/dflash2/layers.{lid}/input_layernorm/output_3"
            nodes.append(
                helper.make_node(
                    "SkipSimplifiedLayerNormalization",
                    [prev_res, prev_mlp_sum, f"dflash2.layers.{lid}.input_layernorm.weight"],
                    [in_norm, "", "", attn_res_base],
                    name=f"/dflash2/layers.{lid}/input_layernorm",
                    domain="com.microsoft",
                    epsilon=eps,
                )
            )

        prep_attn, cf0, cf1 = emit_conv_prepare(lid, "attention_conv", in_norm)
        nodes.extend(
            [
                helper.make_node(
                    "Concat",
                    [prep_attn, "/dflash2/hidden_norm/output_0"],
                    [f"/dflash2/layers.{lid}/attn/qkv_rows/Concat/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/qkv_rows/Concat",
                    axis=0,
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"/dflash2/layers.{lid}/attn/qkv_rows/Concat/output_0",
                        f"dflash2.layers.{lid}.attn.qkv_proj.MatMul.weight.qweight",
                        f"dflash2.layers.{lid}.attn.qkv_proj.MatMul.weight.scales",
                    ],
                    [f"/dflash2/layers.{lid}/attn/qkv_proj/MatMul/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/qkv_proj/MatMul",
                    domain="com.microsoft",
                    K=2560,
                    N=6144,
                    accuracy_level=0,
                    bits=4,
                    block_size=32,
                    weight_prepacked=1,
                ),
                helper.make_node(
                    "Gather",
                    [f"/dflash2/layers.{lid}/attn/qkv_proj/MatMul/output_0", "qkv_row_map"],
                    [f"/dflash2/layers.{lid}/attn/qkv_gather/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/qkv_gather",
                ),
                helper.make_node(
                    "PagedAttention",
                    [
                        f"/dflash2/layers.{lid}/attn/qkv_gather/output_0",
                        "",
                        "",
                        f"past_key_values.{lid}.key",
                        f"past_key_values.{lid}.value",
                        "cumulative_sequence_lengths",
                        "past_sequence_lengths",
                        "block_table",
                        "dflash2.cos_cache",
                        "dflash2.sin_cache",
                        "",
                        "",
                        f"dflash2.layers.{lid}.self_attn.q_norm.weight",
                        f"dflash2.layers.{lid}.self_attn.k_norm.weight",
                        "",
                        "",
                        "attention_metadata",
                    ],
                    [
                        f"/dflash2/layers.{lid}/attn/PagedAttention/output_0",
                        f"present.{lid}.key",
                        f"present.{lid}.value",
                    ],
                    name=f"/dflash2/layers.{lid}/attn/PagedAttention",
                    domain="com.microsoft",
                    num_heads=32,
                    kv_num_heads=8,
                    local_window_size=sliding_window,
                    is_causal=0,
                    do_rotary=1,
                    rotary_interleaved=0,
                    qk_norm_epsilon=eps,
                ),
                helper.make_node(
                    "Gather",
                    [f"/dflash2/layers.{lid}/attn/PagedAttention/output_0", "block_row_index"],
                    [f"/dflash2/layers.{lid}/attn/out_gather/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/out_gather",
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"/dflash2/layers.{lid}/attn/out_gather/output_0",
                        f"dflash2.layers.{lid}.attn.o_proj.MatMul.weight.qweight",
                        f"dflash2.layers.{lid}.attn.o_proj.MatMul.weight.scales",
                    ],
                    [f"/dflash2/layers.{lid}/attn/o_proj/MatMul/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/o_proj/MatMul",
                    domain="com.microsoft",
                    K=4096,
                    N=2560,
                    accuracy_level=0,
                    bits=4,
                    block_size=32,
                    weight_prepacked=1,
                ),
            ]
        )
        attn_sum = emit_conv_finish(
            lid, "attention_conv", f"/dflash2/layers.{lid}/attn/o_proj/MatMul/output_0", cf0, cf1
        )
        post_norm = f"/dflash2/layers.{lid}/post_attention_layernorm/output_0"
        post_res = f"/dflash2/layers.{lid}/post_attention_layernorm/output_3"
        nodes.append(
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                [attn_res_base, attn_sum, f"dflash2.layers.{lid}.post_attention_layernorm.weight"],
                [post_norm, "", "", post_res],
                name=f"/dflash2/layers.{lid}/post_attention_layernorm",
                domain="com.microsoft",
                epsilon=eps,
            )
        )

        prep_mlp, m_cf0, m_cf1 = emit_conv_prepare(lid, "mlp_conv", post_norm)
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                [
                    prep_mlp,
                    f"dflash2.layers.{lid}.mlp.gate_up_proj.MatMul.weight.qweight",
                    f"dflash2.layers.{lid}.mlp.gate_up_proj.MatMul.weight.scales",
                ],
                [f"/dflash2/layers.{lid}/mlp/gate_up_proj/MatMul/output_0"],
                name=f"/dflash2/layers.{lid}/mlp/gate_up_proj/MatMul",
                domain="com.microsoft",
                K=2560,
                N=15360,
                accuracy_level=0,
                bits=4,
                block_size=32,
                weight_prepacked=1,
            )
        )
        if lid == 0:
            nodes.append(make_const("dflash2.const.10", [7680, 7680]))
        nodes.extend(
            [
                helper.make_node(
                    "SwiGLU",
                    [f"/dflash2/layers.{lid}/mlp/gate_up_proj/MatMul/output_0"],
                    [f"/dflash2/layers.{lid}/mlp/act/MulUp/output_0"],
                    name=f"/dflash2/layers.{lid}/mlp/act/SwiGLU",
                    domain="com.microsoft",
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"/dflash2/layers.{lid}/mlp/act/MulUp/output_0",
                        f"dflash2.layers.{lid}.mlp.down_proj.MatMul.weight.qweight",
                        f"dflash2.layers.{lid}.mlp.down_proj.MatMul.weight.scales",
                    ],
                    [f"/dflash2/layers.{lid}/mlp/down_proj/MatMul/output_0"],
                    name=f"/dflash2/layers.{lid}/mlp/down_proj/MatMul",
                    domain="com.microsoft",
                    K=7680,
                    N=2560,
                    accuracy_level=0,
                    bits=4,
                    block_size=32,
                    weight_prepacked=1,
                ),
            ]
        )
        prev_mlp_sum = emit_conv_finish(
            lid, "mlp_conv", f"/dflash2/layers.{lid}/mlp/down_proj/MatMul/output_0", m_cf0, m_cf1
        )
        prev_res = post_res

    nodes.extend(
        [
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                [prev_res, prev_mlp_sum, "dflash2.norm.weight"],
                ["/dflash2/norm/output_0"],
                name="/dflash2/norm",
                domain="com.microsoft",
                epsilon=eps,
            ),
            make_const("dflash2.const.11", [-1, block_size, 2560]),
            helper.make_node(
                "Reshape",
                ["/dflash2/norm/output_0", "dflash2.const.11"],
                ["/dflash2/select/Reshape/output_0"],
                name="/dflash2/select/Reshape",
            ),
            make_const("dflash2.const.12", [block_size]),
            helper.make_node(
                "Slice",
                ["/dflash2/select/Reshape/output_0", "dflash2.const.8", "dflash2.const.12", "dflash2.const.8"],
                ["/dflash2/select/Slice/output_0"],
                name="/dflash2/select/Slice",
            ),
            helper.make_node(
                "Reshape",
                ["/dflash2/select/Slice/output_0", "dflash2.const.6"],
                ["/dflash2/select/Flatten/output_0"],
                name="/dflash2/select/Flatten",
            ),
            helper.make_node(
                "MatMulNBits",
                ["/dflash2/select/Flatten/output_0", "lm_head.weight_t.qweight", "lm_head.weight_t.scales"],
                ["/lm_head/MatMul/output_0"],
                name="/lm_head/MatMul",
                domain="com.microsoft",
                K=2560,
                N=248320,
                accuracy_level=0,
                bits=8,
                block_size=32,
                weight_prepacked=1,
            ),
            helper.make_node(
                "Cast",
                ["/lm_head/MatMul/output_0"],
                ["/dflash2/topk/Cast/output_0"],
                name="/dflash2/topk/Cast",
                to=TensorProto.FLOAT,
            ),
            make_const("dflash2.const.13", [selector_top_k]),
            helper.make_node(
                "TopK",
                ["/dflash2/topk/Cast/output_0", "dflash2.const.13"],
                ["/dflash2/topk/values", "/dflash2/topk/indices"],
                name="/dflash2/topk/TopK",
                axis=-1,
                largest=1,
                sorted=1,
            ),
            make_const("dflash2.const.14", [-1, num_draft_tokens, selector_top_k]),
            helper.make_node(
                "Reshape",
                ["/dflash2/topk/indices", "dflash2.const.14"],
                ["/dflash2/topk/cand/output_0"],
                name="/dflash2/topk/cand",
            ),
            helper.make_node(
                "Reshape",
                ["/dflash2/topk/values", "dflash2.const.14"],
                ["/dflash2/topk/unary/output_0"],
                name="/dflash2/topk/unary",
            ),
            helper.make_node(
                "MatMul",
                ["/dflash2/select/Flatten/output_0", "dflash2.selector.hidden_projection.MatMul.weight"],
                ["/dflash2/selector/hidden_projection/MatMul/output_0"],
                name="/dflash2/selector/hidden_projection/MatMul",
            ),
            make_const("dflash2.const.15", [-1, num_draft_tokens, 1, 256]),
            helper.make_node(
                "Reshape",
                ["/dflash2/selector/hidden_projection/MatMul/output_0", "dflash2.const.15"],
                ["/dflash2/selector/hp/output_0"],
                name="/dflash2/selector/hp",
            ),
            make_const("dflash2.const.16", [-1, block_size]),
            helper.make_node(
                "Reshape",
                ["input_ids", "dflash2.const.16"],
                ["/dflash2/selector/ids/output_0"],
                name="/dflash2/selector/ids",
            ),
            helper.make_node(
                "Slice",
                ["/dflash2/selector/ids/output_0", "dflash2.const.7", "dflash2.const.8", "dflash2.const.8"],
                ["/dflash2/selector/anchor/output_0"],
                name="/dflash2/selector/anchor",
            ),
            make_const("dflash2.const.17", [-1, 1, 1]),
            helper.make_node(
                "Reshape",
                ["/dflash2/selector/anchor/output_0", "dflash2.const.17"],
                ["/dflash2/selector/anchor3/output_0"],
                name="/dflash2/selector/anchor3",
            ),
            make_const("dflash2.const.18", [1, 1, selector_top_k]),
            helper.make_node(
                "Tile",
                ["/dflash2/selector/anchor3/output_0", "dflash2.const.18"],
                ["/dflash2/selector/anchor_tile/output_0"],
                name="/dflash2/selector/anchor_tile",
            ),
            make_const("dflash2.const.19", [num_draft_tokens - 1]),
            helper.make_node(
                "Slice",
                ["/dflash2/topk/cand/output_0", "dflash2.const.7", "dflash2.const.19", "dflash2.const.8"],
                ["/dflash2/selector/prev_tail/output_0"],
                name="/dflash2/selector/prev_tail",
            ),
            helper.make_node(
                "Concat",
                ["/dflash2/selector/anchor_tile/output_0", "/dflash2/selector/prev_tail/output_0"],
                ["/dflash2/selector/prev/output_0"],
                name="/dflash2/selector/prev",
                axis=1,
            ),
            helper.make_node(
                "Gather",
                ["dflash2.candidate_selector.predecessor_codebook", "/dflash2/selector/prev/output_0"],
                ["/dflash2/selector/pred_gather/output_0"],
                name="/dflash2/selector/pred_gather",
            ),
            helper.make_node(
                "Gather",
                ["dflash2.candidate_selector.successor_codebook", "/dflash2/topk/cand/output_0"],
                ["/dflash2/selector/succ_gather/output_0"],
                name="/dflash2/selector/succ_gather",
            ),
            helper.make_node(
                "Mul",
                ["/dflash2/selector/pred_gather/output_0", "/dflash2/selector/hp/output_0"],
                ["/dflash2/selector/pw/output_0"],
                name="/dflash2/selector/pw",
            ),
            helper.make_node(
                "Transpose",
                ["/dflash2/selector/succ_gather/output_0"],
                ["/dflash2/selector/succ_t/output_0"],
                name="/dflash2/selector/succ_t",
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "MatMul",
                ["/dflash2/selector/pw/output_0", "/dflash2/selector/succ_t/output_0"],
                ["/dflash2/selector/pair/output_0"],
                name="/dflash2/selector/pair",
            ),
            helper.make_node(
                "Cast",
                ["/dflash2/selector/pair/output_0"],
                ["/dflash2/selector/pair_cast/output_0"],
                name="/dflash2/selector/pair_cast",
                to=TensorProto.FLOAT,
            ),
            make_const("dflash2.const.20", [-1, num_draft_tokens, 1, selector_top_k]),
            helper.make_node(
                "Reshape",
                ["/dflash2/topk/unary/output_0", "dflash2.const.20"],
                ["/dflash2/selector/unary4/output_0"],
                name="/dflash2/selector/unary4",
            ),
            helper.make_node(
                "Add",
                ["/dflash2/selector/pair_cast/output_0", "/dflash2/selector/unary4/output_0"],
                ["draft_scores"],
                name="/dflash2/selector/scores",
            ),
            helper.make_node(
                "Cast",
                ["/dflash2/topk/cand/output_0"],
                ["draft_candidate_ids"],
                name="/dflash2/selector/cand_cast",
                to=TensorProto.INT32,
            ),
        ]
    )

    inputs = [
        helper.make_tensor_value_info("aux_hidden_states", TensorProto.BFLOAT16, ["num_ctx", 12800]),
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_block"]),
        helper.make_tensor_value_info("qkv_row_map", TensorProto.INT32, ["num_tokens"]),
        helper.make_tensor_value_info("block_row_index", TensorProto.INT32, ["num_block"]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_size + 1"]),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch_size"]),
        helper.make_tensor_value_info("block_table", TensorProto.INT32, ["batch_size", "max_num_blocks"]),
        helper.make_tensor_value_info("attention_metadata", TensorProto.INT32, [3]),
    ]
    for lid in range(5):
        inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{lid}.key", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128]
            )
        )
        inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{lid}.value", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128]
            )
        )

    outputs = [
        helper.make_tensor_value_info(
            "draft_candidate_ids", TensorProto.INT32, ["batch_size", num_draft_tokens, selector_top_k]
        ),
        helper.make_tensor_value_info(
            "draft_scores", TensorProto.FLOAT, ["batch_size", num_draft_tokens, selector_top_k, selector_top_k]
        ),
    ]
    for lid in range(5):
        outputs.append(
            helper.make_tensor_value_info(f"present.{lid}.key", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128])
        )
        outputs.append(
            helper.make_tensor_value_info(f"present.{lid}.value", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128])
        )

    mask_f32 = np.ones((1, block_size, 4, 1, 1), dtype=np.float32)
    mask_f32[0, 0, 1, 0, 0] = 0.0
    mask_f32[0, 0, 3, 0, 0] = 0.0
    mask_u16 = torch.from_numpy(mask_f32).to(torch.bfloat16).view(torch.uint16).numpy()

    ext_names = sorted(
        k for k in tensor_locations if k.startswith(("dflash2.", "/dflash2/", "lm_head.", "model.embed_tokens."))
    )
    inits: list[TensorProto] = []
    for name in ext_names:
        if name == "dflash2.conv_mask_f32":
            inits.append(
                numpy_helper.from_array(
                    np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32), name="dflash2.conv_mask_f32"
                )
            )
            continue
        meta = tensor_locations[name]
        inits.append(make_ext_init(name, meta["location"], meta["offset"], meta["length"], meta["dtype"], meta["dims"]))
    if "dflash2.conv_mask_f32" not in tensor_locations:
        inits.append(
            numpy_helper.from_array(
                np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32), name="dflash2.conv_mask_f32"
            )
        )

    inits.extend(
        [
            make_bf16_tensor("dflash2.zero_row_1x2560", np.zeros((1, 2560), dtype=np.uint16), [1, 2560]),
            numpy_helper.from_array(
                np.array([-1, block_size, 4, 160, 1], dtype=np.int64), name="dflash2.shape_m1_4_160_1"
            ),
            numpy_helper.from_array(np.array([-1, 4, 2560], dtype=np.int64), name="dflash2.shape_m1_4_2560"),
            numpy_helper.from_array(np.array([-1, 2560], dtype=np.int64), name="dflash2.shape_m1_2560"),
            numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="dflash2.split_4x1"),
        ]
    )
    for lid in range(5):
        for kind in ("attention_conv", "mlp_conv"):
            bname = f"/dflash2/layers.{lid}/{kind}/base_kernel_all_1x4x160x16"
            inits.append(make_bf16_tensor(bname, base_kernels_u16[bname], [1, 1, 4, 160, 16]))
    inits.append(make_bf16_tensor("dflash2.conv_tap_mask_1x8x4x1x1", mask_u16, [1, block_size, 4, 1, 1]))

    graph = helper.make_graph(nodes, "dflash2_graph", inputs, outputs, initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="onnxruntime-genai",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


def build_target_model_proto(
    tensor_locations: dict[str, dict],
    embedded_weights: dict[str, TensorProto],
    kv_scales: dict[str, np.ndarray] | None = None,
    *,
    kv_capacity: int | str = "kv_capacity",
    state_capacity: int | str = "state_capacity",
    max_total_sequence_length: int = 262152,
    indexer_state_capacity_attr: int = 8259,
    verify_rows: int = 8,
    state_update_capacity: int = 7,
) -> onnx.ModelProto:
    """Procedurally construct the unified 958-node Qwen3.8-Flash-Next RC5 `model.onnx` target graph.

    By default, cache capacity dimensions (`kv_capacity` and `state_capacity`) are symbolic and
    `contrib_const_dsa_total_seq` is set to `max_total_sequence_length` (`262152`), allowing a single
    unified `model.onnx` to serve 32K (`num_blocks=142`), 128K (`num_blocks=525`), and 256K (`num_blocks=1037`).
    """
    if kv_scales is None:
        kv_scales = load_calibrated_kv_scales()
    eff_indexer_state_cap = int(state_capacity) if isinstance(state_capacity, int) else int(indexer_state_capacity_attr)
    eff_dsa_total_seq = int(kv_capacity) if isinstance(kv_capacity, int) else int(max_total_sequence_length)

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Add",
            ["committed_length", "sequence_length_minus_one"],
            ["capture.seqlens_k"],
            name="capture/Add_seqlens_k",
        ),
        helper.make_node(
            "Concat",
            ["inputs_embeds", "inputs_embeds", "inputs_embeds", "inputs_embeds"],
            ["v_model.Concat_8"],
            name="model/Concat_node_8",
            axis=-1,
        ),
        helper.make_node("Identity", ["past_position_ids"], ["present_position_ids"], name="node_Identity_16"),
    ]

    base = 87
    prev_streams = "v_model.Concat_8"
    prev_mlp_add = ""
    prev_mlp_mul = ""
    prev_post_mix_id = 0
    dsa_idx = 4

    for i in range(48):
        if i == 1:
            base += 149
        elif i > 1:
            base += 147 if ((i - 1) % 4 == 3) else 109

        sg_base = 0 if i == 0 else (5 + 4 * (i - 1))
        mp_base = 0 if i == 0 else (3 + 2 * (i - 1))

        if i == 1:
            nodes.extend(
                [
                    helper.make_node(
                        "Concat",
                        ["past_key_values.1.ple_context", "input_ids"],
                        ["v_model.layers.1.ple.ple_embedding.Concat_195"],
                        name="model/layers.1/ple/ple_embedding/Concat_node_195",
                        axis=1,
                    ),
                    helper.make_node(
                        "Slice",
                        ["v_model.layers.1.ple.ple_embedding.Concat_195", "const_1d_0", "const_1d_1", "const_1d_3"],
                        ["present.1.ple_context"],
                        name="model/layers.1/ple/ple_embedding/Slice_node_196",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        ["ple_embeddings.1", "merged_proj.2.weight_t.qweight", "merged_proj.2.weight_t.scales"],
                        ["skinny_gemm/4.0/merged_out"],
                        name="skinny_gemm/4.0/MatMulNBits",
                        domain="com.microsoft",
                        K=2560,
                        N=12800,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                    helper.make_node(
                        "Split",
                        ["skinny_gemm/4.0/merged_out", "skinny_gemm/4.0/split_lengths"],
                        ["v_model.layers.1.ple.key_proj.MatMul_198", "v_model.layers.1.ple.value_proj.MatMul_208"],
                        name="skinny_gemm/4.0/Split",
                        axis=-1,
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["v_model.layers.1.ple.key_proj.MatMul_198", "v_model.layers.1.ple.norm_key.Add_200.folded"],
                        ["v_model.layers.1.ple.norm_key.BranchwiseRMSNorm_201"],
                        name="model/layers.1/ple/norm_key/BranchwiseRMSNorm_node_201",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        num_branches=4,
                    ),
                    helper.make_node(
                        "Reshape",
                        ["v_model.layers.1.ple.norm_key.BranchwiseRMSNorm_201", "const_1d_19"],
                        ["v_model.layers.1.ple.Reshape_202"],
                        name="model/layers.1/ple/Reshape_node_202",
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        [
                            "v_model.layers.0.HyperConnectionPostMix_194",
                            "v_model.layers.1.ple.norm_query.Add_204.folded",
                        ],
                        ["v_model.layers.1.ple.norm_query.BranchwiseRMSNorm_205"],
                        name="model/layers.1/ple/norm_query/BranchwiseRMSNorm_node_205",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        num_branches=4,
                    ),
                    helper.make_node(
                        "Reshape",
                        ["v_model.layers.1.ple.norm_query.BranchwiseRMSNorm_205", "const_1d_19"],
                        ["v_model.layers.1.ple.Reshape_206"],
                        name="model/layers.1/ple/Reshape_node_206",
                    ),
                    helper.make_node(
                        "Mul",
                        ["v_model.layers.1.ple.Reshape_202", "v_model.layers.1.ple.Reshape_206"],
                        ["v_model.layers.1.ple.Mul_209"],
                        name="model/layers.1/ple/Mul_node_209",
                    ),
                    helper.make_node(
                        "ReduceSum",
                        ["v_model.layers.1.ple.Mul_209", "const_1d_11"],
                        ["v_model.layers.1.ple.ReduceSum_210"],
                        name="model/layers.1/ple/ReduceSum_node_210",
                        keepdims=1,
                    ),
                    helper.make_node(
                        "CastLike",
                        ["const_50.59644256269407_f32", "v_model.layers.1.ple.ReduceSum_210"],
                        ["v_model.layers.1.ple.CastLike_211"],
                        name="model/layers.1/ple/CastLike_node_211",
                    ),
                    helper.make_node(
                        "Div",
                        ["v_model.layers.1.ple.ReduceSum_210", "v_model.layers.1.ple.CastLike_211"],
                        ["v_model.layers.1.ple.Div_212"],
                        name="model/layers.1/ple/Div_node_212",
                    ),
                    helper.make_node(
                        "Sign",
                        ["v_model.layers.1.ple.Div_212"],
                        ["v_model.layers.1.ple.Sign_213"],
                        name="model/layers.1/ple/Sign_node_213",
                    ),
                    helper.make_node(
                        "Abs",
                        ["v_model.layers.1.ple.Div_212"],
                        ["v_model.layers.1.ple.Abs_214"],
                        name="model/layers.1/ple/Abs_node_214",
                    ),
                    helper.make_node(
                        "CastLike",
                        ["const_1e-06_f32", "v_model.layers.1.ple.Div_212"],
                        ["v_model.layers.1.ple.CastLike_215"],
                        name="model/layers.1/ple/CastLike_node_215",
                    ),
                    helper.make_node(
                        "Max",
                        ["v_model.layers.1.ple.Abs_214", "v_model.layers.1.ple.CastLike_215"],
                        ["v_model.layers.1.ple.Max_216"],
                        name="model/layers.1/ple/Max_node_216",
                    ),
                    helper.make_node(
                        "Sqrt",
                        ["v_model.layers.1.ple.Max_216"],
                        ["v_model.layers.1.ple.Sqrt_217"],
                        name="model/layers.1/ple/Sqrt_node_217",
                    ),
                    helper.make_node(
                        "Mul",
                        ["v_model.layers.1.ple.Sign_213", "v_model.layers.1.ple.Sqrt_217"],
                        ["v_model.layers.1.ple.Mul_218"],
                        name="model/layers.1/ple/Mul_node_218",
                    ),
                    helper.make_node(
                        "Sigmoid",
                        ["v_model.layers.1.ple.Mul_218"],
                        ["v_model.layers.1.ple.Sigmoid_219"],
                        name="model/layers.1/ple/Sigmoid_node_219",
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        ["v_model.layers.1.ple.value_proj.MatMul_208", "const_1d_0"],
                        ["v_model.layers.1.ple.Unsqueeze_220"],
                        name="model/layers.1/ple/Unsqueeze_node_220",
                    ),
                    helper.make_node(
                        "Mul",
                        ["v_model.layers.1.ple.Sigmoid_219", "v_model.layers.1.ple.Unsqueeze_220"],
                        ["v_model.layers.1.ple.Mul_221"],
                        name="model/layers.1/ple/Mul_node_221",
                    ),
                    helper.make_node(
                        "Reshape",
                        ["v_model.layers.1.ple.Mul_221", "const_1d_21"],
                        ["v_model.layers.1.ple.Reshape_222"],
                        name="model/layers.1/ple/Reshape_node_222",
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["v_model.layers.1.ple.Reshape_222", "v_model.layers.1.ple.norm_conv.Add_224.folded"],
                        ["v_model.layers.1.ple.norm_conv.BranchwiseRMSNorm_225"],
                        name="model/layers.1/ple/norm_conv/BranchwiseRMSNorm_node_225",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        num_branches=4,
                    ),
                    helper.make_node(
                        "CausalConvWithState",
                        [
                            "v_model.layers.1.ple.norm_conv.BranchwiseRMSNorm_225",
                            "model.layers.1.ple.conv1d.weight",
                            "",
                            "past_key_values.1.ple_conv_state",
                        ],
                        [
                            "v_model.layers.1.ple.conv1d.CausalConvWithState_231_0",
                            "present.1.ple_conv_state",
                            "state_update.1.ple_conv_value",
                        ],
                        name="model/layers.1/ple/conv1d/CausalConvWithState_node_231",
                        domain="com.microsoft",
                        activation="silu",
                        channels_last=1,
                        dilation=3,
                        input_channel_offset=0,
                        input_channels=10240,
                        ndim=1,
                        state_channels_last=0,
                        state_length=9,
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "Add",
                        ["v_model.layers.1.ple.Reshape_222", "v_model.layers.1.ple.conv1d.CausalConvWithState_231_0"],
                        ["v_model.layers.1.ple.Add_233"],
                        name="model/layers.1/ple/Add_node_233",
                    ),
                    helper.make_node(
                        "Add",
                        ["v_model.layers.0.HyperConnectionPostMix_194", "v_model.layers.1.ple.Add_233"],
                        ["v_model.layers.1.Add_234"],
                        name="model/layers.1/Add_node_234",
                    ),
                ]
            )
            prev_streams = "v_model.layers.1.Add_234"

        attn_pre = f"v_model.layers.{i}.attn_hyper_connection.HyperConnectionPreMix_{base + 8}"
        attn_mul = f"v_model.layers.{i}.attn_hyper_connection.Mul_{base + 15}"
        hc_in = [
            prev_streams,
            f"v_model.layers.{i}.attn_hyper_connection.hc_norm.Add_{base}.folded",
            f"model.layers.{i}.attn_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
            f"model.layers.{i}.attn_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
            f"model.layers.{i}.attn_hyper_connection.block_inject_weight.weight_t",
        ]
        if i in (0, 1):
            nodes.append(
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    hc_in,
                    [attn_pre, attn_mul],
                    name=f"model/layers.{i}/attn_hyper_connection/HcBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    epsilon=1e-6,
                    gate_div=4.0,
                    gate_mul=2.0,
                    num_branches=4,
                    reduction_scale=0.25,
                    silu_alpha=0.25,
                )
            )
            cur_streams = prev_streams
        else:
            cur_streams = f"v_model.layers.{i - 1}.HyperConnectionPostMix_{prev_post_mix_id}"
            nodes.append(
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    [*hc_in, prev_mlp_add, prev_mlp_mul],
                    [attn_pre, attn_mul, cur_streams],
                    name=f"model/layers.{i}/attn_hyper_connection/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    epsilon=1e-6,
                    gate_div=4.0,
                    gate_mul=2.0,
                    num_branches=4,
                    reduction_scale=0.25,
                    silu_alpha=0.25,
                )
            )

        if i % 4 != 3:
            z_out = f"v_model.layers.{i}.linear_attn.in_proj_z.MatMul_{base + 22}"
            gdn_n = base + 52
            out_proj_out = f"v_model.layers.{i}.linear_attn.out_proj.MatMul_{base + 62}"
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [attn_pre, f"merged_proj.{mp_base}.weight_t.qweight", f"merged_proj.{mp_base}.weight_t.scales"],
                        [f"merged_proj.{mp_base}.out"],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=2560,
                        N=16480,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                    helper.make_node(
                        "Split",
                        [f"merged_proj.{mp_base}.out", "contrib_const_split_gdn"],
                        [
                            f"merged_proj.{mp_base}.out.qkv_10240",
                            z_out,
                            f"merged_proj.{mp_base}.out.a_48",
                            f"merged_proj.{mp_base}.out.b_48",
                        ],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits/SplitGdn",
                        axis=-1,
                    ),
                    helper.make_node(
                        "CausalConvWithState",
                        [
                            f"merged_proj.{mp_base}.out.qkv_10240",
                            f"model.layers.{i}.linear_attn.conv1d.weight",
                            "",
                            f"past_key_values.{i}.conv_state",
                        ],
                        [
                            f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}/GdnProjPrep/conv_qkv_out",
                            f"present.{i}.conv_state",
                            f"state_update.{i}.conv_value",
                        ],
                        name=f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}/GdnProjPrep/CausalConvWithState",
                        domain="com.microsoft",
                        activation="silu",
                        channels_last=1,
                        dilation=1,
                        input_channel_offset=0,
                        input_channels=10240,
                        ndim=1,
                        state_channels_last=0,
                        state_length=4,
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "GatedDeltaNet",
                        [
                            f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}/GdnProjPrep/conv_qkv_out",
                            "",
                            "",
                            "",
                            f"merged_proj.{mp_base}.out.b_48",
                            f"merged_proj.{mp_base}.out.a_48",
                            f"past_key_values.{i}.recurrent_state",
                            f"model.layers.{i}.linear_attn.A_log.f32",
                            f"model.layers.{i}.linear_attn.dt_bias.f32",
                            "state_update_capture_count",
                        ],
                        [
                            f"v_model.layers.{i}.linear_attn.GatedDeltaNet_{gdn_n}_0",
                            f"present.{i}.recurrent_state",
                            f"state_update.{i}.recurrent_capsule",
                        ],
                        name=f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}",
                        domain="com.microsoft",
                        beta_activation="sigmoid",
                        gate_activation="softplus",
                        qk_l2_norm=1,
                        scale=0.0,
                        update_rule="gated_delta",
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"v_model.layers.{i}.linear_attn.GatedDeltaNet_{gdn_n}_0", "const_1d_17"],
                        [f"v_model.layers.{i}.linear_attn.Reshape_{base + 54}"],
                        name=f"model/layers.{i}/linear_attn/Reshape_node_{base + 54}",
                    ),
                    helper.make_node(
                        "Reshape",
                        [z_out, "const_1d_17"],
                        [f"v_model.layers.{i}.linear_attn.Reshape_{base + 55}"],
                        name=f"model/layers.{i}/linear_attn/Reshape_node_{base + 55}",
                    ),
                    helper.make_node(
                        "GatedRMSNorm",
                        [
                            f"v_model.layers.{i}.linear_attn.Reshape_{base + 54}",
                            f"model.layers.{i}.linear_attn.norm.weight",
                            f"v_model.layers.{i}.linear_attn.Reshape_{base + 55}",
                        ],
                        [f"v_model.layers.{i}.linear_attn.norm.GatedRMSNorm_{base + 56}"],
                        name=f"model/layers.{i}/linear_attn/norm/GatedRMSNorm_node_{base + 56}",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        activation="sigmoid",
                    ),
                    helper.make_node(
                        "Shape",
                        [attn_pre],
                        [f"v_model.layers.{i}.linear_attn.Shape_{base + 57}"],
                        name=f"model/layers.{i}/linear_attn/Shape_node_{base + 57}",
                        end=2,
                        start=0,
                    ),
                    helper.make_node(
                        "Concat",
                        [
                            f"v_model.layers.{i}.linear_attn.Shape_{base + 57}",
                            "v_model.layers.0.linear_attn.CastLike_145",
                        ],
                        [f"v_model.layers.{i}.linear_attn.Concat_{base + 59}"],
                        name=f"model/layers.{i}/linear_attn/Concat_node_{base + 59}",
                        axis=0,
                    ),
                    helper.make_node(
                        "Reshape",
                        [
                            f"v_model.layers.{i}.linear_attn.norm.GatedRMSNorm_{base + 56}",
                            f"v_model.layers.{i}.linear_attn.Concat_{base + 59}",
                        ],
                        [f"v_model.layers.{i}.linear_attn.Reshape_{base + 60}"],
                        name=f"model/layers.{i}/linear_attn/Reshape_node_{base + 60}",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        [
                            f"v_model.layers.{i}.linear_attn.Reshape_{base + 60}",
                            f"model.layers.{i}.linear_attn.out_proj.weight_t.qweight",
                            f"model.layers.{i}.linear_attn.out_proj.weight_t.scales",
                        ],
                        [out_proj_out],
                        name=f"skinny_gemm/{sg_base + 1}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=6144,
                        N=2560,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                ]
            )
            mlp_off = 65
        else:
            iqk_out = f"v_model.layers.{i}.self_attn.indexer.index_qk_proj.MatMul_{base + 22}"
            v_out = f"v_model.layers.{i}.self_attn.v_proj.MatMul_{base + 76}"
            out_proj_out = f"v_model.layers.{i}.self_attn.o_proj.MatMul_{base + 100}"
            val_idx = 21 + dsa_idx
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [attn_pre, f"merged_proj.{mp_base}.weight_t.qweight", f"merged_proj.{mp_base}.weight_t.scales"],
                        [f"merged_proj.{mp_base}.out"],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=2560,
                        N=13952,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                    helper.make_node(
                        "Split",
                        [f"merged_proj.{mp_base}.out", "contrib_const_split_qsa"],
                        [iqk_out, f"merged_proj.{mp_base}.out.q_gate_12288", f"merged_proj.{mp_base}.out.k_512", v_out],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits/SplitQsa",
                        axis=-1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"merged_proj.{mp_base}.out.q_gate_12288", "contrib_const_q_4d"],
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/ReshapeQ4D",
                    ),
                    helper.make_node(
                        "Split",
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d", "const_split_256_256"],
                        [
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/q_only_4d",
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/gate_only_4d",
                        ],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/ReshapeQ4D/SplitQGate",
                        axis=-1,
                    ),
                    helper.make_node(
                        "RMSNormRotary",
                        [
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/q_only_4d",
                            f"v_model.layers.{i}.self_attn.q_norm.Add_{base + 47}.folded",
                            "position_ids",
                            "model.rotary_emb.cos_cache",
                            "model.rotary_emb.sin_cache",
                        ],
                        [f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 73}"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/RMSNormRotaryQ",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        flatten_batch=0,
                        input_start=0,
                        input_width=256,
                        rotary_embedding_dim=64,
                        use_fp32_arithmetic=1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"merged_proj.{mp_base}.out.k_512", "contrib_const_k_4d"],
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/k_4d"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/ReshapeK4D",
                    ),
                    helper.make_node(
                        "RMSNormRotary",
                        [
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/k_4d",
                            f"v_model.layers.{i}.self_attn.k_norm.Add_{base + 61}.folded",
                            "position_ids",
                            "model.rotary_emb.cos_cache",
                            "model.rotary_emb.sin_cache",
                        ],
                        [f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 74}"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/RMSNormRotaryK",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        flatten_batch=0,
                        input_start=0,
                        input_width=256,
                        rotary_embedding_dim=64,
                        use_fp32_arithmetic=1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [iqk_out, "capture.indexer_qk_shape"],
                        [f"capture.{i}.indexer_qk_flat"],
                        name=f"capture/{i}/Reshape_indexer_qk",
                    ),
                    helper.make_node(
                        "PackedSparseAttentionIndexer",
                        [
                            f"capture.{i}.indexer_qk_flat",
                            "",
                            f"v_model.layers.{i}.self_attn.indexer.q_layernorm.Add_{base + 25}.folded.bf16",
                            f"v_model.layers.{i}.self_attn.indexer.k_layernorm.Add_{base + 29}.folded.bf16",
                            "capture.indexer_cos_cache.folded",
                            "capture.indexer_sin_cache.folded",
                            "cumulative_sequence_lengths",
                            "committed_length",
                            "",
                            "",
                            "",
                            "",
                            f"past_key_values.{i}.index_key",
                            f"past_key_values.{i}.index_buffer",
                            "",
                            f"past_key_values.{i}.index_lengths",
                            "state_update_capture_count",
                        ],
                        [
                            f"v_model.layers.{i}.self_attn.indexer.SparseAttentionIndexer_{base + 33}_0",
                            f"capture.{i}.selected_counts",
                            f"present.{i}.index_key",
                            f"present.{i}.index_buffer",
                            "",
                            f"present.{i}.index_lengths",
                            f"state_update.{i}.indexer_value",
                            "",
                            f"validation.qsa.{i}",
                        ],
                        name=f"model/layers.{i}/self_attn/indexer/PackedQSAIndexer_node_{base + 33}",
                        domain="com.microsoft",
                        compress_ratio=4,
                        epsilon=1e-6,
                        policy_mode="qsa",
                        scale=0.0883883461356163,
                        state_capacity=eff_indexer_state_cap,
                        token_budget=2048,
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "DynamicSparseAttention",
                        [
                            f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 73}",
                            f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 74}",
                            v_out,
                            f"past_key_values.{i}.key",
                            f"past_key_values.{i}.value",
                            "",
                            "",
                            f"v_model.layers.{i}.self_attn.indexer.SparseAttentionIndexer_{base + 33}_0",
                            f"capture.{i}.selected_counts",
                            "capture.seqlens_k",
                            "contrib_const_dsa_total_seq",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "committed_length",
                            f"node_GuardedDSA_{dsa_idx}_k_scale",
                            f"node_GuardedDSA_{dsa_idx}_v_scale",
                        ],
                        [
                            f"v_model.layers.{i}.self_attn.DynamicSparseAttention_{base + 95}_0",
                            f"present.{i}.key",
                            f"present.{i}.value",
                            f"validation.{val_idx}.present.{i}.key",
                        ],
                        name=f"node_GuardedDSA_{dsa_idx}",
                        domain="com.microsoft",
                        kv_num_heads=2,
                        num_heads=24,
                        scale=0.0625,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/gate_only_4d", "const_attn_6144_shape"],
                        [f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_3d"],
                        name=f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/ReshapeGate3D",
                    ),
                    helper.make_node(
                        "Sigmoid",
                        [f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_3d"],
                        [f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_sig"],
                        name=f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/SigmoidGate",
                    ),
                    helper.make_node(
                        "Mul",
                        [
                            f"v_model.layers.{i}.self_attn.DynamicSparseAttention_{base + 95}_0",
                            f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_sig",
                        ],
                        [f"v_model.layers.{i}.self_attn.Mul_{base + 98}"],
                        name=f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/MulGate",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        [
                            f"v_model.layers.{i}.self_attn.Mul_{base + 98}",
                            f"model.layers.{i}.self_attn.o_proj.weight_t.qweight",
                            f"model.layers.{i}.self_attn.o_proj.weight_t.scales",
                        ],
                        [out_proj_out],
                        name=f"skinny_gemm/{sg_base + 1}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=6144,
                        N=2560,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                ]
            )
            dsa_idx += 1
            mlp_off = 103

        mb = base + mlp_off
        mlp_pre = f"v_model.layers.{i}.mlp_hyper_connection.HyperConnectionPreMix_{mb + 8}"
        mlp_mul = f"v_model.layers.{i}.mlp_hyper_connection.Mul_{mb + 15}"
        attn_post = f"v_model.layers.{i}.HyperConnectionPostMix_{mb - 2}"
        mlp_add = f"v_model.layers.{i}.mlp.Add_{mb + 41}"
        nodes.extend(
            [
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    [
                        cur_streams,
                        f"v_model.layers.{i}.mlp_hyper_connection.hc_norm.Add_{mb}.folded",
                        f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                        f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                        f"model.layers.{i}.mlp_hyper_connection.block_inject_weight.weight_t",
                        out_proj_out,
                        attn_mul,
                    ],
                    [mlp_pre, mlp_mul, attn_post],
                    name=f"model/layers.{i}/mlp_hyper_connection/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    epsilon=1e-6,
                    gate_div=4.0,
                    gate_mul=2.0,
                    num_branches=4,
                    reduction_scale=0.25,
                    silu_alpha=0.25,
                ),
                helper.make_node(
                    "RouterGateUpProj",
                    [mlp_pre, f"merged_proj.{mp_base + 1}.weight_t"],
                    [
                        f"skinny_gemm/{sg_base + 2}.0/SplitRouterSwishMul/router_3d",
                        f"skinny_gemm/{sg_base + 2}.0/swish_640",
                    ],
                    name=f"skinny_gemm/{sg_base + 2}.0/RouterGateUpProj",
                    domain="com.microsoft",
                    activation="swiglu",
                    activation_alpha=1.0,
                    activation_beta=0.0,
                    limit=0.0,
                    router_size=512,
                ),
                helper.make_node(
                    "Reshape",
                    [f"skinny_gemm/{sg_base + 2}.0/SplitRouterSwishMul/router_3d", "contrib_const_router_2d_shape"],
                    [f"v_model.layers.{i}.mlp.gate.MatMul_{mb + 17}"],
                    name=f"skinny_gemm/{sg_base + 2}.0/SplitRouterSwishMul/ReshapeRouter2D",
                ),
                helper.make_node(
                    "QMoE",
                    [
                        mlp_pre,
                        f"v_model.layers.{i}.mlp.gate.MatMul_{mb + 17}",
                        f"model.layers.{i}.mlp.fc1_experts_weights",
                        f"model.layers.{i}.mlp.fc1_scales",
                        "",
                        f"model.layers.{i}.mlp.fc2_experts_weights",
                        f"model.layers.{i}.mlp.fc2_scales",
                    ],
                    [f"v_model.layers.{i}.mlp.QMoE_{mb + 28}"],
                    name=f"model/layers.{i}/mlp/QMoE_node_{mb + 28}",
                    domain="com.microsoft",
                    activation_type="swiglu",
                    block_size=32,
                    expert_weight_bits=4,
                    k=10,
                    normalize_routing_weights=1,
                    swiglu_fusion=1,
                    weights_prepacked=1,
                ),
                helper.make_node(
                    "MatMul",
                    [
                        f"skinny_gemm/{sg_base + 2}.0/swish_640",
                        f"model.layers.{i}.mlp.shared_expert.down_proj.weight_t",
                    ],
                    [f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/shared_down_out"],
                    name=f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/MatMulBf16",
                ),
                helper.make_node(
                    "GatedAdd",
                    [
                        f"v_model.layers.{i}.mlp.QMoE_{mb + 28}",
                        f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/shared_down_out",
                        mlp_pre,
                        f"model.layers.{i}.mlp.shared_expert_gate.weight_t",
                    ],
                    [mlp_add],
                    name=f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/SharedGateMulAdd/ProjectedSigmoidGateMul/GatedAdd",
                    domain="com.microsoft",
                ),
            ]
        )

        if i == 0:
            pfx0 = "model/layers.0/HyperConnectionPostMix_node_194"
            nodes.extend(
                [
                    helper.make_node(
                        "Reshape",
                        [attn_post, "const_hc_4d_shape"],
                        [f"{pfx0}/streams_4d"],
                        name=f"{pfx0}/ReshapeStreams4D",
                    ),
                    helper.make_node(
                        "Unsqueeze", [mlp_mul, "const_axis_neg1"], [f"{pfx0}/gate_4d"], name=f"{pfx0}/UnsqueezeGate"
                    ),
                    helper.make_node(
                        "Unsqueeze", [mlp_add, "const_axis_neg2"], [f"{pfx0}/block_4d"], name=f"{pfx0}/UnsqueezeBlock"
                    ),
                    helper.make_node(
                        "Mul", [f"{pfx0}/gate_4d", f"{pfx0}/block_4d"], [f"{pfx0}/prod_4d"], name=f"{pfx0}/MulGate"
                    ),
                    helper.make_node(
                        "Add", [f"{pfx0}/streams_4d", f"{pfx0}/prod_4d"], [f"{pfx0}/add_4d"], name=f"{pfx0}/AddStreams"
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"{pfx0}/add_4d", "const_hc_3d_shape"],
                        ["v_model.layers.0.HyperConnectionPostMix_194"],
                        name=f"{pfx0}/ReshapeOut3D",
                    ),
                ]
            )
        prev_streams = attn_post
        prev_mlp_add = mlp_add
        prev_mlp_mul = mlp_mul
        prev_post_mix_id = mb + 42

    pfx_f = "model/hyper_connection_mixer/HcPostBlock/HyperConnectionPostMix"
    val_ins = [f"validation.{25 + k}.present.{3 + 4 * k}.key" for k in range(12)] + [
        f"validation.qsa.{3 + 4 * k}" for k in range(12)
    ]
    nodes.extend(
        [
            helper.make_node(
                "Reshape",
                [prev_streams, "const_hc_4d_shape"],
                [f"{pfx_f}/streams_4d"],
                name=f"{pfx_f}/ReshapeStreams4D",
            ),
            helper.make_node(
                "Unsqueeze", [prev_mlp_mul, "const_axis_neg1"], [f"{pfx_f}/gate_4d"], name=f"{pfx_f}/UnsqueezeGate"
            ),
            helper.make_node(
                "Unsqueeze", [prev_mlp_add, "const_axis_neg2"], [f"{pfx_f}/block_4d"], name=f"{pfx_f}/UnsqueezeBlock"
            ),
            helper.make_node(
                "Mul", [f"{pfx_f}/gate_4d", f"{pfx_f}/block_4d"], [f"{pfx_f}/prod_4d"], name=f"{pfx_f}/MulGate"
            ),
            helper.make_node(
                "Add", [f"{pfx_f}/streams_4d", f"{pfx_f}/prod_4d"], [f"{pfx_f}/add_4d"], name=f"{pfx_f}/AddStreams"
            ),
            helper.make_node(
                "Reshape", [f"{pfx_f}/add_4d", "const_hc_3d_shape"], ["hidden_states"], name=f"{pfx_f}/ReshapeOut3D"
            ),
            helper.make_node(
                "Slice",
                ["hidden_states", "dflash_slice_neg4", "const_1d_1", "const_1d_3"],
                ["hidden_states.last2"],
                name="model/hyper_connection_mixer/Slice_last2",
            ),
            helper.make_node(
                "HyperConnectionLowRankMix",
                [
                    "hidden_states.last2",
                    "v_model.hyper_connection_mixer.hc_norm.Add_5815.folded",
                    "model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                    "model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                    "model.layers.0.attn_hyper_connection.block_inject_weight.weight_t",
                ],
                [
                    "v_model.hyper_connection_mixer.HyperConnectionPreMix_5823",
                    "model/hyper_connection_mixer/HcPostBlock/BranchwiseRMSNorm.dummy_gate",
                ],
                name="model/hyper_connection_mixer/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                domain="com.microsoft",
                epsilon=1e-6,
                gate_div=4.0,
                gate_mul=2.0,
                num_branches=4,
                reduction_scale=0.25,
                silu_alpha=0.25,
            ),
            helper.make_node(
                "MatMulNBits",
                [
                    "v_model.hyper_connection_mixer.HyperConnectionPreMix_5823",
                    "lm_head.weight_t.qweight",
                    "lm_head.weight_t.scales",
                ],
                ["logits"],
                name="lm_head/MatMul_node_5826/SkinnyGemm/MatMulNBits",
                domain="com.microsoft",
                K=2560,
                N=248320,
                accuracy_level=0,
                bits=8,
                block_size=32,
                weight_prepacked=1,
            ),
            helper.make_node("Concat", val_ins, ["validation.aggregate"], name="validation/aggregate", axis=0),
            helper.make_node(
                "ArgMax", ["logits"], ["next_tokens"], name="argmax_rows", axis=-1, keepdims=0, select_last_index=0
            ),
            helper.make_node(
                "SpeculativePrefixCommit",
                [
                    "next_tokens",
                    "input_ids",
                    "validation.aggregate",
                    "fold.ctl",
                    "fold.ctx_commit",
                    "fold.ctx_a",
                    "fold.ptrs",
                ],
                ["fold.tail_done", "fold.ctl_out", "fold.ctx_commit_out"],
                name="fold/SpeculativePrefixCommit",
                domain="com.microsoft",
                head_rows=verify_rows,
            ),
        ]
    )
    for tap_lid in (5, 15, 25, 35, 45):
        post_t = next(
            n.output[2]
            for n in nodes
            if n.name
            == f"model/layers.{tap_lid + 1}/attn_hyper_connection/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix"
        )
        nodes.extend(
            [
                helper.make_node(
                    "Reshape",
                    [post_t, "dflash_aux_reshape_4d"],
                    [f"dflash_aux_layer_{tap_lid}_4d"],
                    name=f"dflash_aux/layer_{tap_lid}/Reshape",
                ),
                helper.make_node(
                    "ReduceMean",
                    [f"dflash_aux_layer_{tap_lid}_4d", "dflash_aux_reduce_axis"],
                    [f"dflash_aux_layer_{tap_lid}_mean"],
                    name=f"dflash_aux/layer_{tap_lid}/ReduceMean",
                    keepdims=0,
                ),
            ]
        )
    nodes.append(
        helper.make_node(
            "Concat",
            [f"dflash_aux_layer_{lid}_mean" for lid in (5, 15, 25, 35, 45)],
            ["aux_hidden_states"],
            name="dflash_aux/Concat",
            axis=-1,
        )
    )

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence_length"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence_length"]),
        helper.make_tensor_value_info("past_position_ids", TensorProto.INT64, ["batch", kv_capacity]),
    ]
    for i in range(48):
        if i % 4 != 3:
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{i}.conv_state", TensorProto.BFLOAT16, ["batch", 10240, 4]
                )
            )
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{i}.recurrent_state", TensorProto.FLOAT, ["batch", 48, 128, 128]
                )
            )
            if i == 1:
                inputs.extend(
                    [
                        helper.make_tensor_value_info(
                            "ple_embeddings.1", TensorProto.BFLOAT16, ["batch", "sequence_length", 2560]
                        ),
                        helper.make_tensor_value_info(
                            "past_key_values.1.ple_conv_state", TensorProto.BFLOAT16, ["batch", 10240, 9]
                        ),
                        helper.make_tensor_value_info("past_key_values.1.ple_context", TensorProto.INT64, ["batch", 2]),
                    ]
                )
        else:
            inputs.extend(
                [
                    helper.make_tensor_value_info(
                        f"past_key_values.{i}.key", TensorProto.INT8, ["batch", 2, kv_capacity, 256]
                    ),
                    helper.make_tensor_value_info(
                        f"past_key_values.{i}.value", TensorProto.INT8, ["batch", 2, kv_capacity, 256]
                    ),
                    helper.make_tensor_value_info(
                        f"past_key_values.{i}.index_key", TensorProto.FLOAT, ["batch", state_capacity, 128]
                    ),
                ]
            )
    inputs.extend(
        [
            helper.make_tensor_value_info("committed_length", TensorProto.INT32, ["batch"]),
            helper.make_tensor_value_info("sequence_length_minus_one", TensorProto.INT32, ["seq_minus_one_len"]),
            helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_plus_one"]),
        ]
    )
    for i in range(3, 48, 4):
        inputs.append(
            helper.make_tensor_value_info(f"past_key_values.{i}.index_buffer", TensorProto.BFLOAT16, ["batch", 7, 128])
        )
        inputs.append(
            helper.make_tensor_value_info(f"past_key_values.{i}.index_lengths", TensorProto.INT32, ["batch", 2])
        )
    inputs.extend(
        [
            helper.make_tensor_value_info("fold.ctl", TensorProto.INT64, ["fold_ctl_len"]),
            helper.make_tensor_value_info("fold.ctx_commit", TensorProto.INT64, ["fold_ctx_commit_len"]),
            helper.make_tensor_value_info("fold.ctx_a", TensorProto.INT64, ["fold_ctx_a_len"]),
            helper.make_tensor_value_info("fold.ptrs", TensorProto.INT64, ["fold_ptrs_len"]),
            helper.make_tensor_value_info(
                "inputs_embeds", TensorProto.BFLOAT16, ["batch_size", "sequence_length", 2560]
            ),
            helper.make_tensor_value_info("state_update_capture_count", TensorProto.INT32, ["batch"]),
        ]
    )

    logits_vi = onnx.ValueInfoProto()
    logits_vi.name = "logits"
    logits_vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
    outputs = [
        logits_vi,
        helper.make_tensor_value_info("present_position_ids", TensorProto.INT64, ["batch", kv_capacity]),
        helper.make_tensor_value_info("hidden_states", TensorProto.BFLOAT16, ["batch", "sequence_length", 10240]),
    ]
    for i in range(48):
        if i % 4 != 3:
            outputs.append(
                helper.make_tensor_value_info(f"present.{i}.conv_state", TensorProto.BFLOAT16, ["batch", 10240, 4])
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"present.{i}.recurrent_state", TensorProto.FLOAT, ["batch", 48, 128, 128]
                )
            )
            if i == 1:
                outputs.append(
                    helper.make_tensor_value_info("present.1.ple_conv_state", TensorProto.BFLOAT16, ["batch", 10240, 9])
                )
                outputs.append(
                    helper.make_tensor_value_info("present.1.ple_context", TensorProto.INT64, ["batch", "_d12"])
                )
        else:
            outputs.extend(
                [
                    helper.make_tensor_value_info(f"present.{i}.key", TensorProto.INT8, ["batch", 2, kv_capacity, 256]),
                    helper.make_tensor_value_info(
                        f"present.{i}.value", TensorProto.INT8, ["batch", 2, kv_capacity, 256]
                    ),
                    helper.make_tensor_value_info(
                        f"present.{i}.index_key", TensorProto.FLOAT, ["batch", state_capacity, 128]
                    ),
                ]
            )
    outputs.append(helper.make_tensor_value_info("validation.aggregate", TensorProto.INT32, ["validation_count"]))
    for i in range(3, 48, 4):
        outputs.append(
            helper.make_tensor_value_info(f"present.{i}.index_buffer", TensorProto.BFLOAT16, ["batch", 7, 128])
        )
        outputs.append(helper.make_tensor_value_info(f"present.{i}.index_lengths", TensorProto.INT32, ["batch", 2]))
    outputs.extend(
        [
            helper.make_tensor_value_info("next_tokens", TensorProto.INT64, ["batch", "sequence_length"]),
            helper.make_tensor_value_info("fold.tail_done", TensorProto.INT32, ["batch"]),
            helper.make_tensor_value_info("fold.ctl_out", TensorProto.INT64, ["fold_ctl_len"]),
            helper.make_tensor_value_info("fold.ctx_commit_out", TensorProto.INT64, ["fold_ctx_commit_len"]),
        ]
    )
    for i in range(48):
        if i == 1:
            outputs.append(
                helper.make_tensor_value_info(
                    "state_update.1.ple_conv_value", TensorProto.BFLOAT16, ["batch", state_update_capacity, 10240]
                )
            )
        if i % 4 != 3:
            outputs.append(
                helper.make_tensor_value_info(
                    f"state_update.{i}.conv_value", TensorProto.BFLOAT16, ["batch", state_update_capacity, 10240]
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"state_update.{i}.recurrent_capsule", TensorProto.FLOAT, ["batch", state_update_capacity * 8240]
                )
            )
        else:
            outputs.append(
                helper.make_tensor_value_info(
                    f"state_update.{i}.indexer_value", TensorProto.BFLOAT16, ["batch", state_update_capacity, 128]
                )
            )
    outputs.append(
        helper.make_tensor_value_info("aux_hidden_states", TensorProto.BFLOAT16, ["batch", "sequence_length", 12800])
    )

    inits: list[TensorProto] = [
        numpy_helper.from_array(np.array([-2], dtype=np.int64), name="const_1d_0"),
        numpy_helper.from_array(np.array([9223372036854775807], dtype=np.int64), name="const_1d_1"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="const_1d_3"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="const_1d_11"),
        numpy_helper.from_array(np.array(1e-6, dtype=np.float32), name="const_1e-06_f32"),
        numpy_helper.from_array(np.array([-1, 128], dtype=np.int64), name="const_1d_17"),
        numpy_helper.from_array(np.array([0, 0, 4, 2560], dtype=np.int64), name="const_1d_19"),
        numpy_helper.from_array(np.array(50.59644256269407, dtype=np.float32), name="const_50.59644256269407_f32"),
        numpy_helper.from_array(np.array([0, 0, -1], dtype=np.int64), name="const_1d_21"),
        numpy_helper.from_array(np.array([6144], dtype=np.int64), name="v_model.layers.0.linear_attn.CastLike_145"),
        numpy_helper.from_array(np.array([-1, 640], dtype=np.int64), name="capture.indexer_qk_shape"),
        numpy_helper.from_array(np.array([10240, 2560], dtype=np.int64), name="skinny_gemm/4.0/split_lengths"),
        numpy_helper.from_array(np.array([0, 0, 24, 512], dtype=np.int64), name="contrib_const_q_4d"),
        numpy_helper.from_array(np.array([0, 0, 2, 256], dtype=np.int64), name="contrib_const_k_4d"),
        numpy_helper.from_array(np.array([eff_dsa_total_seq], dtype=np.int32), name="contrib_const_dsa_total_seq"),
        numpy_helper.from_array(np.array([10240, 6144, 48, 48], dtype=np.int64), name="contrib_const_split_gdn"),
        numpy_helper.from_array(np.array([640, 12288, 512, 512], dtype=np.int64), name="contrib_const_split_qsa"),
        numpy_helper.from_array(np.array([-1, 512], dtype=np.int64), name="contrib_const_router_2d_shape"),
        numpy_helper.from_array(np.array([-verify_rows], dtype=np.int64), name="dflash_slice_neg4"),
        numpy_helper.from_array(np.array([0, 0, 4, 2560], dtype=np.int64), name="dflash_aux_reshape_4d"),
        numpy_helper.from_array(np.array([-2], dtype=np.int64), name="dflash_aux_reduce_axis"),
        numpy_helper.from_array(np.array([0, 0, 4, 2560], dtype=np.int64), name="const_hc_4d_shape"),
        numpy_helper.from_array(np.array([0, 0, 10240], dtype=np.int64), name="const_hc_3d_shape"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="const_axis_neg1"),
        numpy_helper.from_array(np.array([-2], dtype=np.int64), name="const_axis_neg2"),
        numpy_helper.from_array(np.array([256, 256], dtype=np.int64), name="const_split_256_256"),
        numpy_helper.from_array(np.array([0, 0, 6144], dtype=np.int64), name="const_attn_6144_shape"),
        numpy_helper.from_array(np.array([512, 1280], dtype=np.int64), name="const_split_512_1280"),
    ]

    for k_name, arr in sorted(kv_scales.items()):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=k_name))
    for _k_name, t_proto in sorted(embedded_weights.items()):
        t_copy = TensorProto()
        t_copy.CopyFrom(t_proto)
        inits.append(t_copy)

    used_inputs = {inp for n in nodes for inp in n.input if inp}
    existing_init_names = {t.name for t in inits}
    for name in sorted(used_inputs):
        if name in existing_init_names:
            continue
        if name in tensor_locations:
            meta = tensor_locations[name]
            inits.append(
                make_ext_init(name, meta["location"], meta["offset"], meta["length"], meta["dtype"], meta["dims"])
            )

    graph = helper.make_graph(nodes, "qwen4_exp", inputs, outputs, initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="mobius",
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11
    fixed_kv_map = {
        **{
            f"past_key_values.{lid}.{sfx}": f"present.{lid}.{sfx}"
            for lid in sorted((3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47), key=str)
            for sfx in ("key", "value")
        },
        "past_position_ids": "present_position_ids",
    }
    meta_props = (
        ("experiment.cuda_graph_device_lengths", "committed_length + sequence_length_minus_one"),
        ("experiment.cuda_graph_index_rotary", "static-expanded-initializer"),
        ("experiment.cuda_graph_index_state_type", "float32"),
        ("experiment.cuda_graph_indexer", "qwen.capture::PackedQSAIndexer;device lengths;compressed state"),
        ("experiment.cuda_graph_ple_empty_branch", "pruned"),
        ("experiment.cuda_graph_sequence_length", "per fixed binding"),
        ("experiment.fixed_kv_capacity", str(eff_dsa_total_seq)),
        ("experiment.fixed_kv_logical_length", "attention_mask.shape[1]"),
        ("experiment.fixed_kv_mapping", json.dumps(fixed_kv_map)),
        ("experiment.host_positions", "1"),
        (
            "experiment.static_execution",
            "fixedKV+index+positions;immutable-recurrent;deferred-status;no-fullcapture",
        ),
        ("experiment.validation_status_names", json.dumps(val_ins)),
        (
            "mobius.cache_abi",
            "qwen4-exp:position_ids;linear=conv_state,recurrent_state[,ple_conv_state,ple_context];qsa=key,value,index_key",
        ),
        (
            "mobius.hidden_states_abi",
            "pre-final-hyper-mixer;unpruned-sequence;row=tokens-consumed;mtp-input=(hidden[t],token[t+1])",
        ),
        ("mobius.logits_abi", "sequence-row-gather:input=logits_indices;axis=1;full-vocabulary"),
        ("mobius.ple_abi", "external-cpu-fp8:scaled-bfloat16;input=ple_embeddings.{zero_based_layer}"),
    )
    for k_prop, v_prop in meta_props:
        prop = model.metadata_props.add()
        prop.key = k_prop
        prop.value = v_prop
    return model


def _load_safetensors_tensor(shard_path: Path, key: str, device: str = "cpu") -> torch.Tensor:
    with open(shard_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    info = hdr[key]
    s, e = info["data_offsets"]
    shape = tuple(info["shape"])
    dtype_str = info["dtype"]
    base_off = 8 + hlen + s
    if dtype_str == "BF16":
        mm = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=base_off, shape=shape)
        t = torch.from_numpy(mm.copy()).view(torch.bfloat16)
    elif dtype_str == "F32":
        mm = np.memmap(shard_path, dtype=np.float32, mode="r", offset=base_off, shape=shape)
        t = torch.from_numpy(mm.copy())
    elif dtype_str == "I64":
        mm = np.memmap(shard_path, dtype=np.int64, mode="r", offset=base_off, shape=shape)
        t = torch.from_numpy(mm.copy())
    else:
        raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {key}")
    return t.to(device=device) if device != "cpu" else t


class CheckpointTensorLoader:
    """Memory-mapped reader for single-file (`model.safetensors`) or sharded (`model.safetensors.index.json`) checkpoints."""

    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir.resolve()
        idx_path = self.ckpt_dir / "model.safetensors.index.json"
        single_path = self.ckpt_dir / "model.safetensors"
        if idx_path.is_file():
            self.weight_map: dict[str, str] = json.loads(idx_path.read_text())["weight_map"]
        elif single_path.is_file():
            with open(single_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            self.weight_map = {k: "model.safetensors" for k in hdr if k != "__metadata__"}
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors.index.json nor model.safetensors found in {self.ckpt_dir}"
            )
        self._headers: dict[str, tuple[int, dict]] = {}

    def _resolve_key(self, name: str) -> str:
        if name in self.weight_map:
            return name
        if name.startswith("model.layers."):
            alt = "model.language_model." + name[len("model.") :]
            if alt in self.weight_map:
                return alt
        if name.startswith("model.embed_tokens."):
            alt = "model.language_model." + name[len("model.") :]
            if alt in self.weight_map:
                return alt
        if name.startswith("model.hyper_connection_mixer."):
            alt = "model.language_model." + name[len("model.") :]
            if alt in self.weight_map:
                return alt
        raise KeyError(f"Weight {name!r} not found in {self.ckpt_dir}")

    def read_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        key = self._resolve_key(name)
        shard_file = self.weight_map[key]
        if shard_file not in self._headers:
            p = self.ckpt_dir / shard_file
            with open(p, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            self._headers[shard_file] = (8 + hlen, hdr)
        base_off, hdr = self._headers[shard_file]
        info = hdr[key]
        s, _e = info["data_offsets"]
        shape = tuple(info["shape"])
        dtype_str = info["dtype"]
        p = self.ckpt_dir / shard_file
        if dtype_str == "BF16":
            mm = np.memmap(p, dtype=np.uint16, mode="r", offset=base_off + s, shape=shape)
            t = torch.from_numpy(mm).view(torch.bfloat16)
        elif dtype_str == "F32":
            mm = np.memmap(p, dtype=np.float32, mode="r", offset=base_off + s, shape=shape)
            t = torch.from_numpy(mm)
        elif dtype_str == "I64":
            mm = np.memmap(p, dtype=np.int64, mode="r", offset=base_off + s, shape=shape)
            t = torch.from_numpy(mm)
        else:
            raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {key}")
        return t.to(device=device, non_blocking=False) if device != "cpu" else t.clone()


@torch.no_grad()
def build_embedding_from_hf(
    hf_loader: CheckpointTensorLoader, out_dir: Path, vocab_size: int = 248320, hidden_size: int = 2560
) -> dict[str, dict]:
    """Quantize `model.language_model.embed_tokens.weight` to per-row symmetric INT8 in `embedding.onnx.data` and save `embedding.onnx`."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb_f32 = hf_loader.read_tensor("model.language_model.embed_tokens.weight", device=dev).float()
    sc = (emb_f32.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
    q_i8 = torch.round(emb_f32 / sc).clamp(-127.0, 127.0).to(torch.int8).cpu().numpy()
    sc_np = sc.to(torch.float32).cpu().numpy()

    i8_bytes = vocab_size * hidden_size
    s_bytes = vocab_size * 4
    data_path = out_dir / "embedding.onnx.data"
    with open(data_path, "wb") as f:
        f.write(q_i8.tobytes())
        f.write(sc_np.tobytes())

    emb_proto = build_embedding_model_proto(vocab_size=vocab_size, hidden_size=hidden_size)
    onnx.save_model(emb_proto, str(out_dir / "embedding.onnx"), save_as_external_data=False)
    return {
        "model.embed_tokens.weight": {
            "location": "embedding.onnx.data",
            "offset": 0,
            "length": i8_bytes,
            "dtype": TensorProto.INT8,
            "dims": [vocab_size, hidden_size],
        },
        "model.embed_tokens.weight.row_scales": {
            "location": "embedding.onnx.data",
            "offset": i8_bytes,
            "length": s_bytes,
            "dtype": TensorProto.FLOAT,
            "dims": [vocab_size, 1],
        },
    }


@torch.no_grad()
def build_ple_int8_from_hf(hf_loader: CheckpointTensorLoader, out_dir: Path, eos_token_id: int = 248044) -> None:
    """Quantize all 128 PLE n-gram shards from `hf_loader` to `ple/ple_int8_00..09.data` and write `ple_table.txt`."""
    ple_dir = out_dir / "ple"
    ple_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mults = [
        int(x)
        for x in hf_loader.read_tensor("model.language_model.layers.1.ple.ple_embedding.layer_multipliers").tolist()
    ]
    if len(mults) == 3:
        mults.append(0)
    vocabs = [
        int(x)
        for x in hf_loader.read_tensor(
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"
        ).tolist()
    ]
    offsets = [
        int(x)
        for x in hf_loader.read_tensor("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets").tolist()
    ]

    lines = [
        "qwen38_ple_table 1",
        f"n_heads {len(vocabs)}",
        "head_dim 160",
        "heads_per_ngram 8",
        f"eos {eos_token_id}",
        f"scale -{PLE_INT8_SCALE:.12f}",
        "mult " + " ".join(str(x) for x in mults),
        "vocab " + " ".join(str(x) for x in vocabs),
        "offset " + " ".join(str(x) for x in offsets),
    ]

    shards_per_file = [13] * 8 + [12] * 2
    file_idx = 0
    in_file_count = 0
    cur_f = None
    cur_rel = ""
    cur_off = 0
    inv_scale = 1.0 / PLE_INT8_SCALE

    try:
        for s in range(128):
            if cur_f is None:
                cur_rel = f"ple/ple_int8_{file_idx:02d}.data"
                cur_f = open(out_dir / cur_rel, "wb")  # noqa: SIM115
                cur_off = 0
                in_file_count = 0
            t = hf_loader.read_tensor(
                f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight", device=dev
            )
            rows, _width = int(t.shape[0]), int(t.shape[1])
            q_i8 = torch.round(t.float() * inv_scale).clamp(-127.0, 127.0).to(torch.int8).cpu().numpy()
            raw_bytes = q_i8.tobytes()
            cur_f.write(raw_bytes)
            lines.append(f"shard {cur_rel} {cur_off} {rows}")
            cur_off += len(raw_bytes)
            in_file_count += 1
            if in_file_count == shards_per_file[file_idx]:
                cur_f.close()
                cur_f = None
                file_idx += 1
    finally:
        if cur_f is not None:
            cur_f.close()

    (out_dir / "ple_table.txt").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def build_target_non_qmoe_from_hf(
    hf_loader: CheckpointTensorLoader,
    out_dir: Path,
    fast_quant_prepack_fn: Callable,
    max_rotary_rows: int = 262144,
    rope_config: RopeConfig = DEFAULT_ROPE,
) -> tuple[dict[str, dict], dict[str, TensorProto]]:
    """Build `norm_and_rope.data`, `attn_consts.data`, `contrib_weights.data`, `matmul_int8.data`, `matmul_scales.data`
    and embedded layer norms directly from `hf_loader`.
    """
    tensor_locations: dict[str, dict] = {}
    embedded_weights: dict[str, TensorProto] = {}

    norm_writer = BinaryBlobWriter(out_dir / "norm_and_rope.data")
    attn_writer = BinaryBlobWriter(out_dir / "attn_consts.data")
    contrib_writer = BinaryBlobWriter(out_dir / "contrib_weights.data")
    mm_w_writer = BinaryBlobWriter(out_dir / "matmul_int8.data")
    mm_s_writer = BinaryBlobWriter(out_dir / "matmul_scales.data")

    try:
        cos, sin = target_rope(torch.arange(max_rotary_rows, dtype=torch.float32), rope_config)
        cos32, sin32 = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        cos64 = torch.cat([cos32, cos32], dim=-1).contiguous()
        sin64 = torch.cat([sin32, sin32], dim=-1).contiguous()

        tensor_locations["model.rotary_emb.cos_cache"] = norm_writer.write_aligned(
            cos32.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 32]
        )
        tensor_locations["model.rotary_emb.sin_cache"] = norm_writer.write_aligned(
            sin32.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 32]
        )
        tensor_locations["capture.indexer_cos_cache.folded"] = attn_writer.write_aligned(
            cos64.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 64]
        )
        tensor_locations["capture.indexer_sin_cache.folded"] = attn_writer.write_aligned(
            sin64.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 64]
        )

        def write_bf16(writer: BinaryBlobWriter, t: torch.Tensor) -> dict:
            t_c = t.to(torch.bfloat16).contiguous().cpu()
            return writer.write_aligned(t_c.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(t_c.shape))

        def write_int8_matmul(name_base: str, w_nk: torch.Tensor) -> None:
            qb, qs = fast_quant_prepack_fn(w_nk, bits=8, block_size=32, is_qmoe=False)
            qw_name = f"{name_base}.qweight"
            qs_name = f"{name_base}.scales"
            tensor_locations[qw_name] = mm_w_writer.write_aligned(
                qb.view(torch.uint8).numpy().tobytes(), TensorProto.UINT8, list(qb.shape)
            )
            tensor_locations[qs_name] = mm_s_writer.write_aligned(
                qs.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(qs.shape)
            )

        for i in range(48):
            base = layer_base_node_id(i)
            mp_base = 0 if i == 0 else (3 + 2 * (i - 1))
            lpfx = f"model.language_model.layers.{i}"

            if i == 1:
                # PLE norms, conv1d, and merged key_proj + value_proj
                for p_key, oname in (
                    (f"{lpfx}.ple.norm_key.weight", "v_model.layers.1.ple.norm_key.Add_200.folded"),
                    (f"{lpfx}.ple.norm_query.weight", "v_model.layers.1.ple.norm_query.Add_204.folded"),
                    (f"{lpfx}.ple.norm_conv.weight", "v_model.layers.1.ple.norm_conv.Add_224.folded"),
                ):
                    w_n = (1.0 + hf_loader.read_tensor(p_key).float()).to(torch.bfloat16)
                    embedded_weights[oname] = make_bf16_tensor(oname, w_n.view(torch.uint16).numpy(), list(w_n.shape))
                ple_conv = hf_loader.read_tensor(f"{lpfx}.ple.conv1d.weight")
                tensor_locations["model.layers.1.ple.conv1d.weight"] = write_bf16(norm_writer, ple_conv)
                ple_kv = torch.cat(
                    [
                        hf_loader.read_tensor(f"{lpfx}.ple.key_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.ple.value_proj.weight"),
                    ],
                    dim=0,
                )
                write_int8_matmul("merged_proj.2.weight_t", ple_kv)

            # Attn hyper-connection
            w_hc_a = (1.0 + hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.hc_norm.weight").float()).to(
                torch.bfloat16
            )
            hc_a_name = f"v_model.layers.{i}.attn_hyper_connection.hc_norm.Add_{base}.folded"
            embedded_weights[hc_a_name] = make_bf16_tensor(
                hc_a_name, w_hc_a.view(torch.uint16).numpy(), list(w_hc_a.shape)
            )

            tensor_locations[
                f"model.layers.{i}.attn_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.input_mix_weight_down.weight").T
            )
            tensor_locations[
                f"model.layers.{i}.attn_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.input_mix_weight_up.weight").T
            )
            tensor_locations[f"model.layers.{i}.attn_hyper_connection.block_inject_weight.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.block_inject_weight.weight").T
            )

            if i % 4 != 3:
                w_in = torch.cat(
                    [
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_qkv.weight"),
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_z.weight"),
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_b.weight"),
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_a.weight"),
                    ],
                    dim=0,
                )
                write_int8_matmul(f"merged_proj.{mp_base}.weight_t", w_in)
                write_int8_matmul(
                    f"model.layers.{i}.linear_attn.out_proj.weight_t",
                    hf_loader.read_tensor(f"{lpfx}.linear_attn.out_proj.weight"),
                )
                tensor_locations[f"model.layers.{i}.linear_attn.conv1d.weight"] = write_bf16(
                    norm_writer, hf_loader.read_tensor(f"{lpfx}.linear_attn.conv1d.weight")
                )
                tensor_locations[f"model.layers.{i}.linear_attn.norm.weight"] = write_bf16(
                    norm_writer, hf_loader.read_tensor(f"{lpfx}.linear_attn.norm.weight")
                )
                a_log = hf_loader.read_tensor(f"{lpfx}.linear_attn.A_log").float().numpy()
                dt_bias = hf_loader.read_tensor(f"{lpfx}.linear_attn.dt_bias").float().numpy()
                embedded_weights[f"model.layers.{i}.linear_attn.A_log.f32"] = numpy_helper.from_array(
                    a_log, name=f"model.layers.{i}.linear_attn.A_log.f32"
                )
                embedded_weights[f"model.layers.{i}.linear_attn.dt_bias.f32"] = numpy_helper.from_array(
                    dt_bias, name=f"model.layers.{i}.linear_attn.dt_bias.f32"
                )
                mlp_off = 65
            else:
                w_qsa = torch.cat(
                    [
                        hf_loader.read_tensor(f"{lpfx}.self_attn.indexer.index_qk_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.self_attn.q_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.self_attn.k_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.self_attn.v_proj.weight"),
                    ],
                    dim=0,
                )
                write_int8_matmul(f"merged_proj.{mp_base}.weight_t", w_qsa)
                write_int8_matmul(
                    f"model.layers.{i}.self_attn.o_proj.weight_t",
                    hf_loader.read_tensor(f"{lpfx}.self_attn.o_proj.weight"),
                )
                iq_n = (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.indexer.q_layernorm.weight").float()).to(
                    torch.bfloat16
                )
                ik_n = (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.indexer.k_layernorm.weight").float()).to(
                    torch.bfloat16
                )
                iq_name = f"v_model.layers.{i}.self_attn.indexer.q_layernorm.Add_{base + 25}.folded.bf16"
                ik_name = f"v_model.layers.{i}.self_attn.indexer.k_layernorm.Add_{base + 29}.folded.bf16"
                embedded_weights[iq_name] = make_bf16_tensor(iq_name, iq_n.view(torch.uint16).numpy(), list(iq_n.shape))
                embedded_weights[ik_name] = make_bf16_tensor(ik_name, ik_n.view(torch.uint16).numpy(), list(ik_n.shape))

                qn_f32 = (
                    (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.q_norm.weight").float()).numpy().astype(np.float32)
                )
                kn_f32 = (
                    (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.k_norm.weight").float()).numpy().astype(np.float32)
                )
                qn_name = f"v_model.layers.{i}.self_attn.q_norm.Add_{base + 47}.folded"
                kn_name = f"v_model.layers.{i}.self_attn.k_norm.Add_{base + 61}.folded"
                embedded_weights[qn_name] = numpy_helper.from_array(qn_f32, name=qn_name)
                embedded_weights[kn_name] = numpy_helper.from_array(kn_f32, name=kn_name)
                mlp_off = 103

            mb = base + mlp_off
            w_hc_m = (1.0 + hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.hc_norm.weight").float()).to(
                torch.bfloat16
            )
            hc_m_name = f"v_model.layers.{i}.mlp_hyper_connection.hc_norm.Add_{mb}.folded"
            embedded_weights[hc_m_name] = make_bf16_tensor(
                hc_m_name, w_hc_m.view(torch.uint16).numpy(), list(w_hc_m.shape)
            )

            tensor_locations[
                f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.input_mix_weight_down.weight").T
            )
            tensor_locations[
                f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.input_mix_weight_up.weight").T
            )
            tensor_locations[f"model.layers.{i}.mlp_hyper_connection.block_inject_weight.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.block_inject_weight.weight").T
            )

            w_rgu = torch.cat(
                [
                    hf_loader.read_tensor(f"{lpfx}.mlp.gate.weight"),
                    hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert.gate_proj.weight"),
                    hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert.up_proj.weight"),
                ],
                dim=0,
            ).T
            tensor_locations[f"merged_proj.{mp_base + 1}.weight_t"] = write_bf16(norm_writer, w_rgu)
            tensor_locations[f"model.layers.{i}.mlp.shared_expert.down_proj.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert.down_proj.weight").T
            )
            tensor_locations[f"model.layers.{i}.mlp.shared_expert_gate.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert_gate.weight").T
            )

        # Final hyper-connection mixer & LM head
        w_hc_f = (1.0 + hf_loader.read_tensor("model.language_model.hyper_connection_mixer.hc_norm.weight").float()).to(
            torch.bfloat16
        )
        hc_f_name = "v_model.hyper_connection_mixer.hc_norm.Add_5815.folded"
        embedded_weights[hc_f_name] = make_bf16_tensor(hc_f_name, w_hc_f.view(torch.uint16).numpy(), list(w_hc_f.shape))
        tensor_locations["model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16"] = (
            write_bf16(
                contrib_writer,
                hf_loader.read_tensor("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight").T,
            )
        )
        tensor_locations["model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16"] = (
            write_bf16(
                contrib_writer,
                hf_loader.read_tensor("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight").T,
            )
        )
        write_int8_matmul("lm_head.weight_t", hf_loader.read_tensor("lm_head.weight"))
    finally:
        norm_writer.close()
        attn_writer.close()
        contrib_writer.close()
        mm_w_writer.close()
        mm_s_writer.close()

    return tensor_locations, embedded_weights


@torch.no_grad()
def build_dflash2_from_hf(
    dflash_dir: Path,
    out_dir: Path,
    shared_locations: dict[str, dict],
    fast_quant_prepack_fn: Callable,
    max_position: int = 262148,
) -> dict[str, dict]:
    """Quantize and pack `dflash2.onnx.data` and save `dflash2.onnx` from `dflash_dir`."""
    dflash_loader = CheckpointTensorLoader(dflash_dir)
    cfg = json.loads((dflash_dir / "config.json").read_text())
    dfl_cfg = cfg.get("dflash_config", {})
    block_size = int(dfl_cfg.get("block_size", 8))
    num_draft_tokens = block_size - 1
    selector_top_k = int(dfl_cfg.get("selector_top_k", 16))
    sliding_window = int(cfg.get("sliding_window", 2048))
    rope_theta = float(cfg.get("rope_parameters", {}).get("rope_theta", 10000.0))
    head_dim = int(cfg.get("head_dim", 128))
    if (
        head_dim != 128
        or rope_theta != 10000.0
        or cfg.get("rope_parameters", {}).get("rope_type", "default") != "default"
    ):
        raise ValueError("RC5 DFlash2 requires independent default RoPE with head_dim=128 and theta=10000")

    writer = BinaryBlobWriter(out_dir / "dflash2.onnx.data")
    dflash_locations: dict[str, dict] = {
        "model.embed_tokens.weight": shared_locations["model.embed_tokens.weight"],
        "model.embed_tokens.weight.row_scales": shared_locations["model.embed_tokens.weight.row_scales"],
        "lm_head.weight_t.qweight": shared_locations["lm_head.weight_t.qweight"],
        "lm_head.weight_t.scales": shared_locations["lm_head.weight_t.scales"],
    }
    base_kernels_u16: dict[str, np.ndarray] = {}

    try:

        def write_int4(name_base: str, w_nk: torch.Tensor) -> None:
            qb, qs = fast_quant_prepack_fn(w_nk.contiguous(), bits=4, block_size=32, is_qmoe=False)
            dflash_locations[f"{name_base}.qweight"] = writer.write_aligned(
                qb.view(torch.uint8).numpy().tobytes(), TensorProto.UINT8, list(qb.shape)
            )
            dflash_locations[f"{name_base}.scales"] = writer.write_aligned(
                qs.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(qs.shape)
            )

        def write_bf16(name: str, t: torch.Tensor, dims: list[int] | None = None) -> None:
            t_c = t.to(torch.bfloat16).contiguous().cpu()
            dflash_locations[name] = writer.write_aligned(
                t_c.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, dims or list(t_c.shape)
            )

        write_int4("dflash2.fc.MatMul.weight", dflash_loader.read_tensor("fc.weight"))
        for lid in range(5):
            w_qkv = torch.cat(
                [
                    dflash_loader.read_tensor(f"layers.{lid}.self_attn.q_proj.weight"),
                    dflash_loader.read_tensor(f"layers.{lid}.self_attn.k_proj.weight"),
                    dflash_loader.read_tensor(f"layers.{lid}.self_attn.v_proj.weight"),
                ],
                dim=0,
            )
            write_int4(f"dflash2.layers.{lid}.attn.qkv_proj.MatMul.weight", w_qkv)
            write_int4(
                f"dflash2.layers.{lid}.attn.o_proj.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.self_attn.o_proj.weight"),
            )
            w_gu = torch.cat(
                [
                    dflash_loader.read_tensor(f"layers.{lid}.mlp.gate_proj.weight"),
                    dflash_loader.read_tensor(f"layers.{lid}.mlp.up_proj.weight"),
                ],
                dim=0,
            )
            write_int4(f"dflash2.layers.{lid}.mlp.gate_up_proj.MatMul.weight", w_gu)
            write_int4(
                f"dflash2.layers.{lid}.mlp.down_proj.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.mlp.down_proj.weight"),
            )

        for lid in range(5):
            for kind in ("attention_conv", "mlp_conv"):
                bk = dflash_loader.read_tensor(f"layers.{lid}.{kind}.base_kernel").to(torch.bfloat16).contiguous()
                pfx = f"/dflash2/layers.{lid}/{kind}"
                write_bf16(f"{pfx}/finish.base_kernel.0", bk[1, 0].unsqueeze(0), [1, 160, 16])
                write_bf16(f"{pfx}/finish.base_kernel.1", bk[1, 1].unsqueeze(0), [1, 160, 16])
                write_bf16(f"{pfx}/prepare.base_kernel.0", bk[0, 0].unsqueeze(0), [1, 160, 16])
                write_bf16(f"{pfx}/prepare.base_kernel.1", bk[0, 1].unsqueeze(0), [1, 160, 16])
                base_kernels_u16[f"{pfx}/base_kernel_all_1x4x160x16"] = (
                    bk.reshape(1, 1, 4, 160, 16).view(torch.uint16).numpy().copy()
                )

        write_bf16(
            "dflash2.candidate_selector.predecessor_codebook",
            dflash_loader.read_tensor("candidate_selector.predecessor_codebook"),
        )
        write_bf16(
            "dflash2.candidate_selector.successor_codebook",
            dflash_loader.read_tensor("candidate_selector.successor_codebook"),
        )

        cos, sin = dflash_rope(np.arange(max_position, dtype=np.float64), head_dim, rope_theta)
        cos_d, sin_d = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        write_bf16("dflash2.cos_cache", cos_d)
        write_bf16("dflash2.hidden_norm.weight", dflash_loader.read_tensor("hidden_norm.weight"))

        for lid in range(5):
            write_bf16(
                f"dflash2.layers.{lid}.attention_conv.kernel_projection.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.attention_conv.kernel_projection.weight").T,
            )
            write_bf16(
                f"dflash2.layers.{lid}.input_layernorm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.input_layernorm.weight"),
            )
            write_bf16(
                f"dflash2.layers.{lid}.mlp_conv.kernel_projection.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.mlp_conv.kernel_projection.weight").T,
            )
            write_bf16(
                f"dflash2.layers.{lid}.post_attention_layernorm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.post_attention_layernorm.weight"),
            )
            write_bf16(
                f"dflash2.layers.{lid}.self_attn.k_norm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.self_attn.k_norm.weight"),
            )
            write_bf16(
                f"dflash2.layers.{lid}.self_attn.q_norm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.self_attn.q_norm.weight"),
            )

        write_bf16("dflash2.norm.weight", dflash_loader.read_tensor("norm.weight"))
        write_bf16(
            "dflash2.selector.hidden_projection.MatMul.weight",
            dflash_loader.read_tensor("candidate_selector.hidden_projection.weight").T,
        )
        write_bf16("dflash2.sin_cache", sin_d)
    finally:
        writer.close()

    dflash_proto = build_dflash2_model_proto(
        dflash_locations,
        base_kernels_u16,
        block_size=block_size,
        num_draft_tokens=num_draft_tokens,
        selector_top_k=selector_top_k,
        sliding_window=sliding_window,
    )
    onnx.save_model(dflash_proto, str(out_dir / "dflash2.onnx"), save_as_external_data=False)
    return dflash_locations


@torch.no_grad()
def build_expert_profile_from_hf(
    hf_loader: CheckpointTensorLoader,
    out_path: Path,
    layer_kl_weights: list[float] | None = None,
    num_layers: int = 48,
    num_experts: int = 512,
) -> None:
    """Synthesize STRP v1 `expert-profile.bin` ranking `(layer, expert)` uint16 pairs from router gate norms."""
    scores: list[tuple[float, int, int]] = []
    for lid in range(num_layers):
        g = hf_loader.read_tensor(f"model.language_model.layers.{lid}.mlp.gate.weight").float()
        row_norms = torch.linalg.vector_norm(g, dim=1).tolist()
        lw = float(layer_kl_weights[lid]) if layer_kl_weights is not None and lid < len(layer_kl_weights) else 1.0
        for eid, rn in enumerate(row_norms[:num_experts]):
            scores.append((rn * (1.0 + lw), lid, eid))
    scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    n_ranked = len(scores)
    header = struct.pack("<6I", STRP_MAGIC, 1, num_layers, num_experts, n_ranked, n_ranked)
    entries = np.zeros((n_ranked, 2), dtype=np.uint16)
    for idx, (_sc, lid, eid) in enumerate(scores):
        entries[idx, 0] = lid
        entries[idx, 1] = eid
    out_path.write_bytes(header + entries.tobytes())


def write_weights_pool_txt(out_dir: Path, tensor_locations: dict[str, dict]) -> None:
    """Write `weights_pool.txt` mapping the 5 device-pooled initializers across `target` and `dflash2`."""
    pooled_specs = [
        ("model.rotary_emb.cos_cache", ("target",)),
        ("model.rotary_emb.sin_cache", ("target",)),
        ("capture.indexer_cos_cache.folded", ("target",)),
        ("capture.indexer_sin_cache.folded", ("target",)),
        ("lm_head.weight_t.qweight", ("target", "dflash2")),
    ]
    pool_lines = ["qwen38_weights_pool 1"]
    session_lines: list[str] = []
    for pid, (t_name, sessions) in enumerate(pooled_specs):
        meta = tensor_locations[t_name]
        dims_str = " ".join(str(d) for d in meta["dims"])
        pool_lines.append(
            f"pool {pid} {meta['dtype']} {meta['offset']} {meta['length']} {len(meta['dims'])} {dims_str} {meta['location']}"
        )
        for sess in sessions:
            session_lines.append(f"session {sess} {t_name} {pid}")
    (out_dir / "weights_pool.txt").write_text("\n".join(pool_lines + session_lines) + "\n")


def load_bank_metadata_from_dir(bank_dir: Path) -> tuple[dict[str, dict], dict[str, TensorProto]]:
    """Load `(tensor_locations, embedded_weights)` from `bank_dir/bank_metadata.json` or existing ONNX headers."""
    meta_json = bank_dir / "bank_metadata.json"
    if meta_json.is_file():
        data = json.loads(meta_json.read_text())
        tensor_locations = data["tensor_locations"]
        embedded_weights: dict[str, TensorProto] = {}
        for name, item in data["embedded_weights"].items():
            t = TensorProto()
            t.name = name
            t.data_type = int(item["dtype"])
            t.dims.extend(item["dims"])
            t.raw_data = base64.b85decode(item["raw_b85"])
            embedded_weights[name] = t
        return tensor_locations, embedded_weights

    # Fallback also supports a package containing only its explicitly selected fixed graph.
    tensor_locations = {}
    embedded_weights = {}
    target_filename = "model.onnx"
    config_path = bank_dir / "genai_config.json"
    if config_path.is_file():
        target_filename = json.loads(config_path.read_text())["model"]["decoder"]["filename"]
    if not (bank_dir / target_filename).resolve().is_relative_to(bank_dir.resolve()):
        raise ValueError("Target graph must be inside the reused package")
    for m_name in (target_filename, "dflash2.onnx"):
        p = bank_dir / m_name
        if not p.is_file():
            continue
        m = onnx.load(str(p), load_external_data=False)
        for t in m.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if ext and "location" in ext:
                tensor_locations[t.name] = {
                    "location": ext["location"],
                    "offset": int(ext.get("offset", 0)),
                    "length": int(ext.get("length", 0)),
                    "dtype": int(t.data_type),
                    "dims": list(t.dims),
                }
            elif (
                m_name == target_filename
                and ("layers." in t.name or "hyper_connection_mixer" in t.name)
                and t.name != "v_model.layers.0.linear_attn.CastLike_145"
            ):
                t_copy = TensorProto()
                t_copy.CopyFrom(t)
                embedded_weights[t.name] = t_copy
    return tensor_locations, embedded_weights


def save_bank_metadata_to_dir(
    out_dir: Path,
    tensor_locations: dict[str, dict],
    embedded_weights: dict[str, TensorProto],
    position_config: dict | None = None,
) -> None:
    """Save `bank_metadata.json` so subsequent multi-preset builds can reuse binary `.data` banks without parsing ONNX."""
    non_qmoe_locs = {k: v for k, v in tensor_locations.items() if not str(v.get("location", "")).startswith("qmoe_")}
    emb_serialized = {
        name: {
            "dtype": int(t.data_type),
            "dims": list(t.dims),
            "raw_b85": base64.b85encode(t.raw_data).decode("ascii"),
        }
        for name, t in embedded_weights.items()
    }
    metadata = {"tensor_locations": non_qmoe_locs, "embedded_weights": emb_serialized}
    if position_config is not None:
        metadata["position_config"] = position_config
    (out_dir / "bank_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_reused_position_tables(
    out_dir: Path, tensor_locations: dict[str, dict], rope: RopeConfig, target_rows: int, draft_rows: int
) -> None:
    """Construct new positional assets without rewriting or requantizing reused weight banks."""
    writer = BinaryBlobWriter(out_dir / "position_tables.data")
    try:
        cos, sin = target_rope(torch.arange(target_rows, dtype=torch.float32), rope)
        for index, name in enumerate(TARGET_NAMES):
            value = (cos, sin)[index % 2].to(torch.bfloat16)
            if index >= 2:
                value = torch.cat((value, value), dim=-1)
            tensor_locations[name] = writer.write_aligned(
                value.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(value.shape)
            )
        cos, sin = dflash_rope(np.arange(draft_rows, dtype=np.float64))
        for name, table in zip(DRAFT_NAMES, (cos, sin), strict=True):
            value = table.to(torch.bfloat16)
            tensor_locations[name] = writer.write_aligned(
                value.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(value.shape)
            )
    finally:
        writer.close()
