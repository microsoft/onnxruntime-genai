#!/usr/bin/env python3
"""
Convert Qwen3.8-Flash-Next RC2 packages (both NVFP4+FP8 and INT4+INT8 variants)
to use ONLY standard ONNX operators and `com.microsoft` contrib operators
(PRs #33393, #33469, #33470, #33471, #33472, #33473, #33474), eliminating all
`qwen.*` custom operators and `custom_ops_libraries`.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch

ALIGN = 4096


def align_up(v: int, a: int = ALIGN) -> int:
    return (v + a - 1) & ~(a - 1)


def get_attr_map(node: onnx.NodeProto) -> dict:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def set_external(tensor: TensorProto, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> None:
    tensor.data_type = dtype
    del tensor.dims[:]
    tensor.dims.extend(dims)
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
        e = tensor.external_data.add()
        e.key, e.value = k, v


def untile_hc_u8(w_tiled: torch.Tensor, kind: int) -> torch.Tensor:
    if kind == 0:
        return w_tiled.view(8, 20, 10, 128, 2, 8).permute(0, 2, 3, 1, 4, 5).contiguous().view(10240, 320)
    return w_tiled.view(160, 10, 32, 4, 2, 8).permute(1, 2, 3, 0, 4, 5).contiguous().view(320, 10240)


def ensure_ms_opset(model: onnx.ModelProto, version: int = 1) -> None:
    keep = [op for op in model.opset_import if op.domain in ("", "ai.onnx", "com.microsoft")]
    if not any(op.domain == "com.microsoft" for op in keep):
        keep.append(helper.make_opsetid("com.microsoft", version))
    del model.opset_import[:]
    model.opset_import.extend(keep)


class ContribWeightWriter:
    def __init__(self, src_dir: Path, out_dir: Path) -> None:
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.data_file_name = "contrib_weights.data"
        self.data_path = out_dir / self.data_file_name
        self.fh = open(self.data_path, "wb")
        self.pos = 0
        self.cache: dict[tuple, dict] = {}

    def close(self) -> None:
        self.fh.close()
        if self.pos == 0 and self.data_path.exists():
            self.data_path.unlink()

    def write_bytes(self, raw: bytes, dtype: int, dims: list[int]) -> dict:
        pad = align_up(self.pos) - self.pos
        if pad:
            self.fh.write(b"\x00" * pad)
            self.pos += pad
        off = self.pos
        self.fh.write(raw)
        self.pos += len(raw)
        return {
            "location": self.data_file_name,
            "offset": off,
            "length": len(raw),
            "dtype": dtype,
            "dims": list(dims),
        }

    def read_tensor_bytes(self, t: TensorProto) -> tuple[tuple, bytes]:
        ext = {e.key: e.value for e in t.external_data}
        if ext:
            p = self.src_dir / ext["location"]
            st = os.stat(p)
            off = int(ext.get("offset", 0))
            ln = int(ext.get("length", 0)) or (st.st_size - off)
            with open(p, "rb") as f:
                f.seek(off)
                raw = f.read(ln)
            return (st.st_ino, off, ln), raw
        raw = bytes(t.raw_data)
        return (hashlib.sha256(raw).hexdigest(), 0, len(raw)), raw

    def transpose_fp8_weight_and_scale(self, t: TensorProto, w_scale: list[float]) -> tuple[dict, dict]:
        src_key, raw = self.read_tensor_bytes(t)
        ck = (*src_key, "fp8_nk")
        if ck in self.cache:
            return self.cache[ck]["w"], self.cache[ck]["s"]
        k_dim, n_dim = int(t.dims[0]), int(t.dims[1])
        arr_kn = np.frombuffer(raw, dtype=np.uint8).reshape(k_dim, n_dim)
        arr_nk = np.ascontiguousarray(arr_kn.T)
        sc_arr = np.asarray(w_scale, dtype=np.float32).reshape(n_dim, 1)
        w_info = self.write_bytes(arr_nk.tobytes(), TensorProto.FLOAT8E4M3FN, [n_dim, k_dim])
        s_info = self.write_bytes(sc_arr.tobytes(), TensorProto.FLOAT, [n_dim, 1])
        self.cache[ck] = {"w": w_info, "s": s_info}
        return w_info, s_info

    def quantize_bf16_to_fp8_nk(self, t: TensorProto) -> tuple[dict, dict]:
        src_key, raw = self.read_tensor_bytes(t)
        ck = (*src_key, "bf16_to_fp8_nk")
        if ck in self.cache:
            return self.cache[ck]["w"], self.cache[ck]["s"]
        k_dim, n_dim = int(t.dims[0]), int(t.dims[1])
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        w_bf16 = torch.from_numpy(u16).view(torch.bfloat16).reshape(k_dim, n_dim).float()
        amax = w_bf16.abs().amax(dim=0).clamp(min=1e-12)
        w_scale = (amax / 448.0).to(torch.float32)
        wq = (w_bf16 / w_scale[None, :]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        arr_nk = np.ascontiguousarray(wq.numpy().T)
        sc_arr = w_scale.numpy().reshape(n_dim, 1)
        w_info = self.write_bytes(arr_nk.tobytes(), TensorProto.FLOAT8E4M3FN, [n_dim, k_dim])
        s_info = self.write_bytes(sc_arr.tobytes(), TensorProto.FLOAT, [n_dim, 1])
        self.cache[ck] = {"w": w_info, "s": s_info}
        return w_info, s_info

    def untile_hc_fp8_to_bf16(self, t: TensorProto, scale_list: list[float], kind: int) -> dict:
        src_key, raw = self.read_tensor_bytes(t)
        ck = (*src_key, f"hc_untile_{kind}")
        if ck in self.cache:
            return self.cache[ck]
        u8 = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(160, 20480)
        untiled_u8 = untile_hc_u8(u8, kind)
        sc = torch.tensor(scale_list, dtype=torch.float32)
        w_f32 = untiled_u8.view(torch.float8_e4m3fn).float() * sc[None, :]
        w_bf16 = w_f32.to(torch.bfloat16).contiguous()
        dims = [10240, 320] if kind == 0 else [320, 10240]
        info = self.write_bytes(w_bf16.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, dims)
        self.cache[ck] = info
        return info


def make_ext_init(name: str, info: dict) -> TensorProto:
    t = TensorProto()
    t.name = name
    set_external(t, info["location"], info["offset"], info["length"], info["dtype"], info["dims"])
    return t


def convert_model_to_contrib_ops(
    model_path: Path,
    writer: ContribWeightWriter,
    is_mtp: bool = False,
    fp8_mlp_shared: bool = False,
    fp8_shared_down_only: bool = False,
    hc_bf16_inits: dict[str, TensorProto] | None = None,
) -> dict:
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    inits: dict[str, TensorProto] = {t.name: t for t in graph.initializer}
    new_inits: dict[str, TensorProto] = {}
    stats: collections.Counter = collections.Counter()

    def ensure_const(name: str, arr: np.ndarray) -> str:
        if name not in inits and name not in new_inits:
            new_inits[name] = numpy_helper.from_array(arr, name)
        return name

    def ensure_bf16_norm_scale(f32_name: str) -> str:
        bf16_name = f"{f32_name}.bf16"
        if bf16_name in inits or bf16_name in new_inits:
            return bf16_name
        t_src = inits[f32_name]
        if t_src.data_type == TensorProto.BFLOAT16:
            return f32_name
        _, raw = writer.read_tensor_bytes(t_src)
        f32_arr = np.frombuffer(raw, dtype=np.float32).copy()
        bf16_u16 = torch.from_numpy(f32_arr).to(torch.bfloat16).view(torch.uint16).numpy()
        t_bf16 = TensorProto()
        t_bf16.name = bf16_name
        t_bf16.data_type = TensorProto.BFLOAT16
        t_bf16.dims.extend(list(t_src.dims))
        t_bf16.raw_data = bf16_u16.tobytes()
        new_inits[bf16_name] = t_bf16
        return bf16_name

    # Pre-fuse 7/8-node INT8 HC chains into com.microsoft::HyperConnectionLowRankMix when hc_bf16_inits is provided
    if hc_bf16_inits:
        nodes_in = list(graph.node)
        pre_rebuilt: list[onnx.NodeProto] = []
        i = 0
        while i < len(nodes_in):
            # Check 8-node HcPostBlock chain: [HyperConnectionPostMix, BranchwiseRMSNorm, MatMulNBits, ScaledSiLU, MatMulNBits, Sigmoid, HyperConnectionPreMix, HcInjectGate]
            if (
                i + 7 < len(nodes_in)
                and nodes_in[i].op_type == "HyperConnectionPostMix"
                and nodes_in[i + 1].op_type == "BranchwiseRMSNorm"
                and nodes_in[i + 2].op_type == "MatMulNBits"
                and nodes_in[i + 3].op_type == "ScaledSiLU"
                and nodes_in[i + 4].op_type == "MatMulNBits"
                and nodes_in[i + 5].op_type == "Sigmoid"
                and nodes_in[i + 6].op_type == "HyperConnectionPreMix"
                and nodes_in[i + 7].op_type == "HcInjectGate"
                and nodes_in[i + 1].input[0] == nodes_in[i].output[0]
            ):
                n_post, n_norm, n_down, n_silu, n_up, _, n_pre, n_inj = nodes_in[i : i + 8]
                wd_bf16_name = n_down.input[1].removesuffix(".qweight") + ".untiled_bf16"
                wu_bf16_name = n_up.input[1].removesuffix(".qweight") + ".untiled_bf16"
                if wd_bf16_name in hc_bf16_inits and wu_bf16_name in hc_bf16_inits:
                    new_inits[wd_bf16_name] = hc_bf16_inits[wd_bf16_name]
                    new_inits[wu_bf16_name] = hc_bf16_inits[wu_bf16_name]
                    norm_bf16_name = ensure_bf16_norm_scale(n_norm.input[1])
                    norm_attrs = get_attr_map(n_norm)
                    silu_attrs = get_attr_map(n_silu)
                    pre_attrs = get_attr_map(n_pre)
                    inj_attrs = get_attr_map(n_inj)
                    pre_rebuilt.append(
                        helper.make_node(
                            "HyperConnectionLowRankMix",
                            [
                                n_post.input[0],
                                norm_bf16_name,
                                wd_bf16_name,
                                wu_bf16_name,
                                n_inj.input[1],
                                n_post.input[1],
                                n_post.input[2],
                            ],
                            [n_pre.output[0], n_inj.output[0], n_post.output[0]],
                            name=f"{n_norm.name}/HyperConnectionLowRankMix",
                            domain="com.microsoft",
                            num_branches=4,
                            epsilon=float(norm_attrs.get("epsilon", 1e-6)),
                            reduction_scale=float(pre_attrs.get("reduction_scale", 0.25)),
                            silu_alpha=float(silu_attrs.get("alpha", 0.25)),
                            gate_div=float(inj_attrs.get("div", 4.0)),
                            gate_mul=float(inj_attrs.get("mul", 2.0)),
                        )
                    )
                    stats["Int8HcPostChain_to_HyperConnectionLowRankMix"] += 1
                    i += 8
                    continue
            # Check 7-node HcBlock chain: [BranchwiseRMSNorm, MatMulNBits, ScaledSiLU, MatMulNBits, Sigmoid, HyperConnectionPreMix, HcInjectGate]
            if (
                i + 6 < len(nodes_in)
                and nodes_in[i].op_type == "BranchwiseRMSNorm"
                and nodes_in[i + 1].op_type == "MatMulNBits"
                and nodes_in[i + 2].op_type == "ScaledSiLU"
                and nodes_in[i + 3].op_type == "MatMulNBits"
                and nodes_in[i + 4].op_type == "Sigmoid"
                and nodes_in[i + 5].op_type == "HyperConnectionPreMix"
                and nodes_in[i + 6].op_type == "HcInjectGate"
            ):
                n_norm, n_down, n_silu, n_up, _, n_pre, n_inj = nodes_in[i : i + 7]
                wd_bf16_name = n_down.input[1].removesuffix(".qweight") + ".untiled_bf16"
                wu_bf16_name = n_up.input[1].removesuffix(".qweight") + ".untiled_bf16"
                if wd_bf16_name in hc_bf16_inits and wu_bf16_name in hc_bf16_inits:
                    new_inits[wd_bf16_name] = hc_bf16_inits[wd_bf16_name]
                    new_inits[wu_bf16_name] = hc_bf16_inits[wu_bf16_name]
                    norm_bf16_name = ensure_bf16_norm_scale(n_norm.input[1])
                    norm_attrs = get_attr_map(n_norm)
                    silu_attrs = get_attr_map(n_silu)
                    pre_attrs = get_attr_map(n_pre)
                    inj_attrs = get_attr_map(n_inj)
                    pre_rebuilt.append(
                        helper.make_node(
                            "HyperConnectionLowRankMix",
                            [
                                n_norm.input[0],
                                norm_bf16_name,
                                wd_bf16_name,
                                wu_bf16_name,
                                n_inj.input[1],
                            ],
                            [n_pre.output[0], n_inj.output[0]],
                            name=f"{n_norm.name}/HyperConnectionLowRankMix",
                            domain="com.microsoft",
                            num_branches=4,
                            epsilon=float(norm_attrs.get("epsilon", 1e-6)),
                            reduction_scale=float(pre_attrs.get("reduction_scale", 0.25)),
                            silu_alpha=float(silu_attrs.get("alpha", 0.25)),
                            gate_div=float(inj_attrs.get("div", 4.0)),
                            gate_mul=float(inj_attrs.get("mul", 2.0)),
                        )
                    )
                    stats["Int8HcBlockChain_to_HyperConnectionLowRankMix"] += 1
                    i += 7
                    continue
            pre_rebuilt.append(nodes_in[i])
            i += 1
        del graph.node[:]
        graph.node.extend(pre_rebuilt)

    # Pre-scan consumer map to find where SkinnyGemm / MatMulNBits outputs go
    consumers_by_val: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
    producer_by_val: dict[str, onnx.NodeProto] = {}
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumers_by_val[inp].append(n)
        for out in n.output:
            if out:
                producer_by_val[out] = n

    # Track GDN 3-Reshape nodes and DSA 2-Reshape nodes to skip when fused
    skip_nodes: set[int] = set()
    # Map GdnProjPrep output[0] -> GdnRecurrent node & its 3 Reshape nodes
    gdn_info_by_proj_out: dict[str, tuple[onnx.NodeProto, list[onnx.NodeProto]]] = {}
    for n in graph.node:
        if n.op_type == "GdnProjPrep":
            r_nodes = []
            rec_node = None
            for i in range(3):
                cs = consumers_by_val.get(n.output[i], [])
                if len(cs) == 1 and cs[0].op_type == "Reshape":
                    r_nodes.append(cs[0])
                    rcs = consumers_by_val.get(cs[0].output[0], [])
                    if len(rcs) == 1 and rcs[0].op_type == "GdnRecurrent":
                        rec_node = rcs[0]
            if len(r_nodes) == 3 and rec_node is not None:
                for rn in r_nodes:
                    skip_nodes.add(id(rn))
                skip_nodes.add(id(rec_node))
                gdn_info_by_proj_out[n.input[0]] = (rec_node, r_nodes)

    # Track DSA Reshape_546 and Reshape_547 before GuardedDSA
    dsa_raw_sel_by_reshaped: dict[str, str] = {}
    for n in graph.node:
        if n.op_type == "GuardedDSA":
            for idx_in in (5, 6):
                prod = producer_by_val.get(n.input[idx_in])
                if prod is not None and prod.op_type == "Reshape" and len(consumers_by_val[prod.output[0]]) == 1:
                    skip_nodes.add(id(prod))
                    dsa_raw_sel_by_reshaped[n.input[idx_in]] = prod.input[0]

    # Track PLE Transpose_230 and Transpose_232 around CausalConvWithState_node_231
    ple_conv_mid_out: dict[str, str] = {}
    for n in graph.node:
        if n.op_type == "PleMidState":
            skip_nodes.add(id(n))
            ple_conv_mid_out["conv_mid"] = n.output[0]
            ple_conv_mid_out["ctx_mid"] = n.output[1]
    for n in graph.node:
        if n.name in (
            "model/layers.1/ple/conv1d/Transpose_node_230",
            "model/layers.1/ple/conv1d/Transpose_node_232",
        ):
            skip_nodes.add(id(n))

    # Track Gather_cos_1d and Gather_sin_1d (replaced by direct cos_cache/sin_cache in RMSNormRotary)
    rope_cache_by_gathered: dict[str, tuple[str, str]] = {}
    for n in graph.node:
        if n.name in ("model/rotary_emb/Gather_cos_1d", "model/rotary_emb/Gather_sin_1d"):
            skip_nodes.add(id(n))
            rope_cache_by_gathered[n.output[0]] = (n.input[0], n.input[1])

    # Track QsaProjPrep and GdnProjPrep consumers of merged_proj.out
    qsa_prep_by_in: dict[str, onnx.NodeProto] = {}
    gdn_prep_by_in: dict[str, onnx.NodeProto] = {}
    for n in graph.node:
        if n.op_type == "QsaProjPrep":
            qsa_prep_by_in[n.input[0]] = n
        elif n.op_type == "GdnProjPrep":
            gdn_prep_by_in[n.input[0]] = n

    # Map QsaProjPrep merged_proj.out -> q_gate_proj partition name (for SigmoidGateMul)
    q_gate_part_by_merged: dict[str, str] = {}
    # Map GdnProjPrep merged_proj.out -> (qkv_part, z_part, a_part, b_part)
    gdn_parts_by_merged: dict[str, tuple[str, str, str, str]] = {}
    # Map QsaProjPrep merged_proj.out -> (idx_qk_part, q_gate_part, k_part, v_part)
    qsa_parts_by_merged: dict[str, tuple[str, str, str, str]] = {}

    const_q_4d = ensure_const("contrib_const_q_4d", np.array([0, 0, 24, 512], dtype=np.int64))
    const_k_4d = ensure_const("contrib_const_k_4d", np.array([0, 0, 2, 256], dtype=np.int64))
    const_rope_2d = ensure_const("contrib_const_rope_2d", np.array([-1, 32], dtype=np.int64))
    const_pos_zero = ensure_const("contrib_const_pos_zero", np.array([0], dtype=np.int64))
    const_dsa_total_seq = ensure_const("contrib_const_dsa_total_seq", np.array([33028], dtype=np.int32))
    const_split_512_640_640 = ensure_const("contrib_const_split_512_640_640", np.array([512, 640, 640], dtype=np.int64))
    const_split_gdn = ensure_const("contrib_const_split_gdn", np.array([10240, 6144, 48, 48], dtype=np.int64))
    const_split_qsa = ensure_const("contrib_const_split_qsa", np.array([640, 12288, 512, 512], dtype=np.int64))
    const_router_2d_shape = ensure_const("contrib_const_router_2d_shape", np.array([-1, 512], dtype=np.int64))
    const_hc_div = ensure_const("contrib_const_hc_div_4", np.array(4.0 / 256.0, dtype=np.uint16))  # unused if fused

    rebuilt_nodes: list[onnx.NodeProto] = []

    for n in graph.node:
        if id(n) in skip_nodes:
            continue

        # 1. PLE Concat_node_195 -> also emit Slice for present_mid.1.ple_context
        if n.name == "model/layers.1/ple/ple_embedding/Concat_node_195":
            rebuilt_nodes.append(n)
            if "ctx_mid" in ple_conv_mid_out:
                c_s1 = ensure_const("contrib_ple_mid_starts", np.array([1], dtype=np.int64))
                c_e3 = ensure_const("contrib_ple_mid_ends", np.array([3], dtype=np.int64))
                c_ax1 = ensure_const("contrib_ple_mid_axes", np.array([1], dtype=np.int64))
                rebuilt_nodes.append(
                    helper.make_node(
                        "Slice",
                        [n.output[0], c_s1, c_e3, c_ax1],
                        [ple_conv_mid_out["ctx_mid"]],
                        name="model/layers.1/ple/ple_embedding/Slice_mid_context",
                    )
                )
                stats["PleMidState_to_Slice_and_CausalConv"] += 1
            continue

        # 2. PLE CausalConvWithState_node_231 -> direct channels-first state + mid_state output
        if n.name == "model/layers.1/ple/conv1d/CausalConvWithState_node_231":
            rebuilt_nodes.append(
                helper.make_node(
                    "CausalConvWithState",
                    [n.input[0], n.input[1], "", "past_key_values.1.ple_conv_state"],
                    [n.output[0], "present.1.ple_conv_state", ple_conv_mid_out["conv_mid"]],
                    name=n.name,
                    domain="com.microsoft",
                    activation="silu",
                    ndim=1,
                    channels_last=1,
                    dilation=3,
                    input_channels=10240,
                    input_channel_offset=0,
                    state_length=9,
                    state_channels_last=0,
                )
            )
            continue

        # 3. qwen.fused_hc::HcBlock and qwen.fused_hc::HcPostBlock -> com.microsoft::HyperConnectionLowRankMix
        if n.domain == "qwen.fused_hc" and n.op_type in ("HcBlock", "HcPostBlock"):
            is_post = n.op_type == "HcPostBlock"
            attrs = get_attr_map(n)
            norm_w_name = n.input[3 if is_post else 1]
            wd_name = n.input[4 if is_post else 2]
            wi_name = n.input[5 if is_post else 3]
            wu_name = n.input[6 if is_post else 4]

            norm_bf16_name = ensure_bf16_norm_scale(norm_w_name)
            wd_untiled_name = f"{wd_name}.untiled_bf16"
            wu_untiled_name = f"{wu_name}.untiled_bf16"
            if wd_untiled_name not in inits and wd_untiled_name not in new_inits:
                wd_info = writer.untile_hc_fp8_to_bf16(inits[wd_name], list(attrs["wd_scale"]), 0)
                new_inits[wd_untiled_name] = make_ext_init(wd_untiled_name, wd_info)
            if wu_untiled_name not in inits and wu_untiled_name not in new_inits:
                wu_info = writer.untile_hc_fp8_to_bf16(inits[wu_name], list(attrs["wu_scale"]), 1)
                new_inits[wu_untiled_name] = make_ext_init(wu_untiled_name, wu_info)

            if is_post:
                hc_ins = [n.input[0], norm_bf16_name, wd_untiled_name, wu_untiled_name, wi_name, n.input[1], n.input[2]]
                hc_outs = [n.output[1], n.output[2], n.output[0]]
            else:
                hc_ins = [n.input[0], norm_bf16_name, wd_untiled_name, wu_untiled_name, wi_name]
                hc_outs = [n.output[0], n.output[1]]

            rebuilt_nodes.append(
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    hc_ins,
                    hc_outs,
                    name=f"{n.name}/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    num_branches=4,
                    epsilon=float(attrs.get("epsilon", 1e-6)),
                    reduction_scale=float(attrs.get("reduction_scale", 0.25)),
                    silu_alpha=float(attrs.get("silu_alpha", 0.25)),
                    gate_div=float(attrs.get("gate_div", 4.0)),
                    gate_mul=float(attrs.get("gate_mul", 2.0)),
                )
            )
            stats[f"{n.op_type}_to_HyperConnectionLowRankMix"] += 1
            continue

        # 4. qwen.fused::HcInjectGate in INT4+INT8 variant -> MatMul + Div + Sigmoid + Mul
        if n.domain == "qwen.fused" and n.op_type == "HcInjectGate":
            attrs = get_attr_map(n)
            div_val = float(attrs.get("div", 4.0))
            mul_val = float(attrs.get("mul", 2.0))
            div_c = ensure_const(
                "contrib_hc_gate_div_bf16",
                torch.tensor([div_val], dtype=torch.bfloat16).view(torch.uint16).numpy(),
            )
            new_inits[div_c].data_type = TensorProto.BFLOAT16
            mul_c = ensure_const(
                "contrib_hc_gate_mul_bf16",
                torch.tensor([mul_val], dtype=torch.bfloat16).view(torch.uint16).numpy(),
            )
            new_inits[mul_c].data_type = TensorProto.BFLOAT16
            mm_out = f"{n.name}/mm"
            div_out = f"{n.name}/div"
            sig_out = f"{n.name}/sig"
            rebuilt_nodes.extend(
                [
                    helper.make_node("MatMul", [n.input[0], n.input[1]], [mm_out], name=f"{n.name}/MatMul"),
                    helper.make_node("Div", [mm_out, div_c], [div_out], name=f"{n.name}/Div"),
                    helper.make_node("Sigmoid", [div_out], [sig_out], name=f"{n.name}/Sigmoid"),
                    helper.make_node("Mul", [sig_out, mul_c], [n.output[0]], name=f"{n.name}/Mul"),
                ]
            )
            stats["HcInjectGate_to_ONNX"] += 1
            continue

        # 5. MatMulNBits feeding GdnProjPrep or QsaProjPrep in INT4+INT8 variant
        if n.domain == "com.microsoft" and n.op_type == "MatMulNBits":
            rebuilt_nodes.append(n)
            out_name = n.output[0]
            if out_name in gdn_prep_by_in:
                gprep = gdn_prep_by_in[out_name]
                qkv_p = f"{out_name}.qkv_10240"
                z_p = gprep.output[3]
                a_p = f"{out_name}.a_48"
                b_p = f"{out_name}.b_48"
                rebuilt_nodes.append(
                    helper.make_node(
                        "Split",
                        [out_name, const_split_gdn],
                        [qkv_p, z_p, a_p, b_p],
                        name=f"{n.name}/SplitGdn",
                        axis=-1,
                    )
                )
                gdn_parts_by_merged[out_name] = (qkv_p, z_p, a_p, b_p)
            elif out_name in qsa_prep_by_in:
                qprep = qsa_prep_by_in[out_name]
                idx_p = qprep.output[2]
                qg_p = f"{out_name}.q_gate_12288"
                k_p = f"{out_name}.k_512"
                v_p = qprep.output[3]
                rebuilt_nodes.append(
                    helper.make_node(
                        "Split",
                        [out_name, const_split_qsa],
                        [idx_p, qg_p, k_p, v_p],
                        name=f"{n.name}/SplitQsa",
                        axis=-1,
                    )
                )
                qsa_parts_by_merged[out_name] = (idx_p, qg_p, k_p, v_p)
                q_gate_part_by_merged[out_name] = qg_p
            continue

        # 6. qwen.fused_gemm::SkinnyGemm -> com.microsoft::MatMulBlockQuantizedFp8Weight (or BF16 MatMul + Split)
        if n.domain == "qwen.fused_gemm" and n.op_type == "SkinnyGemm":
            attrs = get_attr_map(n)
            w_name = n.input[1]
            t_w = inits[w_name]
            widths = [int(x) for x in attrs.get("widths", [t_w.dims[1]])]
            if t_w.data_type == TensorProto.BFLOAT16 and not fp8_mlp_shared:
                mm_out = f"{n.name}/bf16_mm_out" if len(n.output) > 1 else n.output[0]
                rebuilt_nodes.append(helper.make_node("MatMul", [n.input[0], w_name], [mm_out], name=f"{n.name}/MatMulBf16"))
                if len(n.output) > 1:
                    sp_const = ensure_const(f"{n.name}_split_sizes", np.array(widths, dtype=np.int64))
                    rebuilt_nodes.append(
                        helper.make_node("Split", [mm_out, sp_const], list(n.output), name=f"{n.name}/Split", axis=-1)
                    )
                stats["SkinnyGemm_Bf16_to_MatMul"] += 1
                continue

            k_dim, n_dim = int(t_w.dims[0]), int(t_w.dims[1])
            nk_name = f"{w_name}.fp8_nk"
            sc_name = f"{w_name}.scale_n1"
            if nk_name not in inits and nk_name not in new_inits:
                if t_w.data_type == TensorProto.BFLOAT16:
                    w_info, s_info = writer.quantize_bf16_to_fp8_nk(t_w)
                else:
                    w_scale = [float(x) for x in attrs["w_scale"]]
                    w_info, s_info = writer.transpose_fp8_weight_and_scale(t_w, w_scale)
                new_inits[nk_name] = make_ext_init(nk_name, w_info)
                new_inits[sc_name] = make_ext_init(sc_name, s_info)

            out_name = n.output[0]
            if len(n.output) == 1 and out_name in gdn_prep_by_in:
                gprep = gdn_prep_by_in[out_name]
                qkv_p = f"{out_name}.qkv_10240"
                z_p = gprep.output[3]
                a_p = f"{out_name}.a_48"
                b_p = f"{out_name}.b_48"
                rebuilt_nodes.append(
                    helper.make_node(
                        "MatMulBlockQuantizedFp8Weight",
                        [n.input[0], nk_name, sc_name],
                        [qkv_p, z_p, a_p, b_p],
                        name=f"{n.name}/MatMulBlockQuantizedFp8Weight",
                        domain="com.microsoft",
                        block_size=k_dim,
                        split_sizes=[10240, 6144, 48, 48],
                    )
                )
                gdn_parts_by_merged[out_name] = (qkv_p, z_p, a_p, b_p)
            elif len(n.output) == 1 and out_name in qsa_prep_by_in:
                qprep = qsa_prep_by_in[out_name]
                idx_p = qprep.output[2]
                qg_p = f"{out_name}.q_gate_12288"
                k_p = f"{out_name}.k_512"
                v_p = qprep.output[3]
                rebuilt_nodes.append(
                    helper.make_node(
                        "MatMulBlockQuantizedFp8Weight",
                        [n.input[0], nk_name, sc_name],
                        [idx_p, qg_p, k_p, v_p],
                        name=f"{n.name}/MatMulBlockQuantizedFp8Weight",
                        domain="com.microsoft",
                        block_size=k_dim,
                        split_sizes=[640, 12288, 512, 512],
                    )
                )
                qsa_parts_by_merged[out_name] = (idx_p, qg_p, k_p, v_p)
                q_gate_part_by_merged[out_name] = qg_p
            elif len(n.output) > 1:
                rebuilt_nodes.append(
                    helper.make_node(
                        "MatMulBlockQuantizedFp8Weight",
                        [n.input[0], nk_name, sc_name],
                        list(n.output),
                        name=f"{n.name}/MatMulBlockQuantizedFp8Weight",
                        domain="com.microsoft",
                        block_size=k_dim,
                        split_sizes=widths,
                    )
                )
            else:
                rebuilt_nodes.append(
                    helper.make_node(
                        "MatMulBlockQuantizedFp8Weight",
                        [n.input[0], nk_name, sc_name],
                        list(n.output),
                        name=f"{n.name}/MatMulBlockQuantizedFp8Weight",
                        domain="com.microsoft",
                        block_size=k_dim,
                    )
                )
            stats["SkinnyGemm_to_MatMulBlockQuantizedFp8Weight"] += 1
            continue

        # 7. qwen.fused_gemm::SharedExpertDown -> MatMulBlockQuantizedFp8Weight(shared_expert=1) or SigmoidGateMul + MatMul
        if n.domain == "qwen.fused_gemm" and n.op_type == "SharedExpertDown":
            gate_in, up_in, down_w, hidden_in, gate_w, qmoe_out = n.input[:6]
            t_dw = inits[down_w]
            if fp8_mlp_shared or fp8_shared_down_only or t_dw.data_type == TensorProto.FLOAT8E4M3FN:
                k_dim, n_dim = int(t_dw.dims[0]), int(t_dw.dims[1])
                nk_name = f"{down_w}.fp8_nk"
                sc_name = f"{down_w}.scale_n1"
                if nk_name not in inits and nk_name not in new_inits:
                    if t_dw.data_type == TensorProto.BFLOAT16:
                        w_info, s_info = writer.quantize_bf16_to_fp8_nk(t_dw)
                    else:
                        attrs = get_attr_map(n)
                        w_info, s_info = writer.transpose_fp8_weight_and_scale(t_dw, [float(x) for x in attrs["w_scale"]])
                    new_inits[nk_name] = make_ext_init(nk_name, w_info)
                    new_inits[sc_name] = make_ext_init(sc_name, s_info)
                rebuilt_nodes.append(
                    helper.make_node(
                        "MatMulBlockQuantizedFp8Weight",
                        [up_in, nk_name, sc_name, "", "", "", "", "", "", gate_in, hidden_in, gate_w, qmoe_out],
                        [n.output[0]],
                        name=f"{n.name}/MatMulBlockQuantizedFp8Weight",
                        domain="com.microsoft",
                        block_size=k_dim,
                        shared_expert=1,
                    )
                )
                stats["SharedExpertDown_to_MatMulBlockQuantizedFp8Weight"] += 1
                continue
            silu_out = f"{n.name}/silu_640"
            down_out = f"{n.name}/down_2560"
            rebuilt_nodes.extend(
                [
                    helper.make_node(
                        "SigmoidGateMul",
                        [up_in, gate_in],
                        [silu_out],
                        name=f"{n.name}/SiluGate",
                        domain="com.microsoft",
                        silu_gate=1,
                        sigmoid_in_fp32=1,
                    ),
                    helper.make_node(
                        "MatMul",
                        [silu_out, down_w],
                        [down_out],
                        name=f"{n.name}/DownMatMul",
                    ),
                    helper.make_node(
                        "SigmoidGateMul",
                        [down_out, hidden_in, qmoe_out, "", gate_w],
                        [n.output[0]],
                        name=f"{n.name}/ProjectedSigmoidGateMul",
                        domain="com.microsoft",
                        broadcast_gate=1,
                        sigmoid_in_fp32=1,
                    ),
                ]
            )
            stats["SharedExpertDown_to_SigmoidGateMul"] += 1
            continue

        # 8. qwen.fused::SplitRouterSwishMul and qwen.fused::SharedGateMulAdd in INT4+INT8 variant
        if n.domain == "qwen.fused" and n.op_type == "SplitRouterSwishMul":
            merged_in = n.input[0]
            router_2d_out, swish_out = n.output[:2]
            r_3d = f"{n.name}/router_3d"
            g_3d = f"{n.name}/gate_3d"
            u_3d = f"{n.name}/up_3d"
            rebuilt_nodes.extend(
                [
                    helper.make_node(
                        "Split",
                        [merged_in, const_split_512_640_640],
                        [r_3d, g_3d, u_3d],
                        name=f"{n.name}/Split",
                        axis=-1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [r_3d, const_router_2d_shape],
                        [router_2d_out],
                        name=f"{n.name}/ReshapeRouter2D",
                    ),
                    helper.make_node(
                        "SigmoidGateMul",
                        [u_3d, g_3d],
                        [swish_out],
                        name=f"{n.name}/SiluGate",
                        domain="com.microsoft",
                        silu_gate=1,
                        sigmoid_in_fp32=1,
                    ),
                ]
            )
            stats["SplitRouterSwishMul_to_SigmoidGateMul"] += 1
            continue

        if n.domain == "qwen.fused" and n.op_type == "SharedGateMulAdd":
            hidden_in, gate_w, down_out, qmoe_out = n.input[:4]
            rebuilt_nodes.append(
                helper.make_node(
                    "SigmoidGateMul",
                    [down_out, hidden_in, qmoe_out, "", gate_w],
                    [n.output[0]],
                    name=f"{n.name}/ProjectedSigmoidGateMul",
                    domain="com.microsoft",
                    broadcast_gate=1,
                    sigmoid_in_fp32=1,
                )
            )
            stats["SharedGateMulAdd_to_SigmoidGateMul"] += 1
            continue

        # 9. qwen.fused_attn::GdnProjPrep (+ GdnRecurrent) -> CausalConvWithState + GatedDeltaNet
        if n.domain == "qwen.fused_attn" and n.op_type == "GdnProjPrep":
            merged_out, past_conv, conv_w = n.input[:3]
            qkv_p, _, a_p, b_p = gdn_parts_by_merged[merged_out]
            rec_node, _ = gdn_info_by_proj_out[merged_out]
            conv_qkv_out = f"{n.name}/conv_qkv_out"
            rebuilt_nodes.append(
                helper.make_node(
                    "CausalConvWithState",
                    [qkv_p, conv_w, "", past_conv],
                    [conv_qkv_out, n.output[4], n.output[7]],
                    name=f"{n.name}/CausalConvWithState",
                    domain="com.microsoft",
                    activation="silu",
                    ndim=1,
                    channels_last=1,
                    dilation=1,
                    input_channels=10240,
                    input_channel_offset=0,
                    state_length=4,
                    state_channels_last=0,
                )
            )
            rebuilt_nodes.append(
                helper.make_node(
                    "GatedDeltaNet",
                    [
                        conv_qkv_out,
                        "",
                        "",
                        "",
                        b_p,
                        a_p,
                        rec_node.input[6],
                        rec_node.input[7],
                        rec_node.input[8],
                    ],
                    list(rec_node.output),
                    name=rec_node.name,
                    domain="com.microsoft",
                    update_rule="gated_delta",
                    gate_activation="softplus",
                    beta_activation="sigmoid",
                    qk_l2_norm=1,
                    scale=0.0,
                    state_output="mid_state",
                )
            )
            stats["GdnProjPrep_GdnRecurrent_to_Contrib"] += 1
            continue

        # 10. qwen.fused_attn::QsaProjPrep -> Reshape + com.microsoft::RMSNormRotary
        if n.domain == "qwen.fused_attn" and n.op_type == "QsaProjPrep":
            merged_out, q_norm, k_norm, rope_cos, rope_sin = n.input[:5]
            q_rot_out, k_rot_out, _, _ = n.output[:4]
            _, qg_p, k_p, _ = qsa_parts_by_merged[merged_out]
            attrs = get_attr_map(n)
            eps = float(attrs.get("epsilon", 1e-6))

            if rope_cos in rope_cache_by_gathered and rope_sin in rope_cache_by_gathered:
                cos_2d, pos_tensor = rope_cache_by_gathered[rope_cos]
                sin_2d, _ = rope_cache_by_gathered[rope_sin]
            else:
                cos_2d = f"{n.name}/cos_2d"
                sin_2d = f"{n.name}/sin_2d"
                pos_tensor = const_pos_zero
                rebuilt_nodes.extend(
                    [
                        helper.make_node("Reshape", [rope_cos, const_rope_2d], [cos_2d], name=f"{n.name}/ReshapeCos2D"),
                        helper.make_node("Reshape", [rope_sin, const_rope_2d], [sin_2d], name=f"{n.name}/ReshapeSin2D"),
                    ]
                )

            q_4d = f"{n.name}/q_4d"
            k_4d = f"{n.name}/k_4d"
            rebuilt_nodes.extend(
                [
                    helper.make_node("Reshape", [qg_p, const_q_4d], [q_4d], name=f"{n.name}/ReshapeQ4D"),
                    helper.make_node(
                        "RMSNormRotary",
                        [q_4d, q_norm, pos_tensor, cos_2d, sin_2d],
                        [q_rot_out],
                        name=f"{n.name}/RMSNormRotaryQ",
                        domain="com.microsoft",
                        epsilon=eps,
                        input_start=0,
                        input_width=256,
                        rotary_embedding_dim=64,
                        use_fp32_arithmetic=1,
                        flatten_batch=0,
                    ),
                    helper.make_node("Reshape", [k_p, const_k_4d], [k_4d], name=f"{n.name}/ReshapeK4D"),
                    helper.make_node(
                        "RMSNormRotary",
                        [k_4d, k_norm, pos_tensor, cos_2d, sin_2d],
                        [k_rot_out],
                        name=f"{n.name}/RMSNormRotaryK",
                        domain="com.microsoft",
                        epsilon=eps,
                        input_start=0,
                        input_width=256,
                        rotary_embedding_dim=64,
                        use_fp32_arithmetic=1,
                        flatten_batch=0,
                    ),
                ]
            )
            stats["QsaProjPrep_to_RMSNormRotary"] += 1
            continue

        # 11. qwen.capture::PackedQSAIndexer -> com.microsoft::PackedSparseAttentionIndexer
        if n.domain == "qwen.capture" and n.op_type == "PackedQSAIndexer":
            attrs = get_attr_map(n)
            psai_ins = list(n.input)
            psai_ins[11] = ""  # position_ids omitted for policy_mode='qsa'
            psai_outs = [
                n.output[0],
                n.output[1],
                n.output[2],
                n.output[3],
                "",
                n.output[5],
                "",
                "",
                n.output[7],
                n.output[8],
                n.output[6],
            ]
            rebuilt_nodes.append(
                helper.make_node(
                    "PackedSparseAttentionIndexer",
                    psai_ins,
                    psai_outs,
                    name=n.name,
                    domain="com.microsoft",
                    policy_mode="qsa",
                    compress_ratio=int(attrs.get("compress_ratio", 4)),
                    state_capacity=int(attrs.get("state_capacity", 8257)),
                    token_budget=int(attrs.get("token_budget", 2048)),
                    epsilon=float(attrs.get("epsilon", 1e-6)),
                    scale=float(attrs.get("scale", 0.0883883461356163)),
                )
            )
            stats["PackedQSAIndexer_to_Contrib"] += 1
            continue

        # 12. qwen.static::GuardedDSA -> com.microsoft::DynamicSparseAttention
        if n.domain == "qwen.static" and n.op_type == "GuardedDSA":
            attrs = get_attr_map(n)
            sel_idx_2d = dsa_raw_sel_by_reshaped.get(n.input[5], n.input[5])
            sel_cnt_1d = dsa_raw_sel_by_reshaped.get(n.input[6], n.input[6])
            dsa_ins = [
                n.input[0],
                n.input[1],
                n.input[2],
                n.input[3],
                n.input[4],
                "",
                "",
                sel_idx_2d,
                sel_cnt_1d,
                n.input[7],
                const_dsa_total_seq,
                "",
                "",
                "",
                "",
                "",
                "",
                n.input[8],
            ]
            if "k_scale" in attrs and "v_scale" in attrs:
                ks_name = ensure_const(f"{n.name}_k_scale", np.asarray(attrs["k_scale"], dtype=np.float32))
                vs_name = ensure_const(f"{n.name}_v_scale", np.asarray(attrs["v_scale"], dtype=np.float32))
                dsa_ins.extend([ks_name, vs_name])
            rebuilt_nodes.append(
                helper.make_node(
                    "DynamicSparseAttention",
                    dsa_ins,
                    list(n.output),
                    name=n.name,
                    domain="com.microsoft",
                    num_heads=int(attrs.get("num_heads", 24)),
                    kv_num_heads=int(attrs.get("kv_num_heads", 2)),
                    scale=float(attrs.get("scale", 0.0625)),
                )
            )
            stats["GuardedDSA_to_DynamicSparseAttention"] += 1
            continue

        # 13. qwen.fused_attn::SigmoidGateMul -> com.microsoft::SigmoidGateMul
        if n.domain == "qwen.fused_attn" and n.op_type == "SigmoidGateMul":
            attrs = get_attr_map(n)
            proj_in = n.input[1]
            if proj_in in q_gate_part_by_merged:
                proj_in = q_gate_part_by_merged[proj_in]
                q_off = 0
                tok_str = 12288
            else:
                q_off = int(attrs.get("q_offset", 640))
                tok_str = int(attrs.get("tok_stride", 13952))
            rebuilt_nodes.append(
                helper.make_node(
                    "SigmoidGateMul",
                    [n.input[0], proj_in],
                    list(n.output),
                    name=n.name,
                    domain="com.microsoft",
                    head_dim=int(attrs.get("head_dim", 256)),
                    num_heads=int(attrs.get("num_heads", 24)),
                    q_offset=q_off,
                    q_stride=int(attrs.get("q_stride", 512)),
                    tok_stride=tok_str,
                    sigmoid_in_fp32=1,
                )
            )
            stats["SigmoidGateMul_to_Contrib"] += 1
            continue

        # 14. ArgmaxRows / Bf16ArgMax -> com.microsoft::ArgMaxLastAxis
        if n.op_type in ("ArgmaxRows", "Bf16ArgMax"):
            keepdims = 1 if n.op_type == "Bf16ArgMax" else 0
            rebuilt_nodes.append(
                helper.make_node(
                    "ArgMaxLastAxis",
                    [n.input[0]],
                    [n.output[0]],
                    name=n.name,
                    domain="com.microsoft",
                    keepdims=keepdims,
                )
            )
            stats[f"{n.op_type}_to_ArgMaxLastAxis"] += 1
            continue

        # 15. qwen.fold::{EmbedNorm, FoldMid2, RowSelectBf16x3, FoldTail} & qwen.mtp::MtpMoe
        if n.domain == "qwen.fold" and n.op_type == "EmbedNorm":
            attrs = get_attr_map(n)
            rebuilt_nodes.append(
                helper.make_node(
                    "EmbeddingRMSNorm",
                    list(n.input),
                    list(n.output),
                    name=n.name,
                    domain="com.microsoft",
                    epsilon=float(attrs.get("epsilon", 1e-6)),
                )
            )
            stats["EmbedNorm_to_EmbeddingRMSNorm"] += 1
            continue

        if n.domain == "qwen.fold" and n.op_type == "RowSelectBf16x3":
            rebuilt_nodes.append(
                helper.make_node(
                    "RowSelect",
                    list(n.input),
                    list(n.output),
                    name=n.name,
                    domain="com.microsoft",
                )
            )
            stats["RowSelectBf16x3_to_RowSelect"] += 1
            continue

        if n.domain == "qwen.fold" and n.op_type in ("FoldMid2", "FoldTail"):
            n.domain = "com.microsoft"
            rebuilt_nodes.append(n)
            stats[f"{n.op_type}_to_Contrib"] += 1
            continue

        if n.domain == "qwen.mtp" and n.op_type == "MtpMoe":
            n.domain = "com.microsoft"
            rebuilt_nodes.append(n)
            stats["MtpMoe_to_Contrib"] += 1
            continue

        rebuilt_nodes.append(n)

    del graph.node[:]
    graph.node.extend(rebuilt_nodes)

    # Prune unused initializers
    used_inputs = {inp for n in graph.node for inp in n.input if inp} | {o.name for o in graph.output if o.name}
    inits.update(new_inits)
    kept_inits = [t for name, t in inits.items() if name in used_inputs]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    del graph.value_info[:]
    ensure_ms_opset(model, 1)

    # Verify zero custom ops remain
    remaining_custom = [(n.domain, n.op_type, n.name) for n in graph.node if n.domain not in ("", "ai.onnx", "com.microsoft")]
    if remaining_custom:
        raise RuntimeError(f"Remaining custom ops in {model_path}: {remaining_custom[:10]}")

    onnx.save_model(model, str(model_path), save_as_external_data=False)
    return dict(stats)


def write_weights_pool_flat(out_dir: Path, model_paths: dict[str, Path]) -> dict:
    inv: dict[str, dict] = {}
    counts: collections.Counter = collections.Counter()
    skip_files = {"embedding.onnx.data", "qmoe_scales.data", "matmul_scales.data"}

    for label, mpath in model_paths.items():
        m = onnx.load(str(mpath), load_external_data=False)
        sub_inv = {}
        for t in m.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if not ext or Path(ext["location"]).name in skip_files:
                continue
            p = mpath.parent / ext["location"]
            st = os.stat(p)
            off = int(ext.get("offset", 0))
            ln = int(ext.get("length", 0)) or (st.st_size - off)
            key = (st.st_ino, off, ln, t.data_type, tuple(t.dims))
            sub_inv[t.name] = {
                "key": key,
                "file": ext["location"],
                "dtype": t.data_type,
                "dims": list(t.dims),
                "offset": off,
                "length": ln,
            }
            counts[key] += 1
        inv[label] = sub_inv

    pool_ids: dict[tuple, int] = {}
    lines: list[str] = []
    sessions: list[str] = []
    dev_bytes = 0

    for label, sub_inv in inv.items():
        for name, item in sub_inv.items():
            if counts[item["key"]] < 2:
                continue
            if item["key"] not in pool_ids:
                pid = len(pool_ids)
                pool_ids[item["key"]] = pid
                dims_str = " ".join(str(d) for d in item["dims"])
                lines.append(
                    f"pool {pid} {item['dtype']} {item['offset']} {item['length']} {len(item['dims'])} "
                    f"{dims_str} {item['file']}"
                )
                dev_bytes += item["length"]
            sessions.append(f"session {label} {name} {pool_ids[item['key']]}")

    (out_dir / "weights_pool.txt").write_text("qwen38_weights_pool 1\n" + "\n".join(lines + sessions) + "\n")
    return {
        "initializers": len(pool_ids),
        "device_gib": round(dev_bytes / 2**30, 3),
        "sessions": {l: sum(1 for s in sessions if s.split()[1] == l) for l in inv},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="Source RC2 package directory")
    ap.add_argument("--out", type=Path, required=True, help="Output contrib-op RC2 package directory")
    ap.add_argument("--provider-lib", type=Path, required=True, help="Path to libonnxruntime_providers_cuda.so")
    ap.add_argument("--plane-lib", type=Path, required=True, help="Path to libqwen38_plane.so")
    ap.add_argument("--fp8-mlp-shared", action="store_true", default=False)
    ap.add_argument("--fp8-shared-down-only", action="store_true", default=False)
    ap.add_argument("--reuse-hc-bf16-from", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    src = args.src.resolve(strict=True)
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Hard-link static data files and subdirectories (excluding custom-op libs and hc_tiled.data)
    for item in sorted(src.iterdir()):
        if item.name in ("libs", "provider", "model.onnx", "model_s2.onnx", "model_s3.onnx", "model_s4.onnx", "mtp.onnx", "genai_config.json", "package_manifest.json", "weights_pool.txt", "hc_tiled.data"):
            continue
        dst_item = out / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, copy_function=os.link)
        else:
            os.link(item, dst_item)

    (out / "libs").mkdir(parents=True, exist_ok=True)
    (out / "provider").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.plane_lib, out / "libs" / "libqwen38_plane.so")
    shutil.copy2(args.provider_lib, out / "provider" / "libonnxruntime_providers_cuda.so")

    shutil.copy2(src / "model.onnx", out / "model.onnx")
    shutil.copy2(src / "mtp.onnx", out / "mtp.onnx")

    writer = ContribWeightWriter(src, out)
    hc_bf16_target = None
    hc_bf16_mtp = None
    if args.reuse_hc_bf16_from is not None:
        ref_pkg = args.reuse_hc_bf16_from.resolve(strict=True)
        writer.close()
        if (out / "contrib_weights.data").exists():
            (out / "contrib_weights.data").unlink()
        os.link(ref_pkg / "contrib_weights.data", out / "contrib_weights.data")
        m_ref_t = onnx.load(str(ref_pkg / "model.onnx"), load_external_data=False)
        m_ref_h = onnx.load(str(ref_pkg / "mtp.onnx"), load_external_data=False)
        hc_bf16_target = {t.name: t for t in m_ref_t.graph.initializer if t.name.endswith(".untiled_bf16")}
        hc_bf16_mtp = {t.name: t for t in m_ref_h.graph.initializer if t.name.endswith(".untiled_bf16")}
        target_stats = convert_model_to_contrib_ops(
            out / "model.onnx", writer, is_mtp=False, fp8_mlp_shared=args.fp8_mlp_shared, fp8_shared_down_only=args.fp8_shared_down_only, hc_bf16_inits=hc_bf16_target
        )
        mtp_stats = convert_model_to_contrib_ops(
            out / "mtp.onnx", writer, is_mtp=True, fp8_mlp_shared=args.fp8_mlp_shared, fp8_shared_down_only=args.fp8_shared_down_only, hc_bf16_inits=hc_bf16_mtp
        )
    else:
        try:
            target_stats = convert_model_to_contrib_ops(
                out / "model.onnx", writer, is_mtp=False, fp8_mlp_shared=args.fp8_mlp_shared, fp8_shared_down_only=args.fp8_shared_down_only
            )
            mtp_stats = convert_model_to_contrib_ops(
                out / "mtp.onnx", writer, is_mtp=True, fp8_mlp_shared=args.fp8_mlp_shared, fp8_shared_down_only=args.fp8_shared_down_only
            )
        finally:
            writer.close()

    pool_info = write_weights_pool_flat(out, {"target": out / "model.onnx", "mtp": out / "mtp.onnx"})

    # Update genai_config.json: remove custom_ops_libraries and offload runtime_profiles!
    cfg = json.loads((src / "genai_config.json").read_text())
    cfg.pop("runtime_profiles", None)
    cfg["model"]["decoder"]["session_options"].pop("custom_ops_libraries", None)
    if "fold" in cfg["model"].get("qwen38_graph", {}) and "mtp" in cfg["model"]["qwen38_graph"]["fold"]:
        cfg["model"]["qwen38_graph"]["fold"]["mtp"].get("session_options", {}).pop("custom_ops_libraries", None)
    (out / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    manifest = json.loads((src / "package_manifest.json").read_text()) if (src / "package_manifest.json").exists() else {}
    manifest["contrib_ops_only"] = True
    manifest["custom_ops_libraries"] = {}
    manifest["target_contrib_rewrites"] = target_stats
    manifest["mtp_contrib_rewrites"] = mtp_stats
    manifest["weights_pool"] = pool_info
    manifest["provider_sha256"] = hashlib.sha256((out / "provider" / "libonnxruntime_providers_cuda.so").read_bytes()).hexdigest()
    manifest["plane_sha256"] = hashlib.sha256((out / "libs" / "libqwen38_plane.so").read_bytes()).hexdigest()
    manifest["build_seconds"] = round(time.time() - t0, 2)
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[Done] Built contrib-ops-only RC package {out} in {time.time() - t0:.1f}s")
    print("  target rewrites:", target_stats)
    print("  mtp rewrites:   ", mtp_stats)
    print("  weights_pool:   ", pool_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
