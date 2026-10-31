#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Build the SM80/Ampere-friendly INT4 QMoE + INT8 MatMulNBits variant of Qwen3.8-Flash-Next Engine v12d:
  - Target + MTP MoE experts (48 + 1 layers): com.microsoft::QMoE (INT4 symmetric, block_size=32, weights_prepacked=1, SM80 layout)
  - Dense projections & HyperConnection down/up (402 unique weights): com.microsoft::MatMulNBits (INT8 symmetric, block_size=32, weight_prepacked=1, SM80 layout)
  - PLE embedding tables (48 layers, 128 shards): symmetric INT8 raw binary shards (ple/ple_int8_00..09.data, scale -0.000703105776)
  - CPU-mapped token embedding (embed_tokens.data in host_pool)
  - Calibrated INT8 per-channel KV cache (copied from v12d_bf16lm_engine)
  - Multi-arch custom ops (sm_80, sm_86, sm_89, sm_90/sm_90a) without Hopper-only TMA/cluster ops
  - Self-contained inference-only package (zero .safetensors files and zero unreferenced source weight bytes)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from package_paths import reject_output_overlap

ALIGN = 4096
PLE_INT8_SCALE = 0.000703105776


def align_up(val: int, align: int = ALIGN) -> int:
    return (val + align - 1) // align * align


def _preprocess_weights_for_mixed_gemm_torch(tensor: torch.Tensor, bits: int, sm: int = 80) -> torch.Tensor:
    bits_a = 16
    bits_b = 4 if bits == 4 else 8
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    permutation_map = {
        "16_8": [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        "16_4": [
            0,
            1,
            8,
            9,
            16,
            17,
            24,
            25,
            2,
            3,
            10,
            11,
            18,
            19,
            26,
            27,
            4,
            5,
            12,
            13,
            20,
            21,
            28,
            29,
            6,
            7,
            14,
            15,
            22,
            23,
            30,
            31,
        ],
    }
    mma_shape_n = 8
    b_rows_per_mma = 8 * 16 // bits_b

    num_experts, num_rows, num_cols = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    if num_rows % b_rows_per_mma != 0 or num_cols % mma_shape_n != 0:
        raise ValueError(f"Incompatible shape ({num_rows}, {num_cols}) for bits={bits_b}")

    if sm < 100:
        pmap = permutation_map[f"{bits_a}_{bits_b}"]
        row_idx = [(r // b_rows_per_mma) * b_rows_per_mma + pmap[r % b_rows_per_mma] for r in range(num_rows)]
        tensor = tensor[:, row_idx, :]

    original_shape = tensor.shape
    if bits_b == 4:
        u = tensor.view(torch.uint8)
        high = (u >> 4).permute(0, 2, 1).unsqueeze(2)
        low = ((u << 4) >> 4).permute(0, 2, 1).unsqueeze(2)
        merged = torch.cat([low, high], dim=2).reshape(u.shape[0], -1, u.shape[1])
        merged = merged[:, :, 0::2] + merged[:, :, 1::2] * 16
        tensor = merged.view(torch.int8).reshape(original_shape)
    else:
        tensor = tensor.permute(0, 2, 1).reshape(original_shape)

    interleave = bits_a // bits_b
    if interleave > 1 and sm < 90:
        rows_per_tile = 128 * 8 // bits_a
        elts_in_int32 = 32 // bits_b
        tensor = tensor.reshape(
            num_experts, -1, interleave, num_rows // rows_per_tile, rows_per_tile * 4 // elts_in_int32
        )
        tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)

    if bits_b == 8:
        t = tensor.to(torch.int16)
        t += -256 * (t > 127).to(torch.int16) + 128
        t = t.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(original_shape)
        tensor = t.to(torch.uint8).view(torch.int8)
    else:
        u = tensor.view(torch.uint8)
        high = (u >> 4).unsqueeze(-1)
        low = ((u << 4) >> 4).unsqueeze(-1)
        merged = torch.cat([low, high], dim=-1).reshape(u.shape[0], u.shape[1], -1)
        merged = merged.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(merged.shape)
        merged = merged.to(torch.int16)
        merged += -16 * (merged > 7).to(torch.int16) + 8
        merged = merged[:, :, 0::2] + merged[:, :, 1::2] * 16
        tensor = merged.to(torch.uint8).view(torch.int8)

    return tensor.contiguous()


@torch.no_grad()
def fast_quant_prepack_batch(
    weights_bf16: torch.Tensor,
    bits: int,
    block_size: int = 32,
    is_qmoe: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact SM80 CutlassFpAIntB / QMoE blockwise symmetric quantizer + prepacker on GPU.
    Matches CudaQuantizer.qmoe_prepacked_blockwise_quantize (bits=4) and
    CudaQuantizer.matmulnbits_prepacked_blockwise_quantize (bits=8) bit-for-bit.
    """
    orig_shape = weights_bf16.shape
    if weights_bf16.ndim == 2:
        weights_bf16 = weights_bf16.unsqueeze(0)
    E, N, K = weights_bf16.shape  # noqa: N806 - matrix dimension names
    assert K % block_size == 0 and K % 64 == 0, f"Invalid K={K} for block_size={block_size}"
    w = weights_bf16.to(device="cuda", dtype=torch.bfloat16)

    qmax = float((1 << (bits - 1)) - 1)
    qmin = -qmax if is_qmoe else -float(1 << (bits - 1))

    w_f32 = w.float().reshape(E, N, K // block_size, block_size)
    amax = w_f32.abs().amax(dim=-1)
    scales_bf16 = (amax / qmax).to(torch.bfloat16)
    scales_f32 = scales_bf16.float().unsqueeze(-1)
    q = torch.where(
        scales_f32 != 0,
        torch.round(w_f32 / scales_f32),
        torch.zeros_like(w_f32),
    )
    q_signed = q.clamp(qmin, qmax).to(torch.int16).reshape(E, N, K)

    # Transpose (E, N, K) -> (E, K, N) signed
    signed_t = q_signed.permute(0, 2, 1).contiguous()
    if bits == 4:
        packed_t = (
            ((signed_t[:, :, 0::2] & 0x0F) | ((signed_t[:, :, 1::2] & 0x0F) << 4)).to(torch.uint8).view(torch.int8)
        )
    else:
        packed_t = signed_t.to(torch.uint8).view(torch.int8)

    out_b = _preprocess_weights_for_mixed_gemm_torch(packed_t, bits=bits, sm=80).view(torch.uint8)
    packed_cols = N * bits // 8

    if is_qmoe:
        out_b = out_b.reshape(E, K, packed_cols)
        return out_b.cpu(), scales_bf16.cpu()
    else:
        out_b = out_b.reshape(E, N, K // 32, 32 * bits // 8)
        if len(orig_shape) == 2:
            out_b = out_b.squeeze(0)
            scales_bf16 = scales_bf16.squeeze(0)
        return out_b.cpu(), scales_bf16.cpu()


class HFCheckpointReader:
    """Fast memory-mapped reader for the safetensors shards of the HF checkpoint."""

    def __init__(self, hf_dir: Path):
        self.hf_dir = hf_dir
        idx_path = hf_dir / "model.safetensors.index.json"
        self.weight_map = json.loads(idx_path.read_text())["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}

    def _get_header(self, shard_file: str) -> tuple[int, dict]:
        if shard_file not in self._headers:
            p = self.hf_dir / shard_file
            with open(p, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            self._headers[shard_file] = (8 + hlen, hdr)
        return self._headers[shard_file]

    def read_bf16_tensor(self, name: str, device: str = "cuda") -> torch.Tensor:
        if name not in self.weight_map and name.startswith("model.layers."):
            name = "model.language_model." + name[len("model.") :]
        shard_file = self.weight_map[name]
        base_off, hdr = self._get_header(shard_file)
        info = hdr[name]
        assert info["dtype"] == "BF16", f"Expected BF16 for {name}, got {info['dtype']}"
        s, e = info["data_offsets"]
        shape = tuple(info["shape"])
        p = self.hf_dir / shard_file
        mm = np.memmap(p, dtype=np.uint16, mode="r", offset=base_off + s, shape=shape)
        t = torch.from_numpy(mm)
        if device == "cuda":
            t = t.to(device="cuda", non_blocking=False)
        else:
            t = t.clone()
        return t.view(torch.bfloat16)


def build_ple_int8(hf_reader: HFCheckpointReader, v12d_dir: Path, out_dir: Path) -> None:
    """Quantize all 128 PLE shards from BF16 to symmetric INT8 raw binary files."""
    t0 = time.time()
    ple_dir = out_dir / "ple"
    ple_dir.mkdir(parents=True, exist_ok=True)

    # Group 128 shards across 10 raw binary files (13 shards in 00..07, 12 shards in 08..09)
    shards_per_file = [13] * 8 + [12] * 2
    assert sum(shards_per_file) == 128

    orig_lines = (v12d_dir / "ple_table.txt").read_text().splitlines()
    lines = []
    for line in orig_lines:
        if line.startswith("shard "):
            break
        if line.startswith("scale "):
            lines.append(f"scale -{PLE_INT8_SCALE:.12f}")
        else:
            lines.append(line)

    inv_scale = 1.0 / PLE_INT8_SCALE
    shard = 0
    for file_idx, shard_count in enumerate(shards_per_file):
        rel = f"ple/ple_int8_{file_idx:02d}.data"
        off = 0
        with open(out_dir / rel, "wb") as f:
            for _ in range(shard_count):
                t = hf_reader.read_bf16_tensor(
                    f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                ).to(device="cuda", dtype=torch.bfloat16)  # [rows, 160]
                rows, width = t.shape
                assert width == 160, f"Unexpected PLE shard shape {t.shape}"
                q_i8 = torch.round(t.float() * inv_scale).clamp(-127.0, 127.0).to(torch.int8).cpu().numpy()
                raw_bytes = q_i8.tobytes()
                assert len(raw_bytes) == rows * width

                f.write(raw_bytes)
                lines.append(f"shard {rel} {off} {rows}")
                off += len(raw_bytes)
                shard += 1

    (out_dir / "ple_table.txt").write_text("\n".join(lines) + "\n")
    print(f"[PLE] Quantized 128 PLE shards to INT8 in {time.time() - t0:.1f}s")


def build_qmoe_int4(hf_reader: HFCheckpointReader, target_dir: Path) -> dict[str, dict]:
    """
    Quantize & SM80-prepack all 48 target MoE layers + 1 MTP MoE layer into:
      - target/qmoe_int4.data   (UINT8 packed weights)
      - target/qmoe_scales.data (BFLOAT16 scales)
    Returns metadata dict mapping canonical tensor names to file/offset/length/dtype/dims.
    """
    t0 = time.time()
    w_path = target_dir / "qmoe_int4.data"
    s_path = target_dir / "qmoe_scales.data"
    meta: dict[str, dict] = {}

    w_off = 0
    s_off = 0
    prefixes = [f"model.layers.{i}.mlp" for i in range(48)] + ["mtp.layers.0.mlp"]
    if (
        w_path.exists()
        and s_path.exists()
        and w_path.stat().st_size == 49 * (838860800 + 419430400)
        and s_path.stat().st_size == 49 * (104857600 + 52428800)
    ):
        for pfx in prefixes:
            for suffix, sz, dtype, dims, fname, is_scale in [
                ("fc1_experts_weights", 838860800, TensorProto.UINT8, [512, 2560, 640], "qmoe_int4.data", False),
                ("fc2_experts_weights", 419430400, TensorProto.UINT8, [512, 640, 1280], "qmoe_int4.data", False),
                ("fc1_scales", 104857600, TensorProto.BFLOAT16, [512, 1280, 80], "qmoe_scales.data", True),
                ("fc2_scales", 52428800, TensorProto.BFLOAT16, [512, 2560, 20], "qmoe_scales.data", True),
            ]:
                cur_off = s_off if is_scale else w_off
                meta[f"{pfx}.{suffix}"] = {
                    "location": fname,
                    "offset": cur_off,
                    "length": sz,
                    "dtype": dtype,
                    "dims": dims,
                }
                if is_scale:
                    s_off = cur_off + sz
                else:
                    w_off = cur_off + sz
        print(f"[QMoE INT4] Reusing existing {w_path} and {s_path}")
        return meta

    with open(w_path, "wb") as fw, open(s_path, "wb") as fs:
        for idx, pfx in enumerate(prefixes):
            lt0 = time.time()
            raw_gu = hf_reader.read_bf16_tensor(f"{pfx}.experts.gate_up_proj")  # [512, 1280, 2560]
            raw_dn = hf_reader.read_bf16_tensor(f"{pfx}.experts.down_proj")  # [512, 2560, 640]

            # Interleave gate (0..639) and up (640..1279) rows for swiglu_fusion=1
            half = raw_gu.shape[1] // 2
            interleaved_gu = torch.stack([raw_gu[:, :half, :], raw_gu[:, half:, :]], dim=2).reshape_as(raw_gu)

            fc1_w, fc1_s = fast_quant_prepack_batch(interleaved_gu, bits=4, block_size=32, is_qmoe=True)
            fc2_w, fc2_s = fast_quant_prepack_batch(raw_dn, bits=4, block_size=32, is_qmoe=True)

            for suffix, t_cpu, f_handle, fname, is_scale in [
                ("fc1_experts_weights", fc1_w, fw, "qmoe_int4.data", False),
                ("fc2_experts_weights", fc2_w, fw, "qmoe_int4.data", False),
                ("fc1_scales", fc1_s, fs, "qmoe_scales.data", True),
                ("fc2_scales", fc2_s, fs, "qmoe_scales.data", True),
            ]:
                cur_off = s_off if is_scale else w_off
                aligned_off = align_up(cur_off, ALIGN)
                if aligned_off > cur_off:
                    f_handle.write(b"\x00" * (aligned_off - cur_off))
                    cur_off = aligned_off
                raw = t_cpu.view(torch.uint8).numpy().tobytes()
                f_handle.write(raw)
                dtype = TensorProto.BFLOAT16 if is_scale else TensorProto.UINT8
                meta[f"{pfx}.{suffix}"] = {
                    "location": fname,
                    "offset": cur_off,
                    "length": len(raw),
                    "dtype": dtype,
                    "dims": list(t_cpu.shape),
                }
                if is_scale:
                    s_off = cur_off + len(raw)
                else:
                    w_off = cur_off + len(raw)

            if (idx + 1) % 8 == 0 or idx == len(prefixes) - 1:
                print(
                    f"[QMoE INT4] {idx + 1}/{len(prefixes)} layers quantized "
                    f"({time.time() - lt0:.2f}s/layer, elapsed {time.time() - t0:.1f}s)"
                )

    return meta


def collect_v12b_bf16_map(v12b_dir: Path) -> dict[tuple[str, str], tuple]:
    """Map (sub, init_name) in pkg_v12b_engine to (phys_key, path, offset, length, dtype, dims)."""
    v12b_map = {}
    for sub in ["target", "head_r2", "dense", "head_kv"]:
        m = onnx.load(str(v12b_dir / sub / "model.onnx"), load_external_data=False)
        for t in m.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if not ext:
                continue
            p = v12b_dir / sub / ext["location"]
            st = os.stat(p)
            off = int(ext.get("offset", 0))
            ln = int(ext.get("length", 0)) or (st.st_size - off)
            phys_key = (st.st_ino, off, ln, t.data_type, tuple(t.dims))
            v12b_map[(sub, t.name)] = (phys_key, p, off, ln, t.data_type, tuple(t.dims))
    return v12b_map


def lookup_v12b_weight(v12b_map: dict, sub: str, w_name: str) -> tuple:
    base = w_name
    for sfx in ["_fp8", ".hc_tiled_down", ".hc_tiled_up"]:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    if (sub, base) in v12b_map:
        return v12b_map[(sub, base)]
    if base.startswith("head.") and ("head_r2", base[5:]) in v12b_map:
        return v12b_map[("head_r2", base[5:])]
    for s2 in ["target", "head_r2", "dense", "head_kv"]:
        if (s2, base) in v12b_map:
            return v12b_map[(s2, base)]
    raise KeyError(f"Could not find BF16 weight in v12b for ({sub}, {w_name}) -> base={base}")


def build_matmul_int8(
    v12d_dir: Path,
    v12b_map: dict,
    target_dir: Path,
    keep_bf16_router_shared: bool = False,
) -> dict[tuple, dict]:
    """
    Quantize & SM80-prepack all unique dense projection weights [K, N] into:
      - matmul_int8.data   (UINT8 [N, K//32, 32])
      - matmul_scales.data (BFLOAT16 [N, K//32])
    Returns dict mapping v12b phys_key -> {"qweight": {...}, "scales": {...}, "K": K, "N": N}.
    """
    t0 = time.time()
    unique_weights: dict[tuple, tuple] = {}
    subs = (
        ["target", "head_r2"]
        if not (v12d_dir / "dense/model.onnx").exists()
        else ["target", "head_r2", "dense", "head_kv"]
    )
    for sub in subs:
        m = onnx.load(str(v12d_dir / sub / "model.onnx"), load_external_data=False)
        for n in m.graph.node:
            w_names = []
            if n.op_type == "SkinnyGemm":
                if keep_bf16_router_shared and len(n.output) == 3:
                    continue
                w_names = [n.input[1]]
            elif n.op_type == "SharedExpertDown":
                if keep_bf16_router_shared:
                    continue
                w_names = [n.input[2]]
            elif n.op_type in ("HcBlock", "HcPostBlock"):
                idx = 4 if n.op_type == "HcPostBlock" else 2
                w_names = [n.input[idx], n.input[idx + 2]]
            elif n.op_type in ("DequantWeightBf16", "UntileHcBf16"):
                w_names = [n.input[0]]
            for wn in w_names:
                info = lookup_v12b_weight(v12b_map, sub, wn)
                phys_key = info[0]
                if phys_key not in unique_weights:
                    unique_weights[phys_key] = info

    w_path = target_dir / "matmul_int8.data"
    s_path = target_dir / "matmul_scales.data"
    w_off = 0
    s_off = 0
    quant_map: dict[tuple, dict] = {}

    with open(w_path, "wb") as fw, open(s_path, "wb") as fs:
        for phys_key, (_, p, off, ln, dtype, dims) in unique_weights.items():
            assert dtype == TensorProto.BFLOAT16 and len(dims) == 2, f"Unexpected {dtype} {dims}"
            K, N = dims  # noqa: N806
            with open(p, "rb") as f_in:
                f_in.seek(off)
                raw = f_in.read(ln)
            w_t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).reshape(K, N)
            # MatMul(X, W_t) where W_t is [K, N] -> MatMulNBits expects W = W_t.T of shape [N, K]
            w_nk = w_t.t().contiguous()
            qb, qs = fast_quant_prepack_batch(w_nk, bits=8, block_size=32, is_qmoe=False)

            w_off_aligned = align_up(w_off, ALIGN)
            if w_off_aligned > w_off:
                fw.write(b"\x00" * (w_off_aligned - w_off))
                w_off = w_off_aligned
            qb_bytes = qb.view(torch.uint8).numpy().tobytes()
            fw.write(qb_bytes)
            qw_meta = {
                "location": "matmul_int8.data",
                "offset": w_off,
                "length": len(qb_bytes),
                "dtype": TensorProto.UINT8,
                "dims": list(qb.shape),
            }
            w_off += len(qb_bytes)

            s_off_aligned = align_up(s_off, ALIGN)
            if s_off_aligned > s_off:
                fs.write(b"\x00" * (s_off_aligned - s_off))
                s_off = s_off_aligned
            qs_bytes = qs.view(torch.uint8).numpy().tobytes()
            fs.write(qs_bytes)
            qs_meta = {
                "location": "matmul_scales.data",
                "offset": s_off,
                "length": len(qs_bytes),
                "dtype": TensorProto.BFLOAT16,
                "dims": list(qs.shape),
            }
            s_off += len(qs_bytes)

            quant_map[phys_key] = {
                "qweight": qw_meta,
                "scales": qs_meta,
                "K": K,
                "N": N,
            }

    print(
        f"[MatMulNBits INT8] Quantized & prepacked {len(quant_map)} unique weights "
        f"({w_off / 2**20:.1f} MiB weights, {s_off / 2**20:.1f} MiB scales) in {time.time() - t0:.1f}s"
    )
    return quant_map


def make_ext_tensor(name: str, meta: dict, loc_prefix: str = "") -> TensorProto:
    t = TensorProto()
    t.name = name
    t.data_type = meta["dtype"]
    t.dims.extend(meta["dims"])
    t.data_location = TensorProto.EXTERNAL
    loc = f"{loc_prefix}{meta['location']}" if loc_prefix else meta["location"]
    for k, v in [
        ("location", loc),
        ("offset", str(meta["offset"])),
        ("length", str(meta["length"])),
    ]:
        kv = t.external_data.add()
        kv.key = k
        kv.value = v
    return t


def get_attr_map(node) -> dict:
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode("utf-8")
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            out[a.name] = list(a.floats)
    return out


def remove_dead_nodes_and_inits(model: onnx.ModelProto) -> None:
    graph = model.graph
    needed = {o.name for o in graph.output if o.name}
    kept_rev = []
    for n in reversed(graph.node):
        if any(o in needed for o in n.output if o):
            kept_rev.append(n)
            for inp in n.input:
                if inp:
                    needed.add(inp)
    kept_rev.reverse()
    del graph.node[:]
    graph.node.extend(kept_rev)

    used_inits = {inp for n in graph.node for inp in n.input if inp} | {o.name for o in graph.output if o.name}
    kept_inits = [t for t in graph.initializer if t.name in used_inits]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)


def ensure_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    for op in model.opset_import:
        if op.domain == domain:
            return
    op = model.opset_import.add()
    op.domain = domain
    op.version = version


def prune_unused_opsets(model: onnx.ModelProto) -> None:
    used_domains = {n.domain for n in model.graph.node} | {""}
    kept = [op for op in model.opset_import if op.domain in used_domains]
    del model.opset_import[:]
    model.opset_import.extend(kept)


def rewrite_submodel(
    sub: str,
    v12d_dir: Path,
    v12b_map: dict,
    qmoe_meta: dict[str, dict],
    matmul_meta: dict[tuple, dict],
    keep_bf16_router_shared: bool = False,
    mtp_kv_bf16: bool = True,
) -> onnx.ModelProto:
    model = onnx.load(str(v12d_dir / sub / "model.onnx"), load_external_data=False)
    graph = model.graph
    loc_prefix = ""

    if mtp_kv_bf16:
        for vi in list(graph.input) + list(graph.output):
            if not vi.name.endswith((".key", ".value")):
                continue
            if sub == "head_r2" or vi.name.startswith(("head.past_key_values.", "head.present.")):
                vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
        for n in graph.node:
            if n.op_type == "GuardedDSA" and (sub == "head_r2" or n.input[3].startswith("head.")):
                keep_a = [a for a in n.attribute if a.name not in ("k_scale", "v_scale")]
                del n.attribute[:]
                n.attribute.extend(keep_a)

    # Pre-scan consumer counts and SharedExpertDown pairings
    all_consumed = {o.name for o in graph.output if o.name}
    for n in graph.node:
        for inp in n.input:
            if inp:
                all_consumed.add(inp)

    # Map (shared_gate_out, shared_up_out) -> SharedExpertDown node
    shared_down_by_pair: dict[tuple[str, str], onnx.NodeProto] = {}
    for n in graph.node:
        if n.op_type == "SharedExpertDown":
            shared_down_by_pair[(n.input[0], n.input[1])] = n

    # Map Reshape_175 output -> router_2d output from SplitRouterSwishMul so QMoE can bypass the 4-node Reshape chain
    router_2d_by_gate_out: dict[str, str] = {}
    new_inits: dict[str, TensorProto] = {}

    def add_matmul_nbits_inits(base_name: str, phys_key: tuple) -> tuple[str, str, int, int]:
        m_info = matmul_meta[phys_key]
        qw_name = f"{base_name}.qweight"
        qs_name = f"{base_name}.scales"
        if qw_name not in new_inits:
            new_inits[qw_name] = make_ext_tensor(qw_name, m_info["qweight"], loc_prefix)
        if qs_name not in new_inits:
            new_inits[qs_name] = make_ext_tensor(qs_name, m_info["scales"], loc_prefix)
        return qw_name, qs_name, m_info["K"], m_info["N"]

    def add_qmoe_inits(layer_pfx: str, init_pfx: str) -> tuple[str, str, str, str]:
        names = []
        for sfx in ["fc1_experts_weights", "fc1_scales", "fc2_experts_weights", "fc2_scales"]:
            tname = f"{init_pfx}.{sfx}"
            if tname not in new_inits:
                new_inits[tname] = make_ext_tensor(tname, qmoe_meta[f"{layer_pfx}.{sfx}"], loc_prefix)
            names.append(tname)
        return tuple(names)  # type: ignore[return-value]

    # Pre-scan Reshape nodes that reshape SkinnyGemm router output so QMoE can directly consume router_2d
    producer_by_out = {o: n for n in graph.node for o in n.output if o}

    new_nodes: list[onnx.NodeProto] = []
    for n in graph.node:
        # 1. DequantWeightBf16 / UntileHcBf16 in dense and head_kv -> skipped (handled at consuming MatMul)
        if n.op_type in ("DequantWeightBf16", "UntileHcBf16"):
            continue

        # 2. MatMul in dense and head_kv fed by DequantWeightBf16 or UntileHcBf16
        if n.op_type == "MatMul" and n.input[1] in producer_by_out:
            prod = producer_by_out[n.input[1]]
            if prod.op_type in ("DequantWeightBf16", "UntileHcBf16"):
                w_orig_name = prod.input[0]
                info = lookup_v12b_weight(v12b_map, sub, w_orig_name)
                qw_name, qs_name, K, N = add_matmul_nbits_inits(w_orig_name, info[0])  # noqa: N806
                new_nodes.append(
                    helper.make_node(
                        "MatMulNBits",
                        [n.input[0], qw_name, qs_name],
                        [n.output[0]],
                        name=f"{n.name}/MatMulNBits",
                        domain="com.microsoft",
                        K=K,
                        N=N,
                        bits=8,
                        block_size=32,
                        accuracy_level=0,
                        weight_prepacked=1,
                    )
                )
                continue

        # 3. SkinnyGemm in target and head_r2
        if n.op_type == "SkinnyGemm":
            w_name = n.input[1]
            info = lookup_v12b_weight(v12b_map, sub, w_name)
            attrs = get_attr_map(n)
            widths = attrs.get("widths", [info[5][1]])

            if len(n.output) == 3 and keep_bf16_router_shared:
                assert widths == [512, 640, 640], f"Unexpected 3-way SkinnyGemm widths: {widths}"
                merged_out = f"{n.name}/merged_1792"
                router_2d = n.output[0]
                swish_out = f"{n.name}/swish_640"
                router_2d_by_gate_out[n.output[0]] = swish_out
                _, p_src, off_src, ln_src, dt_src, dims_src = info
                new_inits[w_name] = make_ext_tensor(
                    w_name,
                    {
                        "location": str(p_src),
                        "offset": off_src,
                        "length": ln_src,
                        "dtype": dt_src,
                        "dims": list(dims_src),
                    },
                )
                new_nodes.append(
                    helper.make_node(
                        "MatMul",
                        [n.input[0], w_name],
                        [merged_out],
                        name=f"{n.name}/MatMulBf16",
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "SplitRouterSwishMul",
                        [merged_out],
                        [router_2d, swish_out],
                        name=f"{n.name}/SplitRouterSwishMul",
                        domain="qwen.fused",
                        router_cols=512,
                        ffn_cols=640,
                    )
                )
                continue

            qw_name, qs_name, K, N = add_matmul_nbits_inits(w_name, info[0])  # noqa: N806

            if len(n.output) == 1:
                new_nodes.append(
                    helper.make_node(
                        "MatMulNBits",
                        [n.input[0], qw_name, qs_name],
                        [n.output[0]],
                        name=f"{n.name}/MatMulNBits",
                        domain="com.microsoft",
                        K=K,
                        N=N,
                        bits=8,
                        block_size=32,
                        accuracy_level=0,
                        weight_prepacked=1,
                    )
                )
            elif len(n.output) == 2:
                # PLE key_proj + value_proj [10240, 2560]
                merged_out = f"{n.name}/merged_out"
                split_t_name = f"{n.name}/split_lengths"
                new_inits[split_t_name] = helper.make_tensor(
                    split_t_name, TensorProto.INT64, [len(widths)], np.array(widths, dtype=np.int64)
                )
                new_nodes.append(
                    helper.make_node(
                        "MatMulNBits",
                        [n.input[0], qw_name, qs_name],
                        [merged_out],
                        name=f"{n.name}/MatMulNBits",
                        domain="com.microsoft",
                        K=K,
                        N=N,
                        bits=8,
                        block_size=32,
                        accuracy_level=0,
                        weight_prepacked=1,
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "Split",
                        [merged_out, split_t_name],
                        list(n.output),
                        name=f"{n.name}/Split",
                        axis=-1,
                    )
                )
            elif len(n.output) == 3:
                # Router + shared_expert.gate + shared_expert.up [512, 640, 640]
                assert widths == [512, 640, 640], f"Unexpected 3-way SkinnyGemm widths: {widths}"
                merged_out = f"{n.name}/merged_1792"
                router_2d = n.output[0]
                swish_out = f"{n.name}/swish_640"
                router_2d_by_gate_out[n.output[0]] = swish_out
                new_nodes.append(
                    helper.make_node(
                        "MatMulNBits",
                        [n.input[0], qw_name, qs_name],
                        [merged_out],
                        name=f"{n.name}/MatMulNBits",
                        domain="com.microsoft",
                        K=K,
                        N=N,
                        bits=8,
                        block_size=32,
                        accuracy_level=0,
                        weight_prepacked=1,
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "SplitRouterSwishMul",
                        [merged_out],
                        [router_2d, swish_out],
                        name=f"{n.name}/SplitRouterSwishMul",
                        domain="qwen.fused",
                        router_cols=512,
                        ffn_cols=640,
                    )
                )
            else:
                raise ValueError(f"Unsupported SkinnyGemm outputs: {n.output}")
            continue

        # 4. SharedExpertDown in target and head_r2
        if n.op_type == "SharedExpertDown":
            # inputs: [gate, up, w_down, gx, gw, routed] -> [y]
            gate_out, up_out, w_down_name, gx, gw, routed = n.input
            # Find swish_out from the preceding 3-way SkinnyGemm
            skinny_node = producer_by_out[gate_out]
            swish_out = f"{skinny_node.name}/swish_640"
            info = lookup_v12b_weight(v12b_map, sub, w_down_name)
            shared_down_out = f"{n.name}/shared_down_out"
            if keep_bf16_router_shared:
                _, p_src, off_src, ln_src, dt_src, dims_src = info
                new_inits[w_down_name] = make_ext_tensor(
                    w_down_name,
                    {
                        "location": str(p_src),
                        "offset": off_src,
                        "length": ln_src,
                        "dtype": dt_src,
                        "dims": list(dims_src),
                    },
                )
                new_nodes.append(
                    helper.make_node(
                        "MatMul",
                        [swish_out, w_down_name],
                        [shared_down_out],
                        name=f"{n.name}/MatMulBf16",
                    )
                )
            else:
                qw_name, qs_name, K, N = add_matmul_nbits_inits(w_down_name, info[0])  # noqa: N806
                new_nodes.append(
                    helper.make_node(
                        "MatMulNBits",
                        [swish_out, qw_name, qs_name],
                        [shared_down_out],
                        name=f"{n.name}/MatMulNBits",
                        domain="com.microsoft",
                        K=K,
                        N=N,
                        bits=8,
                        block_size=32,
                        accuracy_level=0,
                        weight_prepacked=1,
                    )
                )
            new_nodes.append(
                helper.make_node(
                    "SharedGateMulAdd",
                    [gx, gw, shared_down_out, routed],
                    [n.output[0]],
                    name=f"{n.name}/SharedGateMulAdd",
                    domain="qwen.fused",
                )
            )
            continue

        # 5. QMoE (NVFP4 -> INT4) in target and dense
        if n.op_type == "QMoE":
            # Extract layer prefix, e.g. model.layers.0.mlp
            fc1_w_old = n.input[2]
            layer_pfx = fc1_w_old.rsplit(".", 1)[0]
            fc1_w, fc1_s, fc2_w, fc2_s = add_qmoe_inits(layer_pfx, layer_pfx)
            router_in = n.input[1]
            # If router_in comes from Reshape(gate_out) where gate_out came from SplitRouterSwishMul (already 2D), bypass Reshape!
            if router_in in producer_by_out and producer_by_out[router_in].op_type == "Reshape":
                reshape_in = producer_by_out[router_in].input[0]
                if reshape_in in router_2d_by_gate_out:
                    router_in = reshape_in
            new_nodes.append(
                helper.make_node(
                    "QMoE",
                    [n.input[0], router_in, fc1_w, fc1_s, "", fc2_w, fc2_s],
                    [n.output[0]],
                    name=n.name,
                    domain="com.microsoft",
                    activation_type="swiglu",
                    normalize_routing_weights=1,
                    k=10,
                    expert_weight_bits=4,
                    block_size=32,
                    swiglu_fusion=1,
                    weights_prepacked=1,
                )
            )
            continue

        # 6. MtpMoe (NVFP4 -> INT4 QMoE) in target and head_r2
        if n.op_type == "MtpMoe":
            init_pfx = "head.mtp.layers.0.mlp" if n.name.startswith("head.") else "mtp.layers.0.mlp"
            fc1_w, fc1_s, fc2_w, fc2_s = add_qmoe_inits("mtp.layers.0.mlp", init_pfx)
            new_nodes.append(
                helper.make_node(
                    "QMoE",
                    [n.input[0], n.input[1], fc1_w, fc1_s, "", fc2_w, fc2_s],
                    [n.output[0]],
                    name=f"{n.name}/QMoE",
                    domain="com.microsoft",
                    activation_type="swiglu",
                    normalize_routing_weights=1,
                    k=10,
                    expert_weight_bits=4,
                    block_size=32,
                    swiglu_fusion=1,
                    weights_prepacked=1,
                )
            )
            continue

        # 7. HcBlock / HcPostBlock -> standard ORT contrib ops + MatMulNBits + qwen.fused::HcInjectGate
        if n.op_type in ("HcBlock", "HcPostBlock"):
            attrs = get_attr_map(n)
            eps = float(attrs.get("epsilon", 1e-6))
            silu_alpha = float(attrs.get("silu_alpha", 0.25))
            red_scale = float(attrs.get("reduction_scale", 0.25))
            gate_div = float(attrs.get("gate_div", 4.0))
            gate_mul = float(attrs.get("gate_mul", 2.0))

            if n.op_type == "HcPostBlock":
                streams, branch, gate_in, norm_w, w_down_tiled, w_inject, w_up_tiled = n.input[:7]
                out_post, out_pre, out_gate = n.output[:3]
                xs = out_post if out_post else f"{n.name}/post_mix"
                new_nodes.append(
                    helper.make_node(
                        "HyperConnectionPostMix",
                        [streams, branch, gate_in],
                        [xs],
                        name=f"{n.name}/HyperConnectionPostMix",
                        domain="com.microsoft",
                        num_branches=4,
                    )
                )
            else:
                streams, norm_w, w_down_tiled, w_inject, w_up_tiled = n.input[:5]
                out_pre, out_gate = n.output[:2]
                xs = streams

            hn = f"{n.name}/hn"
            down_out = f"{n.name}/down_out"
            silu_out = f"{n.name}/silu_out"
            up_out = f"{n.name}/up_out"
            sig_out = f"{n.name}/sig_out"

            down_info = lookup_v12b_weight(v12b_map, sub, w_down_tiled)
            up_info = lookup_v12b_weight(v12b_map, sub, w_up_tiled)
            down_qw, down_qs, dK, dN = add_matmul_nbits_inits(w_down_tiled, down_info[0])  # noqa: N806
            up_qw, up_qs, uK, uN = add_matmul_nbits_inits(w_up_tiled, up_info[0])  # noqa: N806

            new_nodes.append(
                helper.make_node(
                    "BranchwiseRMSNorm",
                    [xs, norm_w],
                    [hn],
                    name=f"{n.name}/BranchwiseRMSNorm",
                    domain="com.microsoft",
                    epsilon=eps,
                    num_branches=4,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [hn, down_qw, down_qs],
                    [down_out],
                    name=f"{n.name}/DownMatMulNBits",
                    domain="com.microsoft",
                    K=dK,
                    N=dN,
                    bits=8,
                    block_size=32,
                    accuracy_level=0,
                    weight_prepacked=1,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "ScaledSiLU",
                    [down_out],
                    [silu_out],
                    name=f"{n.name}/ScaledSiLU",
                    domain="com.microsoft",
                    alpha=silu_alpha,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [silu_out, up_qw, up_qs],
                    [up_out],
                    name=f"{n.name}/UpMatMulNBits",
                    domain="com.microsoft",
                    K=uK,
                    N=uN,
                    bits=8,
                    block_size=32,
                    accuracy_level=0,
                    weight_prepacked=1,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Sigmoid",
                    [up_out],
                    [sig_out],
                    name=f"{n.name}/Sigmoid",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "HyperConnectionPreMix",
                    [hn, sig_out],
                    [out_pre],
                    name=f"{n.name}/HyperConnectionPreMix",
                    domain="com.microsoft",
                    num_branches=4,
                    reduction_scale=red_scale,
                )
            )
            if out_gate and out_gate in all_consumed:
                new_nodes.append(
                    helper.make_node(
                        "HcInjectGate",
                        [hn, w_inject],
                        [out_gate],
                        name=f"{n.name}/HcInjectGate",
                        domain="qwen.fused",
                        div=gate_div,
                        mul=gate_mul,
                    )
                )
            continue

        # 8. ArgmaxRows (qwen.fused_gemm -> qwen.fused)
        if n.op_type == "ArgmaxRows":
            new_nodes.append(
                helper.make_node(
                    "ArgmaxRows",
                    list(n.input),
                    [n.output[0]],
                    name=n.name,
                    domain="qwen.fused",
                )
            )
            continue

        new_nodes.append(n)

    del graph.node[:]
    graph.node.extend(new_nodes)

    existing_init_names = {t.name for t in graph.initializer}
    for name, t in new_inits.items():
        if name in existing_init_names:
            for i, old_t in enumerate(graph.initializer):
                if old_t.name == name:
                    graph.initializer[i].CopyFrom(t)
                    break
        else:
            graph.initializer.append(t)

    remove_dead_nodes_and_inits(model)
    ensure_opset(model, "com.microsoft", 1)
    if any(n.domain == "qwen.fused" for n in graph.node):
        ensure_opset(model, "qwen.fused", 1)
    prune_unused_opsets(model)
    return model


def unify_and_pack_flat_int4_int8_package(
    models: dict[str, onnx.ModelProto],
    v12d_dir: Path,
    out_dir: Path,
    reuse_embed_from: Path,
) -> dict:
    """
    Unify target -> model.onnx and head_r2 -> mtp.onnx for arbitrary
    sequence_length (serving both prefill and decode), wire INT8 CPU embedding.onnx + embedding.onnx.data,
    pack all remaining BF16 initializers (norms, RoPE, router, shared_expert) into norm_and_rope.data
    in the root directory, and write weights_pool.txt.
    """

    def set_ext(tensor: TensorProto, location: str, offset: int, length: int) -> None:
        tensor.ClearField("raw_data")
        del tensor.external_data[:]
        tensor.data_location = TensorProto.EXTERNAL
        for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
            e = tensor.external_data.add()
            e.key, e.value = k, v

    # 1. Link embedding.onnx and embedding.onnx.data from reuse_embed_from
    for fname in ("embedding.onnx", "embedding.onnx.data", "attn_consts.data"):
        src_f = reuse_embed_from / fname
        dst_f = out_dir / fname
        if dst_f.exists() or dst_f.is_symlink():
            dst_f.unlink()
        os.link(src_f, dst_f)

    embed_names = {"model.embed_tokens.weight", "head.mtp.embed_tokens.weight", "mtp.embed_tokens.weight"}
    embed_i8_len = 248320 * 2560
    for sname in ("target", "head_r2"):
        for t in models[sname].graph.initializer:
            if t.name in embed_names:
                t.data_type = TensorProto.INT8
                set_ext(t, "embedding.onnx.data", 0, embed_i8_len)

    # 2. Replace Gather(model.embed_tokens.weight, input_ids) in target with inputs_embeds
    t_graph = models["target"].graph
    gather_candidates = [
        n
        for n in t_graph.node
        if n.op_type == "Gather" and len(n.input) >= 2 and n.input[0] == "model.embed_tokens.weight"
    ]
    if len(gather_candidates) == 1:
        gnode = gather_candidates[0]
        old_out = gnode.output[0]
        t_graph.node.remove(gnode)
        for consumer in t_graph.node:
            for i, inp in enumerate(consumer.input):
                if inp == old_out:
                    consumer.input[i] = "inputs_embeds"
        kept_inits = [t for t in t_graph.initializer if t.name != "model.embed_tokens.weight"]
        del t_graph.initializer[:]
        t_graph.initializer.extend(kept_inits)
        if not any(vi.name == "inputs_embeds" for vi in t_graph.input):
            t_graph.input.append(
                helper.make_tensor_value_info(
                    "inputs_embeds", TensorProto.BFLOAT16, ["batch_size", "sequence_length", 2560]
                )
            )

    # 3. Unify target graph (Slice_last2 before final hyper_connection_mixer + FoldMid2 head_rows=2 + bypass logits_indices / 3D M-RoPE / mask Where_85)
    for idx_n, n in enumerate(list(t_graph.node)):
        if (
            n.op_type == "Gather"
            and len(n.input) >= 2
            and n.input[1] == "logits_indices"
            and idx_n + 1 < len(t_graph.node)
            and t_graph.node[idx_n + 1].input
            and t_graph.node[idx_n + 1].input[0] == n.output[0]
        ):
            t_graph.node[idx_n + 1].input[0] = n.input[0]
            t_graph.node.remove(n)
            break

    rot_nodes = [n for n in t_graph.node if n.name.startswith("model/rotary_emb/")]
    if (
        len(rot_nodes) == 23
        and any(n.output and n.output[0] == "v_model.rotary_emb.Where_30" for n in rot_nodes)
        and any(n.output and n.output[0] == "v_model.rotary_emb.Where_32" for n in rot_nodes)
    ):
        rot_set = {id(n) for n in rot_nodes}
        first_rot = rot_nodes[0]
        g_cos = helper.make_node(
            "Gather",
            ["model.rotary_emb.cos_cache", "position_ids"],
            ["v_model.rotary_emb.Where_30"],
            name="model/rotary_emb/Gather_cos_1d",
            axis=0,
        )
        g_sin = helper.make_node(
            "Gather",
            ["model.rotary_emb.sin_cache", "position_ids"],
            ["v_model.rotary_emb.Where_32"],
            name="model/rotary_emb/Gather_sin_1d",
            axis=0,
        )
        rebuilt_rot = []
        for n in t_graph.node:
            if n is first_rot:
                rebuilt_rot.extend([g_cos, g_sin])
            elif id(n) not in rot_set:
                rebuilt_rot.append(n)
        del t_graph.node[:]
        t_graph.node.extend(rebuilt_rot)

    mask_chain_names = {
        "Constant_node_3",
        "model/Shape_node_62",
        "model/Shape_node_63",
        "model/Sub_node_64",
        "model/Constant_node_77",
        "model/Slice_node_78",
        "model/Constant_node_79",
        "model/Equal_node_80",
        "model/Not_node_81",
        "model/Shape_node_83",
        "model/Expand_node_84",
        "model/Where_node_85",
    }
    if any(n.name == "model/Where_node_85" and list(n.output) == ["v_model.Where_85"] for n in t_graph.node):
        rebuilt_mask = []
        for n in t_graph.node:
            if n.name in mask_chain_names:
                continue
            for i_idx, inp in enumerate(n.input):
                if inp == "v_model.Where_85":
                    n.input[i_idx] = "input_ids"
            rebuilt_mask.append(n)
        del t_graph.node[:]
        t_graph.node.extend(rebuilt_mask)

    t_nodes = []
    has_slice_last2 = any(n.name == "model/hyper_connection_mixer/Slice_last2" for n in t_graph.node)
    for n in t_graph.node:
        if (
            n.op_type in ("BranchwiseRMSNorm", "HcBlock")
            and (
                n.name.startswith("model/hyper_connection_mixer/")
                or (len(n.input) >= 1 and n.input[0] == "hidden_states" and "hyper_connection_mixer" in n.name)
            )
            and not has_slice_last2
        ):
            t_nodes.append(
                helper.make_node(
                    "Slice",
                    ["hidden_states", "const_1d_0", "const_1d_1", "const_1d_3"],
                    ["hidden_states.last2"],
                    name="model/hyper_connection_mixer/Slice_last2",
                )
            )
            n.input[0] = "hidden_states.last2"
            has_slice_last2 = True
        elif n.op_type == "FoldMid2":
            if len(n.input) == 10:
                n.input.append("hidden_states")
            if len(n.output) == 7:
                n.output.append("fold.head_hidden")
            keep_a = [a for a in n.attribute if a.name != "head_rows"]
            del n.attribute[:]
            node_attrs = [*keep_a, helper.make_attribute("head_rows", 2)]
            n.attribute.extend(node_attrs)
        elif n.name == "head.mtp/pre_fc_norm_hidden/BranchwiseRMSNorm_node_15":
            n.input[0] = "fold.head_hidden"
        t_nodes.append(n)
    del t_graph.node[:]
    t_graph.node.extend(t_nodes)

    # 4. Unify head_r2 graph (EmbedNorm dynamic outputs + Slice [-2:] before o_proj and mlp_hyper_connection)
    h_graph = models["head_r2"].graph
    h_init_names = {t.name for t in h_graph.initializer}
    for iname, val in (
        ("slice_last2_starts", np.array([-2], dtype=np.int64)),
        ("slice_last2_ends", np.array([9223372036854775807], dtype=np.int64)),
        ("slice_last2_axes", np.array([1], dtype=np.int64)),
    ):
        if iname not in h_init_names:
            h_graph.initializer.append(numpy_helper.from_array(val, iname))
    h_nodes = []
    for n in h_graph.node:
        if n.name in (
            "capture/Cast_slot_ids",
            "capture/Add_slot_ids",
            "capture/Add_seqlens_k",
            "mtp/hyper_connection_mixer/Slice_last2",
            "mtp/Slice_last2_mul200",
            "mtp/Slice_last2_r23",
            "mtp/Slice_last2_mul117",
        ):
            continue
        if n.op_type == "EmbedNorm":
            if len(n.input) == 3:
                n.input.append("committed_length")
            if len(n.output) == 1:
                n.output.extend(["capture.slot_ids", "capture.seqlens_k", "capture.cumulative_sequence_lengths"])
        elif n.name in ("skinny_gemm/3.0", "skinny_gemm/3.0/MatMulNBits"):
            h_nodes.append(
                helper.make_node(
                    "Slice",
                    ["v_mtp.layers.0.self_attn.Mul_200", "slice_last2_starts", "slice_last2_ends", "slice_last2_axes"],
                    ["v_mtp.layers.0.self_attn.Mul_200.last2"],
                    name="mtp/Slice_last2_mul200",
                )
            )
            h_nodes.append(
                helper.make_node(
                    "Slice",
                    ["v_mtp.Reshape_23", "slice_last2_starts", "slice_last2_ends", "slice_last2_axes"],
                    ["v_mtp.Reshape_23.last2"],
                    name="mtp/Slice_last2_r23",
                )
            )
            h_nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        "v_mtp.layers.0.attn_hyper_connection.Mul_117",
                        "slice_last2_starts",
                        "slice_last2_ends",
                        "slice_last2_axes",
                    ],
                    ["v_mtp.layers.0.attn_hyper_connection.Mul_117.last2"],
                    name="mtp/Slice_last2_mul117",
                )
            )
            n.input[0] = "v_mtp.layers.0.self_attn.Mul_200.last2"
        elif n.name in (
            "mtp/layers.0/mlp_hyper_connection/HcPostBlock",
            "mtp/layers.0/mlp_hyper_connection/HcPostBlock/HyperConnectionPostMix",
        ):
            n.input[0] = "v_mtp.Reshape_23.last2"
            n.input[2] = "v_mtp.layers.0.attn_hyper_connection.Mul_117.last2"
        h_nodes.append(n)
    del h_graph.node[:]
    h_graph.node.extend(h_nodes)
    kept_h_inits = [
        t
        for t in h_graph.initializer
        if t.name not in ("capture.slot_offsets", "capture.rows_minus_one", "capture.cumulative_sequence_lengths")
    ]
    del h_graph.initializer[:]
    h_graph.initializer.extend(kept_h_inits)

    seq_io_names = {
        "input_ids",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "hidden_states",
        "hidden_states_out",
        "logits",
        "next_tokens",
        "next_token",
    }
    for sname in ("target", "head_r2"):
        g = models[sname].graph
        kept_ins = [vi for vi in g.input if vi.name != "logits_indices"]
        del g.input[:]
        g.input.extend(kept_ins)
        for vi in list(g.input) + list(g.output):
            if vi.name in seq_io_names or vi.name.startswith("ple_embeddings."):
                dims = vi.type.tensor_type.shape.dim
                if len(dims) >= 2:
                    dims[1].ClearField("dim_value")
                    dims[1].dim_param = "sequence_length"
        del g.value_info[:]

    # 5. Pack remaining BF16 initializers into norm_and_rope.data in root directory
    kept_root_files = {
        "embedding.onnx.data",
        "qmoe_int4.data",
        "qmoe_scales.data",
        "matmul_int8.data",
        "matmul_scales.data",
        "attn_consts.data",
    }
    norm_rope_path = out_dir / "norm_and_rope.data"
    norm_rope_pos = 0
    norm_rope_map: dict[tuple[int, int, int], int] = {}
    with open(norm_rope_path, "wb") as nrf:
        for sname in ("target", "head_r2"):
            for t in models[sname].graph.initializer:
                ext = {e.key: e.value for e in t.external_data}
                if not ext:
                    continue
                loc = ext["location"]
                if loc in ("attn_consts.data", "head_data/attn_consts.data"):
                    set_ext(t, "attn_consts.data", int(ext.get("offset", 0)), int(ext["length"]))
                    continue
                if Path(loc).name in kept_root_files:
                    set_ext(t, Path(loc).name, int(ext.get("offset", 0)), int(ext["length"]))
                    continue
                src_p = Path(loc) if Path(loc).is_absolute() else (v12d_dir / sname / loc)
                if not src_p.exists():
                    src_p = v12d_dir / "target" / loc
                st = os.stat(src_p)
                off = int(ext.get("offset", 0))
                ln = int(ext.get("length", 0)) or (st.st_size - off)
                pkey = (st.st_ino, off, ln)
                if pkey not in norm_rope_map:
                    pad = (-norm_rope_pos) % ALIGN
                    if pad:
                        nrf.write(b"\x00" * pad)
                        norm_rope_pos += pad
                    with open(src_p, "rb") as fin:
                        fin.seek(off)
                        raw = fin.read(ln)
                    nrf.write(raw)
                    norm_rope_map[pkey] = norm_rope_pos
                    norm_rope_pos += ln
                set_ext(t, "norm_and_rope.data", norm_rope_map[pkey], ln)

    onnx.save_model(models["target"], str(out_dir / "model.onnx"), save_as_external_data=False)
    onnx.save_model(models["head_r2"], str(out_dir / "mtp.onnx"), save_as_external_data=False)

    return write_weights_pool(
        out_dir,
        {"target": out_dir / "model.onnx", "mtp": out_dir / "mtp.onnx"},
    )


def consolidate_remaining_external_initializers(
    models: dict[str, onnx.ModelProto],
    v12d_dir: Path,
    target_dir: Path,
) -> None:
    """
    Consolidate all remaining external initializers (not already in qmoe_*.data or matmul_*.data)
    into self-contained files in target/:
      - embed_tokens.data (model.embed_tokens.weight [248320, 2560] BF16)
      - lm_head.data      (lm_head.weight_t [2560, 248320] BF16)
      - small_weights.data (all remaining conv1d, norm, rotary_emb, block_inject_weight, shared_expert_gate)
    Deduplicates by physical source (inode, offset, length) so shared tensors have identical offsets.
    """
    new_files = {"qmoe_int4.data", "qmoe_scales.data", "matmul_int8.data", "matmul_scales.data"}
    dedup_map: dict[tuple[int, int, int], tuple[str, int, int]] = {}

    f_embed = open(target_dir / "embed_tokens.data", "wb")  # noqa: SIM115 - closed in the finally block below
    f_lm = open(target_dir / "lm_head.data", "wb")  # noqa: SIM115
    f_small = open(target_dir / "small_weights.data", "wb")  # noqa: SIM115
    offsets = {"embed_tokens.data": 0, "lm_head.data": 0, "small_weights.data": 0}
    handles = {"embed_tokens.data": f_embed, "lm_head.data": f_lm, "small_weights.data": f_small}

    try:
        for sub, model in models.items():
            for t in model.graph.initializer:
                ext = {e.key: e.value for e in t.external_data}
                if not ext:
                    continue
                loc = ext["location"]
                if Path(loc).name in new_files:
                    continue

                src_path = v12d_dir / sub / loc
                st = os.stat(src_path)
                off = int(ext.get("offset", 0))
                ln = int(ext.get("length", 0)) or (st.st_size - off)
                phys_key = (st.st_ino, off, ln)

                if phys_key not in dedup_map:
                    with open(src_path, "rb") as f_in:
                        f_in.seek(off)
                        raw = f_in.read(ln)
                    assert len(raw) == ln, f"Short read from {src_path}"

                    dims = tuple(t.dims)
                    if dims == (248320, 2560) and t.data_type == TensorProto.BFLOAT16:
                        dest_file = "embed_tokens.data"
                    elif dims == (2560, 248320) and t.data_type == TensorProto.BFLOAT16:
                        dest_file = "lm_head.data"
                    else:
                        dest_file = "small_weights.data"

                    cur_off = offsets[dest_file]
                    aligned_off = align_up(cur_off, ALIGN)
                    if aligned_off > cur_off:
                        handles[dest_file].write(b"\x00" * (aligned_off - cur_off))
                        cur_off = aligned_off
                    handles[dest_file].write(raw)
                    offsets[dest_file] = cur_off + ln
                    dedup_map[phys_key] = (dest_file, cur_off, ln)

                dest_file, new_off, new_ln = dedup_map[phys_key]
                del t.external_data[:]
                for k, v in [
                    ("location", dest_file),
                    ("offset", str(new_off)),
                    ("length", str(new_ln)),
                ]:
                    kv = t.external_data.add()
                    kv.key = k
                    kv.value = v
    finally:
        f_embed.close()
        f_lm.close()
        f_small.close()

    print(
        f"[Consolidate] embed_tokens.data={offsets['embed_tokens.data'] / 2**20:.2f} MiB, "
        f"lm_head.data={offsets['lm_head.data'] / 2**20:.2f} MiB, "
        f"small_weights.data={offsets['small_weights.data'] / 2**20:.2f} MiB"
    )


def write_weights_pool(out_dir: Path, model_paths: dict[str, Path]) -> dict:
    """
    Build weights_pool.txt for initializers shared >= 2 times across sessions.
    Places embed_tokens.data, qmoe_scales.data, and matmul_scales.data in host_pool:
      - embed_tokens.data is read by CPU Gather and GPU EmbedLookup
      - qmoe_scales.data and matmul_scales.data are transposed once to GPU during session PrePack
        and deduplicated across all 4 sessions via our PrePack scale cache, saving 7.40 GiB of duplicate GPU memory!
    """
    inv: dict[str, dict] = {}
    counts: collections.Counter = collections.Counter()

    for label, mpath in model_paths.items():
        m = onnx.load(str(mpath), load_external_data=False)
        sub_inv = {}
        for t in m.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if not ext or Path(ext["location"]).name in (
                "embedding.onnx.data",
                "qmoe_scales.data",
                "matmul_scales.data",
            ):
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

    host_files = {"embed_tokens.data"}
    pool_ids: dict[tuple, int] = {}
    lines: list[str] = []
    sessions: list[str] = []
    dev_bytes = 0
    host_bytes = 0

    for label, sub_inv in inv.items():
        rel_dir = model_paths[label].parent.relative_to(out_dir)
        for name, item in sub_inv.items():
            if counts[item["key"]] < 2:
                continue
            if item["key"] not in pool_ids:
                pid = len(pool_ids)
                pool_ids[item["key"]] = pid
                dims_str = " ".join(str(d) for d in item["dims"])
                fname = Path(item["file"]).name
                is_host = fname in host_files
                kw = "host_pool" if is_host else "pool"
                lines.append(
                    f"{kw} {pid} {item['dtype']} {item['offset']} {item['length']} {len(item['dims'])} "
                    f"{dims_str} {rel_dir / item['file']}"
                )
                if is_host:
                    host_bytes += item["length"]
                else:
                    dev_bytes += item["length"]
            sessions.append(f"session {label} {name} {pool_ids[item['key']]}")

    (out_dir / "weights_pool.txt").write_text("qwen38_weights_pool 1\n" + "\n".join(lines + sessions) + "\n")
    info = {
        "initializers": len(pool_ids),
        "device_gib": dev_bytes / 2**30,
        "host_gib": host_bytes / 2**30,
        "sessions": {label: sum(1 for s in sessions if s.split()[1] == label) for label in inv},
    }
    print(
        f"[Pool] {info['initializers']} shared initializers: "
        f"{info['device_gib']:.3f} GiB device pool, {info['host_gib']:.3f} GiB host_pool"
    )
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, required=True, help="Qwen3.8-Flash-Next HF checkpoint folder (safetensors)")
    ap.add_argument("--v12b-dir", type=Path, required=True, help="v12b package folder")
    ap.add_argument("--v12d-dir", type=Path, required=True, help="v12d package folder")
    ap.add_argument("--ops-dir", type=Path, required=True, help="folder of the multi-arch custom-op libraries")
    ap.add_argument("--out", type=Path, required=True, help="output package folder")
    ap.add_argument("--reuse-ple-from", type=Path, default=None)
    ap.add_argument("--reuse-qmoe-from", type=Path, default=None)
    ap.add_argument("--reuse-embed-from", type=Path, default=None)
    ap.add_argument("--plane-lib", type=Path, default=None)
    ap.add_argument("--provider-lib", type=Path, default=None)
    ap.add_argument("--keep-bf16-router-shared", action="store_true", default=False)
    ap.add_argument("--clean-flat-package", action="store_true", default=False)
    args = ap.parse_args()

    t_start = time.time()
    out = reject_output_overlap(
        args.out,
        {
            "--hf-dir": args.hf_dir,
            "--v12b-dir": args.v12b_dir,
            "--v12d-dir": args.v12d_dir,
            "--ops-dir": args.ops_dir,
            "--reuse-ple-from": args.reuse_ple_from,
            "--reuse-qmoe-from": args.reuse_qmoe_from,
            "--reuse-embed-from": args.reuse_embed_from,
            "--plane-lib": args.plane_lib,
            "--provider-lib": args.provider_lib,
        },
    )
    has_existing_ple = (out / "ple_table.txt").exists() and len(list((out / "ple").glob("ple_int8_*.data"))) == 10
    if args.clean_flat_package:
        for sub in ["libs", "provider", "ple"]:
            if (out / sub).exists() and sub != "ple":
                shutil.rmtree(out / sub)
            (out / sub).mkdir(parents=True, exist_ok=True)
    else:
        for sub in ["head_r2", "dense", "head_kv", "libs", "provider"]:
            if (out / sub).exists():
                shutil.rmtree(out / sub)
            (out / sub).mkdir(parents=True, exist_ok=True)
        (out / "target").mkdir(parents=True, exist_ok=True)
        (out / "ple").mkdir(parents=True, exist_ok=True)

    hf_reader = HFCheckpointReader(args.hf_dir)

    # 1. PLE INT8 shards
    if has_existing_ple:
        print(f"[PLE] Reusing existing INT8 PLE shards in {out / 'ple'}")
    elif args.reuse_ple_from and (args.reuse_ple_from / "ple_table.txt").exists():
        shutil.copy2(args.reuse_ple_from / "ple_table.txt", out / "ple_table.txt")
        for f in sorted((args.reuse_ple_from / "ple").glob("ple_int8_*.data")):
            dst_f = out / "ple" / f.name
            if dst_f.exists():
                dst_f.unlink()
            os.link(f, dst_f)
        print(f"[PLE] Hardlinked INT8 PLE shards from {args.reuse_ple_from}")
    else:
        build_ple_int8(hf_reader, args.v12d_dir, out)

    data_root = out if args.clean_flat_package else (out / "target")
    if args.reuse_qmoe_from is not None:
        for qf in ("qmoe_int4.data", "qmoe_scales.data"):
            src_qf = (
                (args.reuse_qmoe_from / "target" / qf)
                if (args.reuse_qmoe_from / "target" / qf).exists()
                else (args.reuse_qmoe_from / qf)
            )
            dst_qf = data_root / qf
            if dst_qf.exists():
                dst_qf.unlink()
            os.link(src_qf, dst_qf)

    # 2. QMoE INT4 weights & scales
    qmoe_meta = build_qmoe_int4(hf_reader, data_root)

    # 3. MatMulNBits INT8 weights & scales
    v12b_map = collect_v12b_bf16_map(args.v12b_dir)
    matmul_meta = build_matmul_int8(
        args.v12d_dir, v12b_map, data_root, keep_bf16_router_shared=args.keep_bf16_router_shared
    )

    # 4. Rewrite ONNX graphs
    models: dict[str, onnx.ModelProto] = {}
    subs_to_rewrite = ["target", "head_r2"] if args.clean_flat_package else ["target", "head_r2", "dense", "head_kv"]
    for sub in subs_to_rewrite:
        t0 = time.time()
        models[sub] = rewrite_submodel(
            sub,
            args.v12d_dir,
            v12b_map,
            qmoe_meta,
            matmul_meta,
            keep_bf16_router_shared=args.keep_bf16_router_shared,
            mtp_kv_bf16=True,
        )
        print(
            f"[Graph] Rewrote {sub}/model.onnx: {len(models[sub].graph.node)} nodes, "
            f"{len(models[sub].graph.initializer)} initializers ({time.time() - t0:.1f}s)"
        )

    if args.clean_flat_package:
        if args.reuse_embed_from is None:
            raise SystemExit("--reuse-embed-from is required with --clean-flat-package")
        pool_info = unify_and_pack_flat_int4_int8_package(models, args.v12d_dir, out, args.reuse_embed_from.resolve())
        for fname in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
            src = args.reuse_embed_from / fname
            if src.exists():
                dst = out / fname
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
        needed_libs = [
            "libqwen38_plane.so",
            "libqwen_packed_qsa_indexer_v11.so",
            "libqwen_fused_ops.so",
            "libqwen_fused_attn_r5.so",
            "libqwen_guarded_dsa_r2_v12.so",
            "libqwen_mtp_head.so",
            "libqwen_fold.so",
        ]
        lib_hashes = {}
        for lib_name in needed_libs:
            src_lib = (
                args.plane_lib if (lib_name == "libqwen38_plane.so" and args.plane_lib) else (args.ops_dir / lib_name)
            )
            dst_lib = out / "libs" / lib_name
            shutil.copy2(src_lib, dst_lib)
            lib_hashes[lib_name] = hashlib.sha256(dst_lib.read_bytes()).hexdigest()
        for dll in sorted((args.reuse_embed_from / "libs").glob("*.dll")):
            shutil.copy2(dll, out / "libs" / dll.name)
        prov_src = args.provider_lib or (args.ops_dir / "libonnxruntime_providers_cuda.so")
        if prov_src and prov_src.exists():
            shutil.copy2(prov_src, out / "provider" / "libonnxruntime_providers_cuda.so")
        cfg = json.loads((args.reuse_embed_from / "genai_config.json").read_text())
        cfg["model"]["decoder"]["session_options"]["custom_ops_libraries"] = [
            "libs/libqwen_packed_qsa_indexer_v11.so",
            "libs/libqwen_fused_ops.so",
            "libs/libqwen_fused_attn_r5.so",
            "libs/libqwen_guarded_dsa_r2_v12.so",
            "libs/libqwen_mtp_head.so",
            "libs/libqwen_fold.so",
        ]
        (out / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        manifest = {
            "variant": "qwen_flash_next_int4_int8_rc2",
            "target_gpu": "RTX 3090 / Ampere SM80+ (zero FP8 tensors)",
            "moe_quant": "com.microsoft::QMoE int4 symmetric block_size=32 weights_prepacked=1 (SM80 layout)",
            "matmul_quant": "com.microsoft::MatMulNBits int8 symmetric block_size=32 weight_prepacked=1 (SM80 layout)",
            "sensitive_bf16_weights": "MoE router (mlp.gate) + shared_expert (gate_proj, up_proj, down_proj)",
            "ple_quant": f"int8 symmetric scale={-PLE_INT8_SCALE}",
            "kv_quant": "target=int8_pc calibrated, mtp_head=bf16",
            "lm_head": "int8 MatMulNBits block_size=32",
            "embed_tokens": "int8 per-row CPU embedding.onnx + embedding.onnx.data (607.20 MiB)",
            "weights_pool": pool_info,
            "custom_op_libraries": lib_hashes,
            "build_seconds": round(time.time() - t_start, 2),
        }
        (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[Done] Built flat RC package {out} in {time.time() - t_start:.1f}s")
        return 0

    # 5. Consolidate remaining external initializers into target/{embed_tokens,lm_head,small_weights}.data
    consolidate_remaining_external_initializers(models, args.v12d_dir, out / "target")

    # 6. Symlink data files into head_r2, dense, head_kv and save ONNX models
    data_files = [
        "qmoe_int4.data",
        "qmoe_scales.data",
        "matmul_int8.data",
        "matmul_scales.data",
        "embed_tokens.data",
        "lm_head.data",
        "small_weights.data",
    ]
    for sub in ["head_r2", "dense", "head_kv"]:
        for df in data_files:
            dst = out / sub / df
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.link(out / "target" / df, dst)

    model_paths = {}
    for sub, m in models.items():
        mp = out / sub / "model.onnx"
        onnx.save(m, str(mp))
        model_paths[sub] = mp

    # 7. Write weights_pool.txt
    pool_info = write_weights_pool(out, model_paths)

    # 8. Copy QSA constants, tokenizer/config files, and multi-arch custom op libraries
    shutil.copy2(args.v12d_dir / "qsa_consts.txt", out / "qsa_consts.txt")
    shutil.copy2(args.v12d_dir / "qsa_consts.bin", out / "qsa_consts.bin")
    for fname in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]:
        src = args.v12d_dir / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    needed_libs = [
        "libqwen38_plane.so",
        "libqwen_packed_qsa_indexer_v11.so",
        "libqwen_fused_ops.so",
        "libqwen_fused_attn_r5.so",
        "libqwen_guarded_dsa_r2_v12.so",
        "libqwen_mtp_head.so",
        "libqwen_fold.so",
    ]
    lib_hashes = {}
    for lib_name in needed_libs:
        src_lib = args.ops_dir / lib_name
        dst_lib = out / "libs" / lib_name
        shutil.copy2(src_lib, dst_lib)
        lib_hashes[lib_name] = hashlib.sha256(dst_lib.read_bytes()).hexdigest()

    prov_src = args.ops_dir / "libonnxruntime_providers_cuda.so"
    if prov_src.exists():
        os.link(prov_src, out / "provider" / "libonnxruntime_providers_cuda.so")

    # 9. Write genai_config.json (no Hopper-only custom ops; dense & head_kv need zero custom ops)
    cfg = json.loads((args.v12d_dir / "genai_config.json").read_text())
    cfg["model"]["decoder"]["session_options"]["custom_ops_libraries"] = [
        "libs/libqwen_packed_qsa_indexer_v11.so",
        "libs/libqwen_fused_ops.so",
        "libs/libqwen_fused_attn_r5.so",
        "libs/libqwen_guarded_dsa_r2_v12.so",
        "libs/libqwen_mtp_head.so",
        "libs/libqwen_fold.so",
    ]
    cfg["model"]["qwen38_graph"]["prefill"]["dense"]["session_options"].pop("custom_ops_libraries", None)
    cfg["model"]["qwen38_graph"]["prefill"]["head_kv"]["session_options"].pop("custom_ops_libraries", None)
    (out / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    manifest = {
        "variant": "v12d_int4_int8_bf16lm_engine",
        "moe_quant": "com.microsoft::QMoE int4 symmetric block_size=32 weights_prepacked=1 (SM80 layout)",
        "matmul_quant": "com.microsoft::MatMulNBits int8 symmetric block_size=32 weight_prepacked=1 (SM80 layout)",
        "ple_quant": f"int8 symmetric scale={-PLE_INT8_SCALE}",
        "kv_quant": "int8_pc calibrated",
        "lm_head": "bf16",
        "embed_tokens": "bf16 host_pool (CPU mapped)",
        "weights_pool": pool_info,
        "custom_op_libraries": lib_hashes,
        "build_seconds": round(time.time() - t_start, 2),
    }
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[Done] Built {out} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
