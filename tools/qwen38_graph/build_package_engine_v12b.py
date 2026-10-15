#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Assemble an Engine-facing package of the v12b stack (model.type "qwen38_flash_graph": folded decode cycle, fast dense
prefill, shared weight pool) from an existing v12b package, with optional KV-cache quantization (INT8 or FP8 per-channel)
and 16-bit weight quantization (FP8 E4M3FN or NVFP4 block-16 on attention/shared-expert projections and MTP head experts,
plus dense-prefill weight sharing).

usage: build_package_engine_v12b.py --from-package <dir> --out <dir> --plane-lib <libqwen38_plane.so>
                                    [--stop-token-id N] [--prefill-chunk-tokens N] [--copy]
                                    [--kv-quant {none,int8,fp8}] [--kv-scale-file <json>] [--dsa-lib <so>]
                                    [--weight-quant {none,fp8,nvfp4,fp8_attn_nvfp4_shared}]
                                    [--mtp-moe-quant {none,fp8,nvfp4}] [--share-dense-weights]
                                    [--gemm-lib <so>] [--mtp-lib <so>]
Every path is an argument; nothing is read from the environment. Load the result with og.Config / og.Model / og.Engine and
register the CUDA execution provider of <out>/provider/ (one provider path per process).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPLACED = {"genai_config.json", "package_manifest.json", "README.md"}
PLANE_REL = Path("libs/libqwen38_plane.so")


def link_file(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.is_symlink():
        dst.symlink_to(os.readlink(src))
        return
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def link_tree(src: Path, dst: Path, copy: bool, skip_rel: set[Path] | None = None) -> int:
    count = 0
    skip_rel = skip_rel or set()
    for root, _dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            relative = rel / name
            if relative in (Path(n) for n in REPLACED) or relative == PLANE_REL or relative in skip_rel:
                continue
            link_file(Path(root) / name, dst / relative, copy)
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(model_path: Path) -> dict:
    import onnx  # noqa: PLC0415

    model = onnx.load(str(model_path), load_external_data=False)
    out = {}
    for t in model.graph.initializer:
        ext = {e.key: e.value for e in t.external_data}
        if not ext:
            continue
        path = model_path.parent / ext["location"]
        st = os.stat(path)
        offset = int(ext.get("offset", 0))
        length = int(ext.get("length", 0)) or st.st_size - offset
        out[t.name] = {
            "key": (st.st_ino, offset, length, t.data_type, tuple(t.dims)),
            "file": str(Path(ext["location"])),
            "dtype": t.data_type,
            "dims": list(t.dims),
            "offset": offset,
            "length": length,
        }
    return out


def write_pool(out: Path, models: dict[str, Path]) -> dict:
    """weights_pool.txt: initializers used (same inode / offset / length / dtype / dims) by >= 2 of the sessions."""
    inv = {label: inventory(path) for label, path in models.items()}
    users = defaultdict(set)
    for label, items in inv.items():
        for item in items.values():
            users[item["key"]].add(label)
    pool_ids, lines, sessions = {}, [], []
    total = 0
    for label, items in inv.items():
        rel_dir = models[label].parent.relative_to(out)
        for name, item in items.items():
            if len(users[item["key"]]) < 2:
                continue
            if item["key"] not in pool_ids:
                pool_ids[item["key"]] = len(pool_ids)
                dims = " ".join(str(d) for d in item["dims"])
                lines.append(
                    f"pool {pool_ids[item['key']]} {item['dtype']} {item['offset']} {item['length']} {len(item['dims'])} "
                    f"{dims} {rel_dir / item['file']}".replace("  ", " ")
                )
                total += item["length"]
            sessions.append(f"session {label} {name} {pool_ids[item['key']]}")
    (out / "weights_pool.txt").write_text("qwen38_weights_pool 1\n" + "\n".join(lines + sessions) + "\n")
    return {
        "initializers": len(pool_ids),
        "gib": total / 2**30,
        "sessions": {lbl: sum(1 for s in sessions if s.split()[1] == lbl) for lbl in inv},
    }


def apply_kv_quant(
    source: Path,
    out: Path,
    target_model,
    head_r2_model,
    kv_quant: str,
    scale_file: Path,
) -> dict:
    import numpy as np  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    data = json.loads(scale_file.read_text())
    layer_ids = [int(x) for x in data["layer_ids"]]
    k_scales = {lid: [float(v) for v in s] for lid, s in zip(layer_ids, data["scales"]["k_scales"], strict=True)}
    v_scales = {lid: [float(v) for v in s] for lid, s in zip(layer_ids, data["scales"]["v_scales"], strict=True)}
    head_k = [float(v) for v in data["scales"]["head_k_scales"][0]]
    head_v = [float(v) for v in data["scales"]["head_v_scales"][0]]
    kv_dtype = TensorProto.INT8 if kv_quant == "int8" else TensorProto.FLOAT8E4M3FN

    for model, is_head_only in ((target_model, False), (head_r2_model, True)):
        for vi in list(model.graph.input) + list(model.graph.output):
            if vi.name.endswith((".key", ".value")) and (
                vi.name.startswith(("past_key_values.", "present.", "head.past_key_values.", "head.present."))
            ):
                vi.type.tensor_type.elem_type = kv_dtype
        for node in model.graph.node:
            if node.op_type != "GuardedDSA":
                continue
            pk_name = node.input[3]
            if is_head_only or pk_name.startswith("head."):
                ks, vs = head_k, head_v
            else:
                lid = int(pk_name.split(".")[1])
                ks, vs = k_scales[lid], v_scales[lid]
            keep_attrs = [a for a in node.attribute if a.name not in ("k_scale", "v_scale")]
            del node.attribute[:]
            node.attribute.extend(keep_attrs)
            node.attribute.append(helper.make_attribute("k_scale", ks))
            node.attribute.append(helper.make_attribute("v_scale", vs))

    # Append kscale.<layer> and vscale.<layer> to qsa_consts.bin and qsa_consts.txt
    blob = bytearray((source / "qsa_consts.bin").read_bytes())
    lines = [ln for ln in (source / "qsa_consts.txt").read_text().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not (" kscale." in ln or " vscale." in ln)]

    def add_scale(set_name: str, name: str, vals: list[float]) -> None:
        while len(blob) % 16:
            blob.append(0)
        raw = np.asarray(vals, dtype=np.float32).tobytes()
        lines.append(f"tensor {set_name} {name} {len(blob)} {len(raw)}")
        blob.extend(raw)

    for lid in sorted(layer_ids):
        add_scale("target", f"kscale.{lid}", k_scales[lid])
        add_scale("target", f"vscale.{lid}", v_scales[lid])
    add_scale("head", "kscale.0", head_k)
    add_scale("head", "vscale.0", head_v)

    (out / "qsa_consts.bin").write_bytes(bytes(blob))
    (out / "qsa_consts.txt").write_text("\n".join(lines) + "\n")
    return {
        "kv_quant": kv_quant,
        "kv_dtype": int(kv_dtype),
        "scale_file": str(scale_file),
        "target_layers": len(layer_ids),
        "qsa_consts_bytes": len(blob),
    }


def apply_weight_and_dense_transforms(
    source: Path,
    out: Path,
    target_model,
    head_r2_model,
    dense_model,
    weight_quant: str,
    mtp_moe_quant: str,
    share_dense: bool,
) -> dict:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    e2m1_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)

    quant_path = out / "target" / "quant_weights.data"
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    quant_file = quant_path.open("wb")
    quant_pos = 0
    quant_cache: dict = {}

    def write_quant_blob(raw_bytes: bytes) -> tuple[int, int]:
        nonlocal quant_pos
        pad = (-quant_pos) % 4096
        if pad:
            quant_file.write(b"\0" * pad)
            quant_pos += pad
        off = quant_pos
        quant_file.write(raw_bytes)
        quant_pos += len(raw_bytes)
        return off, len(raw_bytes)

    def load_src_bf16(base_dir: Path, tensor) -> tuple[tuple, torch.Tensor]:
        ext = {e.key: e.value for e in tensor.external_data}
        p = base_dir / ext["location"]
        st = os.stat(p)
        off = int(ext.get("offset", 0))
        length = int(ext.get("length", 0)) or (st.st_size - off)
        key = (st.st_ino, off, length, tuple(tensor.dims))
        with p.open("rb") as f:
            f.seek(off)
            buf = f.read(length)
        arr = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(list(tensor.dims)).to(device)
        return key, arr

    def set_external(tensor, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> None:
        tensor.data_type = dtype
        del tensor.dims[:]
        tensor.dims.extend(dims)
        tensor.ClearField("raw_data")
        del tensor.external_data[:]
        tensor.data_location = TensorProto.EXTERNAL
        for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
            e = tensor.external_data.add()
            e.key, e.value = k, v

    def quantize_2d(base_dir: Path, tensor, qmode: str) -> dict:
        src_key, w_bf16 = load_src_bf16(base_dir, tensor)
        cache_key = (*src_key, qmode)
        if cache_key in quant_cache:
            return quant_cache[cache_key]
        k_dim, n_dim = int(w_bf16.shape[0]), int(w_bf16.shape[1])
        wf = w_bf16.float()
        if qmode == "fp8":
            amax = wf.abs().amax(dim=0).clamp(min=1e-12)
            w_scale = (amax / 448.0).to(torch.float32)
            wq = (wf / w_scale[None, :]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
            raw = wq.cpu().numpy().tobytes()
            off, length = write_quant_blob(raw)
            info = {
                "offset": off,
                "length": length,
                "dtype": TensorProto.FLOAT8E4M3FN,
                "dims": [k_dim, n_dim],
                "n_cols": n_dim,
                "w_scale": [float(x) for x in w_scale.cpu().numpy()],
            }
        elif qmode == "nvfp4":
            assert n_dim % 16 == 0, f"NVFP4 requires N % 16 == 0, got {n_dim}"
            wb = wf.reshape(k_dim, n_dim // 16, 16)
            g_amax = wb.abs().amax(dim=(0, 2)).clamp(min=1e-12)  # [N // 16]
            g_scale = (g_amax / (6.0 * 448.0)).to(torch.float32)  # [N // 16]
            b_amax = wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)  # [K, N // 16, 1]
            b_scale_fp8 = ((b_amax / 6.0) / g_scale[None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            eff_scale = b_scale_fp8.float() * g_scale[None, :, None]
            scaled = wb / eff_scale.clamp(min=1e-12)
            sign_bit = (scaled < 0).to(torch.uint8) << 3
            mag_idx = (scaled.abs().unsqueeze(-1) - e2m1_grid).abs().argmin(dim=-1).to(torch.uint8)
            nibs = (sign_bit | mag_idx).reshape(k_dim, n_dim)
            packed_bytes = nibs[:, 0::2] | (nibs[:, 1::2] << 4)
            scale_bytes = b_scale_fp8.view(torch.uint8).reshape(k_dim, n_dim // 16)
            packed_row = torch.cat([packed_bytes, scale_bytes], dim=-1).contiguous()
            raw = packed_row.cpu().numpy().tobytes()
            off, length = write_quant_blob(raw)
            w_scale_full = g_scale.repeat_interleave(16).cpu().numpy()
            info = {
                "offset": off,
                "length": length,
                "dtype": TensorProto.UINT8,
                "dims": [k_dim, (n_dim * 9) // 16],
                "n_cols": n_dim,
                "w_scale": [float(x) for x in w_scale_full],
            }
        else:
            raise ValueError(f"Unknown 2D quant mode {qmode}")
        quant_cache[cache_key] = info
        return info

    def quantize_3d_moe(base_dir: Path, tensor, qmode: str) -> dict:
        src_key, w_bf16 = load_src_bf16(base_dir, tensor)
        cache_key = (*src_key, f"moe_{qmode}")
        if cache_key in quant_cache:
            return quant_cache[cache_key]
        num_e, num_r, num_k = [int(d) for d in w_bf16.shape]
        # Process in chunks of 64 experts to bound temporary GPU memory
        chunks = []
        g_scale_val = 1.0
        if qmode == "nvfp4":
            # Compute global amax across all experts in chunks
            g_max = 1e-12
            for e0 in range(0, num_e, 64):
                sub = w_bf16[e0 : e0 + 64].float()
                g_max = max(g_max, float(sub.abs().max().item()))
            g_scale_val = g_max / (6.0 * 448.0)

        for e0 in range(0, num_e, 64):
            sub = w_bf16[e0 : e0 + 64].float()
            ec = sub.shape[0]
            if qmode == "fp8":
                amax = sub.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                scale = (amax / 448.0).squeeze(-1).to(torch.float32)  # [ec, R]
                q = (sub / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
                trailer = torch.zeros((ec, num_r, 16), dtype=torch.uint8, device=device)
                trailer[:, :, :4] = scale.contiguous().view(torch.uint8).reshape(ec, num_r, 4)
                chunks.append(torch.cat([q, trailer], dim=-1).cpu().numpy())
            elif qmode == "nvfp4":
                wb = sub.reshape(ec, num_r, num_k // 16, 16)
                b_amax = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                b_scale_fp8 = ((b_amax / 6.0) / g_scale_val).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                eff_scale = b_scale_fp8.float() * g_scale_val
                scaled = wb / eff_scale.clamp(min=1e-12)
                sign_bit = (scaled < 0).to(torch.uint8) << 3
                mag_idx = (scaled.abs().unsqueeze(-1) - e2m1_grid).abs().argmin(dim=-1).to(torch.uint8)
                nibs = (sign_bit | mag_idx).reshape(ec, num_r, num_k)
                packed_bytes = nibs[:, :, 0::2] | (nibs[:, :, 1::2] << 4)
                scale_bytes = b_scale_fp8.view(torch.uint8).reshape(ec, num_r, num_k // 16)
                chunks.append(torch.cat([packed_bytes, scale_bytes], dim=-1).cpu().numpy())
            else:
                raise ValueError(f"Unknown MoE quant mode {qmode}")
        full_arr = np.concatenate(chunks, axis=0)
        off, length = write_quant_blob(full_arr.tobytes())
        info = {
            "offset": off,
            "length": length,
            "dtype": TensorProto.UINT8,
            "dims": list(full_arr.shape),
            "global_scale": float(g_scale_val),
        }
        quant_cache[cache_key] = info
        return info

    def mode_for_node(node) -> str | None:
        if weight_quant == "none":
            return None
        if weight_quant == "fp8":
            return "fp8"
        if weight_quant == "nvfp4":
            return "nvfp4"
        if weight_quant == "fp8_attn_nvfp4_shared":
            return "nvfp4" if node.op_type == "SharedExpertDown" else "fp8"
        raise ValueError(f"Unsupported weight_quant={weight_quant}")

    # Track quantized info by initializer name in target for dense sharing
    target_quant_by_name: dict[str, dict] = {}
    stats = collections.Counter()

    for mlabel, model, base_dir in (
        ("target", target_model, source / "target"),
        ("head_r2", head_r2_model, source / "head_r2"),
    ):
        inits = {t.name: t for t in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type in ("SkinnyGemm", "SharedExpertDown") and weight_quant != "none":
                qmode = mode_for_node(node)
                w_idx = 1 if node.op_type == "SkinnyGemm" else 2
                w_name = node.input[w_idx]
                t = inits[w_name]
                info = quantize_2d(base_dir, t, qmode)
                set_external(t, "quant_weights.data", info["offset"], info["length"], info["dtype"], info["dims"])
                keep_attrs = [a for a in node.attribute if a.name != "w_scale"]
                del node.attribute[:]
                node.attribute.extend(keep_attrs)
                node.attribute.append(helper.make_attribute("w_scale", info["w_scale"]))
                if mlabel == "target":
                    target_quant_by_name[w_name] = info
                stats[f"{mlabel}_{node.op_type}_{qmode}"] += 1
            elif node.op_type == "MtpMoe" and mtp_moe_quant != "none":
                gu_name, dn_name = node.input[2], node.input[3]
                gu_info = quantize_3d_moe(base_dir, inits[gu_name], mtp_moe_quant)
                dn_info = quantize_3d_moe(base_dir, inits[dn_name], mtp_moe_quant)
                set_external(
                    inits[gu_name],
                    "quant_weights.data",
                    gu_info["offset"],
                    gu_info["length"],
                    gu_info["dtype"],
                    gu_info["dims"],
                )
                set_external(
                    inits[dn_name],
                    "quant_weights.data",
                    dn_info["offset"],
                    dn_info["length"],
                    dn_info["dtype"],
                    dn_info["dims"],
                )
                keep_attrs = [a for a in node.attribute if a.name not in ("gate_up_global_scale", "down_global_scale")]
                del node.attribute[:]
                node.attribute.extend(keep_attrs)
                if mtp_moe_quant == "nvfp4":
                    node.attribute.append(helper.make_attribute("gate_up_global_scale", gu_info["global_scale"]))
                    node.attribute.append(helper.make_attribute("down_global_scale", dn_info["global_scale"]))
                stats[f"{mlabel}_MtpMoe_{mtp_moe_quant}"] += 1

    quant_file.close()
    if quant_pos > 0:
        link_file(quant_path, out / "head_r2" / "quant_weights.data", False)
        link_file(quant_path, out / "dense" / "quant_weights.data", False)
    else:
        quant_path.unlink(missing_ok=True)

    # Rewrite dense/model.onnx to share merged_proj, quantized weights, and hc_tiled.data with target/model.onnx
    if share_dense:
        inits_d = {t.name: t for t in dense_model.graph.initializer}
        inits_t = {t.name: t for t in target_model.graph.initializer}

        def weight_info_d(name: str):
            t = inits_d.get(name)
            if t is None or t.data_type != TensorProto.BFLOAT16 or len(t.dims) != 2:
                return None
            meta = {e.key: e.value for e in t.external_data}
            if "location" not in meta:
                return None
            return t, meta

        # 1. Find the 97 shared-input MatMul groups in dense/model.onnx matching merged_proj.0..96.weight_t
        groups = collections.OrderedDict()
        for index, node in enumerate(dense_model.graph.node):
            if node.op_type != "MatMul" or weight_info_d(node.input[1]) is None:
                continue
            t, _ = weight_info_d(node.input[1])
            if t.dims[1] % 8 != 0:
                continue
            groups.setdefault((node.input[0], t.dims[0]), []).append((index, node))
        group_list = [g for g in groups.values() if len(g) >= 2]

        if weight_quant == "none":
            link_file(out / "target" / "merged.data", out / "dense" / "merged.data", False)
        link_file(out / "target" / "hc_tiled.data", out / "dense" / "hc_tiled.data", False)

        replaced_indices = set()
        inserted_at = collections.defaultdict(list)

        for gi, members in enumerate(group_list):
            m_name = f"merged_proj.{gi}.weight_t"
            assert m_name in inits_t, f"Missing {m_name} in target"
            widths = [int(inits_d[n.input[1]].dims[1]) for _, n in members]
            first_idx = members[0][0]
            x_in = members[0][1].input[0]

            t_merged = TensorProto()
            t_merged.CopyFrom(inits_t[m_name])
            dense_model.graph.initializer.append(t_merged)

            split_name = f"merged_proj.{gi}.split_sizes"
            dense_model.graph.initializer.append(numpy_helper.from_array(np.array(widths, dtype=np.int64), split_name))

            if m_name in target_quant_by_name:
                qinfo = target_quant_by_name[m_name]
                w_bf16_name = f"{m_name}.deq_bf16"
                deq_node = helper.make_node(
                    "DequantWeightBf16",
                    [m_name, x_in],
                    [w_bf16_name],
                    name=f"dense_merge/{gi}/Dequant",
                    domain="qwen.fused_gemm",
                    n_cols=qinfo["n_cols"],
                    w_scale=qinfo["w_scale"],
                )
                inserted_at[first_idx].append(deq_node)
                mm_weight = w_bf16_name
            else:
                mm_weight = m_name

            mm_out = f"merged_proj.{gi}.out"
            mm_node = helper.make_node("MatMul", [x_in, mm_weight], [mm_out], name=f"dense_merge/{gi}/MatMul")
            sp_node = helper.make_node(
                "Split",
                [mm_out, split_name],
                [n.output[0] for _, n in members],
                name=f"dense_merge/{gi}/Split",
                axis=-1,
            )
            inserted_at[first_idx].extend([mm_node, sp_node])
            for idx, _ in members:
                replaced_indices.add(idx)
            stats["dense_merged_groups"] += 1

        # 2. Remaining standalone MatMuls in dense: either quantized in target OR tiled in hc_tiled.data
        for index, node in enumerate(dense_model.graph.node):
            if index in replaced_indices or node.op_type != "MatMul":
                continue
            w_name = node.input[1]
            if w_name in target_quant_by_name:
                qinfo = target_quant_by_name[w_name]
                set_external(
                    inits_d[w_name],
                    "quant_weights.data",
                    qinfo["offset"],
                    qinfo["length"],
                    qinfo["dtype"],
                    qinfo["dims"],
                )
                w_bf16_name = f"{w_name}.deq_bf16"
                deq_node = helper.make_node(
                    "DequantWeightBf16",
                    [w_name, node.input[0]],
                    [w_bf16_name],
                    name=f"dense_deq/{index}/Dequant",
                    domain="qwen.fused_gemm",
                    n_cols=qinfo["n_cols"],
                    w_scale=qinfo["w_scale"],
                )
                inserted_at[index].append(deq_node)
                node.input[1] = w_bf16_name
                stats["dense_dequant_matmuls"] += 1
            else:
                for kind, suffix in ((0, "hc_tiled_down"), (1, "hc_tiled_up")):
                    tiled_name = f"{w_name}.{suffix}"
                    if tiled_name in inits_t:
                        t_tiled = TensorProto()
                        t_tiled.CopyFrom(inits_t[tiled_name])
                        dense_model.graph.initializer.append(t_tiled)
                        untiled_out = f"{w_name}.untiled_bf16"
                        untile_node = helper.make_node(
                            "UntileHcBf16",
                            [tiled_name, node.input[0]],
                            [untiled_out],
                            name=f"dense_hc/{index}/Untile",
                            domain="qwen.fused_gemm",
                            kind=kind,
                        )
                        inserted_at[index].append(untile_node)
                        node.input[1] = untiled_out
                        stats[f"dense_untile_{suffix}"] += 1
                        break

        rebuilt_nodes = []
        for index, node in enumerate(dense_model.graph.node):
            if index in inserted_at:
                rebuilt_nodes.extend(inserted_at[index])
            if index not in replaced_indices:
                rebuilt_nodes.append(node)
        del dense_model.graph.node[:]
        dense_model.graph.node.extend(rebuilt_nodes)

        used_inputs = {inp for n in dense_model.graph.node for inp in n.input if inp}
        kept_inits = [t for t in dense_model.graph.initializer if t.name in used_inputs]
        del dense_model.graph.initializer[:]
        dense_model.graph.initializer.extend(kept_inits)

        if all(o.domain != "qwen.fused_gemm" for o in dense_model.opset_import):
            opset = dense_model.opset_import.add()
            opset.domain, opset.version = "qwen.fused_gemm", 1

    return {"quant_bytes": quant_pos, "quant_gib": quant_pos / 2**30, "stats": dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-package", type=Path, required=True, help="existing v12b package folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--plane-lib", type=Path, required=True, help="libqwen38_plane.so built from the GenAI tree (ABI 2)"
    )
    parser.add_argument(
        "--stop-token-id", type=int, default=None, help="replace eos_token_id (fixed-length benchmarks)"
    )
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=None,
        help="model.qwen38_graph.prefill_chunk_tokens (only used when prefill.enabled is false)",
    )
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    parser.add_argument(
        "--kv-quant",
        choices=("none", "int8", "fp8"),
        default="none",
        help="quantize target and head KV caches to calibrated per-channel INT8 (preferred) or FP8 E4M3FN",
    )
    parser.add_argument(
        "--kv-scale-file",
        type=Path,
        default=None,
        help="path to calibrated KV scale JSON (required when --kv-quant != none)",
    )
    parser.add_argument(
        "--dsa-lib",
        type=Path,
        default=None,
        help="optional GuardedDSA library override (libqwen_guarded_dsa_r2_v12.so with KV quant support)",
    )
    parser.add_argument(
        "--weight-quant",
        choices=("none", "fp8", "nvfp4", "fp8_attn_nvfp4_shared"),
        default="none",
        help="quantize BF16 attention and shared-expert projections in target/head_r2/dense",
    )
    parser.add_argument(
        "--mtp-moe-quant",
        choices=("none", "fp8", "nvfp4"),
        default="none",
        help="quantize the 4.69 GiB BF16 MTP head routed experts in target/head_r2",
    )
    parser.add_argument(
        "--share-dense-weights",
        action="store_true",
        default=False,
        help="rewrite dense/model.onnx to share merged_proj, quantized projections, and hc_tiled.data with target",
    )
    parser.add_argument(
        "--gemm-lib",
        type=Path,
        default=None,
        help="optional fused_gemm library override (libqwen_fused_gemm_r5.so with FP8/NVFP4 support)",
    )
    parser.add_argument(
        "--mtp-lib",
        type=Path,
        default=None,
        help="optional mtp_head library override (libqwen_mtp_head.so with FP8/NVFP4 MtpMoe support)",
    )
    args = parser.parse_args()

    source, out = args.from_package.resolve(), args.out.resolve()
    if not (source / "genai_config.json").exists():
        raise SystemExit(f"{source} is not a package (no genai_config.json)")
    if not args.plane_lib.exists():
        raise SystemExit(f"missing plane library: {args.plane_lib}")
    if out == source:
        raise SystemExit("--out must differ from --from-package")
    out.mkdir(parents=True, exist_ok=True)

    share_dense = args.share_dense_weights or (args.weight_quant != "none")
    transform_models = (
        args.kv_quant != "none" or args.weight_quant != "none" or args.mtp_moe_quant != "none" or share_dense
    )

    skip_rel: set[Path] = set()
    if transform_models:
        skip_rel.update({Path("target/model.onnx"), Path("head_r2/model.onnx"), Path("weights_pool.txt")})
        if share_dense:
            skip_rel.add(Path("dense/model.onnx"))
        if args.kv_quant != "none":
            skip_rel.update({Path("qsa_consts.bin"), Path("qsa_consts.txt")})

    linked = link_tree(source, out, args.copy, skip_rel=skip_rel)
    link_file(args.plane_lib.resolve(), out / PLANE_REL, True)

    config = json.loads((source / "genai_config.json").read_text())
    qwen = config["model"]["qwen38_graph"]
    if not qwen.get("fold", {}).get("enabled") or not qwen.get("prefill", {}).get("enabled"):
        raise SystemExit("the source package is not a v12 package (fold.enabled and prefill.enabled are required)")
    if args.stop_token_id is not None:
        config["model"]["eos_token_id"] = [args.stop_token_id]
    if args.prefill_chunk_tokens is not None:
        qwen["prefill_chunk_tokens"] = args.prefill_chunk_tokens

    kv_info = None
    weight_info = None
    pool_info = None

    if transform_models:
        import onnx  # noqa: PLC0415

        target_model = onnx.load(str(source / "target/model.onnx"), load_external_data=False)
        head_r2_model = onnx.load(str(source / "head_r2/model.onnx"), load_external_data=False)
        dense_model = onnx.load(str(source / "dense/model.onnx"), load_external_data=False) if share_dense else None

        if args.kv_quant != "none":
            scale_file = args.kv_scale_file
            if scale_file is None or not scale_file.exists():
                raise SystemExit("--kv-scale-file is required and must exist when --kv-quant != none")
            if args.dsa_lib is not None and not args.dsa_lib.exists():
                raise SystemExit(f"missing GuardedDSA library: {args.dsa_lib}")
            kv_info = apply_kv_quant(source, out, target_model, head_r2_model, args.kv_quant, scale_file.resolve())
            if args.dsa_lib is not None:
                link_file(args.dsa_lib.resolve(), out / "libs" / "libqwen_guarded_dsa_r2_v12.so", True)

        if args.weight_quant != "none" or args.mtp_moe_quant != "none" or share_dense:
            if (
                (args.weight_quant != "none" or share_dense)
                and args.gemm_lib is not None
                and not args.gemm_lib.exists()
            ):
                raise SystemExit(f"missing gemm library: {args.gemm_lib}")
            if args.mtp_moe_quant != "none" and args.mtp_lib is not None and not args.mtp_lib.exists():
                raise SystemExit(f"missing mtp library: {args.mtp_lib}")
            weight_info = apply_weight_and_dense_transforms(
                source,
                out,
                target_model,
                head_r2_model,
                dense_model,
                args.weight_quant,
                args.mtp_moe_quant,
                share_dense,
            )
            if (args.weight_quant != "none" or share_dense) and args.gemm_lib is not None:
                link_file(args.gemm_lib.resolve(), out / "libs" / "libqwen_fused_gemm_r5.so", True)
            if args.mtp_moe_quant != "none" and args.mtp_lib is not None:
                link_file(args.mtp_lib.resolve(), out / "libs" / "libqwen_mtp_head.so", True)

        onnx.save(target_model, str(out / "target/model.onnx"))
        onnx.save(head_r2_model, str(out / "head_r2/model.onnx"))
        if share_dense and dense_model is not None:
            onnx.save(dense_model, str(out / "dense/model.onnx"))
            dense_so = qwen["prefill"]["dense"]["session_options"]
            dense_so["custom_ops_libraries"] = ["libs/libqwen_fused_gemm_r5.so"]
            dense_so["provider_options"] = [{"cuda": {"arena_extend_strategy": "kSameAsRequested"}}]
            qwen["prefill"]["head_kv"]["session_options"]["provider_options"] = [
                {"cuda": {"arena_extend_strategy": "kSameAsRequested"}}
            ]

        models_map = {
            "target": out / "target/model.onnx",
            "head_r2": out / "head_r2/model.onnx",
            "dense": out / "dense/model.onnx",
            "head_kv": out / "head_kv/model.onnx",
        }
        pool_info = write_pool(out, models_map)

    (out / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")

    libs = {}
    for path in sorted((out / "libs").glob("*.so")):
        libs[path.name] = sha256(path)
    provider_dir = out / "provider"
    providers = {p.name: sha256(p) for p in sorted(provider_dir.glob("*.so"))} if provider_dir.exists() else {}
    manifest = {
        "from_package": str(source),
        "stack": "v12b-engine",
        "hard_linked_files": linked,
        "eos_token_id": config["model"]["eos_token_id"],
        "plane_library": {"source": str(args.plane_lib.resolve()), "sha256": libs.get(PLANE_REL.name)},
        "libs_sha256": libs,
        "provider_sha256": providers,
    }
    if kv_info is not None:
        manifest["kv_quant"] = kv_info
    if weight_info is not None:
        manifest["weight_quant"] = weight_info
    if pool_info is not None:
        manifest["weights_pool"] = pool_info
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"package: {out} ({linked} files linked)")
    print(f"plane: {libs.get(PLANE_REL.name)}")
    if pool_info is not None:
        print(f"pool: {pool_info['initializers']} initializers, {pool_info['gib']:.3f} GiB")
    for name, digest in providers.items():
        print(f"provider {name}: {digest}")


if __name__ == "__main__":
    sys.exit(main())
