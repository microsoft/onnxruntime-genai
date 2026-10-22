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


def write_pool(out: Path, models: dict[str, Path], host_map_embed: bool = False, exclude_embed: bool = False) -> dict:
    """weights_pool.txt: initializers used (same inode / offset / length / dtype / dims) by >= 2 of the sessions."""
    inv = {label: inventory(path) for label, path in models.items()}
    users = defaultdict(set)
    for label, items in inv.items():
        for name, item in items.items():
            if exclude_embed and (
                name.endswith("embed_tokens.weight") or Path(item["file"]).name == "embedding.onnx.data"
            ):
                continue
            users[item["key"]].add(label)
    pool_ids, lines, sessions = {}, [], []
    total = 0
    host_total = 0
    for label, items in inv.items():
        rel_dir = models[label].parent.relative_to(out)
        for name, item in items.items():
            if exclude_embed and (
                name.endswith("embed_tokens.weight") or Path(item["file"]).name == "embedding.onnx.data"
            ):
                continue
            if len(users[item["key"]]) < 2:
                continue
            if item["key"] not in pool_ids:
                pool_ids[item["key"]] = len(pool_ids)
                dims = " ".join(str(d) for d in item["dims"])
                is_host_embed = host_map_embed and name.endswith("embed_tokens.weight")
                prefix = "host_pool" if is_host_embed else "pool"
                lines.append(
                    f"{prefix} {pool_ids[item['key']]} {item['dtype']} {item['offset']} {item['length']} {len(item['dims'])} "
                    f"{dims} {rel_dir / item['file']}".replace("  ", " ")
                )
                if is_host_embed:
                    host_total += item["length"]
                else:
                    total += item["length"]
            sessions.append(f"session {label} {name} {pool_ids[item['key']]}")
    (out / "weights_pool.txt").write_text("qwen38_weights_pool 1\n" + "\n".join(lines + sessions) + "\n")
    res = {
        "initializers": len(pool_ids),
        "gib": total / 2**30,
        "sessions": {lbl: sum(1 for s in sessions if s.split()[1] == lbl) for lbl in inv},
    }
    if host_total > 0:
        res["host_gib"] = host_total / 2**30
    return res


def apply_kv_quant(
    source: Path,
    out: Path,
    target_model,
    head_r2_model,
    kv_quant: str,
    scale_file: Path,
    mtp_kv_quant: str = "same",
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
    quantize_head_kv = (mtp_kv_quant != "bf16")

    for model, is_head_only in ((target_model, False), (head_r2_model, True)):
        if is_head_only and not quantize_head_kv:
            continue
        for vi in list(model.graph.input) + list(model.graph.output):
            if not vi.name.endswith((".key", ".value")):
                continue
            if vi.name.startswith(("head.past_key_values.", "head.present.")):
                if quantize_head_kv:
                    vi.type.tensor_type.elem_type = kv_dtype
            elif vi.name.startswith(("past_key_values.", "present.")):
                vi.type.tensor_type.elem_type = kv_dtype
        for node in model.graph.node:
            if node.op_type != "GuardedDSA":
                continue
            pk_name = node.input[3]
            if is_head_only or pk_name.startswith("head."):
                if not quantize_head_kv:
                    continue
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
    if quantize_head_kv:
        add_scale("head", "kscale.0", head_k)
        add_scale("head", "vscale.0", head_v)

    (out / "qsa_consts.bin").write_bytes(bytes(blob))
    (out / "qsa_consts.txt").write_text("\n".join(lines) + "\n")
    return {
        "kv_quant": kv_quant,
        "mtp_kv_quant": "bf16" if not quantize_head_kv else kv_quant,
        "kv_dtype": int(kv_dtype),
        "scale_file": str(scale_file),
        "target_layers": len(layer_ids),
        "qsa_consts_bytes": len(blob),
    }


def _fuse_dense_pointwise_ops(dense_model, base_dir: Path, stats: collections.Counter) -> None:
    """Fold constant Casts/Adds and replace Div->Sigmoid->Mul, Swish->Mul, and shared-expert gate+add with qwen.fused ops."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    graph = dense_model.graph
    init = {t.name: t for t in graph.initializer}

    def load_init_f32(t) -> np.ndarray:
        if t.external_data:
            meta = {e.key: e.value for e in t.external_data}
            p = base_dir / meta["location"]
            off = int(meta.get("offset", 0))
            length = int(meta.get("length", 0)) or (os.stat(p).st_size - off)
            with p.open("rb") as f:
                f.seek(off)
                buf = f.read(length)
        else:
            buf = bytes(t.raw_data)
        if t.data_type == TensorProto.BFLOAT16:
            arr = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).float().numpy()
            return arr.reshape(list(t.dims))
        return numpy_helper.to_array(t).astype(np.float32)

    cast_map: dict[tuple[str, int], str] = {}
    for node in list(graph.node):
        if node.op_type != "Cast" or node.input[0] not in init:
            continue
        to = next((a.i for a in node.attribute if a.name == "to"), None)
        if to != TensorProto.FLOAT:
            continue
        key = (node.input[0], to)
        if key not in cast_map:
            val = load_init_f32(init[node.input[0]])
            name = f"{node.input[0]}.f32"
            t_new = numpy_helper.from_array(val.astype(np.float32), name)
            graph.initializer.append(t_new)
            init[name] = t_new
            cast_map[key] = name
        for other in graph.node:
            for i, inp in enumerate(other.input):
                if inp == node.output[0]:
                    other.input[i] = cast_map[key]
        stats["dense_castfold"] += 1

    producer = {o: n for n in graph.node for o in n.output}
    users = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            users[inp].append(n)
    outputs = {o.name for o in graph.output}

    def scalar(name: str) -> float | None:
        t = init.get(name)
        if t is None or t.external_data or int(np.prod(t.dims or [1])) != 1:
            return None
        return float(load_init_f32(t).reshape(-1)[0])

    def sole(name: str) -> bool:
        return len(users.get(name, [])) == 1 and name not in outputs

    replace = {}
    for node in graph.node:
        if node.op_type == "Mul":
            a, b = node.input
            for sig_name, scale_name in ((a, b), (b, a)):
                sig = producer.get(sig_name)
                scale = scalar(scale_name)
                if sig is not None and sig.op_type == "Sigmoid" and scale is not None and sole(sig_name):
                    div = producer.get(sig.input[0])
                    if div is not None and div.op_type == "Div" and sole(sig.input[0]):
                        d = scalar(div.input[1])
                        if d is not None:
                            replace[id(node)] = helper.make_node(
                                "DivSigmoidMul",
                                [div.input[0]],
                                list(node.output),
                                name=node.name + "/fused",
                                domain="qwen.fused",
                                div=d,
                                mul=scale,
                            )
                            stats["dense_DivSigmoidMul"] += 1
                            break
            else:
                for swish_name, other in ((a, b), (b, a)):
                    sw = producer.get(swish_name)
                    if sw is not None and sw.op_type == "Swish" and sole(swish_name) and other not in init:
                        replace[id(node)] = helper.make_node(
                            "SwishMul",
                            [sw.input[0], other],
                            list(node.output),
                            name=node.name + "/fused",
                            domain="qwen.fused",
                        )
                        stats["dense_SwishMul"] += 1
                        break
        elif node.op_type == "Add":
            for mul_name, routed in ((node.input[0], node.input[1]), (node.input[1], node.input[0])):
                mul = producer.get(mul_name)
                if mul is None or mul.op_type != "Mul" or not sole(mul_name):
                    continue
                for sig_name, shared in ((mul.input[0], mul.input[1]), (mul.input[1], mul.input[0])):
                    sig = producer.get(sig_name)
                    if sig is None or sig.op_type != "Sigmoid" or not sole(sig_name):
                        continue
                    mm = producer.get(sig.input[0])
                    if mm is None or mm.op_type != "MatMul" or not sole(sig.input[0]):
                        continue
                    w = init.get(mm.input[1])
                    if w is None or list(w.dims)[1:] != [1] or w.data_type != TensorProto.BFLOAT16:
                        continue
                    replace[id(node)] = helper.make_node(
                        "SharedGateMulAdd",
                        [mm.input[0], mm.input[1], shared, routed],
                        list(node.output),
                        name=node.name + "/fused",
                        domain="qwen.fused",
                    )
                    stats["dense_SharedGateMulAdd"] += 1
                    break
                else:
                    continue
                break

    if replace:
        rebuilt = [replace.get(id(n), n) for n in graph.node]
        del graph.node[:]
        graph.node.extend(rebuilt)
        if all(o.domain != "qwen.fused" for o in dense_model.opset_import):
            entry = dense_model.opset_import.add()
            entry.domain, entry.version = "qwen.fused", 1

    while True:
        u = defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                u[inp].append(n)
        dead = [n for n in graph.node if all(o not in u and o not in outputs for o in n.output)]
        if not dead:
            break
        dead_ids = {id(n) for n in dead}
        kept = [n for n in graph.node if id(n) not in dead_ids]
        del graph.node[:]
        graph.node.extend(kept)


def apply_weight_and_dense_transforms(
    source: Path,
    out: Path,
    target_model,
    head_r2_model,
    dense_model,
    head_kv_model,
    weight_quant: str,
    mtp_moe_quant: str,
    lm_head_quant: str = "none",
    hc_quant: str = "none",
    share_dense: bool = False,
    fuse_dense_ops: bool = False,
    keep_bf16_router_shared: bool = False,
    keep_bf16_shared_down: bool = False,
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
    stats = collections.Counter()

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

    def tile_hc_bf16(w_bf16: torch.Tensor, kind: int) -> torch.Tensor:
        if kind == 0:
            return w_bf16.view(8, 10, 128, 20, 2, 8).permute(0, 3, 1, 2, 4, 5).contiguous().view(160, 20480)
        return w_bf16.view(10, 32, 4, 160, 2, 8).permute(3, 0, 1, 2, 4, 5).contiguous().view(160, 20480)

    def untile_hc_bf16(w_tiled: torch.Tensor, kind: int) -> torch.Tensor:
        if kind == 0:
            return w_tiled.view(8, 20, 10, 128, 2, 8).permute(0, 2, 3, 1, 4, 5).contiguous().view(10240, 320)
        return w_tiled.view(160, 10, 32, 4, 2, 8).permute(1, 2, 3, 0, 4, 5).contiguous().view(320, 10240)

    def tile_hc_fp8(w_fp8_u8: torch.Tensor, kind: int) -> torch.Tensor:
        if kind == 0:
            return w_fp8_u8.view(8, 10, 128, 20, 2, 8).permute(0, 3, 1, 2, 4, 5).contiguous().view(160, 20480)
        return w_fp8_u8.view(10, 32, 4, 160, 2, 8).permute(3, 0, 1, 2, 4, 5).contiguous().view(160, 20480)

    hc_src_path = source / "target" / "hc_tiled.data"
    hc_dst_path = out / "target" / "hc_tiled.data"
    if hc_src_path.exists():
        shutil.copy2(hc_src_path, hc_dst_path)
    hc_tile_cache: dict = {}

    def append_hc_tiled_bf16(base_dir: Path, tensor, kind: int) -> tuple[int, int]:
        src_key, w_bf16 = load_src_bf16(base_dir, tensor)
        ck = (*src_key, f"hc_tile_{kind}")
        if ck in hc_tile_cache:
            return hc_tile_cache[ck]
        tiled = tile_hc_bf16(w_bf16, kind)
        raw = tiled.view(torch.uint16).cpu().numpy().tobytes()
        with hc_dst_path.open("ab") as hf:
            pos = hf.tell()
            pad = (-pos) % 4096
            if pad:
                hf.write(b"\0" * pad)
                pos += pad
            hf.write(raw)
        hc_tile_cache[ck] = (pos, len(raw))
        return pos, len(raw)

    for mlabel, model, base_dir in (
        ("target", target_model, source / "target"),
        ("head_r2", head_r2_model, source / "head_r2"),
    ):
        inits = {t.name: t for t in model.graph.initializer}
        nodes = list(model.graph.node)
        rebuilt = []
        skip_idx = set()
        for idx, node in enumerate(nodes):
            if idx in skip_idx:
                continue
            if (
                node.op_type in ("HcDown", "HcPostDown")
                and idx + 1 < len(nodes)
                and nodes[idx + 1].op_type == "HcUpPre"
            ):
                up_node = nodes[idx + 1]
                skip_idx.add(idx + 1)
                down_attrs = {a.name: a for a in node.attribute}
                up_attrs = {a.name: a for a in up_node.attribute}
                is_post = node.op_type == "HcPostDown"
                wd_name = node.input[4 if is_post else 2]
                wi_name = node.input[5 if is_post else 3]
                wu_name = up_node.input[2]
                wd_off, wd_len = append_hc_tiled_bf16(base_dir, inits[wd_name], 0)
                wu_off, wu_len = append_hc_tiled_bf16(base_dir, inits[wu_name], 1)
                wd_tiled_name = f"{wd_name}.hc_tiled_down"
                wu_tiled_name = f"{wu_name}.hc_tiled_up"
                set_external(inits[wd_name], "hc_tiled.data", wd_off, wd_len, TensorProto.BFLOAT16, [160, 20480])
                inits[wd_name].name = wd_tiled_name
                inits[wd_tiled_name] = inits.pop(wd_name)
                set_external(inits[wu_name], "hc_tiled.data", wu_off, wu_len, TensorProto.BFLOAT16, [160, 20480])
                inits[wu_name].name = wu_tiled_name
                inits[wu_tiled_name] = inits.pop(wu_name)
                prefix_name = node.name.rsplit("/", 1)[0]
                if is_post:
                    merged_node = helper.make_node(
                        "HcPostBlock",
                        [node.input[0], node.input[1], node.input[2], node.input[3], wd_tiled_name, wi_name, wu_tiled_name],
                        [node.output[0], up_node.output[0], up_node.output[1], node.output[1]],
                        name=f"{prefix_name}/HcPostBlock",
                        domain="qwen.fused_hc",
                    )
                else:
                    merged_node = helper.make_node(
                        "HcBlock",
                        [node.input[0], node.input[1], wd_tiled_name, wi_name, wu_tiled_name],
                        [up_node.output[0], up_node.output[1], node.output[0]],
                        name=f"{prefix_name}/HcBlock",
                        domain="qwen.fused_hc",
                    )
                for a in (
                    down_attrs["epsilon"],
                    up_attrs["gate_div"],
                    up_attrs["gate_mul"],
                    up_attrs["reduction_scale"],
                    down_attrs["silu_alpha"],
                ):
                    merged_node.attribute.append(a)
                rebuilt.append(merged_node)
                stats[f"{mlabel}_head_hc_merged"] += 1
                continue

            if (
                mlabel == "target"
                and node.op_type == "Gather"
                and len(node.input) >= 2
                and node.input[1] == "logits_indices"
                and idx + 1 < len(nodes)
                and nodes[idx + 1].op_type == "MatMul"
                and nodes[idx + 1].input[0] == node.output[0]
            ):
                nodes[idx + 1].input[0] = node.input[0]
                stats["target_bypass_logits_gather"] += 1
                continue

            if node.op_type == "SkinnyGemm" and node.name in ("head.skinny_gemm/3.0", "skinny_gemm/3.0"):
                if all(a.name != "trigger" for a in node.attribute):
                    node.attribute.append(helper.make_attribute("trigger", 1))
                    stats[f"{mlabel}_gemm_triggers_added"] += 1

            if node.op_type == "MatMul" and node.input[1] in ("lm_head.weight_t", "head.lm_head.weight_t"):
                is_head_lm = (mlabel == "head_r2") or (node.input[1] == "head.lm_head.weight_t")
                if lm_head_quant == "fp8" or (lm_head_quant == "head_fp8" and is_head_lm):
                    node = helper.make_node(
                        "SkinnyGemm",
                        list(node.input),
                        list(node.output),
                        name=f"{node.name}/SkinnyGemm",
                        domain="qwen.fused_gemm",
                        seg_outs=[1],
                        widths=[248320],
                    )
                    stats[f"{mlabel}_lm_head_skinny"] += 1

            rebuilt.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)

        # Fuse final hyper_connection_mixer chains into HcPostBlock (in head.mtp / mtp) and HcBlock (in target model)
        dummy_inject = next(
            t.name for t in model.graph.initializer if list(t.dims) == [10240, 4] and t.data_type == TensorProto.BFLOAT16
        )
        nodes_list = list(model.graph.node)
        replace_idx, remove_idx = {}, set()
        for idx in range(len(nodes_list) - 5):
            # Case A: 7-node chain [HyperConnectionPostMix -> BranchwiseRMSNorm -> SkinnyGemm -> ScaledSiLU -> SkinnyGemm -> Sigmoid -> HyperConnectionPreMix]
            # (Used for head.mtp/hyper_connection_mixer in target and mtp/hyper_connection_mixer in head_r2)
            if idx + 6 < len(nodes_list):
                n_post, n_norm, n_down, n_silu, n_up, n_sig, n_pre = nodes_list[idx : idx + 7]
                if (
                    n_post.op_type == "HyperConnectionPostMix"
                    and n_norm.op_type == "BranchwiseRMSNorm"
                    and n_down.op_type == "SkinnyGemm"
                    and n_silu.op_type == "ScaledSiLU"
                    and n_up.op_type == "SkinnyGemm"
                    and n_sig.op_type == "Sigmoid"
                    and n_pre.op_type == "HyperConnectionPreMix"
                    and n_norm.input[0] == n_post.output[0]
                    and n_down.input[0] == n_norm.output[0]
                    and n_silu.input[0] == n_down.output[0]
                    and n_up.input[0] == n_silu.output[0]
                    and n_sig.input[0] == n_up.output[0]
                    and n_pre.input[0] == n_norm.output[0]
                    and n_pre.input[1] == n_sig.output[0]
                    and not n_norm.name.startswith("model/hyper_connection_mixer/")
                ):
                    norm_w = n_norm.input[1]
                    wd_name, wu_name = n_down.input[1], n_up.input[1]
                    wd_off, wd_len = append_hc_tiled_bf16(base_dir, inits[wd_name], 0)
                    wu_off, wu_len = append_hc_tiled_bf16(base_dir, inits[wu_name], 1)
                    wd_tiled_name = f"{wd_name}.hc_tiled_down"
                    wu_tiled_name = f"{wu_name}.hc_tiled_up"
                    set_external(inits[wd_name], "hc_tiled.data", wd_off, wd_len, TensorProto.BFLOAT16, [160, 20480])
                    inits[wd_name].name = wd_tiled_name
                    inits[wd_tiled_name] = inits.pop(wd_name)
                    set_external(inits[wu_name], "hc_tiled.data", wu_off, wu_len, TensorProto.BFLOAT16, [160, 20480])
                    inits[wu_name].name = wu_tiled_name
                    inits[wu_tiled_name] = inits.pop(wu_name)
                    prefix = n_norm.name.rsplit("/hc_norm", 1)[0]
                    ins = list(n_post.input[:3]) + [norm_w, wd_tiled_name, dummy_inject, wu_tiled_name]
                    outs = [n_post.output[0], n_pre.output[0], f"{prefix}.dummy_gate", f"{prefix}.hc_scratch"]
                    replace_idx[idx] = helper.make_node(
                        "HcPostBlock",
                        ins,
                        outs,
                        name=f"{prefix}/HcPostBlock",
                        domain="qwen.fused_hc",
                        epsilon=9.999999974752427e-07,
                        silu_alpha=0.25,
                        reduction_scale=0.25,
                        gate_div=4.0,
                        gate_mul=2.0,
                    )
                    for k in range(idx, idx + 7):
                        remove_idx.add(k)
                    stats[f"{mlabel}_mixer_hc_merged"] += 1
                    continue

            # Case B: 6-node chain [BranchwiseRMSNorm -> SkinnyGemm -> ScaledSiLU -> SkinnyGemm -> Sigmoid -> HyperConnectionPreMix]
            # (Used for model/hyper_connection_mixer in target so Slice_last2 can sit between HyperConnectionPostMix and HcBlock)
            if idx not in remove_idx:
                n_norm, n_down, n_silu, n_up, n_sig, n_pre = nodes_list[idx : idx + 6]
                if (
                    n_norm.op_type == "BranchwiseRMSNorm"
                    and n_down.op_type == "SkinnyGemm"
                    and n_silu.op_type == "ScaledSiLU"
                    and n_up.op_type == "SkinnyGemm"
                    and n_sig.op_type == "Sigmoid"
                    and n_pre.op_type == "HyperConnectionPreMix"
                    and n_down.input[0] == n_norm.output[0]
                    and n_silu.input[0] == n_down.output[0]
                    and n_up.input[0] == n_silu.output[0]
                    and n_sig.input[0] == n_up.output[0]
                    and n_pre.input[0] == n_norm.output[0]
                    and n_pre.input[1] == n_sig.output[0]
                    and n_norm.name.startswith("model/hyper_connection_mixer/")
                ):
                    norm_w = n_norm.input[1]
                    wd_name, wu_name = n_down.input[1], n_up.input[1]
                    wd_off, wd_len = append_hc_tiled_bf16(base_dir, inits[wd_name], 0)
                    wu_off, wu_len = append_hc_tiled_bf16(base_dir, inits[wu_name], 1)
                    wd_tiled_name = f"{wd_name}.hc_tiled_down"
                    wu_tiled_name = f"{wu_name}.hc_tiled_up"
                    set_external(inits[wd_name], "hc_tiled.data", wd_off, wd_len, TensorProto.BFLOAT16, [160, 20480])
                    inits[wd_name].name = wd_tiled_name
                    inits[wd_tiled_name] = inits.pop(wd_name)
                    set_external(inits[wu_name], "hc_tiled.data", wu_off, wu_len, TensorProto.BFLOAT16, [160, 20480])
                    inits[wu_name].name = wu_tiled_name
                    inits[wu_tiled_name] = inits.pop(wu_name)
                    prefix = n_norm.name.rsplit("/hc_norm", 1)[0]
                    ins = [n_norm.input[0], norm_w, wd_tiled_name, dummy_inject, wu_tiled_name]
                    outs = [n_pre.output[0], f"{prefix}.dummy_gate", f"{prefix}.hc_scratch"]
                    replace_idx[idx] = helper.make_node(
                        "HcBlock",
                        ins,
                        outs,
                        name=f"{prefix}/HcBlock",
                        domain="qwen.fused_hc",
                        epsilon=9.999999974752427e-07,
                        silu_alpha=0.25,
                        reduction_scale=0.25,
                        gate_div=4.0,
                        gate_mul=2.0,
                    )
                    for k in range(idx, idx + 6):
                        remove_idx.add(k)
                    stats[f"{mlabel}_mixer_hc_merged"] += 1

        if replace_idx:
            rebuilt_mixer = []
            for idx, n in enumerate(nodes_list):
                if idx in replace_idx:
                    rebuilt_mixer.append(replace_idx[idx])
                elif idx not in remove_idx:
                    rebuilt_mixer.append(n)
            del model.graph.node[:]
            model.graph.node.extend(rebuilt_mixer)
            for a, b in zip(model.graph.node, model.graph.node[1:]):
                if (
                    a.domain == "qwen.fused_gemm"
                    and a.op_type in ("SkinnyGemm", "SharedExpertDown")
                    and b.domain == "qwen.fused_hc"
                    and b.op_type in ("HcBlock", "HcPostBlock")
                    and not any(attr.name == "trigger" for attr in a.attribute)
                ):
                    a.attribute.append(helper.make_attribute("trigger", 1))
                    stats[f"{mlabel}_gemm_triggers_added"] += 1

        if mlabel == "target":
            # Simplify 23-node 3D M-RoPE (t=h=w=position_ids in 1D text prefill/decode) to 2 Gather(axis=0) nodes
            rot_nodes = [n for n in model.graph.node if n.name.startswith("model/rotary_emb/")]
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
                for n in model.graph.node:
                    if n is first_rot:
                        rebuilt_rot.extend([g_cos, g_sin])
                    elif id(n) not in rot_set:
                        rebuilt_rot.append(n)
                del model.graph.node[:]
                model.graph.node.extend(rebuilt_rot)
                stats["target_simplify_1d_rope"] += 1

            # Bypass 12-node attention_mask != 0 Where_node_85 chain (attention_mask is fixed all-ones)
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
            if any(n.name == "model/Where_node_85" and list(n.output) == ["v_model.Where_85"] for n in model.graph.node):
                rebuilt_mask = []
                for n in model.graph.node:
                    if n.name in mask_chain_names:
                        continue
                    for i_idx, inp in enumerate(n.input):
                        if inp == "v_model.Where_85":
                            n.input[i_idx] = "input_ids"
                    rebuilt_mask.append(n)
                del model.graph.node[:]
                model.graph.node.extend(rebuilt_mask)
                stats["target_bypass_mask_where"] += 1

    hc_scales_by_name: dict[str, list[float]] = {}
    if hc_quant == "fp8":
        hc_fp8_path = out / "target" / "hc_tiled_fp8.data"
        hc_fp8_file = hc_fp8_path.open("wb")
        hc_fp8_pos = 0
        hc_fp8_cache: dict = {}

        def quantize_hc_tensor(base_dir: Path, tensor, kind: int) -> dict:
            nonlocal hc_fp8_pos
            ext = {e.key: e.value for e in tensor.external_data}
            p = (out / "target" / "hc_tiled.data") if ext["location"] == "hc_tiled.data" else (base_dir / ext["location"])
            st = os.stat(p)
            off = int(ext.get("offset", 0))
            length = int(ext.get("length", 0)) or (st.st_size - off)
            ck = (st.st_ino, off, length, kind)
            if ck in hc_fp8_cache:
                return hc_fp8_cache[ck]
            with p.open("rb") as f:
                f.seek(off)
                buf = f.read(length)
            w_tiled = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(160, 20480).to(device)
            w_untiled = untile_hc_bf16(w_tiled, kind).float()
            amax = w_untiled.abs().amax(dim=0).clamp(min=1e-12)
            sc = (amax / 448.0).to(torch.float32)
            wq = (w_untiled / sc[None, :]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
            wq_tiled = tile_hc_fp8(wq, kind)
            raw = wq_tiled.cpu().numpy().tobytes()
            pad = (-hc_fp8_pos) % 4096
            if pad:
                hc_fp8_file.write(b"\0" * pad)
                hc_fp8_pos += pad
            new_off = hc_fp8_pos
            hc_fp8_file.write(raw)
            hc_fp8_pos += len(raw)
            info = {
                "offset": new_off,
                "length": len(raw),
                "dtype": TensorProto.FLOAT8E4M3FN,
                "dims": [160, 20480],
                "scale": [float(x) for x in sc.cpu().numpy()],
            }
            hc_fp8_cache[ck] = info
            return info

        for mlabel, model, base_dir in (
            ("target", target_model, source / "target"),
            ("head_r2", head_r2_model, source / "head_r2"),
        ):
            inits = {t.name: t for t in model.graph.initializer}
            for node in model.graph.node:
                if node.op_type not in ("HcBlock", "HcPostBlock"):
                    continue
                is_post = node.op_type == "HcPostBlock"
                wd_name = node.input[4 if is_post else 2]
                wu_name = node.input[6 if is_post else 4]
                wd_info = quantize_hc_tensor(base_dir, inits[wd_name], 0)
                wu_info = quantize_hc_tensor(base_dir, inits[wu_name], 1)
                set_external(inits[wd_name], "hc_tiled.data", wd_info["offset"], wd_info["length"], wd_info["dtype"], wd_info["dims"])
                set_external(inits[wu_name], "hc_tiled.data", wu_info["offset"], wu_info["length"], wu_info["dtype"], wu_info["dims"])
                keep_attrs = [a for a in node.attribute if a.name not in ("wd_scale", "wu_scale")]
                del node.attribute[:]
                node.attribute.extend(keep_attrs)
                node.attribute.append(helper.make_attribute("wd_scale", wd_info["scale"]))
                node.attribute.append(helper.make_attribute("wu_scale", wu_info["scale"]))
                hc_scales_by_name[wd_name] = wd_info["scale"]
                hc_scales_by_name[wu_name] = wu_info["scale"]
                stats[f"{mlabel}_hc_fp8_blocks"] += 1

        hc_fp8_file.close()
        hc_fp8_path.replace(hc_dst_path)

    if hc_dst_path.exists():
        link_file(hc_dst_path, out / "head_r2" / "hc_tiled.data", False)

    def quantize_2d(base_dir: Path, tensor, qmode: str) -> dict:
        src_key, w_bf16 = load_src_bf16(base_dir, tensor)
        cache_key = (*src_key, qmode)
        if cache_key in quant_cache:
            return quant_cache[cache_key]
        k_dim, n_dim = int(w_bf16.shape[0]), int(w_bf16.shape[1])
        wf = w_bf16.float()
        if qmode == "bf16":
            raw = w_bf16.contiguous().view(torch.uint16).cpu().numpy().tobytes()
            off, length = write_quant_blob(raw)
            info = {
                "offset": off,
                "length": length,
                "dtype": TensorProto.BFLOAT16,
                "dims": [k_dim, n_dim],
                "n_cols": n_dim,
                "w_scale": [],
            }
        elif qmode == "fp8":
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
        if node.op_type == "SkinnyGemm" and node.input[1] in ("lm_head.weight_t", "head.lm_head.weight_t"):
            return "fp8"
        if node.op_type == "SharedExpertDown" and (keep_bf16_router_shared or keep_bf16_shared_down):
            return "bf16"
        if node.op_type == "SkinnyGemm" and len(node.output) == 3 and keep_bf16_router_shared:
            return "bf16"
        if weight_quant == "none":
            return None
        if weight_quant == "fp8":
            return "fp8"
        if weight_quant == "nvfp4":
            return "nvfp4"
        if weight_quant == "fp8_attn_nvfp4_shared":
            return "nvfp4" if node.op_type == "SharedExpertDown" else "fp8"
        raise ValueError(f"Unsupported weight_quant={weight_quant}")

    # Track quantized info by initializer name in target for dense and head_kv sharing
    target_quant_by_name: dict[str, dict] = {}

    for mlabel, model, base_dir in (
        ("target", target_model, source / "target"),
        ("head_r2", head_r2_model, source / "head_r2"),
    ):
        inits = {t.name: t for t in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type in ("SkinnyGemm", "SharedExpertDown"):
                qmode = mode_for_node(node)
                if qmode is None:
                    continue
                w_idx = 1 if node.op_type == "SkinnyGemm" else 2
                w_name = node.input[w_idx]
                t = inits[w_name]
                info = quantize_2d(base_dir, t, qmode)
                set_external(t, "quant_weights.data", info["offset"], info["length"], info["dtype"], info["dims"])
                keep_attrs = [a for a in node.attribute if a.name != "w_scale"]
                del node.attribute[:]
                node.attribute.extend(keep_attrs)
                if info["w_scale"]:
                    node.attribute.append(helper.make_attribute("w_scale", info["w_scale"]))
                if mlabel == "target" and info["w_scale"]:
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
        link_file(quant_path, out / "head_kv" / "quant_weights.data", False)
    else:
        quant_path.unlink(missing_ok=True)

    if (out / "target" / "attn_consts.data").exists():
        if (out / "target" / "head_data").exists():
            link_file(out / "target" / "attn_consts.data", out / "target" / "head_data" / "attn_consts.data", False)
        link_file(out / "target" / "attn_consts.data", out / "head_r2" / "attn_consts.data", False)

    gen_data_src = out / "target" / "head_data" / "model.onnx.generated.data"
    if gen_data_src.exists():
        link_file(gen_data_src, out / "head_r2" / "model.onnx.generated.data", False)
        link_file(gen_data_src, out / "dense" / "head_data" / "model.onnx.generated.data", False)
        link_file(gen_data_src, out / "head_kv" / "head_data" / "model.onnx.generated.data", False)
        for m in (target_model, dense_model, head_kv_model):
            if m is None:
                continue
            for t in m.graph.initializer:
                if t.name in ("model.rotary_emb.cos_cache", "mtp.rotary_emb.cos_cache"):
                    set_external(t, "head_data/model.onnx.generated.data", 0, 16777216, TensorProto.BFLOAT16, [262144, 32])
                    stats["dedup_rotary_cache"] += 1
                elif t.name in ("model.rotary_emb.sin_cache", "mtp.rotary_emb.sin_cache"):
                    set_external(
                        t, "head_data/model.onnx.generated.data", 16777216, 16777216, TensorProto.BFLOAT16, [262144, 32]
                    )
                    stats["dedup_rotary_cache"] += 1

    # Rewrite dense/model.onnx to share merged_proj, quantized weights, and hc_tiled.data with target/model.onnx
    if share_dense and dense_model is not None:
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

        # 2. Remaining standalone MatMuls in dense: lm_head SkinnyGemm, quantized in target, OR tiled in hc_tiled.data
        for index, node in enumerate(dense_model.graph.node):
            if index in replaced_indices or node.op_type != "MatMul":
                continue
            w_name = node.input[1]
            if w_name == "lm_head.weight_t" and lm_head_quant == "fp8" and w_name in target_quant_by_name:
                qinfo = target_quant_by_name[w_name]
                set_external(
                    inits_d[w_name],
                    "quant_weights.data",
                    qinfo["offset"],
                    qinfo["length"],
                    qinfo["dtype"],
                    qinfo["dims"],
                )
                sg_node = helper.make_node(
                    "SkinnyGemm",
                    list(node.input),
                    list(node.output),
                    name=f"{node.name}/SkinnyGemm",
                    domain="qwen.fused_gemm",
                    seg_outs=[1],
                    widths=[248320],
                    w_scale=qinfo["w_scale"],
                )
                inserted_at[index].append(sg_node)
                replaced_indices.add(index)
                stats["dense_lm_head_skinny"] += 1
            elif w_name in target_quant_by_name:
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
                        untile_kwargs = {"kind": kind}
                        if tiled_name in hc_scales_by_name:
                            untile_kwargs["w_scale"] = hc_scales_by_name[tiled_name]
                        untile_node = helper.make_node(
                            "UntileHcBf16",
                            [tiled_name, node.input[0]],
                            [untiled_out],
                            name=f"dense_hc/{index}/Untile",
                            domain="qwen.fused_gemm",
                            **untile_kwargs,
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

        if fuse_dense_ops:
            _fuse_dense_pointwise_ops(dense_model, source / "dense", stats)

        used_inputs = {inp for n in dense_model.graph.node for inp in n.input if inp}
        kept_inits = [t for t in dense_model.graph.initializer if t.name in used_inputs]
        del dense_model.graph.initializer[:]
        dense_model.graph.initializer.extend(kept_inits)

        if all(o.domain != "qwen.fused_gemm" for o in dense_model.opset_import):
            opset = dense_model.opset_import.add()
            opset.domain, opset.version = "qwen.fused_gemm", 1

        if head_kv_model is not None:
            link_file(out / "target" / "hc_tiled.data", out / "head_kv" / "hc_tiled.data", False)
            inits_hk = {t.name: t for t in head_kv_model.graph.initializer}
            hk_inserted = collections.defaultdict(list)
            merged0_slices = {
                "mtp.layers.0.self_attn.indexer.index_qk_proj.weight_t": (0, 640),
                "mtp.layers.0.self_attn.k_proj.weight_t": (12928, 512),
                "mtp.layers.0.self_attn.v_proj.weight_t": (13440, 512),
            }
            for index, node in enumerate(head_kv_model.graph.node):
                if node.op_type != "MatMul":
                    continue
                w_name = node.input[1]
                head_target_name = f"head.{w_name}"
                if head_target_name in target_quant_by_name:
                    qinfo = target_quant_by_name[head_target_name]
                    set_external(
                        inits_hk[w_name],
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
                        name=f"hk_deq/{index}/Dequant",
                        domain="qwen.fused_gemm",
                        n_cols=qinfo["n_cols"],
                        w_scale=qinfo["w_scale"],
                    )
                    hk_inserted[index].append(deq_node)
                    node.input[1] = w_bf16_name
                    stats["head_kv_dequant_matmuls"] += 1
                elif w_name in merged0_slices and "head.merged_proj.0.weight_t" in target_quant_by_name:
                    col_offset, n_cols = merged0_slices[w_name]
                    qinfo = target_quant_by_name["head.merged_proj.0.weight_t"]
                    set_external(
                        inits_hk[w_name],
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
                        name=f"hk_deq/{index}/Dequant",
                        domain="qwen.fused_gemm",
                        col_offset=col_offset,
                        n_cols=n_cols,
                        w_scale=qinfo["w_scale"][col_offset : col_offset + n_cols],
                    )
                    hk_inserted[index].append(deq_node)
                    node.input[1] = w_bf16_name
                    stats["head_kv_dequant_matmuls"] += 1
                else:
                    for kind, suffix in ((0, "hc_tiled_down"), (1, "hc_tiled_up")):
                        tiled_target_name = f"head.{w_name}.{suffix}"
                        if tiled_target_name in inits_t:
                            tiled_hk_name = f"{w_name}.{suffix}"
                            t_tiled = TensorProto()
                            t_tiled.CopyFrom(inits_t[tiled_target_name])
                            t_tiled.name = tiled_hk_name
                            head_kv_model.graph.initializer.append(t_tiled)
                            untiled_out = f"{w_name}.untiled_bf16"
                            untile_kwargs = {"kind": kind}
                            if tiled_target_name in hc_scales_by_name:
                                untile_kwargs["w_scale"] = hc_scales_by_name[tiled_target_name]
                            untile_node = helper.make_node(
                                "UntileHcBf16",
                                [tiled_hk_name, node.input[0]],
                                [untiled_out],
                                name=f"hk_hc/{index}/Untile",
                                domain="qwen.fused_gemm",
                                **untile_kwargs,
                            )
                            hk_inserted[index].append(untile_node)
                            node.input[1] = untiled_out
                            stats[f"head_kv_untile_{suffix}"] += 1
                            break

            hk_rebuilt = []
            for index, node in enumerate(head_kv_model.graph.node):
                if index in hk_inserted:
                    hk_rebuilt.extend(hk_inserted[index])
                hk_rebuilt.append(node)
            del head_kv_model.graph.node[:]
            head_kv_model.graph.node.extend(hk_rebuilt)
            hk_used = {inp for n in head_kv_model.graph.node for inp in n.input if inp}
            hk_kept = [t for t in head_kv_model.graph.initializer if t.name in hk_used]
            del head_kv_model.graph.initializer[:]
            head_kv_model.graph.initializer.extend(hk_kept)
            if all(o.domain != "qwen.fused_gemm" for o in head_kv_model.opset_import):
                opset = head_kv_model.opset_import.add()
                opset.domain, opset.version = "qwen.fused_gemm", 1

    return {"quant_bytes": quant_pos, "quant_gib": quant_pos / 2**30, "stats": dict(stats)}


def copy_file_slice(src_fd: int, src_off: int, dst_fd: int, dst_off: int, length: int) -> None:
    rem = length
    r_off = src_off
    w_off = dst_off
    while rem > 0:
        chunk = min(rem, 1 << 26)
        try:
            copied = os.copy_file_range(src_fd, dst_fd, chunk, offset_src=r_off, offset_dst=w_off)
        except OSError:
            copied = 0
        if copied <= 0:
            buf = os.pread(src_fd, chunk, r_off)
            if not buf:
                raise RuntimeError(f"short read at offset {r_off} (remaining {rem})")
            os.pwrite(dst_fd, buf, w_off)
            copied = len(buf)
        r_off += copied
        w_off += copied
        rem -= copied


def clean_and_pack_package(source: Path, out: Path, copy: bool = False, embed_quant: str = "int8") -> dict:
    """Clean up unreferenced safetensors/dense.data/merged.data/generated.data, compact expert scales,
    split model.embed_tokens.weight into a standard CPU embedding.onnx + embedding.onnx.data model
    (INT8 per-row or BF16), unify target and MTP graphs so the same ONNX graph serves both prefill
    and decode, and place all ONNX models (model.onnx, model_s2..s4.onnx, mtp.onnx, embedding.onnx)
    in the root package directory."""
    import numpy as np  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    session_names = ("target", "head_r2")
    models = {}
    src_dirs = {}
    for sname in session_names:
        if sname == "target":
            cands = [out / "target/model.onnx", out / "model.onnx", source / "target/model.onnx", source / "model.onnx"]
        else:
            cands = [out / "head_r2/model.onnx", out / "mtp.onnx", source / "head_r2/model.onnx", source / "mtp.onnx"]
        src_m = next(p for p in cands if p.exists())
        src_dirs[sname] = src_m.parent
        models[sname] = onnx.load(str(src_m), load_external_data=False)

    def set_ext(tensor, location: str, offset: int, length: int) -> None:
        tensor.ClearField("raw_data")
        del tensor.external_data[:]
        tensor.data_location = TensorProto.EXTERNAL
        for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
            e = tensor.external_data.add()
            e.key, e.value = k, v

    (out / "source_weights").mkdir(parents=True, exist_ok=True)
    for obsolete_file in ("qsa_consts.txt", "qsa_consts.bin", "qsa_consts.data"):
        (out / obsolete_file).unlink(missing_ok=True)

    # 1. Extract model.embed_tokens.weight into standard embedding.onnx + embedding.onnx.data (Gather on CPU EP)
    embed_names = {"model.embed_tokens.weight", "head.mtp.embed_tokens.weight", "mtp.embed_tokens.weight"}
    embed_src_path = None
    embed_src_off = 0
    embed_len = 0
    embed_dtype = TensorProto.BFLOAT16
    embed_dims = [248320, 2560]
    for sname in ("target", "head_r2"):
        for t in models[sname].graph.initializer:
            if t.name in embed_names:
                ext = {e.key: e.value for e in t.external_data}
                for base_d in (src_dirs[sname], source / sname, source / "target", source, out):
                    cand_p = base_d / ext["location"]
                    if cand_p.exists():
                        embed_src_path = cand_p
                        break
                embed_src_off = int(ext.get("offset", 0))
                embed_len = int(ext.get("length", 0))
                embed_dtype = t.data_type
                embed_dims = list(t.dims)
                break
        if embed_src_path is not None:
            break
    if embed_src_path is None or embed_len <= 0:
        raise RuntimeError("Failed to locate model.embed_tokens.weight in target/mtp model.onnx")

    dst_embed_data = out / "embedding.onnx.data"
    std_opset = next((o.version for o in models["target"].opset_import if o.domain in ("", "ai.onnx")), 21)

    if embed_quant == "int8":
        with open(embed_src_path, "rb") as fin:
            fin.seek(embed_src_off)
            raw_bf16 = fin.read(embed_len)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        w_bf16 = torch.frombuffer(bytearray(raw_bf16), dtype=torch.bfloat16).reshape(embed_dims).to(device)
        wf = w_bf16.float()
        amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scales_f32 = (amax / 127.0).to(torch.float32)
        q_i8 = torch.round(wf / scales_f32).clamp(-127.0, 127.0).to(torch.int8)
        q_bytes = q_i8.cpu().numpy().tobytes()
        s_bytes = scales_f32.cpu().numpy().tobytes()
        with open(dst_embed_data, "wb") as fout:
            fout.write(q_bytes)
            fout.write(s_bytes)
        embed_len = len(q_bytes) + len(s_bytes)

        embed_init = TensorProto()
        embed_init.name = "model.embed_tokens.weight"
        embed_init.data_type = TensorProto.INT8
        embed_init.dims.extend(embed_dims)
        set_ext(embed_init, "embedding.onnx.data", 0, len(q_bytes))

        scale_init = TensorProto()
        scale_init.name = "model.embed_tokens.scales"
        scale_init.data_type = TensorProto.FLOAT
        scale_init.dims.extend([embed_dims[0], 1])
        set_ext(scale_init, "embedding.onnx.data", len(q_bytes), len(s_bytes))

        emb_nodes = [
            helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["embed_q_i8"], name="/model/embed_tokens/GatherWeight"),
            helper.make_node("Gather", ["model.embed_tokens.scales", "input_ids"], ["embed_scale_f32"], name="/model/embed_tokens/GatherScale"),
            helper.make_node("Cast", ["embed_q_i8"], ["embed_q_f32"], name="/model/embed_tokens/CastWeight", to=TensorProto.FLOAT),
            helper.make_node("Mul", ["embed_q_f32", "embed_scale_f32"], ["embed_out_f32"], name="/model/embed_tokens/MulScale"),
            helper.make_node("Cast", ["embed_out_f32"], ["inputs_embeds"], name="/model/embed_tokens/CastOut", to=TensorProto.BFLOAT16),
        ]
        embedding_model = helper.make_model(
            helper.make_graph(
                emb_nodes,
                "cpu_embedding_int8",
                [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
                [helper.make_tensor_value_info("inputs_embeds", TensorProto.BFLOAT16, ["num_tokens", embed_dims[1]])],
                [embed_init, scale_init],
            ),
            opset_imports=[helper.make_opsetid("", std_opset)],
            ir_version=models["target"].ir_version,
        )
        onnx.save_model(embedding_model, str(out / "embedding.onnx"), save_as_external_data=False)

        for sname in ("target", "head_r2"):
            for t in models[sname].graph.initializer:
                if t.name in embed_names:
                    t.data_type = TensorProto.INT8
                    set_ext(t, "embedding.onnx.data", 0, len(q_bytes))
        embed_dtype_str = "INT8"
    else:
        if embed_src_path.resolve() != dst_embed_data.resolve():
            with open(embed_src_path, "rb") as fin, open(dst_embed_data, "wb") as fout:
                copy_file_slice(fin.fileno(), embed_src_off, fout.fileno(), 0, embed_len)

        embed_init = TensorProto()
        embed_init.name = "model.embed_tokens.weight"
        embed_init.data_type = embed_dtype
        embed_init.dims.extend(embed_dims)
        set_ext(embed_init, "embedding.onnx.data", 0, embed_len)
        gather_node = helper.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["inputs_embeds"],
            name="/model/embed_tokens/Gather",
        )
        embedding_model = helper.make_model(
            helper.make_graph(
                [gather_node],
                "cpu_embedding",
                [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
                [helper.make_tensor_value_info("inputs_embeds", embed_dtype, ["num_tokens", embed_dims[1]])],
                [embed_init],
            ),
            opset_imports=[helper.make_opsetid("", std_opset)],
            ir_version=models["target"].ir_version,
        )
        onnx.save_model(embedding_model, str(out / "embedding.onnx"), save_as_external_data=False)

        for sname in ("target", "head_r2"):
            for t in models[sname].graph.initializer:
                if t.name in embed_names:
                    set_ext(t, "embedding.onnx.data", 0, embed_len)
        embed_dtype_str = "BFLOAT16"

    t_graph = models["target"].graph
    gather_candidates = [
        n for n in t_graph.node if n.op_type == "Gather" and len(n.input) >= 2 and n.input[0] == "model.embed_tokens.weight"
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
                    "inputs_embeds", embed_dtype, ["batch_size", "sequence_length", embed_dims[1]]
                )
            )

    # Unify target/model.onnx for arbitrary query sequence_length (prefill + decode in one graph):
    # - Slice hidden_states[:, -2:, :] before final hyper_connection_mixer + lm_head
    # - Pass hidden_states -> fold.head_hidden through FoldMid2 (head_rows=2)
    t_nodes = []
    has_slice_last2 = any(n.name == "model/hyper_connection_mixer/Slice_last2" for n in t_graph.node)
    for n in t_graph.node:
        if n.name in (
            "model/hyper_connection_mixer/hc_norm/BranchwiseRMSNorm_node_5816",
            "model/hyper_connection_mixer/HcBlock",
        ) and not has_slice_last2:
            t_nodes.append(
                helper.make_node(
                    "Slice",
                    ["hidden_states", "const_1d_0", "const_1d_1", "const_1d_3"],
                    ["hidden_states.last2"],
                    name="model/hyper_connection_mixer/Slice_last2",
                )
            )
            n.input[0] = "hidden_states.last2"
        elif n.op_type == "FoldMid2":
            if len(n.input) == 10:
                n.input.append("hidden_states")
            if len(n.output) == 7:
                n.output.append("fold.head_hidden")
            keep_a = [a for a in n.attribute if a.name != "head_rows"]
            del n.attribute[:]
            n.attribute.extend(keep_a)
            n.attribute.append(helper.make_attribute("head_rows", 2))
        elif n.name == "head.mtp/pre_fc_norm_hidden/BranchwiseRMSNorm_node_15":
            n.input[0] = "fold.head_hidden"
        t_nodes.append(n)
    del t_graph.node[:]
    t_graph.node.extend(t_nodes)

    # Unify head_r2/model.onnx for arbitrary query sequence_length (prefill + decode in one graph):
    # - EmbedNorm dynamically produces capture.slot_ids, capture.seqlens_k, capture.cumulative_sequence_lengths
    # - Slice Mul_200, Reshape_23, Mul_117 to [-2:] before o_proj and MtpMoe
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
        elif n.name == "skinny_gemm/3.0":
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
                    ["v_mtp.layers.0.attn_hyper_connection.Mul_117", "slice_last2_starts", "slice_last2_ends", "slice_last2_axes"],
                    ["v_mtp.layers.0.attn_hyper_connection.Mul_117.last2"],
                    name="mtp/Slice_last2_mul117",
                )
            )
            n.input[0] = "v_mtp.layers.0.self_attn.Mul_200.last2"
        elif n.name == "mtp/layers.0/mlp_hyper_connection/HcPostBlock":
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
    for sname in session_names:
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

    # 2. Place shared quant_weights.data, hc_tiled.data, attn_consts.data, and experts.data in root directory
    for fname in ("quant_weights.data", "hc_tiled.data", "attn_consts.data"):
        cands = [out / fname, out / "target" / fname, source / fname, source / "target" / fname]
        src_f = next(p for p in cands if p.exists())
        if src_f.resolve() != (out / fname).resolve():
            link_file(src_f, out / fname, copy)

    for t in models["target"].graph.initializer:
        ext = {e.key: e.value for e in t.external_data}
        if ext.get("location") == "head_data/attn_consts.data":
            set_ext(t, "attn_consts.data", int(ext.get("offset", 0)), int(ext["length"]))

    exp_cands = [
        source / "source_weights/experts.data",
        source / "target/source_weights/experts.data",
        source / "target/_source_weights/experts.data",
    ]
    src_experts = next(p for p in exp_cands if p.exists())
    link_file(src_experts, out / "source_weights" / "experts.data", copy)

    # 3. Compact source_weights/model.onnx.experts.0..7.data (eliminating 56.25 GiB of dead unreferenced bytes)
    compacted_experts_bytes = 0
    original_experts_bytes = 0
    for idx in range(8):
        rel_exp = f"source_weights/model.onnx.experts.{idx}.data"
        spans = set()
        for t in models["target"].graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if ext.get("location") == rel_exp:
                spans.add((int(ext.get("offset", 0)), int(ext["length"])))
        sorted_spans = sorted(spans)
        src_cands = [
            source / rel_exp,
            source / "target" / rel_exp,
            source / "target" / f"_source_weights/model.onnx.experts.{idx}.data",
        ]
        src_exp = next(p for p in src_cands if p.exists())
        original_experts_bytes += src_exp.stat().st_size
        dst_exp = out / rel_exp
        span_remap = {}
        pos = 0
        with open(src_exp, "rb") as fin, open(dst_exp, "wb") as fout:
            in_fd, out_fd = fin.fileno(), fout.fileno()
            for s_off, s_len in sorted_spans:
                pad = (-pos) % 4096
                if pad:
                    os.pwrite(out_fd, b"\0" * pad, pos)
                    pos += pad
                copy_file_slice(in_fd, s_off, out_fd, pos, s_len)
                span_remap[(s_off, s_len)] = pos
                pos += s_len
        compacted_experts_bytes += pos
        for t in models["target"].graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if ext.get("location") == rel_exp:
                key = (int(ext.get("offset", 0)), int(ext["length"]))
                set_ext(t, rel_exp, span_remap[key], key[1])

    # 4. Pack all remaining BF16 initializers (from model-bf16-*.safetensors, model.onnx.generated.data)
    # into norm_and_rope.data (4096-byte aligned) and deduplicate across model.onnx and mtp.onnx.
    kept_locations = {
        "embedding.onnx.data",
        "quant_weights.data",
        "hc_tiled.data",
        "attn_consts.data",
        "source_weights/experts.data",
        *(f"source_weights/model.onnx.experts.{i}.data" for i in range(8)),
    }
    norm_rope_path = out / "norm_and_rope.data"
    norm_rope_pos = 0
    norm_rope_map: dict[tuple, int] = {}
    packed_tensors = 0
    with open(norm_rope_path, "wb") as nrf:
        nr_fd = nrf.fileno()
        for sname in session_names:
            for t in models[sname].graph.initializer:
                ext = {e.key: e.value for e in t.external_data}
                if not ext:
                    continue
                loc = ext["location"]
                if loc in kept_locations:
                    continue
                s_off = int(ext.get("offset", 0))
                s_len = int(ext["length"])
                dedup_key = (Path(loc).name, s_off, s_len, t.data_type, tuple(t.dims))
                if dedup_key not in norm_rope_map:
                    pad = (-norm_rope_pos) % 4096
                    if pad:
                        os.pwrite(nr_fd, b"\0" * pad, norm_rope_pos)
                        norm_rope_pos += pad
                    src_file = src_dirs[sname] / loc
                    if not src_file.exists():
                        src_file = source / sname / loc
                    if not src_file.exists():
                        src_file = source / "target" / "source_weights" / Path(loc).name
                    with open(src_file, "rb") as fin:
                        copy_file_slice(fin.fileno(), s_off, nr_fd, norm_rope_pos, s_len)
                    norm_rope_map[dedup_key] = norm_rope_pos
                    norm_rope_pos += s_len
                set_ext(t, "norm_and_rope.data", norm_rope_map[dedup_key], s_len)
                packed_tensors += 1

    # 5. Save cleaned unified ONNX models in the root directory, generate S=2..4 multi-draft QMoE graphs, and remove legacy subdirectories
    onnx.save_model(models["target"], str(out / "model.onnx"), save_as_external_data=False)
    link_file(out / "model.onnx", out / "model_s2.onnx", copy)
    from build_package_offload_v3 import build_multidraft_target_onnx  # noqa: PLC0415
    build_multidraft_target_onnx(out / "model_s2.onnx", out / "model_s3.onnx", verify_rows=3)
    build_multidraft_target_onnx(out / "model_s2.onnx", out / "model_s4.onnx", verify_rows=4)
    onnx.save_model(models["head_r2"], str(out / "mtp.onnx"), save_as_external_data=False)
    for obsolete_dir in ("target", "head_r2", "dense", "head_kv"):
        if (out / obsolete_dir).exists():
            shutil.rmtree(out / obsolete_dir)

    return {
        "cpu_embedding": {
            "model_file": "embedding.onnx",
            "data_file": "embedding.onnx.data",
            "operator": "Gather+DequantInt8" if embed_quant == "int8" else "Gather",
            "dtype": embed_dtype_str,
            "shape": embed_dims,
            "bytes": embed_len,
            "mib": embed_len / (1024**2),
            "gib": embed_len / (1024**3),
        },
        "norm_and_rope_data": {
            "file": "norm_and_rope.data",
            "unique_tensors": len(norm_rope_map),
            "total_rewritten_refs": packed_tensors,
            "bytes": norm_rope_pos,
            "mib": norm_rope_pos / (1024**2),
        },
        "compacted_expert_scales": {
            "files": [f"source_weights/model.onnx.experts.{i}.data" for i in range(8)],
            "original_bytes": original_experts_bytes,
            "compacted_bytes": compacted_experts_bytes,
            "saved_gib": (original_experts_bytes - compacted_experts_bytes) / (1024**3),
        },
        "removed_files": [
            "source_weights/model-bf16-00001.safetensors",
            "source_weights/model-bf16-00010.safetensors",
            "source_weights/model-bf16-00011.safetensors",
            "source_weights/model-bf16-00012.safetensors",
            "source_weights/dense.data",
            "target/*",
            "head_r2/*",
            "dense/*",
            "head_kv/*",
        ],
    }


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
        "--lm-head-quant",
        choices=("none", "fp8", "head_fp8"),
        default="none",
        help="quantize lm_head.weight_t and/or head.lm_head.weight_t to FP8 SkinnyGemm",
    )
    parser.add_argument(
        "--hc-quant",
        choices=("none", "fp8"),
        default="none",
        help="quantize [160, 20480] hc_tiled.data hyper-connection weights to FP8 E4M3FN",
    )
    parser.add_argument(
        "--share-dense-weights",
        action="store_true",
        default=False,
        help="rewrite dense/model.onnx and head_kv/model.onnx to share merged_proj, quantized projections, and hc_tiled.data with target",
    )
    parser.add_argument(
        "--fuse-dense-ops",
        action="store_true",
        default=False,
        help="fuse DivSigmoidMul, SwishMul, and SharedGateMulAdd in dense/model.onnx",
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
    parser.add_argument(
        "--hc-lib",
        type=Path,
        default=None,
        help="optional fused_hc library override (libqwen_fused_hc_r5.so)",
    )
    parser.add_argument(
        "--ops-lib",
        type=Path,
        default=None,
        help="optional fused_ops library override (libqwen_fused_ops.so)",
    )
    parser.add_argument(
        "--ops-dir",
        type=Path,
        default=None,
        help="optional directory containing rebuilt libqwen_*.so custom-op libraries",
    )
    parser.add_argument(
        "--provider-lib",
        type=Path,
        default=None,
        help="optional CUDA EP library override (provider/libonnxruntime_providers_cuda.so)",
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=None,
        help="override model.qwen38_graph.prefill.chunk and filter warmup_lengths to <= chunk",
    )
    parser.add_argument(
        "--host-map-embed",
        action="store_true",
        default=False,
        help="place model.embed_tokens.weight in host_pool (CPU host-mapped UVA zero-copy) in weights_pool.txt",
    )
    parser.add_argument(
        "--explicit-cpu-embed",
        action="store_true",
        default=False,
        help="extract model.embed_tokens.weight into explicit target/cpu_embed_tokens.data and map via host_pool",
    )
    parser.add_argument(
        "--clean-package",
        action="store_true",
        default=False,
        help="remove unneeded safetensors/dense.data/merged.data/generated.data, compact expert scales, and extract explicit cpu_embed_tokens.data",
    )
    parser.add_argument(
        "--embed-quant",
        choices=("none", "int8"),
        default="int8",
        help="quantize CPU embedding.onnx.data to symmetric per-row INT8 (607.2 MiB vs 1212.5 MiB BF16)",
    )
    parser.add_argument(
        "--mtp-kv-quant",
        choices=("same", "bf16"),
        default="bf16",
        help="keep MTP head KV cache in BF16 for better performance and draft acceptance while target KV is INT8",
    )
    parser.add_argument(
        "--keep-bf16-router-shared",
        action="store_true",
        default=False,
        help="keep MoE router and shared expert projections (merged_proj.{mlp} and SharedExpertDown) in BF16",
    )
    parser.add_argument(
        "--keep-bf16-shared-down",
        action="store_true",
        default=False,
        help="keep SharedExpertDown projections in BF16",
    )
    args = parser.parse_args()

    source, out = args.from_package.resolve(), args.out.resolve()
    if not (source / "genai_config.json").exists():
        raise SystemExit(f"{source} is not a package (no genai_config.json)")
    if not args.plane_lib.exists():
        raise SystemExit(f"missing plane library: {args.plane_lib}")
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise SystemExit("--out must differ from and not be inside --from-package")
    out.mkdir(parents=True, exist_ok=True)

    share_dense = args.share_dense_weights or (args.weight_quant != "none") or (args.hc_quant != "none")
    clean_pkg = args.clean_package or args.explicit_cpu_embed
    transform_weights = (
        args.weight_quant != "none"
        or args.mtp_moe_quant != "none"
        or args.lm_head_quant != "none"
        or args.hc_quant != "none"
        or share_dense
        or args.fuse_dense_ops
    )
    transform_models = args.kv_quant != "none" or transform_weights

    skip_rel: set[Path] = set()
    if transform_models or clean_pkg:
        skip_rel.update({Path("target/model.onnx"), Path("head_r2/model.onnx"), Path("weights_pool.txt")})
        if share_dense or args.fuse_dense_ops or clean_pkg:
            skip_rel.update({Path("dense/model.onnx"), Path("head_kv/model.onnx")})
        if transform_weights:
            skip_rel.add(Path("target/hc_tiled.data"))
        if args.kv_quant != "none":
            skip_rel.update({Path("qsa_consts.bin"), Path("qsa_consts.txt")})

    if clean_pkg and not transform_models:
        # Only link top-level files, ple/, libs/, and provider/; clean_and_pack_package populates target/head_r2/dense/head_kv
        linked = 0
        for root, _dirs, files in os.walk(source, followlinks=False):
            rel = Path(root).relative_to(source)
            if rel.parts and rel.parts[0] in ("target", "head_r2", "dense", "head_kv"):
                continue
            (out / rel).mkdir(parents=True, exist_ok=True)
            for name in files:
                relative = rel / name
                if relative in (Path(n) for n in REPLACED) or relative == PLANE_REL or relative in skip_rel:
                    continue
                link_file(Path(root) / name, out / relative, args.copy)
                linked += 1
    else:
        linked = link_tree(source, out, args.copy, skip_rel=skip_rel)

    if args.ops_dir is not None and args.ops_dir.exists():
        for so_file in sorted(args.ops_dir.glob("libqwen_*.so")):
            link_file(so_file.resolve(), out / "libs" / so_file.name, True)
    link_file(args.plane_lib.resolve(), out / PLANE_REL, True)
    if args.provider_lib is not None:
        if not args.provider_lib.exists():
            raise SystemExit(f"missing provider library: {args.provider_lib}")
        link_file(args.provider_lib.resolve(), out / "provider" / "libonnxruntime_providers_cuda.so", True)

    config = json.loads((source / "genai_config.json").read_text())
    qwen = config["model"]["qwen38_graph"]
    if not qwen.get("fold", {}).get("enabled") or not qwen.get("prefill", {}).get("enabled"):
        raise SystemExit("the source package is not a v12 package (fold.enabled and prefill.enabled are required)")
    if args.stop_token_id is not None:
        config["model"]["eos_token_id"] = [args.stop_token_id]
    if args.prefill_chunk_tokens is not None:
        qwen["prefill_chunk_tokens"] = args.prefill_chunk_tokens
    if args.prefill_chunk is not None:
        qwen["prefill"]["chunk"] = args.prefill_chunk
        if "warmup_lengths" in qwen["prefill"]:
            qwen["prefill"]["warmup_lengths"] = [
                x for x in qwen["prefill"]["warmup_lengths"] if x <= args.prefill_chunk
            ]

    src_manifest = {}
    if (source / "package_manifest.json").exists():
        try:
            src_manifest = json.loads((source / "package_manifest.json").read_text())
        except Exception:
            src_manifest = {}

    kv_info = src_manifest.get("kv_quant")
    weight_info = src_manifest.get("weight_quant")
    pool_info = src_manifest.get("weights_pool")
    clean_info = None

    if transform_models:
        import onnx  # noqa: PLC0415

        target_model = onnx.load(str(source / "target/model.onnx"), load_external_data=False)
        head_r2_model = onnx.load(str(source / "head_r2/model.onnx"), load_external_data=False)
        dense_model = (
            onnx.load(str(source / "dense/model.onnx"), load_external_data=False)
            if (share_dense or args.fuse_dense_ops)
            else None
        )
        head_kv_model = (
            onnx.load(str(source / "head_kv/model.onnx"), load_external_data=False) if share_dense else None
        )

        if args.kv_quant != "none":
            scale_file = args.kv_scale_file
            if scale_file is None or not scale_file.exists():
                raise SystemExit("--kv-scale-file is required and must exist when --kv-quant != none")
            if args.dsa_lib is not None and not args.dsa_lib.exists():
                raise SystemExit(f"missing GuardedDSA library: {args.dsa_lib}")
            kv_info = apply_kv_quant(
                source,
                out,
                target_model,
                head_r2_model,
                args.kv_quant,
                scale_file.resolve(),
                mtp_kv_quant=args.mtp_kv_quant,
            )
            if args.dsa_lib is not None:
                link_file(args.dsa_lib.resolve(), out / "libs" / "libqwen_guarded_dsa_r2_v12.so", True)

        if transform_weights:
            weight_info = apply_weight_and_dense_transforms(
                source,
                out,
                target_model,
                head_r2_model,
                dense_model,
                head_kv_model,
                args.weight_quant,
                args.mtp_moe_quant,
                lm_head_quant=args.lm_head_quant,
                hc_quant=args.hc_quant,
                share_dense=share_dense,
                fuse_dense_ops=args.fuse_dense_ops,
                keep_bf16_router_shared=args.keep_bf16_router_shared,
                keep_bf16_shared_down=args.keep_bf16_shared_down,
            )
            if args.gemm_lib is not None:
                link_file(args.gemm_lib.resolve(), out / "libs" / "libqwen_fused_gemm_r5.so", True)
            if args.mtp_lib is not None:
                link_file(args.mtp_lib.resolve(), out / "libs" / "libqwen_mtp_head.so", True)
            if args.hc_lib is not None:
                link_file(args.hc_lib.resolve(), out / "libs" / "libqwen_fused_hc_r5.so", True)
            if args.ops_lib is not None:
                link_file(args.ops_lib.resolve(), out / "libs" / "libqwen_fused_ops.so", True)

        onnx.save(target_model, str(out / "target/model.onnx"))
        onnx.save(head_r2_model, str(out / "head_r2/model.onnx"))
        if dense_model is not None:
            onnx.save(dense_model, str(out / "dense/model.onnx"))
            dense_so = qwen["prefill"]["dense"]["session_options"]
            dense_libs = ["libs/libqwen_fused_gemm_r5.so"]
            if args.fuse_dense_ops:
                dense_libs.append("libs/libqwen_fused_ops.so")
            dense_so["custom_ops_libraries"] = dense_libs
            dense_so["provider_options"] = [{"cuda": {"arena_extend_strategy": "kSameAsRequested"}}]
        if head_kv_model is not None:
            onnx.save(head_kv_model, str(out / "head_kv/model.onnx"))
            hk_so = qwen["prefill"]["head_kv"]["session_options"]
            hk_so["custom_ops_libraries"] = ["libs/libqwen_fused_gemm_r5.so"]
            hk_so["provider_options"] = [{"cuda": {"arena_extend_strategy": "kSameAsRequested"}}]

    if clean_pkg:
        clean_info = clean_and_pack_package(source, out, copy=args.copy, embed_quant=args.embed_quant)
        config["model"]["decoder"]["filename"] = "model.onnx"
        old_fold_head = qwen["fold"].pop("head_r2", None) or qwen["fold"].get("mtp", {})
        graph_id = old_fold_head.get("graph_id", 81) if isinstance(old_fold_head, dict) else 81
        qwen["fold"]["mtp"] = {"filename": "mtp.onnx", "graph_id": graph_id}
        if isinstance(old_fold_head, dict) and "session_options" in old_fold_head:
            qwen["fold"]["mtp"]["session_options"] = old_fold_head["session_options"]
        else:
            qwen["fold"]["mtp"]["session_options"] = {
                "log_id": "onnxruntime-genai",
                "graph_optimization_level": "ORT_DISABLE_ALL",
                "provider_options": [
                    {
                        "cuda": {
                            "enable_cuda_graph": "1",
                            "min_num_runs_before_cuda_graph_capture": "4",
                            "arena_extend_strategy": "kSameAsRequested",
                            "arena.extend_strategy": "1",
                        }
                    }
                ],
                "session.use_device_allocator_for_initializers": "1",
            }
        config["model"]["embedding"] = {
            "filename": "embedding.onnx",
            "session_options": {"intra_op_num_threads": 1},
            "inputs": {"input_ids": "input_ids"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }
        qwen["prefill"].pop("dense", None)
        qwen["prefill"].pop("head_kv", None)
        qwen["prefill"].pop("constants", None)

        # Copy STRP v1 expert popularity profile and configure Portable QMoE Offload v4 runtime_profiles
        strata_prof = Path("/home/tianlei/git/Strata/data/expert-profile.bin")
        if strata_prof.exists():
            shutil.copy2(strata_prof, out / "expert-profile.bin")
        config["runtime_profiles"] = [
            {
                "id": "consumer-gpu-24g",
                "eligibility": {
                    "minimum_total_device_memory_bytes": 21474836480,
                    "maximum_total_device_memory_bytes": 26843545600,
                    "minimum_effective_host_memory_bytes": 68719476736,
                    "is_integrated": False,
                },
                "overlay": {
                    "model": {
                        "decoder": {
                            "filename": "model_s4.onnx",
                            "session_options": {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "adaptive",
                                "session.moe.offload.cpu_execution": "auto",
                                "session.moe.offload.device_expert_bytes": "10351538176",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                            },
                        }
                    }
                },
            },
            {
                "id": "consumer-gpu-24g-host-64g",
                "eligibility": {
                    "minimum_total_device_memory_bytes": 21474836480,
                    "maximum_total_device_memory_bytes": 26843545600,
                    "maximum_effective_host_memory_bytes": 68719476735,
                    "is_integrated": False,
                },
                "overlay": {
                    "model": {
                        "decoder": {
                            "filename": "model_s4.onnx",
                            "session_options": {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "adaptive",
                                "session.moe.offload.cpu_execution": "auto",
                                "session.moe.offload.device_expert_bytes": "10351538176",
                                "session.moe.offload.host_expert_bytes": "17179869184",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                            },
                        }
                    }
                },
            },
            {
                "id": "consumer-gpu-30g-ram-48g-32k",
                "eligibility": {
                    "minimum_total_device_memory_bytes": 26843545601,
                    "maximum_total_device_memory_bytes": 32212254720,
                    "maximum_effective_host_memory_bytes": 68719476735,
                    "is_integrated": False,
                },
                "overlay": {
                    "model": {
                        "decoder": {
                            "filename": "model_s4.onnx",
                            "session_options": {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "adaptive",
                                "session.moe.offload.cpu_execution": "auto",
                                "session.moe.offload.device_expert_bytes": "18289511424",
                                "session.moe.offload.host_expert_bytes": "41339060224",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                            },
                        }
                    }
                },
            },
            {
                "id": "consumer-gpu-32g",
                "eligibility": {
                    "minimum_total_device_memory_bytes": 26843545601,
                    "maximum_total_device_memory_bytes": 36507222016,
                    "minimum_effective_host_memory_bytes": 68719476736,
                    "is_integrated": False,
                },
                "overlay": {
                    "model": {
                        "decoder": {
                            "filename": "model_s4.onnx",
                            "session_options": {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "adaptive",
                                "session.moe.offload.cpu_execution": "auto",
                                "session.moe.offload.device_expert_bytes": "19229035520",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                            },
                        }
                    }
                },
            },
        ]

    if transform_models or clean_pkg:
        if clean_pkg:
            models_map = {
                "target": out / "model.onnx",
                "mtp": out / "mtp.onnx",
            }
        else:
            models_map = {
                "target": out / "target/model.onnx",
                "head_r2": out / "head_r2/model.onnx",
                "dense": out / "dense/model.onnx",
                "head_kv": out / "head_kv/model.onnx",
            }
        use_host_embed = (args.host_map_embed or bool(src_manifest.get("weights_pool", {}).get("host_gib"))) and not clean_pkg
        pool_info = write_pool(out, models_map, host_map_embed=use_host_embed, exclude_embed=clean_pkg)

    (out / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")

    libs = {}
    for path in sorted((out / "libs").glob("*.so")):
        libs[path.name] = sha256(path)
    provider_dir = out / "provider"
    providers = {p.name: sha256(p) for p in sorted(provider_dir.glob("*.so"))} if provider_dir.exists() else {}
    manifest = {
        "from_package": str(source),
        "stack": "v12d-engine-rc1" if clean_pkg else "v12b-engine",
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
    if clean_info is not None:
        manifest["clean_package"] = clean_info
    if pool_info is not None:
        manifest["weights_pool"] = pool_info
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"package: {out} ({linked} files linked)")
    print(f"plane: {libs.get(PLANE_REL.name)}")
    if clean_info is not None:
        ce = clean_info["cpu_embedding"]
        cs = clean_info["compacted_expert_scales"]
        nr = clean_info["norm_and_rope_data"]
        print(
            f"clean_package: cpu_embed={ce['model_file']}+{ce['data_file']} ({ce['mib']:.2f} MiB, op={ce['operator']}), "
            f"norm_and_rope={nr['file']} ({nr['mib']:.2f} MiB, {nr['unique_tensors']} tensors), "
            f"expert_scales saved={cs['saved_gib']:.2f} GiB"
        )
    if pool_info is not None:
        host_s = f", host={pool_info['host_gib']:.3f} GiB" if "host_gib" in pool_info else ""
        print(f"pool: {pool_info['initializers']} initializers, device={pool_info['gib']:.3f} GiB{host_s}")
    for name, digest in providers.items():
        print(f"provider {name}: {digest}")


if __name__ == "__main__":
    sys.exit(main())
