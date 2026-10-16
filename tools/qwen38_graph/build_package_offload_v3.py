#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Zero-copy package builder and graph migrator for QMoEOffload v3 (NVFP4 and SharedExpertBlockV1 INT4).

 Supports two migration modes:
 1. NVFP4 v12b package migration (--from-package /data/tlwu/qwen38_flash_next_v12b --out <dir>):
    - Hard-links all weight files (0 GiB disk copy).
    - Rewrites the 48 com.microsoft::QMoE nodes in target/model.onnx and dense/model.onnx to
      com.microsoft::QMoEOffload with explicit layer_index, hidden_size, inter_size, and v3 dispatch controls.
    - Updates weights_pool.txt so the 48 layers of routed expert weights/scales (63.281 GiB) use "cpu"
      mmap placement, and optionally places dense/source_weights/dense.data (5.270 GiB) in "host_mapped"
      UVA placement for 24 GiB VRAM targets.
    - Configures genai_config.json for 24 GiB or 32 GiB VRAM operation.
 2. Standalone SharedExpertBlockV1 v3 compact schema migration (--migrate-shared-onnx <in.onnx> --out-onnx <out.onnx>):
    - Rewrites legacy 26-input QMoEOffload nodes (21->2 expert_blocks, 24->3 fc2_bias, 25->4 optional phase)
      without reading or regenerating the 63.457 GiB external weight payload (Section 5.3 / F8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import onnx
from onnx import helper


REPLACED = {
    Path("genai_config.json"),
    Path("package_manifest.json"),
    Path("README.md"),
    Path("weights_pool.txt"),
    Path("target/model.onnx"),
    Path("dense/model.onnx"),
    Path("libs/libqwen38_plane.so"),
    Path("provider/libonnxruntime_providers_cuda.so"),
}


def link_file(src: Path, dst: Path, copy: bool = False) -> None:
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


def link_tree(src: Path, dst: Path, copy: bool = False) -> int:
    count = 0
    for root, _dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        if rel.name == "_source_weights":
            alias = (dst / rel).parent / "source_weights"
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to("_source_weights")
        for name in files:
            relative = rel / name
            if relative in REPLACED:
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


def migrate_shared_onnx_v3(in_onnx: Path, out_onnx: Path, strip_phase: bool = False, strip_stats: bool = False) -> int:
    """Zero-copy v3 compact shared schema migration (21->2, 24->3, 25->4) per Section 5.3 (F8)."""
    model = onnx.load(str(in_onnx), load_external_data=False)
    migrated = 0
    stat_outputs_removed = set()
    for idx, node in enumerate(model.graph.node):
        if node.domain == "com.microsoft" and node.op_type == "QMoEOffload":
            inputs = list(node.input)
            if len(inputs) >= 22 and inputs[21]:
                x_in = inputs[0]
                router_in = inputs[1]
                blocks_in = inputs[21]
                bias_in = inputs[24] if len(inputs) > 24 else ""
                phase_in = "" if strip_phase else (inputs[25] if len(inputs) > 25 else "")
                new_inputs = [x_in, router_in, blocks_in]
                if bias_in or phase_in:
                    new_inputs.append(bias_in)
                if phase_in:
                    new_inputs.append(phase_in)
                del node.input[:]
                node.input.extend(new_inputs)
                if strip_stats and len(node.output) > 1:
                    for out_name in list(node.output)[1:]:
                        if out_name:
                            stat_outputs_removed.add(out_name)
                    del node.output[1:]
                existing_attrs = {a.name for a in node.attribute}
                if "layout_id" not in existing_attrs:
                    node.attribute.append(helper.make_attribute("layout_id", "ort.qmoe.w4a16.block32.shared.v1"))
                if "layer_index" not in existing_attrs:
                    node.attribute.append(helper.make_attribute("layer_index", migrated))
                migrated += 1
    if stat_outputs_removed:
        kept_outputs = [o for o in model.graph.output if o.name not in stat_outputs_removed]
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out_onnx), save_as_external_data=False)
    return migrated


def load_expert_profile_plan(
    profile_path: Path,
    num_layers: int = 48,
    num_experts: int = 512,
    total_slots: int = 960,
    min_slots_per_layer: int = 4,
    uniform: bool = False,
) -> dict[int, list[int]]:
    """Loads STRP v1 expert-profile.bin and assigns per-layer hot expert lists summing to total_slots."""
    import struct
    import numpy as np

    data = profile_path.read_bytes()
    magic, version, L, E, _slots, n_ranked = struct.unpack("<6I", data[:24])
    if magic != 1347572819 or version != 1 or L != num_layers or E != num_experts:
        raise ValueError(f"Incompatible expert profile header in {profile_path}: {(magic, version, L, E)}")
    pairs = np.frombuffer(data[24 : 24 + n_ranked * 4], dtype=np.uint16).reshape(n_ranked, 2)

    by_layer: dict[int, list[int]] = {l: [] for l in range(num_layers)}
    for l_id, e_id in pairs:
        by_layer[int(l_id)].append(int(e_id))

    if uniform:
        per_layer = total_slots // num_layers
        return {l: by_layer[l][:per_layer] for l in range(num_layers)}

    # Nonuniform static allocation with per-layer floor (v3 Section 8 / P2):
    # Reserve min_slots_per_layer hottest experts in every layer first, then fill remaining budget
    # in global popularity order across all layers.
    selected: dict[int, list[int]] = {l: [] for l in range(num_layers)}
    selected_set: dict[int, set[int]] = {l: set() for l in range(num_layers)}
    budget = total_slots
    floor = min(min_slots_per_layer, total_slots // num_layers, num_experts)
    for l in range(num_layers):
        for e in by_layer[l][:floor]:
            selected[l].append(e)
            selected_set[l].add(e)
            budget -= 1

    for l_id, e_id in pairs:
        if budget <= 0:
            break
        l = int(l_id)
        e = int(e_id)
        if e not in selected_set[l] and len(selected[l]) < num_experts:
            selected[l].append(e)
            selected_set[l].add(e)
            budget -= 1

    return selected


def rewrite_nvfp4_qmoe_to_offload(
    in_onnx: Path,
    out_onnx: Path,
    hidden_size: int = 2560,
    inter_size: int = 640,
    token_tile_size: int = 32,
    extra_attrs: dict | None = None,
    per_layer_initial_experts: dict[int, list[int]] | None = None,
) -> int:
    """Rewrites com.microsoft::QMoE (quant_type='nvfp4') nodes to com.microsoft::QMoEOffload."""
    model = onnx.load(str(in_onnx), load_external_data=False)
    rewritten = 0
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "QMoE":
            layer_idx = rewritten
            node.op_type = "QMoEOffload"
            # Clear unused slot 14 (router_weights) if present in dense/model.onnx
            if len(node.input) > 14 and node.input[14]:
                node.input[14] = ""
            existing_attrs = {a.name for a in node.attribute}
            if "layer_index" not in existing_attrs:
                node.attribute.append(helper.make_attribute("layer_index", layer_idx))
            if "hidden_size" not in existing_attrs:
                node.attribute.append(helper.make_attribute("hidden_size", hidden_size))
            if "inter_size" not in existing_attrs:
                node.attribute.append(helper.make_attribute("inter_size", inter_size))
            if "token_tile_size" not in existing_attrs:
                node.attribute.append(helper.make_attribute("token_tile_size", token_tile_size))
            if "gpu_working_bytes" not in existing_attrs:
                node.attribute.append(helper.make_attribute("gpu_working_bytes", 512 << 20))
            if "pinned_staging_bytes" not in existing_attrs:
                node.attribute.append(helper.make_attribute("pinned_staging_bytes", 64 << 20))
            if per_layer_initial_experts is not None and layer_idx in per_layer_initial_experts:
                init_exps = per_layer_initial_experts[layer_idx]
                node.attribute.append(helper.make_attribute("resident_slots", len(init_exps)))
                node.attribute.append(helper.make_attribute("initial_experts", init_exps))
                existing_attrs.add("resident_slots")
                existing_attrs.add("initial_experts")
            if extra_attrs:
                for k_attr, v_attr in extra_attrs.items():
                    if k_attr not in existing_attrs and v_attr is not None:
                        node.attribute.append(helper.make_attribute(k_attr, v_attr))
            rewritten += 1
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out_onnx), save_as_external_data=False)
    return rewritten


def build_offload_weights_pool(
    source_pkg: Path,
    out_pkg: Path,
    dense_placement: str = "device",
    mtp_moe_placement: str = "device",
    embed_placement: str = "host_mapped",
) -> dict:
    """Rewrites weights_pool.txt with CPU placement for QMoE experts and optional host_mapped for dense.data."""
    src_pool = (source_pkg / "weights_pool.txt").read_text().splitlines()
    assert src_pool[0].strip() == "qwen38_weights_pool 1"

    pool_lines = []
    session_lines = []
    qmoe_pool_ids = set()
    mtp_moe_pool_ids = set()
    dense_bound_names = set()

    for line in src_pool[1:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if parts[0] == "session":
            sess, name, pid = parts[1], parts[2], int(parts[3])
            if ".mlp.fc1_" in name or ".mlp.fc2_" in name:
                qmoe_pool_ids.add(pid)
            if "mtp.layers.0.mlp.experts." in name:
                mtp_moe_pool_ids.add(pid)
            if sess == "dense":
                dense_bound_names.add(name)
            session_lines.append((sess, name, pid))

    aligned_mtp_rel = "target/head_data/source_weights/mtp_moe_aligned.bin"
    need_aligned_mtp_copy = False
    for line in src_pool[1:]:
        parts = line.strip().split()
        if parts and parts[0] == "pool" and int(parts[1]) in mtp_moe_pool_ids:
            if int(parts[3]) % 16 != 0:
                need_aligned_mtp_copy = True

    if mtp_moe_placement != "device" and need_aligned_mtp_copy:
        shared_aligned = source_pkg.parent / f"{source_pkg.name}_mtp_moe_aligned.bin"
        src_st = source_pkg / "target/head_data/source_weights/model-bf16-00011.safetensors"
        expected_size = 3355443200 + 1677721600
        if not shared_aligned.exists() or shared_aligned.stat().st_size != expected_size:
            tmp_aligned = shared_aligned.with_suffix(".tmp")
            with open(src_st, "rb") as fin, open(tmp_aligned, "wb") as fout:
                in_fd = fin.fileno()
                out_fd = fout.fileno()
                for src_off, dst_off, total_len in [
                    (7243385096, 0, 3355443200),
                    (5565663496, 3355443200, 1677721600),
                ]:
                    rem = total_len
                    r_off = src_off
                    w_off = dst_off
                    while rem > 0:
                        chunk = min(rem, 1 << 26)
                        copied = os.copy_file_range(in_fd, out_fd, chunk, offset_src=r_off, offset_dst=w_off)
                        if copied <= 0:
                            raise RuntimeError("os.copy_file_range returned 0")
                        r_off += copied
                        w_off += copied
                        rem -= copied
            tmp_aligned.replace(shared_aligned)
        link_file(shared_aligned, out_pkg / aligned_mtp_rel, copy=False)

    for line in src_pool[1:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if parts[0] == "pool":
            pid = int(parts[1])
            if pid in qmoe_pool_ids:
                pool_lines.append(f"{s} cpu")
            elif pid == 0 and embed_placement != "device":
                pool_lines.append(f"{s} {embed_placement}")
            elif pid in mtp_moe_pool_ids and mtp_moe_placement != "device":
                if not need_aligned_mtp_copy:
                    pool_lines.append(f"{s} {mtp_moe_placement}")
                elif pid == 607:
                    pool_lines.append(
                        f"pool 607 16 0 3355443200 3 512 1280 2560 {aligned_mtp_rel} {mtp_moe_placement}"
                    )
                elif pid == 608:
                    pool_lines.append(
                        f"pool 608 16 3355443200 1677721600 3 512 2560 640 {aligned_mtp_rel} {mtp_moe_placement}"
                    )
            else:
                pool_lines.append(s)

    next_pool_id = len(pool_lines)
    added_dense_entries = 0
    if dense_placement in ("host_mapped", "cpu"):
        dense_model = onnx.load(str(source_pkg / "dense/model.onnx"), load_external_data=False)
        for init in dense_model.graph.initializer:
            if init.name in dense_bound_names:
                continue
            ext = {e.key: e.value for e in init.external_data}
            loc = ext.get("location", "")
            if loc == "source_weights/dense.data":
                offset = int(ext.get("offset", "0"))
                length = int(ext["length"])
                dims = list(init.dims)
                shape_str = f"{len(dims)} " + " ".join(str(d) for d in dims)
                pool_file = "target/source_weights/dense.data"
                pool_lines.append(
                    f"pool {next_pool_id} {init.data_type} {offset} {length} {shape_str} {pool_file} {dense_placement}"
                )
                session_lines.append(("dense", init.name, next_pool_id))
                next_pool_id += 1
                added_dense_entries += 1

    out_lines = ["qwen38_weights_pool 1", *pool_lines]
    for sess, name, pid in session_lines:
        out_lines.append(f"session {sess} {name} {pid}")
    (out_pkg / "weights_pool.txt").write_text("\n".join(out_lines) + "\n")
    return {
        "total_pool_entries": next_pool_id,
        "cpu_qmoe_entries": len(qmoe_pool_ids),
        "added_dense_entries": added_dense_entries,
        "dense_placement": dense_placement,
        "mtp_moe_placement": mtp_moe_placement,
        "embed_placement": embed_placement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--migrate-shared-onnx", type=Path, default=None, help="Input ONNX for standalone v3 shared migration")
    parser.add_argument("--out-onnx", type=Path, default=None, help="Output ONNX for standalone v3 shared migration")
    parser.add_argument("--strip-phase", action="store_true")
    parser.add_argument("--strip-stats", action="store_true")

    parser.add_argument("--from-package", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plane-lib", type=Path, default=None)
    parser.add_argument("--provider-lib", type=Path, default=None)
    parser.add_argument("--stop-token-id", type=int, default=None)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--target-resident-slots", type=int, default=None)
    parser.add_argument("--target-spare-slots", type=int, default=None)
    parser.add_argument("--expert-profile", type=Path, default=None, help="Path to STRP v1 expert-profile.bin")
    parser.add_argument("--total-resident-slots", type=int, default=None, help="Total resident slots across 48 layers")
    parser.add_argument("--min-slots-per-layer", type=int, default=4)
    parser.add_argument("--uniform-profile-slots", action="store_true")
    parser.add_argument("--dense-stream-min-rows", type=int, default=1)
    parser.add_argument("--dense-stream-wave", type=int, default=64)
    parser.add_argument("--dense-gpu-qmoe", action="store_true", default=True, help="Keep stock GPU QMoE in dense/model.onnx using UVA-mapped expert weights")
    parser.add_argument("--dense-offload-qmoe", dest="dense_gpu_qmoe", action="store_false")
    parser.add_argument("--dense-placement", choices=["device", "host_mapped", "cpu"], default="host_mapped")
    parser.add_argument("--mtp-moe-placement", choices=["device", "host_mapped"], default="device")
    parser.add_argument("--embed-placement", choices=["device", "host_mapped"], default="host_mapped")
    parser.add_argument("--vram-limit-gib", type=float, default=None)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    if args.migrate_shared_onnx is not None:
        if args.out_onnx is None:
            raise SystemExit("--out-onnx is required with --migrate-shared-onnx")
        n = migrate_shared_onnx_v3(args.migrate_shared_onnx, args.out_onnx, args.strip_phase, args.strip_stats)
        print(f"Migrated {n} QMoEOffload nodes to v3 compact shared schema -> {args.out_onnx}")
        return

    if args.from_package is None or args.out is None or args.plane_lib is None or args.provider_lib is None:
        raise SystemExit("--from-package, --out, --plane-lib, and --provider-lib are required for package build")

    source = args.from_package.resolve()
    out = args.out.resolve()
    if out == source:
        raise SystemExit("--out must differ from --from-package")
    out.mkdir(parents=True, exist_ok=True)

    linked = link_tree(source, out, args.copy)
    link_file(args.plane_lib.resolve(), out / "libs/libqwen38_plane.so", copy=True)
    link_file(args.provider_lib.resolve(), out / "provider/libonnxruntime_providers_cuda.so", copy=True)

    profile_plan = None
    if args.expert_profile is not None:
        total_slots = args.total_resident_slots or ((args.target_resident_slots or 20) * 48)
        profile_plan = load_expert_profile_plan(
            args.expert_profile.resolve(),
            num_layers=48,
            num_experts=512,
            total_slots=total_slots,
            min_slots_per_layer=args.min_slots_per_layer,
            uniform=args.uniform_profile_slots,
        )

    target_attrs = {
        "resident_slots": args.target_resident_slots,
        "spare_slots": args.target_spare_slots,
        "stream_rows": 0,
    }
    dense_attrs = {
        "resident_slots": 0,
        "spare_slots": 0,
        "stream_rows": args.prefill_chunk,
        "stream_min_tokens": 1,
        "stream_min_rows": args.dense_stream_min_rows,
        "stream_wave": args.dense_stream_wave,
    }
    target_rewritten = rewrite_nvfp4_qmoe_to_offload(
        source / "target/model.onnx",
        out / "target/model.onnx",
        extra_attrs=target_attrs,
        per_layer_initial_experts=profile_plan,
    )
    if args.dense_gpu_qmoe:
        link_file(source / "dense/model.onnx", out / "dense/model.onnx", args.copy)
        dense_rewritten = 0
    else:
        dense_rewritten = rewrite_nvfp4_qmoe_to_offload(
            source / "dense/model.onnx", out / "dense/model.onnx", extra_attrs=dense_attrs
        )
    pool_stats = build_offload_weights_pool(
        source, out, args.dense_placement, args.mtp_moe_placement, args.embed_placement
    )

    config = json.loads((source / "genai_config.json").read_text())
    if args.stop_token_id is not None:
        config["model"]["eos_token_id"] = [args.stop_token_id]

    # Target decoder supports CUDA graph capture via Device-Doorbell Host Worker Protocol (ORT_QMOE_ALLOW_CUDA_GRAPH=1).
    cuda_opts = config["model"]["decoder"]["session_options"]["provider_options"][0]["cuda"]
    cuda_opts["enable_cuda_graph"] = "1"
    cuda_opts["arena_extend_strategy"] = "kSameAsRequested"
    cuda_opts["arena.extend_strategy"] = "1"
    if args.vram_limit_gib is not None and args.vram_limit_gib > 0:
        cuda_opts["gpu_device_mem_limit"] = str(int(args.vram_limit_gib * (1024**3)))

    qwen = config["model"]["qwen38_graph"]
    qwen["fold"]["head_r2"]["session_options"] = {
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
    qwen["prefill"]["chunk"] = args.prefill_chunk
    qwen["prefill"]["warmup_lengths"] = [min(args.prefill_chunk, 512), 128, 37]
    dense_cuda = qwen["prefill"]["dense"]["session_options"]["provider_options"][0]["cuda"]
    dense_cuda["arena_extend_strategy"] = "kSameAsRequested"
    dense_cuda["arena.extend_strategy"] = "1"
    head_kv_cuda = qwen["prefill"]["head_kv"]["session_options"]["provider_options"][0]["cuda"]
    head_kv_cuda["arena_extend_strategy"] = "kSameAsRequested"
    head_kv_cuda["arena.extend_strategy"] = "1"

    (out / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")

    libs = {p.name: sha256(p) for p in sorted((out / "libs").glob("*.so"))}
    providers = {p.name: sha256(p) for p in sorted((out / "provider").glob("*.so"))}
    manifest = {
        "from_package": str(source),
        "stack": "v12b-engine-qmoe-offload-v3",
        "hard_linked_files": linked,
        "target_qmoe_offload_nodes": target_rewritten,
        "dense_qmoe_offload_nodes": dense_rewritten,
        "pool_stats": pool_stats,
        "eos_token_id": config["model"]["eos_token_id"],
        "vram_limit_gib": args.vram_limit_gib,
        "plane_library": {"source": str(args.plane_lib.resolve()), "sha256": libs.get("libqwen38_plane.so")},
        "libs_sha256": libs,
        "provider_sha256": providers,
    }
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Built offload v3 package: {out} (linked={linked}, target_qmoe={target_rewritten}, "
        f"dense_qmoe={dense_rewritten}, pool={pool_stats})"
    )


if __name__ == "__main__":
    main()
