#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Zero-copy package builder and graph migrator for QMoEOffload v3 (NVFP4 and SharedExpertBlockV1 INT4).

Supports two migration modes:
1. NVFP4 v12b package migration (--from-package <pkg_dir> --out <dir>):
   - Hard-links all weight files (0 GiB disk copy).
   - Rewrites the 48 com.microsoft::QMoE nodes in target/model.onnx (or model.onnx) and dense/model.onnx to
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
from pathlib import Path

import onnx
from onnx import helper

REPLACED = {
    Path("genai_config.json"),
    Path("package_manifest.json"),
    Path("README.md"),
    Path("weights_pool.txt"),
    Path("model.onnx"),
    Path("model_s2.onnx"),
    Path("model_s3.onnx"),
    Path("model_s4.onnx"),
    Path("target/model.onnx"),
    Path("target/model_s2.onnx"),
    Path("target/model_s3.onnx"),
    Path("target/model_s4.onnx"),
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
            pass  # hard links can fail across filesystems; fall back to copying
    shutil.copy2(src, dst)


def save_model_fresh(model: onnx.ModelProto, path: Path) -> None:
    """Saves to a new inode: the output tree hard-links the source package, and a write must never go through."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=False)


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
            if name.startswith(".azDownload"):
                continue
            relative = rel / name
            if relative in REPLACED:
                continue
            link_file(Path(root) / name, dst / relative, copy)
            count += 1
    fallback_experts = src.parent / "pkg_v12d_bf16lm_engine/target/source_weights/experts.data"
    if not fallback_experts.exists():
        fallback_experts = src.parent / "qwen_flash_v12c_engine_int8kv_fp8w_fp4moe/target/source_weights/experts.data"
    exp_rels = (
        [Path("source_weights/experts.data")]
        if (src / "model.onnx").exists()
        else [Path("target/source_weights/experts.data")]
    )
    if (src / "dense").exists():
        exp_rels.append(Path("dense/source_weights/experts.data"))
    for exp_rel in exp_rels:
        dst_exp = dst / exp_rel
        if not dst_exp.exists() and fallback_experts.exists():
            link_file(fallback_experts, dst_exp, copy)
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
    for node in model.graph.node:
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
    save_model_fresh(model, out_onnx)
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
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    data = profile_path.read_bytes()
    magic, version, hdr_layers, hdr_experts, _slots, n_ranked = struct.unpack("<6I", data[:24])
    if magic != 1347572819 or version != 1 or hdr_layers != num_layers or hdr_experts != num_experts:
        raise ValueError(
            f"Incompatible expert profile header in {profile_path}: {(magic, version, hdr_layers, hdr_experts)}"
        )
    pairs = np.frombuffer(data[24 : 24 + n_ranked * 4], dtype=np.uint16).reshape(n_ranked, 2)

    by_layer: dict[int, list[int]] = {layer: [] for layer in range(num_layers)}
    for l_id, e_id in pairs:
        by_layer[int(l_id)].append(int(e_id))

    if uniform:
        per_layer = total_slots // num_layers
        return {layer: by_layer[layer][:per_layer] for layer in range(num_layers)}

    selected: dict[int, list[int]] = {layer: [] for layer in range(num_layers)}
    selected_set: dict[int, set[int]] = {layer: set() for layer in range(num_layers)}
    budget = total_slots
    floor = min(min_slots_per_layer, total_slots // num_layers, num_experts)
    for layer in range(num_layers):
        for e in by_layer[layer][:floor]:
            selected[layer].append(e)
            selected_set[layer].add(e)
            budget -= 1

    for l_id, e_id in pairs:
        if budget <= 0:
            break
        layer = int(l_id)
        e = int(e_id)
        if e not in selected_set[layer] and len(selected[layer]) < num_experts:
            selected[layer].append(e)
            selected_set[layer].add(e)
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
            if "allow_cuda_graph" not in existing_attrs:
                node.attribute.append(helper.make_attribute("allow_cuda_graph", 1))
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
    save_model_fresh(model, out_onnx)
    return rewritten


def build_multidraft_target_onnx(in_onnx: Path, out_onnx: Path, verify_rows: int = 3) -> dict:
    """Extends an S=2 folded target/model.onnx to verify_rows in {3, 4} (2 or 3 chained MTP drafts)."""
    import copy  # noqa: PLC0415
    import re  # noqa: PLC0415

    from onnx import numpy_helper  # noqa: PLC0415

    if verify_rows not in (3, 4):
        raise ValueError(f"verify_rows must be 3 or 4, got {verify_rows}")

    model = onnx.load(str(in_onnx), load_external_data=False)
    graph = model.graph
    output_info = {o.name: o for o in graph.output}
    init_names = {i.name for i in graph.initializer}

    def add_graph_output(source_name: str, new_name: str, shape_override: list[int] | None = None) -> None:
        if any(o.name == new_name for o in graph.output):
            return
        info = output_info[source_name]
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(info)
        new_vi.name = new_name
        if shape_override is not None:
            del new_vi.type.tensor_type.shape.dim[:]
            for d in shape_override:
                dim = new_vi.type.tensor_type.shape.dim.add()
                dim.dim_value = d
        graph.output.append(new_vi)
        output_info[new_name] = new_vi

    def layer_of(name: str, suffix: str) -> str:
        m = re.fullmatch(r"present\.(\d+)\." + re.escape(suffix), name)
        if not m:
            raise ValueError(f"Cannot parse layer from {name!r}")
        return m.group(1)

    mids_added = 0
    fold_mid_idx = -1
    for idx, node in enumerate(graph.node):
        if node.name == "model/hyper_connection_mixer/Slice_last2":
            if "const_last_starts" not in init_names:
                graph.initializer.append(
                    numpy_helper.from_array(
                        __import__("numpy").asarray([-verify_rows], dtype=__import__("numpy").int64),
                        name="const_last_starts",
                    )
                )
                init_names.add("const_last_starts")
            node.input[1] = "const_last_starts"
        if node.domain == "qwen.fold" and node.op_type in ("FoldMid", "FoldMid2"):
            keep_a = [a for a in node.attribute if a.name != "head_rows"]
            del node.attribute[:]
            node.attribute.extend(keep_a)
            node.attribute.append(helper.make_attribute("head_rows", verify_rows))
            fold_mid_idx = idx
            break
        if node.domain == "qwen.fused_attn" and node.op_type in ("GdnProjPrep", "GdnConvPrep"):
            state_idx = 4 if node.op_type == "GdnProjPrep" else 3
            src_out = node.output[state_idx]
            lid = layer_of(src_out, "conv_state")
            for m_tag in ("present_mid2", "present_mid3")[: verify_rows - 2]:
                out_name = f"{m_tag}.{lid}.conv_state"
                node.output.append(out_name)
                add_graph_output(src_out, out_name)
                mids_added += 1
        elif node.domain == "qwen.fused_attn" and node.op_type == "GdnRecurrent":
            src_out = node.output[1]
            lid = layer_of(src_out, "recurrent_state")
            for m_tag in ("present_mid2", "present_mid3")[: verify_rows - 2]:
                out_name = f"{m_tag}.{lid}.recurrent_state"
                node.output.append(out_name)
                add_graph_output(src_out, out_name)
                mids_added += 1
        elif node.domain == "qwen.fused_attn" and node.op_type == "PleMidState":
            for m_tag in ("present_mid2", "present_mid3")[: verify_rows - 2]:
                st_name = f"{m_tag}.1.ple_conv_state"
                cx_name = f"{m_tag}.1.ple_context"
                node.output.extend([st_name, cx_name])
                add_graph_output("present.1.ple_conv_state", st_name)
                add_graph_output("present.1.ple_context", cx_name)
                mids_added += 2
        elif node.domain == "qwen.capture" and node.op_type == "PackedQSAIndexer":
            src_buf, src_len = node.output[3], node.output[5]
            lid = layer_of(src_buf, "index_buffer")
            for m_tag in ("present_mid2", "present_mid3")[: verify_rows - 2]:
                b_name = f"{m_tag}.{lid}.index_buffer"
                l_name = f"{m_tag}.{lid}.index_lengths"
                node.output.extend([b_name, l_name])
                add_graph_output(src_buf, b_name)
                add_graph_output(src_len, l_name)
                mids_added += 2

    if fold_mid_idx < 0:
        raise ValueError("FoldMid2 not found in target graph")

    for init in graph.initializer:
        if init.name == "head.capture.cumulative_sequence_lengths":
            init.CopyFrom(
                numpy_helper.from_array(
                    __import__("numpy").asarray([0, verify_rows], dtype=__import__("numpy").int32),
                    name="head.capture.cumulative_sequence_lengths",
                )
            )
    if "head.capture.cu_seqlens_s1" not in init_names:
        graph.initializer.append(
            numpy_helper.from_array(
                __import__("numpy").asarray([0, 1], dtype=__import__("numpy").int32),
                name="head.capture.cu_seqlens_s1",
            )
        )
        init_names.add("head.capture.cu_seqlens_s1")

    stage1_nodes = list(graph.node[fold_mid_idx + 1 :])
    del graph.node[fold_mid_idx + 1 :]

    stage_template = [
        copy.deepcopy(n) for n in stage1_nodes if n.op_type not in ("RowSelectBf16x3", "Concat", "FoldTail")
    ]
    stage1_produced = {o for n in stage1_nodes for o in n.output if o}

    s1_val_inputs = []
    for node in stage1_nodes:
        if node.op_type == "Concat" and node.name == "head.validation/aggregate":
            s1_val_inputs = list(node.input)
            continue
        if node.op_type == "FoldTail":
            continue
        if node.domain == "qwen.capture" and node.op_type == "PackedQSAIndexer":
            node.output[2] = "head.present_s1.0.index_key"
            add_graph_output("head.present.0.index_key", "head.present_s1.0.index_key")
            for m_tag in ("head.present_mid2", "head.present_mid3")[: verify_rows - 2]:
                b_name = f"{m_tag}.0.index_buffer"
                l_name = f"{m_tag}.0.index_lengths"
                node.output.extend([b_name, l_name])
                add_graph_output("head.present.0.index_buffer", b_name)
                add_graph_output("head.present.0.index_lengths", l_name)
        elif node.domain == "qwen.static" and node.op_type == "GuardedDSA":
            node.output[1] = "head.present_s1.0.key"
            node.output[2] = "head.present_s1.0.value"
            add_graph_output("head.present.0.key", "head.present_s1.0.key")
            add_graph_output("head.present.0.value", "head.present_s1.0.value")
        elif node.domain == "qwen.mtp" and node.op_type == "Bf16ArgMax":
            node.output[0] = "head.s1.next_token"
        graph.node.append(node)

    all_val_inputs = list(s1_val_inputs)
    num_extra_stages = verify_rows - 2

    for extra_idx in range(num_extra_stages):
        stage_num = extra_idx + 2
        prefix = f"head.s{stage_num}."
        is_last_stage = extra_idx == num_extra_stages - 1
        prev_tok = "head.s1.next_token" if stage_num == 2 else "head.s2.next_token"
        prev_hidden = "head.hidden_states_out" if stage_num == 2 else "head.s2.head.hidden_states_out"
        prev_key = "head.present_s1.0.key" if stage_num == 2 else "head.present_s2.0.key"
        prev_val = "head.present_s1.0.value" if stage_num == 2 else "head.present_s2.0.value"
        prev_ikey = "head.present_s1.0.index_key" if stage_num == 2 else "head.present_s2.0.index_key"

        if stage_num == 2:
            prep_inputs = [
                prev_tok,
                "fold.ctl",
                "fold.sel",
                "head.mtp.rotary_emb.cos_cache",
                "head.mtp.rotary_emb.sin_cache",
                "head.present_mid.0.index_buffer",
                "head.present_mid.0.index_lengths",
                "head.present_mid2.0.index_buffer",
                "head.present_mid2.0.index_lengths",
            ]
            if verify_rows == 4:
                prep_inputs.extend(["head.present_mid3.0.index_buffer", "head.present_mid3.0.index_lengths"])
            prep_inputs.extend(["head.present.0.index_buffer", "head.present.0.index_lengths"])
        else:
            prep_inputs = [
                prev_tok,
                "fold.ctl",
                "fold.sel",
                "head.mtp.rotary_emb.cos_cache",
                "head.mtp.rotary_emb.sin_cache",
                "head.s2.out_index_buffer",
                "head.s2.out_index_lengths",
            ]

        prep_outputs = [
            f"{prefix}head_ids",
            f"{prefix}head_c",
            f"{prefix}slot_ids",
            f"{prefix}seqlens_k",
            f"{prefix}rope_cos",
            f"{prefix}rope_sin",
            f"{prefix}in_index_buffer",
            f"{prefix}in_index_lengths",
        ]
        graph.node.append(
            helper.make_node(
                "FoldDraftPrep",
                prep_inputs,
                prep_outputs,
                domain="qwen.fold",
                name=f"fold/FoldDraftPrep_s{stage_num}",
                stage=stage_num - 1,
            )
        )

        input_remap = {
            "fold.head_ids": f"{prefix}head_ids",
            "hidden_states": prev_hidden,
            "fold.head_hidden": prev_hidden,
            "head.capture.rope_cos": f"{prefix}rope_cos",
            "head.capture.rope_sin": f"{prefix}rope_sin",
            "head.capture.cumulative_sequence_lengths": "head.capture.cu_seqlens_s1",
            "fold.head_c": f"{prefix}head_c",
            "head.capture.slot_ids": f"{prefix}slot_ids",
            "head.capture.seqlens_k": f"{prefix}seqlens_k",
            "head.past_key_values.0.index_key": prev_ikey,
            "head.past_key_values.0.index_buffer": f"{prefix}in_index_buffer",
            "head.past_key_values.0.index_lengths": f"{prefix}in_index_lengths",
            "head.past_key_values.0.key": prev_key,
            "head.past_key_values.0.value": prev_val,
            "head.v_mtp.layers.0.self_attn.Mul_200.sel": f"{prefix}head.v_mtp.layers.0.self_attn.Mul_200",
            "head.v_mtp.Reshape_23.sel": f"{prefix}head.v_mtp.Reshape_23",
            "head.v_mtp.layers.0.attn_hyper_connection.Mul_117.sel": f"{prefix}head.v_mtp.layers.0.attn_hyper_connection.Mul_117",
        }

        for tmpl in stage_template:
            n = copy.deepcopy(tmpl)
            n.name = f"{prefix}{n.name}"
            for i, inp in enumerate(n.input):
                if not inp:
                    continue
                if inp in input_remap:
                    n.input[i] = input_remap[inp]
                elif inp in stage1_produced and inp not in init_names:
                    n.input[i] = f"{prefix}{inp}"
            for i, out in enumerate(n.output):
                if out:
                    n.output[i] = f"{prefix}{out}"

            if n.domain == "qwen.capture" and n.op_type == "PackedQSAIndexer":
                del n.output[7:]
                if is_last_stage:
                    n.output[2] = "head.present.0.index_key"
                else:
                    n.output[2] = f"head.present_s{stage_num}.0.index_key"
                    add_graph_output("head.present.0.index_key", n.output[2])
                n.output[3] = f"{prefix}out_index_buffer"
                n.output[5] = f"{prefix}out_index_lengths"
                n.output[6] = f"{prefix}validation.qsa"
                all_val_inputs.append(n.output[6])
            elif n.domain == "qwen.static" and n.op_type == "GuardedDSA":
                if is_last_stage:
                    n.output[1] = "head.present.0.key"
                    n.output[2] = "head.present.0.value"
                else:
                    n.output[1] = f"head.present_s{stage_num}.0.key"
                    n.output[2] = f"head.present_s{stage_num}.0.value"
                    add_graph_output("head.present.0.key", n.output[1])
                    add_graph_output("head.present.0.value", n.output[2])
                n.output[3] = f"{prefix}validation.dsa"
                all_val_inputs.append(n.output[3])
            elif n.domain == "qwen.mtp" and n.op_type == "Bf16ArgMax":
                n.output[0] = "head.next_token" if is_last_stage else f"{prefix}next_token"

            graph.node.append(n)

    for o in graph.output:
        if o.name == "head.validation.aggregate":
            del o.type.tensor_type.shape.dim[:]
            dim = o.type.tensor_type.shape.dim.add()
            dim.dim_value = len(all_val_inputs)

    graph.node.append(
        helper.make_node(
            "Concat",
            all_val_inputs,
            ["head.validation.aggregate"],
            name="head.validation/aggregate",
            axis=0,
        )
    )
    tail_ctl_slot = 8 if verify_rows == 3 else 9
    graph.node.append(
        helper.make_node(
            "FoldTail",
            ["head.next_token", "head.validation.aggregate", "fold.ctl"],
            ["fold.tail_done"],
            domain="qwen.fold",
            name="fold/FoldTail",
            ctl_slot=tail_ctl_slot,
        )
    )

    del graph.value_info[:]
    save_model_fresh(model, out_onnx)
    return {
        "verify_rows": verify_rows,
        "mids_added": mids_added,
        "extra_head_stages": num_extra_stages,
        "total_nodes": len(graph.node),
        "total_outputs": len(graph.output),
    }


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
    embed_pool_ids = set()
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
            if name.endswith("embed_tokens.weight"):
                embed_pool_ids.add(pid)
            if sess == "dense":
                dense_bound_names.add(name)
            session_lines.append((sess, name, pid))

    aligned_mtp_rel = "target/head_data/source_weights/mtp_moe_aligned.bin"
    need_aligned_mtp_copy = False
    for line in src_pool[1:]:
        parts = line.strip().split()
        if parts and parts[0] in ("pool", "host_pool") and int(parts[1]) in mtp_moe_pool_ids:
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
        if parts[0] in ("pool", "host_pool"):
            pid = int(parts[1])
            rank = int(parts[5])
            base_tokens = ["pool", *parts[1 : 7 + rank]]
            base_s = " ".join(base_tokens)
            default_placement = (
                "host_mapped" if parts[0] == "host_pool" else (parts[7 + rank] if len(parts) > 7 + rank else "device")
            )
            if pid in qmoe_pool_ids:
                pool_lines.append(f"{base_s} cpu")
            elif pid in embed_pool_ids:
                if embed_placement != "device":
                    pool_lines.append(f"{base_s} {embed_placement}")
                else:
                    pool_lines.append(base_s)
            elif pid in mtp_moe_pool_ids and mtp_moe_placement != "device":
                if not need_aligned_mtp_copy:
                    pool_lines.append(f"{base_s} {mtp_moe_placement}")
                elif pid == 607:
                    pool_lines.append(f"pool 607 16 0 3355443200 3 512 1280 2560 {aligned_mtp_rel} {mtp_moe_placement}")
                elif pid == 608:
                    pool_lines.append(
                        f"pool 608 16 3355443200 1677721600 3 512 2560 640 {aligned_mtp_rel} {mtp_moe_placement}"
                    )
            elif default_placement != "device":
                pool_lines.append(f"{base_s} {default_placement}")
            else:
                pool_lines.append(base_s)

    next_pool_id = len(pool_lines)
    target_bound_names = {name for sess, name, _ in session_lines if sess == "target"}
    flat_root = (source_pkg / "model.onnx").exists()
    target_onnx_src = (source_pkg / "model.onnx") if flat_root else (source_pkg / "target/model.onnx")
    target_model = onnx.load(str(target_onnx_src), load_external_data=False)
    for init in target_model.graph.initializer:
        if init.name in target_bound_names:
            continue
        if ".mlp.fc1_" in init.name or ".mlp.fc2_" in init.name:
            ext = {e.key: e.value for e in init.external_data}
            loc = ext.get("location", "")
            if not loc:
                continue
            offset = int(ext.get("offset", "0"))
            length = int(ext["length"])
            dims = list(init.dims)
            shape_str = f"{len(dims)} " + " ".join(str(d) for d in dims)
            pool_file = loc if flat_root else f"target/{loc}"
            pool_lines.append(f"pool {next_pool_id} {init.data_type} {offset} {length} {shape_str} {pool_file} cpu")
            session_lines.append(("target", init.name, next_pool_id))
            qmoe_pool_ids.add(next_pool_id)
            next_pool_id += 1

    added_dense_entries = 0
    if dense_placement in ("host_mapped", "cpu") and (source_pkg / "dense/model.onnx").exists():
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
    parser.add_argument(
        "--migrate-shared-onnx", type=Path, default=None, help="Input ONNX for standalone v3 shared migration"
    )
    parser.add_argument("--out-onnx", type=Path, default=None, help="Output ONNX for standalone v3 shared migration")
    parser.add_argument("--strip-phase", action="store_true")
    parser.add_argument("--strip-stats", action="store_true")

    parser.add_argument("--from-package", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plane-lib", type=Path, default=None)
    parser.add_argument("--provider-lib", type=Path, default=None)
    parser.add_argument(
        "--ops-dir",
        type=Path,
        default=None,
        help="Directory containing rebased libqwen_*.so custom op libraries",
    )
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
    parser.add_argument(
        "--dense-gpu-qmoe",
        action="store_true",
        default=True,
        help="Keep stock GPU QMoE in dense/model.onnx using UVA-mapped expert weights",
    )
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
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise SystemExit("--out must not be equal to or nested inside --from-package")
    out.mkdir(parents=True, exist_ok=True)

    linked = link_tree(source, out, args.copy)
    if args.ops_dir is not None and args.ops_dir.exists():
        for so_file in sorted(args.ops_dir.glob("libqwen_*.so")):
            link_file(so_file.resolve(), out / "libs" / so_file.name, copy=True)
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
    flat_root = (source / "model.onnx").exists()
    target_src = (source / "model.onnx") if flat_root else (source / "target/model.onnx")
    target_out_dir = out if flat_root else (out / "target")
    target_rewritten = rewrite_nvfp4_qmoe_to_offload(
        target_src,
        target_out_dir / "model_s2.onnx",
        extra_attrs=target_attrs,
        per_layer_initial_experts=profile_plan,
    )
    build_multidraft_target_onnx(target_out_dir / "model_s2.onnx", target_out_dir / "model_s3.onnx", verify_rows=3)
    build_multidraft_target_onnx(target_out_dir / "model_s2.onnx", target_out_dir / "model_s4.onnx", verify_rows=4)
    link_file(target_out_dir / "model_s4.onnx", target_out_dir / "model.onnx", copy=True)
    if (source / "dense/model.onnx").exists():
        if args.dense_gpu_qmoe:
            link_file(source / "dense/model.onnx", out / "dense/model.onnx", args.copy)
            dense_rewritten = 0
        else:
            dense_rewritten = rewrite_nvfp4_qmoe_to_offload(
                source / "dense/model.onnx", out / "dense/model.onnx", extra_attrs=dense_attrs
            )
    else:
        dense_rewritten = 0
        for obsolete_dir in ("target", "head_r2", "dense", "head_kv") if flat_root else ("dense", "head_kv"):
            if (out / obsolete_dir).exists():
                shutil.rmtree(out / obsolete_dir)
    pool_stats = build_offload_weights_pool(
        source, out, args.dense_placement, args.mtp_moe_placement, args.embed_placement
    )

    config = json.loads((source / "genai_config.json").read_text())
    if args.stop_token_id is not None:
        config["model"]["eos_token_id"] = [args.stop_token_id]

    cuda_opts = config["model"]["decoder"]["session_options"]["provider_options"][0]["cuda"]
    cuda_opts["enable_cuda_graph"] = "1"
    cuda_opts["arena_extend_strategy"] = "kSameAsRequested"
    cuda_opts["arena.extend_strategy"] = "1"
    if args.vram_limit_gib is not None and args.vram_limit_gib > 0:
        cuda_opts["gpu_device_mem_limit"] = str(int(args.vram_limit_gib * (1024**3)))

    qwen = config["model"]["qwen38_graph"]
    fold_head_key = "mtp" if "mtp" in qwen["fold"] else ("drafter" if "drafter" in qwen["fold"] else "head_r2")
    qwen["fold"][fold_head_key]["session_options"] = {
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
    if "dense" in qwen["prefill"]:
        dense_cuda = qwen["prefill"]["dense"]["session_options"]["provider_options"][0]["cuda"]
        dense_cuda["arena_extend_strategy"] = "kSameAsRequested"
        dense_cuda["arena.extend_strategy"] = "1"
    if "head_kv" in qwen["prefill"]:
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
