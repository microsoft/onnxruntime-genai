#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Zero-copy QMoE-offload package builder: standard com.microsoft::QMoE nodes plus session.moe.offload.* options.

The target runtime is the ONNX Runtime CUDA provider with QMoE offload (microsoft/onnxruntime#33450 -> #33622). It has no
QMoEOffload operator (creating a session with one fails) and the QMoE schema has no offload attribute: exporters emit
standard com.microsoft::QMoE nodes and the offload is configured with `session.moe.offload.*` session options.

NVFP4 v12b package conversion (--from-package <pkg_dir> --out <dir>):
   - Hard-links all weight files (0 GiB disk copy).
   - Checks that the runtime can offload the 48 com.microsoft::QMoE nodes of target/model.onnx (or model.onnx) and keeps
     them as they are, apart from clearing the router_weights input (input 14), which the offload backend does not take.
     The graph is written as model_s2.onnx; model_s3.onnx and model_s4.onnx extend it to 3 and 4 verify rows.
   - Updates weights_pool.txt so the 48 layers of routed expert weights/scales (63.281 GiB) use "cpu"
     mmap placement, and optionally places dense/source_weights/dense.data (5.270 GiB) in "host_mapped"
     UVA placement for 24 GiB VRAM targets.
   - Configures genai_config.json for 24 GiB or 32 GiB VRAM operation and sets the decoder's session.moe.offload.*
     options: mode, placement, the byte budgets, the expert-popularity file and per-layer resident-slot caps.
     The model.qwen38_graph section of the source is moved to the config surface the engine accepts (see
     package_config.py): the MTP head session is fold.head_r2, and there are no CUDA graph ids.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from pathlib import Path

import onnx
from onnx import helper
from package_config import migrate_qwen38_graph_config
from package_paths import reject_output_overlap

OFFLOAD_MODES = ("required", "auto")
OFFLOAD_PLACEMENTS = ("adaptive", "static")
OFFLOAD_OPTION_PREFIX = "session.moe.offload."
EXPERT_PROFILE_NAME = "expert-profile.bin"
ALIGNED_MTP_RELATIVE = "target/head_data/source_weights/mtp_moe_aligned.bin"
ROUTER_WEIGHTS_INPUT = 14
# Inputs of the QMoE schema that the offload backend does not take: fc3 and zero-point banks, router_weights and the
# activation scales.
UNSUPPORTED_OFFLOAD_INPUTS = (8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20)
OFFLOAD_WEIGHT_INPUTS = (2, 3, 5, 6)
NVFP4_GLOBAL_SCALE_INPUTS = (15, 16)

REMOVED_OPTIONS = {
    "--migrate-shared-onnx": "it rewrites legacy 26-input QMoEOffload graphs, and the runtime has no QMoEOffload operator. "
    "No data-preserving conversion of their expert_blocks input to standard QMoE exists; INT4 packages come from "
    "build_package_v12d_int4_int8.py",
    "--out-onnx": "it belongs to --migrate-shared-onnx",
    "--strip-phase": "it belongs to --migrate-shared-onnx",
    "--strip-stats": "it belongs to --migrate-shared-onnx",
    "--target-spare-slots": "spare slots are the runtime's decision; choose --offload-placement static (none) or adaptive",
    "--dense-offload-qmoe": "it rewrote dense/model.onnx to the QMoEOffload operator, which the runtime does not have; "
    "the dense graph keeps the stock QMoE on UVA-mapped expert weights",
    "--dense-stream-min-rows": "it set attributes of the QMoEOffload operator, which the runtime does not have",
    "--dense-stream-wave": "it set attributes of the QMoEOffload operator, which the runtime does not have",
}

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


def load_expert_profile_plan(
    profile_path: Path,
    num_layers: int = 48,
    num_experts: int = 512,
    total_slots: int = 960,
    min_slots_per_layer: int = 4,
    uniform: bool = False,
) -> dict[int, list[int]]:
    """Loads an STRP v1 expert-profile.bin and picks the hottest experts of every layer, `total_slots` in total.

    The profile ranks (layer, expert) pairs from the most to the least popular, and every layer keeps a prefix of its
    own ranking. Uniform: each layer gets total_slots // num_layers experts, and each of the total_slots % num_layers
    remaining slots goes to a different layer, the ones whose next expert ranks highest. Otherwise every layer first
    gets min_slots_per_layer (at most total_slots // num_layers), then the globally most popular experts take the rest.
    A request the profile cannot fill raises ValueError, so the plan always sums to exactly `total_slots`.
    """
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    if num_layers <= 0 or num_experts <= 0:
        raise ValueError(f"num_layers and num_experts must be positive, got {num_layers} and {num_experts}")
    if total_slots < 0 or min_slots_per_layer < 0:
        raise ValueError(
            f"total_slots and min_slots_per_layer must not be negative: {total_slots}, {min_slots_per_layer}"
        )

    data = profile_path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{profile_path} is {len(data)} bytes, shorter than the 24-byte STRP header")
    magic, version, hdr_layers, hdr_experts, _slots, n_ranked = struct.unpack("<6I", data[:24])
    if magic != 1347572819 or version != 1 or hdr_layers != num_layers or hdr_experts != num_experts:
        raise ValueError(
            f"Incompatible expert profile header in {profile_path}: {(magic, version, hdr_layers, hdr_experts)}"
        )
    if len(data) < 24 + n_ranked * 4:
        raise ValueError(f"{profile_path} ranks {n_ranked} experts but holds only {(len(data) - 24) // 4}")
    pairs = np.frombuffer(data[24 : 24 + n_ranked * 4], dtype=np.uint16).reshape(n_ranked, 2)
    if n_ranked and (int(pairs[:, 0].max()) >= num_layers or int(pairs[:, 1].max()) >= num_experts):
        raise ValueError(f"{profile_path} ranks a layer or expert outside {num_layers} layers x {num_experts} experts")

    ranked: list[list[int]] = [[] for _ in range(num_layers)]
    position: list[list[int]] = [[] for _ in range(num_layers)]
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for rank, (layer, expert) in enumerate(pairs.tolist()):
        if (layer, expert) not in seen:
            seen.add((layer, expert))
            ranked[layer].append(expert)
            position[layer].append(rank)
            order.append((layer, expert))

    if total_slots > len(order):
        raise ValueError(f"{total_slots} resident slots requested but {profile_path} ranks only {len(order)} experts")

    if uniform:
        base, remainder = divmod(total_slots, num_layers)
        short = [layer for layer in range(num_layers) if len(ranked[layer]) < base]
        if short:
            raise ValueError(
                f"{base} slots per layer requested but layers {short[:8]} rank fewer experts in the profile"
            )
        candidates = sorted(
            (layer for layer in range(num_layers) if len(ranked[layer]) > base), key=lambda layer: position[layer][base]
        )
        if len(candidates) < remainder:
            raise ValueError(
                f"{total_slots} slots over {num_layers} layers leave {remainder} extra slots, but only "
                f"{len(candidates)} layers rank more than {base} experts"
            )
        extra = set(candidates[:remainder])
        selected = {layer: ranked[layer][: base + (layer in extra)] for layer in range(num_layers)}
    else:
        floor = min(min_slots_per_layer, total_slots // num_layers)
        selected = {layer: ranked[layer][:floor] for layer in range(num_layers)}
        chosen = {layer: set(experts) for layer, experts in selected.items()}
        budget = total_slots - sum(len(experts) for experts in selected.values())
        for layer, expert in order:
            if budget <= 0:
                break
            if expert not in chosen[layer]:
                selected[layer].append(expert)
                chosen[layer].add(expert)
                budget -= 1

    planned = sum(len(experts) for experts in selected.values())
    if planned != total_slots:
        raise RuntimeError(f"expert profile plan holds {planned} slots, not the {total_slots} requested")
    return selected


def qmoe_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.domain == "com.microsoft" and node.op_type == "QMoE"]


def node_label(name: str, ordinal: int) -> str:
    return repr(name) if name else f"#{ordinal} (unnamed)"


def qmoe_attribute(node: onnx.NodeProto, name: str, default=None):
    """Reads a node attribute like the runtime does: strings come back as str and an absent one as `default`."""
    for attribute in node.attribute:
        if attribute.name == name:
            value = helper.get_attribute_value(attribute)
            return value.decode() if isinstance(value, bytes) else value
    return default


def parse_layer_index(name: str) -> int | None:
    """Mirrors the runtime's layer-number extraction: digits after "layers.", "layer." or "blocks." that end the name or
    are followed by a non-alphanumeric character, e.g. 5 in "model.layers.5.mlp". Returns None when there is none."""
    for prefix in ("layers.", "layer.", "blocks."):
        search_from = 0
        while search_from < len(name):
            position = name.find(prefix, search_from)
            if position < 0:
                break
            end = position + len(prefix)
            digits_end = end
            while digits_end < len(name) and name[digits_end] in "0123456789":
                digits_end += 1
            if digits_end > end:
                value = int(name[end:digits_end])
                if value > 2**31 - 1:
                    return None
                following = name[digits_end] if digits_end < len(name) else ""
                if not following or not (following.isascii() and following.isalnum()):
                    return value
            search_from = end
    return None


def qmoe_offload_problems(node: onnx.NodeProto, constants: dict[str, onnx.TensorProto]) -> list[str]:
    """Lists the reasons the runtime's QMoE offload (onnxruntime#33622, moe_quantization.cc) rejects `node`.

    `constants` maps the names of the graph's constant initializers to them. An empty list means the node is eligible;
    `session.moe.offload.mode=required` fails session creation for a node with reasons, and `auto` runs that node
    without offload.
    """
    problems = []

    def slot(index: int) -> str:
        return node.input[index] if index < len(node.input) else ""

    quant_type = qmoe_attribute(node, "quant_type", "int")
    bits = qmoe_attribute(node, "expert_weight_bits", 4)
    fc1_bits = qmoe_attribute(node, "fc1_expert_weight_bits", bits)
    fc2_bits = qmoe_attribute(node, "fc2_expert_weight_bits", bits)
    block_size = qmoe_attribute(node, "block_size", -1)
    prepacked = qmoe_attribute(node, "weights_prepacked", -1)
    nvfp4_row_major = quant_type == "nvfp4" and prepacked == 1
    int4_prepacked = quant_type == "int" and bits == fc1_bits == fc2_bits == 4 and prepacked != 0 and block_size == 32
    int3_raw = (
        quant_type == "int" and bits == fc1_bits == fc2_bits == 3 and prepacked == 0 and block_size in (32, 64, 128)
    )
    if quant_type == "nvfp4" and block_size not in (-1, 16):
        problems.append(f"block_size={block_size}: quant_type='nvfp4' takes block_size 16")
    if not (nvfp4_row_major or int4_prepacked or int3_raw):
        problems.append(
            f"quant_type={quant_type!r} expert_weight_bits={bits} block_size={block_size} weights_prepacked={prepacked} "
            "is not NVFP4 with weights_prepacked=1, INT4 with block_size=32 and prepacked weights, or raw INT3"
        )
    elif not (bits == 4 or int3_raw):
        problems.append(f"expert_weight_bits={bits} must be 4 for NVFP4")

    activation = qmoe_attribute(node, "activation_type", "relu")
    swiglu_fusion = qmoe_attribute(node, "swiglu_fusion", 0)
    if activation != "swiglu" or swiglu_fusion not in (0, 1):
        problems.append(
            f"activation_type={activation!r} swiglu_fusion={swiglu_fusion}: only swiglu with swiglu_fusion 0 or 1 is offloaded"
        )
    if qmoe_attribute(node, "use_sparse_mixer", 0) == 1:
        problems.append("use_sparse_mixer=1 is not offloaded")

    unsupported = [index for index in UNSUPPORTED_OFFLOAD_INPUTS if slot(index)]
    if unsupported:
        problems.append(f"inputs {unsupported} are not supported by the offload backend")

    not_constant = [index for index in OFFLOAD_WEIGHT_INPUTS if slot(index) not in constants]
    if not_constant:
        problems.append(f"inputs {not_constant} (expert weights and scales) must be constant graph initializers")
    else:
        for index in (2, 5):
            rank = len(constants[slot(index)].dims)
            if rank != 3:
                problems.append(f"input {index} has rank {rank}, expected 3 (experts x rows x columns)")
    if quant_type == "nvfp4":
        missing = [index for index in NVFP4_GLOBAL_SCALE_INPUTS if slot(index) not in constants]
        if missing:
            problems.append(
                f"NVFP4 needs the constant global scales {list(NVFP4_GLOBAL_SCALE_INPUTS)}, missing {missing}"
            )
    return problems


def prepare_qmoe_for_offload(model: onnx.ModelProto) -> dict:
    """Readies the com.microsoft::QMoE nodes of `model` (edited in place) for the runtime's offload backend.

    The runtime offloads standard QMoE nodes and has no offload operator or attribute, so the nodes keep their op type and
    attributes. The only edit is clearing the optional router_weights input (14), which the backend does not take: the
    nodes then weight the selected experts by router_probs, like the fresh nodes of build_package_v12d_int4_int8.py.
    Returns {"nodes": the names of the QMoE nodes in graph order, "router_weights_cleared": count,
    "rejected": {position in "nodes": reasons}} for the nodes the backend would still reject.
    """
    graph_inputs = {value.name for value in model.graph.input}
    # An initializer that is also a graph input can be overridden at run time, so the runtime does not treat it as constant.
    constants = {
        init.name: init for init in model.graph.initializer if model.ir_version < 4 or init.name not in graph_inputs
    }
    nodes = qmoe_nodes(model)
    cleared = 0
    for node in nodes:
        if len(node.input) > ROUTER_WEIGHTS_INPUT and node.input[ROUTER_WEIGHTS_INPUT]:
            node.input[ROUTER_WEIGHTS_INPUT] = ""
            cleared += 1
    rejected = {}
    for ordinal, node in enumerate(nodes):
        problems = qmoe_offload_problems(node, constants)
        if problems:
            rejected[ordinal] = problems
    return {"nodes": [node.name for node in nodes], "router_weights_cleared": cleared, "rejected": rejected}


def format_node_overrides(caps: dict[str, int]) -> str:
    """Formats per-node resident-slot caps as the value of session.moe.offload.node_overrides.

    The runtime splits entries at ';', takes the name up to the first ':' and trims ASCII whitespace around it, so a name
    that has one of those cannot be written.
    """
    entries = []
    for name, cap in caps.items():
        if not name or name != name.strip(" \t\n\v\f\r") or ";" in name or ":" in name:
            raise ValueError(f"node name {name!r} cannot appear in session.moe.offload.node_overrides")
        if not 0 <= cap <= 2**31 - 1:
            raise ValueError(f"resident slot cap {cap} of node {name!r} is outside 0..{2**31 - 1}")
        entries.append(f"{name}:max_resident_slots={cap}")
    return ";".join(entries)


def offload_session_options(
    mode: str,
    placement: str,
    device_expert_bytes: str | None = None,
    host_expert_bytes: str | None = None,
    popularity_file: str | None = None,
    node_overrides: str | None = None,
) -> dict[str, str]:
    """The session.moe.offload.* entries (all values are strings) that configure the runtime's QMoE offload."""
    if mode not in OFFLOAD_MODES:
        raise ValueError(f"offload mode {mode!r} is not one of {OFFLOAD_MODES}")
    if placement not in OFFLOAD_PLACEMENTS:
        raise ValueError(f"offload placement {placement!r} is not one of {OFFLOAD_PLACEMENTS}")
    options = {"mode": mode, "placement": placement, "cpu_execution": "auto"}
    optional = {
        "device_expert_bytes": device_expert_bytes,
        "host_expert_bytes": host_expert_bytes,
        "popularity_file": popularity_file,
        "node_overrides": node_overrides,
    }
    options.update({key: value for key, value in optional.items() if value})
    return {OFFLOAD_OPTION_PREFIX + key: value for key, value in options.items()}


def budget_bytes(text: str) -> str:
    """argparse type of the byte budgets: "auto" or a non-negative integer, as the runtime parses them."""
    if text.lower() == "auto":
        return "auto"
    if text.isascii() and text.isdigit() and int(text) < 2**64:
        return str(int(text))
    raise argparse.ArgumentTypeError(f"{text!r} is not 'auto' or a number of bytes between 0 and {2**64 - 1}")


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


def check_pool_bindings(pool_ids: list[int], sessions: list[tuple[str, str, int]], where: str) -> None:
    """Raises ValueError unless every pool id is defined once and every session binding names a defined pool."""
    duplicated = sorted(pid for pid, count in collections.Counter(pool_ids).items() if count > 1)
    if duplicated:
        raise ValueError(f"{where}: pool ids {duplicated} are defined more than once")
    defined = set(pool_ids)
    dangling = [(sess, name, pid) for sess, name, pid in sessions if pid not in defined]
    if dangling:
        shown = ", ".join(f"session {sess} {name} {pid}" for sess, name, pid in dangling[:5])
        raise ValueError(f"{where}: {len(dangling)} session bindings name a pool that is not defined, e.g. {shown}")


def plan_aligned_copy(lengths: list[tuple[int, int]], alignment: int = 4096) -> tuple[dict[int, int], int]:
    """Packs ranges back to back, each starting on an `alignment` boundary. Takes (key, length) pairs in order and
    returns ({key: destination offset}, size of the packed file)."""
    offsets: dict[int, int] = {}
    cursor = 0
    for key, length in lengths:
        cursor = -(-cursor // alignment) * alignment
        offsets[key] = cursor
        cursor += length
    return offsets, cursor


def check_file_ranges(ranges: list[tuple[int, Path, int, int, int]]) -> None:
    """Raises ValueError naming the pool whose (pool id, source file, source offset, _, length) range is not in its file."""
    for pid, src, src_offset, _dst_offset, length in ranges:
        if not src.is_file():
            raise ValueError(f"pool {pid}: source file {src} does not exist")
        size = src.stat().st_size
        if src_offset < 0 or length < 0 or src_offset + length > size:
            raise ValueError(f"pool {pid}: bytes {src_offset}..{src_offset + length} lie outside {src} ({size} bytes)")


def copy_file_ranges(dst: Path, ranges: list[tuple[int, Path, int, int, int]]) -> None:
    """Writes `dst` from (pool id, source file, source offset, destination offset, length) ranges.

    Every range is checked against its source file before `dst` is created, so an out-of-bounds pool fails with its id
    instead of leaving a short file.
    """
    check_file_ranges(ranges)
    with open(dst, "wb") as fout:
        for _pid, src, src_offset, dst_offset, length in ranges:
            with open(src, "rb") as fin:
                remaining = length
                while remaining > 0:
                    copied = os.copy_file_range(
                        fin.fileno(),
                        fout.fileno(),
                        min(remaining, 1 << 26),
                        offset_src=src_offset + length - remaining,
                        offset_dst=dst_offset + length - remaining,
                    )
                    if copied <= 0:
                        raise RuntimeError(f"os.copy_file_range stopped early copying {src}")
                    remaining -= copied
        fout.truncate(max((dst_offset + length for _pid, _src, _off, dst_offset, length in ranges), default=0))


def relocate_unaligned_mtp_pools(
    source_pkg: Path, out_pkg: Path, pool_records: list[list[str]], mtp_pool_ids: set[int], placement: str
) -> dict[int, str]:
    """Moves every MTP MoE pool that starts at a non-16-byte-aligned file offset into one aligned file.

    `pool_records` holds the split `pool`/`host_pool` lines of weights_pool.txt. The ids are whatever the source package
    uses: each moved pool gets a record derived from its own line, pointing at its range of the aligned file, and the
    result maps pool id -> that record. The file sits next to the source package and is named after the layout, so a
    changed layout never reuses a stale copy; the output package hard-links it.
    """
    unaligned = sorted(
        (parts for parts in pool_records if int(parts[1]) in mtp_pool_ids and int(parts[3]) % 16 != 0),
        key=lambda parts: int(parts[1]),
    )
    if not unaligned:
        return {}
    offsets, total_size = plan_aligned_copy([(int(parts[1]), int(parts[4])) for parts in unaligned])
    ranges = [
        (int(parts[1]), source_pkg / parts[6 + int(parts[5])], int(parts[3]), offsets[int(parts[1])], int(parts[4]))
        for parts in unaligned
    ]
    check_file_ranges(ranges)
    # The file name identifies the layout and the source files, so a changed package never reuses a stale copy.
    layout = []
    for (_pid, src, offset, dst_offset, length), parts in zip(ranges, unaligned, strict=True):
        stat = src.stat()
        layout.append([parts[6 + int(parts[5])], offset, length, dst_offset, stat.st_size, stat.st_mtime_ns])
    digest = hashlib.sha256(json.dumps(sorted(layout)).encode()).hexdigest()[:12]
    shared_aligned = source_pkg.parent / f"{source_pkg.name}_mtp_moe_aligned-{digest}.bin"
    if not shared_aligned.exists() or shared_aligned.stat().st_size != total_size:
        tmp_aligned = shared_aligned.with_suffix(".tmp")
        try:
            copy_file_ranges(tmp_aligned, ranges)
        except BaseException:
            tmp_aligned.unlink(missing_ok=True)
            raise
        tmp_aligned.replace(shared_aligned)
    link_file(shared_aligned, out_pkg / ALIGNED_MTP_RELATIVE, copy=False)
    records = {}
    for parts in unaligned:
        pid = int(parts[1])
        rank = int(parts[5])
        head = ["pool", parts[1], parts[2], str(offsets[pid]), parts[4], *parts[5 : 6 + rank]]
        records[pid] = " ".join([*head, ALIGNED_MTP_RELATIVE, placement])
    return records


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

    source_records = [
        parts for parts in (line.split() for line in src_pool[1:]) if parts and parts[0] in ("pool", "host_pool")
    ]
    check_pool_bindings([int(parts[1]) for parts in source_records], session_lines, "the source weights_pool.txt")
    aligned_mtp_records = {}
    if mtp_moe_placement != "device":
        aligned_mtp_records = relocate_unaligned_mtp_pools(
            source_pkg, out_pkg, source_records, mtp_moe_pool_ids, mtp_moe_placement
        )

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
                pool_lines.append(aligned_mtp_records.get(pid, f"{base_s} {mtp_moe_placement}"))
            elif default_placement != "device":
                pool_lines.append(f"{base_s} {default_placement}")
            else:
                pool_lines.append(base_s)

    next_pool_id = max((int(parts[1]) for parts in source_records), default=-1) + 1
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
    check_pool_bindings([int(line.split()[1]) for line in pool_lines], session_lines, "the rewritten weights_pool.txt")
    (out_pkg / "weights_pool.txt").write_text("\n".join(out_lines) + "\n")
    return {
        "total_pool_entries": len(pool_lines),
        "cpu_qmoe_entries": len(qmoe_pool_ids),
        "added_dense_entries": added_dense_entries,
        "dense_placement": dense_placement,
        "mtp_moe_placement": mtp_moe_placement,
        "embed_placement": embed_placement,
    }


def check_layer_numbering(model: onnx.ModelProto, not_offloaded: frozenset[int] = frozenset()) -> None:
    """Refuses a graph whose QMoE nodes the runtime would number differently from their position in the graph.

    The expert profile ranks experts by layer number, and the runtime matches them to nodes by the layer it reads from a
    node's name (digits after "layers.", "layer." or "blocks.", else the same in the name of its fc1 weights input). A node
    without one is numbered in registration order, counting only the nodes the runtime offloads.
    """
    registered: list[int] = []
    for ordinal, node in enumerate(qmoe_nodes(model)):
        if ordinal in not_offloaded:
            continue
        number = parse_layer_index(node.name)
        if number is None and len(node.input) > 2:
            number = parse_layer_index(node.input[2])
        if number is None:
            number = len(registered)
            while number in registered:
                number += 1
        if number != ordinal:
            raise ValueError(
                f"QMoE node {node_label(node.name, ordinal)} is number {ordinal} in the graph but the runtime "
                f"numbers it layer {number}"
            )
        registered.append(number)


def qmoe_expert_counts(model: onnx.ModelProto) -> set[int]:
    """The number of experts (leading dimension of the fc1 weights) of the QMoE nodes."""
    initializers = {init.name: init for init in model.graph.initializer}
    counts = set()
    for node in qmoe_nodes(model):
        weights = initializers.get(node.input[2]) if len(node.input) > 2 else None
        if weights is not None and len(weights.dims) == 3:
            counts.add(weights.dims[0])
    return counts


def plan_resident_caps(
    args: argparse.Namespace, model: onnx.ModelProto, node_count: int, not_offloaded: frozenset[int] = frozenset()
) -> dict[int, int]:
    """The resident-slot caps asked for on the command line, keyed by position in the graph's QMoE nodes.

    `--target-resident-slots N` caps every layer at N experts. With `--expert-profile` and a slot count, the profile
    decides how the slots (`--total-resident-slots`, else N per layer) are spread over the layers, and each layer is capped
    at its share; the runtime then keeps the hottest experts of the profile resident up to that cap. Without a slot count
    there is no cap and the runtime plans the residency from its byte budgets.
    """
    if args.expert_profile is None or (args.total_resident_slots is None and args.target_resident_slots is None):
        return (
            dict.fromkeys(range(node_count), args.target_resident_slots)
            if args.target_resident_slots is not None
            else {}
        )
    counts = qmoe_expert_counts(model)
    if len(counts) != 1:
        raise ValueError(f"--expert-profile needs QMoE nodes with one expert count, the graph has {sorted(counts)}")
    check_layer_numbering(model, not_offloaded)
    total = args.total_resident_slots
    if total is None:
        total = args.target_resident_slots * node_count
    try:
        plan = load_expert_profile_plan(
            args.expert_profile.resolve(),
            num_layers=node_count,
            num_experts=counts.pop(),
            total_slots=total,
            min_slots_per_layer=4 if args.min_slots_per_layer is None else args.min_slots_per_layer,
            uniform=args.uniform_profile_slots,
        )
    except ValueError as error:
        raise ValueError(f"--expert-profile: {error}") from error
    return {layer: len(experts) for layer, experts in plan.items()}


def check_slot_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for flag in ("target_resident_slots", "total_resident_slots", "min_slots_per_layer"):
        if getattr(args, flag) is not None and getattr(args, flag) < 0:
            parser.error(f"--{flag.replace('_', '-')} must not be negative")
    if args.expert_profile is None:
        for flag, given in (
            ("--total-resident-slots", args.total_resident_slots is not None),
            ("--min-slots-per-layer", args.min_slots_per_layer is not None),
            ("--uniform-profile-slots", args.uniform_profile_slots),
        ):
            if given:
                parser.error(f"{flag} needs --expert-profile")
    elif (args.uniform_profile_slots or args.min_slots_per_layer is not None) and (
        args.total_resident_slots is None and args.target_resident_slots is None
    ):
        parser.error(
            "--uniform-profile-slots and --min-slots-per-layer need --total-resident-slots or --target-resident-slots"
        )
    if args.uniform_profile_slots and args.min_slots_per_layer is not None:
        parser.error("--min-slots-per-layer does not apply with --uniform-profile-slots")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in REMOVED_OPTIONS:
        parser.add_argument(flag, nargs="?", const="", default=None, help=argparse.SUPPRESS)

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
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=512,
        help="model.qwen38_graph.prefill.chunk: at most the kv_capacity of the source package when it has prefill.enabled, "
        "an even number otherwise",
    )
    parser.add_argument(
        "--offload-mode",
        choices=OFFLOAD_MODES,
        default="required",
        help="session.moe.offload.mode. required: offload every QMoE node, and fail the build if the runtime would reject "
        "one. auto: the runtime offloads what it can and keeps the other nodes on the GPU",
    )
    parser.add_argument(
        "--offload-placement",
        choices=OFFLOAD_PLACEMENTS,
        default="adaptive",
        help="session.moe.offload.placement. static: a fixed resident set. adaptive: spare slots promote hot experts",
    )
    parser.add_argument(
        "--device-expert-bytes",
        type=budget_bytes,
        default=None,
        help="session.moe.offload.device_expert_bytes: GPU bytes for resident experts ('auto' or a byte count)",
    )
    parser.add_argument(
        "--host-expert-bytes",
        type=budget_bytes,
        default=None,
        help="session.moe.offload.host_expert_bytes: host bytes for cached experts ('auto' or a byte count)",
    )
    parser.add_argument(
        "--target-resident-slots",
        type=int,
        default=None,
        help="Cap on the resident experts of every layer (a max_resident_slots node override). With --expert-profile it is "
        "the per-layer average of --total-resident-slots",
    )
    parser.add_argument(
        "--expert-profile",
        type=Path,
        default=None,
        help="STRP v1 expert-profile.bin; copied next to the decoder graph and set as session.moe.offload.popularity_file",
    )
    parser.add_argument(
        "--total-resident-slots",
        type=int,
        default=None,
        help="Total resident experts over all layers, spread by --expert-profile (needs --expert-profile)",
    )
    parser.add_argument(
        "--min-slots-per-layer",
        type=int,
        default=None,
        help="Floor of the per-layer share of --total-resident-slots when spreading by --expert-profile (default 4)",
    )
    parser.add_argument(
        "--uniform-profile-slots",
        action="store_true",
        help="Share --total-resident-slots evenly over the layers instead of by popularity",
    )
    parser.add_argument(
        "--dense-gpu-qmoe",
        action="store_true",
        default=True,
        help="Keep stock GPU QMoE in dense/model.onnx using UVA-mapped expert weights (always on)",
    )
    parser.add_argument("--dense-placement", choices=["device", "host_mapped", "cpu"], default="host_mapped")
    parser.add_argument("--mtp-moe-placement", choices=["device", "host_mapped"], default="device")
    parser.add_argument("--embed-placement", choices=["device", "host_mapped"], default="host_mapped")
    parser.add_argument("--vram-limit-gib", type=float, default=None)
    parser.add_argument("--copy", action="store_true")
    return parser


def migrate_source_config(config: dict, prefill_chunk: int) -> list[str]:
    """Moves `model.qwen38_graph` of the source config to the current config surface and returns what it changed.

    The builder rewrites `fold.head_r2.session_options` and `prefill.chunk`, so the source needs both sections. The engine
    accepts a chunk of 1 to kv_capacity with `prefill.enabled`, and an even chunk without it.
    """
    qwen = config["model"].get("qwen38_graph")
    if not isinstance(qwen, dict):
        raise SystemExit("genai_config.json of --from-package has no model.qwen38_graph section")
    changes = migrate_qwen38_graph_config(qwen)
    fold = qwen.get("fold")
    if not isinstance(fold, dict) or not isinstance(fold.get("head_r2"), dict):
        raise SystemExit("genai_config.json of --from-package has no model.qwen38_graph.fold.head_r2 section")
    prefill = qwen.get("prefill")
    if not isinstance(prefill, dict):
        raise SystemExit("genai_config.json of --from-package has no model.qwen38_graph.prefill section")
    if prefill_chunk <= 0:
        raise SystemExit("--prefill-chunk must be positive")
    kv_capacity = qwen.get("kv_capacity")
    if prefill.get("enabled"):
        if isinstance(kv_capacity, int) and prefill_chunk > kv_capacity:
            raise SystemExit(
                f"--prefill-chunk ({prefill_chunk}) must not exceed model.qwen38_graph.kv_capacity ({kv_capacity})"
            )
    elif prefill_chunk % 2:
        raise SystemExit(
            f"--prefill-chunk ({prefill_chunk}) must be even: prefill.enabled is false in --from-package, "
            "so the decode graphs ingest the prompt"
        )
    return changes


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for flag, reason in REMOVED_OPTIONS.items():
        if getattr(args, flag.lstrip("-").replace("-", "_")) is not None:
            parser.error(f"{flag} was removed: {reason}")

    if args.from_package is None or args.out is None or args.plane_lib is None or args.provider_lib is None:
        raise SystemExit("--from-package, --out, --plane-lib, and --provider-lib are required for package build")
    check_slot_arguments(parser, args)

    source = args.from_package.resolve()
    out = reject_output_overlap(
        args.out,
        {
            "--from-package": args.from_package,
            "--plane-lib": args.plane_lib,
            "--provider-lib": args.provider_lib,
            "--ops-dir": args.ops_dir,
            "--expert-profile": args.expert_profile,
        },
    )
    flat_root = (source / "model.onnx").exists()
    target_src = (source / "model.onnx") if flat_root else (source / "target/model.onnx")
    target_out_dir = out if flat_root else (out / "target")
    required_files = {
        "genai_config.json of --from-package": source / "genai_config.json",
        "weights_pool.txt of --from-package": source / "weights_pool.txt",
        "the target graph of --from-package": target_src,
        "--plane-lib": args.plane_lib,
        "--provider-lib": args.provider_lib,
        "--expert-profile": args.expert_profile,
    }
    for label, path in required_files.items():
        if path is not None and not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    config = json.loads((source / "genai_config.json").read_text())
    for change in migrate_source_config(config, args.prefill_chunk):
        print(f"genai_config: {change}")

    # Everything that can reject the request runs before the output folder is created.
    model = onnx.load(str(target_src), load_external_data=False)
    graphs = {target_src: model}
    if (source / "dense/model.onnx").exists():
        graphs[source / "dense/model.onnx"] = onnx.load(str(source / "dense/model.onnx"), load_external_data=False)
    for path, graph_model in graphs.items():
        legacy = sum(node.op_type == "QMoEOffload" for node in graph_model.graph.node)
        if legacy:
            raise SystemExit(
                f"{path} has {legacy} QMoEOffload nodes. The runtime has no such operator, so the graph cannot load: "
                "rebuild the package with standard com.microsoft::QMoE nodes"
            )
    qmoe = prepare_qmoe_for_offload(model)
    names, rejected = qmoe["nodes"], qmoe["rejected"]
    if not names:
        raise SystemExit(f"{target_src} has no com.microsoft::QMoE node to offload")
    if rejected:
        first = next(iter(rejected))
        summary = (
            f"{len(rejected)} of {len(names)} QMoE nodes cannot be offloaded by the runtime; "
            f"first, {node_label(names[first], first)}: {'; '.join(rejected[first])}"
        )
        if args.offload_mode == "required":
            raise SystemExit(f"{summary}. Fix the graph, or use --offload-mode auto to keep those nodes on the GPU.")
        print(f"warning: {summary}. They stay on the GPU.")
    try:
        overrides = {}
        for ordinal, cap in plan_resident_caps(args, model, len(names), frozenset(rejected)).items():
            # A node the runtime does not offload is not registered with it, and an override naming it is an error.
            if ordinal in rejected:
                continue
            if not names[ordinal] or names[ordinal] in overrides:
                raise ValueError(
                    "per-node resident-slot caps need uniquely named QMoE nodes, "
                    f"but {node_label(names[ordinal], ordinal)} is unnamed or shares its name"
                )
            overrides[names[ordinal]] = cap
        node_overrides = format_node_overrides(overrides)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    offload_options = offload_session_options(
        args.offload_mode,
        args.offload_placement,
        args.device_expert_bytes,
        args.host_expert_bytes,
        EXPERT_PROFILE_NAME if args.expert_profile is not None else None,
        node_overrides,
    )

    out.mkdir(parents=True, exist_ok=True)

    linked = link_tree(source, out, args.copy)
    if args.ops_dir is not None and args.ops_dir.exists():
        for so_file in sorted(args.ops_dir.glob("libqwen_*.so")):
            link_file(so_file.resolve(), out / "libs" / so_file.name, copy=True)
    link_file(args.plane_lib.resolve(), out / "libs/libqwen38_plane.so", copy=True)
    link_file(args.provider_lib.resolve(), out / "provider/libonnxruntime_providers_cuda.so", copy=True)

    save_model_fresh(model, target_out_dir / "model_s2.onnx")
    build_multidraft_target_onnx(target_out_dir / "model_s2.onnx", target_out_dir / "model_s3.onnx", verify_rows=3)
    build_multidraft_target_onnx(target_out_dir / "model_s2.onnx", target_out_dir / "model_s4.onnx", verify_rows=4)
    link_file(target_out_dir / "model_s4.onnx", target_out_dir / "model.onnx", copy=True)
    if args.expert_profile is not None:
        # A copy: the output may hard-link the source package, and nothing here may write through to it.
        link_file(args.expert_profile.resolve(), target_out_dir / EXPERT_PROFILE_NAME, copy=True)
    if (source / "dense/model.onnx").exists():
        link_file(source / "dense/model.onnx", out / "dense/model.onnx", args.copy)
    else:
        for obsolete_dir in ("target", "head_r2", "dense", "head_kv") if flat_root else ("dense", "head_kv"):
            if (out / obsolete_dir).exists():
                shutil.rmtree(out / obsolete_dir)
    pool_stats = build_offload_weights_pool(
        source, out, args.dense_placement, args.mtp_moe_placement, args.embed_placement
    )

    if args.stop_token_id is not None:
        config["model"]["eos_token_id"] = [args.stop_token_id]

    cuda_opts = config["model"]["decoder"]["session_options"]["provider_options"][0]["cuda"]
    cuda_opts["enable_cuda_graph"] = "1"
    cuda_opts["arena_extend_strategy"] = "kSameAsRequested"
    cuda_opts["arena.extend_strategy"] = "1"
    if args.vram_limit_gib is not None and args.vram_limit_gib > 0:
        cuda_opts["gpu_device_mem_limit"] = str(int(args.vram_limit_gib * (1024**3)))
    config["model"]["decoder"]["session_options"].update(offload_options)

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
        "stack": "v12b-engine-qmoe-offload-v4",
        "hard_linked_files": linked,
        "target_qmoe_nodes": len(names),
        "router_weights_cleared": qmoe["router_weights_cleared"],
        "qmoe_nodes_not_offloaded": [node_label(names[ordinal], ordinal) for ordinal in sorted(rejected)],
        "offload_session_options": offload_options,
        "pool_stats": pool_stats,
        "eos_token_id": config["model"]["eos_token_id"],
        "vram_limit_gib": args.vram_limit_gib,
        "plane_library": {"source": str(args.plane_lib.resolve()), "sha256": libs.get("libqwen38_plane.so")},
        "libs_sha256": libs,
        "provider_sha256": providers,
    }
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Built offload package: {out} (linked={linked}, target_qmoe={len(names)}, "
        f"router_weights_cleared={qmoe['router_weights_cleared']}, not_offloaded={len(rejected)}, "
        f"offload={offload_options}, pool={pool_stats})"
    )


if __name__ == "__main__":
    main()
