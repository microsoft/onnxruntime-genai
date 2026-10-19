#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Assemble a model.type "qwen38_flash_graph" package for the v12 stack: fold decode (head inside the verify graph, graph ids
71-73), fast prefill (dense prefill session + hand-off + batched head KV prefill), one device copy of the shared weights.

Everything is hard-linked from the research bundles (same file system); every input path is an argument, so another kernel
library or bundle is a rebuild of the package (links only). Layout:

    genai_config.json            target/ (fold model)   head_r2/ (standalone R=2 head)   dense/ (dense prefill model)   head_kv/
    libs/*.so                    ple/ + ple_table.txt   weights_pool.txt (shared initializers)   qsa_consts.{txt,bin}
    tokenizer files

usage: build_package_v12.py --out <pkg dir> [--fold-bundle DIR] [--head-r2-bundle DIR] [--dense-bundle DIR] [--head-kv-bundle DIR]
                            [--lib-<name> FILE ...] [--plane-lib FILE] [--no-prefill] [--stop-token-ids ID,ID] [--copy]
Needs the research Python env (onnx; mobius for the PLE table).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
from collections import defaultdict
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_package", HERE / "build_package.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

CG = bp.CG
FOLD_BUNDLE = CG / "mtp_runner/fold/bundle/fold_v12all/target"
HEAD_R2_BUNDLE = CG / "mtp_runner/fold/bundle/head_r2f"
DENSE_BUNDLE = Path(
    "/home/tianlei/git/eval_runs/qwen38_mobius_nvfp4_mtp_20261005/models_v6_native_sparse_20261007_01/target"
)
HEAD_KV_BUNDLE = CG / "mtp_runner/prefill/bundle/head_kv"

# name -> default path; the order is the registration order (op domains do not overlap)
LIBS = {
    "qsa": CG / "fused_attn/r2/build/libqwen_packed_qsa_indexer_v11.so",  # qwen.capture (unchanged in v12)
    "fused_ops": CG / "fused_ops/build/libqwen_fused_ops.so",  # qwen.fused
    "fused_attn": CG / "v12/lib/libqwen_fused_attn_r3.so",  # qwen.fused_attn (r3)
    "fused_hc": CG / "fused_hc/build/libqwen_fused_hc_r3.so",  # qwen.fused_hc (r3)
    "fused_gemm": CG / "fused_gemm/build/libqwen_fused_gemm_r4.so",  # qwen.fused_gemm (r4: trigger + M <= 8)
    "dsa": CG / "v12/dsa/build/libqwen_guarded_dsa_r2_v12.so",  # qwen.static (GuardedDSA r2, memset -> kernel)
    "mtp_head": CG / "mtp_head/build/libqwen_mtp_head.so",  # qwen.mtp
    "fold": CG / "mtp_runner/fold/build/libqwen_fold.so",  # qwen.fold
}


# ---- initializer inventory / shared pool ----------------------------------------------------------------------------
def inventory(model_path: Path) -> dict:
    model = onnx.load(str(model_path), load_external_data=False)
    out = {}
    for t in model.graph.initializer:
        ext = {e.key: e.value for e in t.external_data}
        if not ext:
            continue  # inline initializers are loaded by ORT itself
        path = model_path.parent / ext["location"]
        st = os.stat(path)
        offset = int(ext.get("offset", 0))
        length = int(ext.get("length", 0)) or st.st_size - offset
        out[t.name] = {
            "key": (st.st_dev, st.st_ino, offset, length, t.data_type, tuple(t.dims)),
            "file": str(Path(ext["location"])),
            "dtype": t.data_type,
            "dims": list(t.dims),
            "offset": offset,
            "length": length,
        }
        assert not any(c.isspace() for c in t.name), t.name
    return out


def write_pool(out: Path, models: dict[str, Path], source_models: dict[str, Path] | None = None) -> dict:
    """weights_pool.txt: initializers used (same inode / offset / length / dtype / dims) by >= 2 of the sessions."""
    inv = {label: inventory(path) for label, path in models.items()}
    if source_models is not None:
        for label, items in inv.items():
            source = inventory(source_models[label])
            for name, item in items.items():
                original = source[name]
                if any(item[field] != original[field] for field in ("dtype", "dims", "offset", "length")):
                    raise ValueError(f"copied initializer metadata differs from source: {label}:{name}")
                item["key"] = original["key"]
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
        "sessions": {label: sum(1 for session in sessions if session.split()[1] == label) for label in inv},
    }


# ---- QSA hand-off constants -------------------------------------------------------------------------------------------
def _tensor_bytes(tensor, base: Path) -> bytes:
    assert tensor.data_type == 16, tensor.data_type  # bf16
    if tensor.external_data:
        info = {e.key: e.value for e in tensor.external_data}
        with open(base / info["location"], "rb") as f:
            f.seek(int(info.get("offset", 0)))
            return f.read(int(info["length"]) if "length" in info else -1)
    if tensor.HasField("raw_data"):
        return bytes(tensor.raw_data)
    return struct.pack(f"<{len(tensor.int32_data)}H", *tensor.int32_data)


def qsa_nodes(model_path: Path, head: bool):
    model = onnx.load(str(model_path), load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "PackedQSAIndexer":
            continue
        is_head = node.input[3].startswith("head.") or "mtp" in node.input[3]
        if is_head != head:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        layer = int(node.input[3].split("layers.")[1].split(".")[0])
        yield (
            layer,
            attrs,
            [_tensor_bytes(inits[node.input[i]], model_path.parent) for i in (3, 4, 5)],
            list(inits[node.input[4]].dims),
        )


def write_qsa_constants(out: Path, target_model: Path, head_model: Path) -> dict:
    blob, index = bytearray(), []
    eps = width = None
    sets = {}
    for set_name, path, head in (("target", target_model, False), ("head", head_model, True)):
        cos = sin = None
        layers = {}
        for layer, attrs, (knorm, c, s), cos_dims in qsa_nodes(path, head):
            if eps is None:
                eps = float(attrs["epsilon"])
            assert abs(eps - float(attrs["epsilon"])) < 1e-12
            layers[layer] = knorm
            if cos is None:
                cos, sin = c, s
                if set_name == "target":
                    width = int(cos_dims[-1])
                    assert len(c) == 2 * int(cos_dims[0]) * width and int(cos_dims[0]) >= 33028, cos_dims
            else:
                assert cos == c and sin == s, "QSA rotary tables differ between layers"
        assert layers, f"no PackedQSAIndexer nodes in {path}"
        sets[set_name] = (layers, cos, sin)

    def add(set_name, name, data):
        while len(blob) % 16:
            blob.append(0)
        index.append(f"tensor {set_name} {name} {len(blob)} {len(data)}")
        blob.extend(data)

    for set_name, (layers, cos, sin) in sets.items():
        add(set_name, "cos", cos)
        add(set_name, "sin", sin)
        for layer, knorm in sorted(layers.items()):
            add(set_name, f"knorm.{layer}", knorm)
    (out / "qsa_consts.bin").write_bytes(bytes(blob))
    (out / "qsa_consts.txt").write_text(
        "\n".join(["qwen38_qsa_consts 1", "file qsa_consts.bin", f"epsilon {eps:.9g}", f"width {width}", *index]) + "\n"
    )
    same = sets["target"][1] == sets["head"][1] and sets["target"][2] == sets["head"][2]
    return {
        "layers": sorted(sets["target"][0]),
        "width": width,
        "epsilon": eps,
        "head_tables_equal_target": same,
        "bytes": len(blob),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold-bundle", type=Path, default=FOLD_BUNDLE)
    parser.add_argument("--head-r2-bundle", type=Path, default=HEAD_R2_BUNDLE)
    parser.add_argument("--dense-bundle", type=Path, default=DENSE_BUNDLE)
    parser.add_argument("--head-kv-bundle", type=Path, default=HEAD_KV_BUNDLE)
    parser.add_argument("--plane-lib", type=Path, default=bp.DEFAULT_PLANE)
    parser.add_argument("--checkpoint", type=Path, default=bp.CHECKPOINT)
    parser.add_argument("--tokenizer-from", type=Path, default=bp.TOKENIZER_FROM)
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="fold decode only; the prompt runs through the muted fold graph (S=2 steps)",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help="do not share initializers between the sessions (does not fit one GPU with prefill)",
    )
    parser.add_argument("--stop-token-ids", default="248046,248044")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--skip-ple", action="store_true")
    for name, default in LIBS.items():
        parser.add_argument(f"--lib-{name.replace('_', '-')}", type=Path, default=default)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    bundles = {"target": args.fold_bundle, "head_r2": args.head_r2_bundle}
    if not args.no_prefill:
        bundles.update(dense=args.dense_bundle, head_kv=args.head_kv_bundle)
    bp.validate_bundle_destinations([(source, out / label) for label, source in bundles.items()])
    out.mkdir(parents=True, exist_ok=True)

    libs = {name: getattr(args, f"lib_{name}") for name in LIBS}
    for name, path in libs.items():
        if not path.exists():
            raise SystemExit(f"missing library {name}: {path}")
    if not args.plane_lib.exists():
        raise SystemExit(f"missing plane library: {args.plane_lib} (build the qwen38_plane target)")

    bp.link_tree(args.fold_bundle, out / "target", args.copy)
    bp.link_tree(args.head_r2_bundle, out / "head_r2", args.copy)
    models = {"target": out / "target/model.onnx", "head_r2": out / "head_r2/model.onnx"}
    if not args.no_prefill:
        bp.link_tree(args.dense_bundle, out / "dense", args.copy)
        bp.link_tree(args.head_kv_bundle, out / "head_kv", args.copy)
        models["dense"] = out / "dense/model.onnx"
        models["head_kv"] = out / "head_kv/model.onnx"
    (out / "libs").mkdir(exist_ok=True)
    rel_libs = {}
    for name, path in libs.items():
        bp.link_file(path, out / "libs" / path.name, args.copy)
        rel_libs[name] = f"libs/{path.name}"
    bp.link_file(args.plane_lib, out / "libs/libqwen38_plane.so", args.copy)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = args.tokenizer_from / name
        if not source.exists():
            source = args.checkpoint / name
        bp.link_file(source, out / name, args.copy)
    bp.link_file(args.checkpoint / "generation_config.json", out / "generation_config.json", args.copy)
    ple = None if args.skip_ple else bp.write_ple_table(args.checkpoint, out, args.copy)

    pool = None
    if not args.no_pool:
        pool = write_pool(out, models, {label: source / "model.onnx" for label, source in bundles.items()})
    consts = None
    if not args.no_prefill:
        consts = write_qsa_constants(
            out,
            out / "target/model.onnx",
            out / "head_r2/model.onnx",
        )

    custom_libs = [rel_libs[n] for n in LIBS]
    provider = {
        "cuda": {
            "enable_cuda_graph": "1",
            "min_num_runs_before_cuda_graph_capture": "4",
            "arena_extend_strategy": "1",
        }
    }
    plain_session = {
        "log_id": "onnxruntime-genai",
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "provider_options": [{"cuda": {}}],
        "session.use_device_allocator_for_initializers": "1",
    }
    qg = {
        "plane_library": "libs/libqwen38_plane.so",
        "ple_table": "ple_table.txt",
        "kv_capacity": bp.KV_CAPACITY,
        "warmup": not args.no_warmup,
        "mtp": {"enabled": True, "prefill_head": True},
        "fold": {"enabled": True, "target_graph_id": 71, "head_r2": {"filename": "head_r2/model.onnx", "graph_id": 81}},
    }
    qg["prefill"] = {"weights_pool": "weights_pool.txt"} if pool else {}
    if not args.no_prefill:
        qg["prefill"].update(
            {
                "enabled": True,
                "chunk": args.chunk,
                "constants": "qsa_consts.txt",
                "dense": {"filename": "dense/model.onnx", "session_options": plain_session},
                "head_kv": {"filename": "head_kv/model.onnx", "session_options": plain_session},
                "warmup_lengths": [args.chunk, 2048, 512, 128, 37],
            }
        )
    config = {
        "model": {
            "type": "qwen38_flash_graph",
            "bos_token_id": 248044,
            "pad_token_id": 248044,
            "eos_token_id": [int(x) for x in args.stop_token_ids.split(",")],
            "vocab_size": 248320,
            "context_length": bp.KV_CAPACITY - 4,
            "decoder": {
                "filename": "target/model.onnx",
                "head_size": 256,
                "hidden_size": 2560,
                "num_attention_heads": 24,
                "num_hidden_layers": 48,
                "num_key_value_heads": 2,
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "provider_options": [provider],
                    "session.use_device_allocator_for_initializers": "1",
                    "custom_ops_libraries": custom_libs,
                },
            },
            "qwen38_graph": qg,
        },
        "search": {
            "do_sample": False,
            "top_k": 1,
            "temperature": 1.0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_length": bp.KV_CAPACITY - 4,
            "min_length": 0,
            "past_present_share_buffer": False,
        },
    }
    (out / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "out": str(out),
        "libs": {k: str(v) for k, v in libs.items()},
        "plane": str(args.plane_lib),
        "ple": ple,
        "pool": pool,
        "qsa_constants": consts,
        "fold_bundle": str(args.fold_bundle),
        "head_r2_bundle": str(args.head_r2_bundle),
        "dense_bundle": None if args.no_prefill else str(args.dense_bundle),
        "head_kv_bundle": None if args.no_prefill else str(args.head_kv_bundle),
    }
    (out / "package_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
