#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Assemble an ONNX Runtime GenAI package for model.type "qwen38_flash_graph" (Qwen3.8-Flash-Next NVFP4,
CUDA-graph decode + MTP(1)) from the research bundles. Files are hard-linked where possible (same file system), so the
74 GB target, the head bundle, the custom-op libraries and the PLE safetensors cost no extra disk.

Every input path is an argument; the defaults are the current research stack (CG = eval_runs/qwen38_cuda_graph_20261010_01).
Swap a kernel library or a bundle by passing another path and rebuilding the package (links only).

usage: build_package.py --out <pkg dir> [--target-bundle DIR] [--head-bundle DIR] [--lib-<name> FILE ...]
                        [--plane-lib FILE] [--checkpoint DIR] [--tokenizer-from DIR] [--no-mtp] [--copy]
Needs (for the PLE table only) the research Python environment: mobius, transformers config loader.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

CG = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01")
CHECKPOINT = Path("/home/tianlei/git/Qwen3.8-Flash-Next-NVFP4")
TOKENIZER_FROM = Path("/home/tianlei/git/eval_runs/qwen38_flash_next_nvfp4_20261004/genai")
DEFAULT_PLANE = Path(__file__).resolve().parents[2] / "build/cuda130/Release/libqwen38_plane.so"

KV_CAPACITY = 33028

# name -> default path; order is the registration order
TARGET_LIBS = {
    "qsa": CG / "fused_attn/r2/build/libqwen_packed_qsa_indexer_v11.so",  # qwen.capture
    "fused_ops": CG / "fused_ops/build/libqwen_fused_ops.so",  # qwen.fused
    "fused_attn": CG / "fused_attn/r2/build/libqwen_fused_attn_r2.so",  # qwen.fused_attn
    "fused_hc": CG / "fused_hc/build/libqwen_fused_hc_r2.so",  # qwen.fused_hc
    "fused_gemm": CG / "fused_gemm/build/libqwen_fused_gemm_r2.so",  # qwen.fused_gemm
    "dsa": CG / "fused_attn/r2/build/libqwen_guarded_dsa_r2.so",  # qwen.static (GuardedDSA)
}
HEAD_EXTRA_LIBS = {"mtp_head": CG / "mtp_head/build/libqwen_mtp_head.so"}  # qwen.mtp


def link_tree(src: Path, dst: Path, copy: bool) -> None:
    """Replicate a directory tree with hard links (symlinks kept, copies when linking is impossible)."""
    validate_bundle_destinations([(src, dst)])
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in list(dirs):
            source = Path(root) / name
            if source.is_symlink():
                (dst / rel / name).symlink_to(os.readlink(source), target_is_directory=True)
                dirs.remove(name)
        for name in files:
            s, d = Path(root) / name, dst / rel / name
            if s.is_symlink():
                d.symlink_to(os.readlink(s))
            else:
                link_file(s, d, copy)


def validate_bundle_destinations(bundles: list[tuple[Path, Path]]) -> None:
    """Reject mixed-package rebuilds before any package files are written."""
    sources = [src.resolve() for src, _ in bundles]
    for src, dst in bundles:
        if not src.is_dir():
            raise SystemExit(f"missing source bundle: {src}")
        if os.path.lexists(dst):
            raise SystemExit(f"bundle destination already exists: {dst}; choose a fresh output directory")
        if any(dst.resolve().is_relative_to(source) for source in sources):
            raise SystemExit(f"bundle destination is inside a source bundle: {dst}")


def link_file(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_ple_table(checkpoint: Path, out_dir: Path, copy: bool) -> dict:
    """ple_table.txt: parameters of Qwen4ExpCPUPLELookup + (file, offset, rows) of every FP8 shard of the single PLE layer."""
    from mobius._configs import Qwen4ExpConfig  # noqa: PLC0415
    from mobius.generation._qwen4_exp_ple import Qwen4ExpCPUPLELookup  # noqa: PLC0415
    from mobius.integrations.transformers._builder import _load_transformers_config  # noqa: PLC0415

    hf_config, _ = _load_transformers_config(str(checkpoint), revision=None, trust_remote_code=False)
    config = Qwen4ExpConfig.from_transformers(hf_config)
    layer_ids = [int(i) - 1 for i in (config.ple_layer_ids or [])]
    if len(layer_ids) != 1:
        raise SystemExit(f"expected exactly one PLE layer, got {layer_ids}")
    lookup = Qwen4ExpCPUPLELookup(str(checkpoint), config, layer_ids[0])
    ple_dir = out_dir / "ple"
    ple_dir.mkdir(exist_ok=True)
    lines = [
        "qwen38_ple_table 1",
        f"n_heads {lookup._num_heads}",
        f"head_dim {lookup._head_dim}",
        f"heads_per_ngram {lookup._heads_per_ngram}",
        f"eos {int(lookup._eos_token_id)}",
        f"scale {float(lookup._scale.float()):.9g}",
    ]
    mult = [int(m) for m in lookup._multipliers]
    mult += [0] * (4 - len(mult))
    lines.append("mult " + " ".join(str(m) for m in mult[:4]))
    lines.append("vocab " + " ".join(str(int(v)) for v in lookup._vocab_sizes))
    lines.append("offset " + " ".join(str(int(v)) for v in lookup._offsets))
    linked = {}
    for shard in lookup._shards:
        source = Path(shard.rows.filename)
        if source.name not in linked:
            link_file(source, ple_dir / source.name, copy)
            linked[source.name] = True
        lines.append(f"shard ple/{source.name} {int(shard.rows.offset)} {int(shard.end_row - shard.first_row)}")
    (out_dir / "ple_table.txt").write_text("\n".join(lines) + "\n")
    return {"layer": layer_ids[0], "shards": len(lookup._shards), "files": sorted(linked)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, default=CG / "models_capture_v11_s2_bundle/target")
    parser.add_argument("--head-bundle", type=Path, default=CG / "mtp_head/bundle/final")
    parser.add_argument("--plane-lib", type=Path, default=DEFAULT_PLANE)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tokenizer-from", type=Path, default=TOKENIZER_FROM)
    parser.add_argument(
        "--no-mtp",
        action="store_true",
        help="default to plain S=1 graph decode (mtp.enabled=false); the head stays packaged",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--stop-token-ids",
        default="248046,248044",
        help="comma separated stop tokens written to genai_config.json (<|im_end|>, <|endoftext|>); use an unreachable id "
        "(e.g. 248319) for a fixed-length/ignore-EOS benchmark package",
    )
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    parser.add_argument("--skip-ple", action="store_true", help="do not (re)write ple_table.txt")
    for name, default in {**TARGET_LIBS, **HEAD_EXTRA_LIBS}.items():
        parser.add_argument(f"--lib-{name.replace('_', '-')}", type=Path, default=default)
    args = parser.parse_args()
    out: Path = args.out
    validate_bundle_destinations([(args.target_bundle, out / "target"), (args.head_bundle, out / "mtp_head")])
    out.mkdir(parents=True, exist_ok=True)

    libs = {name: getattr(args, f"lib_{name}") for name in {**TARGET_LIBS, **HEAD_EXTRA_LIBS}}
    for name, path in libs.items():
        if not path.exists():
            raise SystemExit(f"missing library {name}: {path}")
    if not args.plane_lib.exists():
        raise SystemExit(f"missing plane library: {args.plane_lib} (build the qwen38_plane target)")

    link_tree(args.target_bundle, out / "target", args.copy)
    link_tree(args.head_bundle, out / "mtp_head", args.copy)
    (out / "libs").mkdir(exist_ok=True)
    rel_libs = {}
    for name, path in libs.items():
        link_file(path, out / "libs" / path.name, args.copy)
        rel_libs[name] = f"libs/{path.name}"
    link_file(args.plane_lib, out / "libs" / "libqwen38_plane.so", args.copy)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = args.tokenizer_from / name
        if not source.exists():
            source = args.checkpoint / name
        link_file(source, out / name, args.copy)
    link_file(args.checkpoint / "generation_config.json", out / "generation_config.json", args.copy)

    ple = None
    if not args.skip_ple:
        ple = write_ple_table(args.checkpoint, out, args.copy)

    target_libs = [rel_libs[n] for n in TARGET_LIBS]
    head_libs = [*target_libs, rel_libs["mtp_head"]]
    provider = {
        "cuda": {
            "enable_cuda_graph": "1",
            "min_num_runs_before_cuda_graph_capture": "4",
            "arena_extend_strategy": "1",
        }
    }
    config = {
        "model": {
            "type": "qwen38_flash_graph",
            "bos_token_id": 248044,
            "pad_token_id": 248044,
            "eos_token_id": [int(x) for x in args.stop_token_ids.split(",")],
            "vocab_size": 248320,
            "context_length": KV_CAPACITY - 4,
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
                    "custom_ops_libraries": target_libs,
                },
            },
            "qwen38_graph": {
                "plane_library": "libs/libqwen38_plane.so",
                "ple_table": "ple_table.txt",
                "kv_capacity": KV_CAPACITY,
                "warmup": not args.no_warmup,
                "mtp": {"enabled": not args.no_mtp, "prefill_head": True},
            },
        },
        "search": {
            "do_sample": False,
            "top_k": 1,
            "temperature": 1.0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_length": KV_CAPACITY - 4,
            "min_length": 0,
            "past_present_share_buffer": False,
        },
    }
    config["model"]["qwen38_graph"]["head"] = {
        "filename_s1": "mtp_head/model_s1.onnx",
        "filename_s2": "mtp_head/model_s2.onnx",
        "session_options": {"custom_ops_libraries": head_libs},
    }
    (out / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "out": str(out),
        "libs": rel_libs,
        "ple": ple,
        "target_bundle": str(args.target_bundle),
        "head_bundle": str(args.head_bundle),
    }
    (out / "package_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
