#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Assemble an ONNX Runtime GenAI package for model.type "qwen38_flash_graph" (Qwen3.8-Flash-Next NVFP4,
CUDA-graph decode + MTP(1), S=1 / S=2 chain-verify path) from its pre-built parts. Files are hard-linked where possible
(same file system), so the target bundle, the head bundle, the custom-op libraries and the PLE safetensors cost no extra disk.

Required external artifacts (none of them is part of this repository; see README.md):
  * --target-bundle: folder with the exported target graph `model.onnx` (+ external data) of the static-capture export;
  * --head-bundle: folder with the MTP head graphs `model_s1.onnx` / `model_s2.onnx`;
  * --lib-<name>: the custom-op libraries the two graphs reference (domains qwen.capture / qwen.fused* / qwen.static /
    qwen.mtp), built from https://github.com/tianleiwu/qwen38-flash-ops;
  * --plane-lib: libqwen38_plane.so, built from this tree with -DENABLE_QWEN38_PLANE=ON ($QWEN38_PLANE_LIB);
  * --checkpoint: the Hugging Face checkpoint folder (config, generation_config.json, tokenizer files and the FP8 PLE
    safetensors); --tokenizer-from: optional folder that takes precedence for tokenizer.json / tokenizer_config.json /
    chat_template.jinja.
The PLE table (`ple_table.txt`) is written with the `mobius` package (pip install of the model builder's runtime package) and
the transformers config loader; pass --skip-ple when the package already has one.

For the v12b stack (folded decode cycle + fast prefill) use build_package_engine_v12b.py on an existing v12b package.

usage: build_package.py --out <pkg dir> --target-bundle DIR --head-bundle DIR --plane-lib FILE --checkpoint DIR
                        --lib-qsa FILE --lib-fused-ops FILE --lib-fused-attn FILE --lib-fused-hc FILE --lib-fused-gemm FILE
                        --lib-dsa FILE --lib-mtp-head FILE [--tokenizer-from DIR] [--no-mtp] [--copy]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_PLANE = Path(os.environ["QWEN38_PLANE_LIB"]) if "QWEN38_PLANE_LIB" in os.environ else None

KV_CAPACITY = 33028

# Custom-op libraries of the target graph, in registration order (the domain each one provides is noted).
TARGET_LIBS = {
    "qsa": "qwen.capture",
    "fused_ops": "qwen.fused",
    "fused_attn": "qwen.fused_attn",
    "fused_hc": "qwen.fused_hc",
    "fused_gemm": "qwen.fused_gemm",
    "dsa": "qwen.static",
}
HEAD_EXTRA_LIBS = {"mtp_head": "qwen.mtp"}


def link_tree(src: Path, dst: Path, copy: bool) -> None:
    """Replicate a directory tree with hard links (symlinks kept, copies when linking is impossible)."""
    for root, _dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            s, d = Path(root) / name, dst / rel / name
            if s.is_symlink():
                if d.exists() or d.is_symlink():
                    d.unlink()
                d.symlink_to(os.readlink(s))
            else:
                link_file(s, d, copy)


def link_file(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # hard links can fail across filesystems; fall back to copying
    shutil.copy2(src, dst)


def write_ple_table(checkpoint: Path, out_dir: Path, copy: bool) -> dict:
    """ple_table.txt: parameters of Qwen4ExpCPUPLELookup + (file, offset, rows) of every FP8 shard of the single PLE layer."""
    from mobius._configs import Qwen4ExpConfig  # noqa: PLC0415 - only needed for the PLE table
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
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument("--head-bundle", type=Path, required=True)
    parser.add_argument("--plane-lib", type=Path, default=DEFAULT_PLANE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-from", type=Path, default=None)
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
    for name, domain in {**TARGET_LIBS, **HEAD_EXTRA_LIBS}.items():
        parser.add_argument(f"--lib-{name.replace('_', '-')}", type=Path, required=True, help=f"provides {domain}")
    args = parser.parse_args()
    out: Path = args.out.resolve()
    target_bundle: Path = args.target_bundle.resolve()
    head_bundle: Path = args.head_bundle.resolve()
    checkpoint: Path = args.checkpoint.resolve()

    for label, src_dir in (
        ("--target-bundle", target_bundle),
        ("--head-bundle", head_bundle),
        ("--checkpoint", checkpoint),
    ):
        if out == src_dir or out.is_relative_to(src_dir) or src_dir.is_relative_to(out):
            raise SystemExit(f"--out ({out}) must not be equal to or nested inside {label} ({src_dir})")
    if args.tokenizer_from is not None:
        tok_dir = args.tokenizer_from.resolve()
        if out == tok_dir or out.is_relative_to(tok_dir) or tok_dir.is_relative_to(out):
            raise SystemExit(f"--out ({out}) must not be equal to or nested inside --tokenizer-from ({tok_dir})")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out directory already exists and is not empty: {out}")

    libs = {name: getattr(args, f"lib_{name}") for name in {**TARGET_LIBS, **HEAD_EXTRA_LIBS}}
    seen_basenames: dict[str, str] = {"libqwen38_plane.so": "--plane-lib"}
    for name, path in libs.items():
        if not path.exists():
            raise SystemExit(f"missing library {name}: {path}")
        flag = f"--lib-{name.replace('_', '-')}"
        if path.name in seen_basenames:
            raise SystemExit(
                f"duplicate custom-op library basename '{path.name}' between {seen_basenames[path.name]} and {flag}"
            )
        seen_basenames[path.name] = flag
    if args.plane_lib is None or not args.plane_lib.exists():
        raise SystemExit(
            f"missing plane library: {args.plane_lib} (configure with -DENABLE_QWEN38_PLANE=ON, build the "
            "qwen38_plane target and pass --plane-lib or set QWEN38_PLANE_LIB)"
        )

    out.mkdir(parents=True, exist_ok=True)
    link_tree(target_bundle, out / "target", args.copy)
    link_tree(head_bundle, out / "mtp_head", args.copy)
    (out / "libs").mkdir(exist_ok=True)
    rel_libs = {}
    for name, path in libs.items():
        link_file(path, out / "libs" / path.name, args.copy)
        rel_libs[name] = f"libs/{path.name}"
    link_file(args.plane_lib, out / "libs" / "libqwen38_plane.so", args.copy)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = args.tokenizer_from / name if args.tokenizer_from else args.checkpoint / name
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
            "arena_extend_strategy": "kSameAsRequested",
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
