#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Assemble an Engine-facing package of the v12b stack (model.type "qwen38_flash_graph": folded decode cycle, fast dense
prefill, shared weight pool) from an existing v12b package.

The source package (a folder with genai_config.json, target/, head_r2/, dense/, head_kv/, libs/, provider/, weights_pool.txt,
qsa_consts.*, ple_table.txt, ple/ ...) is replicated with hard links (no extra disk; copies when linking is impossible). Only
three things differ in the result: the data-plane library (libs/libqwen38_plane.so is replaced by the one built from the same
GenAI tree as the Engine, ABI version 2), the end-of-sequence ids (--stop-token-id replaces them with an unreachable id for
fixed-length benchmarks), and package_manifest.json, which records the SHA256 of the plane library, the provider and every
custom-op library the package uses.

usage: build_package_engine_v12b.py --from-package <dir> --out <dir> --plane-lib <libqwen38_plane.so>
                                    [--stop-token-id N] [--prefill-chunk-tokens N] [--copy]
Every path is an argument; nothing is read from the environment. Load the result with og.Config / og.Model / og.Engine and
register the CUDA execution provider of <out>/provider/ (one provider path per process).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
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


def link_tree(src: Path, dst: Path, copy: bool) -> int:
    count = 0
    for root, _dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            relative = rel / name
            if relative in (Path(n) for n in REPLACED) or relative == PLANE_REL:
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
    args = parser.parse_args()

    source, out = args.from_package.resolve(), args.out.resolve()
    if not (source / "genai_config.json").exists():
        raise SystemExit(f"{source} is not a package (no genai_config.json)")
    if not args.plane_lib.exists():
        raise SystemExit(f"missing plane library: {args.plane_lib}")
    if out == source:
        raise SystemExit("--out must differ from --from-package")
    out.mkdir(parents=True, exist_ok=True)

    linked = link_tree(source, out, args.copy)
    link_file(args.plane_lib.resolve(), out / PLANE_REL, True)  # a copy: the file is small and owned by the build tree

    config = json.loads((source / "genai_config.json").read_text())
    qwen = config["model"]["qwen38_graph"]
    if not qwen.get("fold", {}).get("enabled") or not qwen.get("prefill", {}).get("enabled"):
        raise SystemExit("the source package is not a v12 package (fold.enabled and prefill.enabled are required)")
    if args.stop_token_id is not None:
        config["model"]["eos_token_id"] = [args.stop_token_id]
    if args.prefill_chunk_tokens is not None:
        qwen["prefill_chunk_tokens"] = args.prefill_chunk_tokens
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
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"package: {out} ({linked} files linked)")
    print(f"plane: {libs.get(PLANE_REL.name)}")
    for name, digest in providers.items():
        print(f"provider {name}: {digest}")


if __name__ == "__main__":
    sys.exit(main())
