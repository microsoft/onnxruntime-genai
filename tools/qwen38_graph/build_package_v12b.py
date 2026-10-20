#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Assemble the v12b package: build_package_v12.py with the v12b kernel libraries (accurate bf16 Sigmoid) and the fixed
CUDA EP provider linked into <pkg>/provider/.

v12b = v12 bundles unchanged + libqwen_fused_{attn,hc,gemm}_r5.so + provider a5ce286d... (qwen38-prefill-fix 453ac97a68); the head, fold,
DSA and indexer libraries and the (unused, test-chain only) qwen.fused library are the v12 ones. GenAI itself is unchanged.
Every path is still an argument (--lib-*, --fold-bundle, ...); --provider only decides what is linked as <pkg>/provider/.

usage: build_package_v12b.py --out <pkg dir> [--v12b-dir DIR] [--provider FILE] [build_package_v12.py arguments]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_package_v12", HERE / "build_package_v12.py")
v12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v12)
bp = v12.bp

V12B = Path("/home/tianlei/git/eval_runs/qwen38_v12b_20261012_01")


def _pop(argv: list[str], flag: str, default):
    for i, arg in enumerate(argv):
        if arg == flag:
            value = argv[i + 1]
            del argv[i : i + 2]
            return type(default)(value)
        if arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
            del argv[i]
            return type(default)(value)
    return default


def main() -> None:
    argv = sys.argv[1:]
    v12b_dir = _pop(argv, "--v12b-dir", V12B)
    provider = _pop(argv, "--provider", v12b_dir / "runtime_fixed/ort_home/lib/libonnxruntime_providers_cuda.so")
    for name, rel in (
        ("fused_attn", "lib/libqwen_fused_attn_r5.so"),
        ("fused_hc", "lib/libqwen_fused_hc_r5.so"),
        ("fused_gemm", "lib/libqwen_fused_gemm_r5.so"),
    ):
        v12.LIBS[name] = v12b_dir / rel
    out = Path(_pop(list(argv), "--out", "")).resolve()
    if not any(flag in argv for flag in ("-h", "--help")) and not provider.is_file():
        raise SystemExit(f"missing provider: {provider}")
    sys.argv = [sys.argv[0], *argv]
    v12.main()
    bp.link_file(provider, out / "provider/libonnxruntime_providers_cuda.so", "--copy" in argv)
    manifest = out / "package_manifest.json"
    data = json.loads(manifest.read_text())
    data["stack"] = "v12b"
    data["provider"] = str(provider)
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    print(f"provider linked: {out / 'provider/libonnxruntime_providers_cuda.so'}")


if __name__ == "__main__":
    main()
