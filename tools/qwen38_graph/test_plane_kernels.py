#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Kernel-level tests of the fast prefill additions of libqwen38_plane.so (ABI 2) against the research (python/torch)
implementations:

  qwen38_plane_ple_rows       batched PLE rows == Qwen4ExpCPUPLELookup (bitwise), incl. EOS masking and a carried-in context
  qwen38_plane_pool_keys      pooled / RMS-normalised / half-RoPE compressed index keys vs handoff_lib.pooled_block_keys (fp32)
  qwen38_plane_transpose_f32  GDN state transpose (bitwise)

usage (research env: mobius, torch; any GPU, < 2 GB):
    test_plane_kernels.py <package dir> --checkpoint <HF checkpoint> --runner-dir <mtp_runner dir (plane.py)>
                          --prefill-dir <dir with handoff_lib.py> [--lib libqwen38_plane.so] [--out json]
"""

import argparse
import ctypes
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

c_vp = ctypes.c_void_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    ap.add_argument("--lib", type=Path)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--runner-dir", type=Path, required=True)
    ap.add_argument("--prefill-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.runner_dir))
    sys.path.insert(0, str(args.prefill_dir))
    lib = ctypes.CDLL(str(args.lib or args.package / "libs/libqwen38_plane.so"))
    ci = ctypes.c_int
    lib.qwen38_plane_ple_rows.argtypes = [c_vp, c_vp, c_vp, ci, c_vp, c_vp, c_vp]
    lib.qwen38_plane_pool_keys.argtypes = [c_vp, c_vp, ci, c_vp, c_vp, c_vp, ci, ctypes.c_float, ci, c_vp]
    lib.qwen38_plane_transpose_f32.argtypes = [c_vp, c_vp, c_vp, ci, ci]
    torch.cuda.init()
    stream = torch.cuda.current_stream().cuda_stream
    result = {}

    # ---- PLE rows ----
    plane = importlib.import_module("plane")  # research data plane wrapper (PleDevice builds the PleParams of a lookup)
    config_class = importlib.import_module("mobius._configs").Qwen4ExpConfig
    lookup_class = importlib.import_module("mobius.generation._qwen4_exp_ple").Qwen4ExpCPUPLELookup
    load_config = importlib.import_module("mobius.integrations.transformers._builder")._load_transformers_config

    checkpoint = str(args.checkpoint)
    hf_config, _ = load_config(checkpoint, revision=None, trust_remote_code=False)
    config = config_class.from_transformers(hf_config)
    layer = next(int(i) - 1 for i in config.ple_layer_ids)
    lookup = lookup_class(checkpoint, config, layer)
    dev = plane.PleDevice(lookup, 33028)
    assert lib.qwen38_plane_sizeof_ple_params() == ctypes.sizeof(plane.PleParams)
    rng = np.random.default_rng(1)
    eos = int(lookup._eos_token_id)
    ok_ple = True
    for n, with_eos, carried in (
        (1, False, False),
        (2, False, True),
        (37, True, True),
        (1000, True, False),
        (4097, True, True),
    ):
        tokens = rng.integers(1000, 200000, n).tolist()
        if with_eos:
            for i in rng.integers(0, n, max(1, n // 50)):
                tokens[int(i)] = eos
        ctx0 = lookup.initial_context()
        if carried:
            ctx0 = np.asarray([[1234, eos if with_eos else 4321]], dtype=np.int64)
        ref, ctx_ref = lookup([tokens], ctx0)
        ref = ref.to("cuda").reshape(n, -1)
        tok_d = torch.tensor(tokens, dtype=torch.int64, device="cuda")
        ctx_in = torch.tensor(np.asarray(ctx0, dtype=np.int64).reshape(2), device="cuda")
        ctx_out = torch.zeros(2, dtype=torch.int64, device="cuda")
        out = torch.empty((n, dev.width), dtype=torch.bfloat16, device="cuda")
        lib.qwen38_plane_ple_rows(
            stream, dev.ptr, tok_d.data_ptr(), n, ctx_in.data_ptr(), ctx_out.data_ptr(), out.data_ptr()
        )
        torch.cuda.synchronize()
        same = bool(torch.equal(out.view(torch.int16), ref.to(torch.bfloat16).view(torch.int16)))
        ctx_same = np.asarray(ctx_ref).reshape(-1).tolist() == ctx_out.cpu().tolist()
        ok_ple &= same and ctx_same
        print(
            f"ple_rows n={n} eos={with_eos} carried={carried}: rows bitwise equal {same}, context equal {ctx_same}",
            flush=True,
        )
    result["ple_rows"] = ok_ple

    # ---- pooled keys ----
    handoff_lib = importlib.import_module("handoff_lib")

    index, consts = {}, {}
    for line in (args.package / "qsa_consts.txt").read_text().splitlines():
        f = line.split()
        if f and f[0] == "tensor":
            index[(f[1], f[2])] = (int(f[3]), int(f[4]))
        elif f and f[0] in ("epsilon", "width"):
            consts[f[0]] = float(f[1])
    blob = np.fromfile(args.package / "qsa_consts.bin", dtype=np.uint8)

    def tensor(set_name, name, shape):
        off, nbytes = index[(set_name, name)]
        return torch.from_numpy(blob[off : off + nbytes].copy()).view(torch.bfloat16).reshape(shape).to("cuda")

    width = int(consts["width"])
    ok_pool = True
    for set_name, layer_id in (("target", 3), ("target", 47), ("head", 0)):
        cos = tensor(set_name, "cos", (-1, width))
        sin = tensor(set_name, "sin", (-1, width))
        knorm = tensor(set_name, f"knorm.{layer_id}", (128,))
        for nb in (1, 7, 513):
            raw = (torch.randn(nb * 4, 128, device="cuda") * (3.0 if nb == 513 else 1.0)).to(torch.bfloat16)
            ref = handoff_lib.pooled_block_keys(raw, knorm, cos, sin, consts["epsilon"])
            out = torch.empty((nb, 128), dtype=torch.float32, device="cuda")
            lib.qwen38_plane_pool_keys(
                stream,
                raw.data_ptr(),
                nb,
                knorm.data_ptr(),
                cos.data_ptr(),
                sin.data_ptr(),
                width,
                ctypes.c_float(consts["epsilon"]),
                0,
                out.data_ptr(),
            )
            torch.cuda.synchronize()
            diff = (out - ref).abs()
            rel = (diff / ref.abs().clamp_min(1e-30)).max().item()
            frac = (out == ref).float().mean().item()
            ok_pool &= diff.max().item() < 1e-5 and rel < 1e-4
            print(
                f"pool_keys {set_name} layer {layer_id} nb={nb}: max abs diff {diff.max().item():.3e}, max rel {rel:.3e}, "
                f"exactly equal {100 * frac:.2f} %",
                flush=True,
            )
    result["pool_keys"] = ok_pool

    # ---- transpose ----
    src = torch.randn((48, 128, 128), dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    lib.qwen38_plane_transpose_f32(stream, src.data_ptr(), dst.data_ptr(), 128, 48)
    torch.cuda.synchronize()
    result["transpose"] = bool(torch.equal(dst, src.transpose(-1, -2)))
    print("transpose bitwise equal", result["transpose"], flush=True)
    print(json.dumps(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
