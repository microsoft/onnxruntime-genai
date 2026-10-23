#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Kernel-level tests of the integrated CUDA data plane against the research (python/torch) implementations:

  mtp_ple_rows      batched PLE rows == Qwen4ExpCPUPLELookup (bitwise), incl. EOS masking and a carried-in context
  mtp_pool_keys     pooled / RMS-normalised / half-RoPE compressed index keys vs handoff_lib.pooled_block_keys (fp32)
  mtp_transpose_f32 GDN state transpose (bitwise)

usage (mobius, torch; any GPU, < 2 GB): test_plane_kernels.py <package dir>
  --plane-module <plane.py> --handoff-module <handoff_lib.py> --checkpoint <directory>
  [--lib libonnxruntime-genai-cuda.so] [--out json]
"""

import argparse
import ctypes
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

c_vp = ctypes.c_void_p


class PlaneFunctions(ctypes.Structure):
    _fields_ = [(name, c_vp) for name in (
        "sizeof_params", "set_device", "error_string", "malloc_", "free_", "host_alloc", "host_free",
        "host_register", "host_unregister", "host_get_device_pointer", "memset_", "memcpy_h2d", "memcpy_d2h",
        "memcpy_d2d", "stream_sync", "prep", "argmax_part", "post_tok", "post1", "head_prep", "head_post",
        "or_status", "memcpy2d_d2d", "stream_create", "stream_destroy", "ple_rows", "pool_keys", "transpose_f32",
        "embed_rows", "dflash_select", "argmax_finish",
    )]


def load_plane_api(path=None):
    if path is None:
        import onnxruntime_genai as og
        path = Path(og.__file__).with_name("libonnxruntime-genai-cuda.so")
    library = ctypes.CDLL(str(path))
    getter = library.GetQwen38PlaneFunctions
    getter.argtypes = [ctypes.c_uint32, ctypes.c_size_t]
    getter.restype = ctypes.POINTER(PlaneFunctions)
    pointer = getter(1, ctypes.sizeof(PlaneFunctions))
    if not pointer:
        raise RuntimeError("GenAI CUDA data-plane version/size mismatch")
    table = pointer.contents
    if any(not getattr(table, name) for name, _ in table._fields_):
        raise RuntimeError("GenAI CUDA data-plane interface is incomplete")
    ci = ctypes.c_int
    signatures = {
        "sizeof_params": (ci, []),
        "ple_rows": (None, [c_vp, c_vp, c_vp, ci, c_vp, c_vp, c_vp]),
        "pool_keys": (None, [c_vp, c_vp, ci, c_vp, c_vp, c_vp, ci, ctypes.c_float, ci, c_vp]),
        "transpose_f32": (None, [c_vp, c_vp, c_vp, ci, ci]),
        "argmax_part": (None, [c_vp, c_vp, ci, c_vp, c_vp, ci]),
        "dflash_select": (ci, [c_vp, c_vp, c_vp, c_vp, ci, ci]),
        "argmax_finish": (ci, [c_vp, c_vp, c_vp, ci, c_vp]),
        "malloc_": (ci, [ctypes.POINTER(c_vp), ctypes.c_size_t]),
        "free_": (ci, [c_vp]),
        "memcpy_h2d": (ci, [c_vp, c_vp, ctypes.c_size_t, c_vp]),
        "memcpy_d2h": (ci, [c_vp, c_vp, ctypes.c_size_t, c_vp]),
        "stream_sync": (ci, [c_vp]),
    }
    api = SimpleNamespace(_library=library, _getter=getter)
    for name, (result, args) in signatures.items():
        prefix = "mtp_" if name in ("sizeof_params", "ple_rows", "pool_keys", "transpose_f32", "argmax_part") else "qg_"
        setattr(api, prefix + name, ctypes.CFUNCTYPE(result, *args)(getattr(table, name)))
    return api


def reference_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.suffix != ".py":
        raise argparse.ArgumentTypeError(f"expected an existing Python reference module: {path}")
    return path


def checkpoint_dir(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise argparse.ArgumentTypeError(f"expected a checkpoint directory containing config.json: {path}")
    return path


def load_reference(name, path, required):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for symbol in required:
        if not callable(getattr(module, symbol, None)):
            raise ValueError(f"{path} must provide {symbol}")
    return module


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    ap.add_argument("--lib", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plane-module", type=reference_file, required=True, help="trusted plane.py reference file")
    ap.add_argument(
        "--handoff-module", type=reference_file, required=True, help="trusted handoff_lib.py reference file"
    )
    ap.add_argument("--checkpoint", type=checkpoint_dir, required=True, help="local Hugging Face checkpoint directory")
    args = ap.parse_args()
    plane = load_reference("plane", args.plane_module, ("PleDevice", "PleParams"))
    handoff_lib = load_reference("handoff_lib", args.handoff_module, ("pooled_block_keys",))
    lib = load_plane_api(args.lib)
    torch.cuda.init()
    stream = torch.cuda.current_stream().cuda_stream
    result = {}

    # ---- PLE rows ----
    from mobius._configs import Qwen4ExpConfig  # noqa: PLC0415
    from mobius.generation._qwen4_exp_ple import Qwen4ExpCPUPLELookup  # noqa: PLC0415
    from mobius.integrations.transformers._builder import _load_transformers_config  # noqa: PLC0415

    checkpoint = str(args.checkpoint)
    hf_config, _ = _load_transformers_config(checkpoint, revision=None, trust_remote_code=False)
    config = Qwen4ExpConfig.from_transformers(hf_config)
    layer = next(int(i) - 1 for i in config.ple_layer_ids)
    lookup = Qwen4ExpCPUPLELookup(checkpoint, config, layer)
    dev = plane.PleDevice(lookup, 33028)
    assert lib.mtp_sizeof_params() == ctypes.sizeof(plane.PleParams)
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
        lib.mtp_ple_rows(stream, dev.ptr, tok_d.data_ptr(), n, ctx_in.data_ptr(), ctx_out.data_ptr(), out.data_ptr())
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
            lib.mtp_pool_keys(
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
    lib.mtp_transpose_f32(stream, src.data_ptr(), dst.data_ptr(), 128, 48)
    torch.cuda.synchronize()
    result["transpose"] = bool(torch.equal(dst, src.transpose(-1, -2)))
    print("transpose bitwise equal", result["transpose"], flush=True)
    print(json.dumps(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
