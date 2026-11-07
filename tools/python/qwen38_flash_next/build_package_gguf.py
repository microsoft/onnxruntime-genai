#!/usr/bin/env python3
"""
Export Unsloth Qwen3.8-Flash-Next GGUF models (UD-IQ1_S, UD-IQ2_XS, UD-IQ3_XXS)
and the corresponding IQ4_NL engram GGUF to ONNX Runtime GenAI
model-data packages (`model.type = "qwen38_flash_graph"`):

  - Routed MoE experts (48 layers x 3 tensors: gate, up, down):
      `com.microsoft::MoEPacked` pointing directly to the `.gguf` shard byte offsets
  - Dense projections & LM Head (98 nodes in model.onnx + lm_head in dflash2.onnx):
      `com.microsoft::MatMulPacked` pointing directly to the `.gguf` shard byte offsets
  - Token embedding (`token_embd.weight` in model.onnx, dflash2.onnx, embedding.onnx):
      `com.microsoft::GatherPacked` pointing directly to the `.gguf` shard byte offset
  - N-gram PLE table (`per_layer_token_embd.weight` in engram GGUF):
      `ple_table.txt` (`scale -20.0` = IQ4_NL) pointing directly to the engram `.gguf`
  - Small weights (HyperConnectionLowRankMix, RouterGateUpProj, shared expert down_proj,
    CausalConvWithState, GatedDeltaNet A_log/dt_bias, RMSNorms):
      Converted from the GGUF tensors to `gguf_small_weights.data` & inline initializers
      to reuse the existing optimized contrib operators.

Install compatible official runtime packages or build the runtimes from source
separately. Generated model packages contain model assets only, not runtime libraries.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_package import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_MAX_CONTEXT_LENGTH,
    DFLASH2_RESERVED_BLOCKS,
    NUM_BLOCKS_32K,
    NUM_BLOCKS_128K,
    NUM_BLOCKS_256K,
    PROTOCOL_BOUNDARY_TOKEN_IDS,
    RC5_PREFILL_CONFIG,
    align_up,
    default_base_genai_config,
    ensure_unified_dynamic_cache_model,
    make_ext_init,
)
from package_config import migrate_qwen38_graph_config  # noqa: E402
from package_paths import reject_output_overlap  # noqa: E402

# Exact block counts at block_size=256 with DFLASH2_RESERVED_BLOCKS=13:
#   32K (32,768 tokens -> min 141 blocks, kv_cap=32,776; RC5 33,024 tokens -> 142 blocks, kv_cap=33,032)
#   64K (65,536 tokens -> min 269 blocks, kv_cap=65,544; RC5 65,792 tokens -> 270 blocks, kv_cap=65,800)
#   128K (131,072 tokens -> 525 blocks, kv_cap=131,080)
#   256K (262,144 tokens -> 1037 blocks, kv_cap=262,152)
MIN_NUM_BLOCKS_32K = 128 + DFLASH2_RESERVED_BLOCKS  # 141
MIN_NUM_BLOCKS_64K = 256 + DFLASH2_RESERVED_BLOCKS  # 269
NUM_BLOCKS_64K = 257 + DFLASH2_RESERVED_BLOCKS  # 270
MIN_NUM_BLOCKS_128K = NUM_BLOCKS_128K  # 525
MIN_NUM_BLOCKS_256K = NUM_BLOCKS_256K  # 1037

GGML_TYPE_INFO: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(arr), np.uint16(0x7FC0), u16)


def parse_gguf_header(path: Path) -> tuple[dict, list[dict], int]:
    with open(path, "rb") as rawf:
        buf = rawf.read(16 * 1024 * 1024)
    f = io.BytesIO(buf)
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF", f"Not a GGUF file: {path}"

    def read_str() -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")

    def skip_val(vtype: int) -> object:
        if vtype in (0, 1, 7):
            return struct.unpack("<B", f.read(1))[0]
        if vtype in (2, 3):
            return struct.unpack("<H", f.read(2))[0]
        if vtype in (4, 5):
            return struct.unpack("<I", f.read(4))[0]
        if vtype == 6:
            return struct.unpack("<f", f.read(4))[0]
        if vtype == 8:
            return read_str()
        if vtype == 9:
            atype, alen = struct.unpack("<IQ", f.read(12))
            if atype in (0, 1, 7):
                f.seek(alen, 1)
            elif atype in (2, 3):
                f.seek(alen * 2, 1)
            elif atype in (4, 5, 6):
                f.seek(alen * 4, 1)
            elif atype in (10, 11, 12):
                f.seek(alen * 8, 1)
            else:
                for _ in range(alen):
                    skip_val(atype)
            return f"arr[{alen}]"
        if vtype in (10, 11, 12):
            return struct.unpack("<Q", f.read(8))[0]
        raise ValueError(f"Unknown GGUF kv type {vtype}")

    kvs = {}
    for _ in range(n_kv):
        k = read_str()
        vt = struct.unpack("<I", f.read(4))[0]
        kvs[k] = skip_val(vt)

    align = int(kvs.get("general.alignment", 32))
    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ttype, rel_off = struct.unpack("<IQ", f.read(12))
        tensors.append({"name": name, "dims": dims, "ggml_type": ttype, "rel_offset": rel_off})

    pos = f.tell()
    data_offset = (pos + align - 1) // align * align
    for t in tensors:
        tname, blck_elems, blck_bytes = GGML_TYPE_INFO[t["ggml_type"]]
        d0 = t["dims"][0]
        assert d0 % blck_elems == 0, f"{t['name']}: d0={d0} not divisible by {blck_elems}"
        row_bytes = (d0 // blck_elems) * blck_bytes
        c_dims = [int(d) for d in reversed(t["dims"][1:])]
        n_rows = 1
        for d in c_dims:
            n_rows *= d
        t["type_name"] = tname
        t["row_bytes"] = row_bytes
        t["n_bytes"] = n_rows * row_bytes
        t["data_offset"] = data_offset + t["rel_offset"]
        t["onnx_u8_shape"] = [*c_dims, row_bytes] if c_dims else [row_bytes]

    return kvs, tensors, data_offset


def dequant_gguf_tensor_f32(shard_path: Path, tinfo: dict) -> np.ndarray:
    dims = tinfo["dims"]
    c_shape = tuple(int(d) for d in reversed(dims))
    rows = int(np.prod(c_shape[:-1])) if len(c_shape) > 1 else 1
    cols = int(c_shape[-1])
    with open(shard_path, "rb") as f:
        f.seek(tinfo["data_offset"])
        raw = f.read(tinfo["n_bytes"])
    ttype = tinfo["ggml_type"]
    if ttype == 0:  # F32
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(c_shape)
    if ttype == 30:  # BF16
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).copy().reshape(c_shape)
    if ttype == 8:  # Q8_0
        n_blocks = rows * (cols // 32)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 34)
        d = arr[:, :2].copy().view(np.float16).astype(np.float32)
        qs = arr[:, 2:].view(np.int8).astype(np.float32)
        return (d * qs).reshape(c_shape)
    if ttype == 14:  # Q6_K
        n_blocks = rows * (cols // 256)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 210)
        ql = arr[:, :128]
        qh = arr[:, 128:192]
        sc = arr[:, 192:208].view(np.int8).astype(np.float32)
        d = arr[:, 208:210].copy().view(np.float16).astype(np.float32)
        out = np.zeros((n_blocks, 256), dtype=np.float32)
        for n in range(2):
            ql_n = ql[:, n * 64 : (n + 1) * 64]
            qh_n = qh[:, n * 32 : (n + 1) * 32]
            sc_n = sc[:, n * 8 : (n + 1) * 8]
            for k_idx in range(32):
                is_idx = k_idx // 16
                q1 = ((ql_n[:, k_idx] & 0xF) | (((qh_n[:, k_idx] >> 0) & 3) << 4)).astype(np.int32) - 32
                q2 = ((ql_n[:, k_idx + 32] & 0xF) | (((qh_n[:, k_idx] >> 2) & 3) << 4)).astype(np.int32) - 32
                q3 = ((ql_n[:, k_idx] >> 4) | (((qh_n[:, k_idx] >> 4) & 3) << 4)).astype(np.int32) - 32
                q4 = ((ql_n[:, k_idx + 32] >> 4) | (((qh_n[:, k_idx] >> 6) & 3) << 4)).astype(np.int32) - 32
                out[:, n * 128 + k_idx + 0] = d[:, 0] * sc_n[:, is_idx + 0] * q1
                out[:, n * 128 + k_idx + 32] = d[:, 0] * sc_n[:, is_idx + 2] * q2
                out[:, n * 128 + k_idx + 64] = d[:, 0] * sc_n[:, is_idx + 4] * q3
                out[:, n * 128 + k_idx + 96] = d[:, 0] * sc_n[:, is_idx + 6] * q4
        return out.reshape(c_shape)
    raise ValueError(f"Unsupported small-weight GGUF type {ttype} for {tinfo['name']}")


def set_inline_f32(tensor: TensorProto, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    tensor.data_type = TensorProto.FLOAT
    del tensor.dims[:]
    tensor.dims.extend(arr.shape)
    del tensor.external_data[:]
    tensor.data_location = TensorProto.DEFAULT
    tensor.raw_data = arr.tobytes()


def set_inline_bf16(tensor: TensorProto, arr: np.ndarray) -> None:
    u16 = f32_to_bf16_u16(arr)
    tensor.data_type = TensorProto.BFLOAT16
    del tensor.dims[:]
    tensor.dims.extend(u16.shape)
    del tensor.external_data[:]
    tensor.data_location = TensorProto.DEFAULT
    tensor.raw_data = u16.tobytes()


def estimate_gguf_peak_vram_and_thresholds(total_weight_bytes: int, moe_bytes: int = 0) -> dict:
    """Estimate peak GPU VRAM (`cudaMemGetInfo` used bytes) and `runtime_profiles` VRAM thresholds for GGUF packages.

    Calibrated against empirical H200 measurements across all three GGUF variants:
      - `UD-IQ1_S`   (weights=35.656 GiB / 38.286 GB):
          32K (num_blocks=142, chunk=4096): peak = 41.393 GiB (44.446 GB / 42,387 MiB)
          32K (num_blocks=142, chunk=8192): peak = 43.432 GiB (46.635 GB / 44,475 MiB)
          64K (num_blocks=270, chunk=4096): peak = 41.741 GiB (44.819 GB / 42,743 MiB)
          128K (num_blocks=525, chunk=4096): peak = 42.845 GiB (46.004 GB / 43,873 MiB)
          256K (num_blocks=1037, chunk=4096): peak = 45.538 GiB (48.896 GB / 46,631 MiB)
      - `UD-IQ2_XS`  (weights=44.460 GiB / 47.739 GB):
          32K (num_blocks=142, chunk=4096): peak = 50.138 GiB (53.835 GB / 51,341 MiB)
          32K (num_blocks=142, chunk=8192): peak = 52.177 GiB (56.024 GB / 53,429 MiB)
          64K (num_blocks=270, chunk=4096): peak = 50.485 GiB (54.208 GB / 51,697 MiB)
          128K (num_blocks=525, chunk=4096): peak = 51.589 GiB (55.393 GB / 52,827 MiB)
          256K (num_blocks=1037, chunk=4096): peak = 54.280 GiB (58.283 GB / 55,583 MiB)
      - `UD-IQ3_XXS` (weights=51.730 GiB / 55.545 GB):
          32K (num_blocks=142, chunk=4096): peak = 57.444 GiB (61.680 GB / 58,823 MiB)
          32K (num_blocks=142, chunk=8192): peak = 59.481 GiB (63.868 GB / 60,909 MiB)
          64K (num_blocks=270, chunk=4096): peak = 57.792 GiB (62.054 GB / 59,179 MiB)
          128K (num_blocks=525, chunk=4096): peak = 58.893 GiB (63.236 GB / 60,307 MiB)
          256K (num_blocks=1037, chunk=4096): peak = 61.587 GiB (66.128 GB / 63,065 MiB)
    """
    peak_32k_bytes = total_weight_bytes + 6_160_000_000
    peak_32k_c8192_bytes = total_weight_bytes + 8_350_000_000
    peak_64k_bytes = total_weight_bytes + 6_535_000_000
    peak_128k_bytes = total_weight_bytes + 7_720_000_000
    peak_256k_bytes = total_weight_bytes + 10_610_000_000

    gb = 1_000_000_000
    min_vram_32k = ((peak_32k_bytes + 2 * gb + gb - 1) // gb) * gb
    min_vram_64k = max(min_vram_32k + gb, ((peak_64k_bytes + 2 * gb + gb - 1) // gb) * gb)
    min_vram_128k = max(min_vram_64k + gb, ((peak_128k_bytes + 2 * gb + gb - 1) // gb) * gb)
    min_vram_256k = max(min_vram_128k + 3 * gb, ((peak_256k_bytes + 2_500_000_000 + gb - 1) // gb) * gb)

    return {
        "total_weight_bytes": total_weight_bytes,
        "total_weight_gib": round(total_weight_bytes / 2**30, 3),
        "total_weight_gb": round(total_weight_bytes / 1e9, 3),
        "moe_bytes": moe_bytes,
        "moe_gib": round(moe_bytes / 2**30, 3),
        "min_num_blocks_32k": MIN_NUM_BLOCKS_32K,
        "min_num_blocks_64k": MIN_NUM_BLOCKS_64K,
        "min_num_blocks_128k": MIN_NUM_BLOCKS_128K,
        "min_num_blocks_256k": MIN_NUM_BLOCKS_256K,
        "num_blocks_32k": NUM_BLOCKS_32K,
        "num_blocks_64k": NUM_BLOCKS_64K,
        "num_blocks_128k": NUM_BLOCKS_128K,
        "num_blocks_256k": NUM_BLOCKS_256K,
        "peak_32k_bytes": peak_32k_bytes,
        "peak_32k_gib": round(peak_32k_bytes / 2**30, 3),
        "peak_32k_gb": round(peak_32k_bytes / 1e9, 3),
        "peak_32k_c8192_bytes": peak_32k_c8192_bytes,
        "peak_32k_c8192_gib": round(peak_32k_c8192_bytes / 2**30, 3),
        "peak_32k_c8192_gb": round(peak_32k_c8192_bytes / 1e9, 3),
        "peak_64k_bytes": peak_64k_bytes,
        "peak_64k_gib": round(peak_64k_bytes / 2**30, 3),
        "peak_64k_gb": round(peak_64k_bytes / 1e9, 3),
        "peak_128k_bytes": peak_128k_bytes,
        "peak_128k_gib": round(peak_128k_bytes / 2**30, 3),
        "peak_128k_gb": round(peak_128k_bytes / 1e9, 3),
        "peak_256k_bytes": peak_256k_bytes,
        "peak_256k_gib": round(peak_256k_bytes / 2**30, 3),
        "peak_256k_gb": round(peak_256k_bytes / 1e9, 3),
        "min_vram_32k_bytes": min_vram_32k,
        "min_vram_64k_bytes": min_vram_64k,
        "min_vram_128k_bytes": min_vram_128k,
        "min_vram_256k_bytes": min_vram_256k,
    }


def default_gguf_runtime_profiles(total_weight_bytes: int, decoder_filename: str = "model.onnx") -> list[dict]:
    """Hardware-matched `runtime_profiles` sizing `num_blocks` (`32K`/`64K`/`128K`/`256K`) to the GGUF model's VRAM footprint."""
    vram_info = estimate_gguf_peak_vram_and_thresholds(total_weight_bytes)
    min_32k = vram_info["min_vram_32k_bytes"]
    min_64k = vram_info["min_vram_64k_bytes"]
    min_128k = vram_info["min_vram_128k_bytes"]
    min_256k = vram_info["min_vram_256k_bytes"]
    return [
        {
            "id": "gpu-32k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_32k,
                "maximum_total_device_memory_bytes": min_64k - 1,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_32K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 4096,
                    }
                },
            },
        },
        {
            "id": "gpu-64k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_64k,
                "maximum_total_device_memory_bytes": min_128k - 1,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_64K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "gpu-128k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_128k,
                "maximum_total_device_memory_bytes": min_256k - 1,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_128K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "gpu-256k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_256k,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_256K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
    ]


def configure_gguf_genai_config(
    config: dict,
    *,
    total_weight_bytes: int,
    include_runtime_profiles: bool = True,
) -> dict:
    """Update `genai_config.json` in place using the canonical RC5 schema and GGUF `runtime_profiles`."""
    model_sec = config.setdefault("model", {})
    model_sec.pop("embedding", None)
    for token_key, token_val in PROTOCOL_BOUNDARY_TOKEN_IDS.items():
        model_sec[token_key] = token_val
    model_sec["context_length"] = DEFAULT_MAX_CONTEXT_LENGTH

    dec_so = model_sec.setdefault("decoder", {}).setdefault("session_options", {})
    dec_so.pop("ep.cuda.qmoe_int_dequant_max_scratch_bytes", None)
    dec_so.pop("ep.cuda.qmoe_row_tile_size", None)
    for po in dec_so.get("provider_options", []):
        if isinstance(po, dict) and isinstance(po.get("cuda"), dict):
            po["cuda"].pop("min_num_runs_before_cuda_graph_capture", None)

    qwen = model_sec.get("qwen38_graph")
    if isinstance(qwen, dict):
        migrate_qwen38_graph_config(qwen)
        prefill = qwen.setdefault("prefill", {})
        if prefill.get("enabled", True):
            prefill["chunk"] = RC5_PREFILL_CONFIG["chunk"]
            prefill["warmup_lengths"] = list(RC5_PREFILL_CONFIG["warmup_lengths"])

    config["engine"] = json.loads(json.dumps(DEFAULT_ENGINE_CONFIG))
    config.setdefault("search", {})["max_length"] = DEFAULT_MAX_CONTEXT_LENGTH
    if include_runtime_profiles:
        dec_fname = model_sec.get("decoder", {}).get("filename", "model.onnx")
        config["runtime_profiles"] = default_gguf_runtime_profiles(total_weight_bytes, dec_fname)
    else:
        config.pop("runtime_profiles", None)
    return config


class SmallWeightBlobWriter:
    def __init__(self, out_dir: Path, filename: str = "gguf_small_weights.data") -> None:
        self.filename = filename
        self.path = out_dir / filename
        self.fh = open(self.path, "wb")  # noqa: SIM115
        self.pos = 0

    def write_bf16(self, arr_f32: np.ndarray) -> tuple[str, int, int, int, list[int]]:
        u16 = f32_to_bf16_u16(arr_f32)
        raw = u16.tobytes()
        pad = align_up(self.pos) - self.pos
        if pad:
            self.fh.write(b"\x00" * pad)
            self.pos += pad
        off = self.pos
        self.fh.write(raw)
        self.pos += len(raw)
        return self.filename, off, len(raw), TensorProto.BFLOAT16, list(u16.shape)

    def write_raw(self, raw: bytes, dtype: int, dims: list[int]) -> tuple[str, int, int, int, list[int]]:
        pad = align_up(self.pos) - self.pos
        if pad:
            self.fh.write(b"\x00" * pad)
            self.pos += pad
        off = self.pos
        self.fh.write(raw)
        self.pos += len(raw)
        return self.filename, off, len(raw), dtype, list(dims)

    def close(self) -> None:
        self.fh.close()


def build_gguf_package(
    variant_name: str,
    gguf_dir: Path,
    engram_gguf: Path,
    base_pkg: Path,
    out_dir: Path,
) -> dict:
    t0 = time.time()
    out_dir = reject_output_overlap(
        out_dir,
        {
            "--gguf-root variant": gguf_dir,
            "engram GGUF": engram_gguf,
            "--base-pkg": base_pkg,
        },
    )
    if out_dir.exists() or out_dir.is_symlink():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. Parse all GGUF shards in gguf_dir and hard-link non-empty shards + engram into out_dir
    gguf_tensors: dict[str, dict] = {}
    shard_files: list[str] = []
    for shard_path in sorted(gguf_dir.glob("*.gguf")):
        _, ts, _ = parse_gguf_header(shard_path)
        if not ts or (len(ts) == 1 and ts[0]["name"] == "per_layer_token_embd.weight"):
            continue
        dst_shard = out_dir / shard_path.name
        os.link(shard_path, dst_shard)
        shard_files.append(shard_path.name)
        for t in ts:
            t["shard_file"] = shard_path.name
            t["shard_path"] = shard_path
            gguf_tensors[t["name"]] = t

    # Hard-link engram GGUF
    engram_dst_name = "Qwen3.8-Flash-Next-engram-IQ4_NL.gguf"
    os.link(engram_gguf, out_dir / engram_dst_name)
    _, engram_ts, engram_doff = parse_gguf_header(engram_gguf)
    assert len(engram_ts) == 1 and engram_ts[0]["name"] == "per_layer_token_embd.weight"
    assert engram_ts[0]["ggml_type"] == 20  # IQ4_NL

    print(f"[{variant_name}] Indexed {len(gguf_tensors)} tensors across {shard_files} + {engram_dst_name}")

    # 2. Copy static non-weight assets from base_pkg
    for fname in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "expert-profile.bin",
        "attn_consts.data",
        "dflash2.onnx.data",
    ):
        src = base_pkg / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    # 3. Write ple_table.txt pointing directly to Qwen3.8-Flash-Next-engram-IQ4_NL.gguf (scale -20.0 = IQ4_NL)
    ple_lines = [
        "qwen38_ple_table 1",
        "n_heads 16",
        "head_dim 160",
        "heads_per_ngram 8",
        "eos 248044",
        "scale -20.0",
        "mult 23703573157769 20109073645365 8052911324071 0",
        "vocab 20000003 20000023 20000033 20000047 20000059 20000063 20000069 20000077 20000081 20000093 20000107 20000147 20000153 20000159 20000161 20000171",
        "offset 0 20000003 40000026 60000059 80000106 100000165 120000228 140000297 160000374 180000455 200000548 220000655 240000802 260000955 280001114 300001275",
    ]
    ple_row_bytes = (160 // 32) * 18  # 90 bytes per row
    rows_per_shard = 2500012
    total_ple_rows = 300001275 + 20000171  # 320,001,446
    cur_row = 0
    for _s_idx in range(128):
        r_cnt = min(rows_per_shard, total_ple_rows - cur_row)
        if r_cnt <= 0:
            break
        byte_off = engram_doff + cur_row * ple_row_bytes
        ple_lines.append(f"shard {engram_dst_name} {byte_off} {r_cnt}")
        cur_row += r_cnt
    (out_dir / "ple_table.txt").write_text("\n".join(ple_lines) + "\n")

    # 4. Convert small weights from GGUF + copy RoPE caches into gguf_small_weights.data
    small_writer = SmallWeightBlobWriter(out_dir, "gguf_small_weights.data")
    base_model = onnx.load(str(base_pkg / "model.onnx"), load_external_data=False)
    base_inits = {t.name: t for t in base_model.graph.initializer}

    def read_base_ext_bytes(init_name: str) -> bytes:
        t = base_inits[init_name]
        ext = {e.key: e.value for e in t.external_data}
        with open(base_pkg / ext["location"], "rb") as f:
            f.seek(int(ext.get("offset", 0)))
            return f.read(int(ext["length"]))

    # Copy cos_cache and sin_cache (32 MiB total) into gguf_small_weights.data at offsets 0 and 16777216
    cos_info = small_writer.write_raw(
        read_base_ext_bytes("model.rotary_emb.cos_cache"), TensorProto.BFLOAT16, [262144, 32]
    )
    sin_info = small_writer.write_raw(
        read_base_ext_bytes("model.rotary_emb.sin_cache"), TensorProto.BFLOAT16, [262144, 32]
    )

    true_perm = np.arange(48).reshape(16, 3).T.flatten()
    inv_perm = np.argsort(true_perm)

    new_inits: dict[str, TensorProto] = {}
    gguf_init_map: dict[str, dict] = {}

    def register_gguf_direct_init(onnx_name: str, gguf_name: str, share: bool = False) -> tuple[TensorProto, dict]:
        tinfo = gguf_tensors[gguf_name]
        init_tp = make_ext_init(
            onnx_name,
            tinfo["shard_file"],
            tinfo["data_offset"],
            tinfo["n_bytes"],
            TensorProto.UINT8,
            tinfo["onnx_u8_shape"],
        )
        new_inits[onnx_name] = init_tp
        entry = {
            "file": tinfo["shard_file"],
            "gguf_name": gguf_name,
            "data_type": int(TensorProto.UINT8),
            "shape": tinfo["onnx_u8_shape"],
        }
        if share:
            entry["share"] = True
        gguf_init_map[onnx_name] = entry
        return init_tp, tinfo

    def add_small_bf16_init(onnx_name: str, arr_f32: np.ndarray) -> None:
        loc, off, ln, dt, dims = small_writer.write_bf16(arr_f32)
        new_inits[onnx_name] = make_ext_init(onnx_name, loc, off, ln, dt, dims)

    # Register RoPE and indexer caches
    new_inits["model.rotary_emb.cos_cache"] = make_ext_init("model.rotary_emb.cos_cache", *cos_info)
    new_inits["model.rotary_emb.sin_cache"] = make_ext_init("model.rotary_emb.sin_cache", *sin_info)
    new_inits["capture.indexer_cos_cache.folded"] = base_inits["capture.indexer_cos_cache.folded"]
    new_inits["capture.indexer_sin_cache.folded"] = base_inits["capture.indexer_sin_cache.folded"]

    # Copy all inline initializers from base_model first, then update norms / A_log / dt_bias from GGUF
    for name, t in base_inits.items():
        ext = {e.key: e.value for e in t.external_data}
        if "location" not in ext:
            new_inits[name] = t

    # Update inline norms and GDN parameters from GGUF
    for name in list(new_inits.keys()):
        m_hc_attn = re.match(r"^v_model\.layers\.(\d+)\.attn_hyper_connection\.hc_norm\..*\.folded$", name)
        m_hc_mlp = re.match(r"^v_model\.layers\.(\d+)\.mlp_hyper_connection\.hc_norm\..*\.folded$", name)
        m_q_norm = re.match(r"^v_model\.layers\.(\d+)\.self_attn\.q_norm\..*\.folded$", name)
        m_k_norm = re.match(r"^v_model\.layers\.(\d+)\.self_attn\.k_norm\..*\.folded$", name)
        m_iq_norm = re.match(r"^v_model\.layers\.(\d+)\.self_attn\.indexer\.q_layernorm\..*\.folded\.bf16$", name)
        m_ik_norm = re.match(r"^v_model\.layers\.(\d+)\.self_attn\.indexer\.k_layernorm\..*\.folded\.bf16$", name)
        m_alog = re.match(r"^model\.layers\.(\d+)\.linear_attn\.A_log\.f32$", name)
        m_dt = re.match(r"^model\.layers\.(\d+)\.linear_attn\.dt_bias\.f32$", name)
        if m_hc_attn:
            lid = int(m_hc_attn.group(1))
            ti = gguf_tensors[f"blk.{lid}.hc_attn_norm.weight"]
            set_inline_bf16(new_inits[name], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())
        elif m_hc_mlp:
            lid = int(m_hc_mlp.group(1))
            ti = gguf_tensors[f"blk.{lid}.hc_ffn_norm.weight"]
            set_inline_bf16(new_inits[name], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())
        elif m_q_norm:
            lid = int(m_q_norm.group(1))
            ti = gguf_tensors[f"blk.{lid}.attn_q_norm.weight"]
            set_inline_f32(new_inits[name], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())
        elif m_k_norm:
            lid = int(m_k_norm.group(1))
            ti = gguf_tensors[f"blk.{lid}.attn_k_norm.weight"]
            set_inline_f32(new_inits[name], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())
        elif m_iq_norm:
            lid = int(m_iq_norm.group(1))
            ti = gguf_tensors[f"blk.{lid}.indexer.q_norm.weight"]
            set_inline_bf16(new_inits[name], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())
        elif m_ik_norm:
            lid = int(m_ik_norm.group(1))
            ti = gguf_tensors[f"blk.{lid}.indexer.k_norm.weight"]
            set_inline_bf16(new_inits[name], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())
        elif m_alog:
            lid = int(m_alog.group(1))
            ti = gguf_tensors[f"blk.{lid}.ssm_a"]
            ssm_a = dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten()[inv_perm]
            set_inline_f32(new_inits[name], np.log(-ssm_a))
        elif m_dt:
            lid = int(m_dt.group(1))
            ti = gguf_tensors[f"blk.{lid}.ssm_dt.bias"]
            dt_b = dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten()[inv_perm]
            set_inline_f32(new_inits[name], dt_b)

    # PLE inline norms & final HC norm
    for gname, oname in (
        ("blk.1.ple_norm_key.weight", "v_model.layers.1.ple.norm_key.Add_200.folded"),
        ("blk.1.ple_norm_query.weight", "v_model.layers.1.ple.norm_query.Add_204.folded"),
        ("blk.1.ple_norm_conv.weight", "v_model.layers.1.ple.norm_conv.Add_224.folded"),
        ("output_hc_norm.weight", "v_model.hyper_connection_mixer.hc_norm.Add_5815.folded"),
    ):
        ti = gguf_tensors[gname]
        set_inline_bf16(new_inits[oname], dequant_gguf_tensor_f32(ti["shard_path"], ti).flatten())

    # Convert small external weights per layer from GGUF
    for lid in range(48):
        # HyperConnection attn & mlp down/up/inject
        for kind_onnx, kind_gguf in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            ti_down = gguf_tensors[f"blk.{lid}.{kind_gguf}_down.weight"]
            w_down = dequant_gguf_tensor_f32(ti_down["shard_path"], ti_down).T  # [10240, 320]
            add_small_bf16_init(
                f"model.layers.{lid}.{kind_onnx}.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16", w_down
            )

            ti_up = gguf_tensors[f"blk.{lid}.{kind_gguf}_up.weight"]
            w_up = dequant_gguf_tensor_f32(ti_up["shard_path"], ti_up).T  # [320, 10240]
            add_small_bf16_init(
                f"model.layers.{lid}.{kind_onnx}.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16", w_up
            )

            ti_inj = gguf_tensors[f"blk.{lid}.{kind_gguf}_inject.weight"]
            w_inj = dequant_gguf_tensor_f32(ti_inj["shard_path"], ti_inj).T  # [10240, 4]
            add_small_bf16_init(f"model.layers.{lid}.{kind_onnx}.block_inject_weight.weight_t", w_inj)

        # RouterGateUpProj merged weight [2560, 1792] = [router(512) | shared_gate(640) | shared_up(640)]
        ti_r = gguf_tensors[f"blk.{lid}.ffn_gate_inp.weight"]
        ti_sg = gguf_tensors[f"blk.{lid}.ffn_gate_shexp.weight"]
        ti_su = gguf_tensors[f"blk.{lid}.ffn_up_shexp.weight"]
        w_r = dequant_gguf_tensor_f32(ti_r["shard_path"], ti_r).T  # [2560, 512]
        w_sg = dequant_gguf_tensor_f32(ti_sg["shard_path"], ti_sg).T  # [2560, 640]
        w_su = dequant_gguf_tensor_f32(ti_su["shard_path"], ti_su).T  # [2560, 640]
        w_rgup = np.concatenate([w_r, w_sg, w_su], axis=1)
        rgup_idx = lid * 2 + 1 if lid == 0 else lid * 2 + 2
        add_small_bf16_init(f"merged_proj.{rgup_idx}.weight_t", w_rgup)

        # Shared expert down_proj [640, 2560] and gate [2560, 1]
        ti_sd = gguf_tensors[f"blk.{lid}.ffn_down_shexp.weight"]
        w_sd = dequant_gguf_tensor_f32(ti_sd["shard_path"], ti_sd).T  # [640, 2560]
        add_small_bf16_init(f"model.layers.{lid}.mlp.shared_expert.down_proj.weight_t", w_sd)

        ti_sgate = gguf_tensors[f"blk.{lid}.ffn_gate_inp_shexp.weight"]
        w_sgate = dequant_gguf_tensor_f32(ti_sgate["shard_path"], ti_sgate).reshape(2560, 1)
        add_small_bf16_init(f"model.layers.{lid}.mlp.shared_expert_gate.weight_t", w_sgate)

        # GDN conv1d, norm, alpha, beta (for GDN layers lid % 4 != 3) or QSA indexer_qk (for lid % 4 == 3)
        if lid % 4 != 3:
            ti_conv = gguf_tensors[f"blk.{lid}.ssm_conv1d.weight"]
            w_conv = dequant_gguf_tensor_f32(ti_conv["shard_path"], ti_conv).reshape(10240, 4).copy()
            w_conv[4096:] = w_conv[4096:].reshape(48, 128, 4)[inv_perm].reshape(6144, 4)
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.conv1d.weight", w_conv.reshape(10240, 1, 4))

            ti_snorm = gguf_tensors[f"blk.{lid}.ssm_norm.weight"]
            w_snorm = dequant_gguf_tensor_f32(ti_snorm["shard_path"], ti_snorm).flatten()
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.norm.weight", w_snorm)

            ti_alpha = gguf_tensors[f"blk.{lid}.ssm_alpha.weight"]
            w_alpha = dequant_gguf_tensor_f32(ti_alpha["shard_path"], ti_alpha)[inv_perm].T  # [2560, 48]
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.alpha.weight_t", w_alpha)

            ti_beta = gguf_tensors[f"blk.{lid}.ssm_beta.weight"]
            w_beta = dequant_gguf_tensor_f32(ti_beta["shard_path"], ti_beta)[inv_perm].T  # [2560, 48]
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.beta.weight_t", w_beta)
        else:
            ti_iq = gguf_tensors[f"blk.{lid}.indexer.q_proj.weight"]
            ti_ik = gguf_tensors[f"blk.{lid}.indexer.k_proj.weight"]
            w_iq = dequant_gguf_tensor_f32(ti_iq["shard_path"], ti_iq).T  # [2560, 512]
            w_ik = dequant_gguf_tensor_f32(ti_ik["shard_path"], ti_ik).T  # [2560, 128]
            add_small_bf16_init(
                f"model.layers.{lid}.self_attn.indexer_qk.weight_t", np.concatenate([w_iq, w_ik], axis=1)
            )

    # Layer 1 PLE conv1d & output HC down/up
    ti_ple_conv = gguf_tensors["blk.1.ple_conv1d.weight"]
    w_ple_conv = dequant_gguf_tensor_f32(ti_ple_conv["shard_path"], ti_ple_conv).reshape(10240, 1, 4)
    add_small_bf16_init("model.layers.1.ple.conv1d.weight", w_ple_conv)

    ti_out_down = gguf_tensors["output_hc_down.weight"]
    add_small_bf16_init(
        "model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
        dequant_gguf_tensor_f32(ti_out_down["shard_path"], ti_out_down).T,
    )
    ti_out_up = gguf_tensors["output_hc_up.weight"]
    add_small_bf16_init(
        "model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
        dequant_gguf_tensor_f32(ti_out_up["shard_path"], ti_out_up).T,
    )

    small_writer.close()

    # 5. Register token_embd.weight and rewrite model.onnx nodes (GatherPacked, MatMulPacked, MoEPacked)
    _, emb_tinfo = register_gguf_direct_init("model.embed_tokens.weight", "token_embd.weight", share=True)

    new_nodes: list[onnx.NodeProto] = []
    # Prepend GatherPacked node so model.onnx computes inputs_embeds on GPU directly from token_embd.weight
    gather_node = helper.make_node(
        "GatherPacked",
        ["model.embed_tokens.weight", "input_ids"],
        ["inputs_embeds"],
        name="model/embed_tokens/GatherPacked",
        domain="com.microsoft",
        K=2560,
        packed_type=int(emb_tinfo["ggml_type"]),
        output_dtype=int(TensorProto.BFLOAT16),
    )
    new_nodes.append(gather_node)

    # Helper shape constants for GDN [3, 16, 128] <-> [16, 3, 128] V-head permutation in the ONNX graph
    new_inits["const_gdn_split_4096_6144"] = numpy_helper.from_array(
        np.array([4096, 6144], dtype=np.int64), name="const_gdn_split_4096_6144"
    )
    new_inits["const_gdn_v_3_16_128"] = numpy_helper.from_array(
        np.array([0, 0, 3, 16, 128], dtype=np.int64), name="const_gdn_v_3_16_128"
    )
    new_inits["const_gdn_v_6144"] = numpy_helper.from_array(
        np.array([0, 0, 6144], dtype=np.int64), name="const_gdn_v_6144"
    )
    new_inits["const_gdn_out_16_3_128"] = numpy_helper.from_array(
        np.array([-1, 16, 3, 128], dtype=np.int64), name="const_gdn_out_16_3_128"
    )

    # Map merged_proj indices to layers and consumer Split nodes in base_model
    node_by_input: dict[str, list[onnx.NodeProto]] = {}
    for n in base_model.graph.node:
        for inp in n.input:
            node_by_input.setdefault(inp, []).append(n)

    removed_split_names: set[str] = set()
    gdn_output_tensor_names: set[str] = set()
    for n in base_model.graph.node:
        if n.op_type == "GatedDeltaNet":
            gdn_output_tensor_names.add(n.output[0])

    matmul_packed_count = 0
    moe_packed_count = 0

    for n in base_model.graph.node:
        if n.name in removed_split_names:
            continue
        if n.op_type == "Reshape" and len(n.input) >= 2 and n.input[0] in gdn_output_tensor_names:
            # Permute GatedDeltaNet output from HF [16, 3, 128] order to GGUF [3, 16, 128] order
            # so both in_proj_z (attn_gate.weight) and out_proj (ssm_out.weight) are consumed unpermuted.
            gdn_out = n.input[0]
            r4d = f"{gdn_out}.reshape_16_3_128"
            t4d = f"{gdn_out}.transpose_3_16_128"
            new_nodes.append(
                helper.make_node("Reshape", [gdn_out, "const_gdn_out_16_3_128"], [r4d], name=f"{n.name}/Reshape4D")
            )
            new_nodes.append(
                helper.make_node("Transpose", [r4d], [t4d], name=f"{n.name}/TransposeVHeads", perm=[0, 2, 1, 3])
            )
            new_nodes.append(helper.make_node("Reshape", [t4d, n.input[1]], list(n.output), name=n.name))
        elif n.op_type == "QMoE":
            fc1_w_name = n.input[2]
            m_layer = re.match(r"^model\.layers\.(\d+)\.mlp\.fc1_experts_weights$", fc1_w_name)
            assert m_layer is not None, f"Unexpected QMoE weight {fc1_w_name}"
            lid = int(m_layer.group(1))
            gw_name = f"model.layers.{lid}.mlp.gate_experts_weights"
            uw_name = f"model.layers.{lid}.mlp.up_experts_weights"
            dw_name = f"model.layers.{lid}.mlp.down_experts_weights"
            _, gw_info = register_gguf_direct_init(gw_name, f"blk.{lid}.ffn_gate_exps.weight")
            _, uw_info = register_gguf_direct_init(uw_name, f"blk.{lid}.ffn_up_exps.weight")
            _, dw_info = register_gguf_direct_init(dw_name, f"blk.{lid}.ffn_down_exps.weight")
            moe_node = helper.make_node(
                "MoEPacked",
                [n.input[0], n.input[1], gw_name, uw_name, dw_name],
                list(n.output),
                name=n.name.replace("QMoE", "MoEPacked"),
                domain="com.microsoft",
                k=10,
                normalize_routing_weights=1,
                activation_type="swiglu",
                gate_type=int(gw_info["ggml_type"]),
                up_type=int(uw_info["ggml_type"]),
                down_type=int(dw_info["ggml_type"]),
            )
            new_nodes.append(moe_node)
            moe_packed_count += 1
        elif n.op_type == "MatMulNBits":
            w_name = n.input[1]
            a_name = n.input[0]
            out_name = n.output[0]
            m_gdn_out = re.match(r"^model\.layers\.(\d+)\.linear_attn\.out_proj\.weight_t\.qweight$", w_name)
            m_qsa_out = re.match(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight_t\.qweight$", w_name)
            if m_gdn_out:
                lid = int(m_gdn_out.group(1))
                b_name = f"model.layers.{lid}.linear_attn.out_proj.gguf_weight"
                _, b_info = register_gguf_direct_init(b_name, f"blk.{lid}.ssm_out.weight")
                new_nodes.append(
                    helper.make_node(
                        "MatMulPacked",
                        [a_name, b_name],
                        [out_name],
                        name=n.name.replace("MatMulNBits", "MatMulPacked"),
                        domain="com.microsoft",
                        K=6144,
                        N=2560,
                        packed_type=int(b_info["ggml_type"]),
                    )
                )
                matmul_packed_count += 1
            elif m_qsa_out:
                lid = int(m_qsa_out.group(1))
                b_name = f"model.layers.{lid}.self_attn.o_proj.gguf_weight"
                _, b_info = register_gguf_direct_init(b_name, f"blk.{lid}.attn_output.weight")
                new_nodes.append(
                    helper.make_node(
                        "MatMulPacked",
                        [a_name, b_name],
                        [out_name],
                        name=n.name.replace("MatMulNBits", "MatMulPacked"),
                        domain="com.microsoft",
                        K=6144,
                        N=2560,
                        packed_type=int(b_info["ggml_type"]),
                    )
                )
                matmul_packed_count += 1
            elif w_name == "lm_head.weight_t.qweight":
                b_name = "lm_head.gguf_weight"
                _, b_info = register_gguf_direct_init(b_name, "output.weight", share=True)
                new_nodes.append(
                    helper.make_node(
                        "MatMulPacked",
                        [a_name, b_name],
                        [out_name],
                        name=n.name.replace("MatMulNBits", "MatMulPacked"),
                        domain="com.microsoft",
                        K=2560,
                        N=248320,
                        packed_type=int(b_info["ggml_type"]),
                    )
                )
                matmul_packed_count += 1
            elif w_name == "merged_proj.2.weight_t.qweight":
                # Layer 1 PLE key + value: replace merged MatMulNBits + Split with two single-weight MatMulPacked nodes
                split_consumers = [c for c in node_by_input.get(out_name, []) if c.op_type == "Split"]
                assert len(split_consumers) == 1
                sp_node = split_consumers[0]
                removed_split_names.add(sp_node.name)
                key_out, val_out = sp_node.output[0], sp_node.output[1]

                b0_name = "model.layers.1.ple.key_proj.gguf_weight"
                b1_name = "model.layers.1.ple.value_proj.gguf_weight"
                _, b0_info = register_gguf_direct_init(b0_name, "blk.1.ple_key.weight")
                _, b1_info = register_gguf_direct_init(b1_name, "blk.1.ple_value.weight")
                new_nodes.append(
                    helper.make_node(
                        "MatMulPacked",
                        [a_name, b0_name],
                        [key_out],
                        name=f"{n.name}/key_proj/MatMulPacked",
                        domain="com.microsoft",
                        K=2560,
                        N=10240,
                        packed_type=int(b0_info["ggml_type"]),
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "MatMulPacked",
                        [a_name, b1_name],
                        [val_out],
                        name=f"{n.name}/value_proj/MatMulPacked",
                        domain="com.microsoft",
                        K=2560,
                        N=2560,
                        packed_type=int(b1_info["ggml_type"]),
                    )
                )
                matmul_packed_count += 2
            else:
                # Merged GDN or QSA input projection: replace merged MatMulNBits + Split with per-tensor ops
                split_consumers = [c for c in node_by_input.get(out_name, []) if c.op_type == "Split"]
                assert len(split_consumers) == 1, f"Expected 1 Split consumer for {out_name}"
                sp_node = split_consumers[0]
                removed_split_names.add(sp_node.name)

                m_in_layer = re.search(r"layers\.(\d+)\.", a_name)
                assert m_in_layer is not None, f"Cannot infer layer from {a_name} for {w_name}"
                lid = int(m_in_layer.group(1))
                if lid % 4 != 3:
                    # GDN input projection: in base_model, sp_node.output[2] (*.a_48) is wired to
                    # GatedDeltaNet input[5] (beta = ssm_beta) and sp_node.output[3] (*.b_48) is wired
                    # to GatedDeltaNet input[4] (decay = ssm_alpha).
                    qkv_out, gate_out, b_out, a_out = (
                        sp_node.output[0],
                        sp_node.output[1],
                        sp_node.output[2],
                        sp_node.output[3],
                    )
                    qkv_w_name = f"model.layers.{lid}.linear_attn.qkv.gguf_weight"
                    gate_w_name = f"model.layers.{lid}.linear_attn.gate.gguf_weight"
                    _, qkv_info = register_gguf_direct_init(qkv_w_name, f"blk.{lid}.attn_qkv.weight")
                    _, gate_info = register_gguf_direct_init(gate_w_name, f"blk.{lid}.attn_gate.weight")

                    qkv_raw = f"{qkv_out}.gguf_raw"
                    qk_part = f"{qkv_out}.qk_4096"
                    v_gguf = f"{qkv_out}.v_gguf_6144"
                    v_5d = f"{qkv_out}.v_3_16_128"
                    v_5d_t = f"{qkv_out}.v_16_3_128"
                    v_hf = f"{qkv_out}.v_hf_6144"

                    new_nodes.append(
                        helper.make_node(
                            "MatMulPacked",
                            [a_name, qkv_w_name],
                            [qkv_raw],
                            name=f"{n.name}/qkv/MatMulPacked",
                            domain="com.microsoft",
                            K=2560,
                            N=10240,
                            packed_type=int(qkv_info["ggml_type"]),
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "Split",
                            [qkv_raw, "const_gdn_split_4096_6144"],
                            [qk_part, v_gguf],
                            name=f"{n.name}/qkv/SplitQKV",
                            axis=-1,
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "Reshape",
                            [v_gguf, "const_gdn_v_3_16_128"],
                            [v_5d],
                            name=f"{n.name}/qkv/ReshapeV5D",
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "Transpose",
                            [v_5d],
                            [v_5d_t],
                            name=f"{n.name}/qkv/TransposeVHeads",
                            perm=[0, 1, 3, 2, 4],
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "Reshape",
                            [v_5d_t, "const_gdn_v_6144"],
                            [v_hf],
                            name=f"{n.name}/qkv/ReshapeVFlat",
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "Concat",
                            [qk_part, v_hf],
                            [qkv_out],
                            name=f"{n.name}/qkv/ConcatQKV",
                            axis=-1,
                        )
                    )

                    new_nodes.append(
                        helper.make_node(
                            "MatMulPacked",
                            [a_name, gate_w_name],
                            [gate_out],
                            name=f"{n.name}/gate/MatMulPacked",
                            domain="com.microsoft",
                            K=2560,
                            N=6144,
                            packed_type=int(gate_info["ggml_type"]),
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "MatMul",
                            [a_name, f"model.layers.{lid}.linear_attn.alpha.weight_t"],
                            [a_out],
                            name=f"{n.name}/alpha/MatMul",
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "MatMul",
                            [a_name, f"model.layers.{lid}.linear_attn.beta.weight_t"],
                            [b_out],
                            name=f"{n.name}/beta/MatMul",
                        )
                    )
                    matmul_packed_count += 2
                else:
                    # QSA input projection: indexer_qk (640), attn_q (12288), attn_k (512), attn_v (512)
                    idx_qk_out, q_out, k_out, v_out = (
                        sp_node.output[0],
                        sp_node.output[1],
                        sp_node.output[2],
                        sp_node.output[3],
                    )
                    qw_name = f"model.layers.{lid}.self_attn.q.gguf_weight"
                    kw_name = f"model.layers.{lid}.self_attn.k.gguf_weight"
                    vw_name = f"model.layers.{lid}.self_attn.v.gguf_weight"
                    _, qw_info = register_gguf_direct_init(qw_name, f"blk.{lid}.attn_q.weight")
                    _, kw_info = register_gguf_direct_init(kw_name, f"blk.{lid}.attn_k.weight")
                    _, vw_info = register_gguf_direct_init(vw_name, f"blk.{lid}.attn_v.weight")

                    new_nodes.append(
                        helper.make_node(
                            "MatMul",
                            [a_name, f"model.layers.{lid}.self_attn.indexer_qk.weight_t"],
                            [idx_qk_out],
                            name=f"{n.name}/indexer_qk/MatMul",
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "MatMulPacked",
                            [a_name, qw_name],
                            [q_out],
                            name=f"{n.name}/q_proj/MatMulPacked",
                            domain="com.microsoft",
                            K=2560,
                            N=12288,
                            packed_type=int(qw_info["ggml_type"]),
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "MatMulPacked",
                            [a_name, kw_name],
                            [k_out],
                            name=f"{n.name}/k_proj/MatMulPacked",
                            domain="com.microsoft",
                            K=2560,
                            N=512,
                            packed_type=int(kw_info["ggml_type"]),
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "MatMulPacked",
                            [a_name, vw_name],
                            [v_out],
                            name=f"{n.name}/v_proj/MatMulPacked",
                            domain="com.microsoft",
                            K=2560,
                            N=512,
                            packed_type=int(vw_info["ggml_type"]),
                        )
                    )
                    matmul_packed_count += 3
        else:
            new_nodes.append(n)

    assert moe_packed_count == 48, f"Expected 48 MoEPacked nodes, got {moe_packed_count}"
    assert matmul_packed_count == 159, f"Expected 159 MatMulPacked nodes, got {matmul_packed_count}"

    # Update model.onnx graph inputs (remove inputs_embeds since GatherPacked computes it inside model.onnx)
    kept_inputs = [vi for vi in base_model.graph.input if vi.name != "inputs_embeds"]
    del base_model.graph.input[:]
    base_model.graph.input.extend(kept_inputs)

    del base_model.graph.node[:]
    base_model.graph.node.extend(new_nodes)

    # Keep only referenced initializers in model.onnx and ensure dynamic cache dimensions (32K/64K/128K/256K)
    used_inputs = {inp for n in base_model.graph.node for inp in n.input if inp}
    final_inits = [init_tp for name, init_tp in new_inits.items() if name in used_inputs]
    del base_model.graph.initializer[:]
    base_model.graph.initializer.extend(final_inits)
    ensure_unified_dynamic_cache_model(base_model)

    onnx.save(base_model, str(out_dir / "model.onnx"))

    # Compute total GPU weight bytes (MoEPacked + MatMulPacked + GatherPacked + gguf_small_weights.data)
    moe_weight_bytes = 0
    for n in base_model.graph.node:
        if n.op_type == "MoEPacked":
            for wi in n.input[2:5]:
                moe_weight_bytes += gguf_tensors[gguf_init_map[wi]["gguf_name"]]["n_bytes"]
    gguf_direct_bytes = sum(
        gguf_tensors[meta["gguf_name"]]["n_bytes"] for name, meta in gguf_init_map.items() if name in used_inputs
    )
    small_weight_bytes = (out_dir / "gguf_small_weights.data").stat().st_size
    total_weight_bytes = gguf_direct_bytes + small_weight_bytes
    vram_info = estimate_gguf_peak_vram_and_thresholds(total_weight_bytes, moe_bytes=moe_weight_bytes)

    # Also build standalone embedding.onnx using GatherPacked on CPU EP
    emb_graph = helper.make_graph(
        [
            helper.make_node(
                "GatherPacked",
                ["model.embed_tokens.weight", "input_ids"],
                ["inputs_embeds"],
                name="/model/embed_tokens/GatherPacked",
                domain="com.microsoft",
                K=2560,
                packed_type=int(emb_tinfo["ggml_type"]),
                output_dtype=int(TensorProto.BFLOAT16),
            )
        ],
        "embedding_gguf",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.BFLOAT16, ["num_tokens", 2560])],
        [
            make_ext_init(
                "model.embed_tokens.weight",
                emb_tinfo["shard_file"],
                emb_tinfo["data_offset"],
                emb_tinfo["n_bytes"],
                TensorProto.UINT8,
                emb_tinfo["onnx_u8_shape"],
            )
        ],
    )
    emb_model = helper.make_model(
        emb_graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save(emb_model, str(out_dir / "embedding.onnx"))

    # 6. Rewrite dflash2.onnx to use GatherPacked (token_embd.weight) and MatMulPacked (output.weight)
    dflash_model = onnx.load(str(base_pkg / "dflash2.onnx"), load_external_data=False)
    dflash_nodes: list[onnx.NodeProto] = []
    skip_dflash_embed_nodes = {
        "/dflash2/embed_tokens/Gather_i8",
        "/dflash2/embed_tokens/Gather_scale",
        "/dflash2/embed_tokens/Cast_f32",
        "/dflash2/embed_tokens/Mul_dequant",
        "/dflash2/embed_tokens/Cast",
    }
    lm_tinfo = gguf_tensors["output.weight"]
    for n in dflash_model.graph.node:
        if n.name == "/dflash2/embed_tokens/Gather_i8":
            dflash_nodes.append(
                helper.make_node(
                    "GatherPacked",
                    ["model.embed_tokens.weight", "input_ids"],
                    ["/dflash2/embed_tokens/Cast/output_0"],
                    name="/dflash2/embed_tokens/GatherPacked",
                    domain="com.microsoft",
                    K=2560,
                    packed_type=int(emb_tinfo["ggml_type"]),
                    output_dtype=int(TensorProto.BFLOAT16),
                )
            )
        elif n.name in skip_dflash_embed_nodes:
            continue
        elif n.name == "/lm_head/MatMul":
            dflash_nodes.append(
                helper.make_node(
                    "MatMulPacked",
                    [n.input[0], "lm_head.gguf_weight"],
                    list(n.output),
                    name="/lm_head/MatMulPacked",
                    domain="com.microsoft",
                    K=2560,
                    N=248320,
                    packed_type=int(lm_tinfo["ggml_type"]),
                )
            )
        else:
            dflash_nodes.append(n)

    del dflash_model.graph.node[:]
    dflash_model.graph.node.extend(dflash_nodes)

    dflash_used = {inp for n in dflash_model.graph.node for inp in n.input if inp}
    dflash_inits = [
        t
        for t in dflash_model.graph.initializer
        if t.name in dflash_used and t.name not in ("model.embed_tokens.weight", "lm_head.gguf_weight")
    ]
    dflash_inits.append(
        make_ext_init(
            "model.embed_tokens.weight",
            emb_tinfo["shard_file"],
            emb_tinfo["data_offset"],
            emb_tinfo["n_bytes"],
            TensorProto.UINT8,
            emb_tinfo["onnx_u8_shape"],
        )
    )
    dflash_inits.append(
        make_ext_init(
            "lm_head.gguf_weight",
            lm_tinfo["shard_file"],
            lm_tinfo["data_offset"],
            lm_tinfo["n_bytes"],
            TensorProto.UINT8,
            lm_tinfo["onnx_u8_shape"],
        )
    )
    del dflash_model.graph.initializer[:]
    dflash_model.graph.initializer.extend(dflash_inits)
    onnx.save(dflash_model, str(out_dir / "dflash2.onnx"))

    # 7. Write weights_pool.txt (shares RoPE caches, indexer caches, token_embd.weight, and lm_head.gguf_weight on GPU!)
    pool_lines = [
        "qwen38_weights_pool 1",
        f"pool 0 16 {cos_info[1]} {cos_info[2]} 2 262144 32 {cos_info[0]}",
        f"pool 1 16 {sin_info[1]} {sin_info[2]} 2 262144 32 {sin_info[0]}",
        "pool 2 16 0 33554432 2 262144 64 attn_consts.data",
        "pool 3 16 33554432 33554432 2 262144 64 attn_consts.data",
        f"pool 4 2 {emb_tinfo['data_offset']} {emb_tinfo['n_bytes']} 2 {emb_tinfo['onnx_u8_shape'][0]} {emb_tinfo['onnx_u8_shape'][1]} {emb_tinfo['shard_file']}",
        f"pool 5 2 {lm_tinfo['data_offset']} {lm_tinfo['n_bytes']} 2 {lm_tinfo['onnx_u8_shape'][0]} {lm_tinfo['onnx_u8_shape'][1]} {lm_tinfo['shard_file']}",
        "session target model.rotary_emb.cos_cache 0",
        "session target model.rotary_emb.sin_cache 1",
        "session target capture.indexer_cos_cache.folded 2",
        "session target capture.indexer_sin_cache.folded 3",
        "session target model.embed_tokens.weight 4",
        "session dflash2 model.embed_tokens.weight 4",
        "session target lm_head.gguf_weight 5",
        "session dflash2 lm_head.gguf_weight 5",
    ]
    (out_dir / "weights_pool.txt").write_text("\n".join(pool_lines) + "\n")

    # 8. Write genai_config.json with RC5 protocol boundary tokens, dynamic_batching, and 32K/64K/128K/256K runtime_profiles
    base_cfg_path = base_pkg / "genai_config.json"
    config = json.loads(base_cfg_path.read_text()) if base_cfg_path.exists() else default_base_genai_config()
    configure_gguf_genai_config(config, total_weight_bytes=total_weight_bytes, include_runtime_profiles=True)
    (out_dir / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")

    # 9. Write package_manifest.json
    manifest = {
        "variant": variant_name,
        "source_gguf_dir": str(gguf_dir),
        "gguf_shards": shard_files,
        "engram_gguf": engram_dst_name,
        "ops": {
            "MoEPacked": moe_packed_count,
            "MatMulPacked": matmul_packed_count,
            "GatherPacked": 1,
        },
        "vram_and_blocks": vram_info,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{variant_name}] Built package at {out_dir} in {manifest['elapsed_seconds']}s")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf-root", type=Path, default=Path("~/models/qwen_flash_next_ggufs"))
    ap.add_argument(
        "--base-pkg",
        type=Path,
        default=Path("~/models/qwen_flash_next_int4_int8_rc5_contrib_dflash2"),
    )
    ap.add_argument("--out-root", type=Path, default=Path("~/models"))
    ap.add_argument("--variants", default="UD-IQ1_S,UD-IQ2_XS,UD-IQ3_XXS")
    args = ap.parse_args()

    engram_gguf = args.gguf_root / "engram/Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file"
    assert engram_gguf.exists(), f"Missing {engram_gguf}"

    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        v_dir = args.gguf_root / v
        assert v_dir.is_dir(), f"Missing GGUF variant dir {v_dir}"
        slug = v.lower().replace("-", "_")
        pkg_name = f"qwen_flash_next_gguf_{slug}"
        out_dir = args.out_root / pkg_name
        build_gguf_package(pkg_name, v_dir, engram_gguf, args.base_pkg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
