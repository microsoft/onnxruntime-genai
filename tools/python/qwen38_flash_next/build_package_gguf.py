#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Export Unsloth Qwen3.8-Flash-Next GGUF models (UD-IQ1_S, UD-IQ2_XS, UD-IQ3_XXS)
and the corresponding IQ4_NL or Q8_0 engram GGUF to ONNX Runtime GenAI
model-data packages (`model.type = "qwen38_flash_graph"`):

  - Routed MoE experts (48 layers x 3 tensors: gate, up, down):
      `com.microsoft::MoEPacked` pointing directly to the `.gguf` shard byte offsets
  - Dense projections & LM Head (98 nodes in model.onnx + lm_head in dflash2.onnx):
      `com.microsoft::MatMulPacked` pointing directly to the `.gguf` shard byte offsets
  - Token embedding (`token_embd.weight` in model.onnx, dflash2.onnx, embedding.onnx):
      `com.microsoft::GatherPacked` pointing directly to the `.gguf` shard byte offset
  - N-gram PLE table (`per_layer_token_embd.weight` in engram GGUF):
      `ple_table.txt` (`format iq4_nl` or `format q8_0`) pointing directly to the engram `.gguf`
  - Small weights (HyperConnectionLowRankMix, RouterGateUpProj, shared expert down_proj,
    CausalConvWithState, GatedDeltaNet A_log/dt_bias, RMSNorms):
      Converted from the GGUF tensors (or lossless HF BF16) to `gguf_small_weights.data` &
      inline initializers, with indexer caches in an output-owned `attn_consts.data`.

Package contents (model assets only):
  - ONNX graphs: `model.onnx` (plus optional `model_int8kv.onnx`), `dflash2.onnx`, `embedding.onnx`
  - Weight & cache blobs: linked/copied `.gguf` model and engram shards, `gguf_small_weights.data`,
    `attn_consts.data` (or consolidated `model.onnx.data`), `dflash2.onnx.data`, `expert-profile.bin`
  - Manifests & tokenizer metadata: `genai_config.json`, `package_manifest.json`, `ple_table.txt`,
    `weights_pool.txt`, `chat_template.jinja`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`

Runtime distribution libraries (`libonnxruntime*`, `libonnxruntime_providers_*`) and standalone plane
binaries (`libqwen38_plane*`) are excluded from generated model packages. Install compatible runtime
packages or build from source separately: register the CUDA EP plugin via
`og.register_execution_provider_library("CUDAExecutionProvider", ...)` (`--provider` or `$QWEN_PROVIDER`),
and `model.qwen38_graph.plane_library` defaults to `""` so `qwen38_flash_graph` resolves the CUDA data
plane compiled into `onnxruntime-genai-cuda`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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
    link_file,
    make_ext_init,
)
from package_config import migrate_qwen38_graph_config  # noqa: E402
from package_io import (  # noqa: E402
    GGML_TYPE_INFO,
    GGUF_MIN_NUM_BLOCKS_32K,
    GGUF_MIN_NUM_BLOCKS_64K,
    GGUF_MIN_NUM_BLOCKS_128K,
    GGUF_MIN_NUM_BLOCKS_256K,
    GGUF_NUM_BLOCKS_64K,
    CheckpointTensorLoader,
    compute_gguf_model_weight_stats,
    default_gguf_runtime_profiles,
    estimate_gguf_peak_vram_and_thresholds,
    format_gguf_ple_table_lines,
    parse_gguf_header_stream,
    read_ple_table_spec,
    validate_engram_gguf,
)
from package_paths import reject_output_overlap  # noqa: E402
from precision_options import (  # noqa: E402
    add_precision_cli_args,
    apply_precision_config_to_package,
    precision_config_needs_hf_checkpoint,
    resolve_precision_config,
    validate_gguf_precision_config,
)

MIN_NUM_BLOCKS_32K = GGUF_MIN_NUM_BLOCKS_32K
MIN_NUM_BLOCKS_64K = GGUF_MIN_NUM_BLOCKS_64K
MIN_NUM_BLOCKS_128K = GGUF_MIN_NUM_BLOCKS_128K
MIN_NUM_BLOCKS_256K = GGUF_MIN_NUM_BLOCKS_256K
NUM_BLOCKS_64K = GGUF_NUM_BLOCKS_64K

__all__ = [
    "BASE_PACKAGE_MODEL_ASSETS",
    "DFLASH2_RESERVED_BLOCKS",
    "GGML_TYPE_INFO",
    "MIN_NUM_BLOCKS_32K",
    "MIN_NUM_BLOCKS_64K",
    "MIN_NUM_BLOCKS_128K",
    "MIN_NUM_BLOCKS_256K",
    "NUM_BLOCKS_32K",
    "NUM_BLOCKS_64K",
    "NUM_BLOCKS_128K",
    "NUM_BLOCKS_256K",
    "SmallWeightBlobWriter",
    "build_gguf_package",
    "configure_gguf_genai_config",
    "default_base_genai_config",
    "default_gguf_runtime_profiles",
    "dequant_gguf_tensor_f32",
    "estimate_gguf_peak_vram_and_thresholds",
    "f32_to_bf16_u16",
    "main",
    "parse_gguf_header",
]

BASE_PACKAGE_MODEL_ASSETS: tuple[str, ...] = (
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "expert-profile.bin",
    "dflash2.onnx.data",
)


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(arr), np.uint16(0x7FC0), u16)


def parse_gguf_header(path: Path) -> tuple[dict, list[dict], int]:
    """Stream-parse a GGUF header of arbitrary metadata size and validate all tensor byte ranges."""
    return parse_gguf_header_stream(Path(path).expanduser(), validate_bounds=True)


def dequant_gguf_tensor_f32(shard_path: Path, tinfo: dict) -> np.ndarray:
    dims = tinfo["dims"]
    c_shape = tuple(int(d) for d in reversed(dims))
    rows = int(np.prod(c_shape[:-1])) if len(c_shape) > 1 else 1
    cols = int(c_shape[-1])
    with open(shard_path, "rb") as f:
        f.seek(tinfo["data_offset"])
        raw = f.read(tinfo["n_bytes"])
    if len(raw) != tinfo["n_bytes"]:
        raise ValueError(
            f"Truncated tensor payload for {tinfo['name']} in {shard_path}: "
            f"expected {tinfo['n_bytes']} bytes, got {len(raw)}"
        )
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


def configure_gguf_genai_config(
    config: dict,
    *,
    total_weight_bytes: int,
    include_runtime_profiles: bool = True,
    kv_cache_dtype: str = "int8",
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

    prefill_chunk_long = int(RC5_PREFILL_CONFIG["chunk"])
    qwen = model_sec.get("qwen38_graph")
    if isinstance(qwen, dict):
        migrate_qwen38_graph_config(qwen)
        # Generated packages exclude standalone binaries; empty string selects the plane built into onnxruntime-genai-cuda.
        qwen["plane_library"] = ""
        prefill = qwen.setdefault("prefill", {})
        if prefill.get("enabled", True):
            prefill["chunk"] = RC5_PREFILL_CONFIG["chunk"]
            prefill["warmup_lengths"] = list(RC5_PREFILL_CONFIG["warmup_lengths"])
        elif prefill.get("chunk"):
            prefill_chunk_long = int(prefill["chunk"])

    config["engine"] = json.loads(json.dumps(DEFAULT_ENGINE_CONFIG))
    config.setdefault("search", {})["max_length"] = DEFAULT_MAX_CONTEXT_LENGTH
    if include_runtime_profiles:
        dec_fname = model_sec.get("decoder", {}).get("filename", "model.onnx")
        config["runtime_profiles"] = default_gguf_runtime_profiles(
            total_weight_bytes,
            dec_fname,
            kv_cache_dtype=kv_cache_dtype,
            prefill_chunk_32k=4096,
            prefill_chunk_long=prefill_chunk_long,
        )
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


def _read_base_tensor(base_pkg: Path, tensor: TensorProto) -> tuple[bytes, int, list[int]]:
    dtype = int(tensor.data_type)
    dims = [int(d) for d in tensor.dims]
    if tensor.data_location == TensorProto.EXTERNAL:
        ext = {e.key: e.value for e in tensor.external_data}
        loc = ext["location"]
        off = int(ext.get("offset", 0))
        ln = int(ext["length"])
        src_path = base_pkg / loc
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing external data file {src_path} for initializer {tensor.name}")
        with open(src_path, "rb") as f:
            f.seek(off)
            raw = f.read(ln)
        if len(raw) != ln:
            raise ValueError(
                f"Truncated external data for {tensor.name} in {src_path}: expected {ln} bytes at offset {off}, got {len(raw)}"
            )
        return raw, dtype, dims
    raw = bytes(tensor.raw_data) if tensor.raw_data else numpy_helper.to_array(tensor).tobytes()
    return raw, dtype, dims


def _format_pool_entry(pool_id: int, tensor: TensorProto) -> str:
    ext = {e.key: e.value for e in tensor.external_data}
    loc = ext["location"]
    off = int(ext.get("offset", 0))
    ln = int(ext["length"])
    dims = [int(d) for d in tensor.dims]
    dims_str = " ".join(str(d) for d in dims)
    return f"pool {pool_id} {int(tensor.data_type)} {off} {ln} {len(dims)} {dims_str} {loc}"


def build_gguf_package(
    variant_name: str,
    gguf_dir: Path,
    engram_gguf: Path,
    base_pkg: Path,
    out_dir: Path,
    *,
    small_weights_source: str = "gguf",
    hf_dir: Path | None = None,
) -> dict:
    t0 = time.time()
    gguf_dir = Path(gguf_dir).expanduser()
    engram_gguf = Path(engram_gguf).expanduser()
    base_pkg = Path(base_pkg).expanduser()
    out_dir = Path(out_dir).expanduser()
    effective_hf_dir = (
        Path(hf_dir).expanduser() if hf_dir is not None else (Path.home() / "models" / "Qwen3.8-Flash-Next")
    )

    out_dir = reject_output_overlap(
        out_dir,
        {
            "--gguf-root variant": gguf_dir,
            "engram GGUF": engram_gguf,
            "--base-pkg": base_pkg,
            "--hf-dir": effective_hf_dir,
        },
    )
    if small_weights_source not in ("gguf", "hf_bf16"):
        raise ValueError(f"Unsupported small_weights_source={small_weights_source!r}; expected 'gguf' or 'hf_bf16'")
    if not (base_pkg / "model.onnx").is_file():
        raise FileNotFoundError(f"Missing target model.onnx in --base-pkg: {base_pkg / 'model.onnx'}")
    if not (base_pkg / "dflash2.onnx").is_file():
        raise FileNotFoundError(f"Missing dflash2.onnx in --base-pkg: {base_pkg / 'dflash2.onnx'}")

    hf_reader = CheckpointTensorLoader(effective_hf_dir) if small_weights_source == "hf_bf16" else None

    base_model = onnx.load(str(base_pkg / "model.onnx"), load_external_data=False)
    base_props = {p.key: p.value for p in base_model.metadata_props}
    if json.loads(base_props.get("qwen38.position_config", "{}")).get("capacity") is not None:
        raise ValueError("--base-pkg cannot convert an explicit-capacity package")
    base_inits = {t.name: t for t in base_model.graph.initializer}
    for req_cache in (
        "model.rotary_emb.cos_cache",
        "model.rotary_emb.sin_cache",
        "capture.indexer_cos_cache.folded",
        "capture.indexer_sin_cache.folded",
    ):
        if req_cache not in base_inits:
            raise KeyError(f"Missing required initializer {req_cache!r} in {base_pkg / 'model.onnx'}")

    # Validate PLE specification, engram GGUF, and all model GGUF shards before modifying out_dir
    ple_spec = read_ple_table_spec(base_pkg / "ple_table.txt" if (base_pkg / "ple_table.txt").is_file() else None)
    engram_dst_name, ple_format, ple_row_bytes, engram_doff = validate_engram_gguf(engram_gguf, ple_spec)

    parsed_shards: list[tuple[Path, list[dict]]] = []
    for shard_path in sorted(gguf_dir.glob("*.gguf")):
        _, ts, _ = parse_gguf_header(shard_path)
        if not ts or (len(ts) == 1 and ts[0]["name"] == "per_layer_token_embd.weight"):
            continue
        parsed_shards.append((shard_path, ts))
    if not parsed_shards:
        raise ValueError(f"No model GGUF shards found in {gguf_dir}")

    # Pre-read retained RoPE and indexer caches from base_pkg before removing out_dir
    cos_raw, cos_dt, cos_dims = _read_base_tensor(base_pkg, base_inits["model.rotary_emb.cos_cache"])
    sin_raw, sin_dt, sin_dims = _read_base_tensor(base_pkg, base_inits["model.rotary_emb.sin_cache"])
    icos_raw, icos_dt, icos_dims = _read_base_tensor(base_pkg, base_inits["capture.indexer_cos_cache.folded"])
    isin_raw, isin_dt, isin_dims = _read_base_tensor(base_pkg, base_inits["capture.indexer_sin_cache.folded"])

    if out_dir.exists() or out_dir.is_symlink():
        if out_dir.is_symlink() or out_dir.is_file():
            out_dir.unlink()
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. Link (or copy across filesystems) validated model GGUF shards + engram GGUF into out_dir
    gguf_tensors: dict[str, dict] = {}
    shard_files: list[str] = []
    for shard_path, ts in parsed_shards:
        dst_shard = out_dir / shard_path.name
        link_file(shard_path, dst_shard)
        shard_files.append(shard_path.name)
        for t in ts:
            t["shard_file"] = shard_path.name
            t["shard_path"] = shard_path
            gguf_tensors[t["name"]] = t

    link_file(engram_gguf, out_dir / engram_dst_name)
    print(f"[{variant_name}] Indexed {len(gguf_tensors)} tensors across {shard_files} + {engram_dst_name}")

    # 2. Copy static model assets only from base_pkg (runtime distributions and standalone plane binaries are excluded)
    for fname in BASE_PACKAGE_MODEL_ASSETS:
        src = base_pkg / fname
        if src.exists():
            shutil.copy2(src.resolve(), out_dir / fname)

    # 3. Write ple_table.txt pointing directly to engram GGUF. GGUF blocks carry their own fp16 scales, so the table names
    #    its format and has no scale line (a scale of -8.0 or -20.0 alone would read as INT8).
    ple_lines = format_gguf_ple_table_lines(engram_dst_name, ple_format, engram_doff, ple_row_bytes, ple_spec)
    (out_dir / "ple_table.txt").write_text("\n".join(ple_lines) + "\n")

    # 4. Convert small weights from GGUF + copy RoPE/indexer caches into output-owned blobs
    small_writer = SmallWeightBlobWriter(out_dir, "gguf_small_weights.data")
    cos_info = small_writer.write_raw(cos_raw, cos_dt, cos_dims)
    sin_info = small_writer.write_raw(sin_raw, sin_dt, sin_dims)

    attn_writer = SmallWeightBlobWriter(out_dir, "attn_consts.data")
    try:
        icos_info = attn_writer.write_raw(icos_raw, icos_dt, icos_dims)
        isin_info = attn_writer.write_raw(isin_raw, isin_dt, isin_dims)
    finally:
        attn_writer.close()

    num_gdn_v_heads = int(np.prod(gguf_tensors["blk.0.ssm_a"]["dims"]))
    num_gdn_k_heads = num_gdn_v_heads // 3
    true_perm = np.arange(num_gdn_v_heads).reshape(num_gdn_k_heads, 3).T.flatten()
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

    # Register RoPE and indexer caches in output-owned blobs
    new_inits["model.rotary_emb.cos_cache"] = make_ext_init("model.rotary_emb.cos_cache", *cos_info)
    new_inits["model.rotary_emb.sin_cache"] = make_ext_init("model.rotary_emb.sin_cache", *sin_info)
    new_inits["capture.indexer_cos_cache.folded"] = make_ext_init("capture.indexer_cos_cache.folded", *icos_info)
    new_inits["capture.indexer_sin_cache.folded"] = make_ext_init("capture.indexer_sin_cache.folded", *isin_info)

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

    # Convert small external weights per layer from GGUF (or lossless HF BF16 when small_weights_source == "hf_bf16")
    gdn_v_dim = int(gguf_tensors["blk.0.attn_gate.weight"]["dims"][1])
    gdn_qkv_dim = int(gguf_tensors["blk.0.attn_qkv.weight"]["dims"][1])
    gdn_qk_dim = gdn_qkv_dim - gdn_v_dim
    gdn_head_v_dim = gdn_v_dim // num_gdn_v_heads

    for lid in range(48):
        lpfx = f"model.language_model.layers.{lid}"
        # HyperConnection attn & mlp down/up/inject
        for kind_onnx, kind_gguf in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            if hf_reader is not None:
                w_down = (
                    hf_reader.read_bf16_tensor(f"{lpfx}.{kind_onnx}.input_mix_weight_down.weight").float().numpy().T
                )
                w_up = hf_reader.read_bf16_tensor(f"{lpfx}.{kind_onnx}.input_mix_weight_up.weight").float().numpy().T
                w_inj = hf_reader.read_bf16_tensor(f"{lpfx}.{kind_onnx}.block_inject_weight.weight").float().numpy().T
            else:
                ti_down = gguf_tensors[f"blk.{lid}.{kind_gguf}_down.weight"]
                w_down = dequant_gguf_tensor_f32(ti_down["shard_path"], ti_down).T
                ti_up = gguf_tensors[f"blk.{lid}.{kind_gguf}_up.weight"]
                w_up = dequant_gguf_tensor_f32(ti_up["shard_path"], ti_up).T
                ti_inj = gguf_tensors[f"blk.{lid}.{kind_gguf}_inject.weight"]
                w_inj = dequant_gguf_tensor_f32(ti_inj["shard_path"], ti_inj).T

            add_small_bf16_init(
                f"model.layers.{lid}.{kind_onnx}.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16", w_down
            )
            add_small_bf16_init(
                f"model.layers.{lid}.{kind_onnx}.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16", w_up
            )
            add_small_bf16_init(f"model.layers.{lid}.{kind_onnx}.block_inject_weight.weight_t", w_inj)

        # RouterGateUpProj merged weight [hidden, router + shared_gate + shared_up]
        if hf_reader is not None:
            w_r = hf_reader.read_bf16_tensor(f"{lpfx}.mlp.gate.weight").float().numpy().T
            w_sg = hf_reader.read_bf16_tensor(f"{lpfx}.mlp.shared_expert.gate_proj.weight").float().numpy().T
            w_su = hf_reader.read_bf16_tensor(f"{lpfx}.mlp.shared_expert.up_proj.weight").float().numpy().T
            w_sd = hf_reader.read_bf16_tensor(f"{lpfx}.mlp.shared_expert.down_proj.weight").float().numpy().T
            w_sgate = hf_reader.read_bf16_tensor(f"{lpfx}.mlp.shared_expert_gate.weight").float().numpy().reshape(-1, 1)
        else:
            ti_r = gguf_tensors[f"blk.{lid}.ffn_gate_inp.weight"]
            ti_sg = gguf_tensors[f"blk.{lid}.ffn_gate_shexp.weight"]
            ti_su = gguf_tensors[f"blk.{lid}.ffn_up_shexp.weight"]
            w_r = dequant_gguf_tensor_f32(ti_r["shard_path"], ti_r).T
            w_sg = dequant_gguf_tensor_f32(ti_sg["shard_path"], ti_sg).T
            w_su = dequant_gguf_tensor_f32(ti_su["shard_path"], ti_su).T
            ti_sd = gguf_tensors[f"blk.{lid}.ffn_down_shexp.weight"]
            w_sd = dequant_gguf_tensor_f32(ti_sd["shard_path"], ti_sd).T
            ti_sgate = gguf_tensors[f"blk.{lid}.ffn_gate_inp_shexp.weight"]
            w_sgate = dequant_gguf_tensor_f32(ti_sgate["shard_path"], ti_sgate).reshape(-1, 1)

        w_rgup = np.concatenate([w_r, w_sg, w_su], axis=1)
        rgup_idx = lid * 2 + 1 if lid == 0 else lid * 2 + 2
        add_small_bf16_init(f"merged_proj.{rgup_idx}.weight_t", w_rgup)
        add_small_bf16_init(f"model.layers.{lid}.mlp.shared_expert.down_proj.weight_t", w_sd)
        add_small_bf16_init(f"model.layers.{lid}.mlp.shared_expert_gate.weight_t", w_sgate)

        # GDN conv1d, norm, alpha, beta (for GDN layers lid % 4 != 3) or QSA indexer_qk (for lid % 4 == 3)
        if lid % 4 != 3:
            ti_conv = gguf_tensors[f"blk.{lid}.ssm_conv1d.weight"]
            raw_conv = dequant_gguf_tensor_f32(ti_conv["shard_path"], ti_conv)
            conv_ch = raw_conv.size // 4
            w_conv = raw_conv.reshape(conv_ch, 4).copy()
            w_conv[gdn_qk_dim:] = (
                w_conv[gdn_qk_dim:].reshape(num_gdn_v_heads, gdn_head_v_dim, 4)[inv_perm].reshape(gdn_v_dim, 4)
            )
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.conv1d.weight", w_conv.reshape(conv_ch, 1, 4))

            ti_snorm = gguf_tensors[f"blk.{lid}.ssm_norm.weight"]
            w_snorm = dequant_gguf_tensor_f32(ti_snorm["shard_path"], ti_snorm).flatten()
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.norm.weight", w_snorm)

            ti_alpha = gguf_tensors[f"blk.{lid}.ssm_alpha.weight"]
            w_alpha = dequant_gguf_tensor_f32(ti_alpha["shard_path"], ti_alpha)[inv_perm].T
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.alpha.weight_t", w_alpha)

            ti_beta = gguf_tensors[f"blk.{lid}.ssm_beta.weight"]
            w_beta = dequant_gguf_tensor_f32(ti_beta["shard_path"], ti_beta)[inv_perm].T
            add_small_bf16_init(f"model.layers.{lid}.linear_attn.beta.weight_t", w_beta)
        else:
            ti_iq = gguf_tensors[f"blk.{lid}.indexer.q_proj.weight"]
            ti_ik = gguf_tensors[f"blk.{lid}.indexer.k_proj.weight"]
            w_iq = dequant_gguf_tensor_f32(ti_iq["shard_path"], ti_iq).T
            w_ik = dequant_gguf_tensor_f32(ti_ik["shard_path"], ti_ik).T
            add_small_bf16_init(
                f"model.layers.{lid}.self_attn.indexer_qk.weight_t", np.concatenate([w_iq, w_ik], axis=1)
            )

    # Layer 1 PLE conv1d & output HC down/up
    ti_ple_conv = gguf_tensors["blk.1.ple_conv1d.weight"]
    raw_ple_conv = dequant_gguf_tensor_f32(ti_ple_conv["shard_path"], ti_ple_conv)
    w_ple_conv = raw_ple_conv.reshape(raw_ple_conv.size // 4, 1, 4)
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
    emb_k = int(emb_tinfo["dims"][0])

    new_nodes: list[onnx.NodeProto] = []
    # Prepend GatherPacked node so model.onnx computes inputs_embeds on GPU directly from token_embd.weight
    gather_node = helper.make_node(
        "GatherPacked",
        ["model.embed_tokens.weight", "input_ids"],
        ["inputs_embeds"],
        name="model/embed_tokens/GatherPacked",
        domain="com.microsoft",
        K=emb_k,
        packed_type=int(emb_tinfo["ggml_type"]),
        output_dtype=int(TensorProto.BFLOAT16),
    )
    new_nodes.append(gather_node)

    # Helper shape constants for GDN [3, num_k_heads, head_v_dim] <-> [num_k_heads, 3, head_v_dim] V-head permutation
    new_inits["const_gdn_split_4096_6144"] = numpy_helper.from_array(
        np.array([gdn_qk_dim, gdn_v_dim], dtype=np.int64), name="const_gdn_split_4096_6144"
    )
    new_inits["const_gdn_v_3_16_128"] = numpy_helper.from_array(
        np.array([0, 0, 3, num_gdn_k_heads, gdn_head_v_dim], dtype=np.int64), name="const_gdn_v_3_16_128"
    )
    new_inits["const_gdn_v_6144"] = numpy_helper.from_array(
        np.array([0, 0, gdn_v_dim], dtype=np.int64), name="const_gdn_v_6144"
    )
    new_inits["const_gdn_out_16_3_128"] = numpy_helper.from_array(
        np.array([-1, num_gdn_k_heads, 3, gdn_head_v_dim], dtype=np.int64), name="const_gdn_out_16_3_128"
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
            moe_k = int(next((helper.get_attribute_value(a) for a in n.attribute if a.name == "k"), 10))
            moe_node = helper.make_node(
                "MoEPacked",
                [n.input[0], n.input[1], gw_name, uw_name, dw_name],
                list(n.output),
                name=n.name.replace("QMoE", "MoEPacked"),
                domain="com.microsoft",
                k=moe_k,
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
                        K=int(b_info["dims"][0]),
                        N=int(b_info["dims"][1]),
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
                        K=int(b_info["dims"][0]),
                        N=int(b_info["dims"][1]),
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
                        K=int(b_info["dims"][0]),
                        N=int(b_info["dims"][1]),
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
                        K=int(b0_info["dims"][0]),
                        N=int(b0_info["dims"][1]),
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
                        K=int(b1_info["dims"][0]),
                        N=int(b1_info["dims"][1]),
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
                            K=int(qkv_info["dims"][0]),
                            N=int(qkv_info["dims"][1]),
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
                            K=int(gate_info["dims"][0]),
                            N=int(gate_info["dims"][1]),
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
                    # QSA input projection: indexer_qk, attn_q, attn_k, attn_v
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
                            K=int(qw_info["dims"][0]),
                            N=int(qw_info["dims"][1]),
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
                            K=int(kw_info["dims"][0]),
                            N=int(kw_info["dims"][1]),
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
                            K=int(vw_info["dims"][0]),
                            N=int(vw_info["dims"][1]),
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
    ensure_unified_dynamic_cache_model(base_model, 262152, 8259)

    onnx.save(base_model, str(out_dir / "model.onnx"))

    # Compute total GPU weight bytes (MoEPacked + MatMulPacked + GatherPacked + gguf_small_weights.data)
    weight_stats = compute_gguf_model_weight_stats(base_model)
    moe_weight_bytes = weight_stats["moe_bytes"]
    gguf_direct_bytes = weight_stats["gguf_bytes"]
    small_weight_bytes = (out_dir / "gguf_small_weights.data").stat().st_size
    total_weight_bytes = gguf_direct_bytes + small_weight_bytes

    kv_vi = next((vi for vi in base_model.graph.input if re.match(r"^past_key_values\.\d+\.key$", vi.name)), None)
    kv_elem = kv_vi.type.tensor_type.elem_type if kv_vi is not None else TensorProto.INT8
    base_kv_dtype = (
        "fp16" if kv_elem == TensorProto.FLOAT16 else ("bf16" if kv_elem == TensorProto.BFLOAT16 else "int8")
    )
    vram_info = estimate_gguf_peak_vram_and_thresholds(
        total_weight_bytes,
        moe_bytes=moe_weight_bytes,
        kv_cache_dtype=base_kv_dtype,
    )
    vram_info["baseline_small_file_bytes"] = small_weight_bytes
    vram_info["baseline_non_gguf_tensor_bytes"] = weight_stats["non_gguf_bytes"]

    # Also build standalone embedding.onnx using GatherPacked on CPU EP
    emb_graph = helper.make_graph(
        [
            helper.make_node(
                "GatherPacked",
                ["model.embed_tokens.weight", "input_ids"],
                ["inputs_embeds"],
                name="/model/embed_tokens/GatherPacked",
                domain="com.microsoft",
                K=emb_k,
                packed_type=int(emb_tinfo["ggml_type"]),
                output_dtype=int(TensorProto.BFLOAT16),
            )
        ],
        "embedding_gguf",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.BFLOAT16, ["num_tokens", emb_k])],
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
                    K=emb_k,
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
                    K=int(lm_tinfo["dims"][0]),
                    N=int(lm_tinfo["dims"][1]),
                    packed_type=int(lm_tinfo["ggml_type"]),
                )
            )
        else:
            dflash_nodes.append(n)

    del dflash_model.graph.node[:]
    dflash_model.graph.node.extend(dflash_nodes)

    dflash_used = {inp for n in dflash_model.graph.node for inp in n.input if inp}
    dflash_inits: list[TensorProto] = []
    dflash_blob_path = out_dir / "dflash2.onnx.data"
    for t in dflash_model.graph.initializer:
        if t.name not in dflash_used or t.name in ("model.embed_tokens.weight", "lm_head.gguf_weight"):
            continue
        kept_t = t
        if kept_t.data_location == TensorProto.EXTERNAL:
            ext = {e.key: e.value for e in kept_t.external_data}
            loc = ext.get("location", "")
            off = int(ext.get("offset", 0))
            ln = int(ext["length"])
            if (
                loc != "dflash2.onnx.data"
                or not dflash_blob_path.is_file()
                or off + ln > dflash_blob_path.stat().st_size
            ):
                raw_df, dt_df, dims_df = _read_base_tensor(base_pkg, kept_t)
                cur_sz = dflash_blob_path.stat().st_size if dflash_blob_path.is_file() else 0
                new_off = align_up(cur_sz)
                with open(dflash_blob_path, "ab") as df_fh:
                    if new_off > cur_sz:
                        df_fh.write(b"\x00" * (new_off - cur_sz))
                    df_fh.write(raw_df)
                kept_t = make_ext_init(kept_t.name, "dflash2.onnx.data", new_off, len(raw_df), dt_df, dims_df)
        dflash_inits.append(kept_t)

    dflash_emb_init = make_ext_init(
        "model.embed_tokens.weight",
        emb_tinfo["shard_file"],
        emb_tinfo["data_offset"],
        emb_tinfo["n_bytes"],
        TensorProto.UINT8,
        emb_tinfo["onnx_u8_shape"],
    )
    dflash_lm_init = make_ext_init(
        "lm_head.gguf_weight",
        lm_tinfo["shard_file"],
        lm_tinfo["data_offset"],
        lm_tinfo["n_bytes"],
        TensorProto.UINT8,
        lm_tinfo["onnx_u8_shape"],
    )
    dflash_inits.append(dflash_emb_init)
    dflash_inits.append(dflash_lm_init)
    del dflash_model.graph.initializer[:]
    dflash_model.graph.initializer.extend(dflash_inits)
    onnx.save(dflash_model, str(out_dir / "dflash2.onnx"))

    # 7. Write weights_pool.txt generated from actual tensor metadata
    pool_lines = [
        "qwen38_weights_pool 1",
        _format_pool_entry(0, new_inits["model.rotary_emb.cos_cache"]),
        _format_pool_entry(1, new_inits["model.rotary_emb.sin_cache"]),
        _format_pool_entry(2, new_inits["capture.indexer_cos_cache.folded"]),
        _format_pool_entry(3, new_inits["capture.indexer_sin_cache.folded"]),
        _format_pool_entry(4, new_inits.get("model.embed_tokens.weight", dflash_emb_init)),
        _format_pool_entry(5, new_inits.get("lm_head.gguf_weight", dflash_lm_init)),
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
    configure_gguf_genai_config(
        config,
        total_weight_bytes=total_weight_bytes,
        include_runtime_profiles=True,
        kv_cache_dtype=base_kv_dtype,
    )
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


def build_arg_parser() -> argparse.ArgumentParser:
    home_models = Path.home() / "models"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf-root", type=Path, default=home_models / "qwen_flash_next_ggufs")
    ap.add_argument(
        "--base-pkg",
        type=Path,
        default=home_models / "qwen_flash_next_int4_int8_rc5_contrib_dflash2",
    )
    ap.add_argument("--hf-dir", type=Path, default=home_models / "Qwen3.8-Flash-Next")
    ap.add_argument("--out-root", type=Path, default=home_models)
    ap.add_argument("--variants", default="UD-IQ1_S,UD-IQ2_XS,UD-IQ3_XXS")
    ap.add_argument("--engram-gguf", type=Path, default=None, help="Explicit path to engram .gguf_file")
    ap.add_argument("--pkg-suffix", default="", help="Optional suffix for output package directory name")
    add_precision_cli_args(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    gguf_root = Path(args.gguf_root).expanduser()
    base_pkg = Path(args.base_pkg).expanduser()
    hf_dir = Path(args.hf_dir).expanduser()
    out_root = Path(args.out_root).expanduser()
    explicit_engram = Path(args.engram_gguf).expanduser() if args.engram_gguf is not None else None
    args.engram_gguf = explicit_engram

    pcfg = resolve_precision_config(args)
    validate_gguf_precision_config(pcfg)

    if explicit_engram is not None:
        engram_gguf = explicit_engram
    elif pcfg.ple_engram == "q8_0":
        engram_gguf = gguf_root / "engram" / "Qwen3.8-Flash-Next-engram-Q8_0.gguf_file"
    else:
        engram_gguf = gguf_root / "engram" / "Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file"
    pcfg.engram_gguf_path = engram_gguf
    if not engram_gguf.exists():
        raise FileNotFoundError(f"Missing engram GGUF: {engram_gguf}")

    eff_small_src = "hf_bf16" if pcfg.small_weights_source == "hf_bf16" else "gguf"
    needs_hf = eff_small_src == "hf_bf16" or (pcfg.is_non_default() and precision_config_needs_hf_checkpoint(pcfg))

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    variant_jobs: list[tuple[str, Path, Path]] = []
    all_variant_dirs = {f"--gguf-root variant {v}": gguf_root / v for v in variants}
    for v in variants:
        v_dir = gguf_root / v
        if not v_dir.is_dir():
            raise FileNotFoundError(f"Missing GGUF variant dir {v_dir}")
        slug = v.lower().replace("-", "_")
        pkg_name = f"qwen_flash_next_gguf_{slug}{args.pkg_suffix}"
        out_dir = out_root / pkg_name
        reject_output_overlap(
            out_dir,
            {
                "--gguf-root": gguf_root,
                **all_variant_dirs,
                "engram GGUF": engram_gguf,
                "--base-pkg": base_pkg,
                "--hf-dir": hf_dir,
            },
        )
        variant_jobs.append((pkg_name, v_dir, out_dir))

    if needs_hf:
        CheckpointTensorLoader(hf_dir)

    default_hf_dir = Path.home() / "models" / "Qwen3.8-Flash-Next"
    for pkg_name, v_dir, out_dir in variant_jobs:
        if eff_small_src == "gguf" and hf_dir == default_hf_dir:
            build_gguf_package(
                pkg_name,
                v_dir,
                engram_gguf,
                base_pkg,
                out_dir,
            )
        else:
            build_gguf_package(
                pkg_name,
                v_dir,
                engram_gguf,
                base_pkg,
                out_dir,
                small_weights_source=eff_small_src,
                hf_dir=hf_dir,
            )
        if pcfg.is_non_default():
            apply_precision_config_to_package(out_dir, pcfg, hf_dir=hf_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
