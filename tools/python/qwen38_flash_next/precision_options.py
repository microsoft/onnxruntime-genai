#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Flexible component-level precision configuration and post-build ONNX/package transformer
for Qwen3.8-Flash-Next RC5 packages built by `build_package.py`.

Supports independent control over:
  - `--precision-profile {default,fp32,fp32_fp16kv}`: preset combinations
  - `--kv-cache-dtype {int8,bf16,fp16}`: QSA full-attention KV cache element type (12 layers)
  - `--rotary-cache-dtype {bf16,fp32}`: QSA `model.rotary_emb.{cos,sin}_cache` element type (`[262144, 32]`)
  - `--indexer-dtype {default,int8,bf16,fp32}`: QSA Lightning Indexer projection, norms, RoPE caches & buffers
  - `--gdn-gate-dtype {default,int8,bf16,fp32}`: GatedDeltaNet `alpha`/`beta` (`in_proj_a`/`in_proj_b`) projections & `TG` inputs
  - `--gdn-precision {bf16,fp32}`: `"ep.cuda.gated_delta_net_bf16_precision"` session option
  - `--ple-engram {default,int8,iq4_nl,q8_0}`: Per-Layer Embedding (PLE) engram table quantization
  - `--ple-norm-dtype {bf16,fp32}`: Layer 1 PLE `BranchwiseRMSNorm` scale precision (`M = BFloat16` vs `float`)
  - `--small-weights-source {default,gguf,hf_bf16}`: GGUF small weights source (`hc_*`, `shared_expert.*`)

`--qsa-qkv-dtype` other than `default` requires `--indexer-dtype fp32`: the FP16 QSA projections replace the fused INT8
`SplitQsa` projection, which also produces the indexer projection.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from package_io import (
    CONTEXT_TIERS,
    AlignedBlobWriter,
    HFCheckpointReader,
    build_context_tier_model,
    ensure_unified_dynamic_cache_model,
    make_ext_init,
    read_gguf_tensor_infos,
    save_model_replacing,
    set_inline_f32,
    write_text_replacing,
)

DEFAULT_HF_DIR = Path("/data/tlwu/Qwen3.8-Flash-Next")
DEFAULT_ENGRAM_Q8_GGUF = Path("/data/tlwu/qwen_flash_next_ggufs/engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file")
DEFAULT_ENGRAM_IQ4_GGUF = Path("/data/tlwu/qwen_flash_next_ggufs/engram/Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file")


@dataclasses.dataclass
class PrecisionConfig:
    profile: str = "default"
    kv_cache_dtype: str = "int8"  # int8 | bf16 | fp16
    rotary_cache_dtype: str = "bf16"  # bf16 | fp32
    qsa_qkv_dtype: str = "default"  # default | v_rc5_fp16 | kv_fp16 | fp16
    indexer_dtype: str = "default"  # default | int8 | bf16 | fp32
    gdn_gate_dtype: str = "default"  # default | int8 | bf16 | fp32
    gdn_precision: str = "bf16"  # bf16 | fp32
    ple_engram: str = "default"  # default | int8 | iq4_nl | q8_0
    ple_norm_dtype: str = "bf16"  # bf16 | fp32
    small_weights_source: str = "default"  # default | gguf | hf_bf16
    # emit model_int8kv.onnx alongside model.onnx when kv_cache_dtype in ("bf16", "fp16")
    emit_int8kv_variant: bool = False
    consolidate_data_files: bool = False  # consolidate target .data files into a single model.onnx.data
    engram_gguf_path: Path | None = None

    def is_non_default(self) -> bool:
        return (
            self.profile != "default"
            or self.kv_cache_dtype != "int8"
            or self.rotary_cache_dtype != "bf16"
            or self.qsa_qkv_dtype != "default"
            or self.indexer_dtype not in ("default", "int8", "bf16")
            or self.gdn_gate_dtype not in ("default", "int8", "bf16")
            or self.gdn_precision != "bf16"
            or self.ple_engram not in ("default",)
            or self.ple_norm_dtype != "bf16"
            or self.small_weights_source not in ("default", "gguf")
            or self.emit_int8kv_variant
            or self.consolidate_data_files
        )

    def validate(self) -> None:
        if self.qsa_qkv_dtype != "default" and self.indexer_dtype != "fp32":
            raise ValueError(
                f"--qsa-qkv-dtype {self.qsa_qkv_dtype} requires --indexer-dtype fp32 (got {self.indexer_dtype}): "
                "the FP16 QSA projections replace the fused INT8 SplitQsa projection that also produces the indexer projection"
            )

    def to_dict(self) -> dict[str, object]:
        d = dataclasses.asdict(self)
        if d["engram_gguf_path"] is not None:
            d["engram_gguf_path"] = str(d["engram_gguf_path"])
        return d


def add_precision_cli_args(ap: argparse.ArgumentParser) -> None:
    grp = ap.add_argument_group("Component Precision & Long-Context Options")
    grp.add_argument(
        "--precision-profile",
        choices=("default", "fp32", "fp32_fp16kv", "fp32_fp16kv_v2", "fp32_int8kv_v2"),
        default="default",
        help=(
            "Preset precision profile: 'default' (standard INT8 KV / BF16 kernels), "
            "'fp32' (BF16 KV + FP32 Indexer + FP32 GDN gates/state + Q8_0 Engram + FP32 PLE norm + HF BF16 small weights), "
            "'fp32_fp16kv' (same as 'fp32' with FP16 KV cache + FP32 QSA rotary caches and FP16 RMSNormRotary Q/K outputs), "
            "'fp32_fp16kv_v2' (RC5-calibrated FP16 V + FP16 KV cache + FP32 QSA rotary caches + FP32 Indexer + FP32 GDN gates/state + native INT8 PLE), "
            "or 'fp32_int8kv_v2' (same as 'fp32_fp16kv_v2' with INT8 KV cache for VRAM-limited offload). "
            "Any individual component flag below overrides the profile."
        ),
    )
    grp.add_argument(
        "--kv-cache-dtype",
        choices=("int8", "bf16", "fp16"),
        default=None,
        help="QSA full-attention KV cache dtype across the 12 QSA layers (int8, bf16, or fp16)",
    )
    grp.add_argument(
        "--rotary-cache-dtype",
        choices=("bf16", "fp32"),
        default=None,
        help="Element type for QSA model.rotary_emb.{cos,sin}_cache [262144, 32] (bf16 or fp32)",
    )
    grp.add_argument(
        "--qsa-qkv-dtype",
        choices=("default", "v_rc5_fp16", "kv_fp16", "fp16"),
        default=None,
        help="QSA full-attention projection precision across the 12 QSA layers (default, v_rc5_fp16 for lossless RC5-calibrated FP16 V, kv_fp16 for FP16 K/V, or fp16 for FP16 Q/K/V/O)",
    )
    grp.add_argument(
        "--indexer-dtype",
        choices=("default", "int8", "bf16", "fp32"),
        default=None,
        help="QSA Lightning Indexer precision (default/int8/bf16 or fp32 for 100%% FP32 Indexer)",
    )
    grp.add_argument(
        "--gdn-gate-dtype",
        choices=("default", "int8", "bf16", "fp32"),
        default=None,
        help="GatedDeltaNet alpha/beta gate projection & TG input precision (default/int8/bf16 or fp32)",
    )
    grp.add_argument(
        "--gdn-precision",
        choices=("bf16", "fp32"),
        default=None,
        help="ep.cuda.gated_delta_net_bf16_precision session option (bf16 or fp32)",
    )
    grp.add_argument(
        "--ple-engram",
        choices=("default", "int8", "iq4_nl", "q8_0"),
        default=None,
        help="Per-Layer Embedding (PLE) engram table format (default, int8, iq4_nl, or q8_0)",
    )
    grp.add_argument(
        "--ple-norm-dtype",
        choices=("bf16", "fp32"),
        default=None,
        help="Layer 1 PLE BranchwiseRMSNorm scale dtype (bf16 or fp32)",
    )
    grp.add_argument(
        "--small-weights-source",
        choices=("default", "gguf", "hf_bf16"),
        default=None,
        help="For GGUF builds: source for small weights hc_* and shared_expert.* (gguf or hf_bf16)",
    )
    grp.add_argument(
        "--emit-int8kv-variant",
        dest="emit_int8kv_variant",
        action="store_true",
        default=None,
        help="When building an FP16/BF16 KV cache package, also emit model_int8kv.onnx sharing all weights for VRAM-limited offload",
    )
    grp.add_argument(
        "--no-emit-int8kv-variant",
        dest="emit_int8kv_variant",
        action="store_false",
        default=None,
        help="Disable emitting model_int8kv.onnx alongside model.onnx",
    )
    grp.add_argument(
        "--consolidate-data-files",
        dest="consolidate_data_files",
        action="store_true",
        default=None,
        help="Consolidate all target decoder external .data files into a single model.onnx.data file",
    )
    grp.add_argument(
        "--no-consolidate-data-files",
        dest="consolidate_data_files",
        action="store_false",
        default=None,
        help="Do not consolidate target decoder external .data files into model.onnx.data",
    )


def resolve_precision_config(args: argparse.Namespace) -> PrecisionConfig:
    prof = getattr(args, "precision_profile", "default") or "default"
    if prof == "fp32":
        cfg = PrecisionConfig(
            profile="fp32",
            kv_cache_dtype="bf16",
            rotary_cache_dtype="bf16",
            indexer_dtype="fp32",
            gdn_gate_dtype="fp32",
            gdn_precision="fp32",
            ple_engram="q8_0",
            ple_norm_dtype="fp32",
            small_weights_source="hf_bf16",
        )
    elif prof == "fp32_fp16kv":
        cfg = PrecisionConfig(
            profile="fp32_fp16kv",
            kv_cache_dtype="fp16",
            rotary_cache_dtype="fp32",
            indexer_dtype="fp32",
            gdn_gate_dtype="fp32",
            gdn_precision="fp32",
            ple_engram="q8_0",
            ple_norm_dtype="fp32",
            small_weights_source="hf_bf16",
        )
    elif prof == "fp32_fp16kv_v2":
        cfg = PrecisionConfig(
            profile="fp32_fp16kv_v2",
            kv_cache_dtype="fp16",
            rotary_cache_dtype="fp32",
            qsa_qkv_dtype="v_rc5_fp16",
            indexer_dtype="fp32",
            gdn_gate_dtype="fp32",
            gdn_precision="fp32",
            ple_engram="default",
            ple_norm_dtype="fp32",
            small_weights_source="hf_bf16",
            emit_int8kv_variant=True,
            consolidate_data_files=True,
        )
    elif prof == "fp32_int8kv_v2":
        cfg = PrecisionConfig(
            profile="fp32_int8kv_v2",
            kv_cache_dtype="int8",
            rotary_cache_dtype="fp32",
            qsa_qkv_dtype="v_rc5_fp16",
            indexer_dtype="fp32",
            gdn_gate_dtype="fp32",
            gdn_precision="fp32",
            ple_engram="default",
            ple_norm_dtype="fp32",
            small_weights_source="hf_bf16",
        )
    else:
        cfg = PrecisionConfig(profile="default")

    if getattr(args, "kv_cache_dtype", None) is not None:
        cfg.kv_cache_dtype = args.kv_cache_dtype
    if getattr(args, "rotary_cache_dtype", None) is not None:
        cfg.rotary_cache_dtype = args.rotary_cache_dtype
    elif cfg.kv_cache_dtype == "fp16" and prof == "default":
        # Default rotary_cache_dtype to fp32 when fp16 KV cache is explicitly requested
        cfg.rotary_cache_dtype = "fp32"
    if getattr(args, "qsa_qkv_dtype", None) is not None:
        cfg.qsa_qkv_dtype = args.qsa_qkv_dtype
    if getattr(args, "indexer_dtype", None) is not None:
        cfg.indexer_dtype = args.indexer_dtype
    if getattr(args, "gdn_gate_dtype", None) is not None:
        cfg.gdn_gate_dtype = args.gdn_gate_dtype
    if getattr(args, "gdn_precision", None) is not None:
        cfg.gdn_precision = args.gdn_precision
    if getattr(args, "ple_engram", None) is not None:
        cfg.ple_engram = args.ple_engram
    if getattr(args, "ple_norm_dtype", None) is not None:
        cfg.ple_norm_dtype = args.ple_norm_dtype
    if getattr(args, "small_weights_source", None) is not None:
        cfg.small_weights_source = args.small_weights_source
    if getattr(args, "emit_int8kv_variant", None) is not None:
        cfg.emit_int8kv_variant = bool(args.emit_int8kv_variant)
    if getattr(args, "consolidate_data_files", None) is not None:
        cfg.consolidate_data_files = bool(args.consolidate_data_files)
    if getattr(args, "engram_gguf", None) is not None:
        cfg.engram_gguf_path = args.engram_gguf
    cfg.validate()
    return cfg


def consolidate_package_external_data(
    pkg_dir: Path,
    target_data_name: str = "model.onnx.data",
) -> dict[str, object]:
    """Consolidate all target decoder external `.data` files into a single `4096`-byte page-aligned `model.onnx.data` file,
    leaving only:
      - `embedding.onnx` + `embedding.onnx.data`
      - `dflash2.onnx` + `dflash2.onnx.data`
      - `model.onnx` (+ `model_int8kv.onnx` and context-tier `model_*.onnx`) + `model.onnx.data`

    The data files referenced by the decoder graphs (`model.onnx` and every `model_*.onnx` variant) are merged, including an
    existing `target_data_name`, whose slices are relocated as well. All other graphs of the package and `weights_pool.txt` are
    rewritten wherever they reference a merged file, so no file that is still referenced is removed.
    """
    model_path = pkg_dir / "model.onnx"
    if not model_path.exists():
        return {"consolidated": False}

    decoder_paths = [model_path, *sorted(pkg_dir.glob("model_*.onnx"))]
    other_paths = sorted(p for p in pkg_dir.glob("*.onnx") if p not in decoder_paths)
    graphs = {p: onnx.load(str(p), load_external_data=False) for p in decoder_paths + other_paths}

    keep_separate = {"embedding.onnx.data", "dflash2.onnx.data"}
    slices_ordered: list[tuple[str, int, int]] = []
    seen_slices: set[tuple[str, int, int]] = set()
    old_files: set[str] = set()

    def add_slice(key: tuple[str, int, int]) -> None:
        if key not in seen_slices:
            seen_slices.add(key)
            slices_ordered.append(key)

    def external_inits(m_proto: onnx.ModelProto):
        for t in m_proto.graph.initializer:
            if t.data_location == TensorProto.EXTERNAL:
                yield t, {kv.key: kv.value for kv in t.external_data}

    for p in decoder_paths:
        for _t, ext in external_inits(graphs[p]):
            loc = ext.get("location", "")
            if not loc or loc in keep_separate or loc.endswith(".gguf") or "/" in loc:
                continue
            old_files.add(loc)
            add_slice((loc, int(ext.get("offset", 0)), int(ext["length"])))
    for p in other_paths:
        for _t, ext in external_inits(graphs[p]):
            loc = ext.get("location", "")
            if loc in old_files:
                add_slice((loc, int(ext.get("offset", 0)), int(ext["length"])))

    pool_path = pkg_dir / "weights_pool.txt"
    pool_lines = pool_path.read_text().splitlines() if pool_path.exists() else []
    for ln in pool_lines:
        parts = ln.split()
        if len(parts) >= 8 and parts[0] == "pool" and parts[-1] in old_files:
            add_slice((parts[-1], int(parts[3]), int(parts[4])))

    if not old_files - {target_data_name}:
        return {"consolidated": False}

    page_align = 4096
    tmp_data_path = pkg_dir / f"{target_data_name}.tmp.{os.getpid()}"
    slice_new_offset: dict[tuple[str, int, int], int] = {}
    cur_off = 0
    try:
        with ExitStack() as stack:
            out_f = stack.enter_context(open(tmp_data_path, "wb"))
            out_fd = out_f.fileno()
            in_files = {}
            for loc, off, length in slices_ordered:
                pad = (page_align - (cur_off % page_align)) % page_align
                if pad > 0:
                    os.write(out_fd, b"\x00" * pad)
                    cur_off += pad
                slice_new_offset[(loc, off, length)] = cur_off
                if loc not in in_files:
                    in_files[loc] = stack.enter_context(open(pkg_dir / loc, "rb"))
                in_fd = in_files[loc].fileno()
                rem = length
                in_pos = off
                while rem > 0:
                    chunk = min(rem, 1 << 30)
                    sent = os.sendfile(out_fd, in_fd, in_pos, chunk)
                    if sent <= 0:
                        raise RuntimeError(f"sendfile returned {sent} for {loc} at {in_pos}")
                    in_pos += sent
                    rem -= sent
                cur_off += length
            tail_pad = (page_align - (cur_off % page_align)) % page_align
            if tail_pad > 0:
                os.write(out_fd, b"\x00" * tail_pad)
                cur_off += tail_pad
        tmp_data_path.replace(pkg_dir / target_data_name)
    finally:
        if tmp_data_path.exists():
            tmp_data_path.unlink()

    def rewrite_proto_inits(m_proto: onnx.ModelProto) -> bool:
        changed = False
        for t, ext in external_inits(m_proto):
            loc = ext.get("location", "")
            if loc not in old_files:
                continue
            key = (loc, int(ext.get("offset", 0)), int(ext["length"]))
            del t.external_data[:]
            for k, v in (
                ("location", target_data_name),
                ("offset", str(slice_new_offset[key])),
                ("length", str(key[2])),
            ):
                entry = t.external_data.add()
                entry.key = k
                entry.value = v
            changed = True
        return changed

    for p, m_proto in graphs.items():
        if rewrite_proto_inits(m_proto):
            save_model_replacing(m_proto, p)

    if pool_lines:
        new_pool_lines: list[str] = []
        for ln in pool_lines:
            parts = ln.split()
            if len(parts) >= 8 and parts[0] == "pool" and parts[-1] in old_files:
                key = (parts[-1], int(parts[3]), int(parts[4]))
                parts[3] = str(slice_new_offset[key])
                parts[-1] = target_data_name
                new_pool_lines.append(" ".join(parts))
            else:
                new_pool_lines.append(ln)
        write_text_replacing(pool_path, "\n".join(new_pool_lines) + "\n")

    removed_files = sorted(old_files - {target_data_name})
    for loc in removed_files:
        p_old = pkg_dir / loc
        if p_old.exists() or p_old.is_symlink():
            p_old.unlink()

    return {
        "consolidated": True,
        "target_data_file": target_data_name,
        "total_bytes": cur_off,
        "removed_files": removed_files,
    }


def compute_f32_rope_caches(max_rotary_rows: int = 262144) -> tuple[bytes, bytes, bytes, bytes]:
    """Return `(cos64_f32_bytes, sin64_f32_bytes, cos32_f32_bytes, sin32_f32_bytes)` in `float32`."""
    dim = 64
    inv = 1.0 / (10000000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    pos = np.arange(max_rotary_rows, dtype=np.float32)[:, None]
    freqs = pos * inv[None, :]
    cos32 = np.ascontiguousarray(np.cos(freqs), dtype=np.float32)
    sin32 = np.ascontiguousarray(np.sin(freqs), dtype=np.float32)
    cos64 = np.ascontiguousarray(np.concatenate([cos32, cos32], axis=-1), dtype=np.float32)
    sin64 = np.ascontiguousarray(np.concatenate([sin32, sin32], axis=-1), dtype=np.float32)
    return cos64.tobytes(), sin64.tobytes(), cos32.tobytes(), sin32.tobytes()


def write_gguf_ple_table(out_dir: Path, engram_kind: str, engram_gguf: Path | None = None) -> str:
    """Write `ple_table.txt` for `q8_0` (`scale -8.0`) or `iq4_nl` (`scale -20.0`) engram GGUF."""
    if engram_gguf is None:
        engram_gguf = DEFAULT_ENGRAM_Q8_GGUF if engram_kind == "q8_0" else DEFAULT_ENGRAM_IQ4_GGUF
    if not engram_gguf.is_file():
        raise FileNotFoundError(f"Missing engram GGUF: {engram_gguf}")
    engram_ts, engram_doff = read_gguf_tensor_infos(engram_gguf)
    if len(engram_ts) != 1 or engram_ts[0]["name"] != "per_layer_token_embd.weight":
        raise ValueError(
            f"{engram_gguf} must hold only per_layer_token_embd.weight, got {[t['name'] for t in engram_ts]}"
        )
    ggml_type = engram_ts[0]["ggml_type"]
    if ggml_type == 8:
        engram_dst_name = "Qwen3.8-Flash-Next-engram-Q8_0.gguf"
        ple_scale = "-8.0"
        ple_row_bytes = (160 // 32) * 34
    elif ggml_type == 20:
        engram_dst_name = "Qwen3.8-Flash-Next-engram-IQ4_NL.gguf"
        ple_scale = "-20.0"
        ple_row_bytes = (160 // 32) * 18
    else:
        raise ValueError(f"Unsupported engram ggml_type={ggml_type}")

    dst = out_dir / engram_dst_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(engram_gguf, dst)
    except OSError:
        shutil.copy2(engram_gguf, dst)

    if (out_dir / "ple").is_dir():
        shutil.rmtree(out_dir / "ple")

    ple_lines = [
        "qwen38_ple_table 1",
        "n_heads 16",
        "head_dim 160",
        "heads_per_ngram 8",
        "eos 248044",
        f"scale {ple_scale}",
        "mult 23703573157769 20109073645365 8052911324071 0",
        "vocab 20000003 20000023 20000033 20000047 20000059 20000063 20000069 20000077 20000081 20000093 20000107 20000147 20000153 20000159 20000161 20000171",
        "offset 0 20000003 40000026 60000059 80000106 100000165 120000228 140000297 160000374 180000455 200000548 220000655 240000802 260000955 280001114 300001275",
    ]
    rows_per_shard = 2500012
    total_ple_rows = 300001275 + 20000171
    cur_row = 0
    for _s_idx in range(128):
        r_cnt = min(rows_per_shard, total_ple_rows - cur_row)
        if r_cnt <= 0:
            break
        byte_off = engram_doff + cur_row * ple_row_bytes
        ple_lines.append(f"shard {engram_dst_name} {byte_off} {r_cnt}")
        cur_row += r_cnt
    write_text_replacing(out_dir / "ple_table.txt", "\n".join(ple_lines) + "\n")
    return engram_dst_name


def apply_precision_config_to_package(
    pkg_dir: Path,
    pcfg: PrecisionConfig,
    hf_dir: Path = DEFAULT_HF_DIR,
) -> dict[str, object]:
    """Apply `pcfg` component precision overrides in-place to a built RC5 or GGUF package at `pkg_dir`."""
    if not pcfg.is_non_default():
        return {"applied": False, "config": pcfg.to_dict()}
    pcfg.validate()

    hf_reader = HFCheckpointReader(hf_dir)
    model_path = pkg_dir / "model.onnx"
    model = onnx.load(str(model_path), load_external_data=False)
    inits_by_name = {t.name: t for t in model.graph.initializer}

    # 1. Optional PLE engram override (q8_0 or iq4_nl)
    if pcfg.ple_engram in ("q8_0", "iq4_nl"):
        write_gguf_ple_table(pkg_dir, pcfg.ple_engram, pcfg.engram_gguf_path)

    # 2. Optional RoPE / Indexer cache, FP32 GDN/Indexer projections, and FP16 QSA projections
    #    All precision-override tensors are packed into the single `attn_consts.data` file.
    cos64_f32_bytes, sin64_f32_bytes, cos32_f32_bytes, sin32_f32_bytes = compute_f32_rope_caches(262144)
    pool_path = pkg_dir / "weights_pool.txt"
    pool_lines = pool_path.read_text().splitlines() if pool_path.exists() else []

    ac_path = pkg_dir / "attn_consts.data"
    old_ac_bytes = b""
    if ac_path.is_symlink() or ac_path.exists():
        if pcfg.indexer_dtype != "fp32":
            old_ac_bytes = ac_path.read_bytes()
        ac_path.unlink()

    for legacy_blob in (
        "fp32_gates_and_indexer.data",
        "fp16_qsa_v_rc5.data",
        "fp16_qsa_kv.data",
        "fp16_qsa_qkv_o.data",
    ):
        p_leg = pkg_dir / legacy_blob
        if p_leg.is_symlink() or p_leg.exists():
            p_leg.unlink()

    ac_writer = AlignedBlobWriter(pkg_dir, "attn_consts.data")

    def add_ac_ext(onnx_name: str, raw_bytes: bytes, dt: int, dims: list[int]) -> tuple[str, int, int, int, list[int]]:
        info = ac_writer.write_raw(raw_bytes, dt, dims)
        inits_by_name[onnx_name] = make_ext_init(onnx_name, *info)
        return info

    if pcfg.indexer_dtype == "fp32":
        c64_info = add_ac_ext("capture.indexer_cos_cache.folded", cos64_f32_bytes, TensorProto.FLOAT, [262144, 64])
        s64_info = add_ac_ext("capture.indexer_sin_cache.folded", sin64_f32_bytes, TensorProto.FLOAT, [262144, 64])
        new_pool: list[str] = []
        for ln in pool_lines:
            if ln.startswith("pool 2 "):
                new_pool.append(f"pool 2 1 {c64_info[1]} {c64_info[2]} 2 262144 64 attn_consts.data")
            elif ln.startswith("pool 3 "):
                new_pool.append(f"pool 3 1 {s64_info[1]} {s64_info[2]} 2 262144 64 attn_consts.data")
            else:
                new_pool.append(ln)
        pool_lines = new_pool
    elif old_ac_bytes:
        ac_writer.write_raw(old_ac_bytes, TensorProto.UINT8, [len(old_ac_bytes)])

    if pcfg.rotary_cache_dtype == "fp32":
        c32_info = add_ac_ext("model.rotary_emb.cos_cache", cos32_f32_bytes, TensorProto.FLOAT, [262144, 32])
        s32_info = add_ac_ext("model.rotary_emb.sin_cache", sin32_f32_bytes, TensorProto.FLOAT, [262144, 32])
        new_pool = []
        for ln in pool_lines:
            if ln.startswith("pool 0 "):
                new_pool.append(f"pool 0 1 {c32_info[1]} {c32_info[2]} 2 262144 32 attn_consts.data")
            elif ln.startswith("pool 1 "):
                new_pool.append(f"pool 1 1 {s32_info[1]} {s32_info[2]} 2 262144 32 attn_consts.data")
            else:
                new_pool.append(ln)
        pool_lines = new_pool

    if pool_lines:
        write_text_replacing(pool_path, "\n".join(pool_lines) + "\n")

    # 3. Optional FP32 GDN gate projections & FP32 Indexer projections (packed into attn_consts.data)
    if pcfg.gdn_gate_dtype == "fp32" or pcfg.indexer_dtype == "fp32":
        for lid in range(48):
            lpfx = f"model.language_model.layers.{lid}"
            if lid % 4 != 3 and pcfg.gdn_gate_dtype == "fp32":
                w_a = np.ascontiguousarray(
                    hf_reader.read_bf16_tensor(f"{lpfx}.linear_attn.in_proj_a.weight").float().numpy().T,
                    dtype=np.float32,
                )
                w_b = np.ascontiguousarray(
                    hf_reader.read_bf16_tensor(f"{lpfx}.linear_attn.in_proj_b.weight").float().numpy().T,
                    dtype=np.float32,
                )
                add_ac_ext(
                    f"model.layers.{lid}.linear_attn.alpha.weight_t.f32",
                    w_a.tobytes(),
                    TensorProto.FLOAT,
                    list(w_a.shape),
                )
                add_ac_ext(
                    f"model.layers.{lid}.linear_attn.beta.weight_t.f32",
                    w_b.tobytes(),
                    TensorProto.FLOAT,
                    list(w_b.shape),
                )
            elif lid % 4 == 3 and pcfg.indexer_dtype == "fp32":
                w_iqk = np.ascontiguousarray(
                    hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.indexer.index_qk_proj.weight").float().numpy().T,
                    dtype=np.float32,
                )
                add_ac_ext(
                    f"model.layers.{lid}.self_attn.indexer_qk.weight_t.f32",
                    w_iqk.tobytes(),
                    TensorProto.FLOAT,
                    list(w_iqk.shape),
                )

    # 3b. Optional FP16 QSA V (`v_rc5_fp16`), K/V (`kv_fp16`), or Q/K/V/O (`fp16`) projections (packed into attn_consts.data)
    if pcfg.qsa_qkv_dtype == "v_rc5_fp16":
        import torch

        for lid in range(3, 48, 4):
            lpfx = f"model.language_model.layers.{lid}"
            w_v_hf = hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.v_proj.weight")
            w_f32 = w_v_hf.float().reshape(512, 2560 // 32, 32)
            scales_bf16 = (w_f32.abs().amax(dim=-1) / 127.0).to(torch.bfloat16)
            scales_f32 = scales_bf16.float().unsqueeze(-1)
            q_signed = torch.round(w_f32 / scales_f32).clamp(-128.0, 127.0)
            w_bf16_deq = (q_signed.to(torch.bfloat16) * scales_bf16.unsqueeze(-1)).float().reshape(512, 2560)
            w_v_fp16_t = np.ascontiguousarray(w_bf16_deq.numpy().T, dtype=np.float16)
            add_ac_ext(
                f"model.layers.{lid}.self_attn.v.rc5_deq.fp16",
                w_v_fp16_t.tobytes(),
                TensorProto.FLOAT16,
                list(w_v_fp16_t.shape),
            )
    elif pcfg.qsa_qkv_dtype == "kv_fp16":
        for lid in range(3, 48, 4):
            lpfx = f"model.language_model.layers.{lid}"
            w_k = hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.k_proj.weight").float().numpy().astype(np.float16).T
            w_v = hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.v_proj.weight").float().numpy().astype(np.float16).T
            w_kv = np.ascontiguousarray(np.concatenate([w_k, w_v], axis=1), dtype=np.float16)
            add_ac_ext(
                f"model.layers.{lid}.self_attn.kv.weight_t.fp16", w_kv.tobytes(), TensorProto.FLOAT16, list(w_kv.shape)
            )
        inits_by_name["const_split_qsa_kv_512_512"] = numpy_helper.from_array(
            np.array([512, 512], dtype=np.int64), name="const_split_qsa_kv_512_512"
        )
    elif pcfg.qsa_qkv_dtype == "fp16":
        for lid in range(3, 48, 4):
            lpfx = f"model.language_model.layers.{lid}"
            w_q = hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.q_proj.weight").float().numpy().astype(np.float16).T
            w_k = hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.k_proj.weight").float().numpy().astype(np.float16).T
            w_v = hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.v_proj.weight").float().numpy().astype(np.float16).T
            w_qkv = np.ascontiguousarray(np.concatenate([w_q, w_k, w_v], axis=1), dtype=np.float16)
            add_ac_ext(
                f"model.layers.{lid}.self_attn.qkv.weight_t.fp16",
                w_qkv.tobytes(),
                TensorProto.FLOAT16,
                list(w_qkv.shape),
            )
            w_o = np.ascontiguousarray(
                hf_reader.read_bf16_tensor(f"{lpfx}.self_attn.o_proj.weight").float().numpy().astype(np.float16).T,
                dtype=np.float16,
            )
            add_ac_ext(
                f"model.layers.{lid}.self_attn.o_proj.weight_t.fp16",
                w_o.tobytes(),
                TensorProto.FLOAT16,
                list(w_o.shape),
            )
        inits_by_name["const_split_qsa_qkv_12288_512_512"] = numpy_helper.from_array(
            np.array([12288, 512, 512], dtype=np.int64), name="const_split_qsa_qkv_12288_512_512"
        )

    # 4. Inline norms (indexer q/k layernorm and Layer 1 PLE norms)
    if pcfg.indexer_dtype == "fp32":
        for name in list(inits_by_name.keys()):
            m_iq = re.match(r"^v_model\.layers\.(\d+)\.self_attn\.indexer\.q_layernorm\..*\.folded\.bf16$", name)
            m_ik = re.match(r"^v_model\.layers\.(\d+)\.self_attn\.indexer\.k_layernorm\..*\.folded\.bf16$", name)
            if m_iq:
                lid = int(m_iq.group(1))
                w = 1.0 + hf_reader.read_bf16_tensor(
                    f"model.language_model.layers.{lid}.self_attn.indexer.q_layernorm.weight"
                ).float().numpy().astype(np.float32)
                set_inline_f32(inits_by_name[name], w.flatten())
            elif m_ik:
                lid = int(m_ik.group(1))
                w = 1.0 + hf_reader.read_bf16_tensor(
                    f"model.language_model.layers.{lid}.self_attn.indexer.k_layernorm.weight"
                ).float().numpy().astype(np.float32)
                set_inline_f32(inits_by_name[name], w.flatten())

    if pcfg.ple_norm_dtype == "fp32":
        for p_key, oname in (
            ("model.language_model.layers.1.ple.norm_key.weight", "v_model.layers.1.ple.norm_key.Add_200.folded"),
            ("model.language_model.layers.1.ple.norm_query.weight", "v_model.layers.1.ple.norm_query.Add_204.folded"),
            ("model.language_model.layers.1.ple.norm_conv.weight", "v_model.layers.1.ple.norm_conv.Add_224.folded"),
        ):
            if oname in inits_by_name:
                w = 1.0 + hf_reader.read_bf16_tensor(p_key).float().numpy().astype(np.float32)
                set_inline_f32(inits_by_name[oname], w.flatten())

    kv_scale_inits: dict[str, TensorProto] = {}
    orig_dsa_scale_inputs: dict[str, list[str]] = {}
    for name, tp in inits_by_name.items():
        if re.match(r"^node_GuardedDSA_\d+_(k|v)_scale$", name):
            tp_copy = TensorProto()
            tp_copy.CopyFrom(tp)
            kv_scale_inits[name] = tp_copy
    for n in model.graph.node:
        if n.op_type == "DynamicSparseAttention" and len(n.input) >= 20 and n.input[18] and n.input[19]:
            orig_dsa_scale_inputs[n.name] = [n.input[18], n.input[19]]

    if pcfg.kv_cache_dtype in ("bf16", "fp16"):
        for name in list(inits_by_name.keys()):
            if re.match(r"^node_GuardedDSA_\d+_(k|v)_scale$", name):
                del inits_by_name[name]

    # 5. Rewrite nodes in model.graph.node
    node_by_output = {o: n for n in model.graph.node for o in n.output}
    existing_node_names = {n.name for n in model.graph.node}
    removed_qsa_mm_names: set[str] = set()
    if pcfg.qsa_qkv_dtype == "fp16" and pcfg.indexer_dtype == "fp32":
        for n in model.graph.node:
            if n.op_type == "Split" and n.name.endswith("/SplitQsa"):
                mm_node = node_by_output.get(n.input[0])
                if mm_node is not None and mm_node.op_type == "MatMulNBits":
                    removed_qsa_mm_names.add(mm_node.name)

    new_nodes: list[onnx.NodeProto] = []
    for n in model.graph.node:
        if n.name in removed_qsa_mm_names:
            continue
        if n.op_type == "RMSNormRotary":
            out_fp16 = (
                1 if (pcfg.kv_cache_dtype == "fp16" or pcfg.qsa_qkv_dtype in ("v_rc5_fp16", "kv_fp16", "fp16")) else 0
            )
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            attrs["output_fp16"] = out_fp16
            new_nodes.append(
                helper.make_node(
                    "RMSNormRotary",
                    list(n.input),
                    list(n.output),
                    name=n.name,
                    domain=n.domain,
                    **attrs,
                )
            )
        elif n.op_type == "DynamicSparseAttention" and (
            pcfg.kv_cache_dtype in ("bf16", "fp16") or pcfg.qsa_qkv_dtype in ("v_rc5_fp16", "kv_fp16", "fp16")
        ):
            dsa_inputs = list(n.input[:18]) if pcfg.kv_cache_dtype in ("bf16", "fp16") else list(n.input)
            if pcfg.qsa_qkv_dtype == "fp16":
                dsa_out_orig = n.output[0]
                dsa_out_bf16 = f"{dsa_out_orig}.bf16"
                dsa_outputs = [dsa_out_bf16] + list(n.output[1:])
                new_nodes.extend(
                    [
                        helper.make_node(
                            "DynamicSparseAttention",
                            dsa_inputs,
                            dsa_outputs,
                            name=n.name,
                            domain=n.domain,
                            kv_num_heads=2,
                            num_heads=24,
                            scale=0.0625,
                            output_bf16=1,
                        ),
                        helper.make_node(
                            "Cast",
                            [dsa_out_bf16],
                            [dsa_out_orig],
                            name=f"{n.name}/CastOutputF16",
                            to=TensorProto.FLOAT16,
                        ),
                    ]
                )
            elif pcfg.qsa_qkv_dtype in ("v_rc5_fp16", "kv_fp16"):
                new_nodes.append(
                    helper.make_node(
                        "DynamicSparseAttention",
                        dsa_inputs,
                        list(n.output),
                        name=n.name,
                        domain=n.domain,
                        kv_num_heads=2,
                        num_heads=24,
                        scale=0.0625,
                        output_bf16=1,
                    )
                )
            else:
                new_nodes.append(
                    helper.make_node(
                        "DynamicSparseAttention",
                        dsa_inputs,
                        list(n.output),
                        name=n.name,
                        domain=n.domain,
                        kv_num_heads=2,
                        num_heads=24,
                        scale=0.0625,
                    )
                )
        elif (
            n.op_type == "MatMulNBits"
            and pcfg.qsa_qkv_dtype == "fp16"
            and len(n.input) >= 2
            and re.match(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight_t\.qweight$", n.input[1])
        ):
            m_o = re.match(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight_t\.qweight$", n.input[1])
            assert m_o is not None
            lid = int(m_o.group(1))
            o_out = n.output[0]
            o_fp16 = f"{o_out}.fp16"
            new_nodes.extend(
                [
                    helper.make_node(
                        "MatMul",
                        [n.input[0], f"model.layers.{lid}.self_attn.o_proj.weight_t.fp16"],
                        [o_fp16],
                        name=f"{n.name}/o_proj/MatMulF16",
                    ),
                    helper.make_node(
                        "Cast",
                        [o_fp16],
                        [o_out],
                        name=f"{n.name}/o_proj/CastOutBF16",
                        to=TensorProto.BFLOAT16,
                    ),
                ]
            )
        elif (
            n.op_type == "Split"
            and n.name.endswith("/SplitGdn")
            and pcfg.gdn_gate_dtype == "fp32"
            and f"{n.name}/beta/MatMulF32" not in existing_node_names
        ):
            mm_node = node_by_output[n.input[0]]
            a_name = mm_node.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", a_name)
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            qkv_out, z_out, b_out, a_out = n.output[0], n.output[1], n.output[2], n.output[3]
            unused_b = f"{b_out}.unused_int8"
            unused_a = f"{a_out}.unused_int8"
            a_f32_name = f"{a_name}.cast_f32"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Split", list(n.input), [qkv_out, z_out, unused_b, unused_a], name=n.name, axis=-1
                    ),
                    helper.make_node(
                        "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_f32_name, f"model.layers.{lid}.linear_attn.beta.weight_t.f32"],
                        [b_out],
                        name=f"{n.name}/beta/MatMulF32",
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_f32_name, f"model.layers.{lid}.linear_attn.alpha.weight_t.f32"],
                        [a_out],
                        name=f"{n.name}/alpha/MatMulF32",
                    ),
                ]
            )
        elif (
            n.op_type == "Split"
            and n.name.endswith("/SplitQsa")
            and pcfg.qsa_qkv_dtype == "v_rc5_fp16"
            and pcfg.indexer_dtype == "fp32"
        ):
            mm_node = node_by_output[n.input[0]]
            a_name = mm_node.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", a_name)
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            iqk_out, q_gate_out, k_out, v_out = n.output[0], n.output[1], n.output[2], n.output[3]
            unused_iqk = iqk_out if iqk_out.endswith(".unused_int8") else f"{iqk_out}.unused_int8"
            unused_v = f"{v_out}.unused_int8"
            a_fp16_name = f"{a_name}.cast_fp16"
            new_nodes.append(
                helper.make_node(
                    "Split",
                    list(n.input),
                    [unused_iqk, q_gate_out, k_out, unused_v],
                    name=n.name,
                    axis=-1,
                )
            )
            if f"{n.name}/indexer_qk/MatMulF32" not in existing_node_names:
                a_f32_name = f"{a_name}.cast_f32"
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                        ),
                        helper.make_node(
                            "MatMul",
                            [a_f32_name, f"model.layers.{lid}.self_attn.indexer_qk.weight_t.f32"],
                            [iqk_out],
                            name=f"{n.name}/indexer_qk/MatMulF32",
                        ),
                    ]
                )
            new_nodes.extend(
                [
                    helper.make_node(
                        "Cast", [a_name], [a_fp16_name], name=f"{n.name}/CastInputF16", to=TensorProto.FLOAT16
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_fp16_name, f"model.layers.{lid}.self_attn.v.rc5_deq.fp16"],
                        [v_out],
                        name=f"{n.name}/v/MatMulF16",
                    ),
                ]
            )
        elif (
            n.op_type == "Split"
            and n.name.endswith("/SplitQsa")
            and pcfg.qsa_qkv_dtype == "kv_fp16"
            and pcfg.indexer_dtype == "fp32"
        ):
            mm_node = node_by_output[n.input[0]]
            a_name = mm_node.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", a_name)
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            iqk_out, q_gate_out, k_out, v_out = n.output[0], n.output[1], n.output[2], n.output[3]
            unused_iqk = iqk_out if iqk_out.endswith(".unused_int8") else f"{iqk_out}.unused_int8"
            unused_k = f"{k_out}.unused_int8"
            unused_v = f"{v_out}.unused_int8"
            a_fp16_name = f"{a_name}.cast_fp16"
            kv_fp16_out = f"{n.input[0]}.kv_fp16"
            new_nodes.append(
                helper.make_node(
                    "Split",
                    list(n.input),
                    [unused_iqk, q_gate_out, unused_k, unused_v],
                    name=n.name,
                    axis=-1,
                )
            )
            if f"{n.name}/indexer_qk/MatMulF32" not in existing_node_names:
                a_f32_name = f"{a_name}.cast_f32"
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                        ),
                        helper.make_node(
                            "MatMul",
                            [a_f32_name, f"model.layers.{lid}.self_attn.indexer_qk.weight_t.f32"],
                            [iqk_out],
                            name=f"{n.name}/indexer_qk/MatMulF32",
                        ),
                    ]
                )
            new_nodes.extend(
                [
                    helper.make_node(
                        "Cast", [a_name], [a_fp16_name], name=f"{n.name}/CastInputF16", to=TensorProto.FLOAT16
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_fp16_name, f"model.layers.{lid}.self_attn.kv.weight_t.fp16"],
                        [kv_fp16_out],
                        name=f"{n.name}/kv/MatMulF16",
                    ),
                    helper.make_node(
                        "Split",
                        [kv_fp16_out, "const_split_qsa_kv_512_512"],
                        [k_out, v_out],
                        name=f"{n.name}/kv/SplitKvF16",
                        axis=-1,
                    ),
                ]
            )
        elif (
            n.op_type == "Split"
            and n.name.endswith("/SplitQsa")
            and pcfg.qsa_qkv_dtype == "fp16"
            and pcfg.indexer_dtype == "fp32"
        ):
            mm_node = node_by_output[n.input[0]]
            a_name = mm_node.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", a_name)
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            iqk_out, q_gate_out, k_out, v_out = n.output[0], n.output[1], n.output[2], n.output[3]
            a_fp16_name = f"{a_name}.cast_fp16"
            qkv_fp16_out = f"{n.input[0]}.qkv_fp16"
            if f"{n.name}/indexer_qk/MatMulF32" not in existing_node_names:
                a_f32_name = f"{a_name}.cast_f32"
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                        ),
                        helper.make_node(
                            "MatMul",
                            [a_f32_name, f"model.layers.{lid}.self_attn.indexer_qk.weight_t.f32"],
                            [iqk_out],
                            name=f"{n.name}/indexer_qk/MatMulF32",
                        ),
                    ]
                )
            new_nodes.extend(
                [
                    helper.make_node(
                        "Cast", [a_name], [a_fp16_name], name=f"{n.name}/CastInputF16", to=TensorProto.FLOAT16
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_fp16_name, f"model.layers.{lid}.self_attn.qkv.weight_t.fp16"],
                        [qkv_fp16_out],
                        name=f"{n.name}/qkv/MatMulF16",
                    ),
                    helper.make_node(
                        "Split",
                        [qkv_fp16_out, "const_split_qsa_qkv_12288_512_512"],
                        [q_gate_out, k_out, v_out],
                        name=f"{n.name}/qkv/SplitQsaF16",
                        axis=-1,
                    ),
                ]
            )
        elif (
            n.op_type == "Split"
            and n.name.endswith("/SplitQsa")
            and pcfg.indexer_dtype == "fp32"
            and f"{n.name}/indexer_qk/MatMulF32" not in existing_node_names
        ):
            mm_node = node_by_output[n.input[0]]
            a_name = mm_node.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", a_name)
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            iqk_out, q_gate_out, k_out, v_out = n.output[0], n.output[1], n.output[2], n.output[3]
            unused_iqk = f"{iqk_out}.unused_int8"
            a_f32_name = f"{a_name}.cast_f32"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Split", list(n.input), [unused_iqk, q_gate_out, k_out, v_out], name=n.name, axis=-1
                    ),
                    helper.make_node(
                        "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_f32_name, f"model.layers.{lid}.self_attn.indexer_qk.weight_t.f32"],
                        [iqk_out],
                        name=f"{n.name}/indexer_qk/MatMulF32",
                    ),
                ]
            )
        elif n.op_type == "MatMul" and n.name.endswith("/alpha/MatMul") and pcfg.gdn_gate_dtype == "fp32":
            a_name = n.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", n.input[1])
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            a_f32_name = f"{a_name}.cast_f32"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_f32_name, f"model.layers.{lid}.linear_attn.alpha.weight_t.f32"],
                        list(n.output),
                        name=f"{n.name}F32",
                    ),
                ]
            )
        elif n.op_type == "MatMul" and n.name.endswith("/beta/MatMul") and pcfg.gdn_gate_dtype == "fp32":
            a_name = n.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", n.input[1])
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            a_f32_name = f"{a_name}.beta_cast_f32"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_f32_name, f"model.layers.{lid}.linear_attn.beta.weight_t.f32"],
                        list(n.output),
                        name=f"{n.name}F32",
                    ),
                ]
            )
        elif n.op_type == "MatMul" and n.name.endswith("/indexer_qk/MatMul") and pcfg.indexer_dtype == "fp32":
            a_name = n.input[0]
            m_in_layer = re.search(r"layers\.(\d+)\.", n.input[1])
            assert m_in_layer is not None
            lid = int(m_in_layer.group(1))
            a_f32_name = f"{a_name}.cast_f32"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Cast", [a_name], [a_f32_name], name=f"{n.name}/CastInputF32", to=TensorProto.FLOAT
                    ),
                    helper.make_node(
                        "MatMul",
                        [a_f32_name, f"model.layers.{lid}.self_attn.indexer_qk.weight_t.f32"],
                        list(n.output),
                        name=f"{n.name}F32",
                    ),
                ]
            )
        else:
            new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # 6. Update graph inputs and outputs for KV cache and Indexer
    kv_elem_type = (
        TensorProto.FLOAT16
        if pcfg.kv_cache_dtype == "fp16"
        else (TensorProto.BFLOAT16 if pcfg.kv_cache_dtype == "bf16" else TensorProto.INT8)
    )
    for vi in model.graph.input:
        if re.match(r"^past_key_values\.\d+\.(key|value)$", vi.name) and pcfg.kv_cache_dtype in ("bf16", "fp16"):
            vi.type.tensor_type.elem_type = kv_elem_type
        elif re.match(r"^past_key_values\.\d+\.index_buffer$", vi.name) and pcfg.indexer_dtype == "fp32":
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    for vi in model.graph.output:
        if re.match(r"^present\.\d+\.(key|value)$", vi.name) and pcfg.kv_cache_dtype in ("bf16", "fp16"):
            vi.type.tensor_type.elem_type = kv_elem_type
        elif re.match(r"^present\.\d+\.index_buffer$", vi.name) and pcfg.indexer_dtype == "fp32":
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
        elif re.match(r"^state_update\.\d+\.indexer_value$", vi.name) and pcfg.indexer_dtype == "fp32":
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    used_inputs = {inp for n in model.graph.node for inp in n.input if inp}
    final_inits = [init_tp for name, init_tp in inits_by_name.items() if name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(final_inits)
    ensure_unified_dynamic_cache_model(model)
    save_model_replacing(model, model_path)

    # Context-tier graphs are copies of model.onnx; rebuild them so they match the transformed graph and attn_consts.data.
    rebuilt_tiers: list[str] = []
    for tier, (kv_cap, st_cap) in CONTEXT_TIERS.items():
        tier_path = pkg_dir / f"model_{tier}.onnx"
        if tier_path.exists():
            save_model_replacing(build_context_tier_model(model, kv_cap, st_cap), tier_path)
            rebuilt_tiers.append(tier)

    emitted_int8kv_onnx: str | None = None
    if (
        pcfg.emit_int8kv_variant
        and pcfg.kv_cache_dtype in ("bf16", "fp16")
        and kv_scale_inits
        and orig_dsa_scale_inputs
    ):
        model_int8kv = onnx.ModelProto()
        model_int8kv.CopyFrom(model)
        for n in model_int8kv.graph.node:
            if n.op_type == "DynamicSparseAttention" and n.name in orig_dsa_scale_inputs:
                while len(n.input) < 18:
                    n.input.append("")
                del n.input[18:]
                n.input.extend(orig_dsa_scale_inputs[n.name])
        for vi in model_int8kv.graph.input:
            if re.match(r"^past_key_values\.\d+\.(key|value)$", vi.name):
                vi.type.tensor_type.elem_type = TensorProto.INT8
        for vi in model_int8kv.graph.output:
            if re.match(r"^present\.\d+\.(key|value)$", vi.name):
                vi.type.tensor_type.elem_type = TensorProto.INT8
        int8_inits_map = {t.name: t for t in model_int8kv.graph.initializer}
        int8_inits_map.update(kv_scale_inits)
        used_int8_inputs = {inp for n in model_int8kv.graph.node for inp in n.input if inp}
        final_int8_inits = [init_tp for name, init_tp in int8_inits_map.items() if name in used_int8_inputs]
        del model_int8kv.graph.initializer[:]
        model_int8kv.graph.initializer.extend(final_int8_inits)
        ensure_unified_dynamic_cache_model(model_int8kv)
        emitted_int8kv_onnx = "model_int8kv.onnx"
        save_model_replacing(model_int8kv, pkg_dir / emitted_int8kv_onnx)

    # 7. Update genai_config.json session options and offload runtime_profiles
    cfg_path = pkg_dir / "genai_config.json"
    if cfg_path.exists():
        gcfg = json.loads(cfg_path.read_text())
        gcfg["model"]["decoder"]["session_options"]["ep.cuda.gated_delta_net_bf16_precision"] = pcfg.gdn_precision
        if emitted_int8kv_onnx is not None:
            gcfg["model"]["decoder"]["filename"] = "model.onnx"
            for prof_entry in gcfg.get("runtime_profiles", []):
                dec_ov = prof_entry.get("overlay", {}).get("model", {}).get("decoder", {})
                sopts = dec_ov.get("session_options", {})
                if sopts.get("session.moe.offload.mode") == "required":
                    dec_ov["filename"] = emitted_int8kv_onnx
                elif "filename" in dec_ov:
                    del dec_ov["filename"]
        write_text_replacing(cfg_path, json.dumps(gcfg, indent=2) + "\n")

    # Copy tactic cache TSV for model_int8kv.onnx if model.onnx tactic cache exists
    if emitted_int8kv_onnx is not None:
        src_tsv = pkg_dir / "model.onnx.matmulnbits_fpa_intb.tsv"
        dst_tsv = pkg_dir / f"{emitted_int8kv_onnx}.matmulnbits_fpa_intb.tsv"
        if src_tsv.exists() and not dst_tsv.exists():
            shutil.copy2(src_tsv, dst_tsv)

    # 8. Optionally consolidate all target external .data files into model.onnx.data
    consolidation_info = None
    if pcfg.consolidate_data_files:
        consolidation_info = consolidate_package_external_data(pkg_dir, "model.onnx.data")

    # 9. Update package_manifest.json
    manifest_path = pkg_dir / "package_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["precision_config"] = pcfg.to_dict()
        if emitted_int8kv_onnx is not None:
            manifest["onnx_variants"] = {
                f"{pcfg.kv_cache_dtype}_kv": "model.onnx",
                "int8_kv": emitted_int8kv_onnx,
            }
        if consolidation_info is not None and consolidation_info.get("consolidated"):
            manifest["external_data_files"] = ["model.onnx.data", "dflash2.onnx.data", "embedding.onnx.data"]
        write_text_replacing(manifest_path, json.dumps(manifest, indent=2) + "\n")

    return {
        "applied": True,
        "config": pcfg.to_dict(),
        "emitted_int8kv_onnx": emitted_int8kv_onnx,
        "rebuilt_context_tiers": rebuilt_tiers,
        "consolidation": consolidation_info,
    }
