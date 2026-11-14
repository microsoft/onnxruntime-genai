# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Checkpoint, ONNX initializer, context-tier and file helpers shared by `build_package.py` and `precision_options.py`.

This module imports neither script, so both can use it without an import cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from rope import ContextCapacity

ALIGN = 4096

CONTEXT_TIERS: dict[str, tuple[int, int]] = {
    "2k": (2312, 2312 // 4 + 1),
    "8k": (8456, 8456 // 4 + 1),
    "16k": (16648, 16648 // 4 + 1),
    "32k": (33032, 33032 // 4 + 1),
    "64k": (65800, 65800 // 4 + 1),
    "128k": (131080, 131080 // 4 + 1),
    "256k": (262152, 262152 // 4 + 1),
}
SRC_KV_CAPS = {33028, 33032, 131080, 133128, 262152}
SRC_STATE_CAPS = {8257, 8259, 32771, 33283, 65539}


def align_up(val: int, align: int = ALIGN) -> int:
    return (val + align - 1) // align * align


FINGERPRINT_THREADS = 16


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the complete contents of `path`.

    Always read the file: writes through an already-dirty writable mmap can change its contents without changing its stat
    metadata, so a stat-based digest cache cannot safely validate mutable checkpoints.
    """
    h = hashlib.sha256()
    buf = bytearray(4 << 20)
    view = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        while n := f.readinto(buf):
            h.update(view[:n])
    return h.hexdigest()


class HFCheckpointReader:
    """Unified safetensors reader supporting single-file (`model.safetensors`), sharded (`model.safetensors.index.json`),
    `model.layers.*` <-> `model.language_model.layers.*` name resolution, and `BF16` / `F32` / `I64` / `F8_E4M3` weights.
    """

    def __init__(self, hf_dir: Path, *, require_checkpoint: bool = False) -> None:
        self.hf_dir = Path(hf_dir)
        self.ckpt_dir = self.hf_dir.resolve() if self.hf_dir.exists() else self.hf_dir
        self._header_cache: dict[str, tuple[int, dict]] = {}
        idx_path = self.hf_dir / "model.safetensors.index.json"
        single_path = self.hf_dir / "model.safetensors"
        if idx_path.is_file():
            self.weight_map: dict[str, str] = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
        elif single_path.is_file():
            base_off, hdr = self._get_header("model.safetensors")
            _ = base_off
            self.weight_map = {k: "model.safetensors" for k in hdr if k != "__metadata__"}
        elif require_checkpoint:
            raise FileNotFoundError(
                f"Neither model.safetensors.index.json nor model.safetensors found in {self.ckpt_dir}"
            )
        else:
            self.weight_map = {}

    def _get_header(self, filename: str) -> tuple[int, dict]:
        if filename not in self._header_cache:
            path = self.hf_dir / filename
            with open(path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len).decode("utf-8"))
            self._header_cache[filename] = (8 + hdr_len, hdr)
        return self._header_cache[filename]

    @staticmethod
    def _candidate_names(name: str) -> list[str]:
        candidates = [name]
        if name.startswith("model.language_model."):
            suffix = name[len("model.language_model.") :]
            candidates.extend((f"model.{suffix}", suffix))
        elif name.startswith("model."):
            suffix = name[len("model.") :]
            candidates.extend((f"model.language_model.{suffix}", suffix))
        else:
            candidates.extend((f"model.{name}", f"model.language_model.{name}"))
        if not name.endswith(".weight"):
            for base in list(candidates):
                candidates.append(f"{base}.weight")
        return candidates

    def _resolve_key(self, name: str) -> str:
        for cand in self._candidate_names(name):
            if cand in self.weight_map:
                return cand
        raise KeyError(f"Weight {name!r} not found in {self.hf_dir}")

    def read_raw_tensor(self, name: str) -> tuple[str, list[int], bytes]:
        key = self._resolve_key(name)
        shard = self.weight_map[key]
        base_off, hdr = self._get_header(shard)
        info = hdr[key]
        start, end = info["data_offsets"]
        with open(self.hf_dir / shard, "rb") as f:
            f.seek(base_off + start)
            raw = f.read(end - start)
        return info["dtype"], info["shape"], raw

    def read_tensor(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        try:
            key = self._resolve_key(name)
        except KeyError:
            scale_key = None
            for cand in self._candidate_names(name):
                if f"{cand}_scale_inv" in self.weight_map and cand in self.weight_map:
                    scale_key = cand
                    break
            if scale_key is None:
                raise
            key = scale_key

        if f"{key}_scale_inv" in self.weight_map:
            dtype_w, shape_w, raw_w = self.read_raw_tensor(key)
            dtype_s, shape_s, raw_s = self.read_raw_tensor(f"{key}_scale_inv")
            if dtype_w != "F8_E4M3":
                raise ValueError(f"Expected F8_E4M3 for {key}, got {dtype_w}")
            w_u8 = torch.from_numpy(np.frombuffer(raw_w, dtype=np.uint8).copy()).reshape(shape_w).to(device)
            w_f32 = w_u8.view(torch.float8_e4m3fn).float()
            s_dtype = np.float32 if dtype_s == "F32" else np.uint16
            s_arr = np.frombuffer(raw_s, dtype=s_dtype).copy()
            if dtype_s == "BF16":
                s_f32 = torch.from_numpy(s_arr).view(torch.bfloat16).reshape(shape_s).to(device).float()
            else:
                s_f32 = torch.from_numpy(s_arr).reshape(shape_s).to(device).float()
            if w_f32.dim() == 2:
                rows, cols = w_f32.shape
                sn, sk = s_f32.shape
                s_exp = s_f32.repeat_interleave(rows // sn, dim=0).repeat_interleave(cols // sk, dim=1)
                return (w_f32 * s_exp).to(torch.bfloat16)
            if w_f32.dim() == 3:
                _num_experts, rows, cols = w_f32.shape
                _, sn, sk = s_f32.shape
                s_exp = s_f32.repeat_interleave(rows // sn, dim=1).repeat_interleave(cols // sk, dim=2)
                return (w_f32 * s_exp).to(torch.bfloat16)
            raise ValueError(f"Unsupported rank {w_f32.dim()} for F8_E4M3 tensor {key}")

        dtype_str, shape, raw = self.read_raw_tensor(key)
        if dtype_str == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            t = torch.from_numpy(arr).reshape(shape)
        elif dtype_str == "I64":
            arr = np.frombuffer(raw, dtype=np.int64).copy()
            t = torch.from_numpy(arr).reshape(shape)
        else:
            raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {key}")
        return t.to(device=device, non_blocking=False) if str(device) != "cpu" else t

    def read_bf16_tensor(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        t = self.read_tensor(name, device=device)
        if t.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 for {name}, got {t.dtype}")
        return t

    def checkpoint_fingerprint(self) -> str:
        """Return a SHA-256 of the weight map and the complete contents of every shard.

        Weight banks are reused only while their recorded fingerprint matches, so every byte is hashed: a change anywhere
        in a shard must produce a new fingerprint. The shards are hashed in parallel.
        """
        h = hashlib.sha256()
        for item_k, item_v in sorted(self.weight_map.items()):
            h.update(f"{item_k}:{item_v}\n".encode())
        shards = [shard for shard in sorted(set(self.weight_map.values())) if (self.hf_dir / shard).is_file()]
        with ThreadPoolExecutor(max_workers=max(1, min(FINGERPRINT_THREADS, len(shards)))) as ex:
            digests = list(ex.map(file_sha256, [self.hf_dir / shard for shard in shards]))
        for shard, digest in zip(shards, digests, strict=True):
            h.update(f"{shard}:{digest}\n".encode())
        return h.hexdigest()


class CheckpointTensorLoader(HFCheckpointReader):
    """Memory-mapped reader for single-file (`model.safetensors`) or sharded (`model.safetensors.index.json`) checkpoints."""

    def __init__(self, ckpt_dir: Path) -> None:
        super().__init__(ckpt_dir, require_checkpoint=True)


def checkpoint_fingerprint(hf_dir: Path | None) -> str | None:
    if hf_dir is None or not Path(hf_dir).is_dir():
        return None
    reader = HFCheckpointReader(Path(hf_dir))
    if not reader.weight_map:
        return None
    return reader.checkpoint_fingerprint()


def set_ext_data(tensor: TensorProto, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> None:
    tensor.data_type = dtype
    del tensor.dims[:]
    tensor.dims.extend(dims)
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
        e = tensor.external_data.add()
        e.key, e.value = k, v


def make_ext_init(name: str, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> TensorProto:
    t = TensorProto()
    t.name = name
    set_ext_data(t, location, offset, length, dtype, dims)
    return t


def set_inline_f32(tensor: TensorProto, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    tensor.data_type = TensorProto.FLOAT
    del tensor.dims[:]
    tensor.dims.extend(arr.shape)
    del tensor.external_data[:]
    tensor.data_location = TensorProto.DEFAULT
    tensor.raw_data = arr.tobytes()


class AlignedBlobWriter:
    """Writes raw tensors to a new external data file, each at a `ALIGN`-byte aligned offset."""

    def __init__(self, out_dir: Path, filename: str) -> None:
        self.filename = filename
        self.path = out_dir / filename
        self.path.write_bytes(b"")
        self.pos = 0

    def write_raw(self, raw: bytes, dtype: int, dims: list[int]) -> tuple[str, int, int, int, list[int]]:
        off = align_up(self.pos)
        with open(self.path, "ab") as fh:
            fh.write(b"\x00" * (off - self.pos))
            fh.write(raw)
        self.pos = off + len(raw)
        return self.filename, off, len(raw), dtype, list(dims)


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

DEFAULT_PLE_VOCAB: tuple[int, ...] = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)
DEFAULT_PLE_OFFSETS: tuple[int, ...] = (
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
)
DEFAULT_PLE_MULT: tuple[int, ...] = (23703573157769, 20109073645365, 8052911324071, 0)
DEFAULT_PLE_ROWS_PER_SHARD = 2500012


def parse_gguf_header_stream(path: Path, *, validate_bounds: bool = True) -> tuple[dict[str, object], list[dict], int]:
    """Stream-parse a GGUF v2/v3 header of arbitrary size and optionally validate tensor byte ranges against file size."""
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size

        def read_exact(size: int) -> bytes:
            buf = f.read(size)
            if len(buf) != size:
                raise ValueError(f"Truncated GGUF header in {path}")
            return buf

        def skip_exact(size: int) -> None:
            if size < 0 or f.tell() + size > file_size:
                raise ValueError(f"Truncated GGUF metadata in {path}")
            if size > 0:
                f.seek(size, os.SEEK_CUR)

        def read(fmt: str):
            return struct.unpack(fmt, read_exact(struct.calcsize(fmt)))

        def read_str(*, skip_only: bool = False) -> str:
            (n,) = read("<Q")
            if skip_only:
                skip_exact(n)
                return ""
            return read_exact(n).decode("utf-8", errors="replace")

        scalar_formats = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            7: "<B",
            10: "<Q",
            11: "<q",
            12: "<d",
        }

        def read_value(vtype: int, *, skip_only: bool = False) -> object:
            if vtype in scalar_formats:
                fmt = scalar_formats[vtype]
                if skip_only:
                    skip_exact(struct.calcsize(fmt))
                    return None
                return read(fmt)[0]
            if vtype == 8:
                return read_str(skip_only=skip_only)
            if vtype == 9:
                atype, alen = read("<IQ")
                if atype in scalar_formats:
                    skip_exact(alen * struct.calcsize(scalar_formats[atype]))
                else:
                    for _ in range(alen):
                        read_value(atype, skip_only=True)
                return f"arr[{alen}]"
            raise ValueError(f"Unknown GGUF value type {vtype} in {path}")

        magic, version, n_tensors, n_kv = read("<4sIQQ")
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version} in {path}")

        kvs: dict[str, object] = {}
        for _ in range(n_kv):
            key = read_str()
            (vtype,) = read("<I")
            kvs[key] = read_value(vtype)

        tensors: list[dict] = []
        for _ in range(n_tensors):
            name = read_str()
            (n_dims,) = read("<I")
            dims = [read("<Q")[0] for _ in range(n_dims)]
            ggml_type, rel_offset = read("<IQ")
            tensors.append({"name": name, "dims": dims, "ggml_type": ggml_type, "rel_offset": rel_offset})

        alignment = int(kvs.get("general.alignment", 32))
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"Invalid GGUF alignment {alignment} in {path}")
        data_offset = align_up(f.tell(), alignment)

    if validate_bounds and tensors and data_offset > file_size:
        raise ValueError(
            f"Truncated GGUF file {path}: tensor data section starts at byte {data_offset}, past file size {file_size}"
        )

    for t in tensors:
        t["data_offset"] = data_offset + t["rel_offset"]
        type_info = GGML_TYPE_INFO.get(t["ggml_type"])
        if type_info is None:
            if validate_bounds:
                raise ValueError(f"Unsupported GGUF tensor type {t['ggml_type']} for {t['name']} in {path}")
            continue
        if not t["dims"] or any(int(d) <= 0 for d in t["dims"]):
            raise ValueError(f"Invalid GGUF tensor dimensions {t['dims']} for {t['name']} in {path}")
        tname, blck_elems, blck_bytes = type_info
        d0 = int(t["dims"][0])
        if d0 % blck_elems != 0:
            raise ValueError(f"{t['name']} in {path}: d0={d0} is not divisible by block size {blck_elems}")
        row_bytes = (d0 // blck_elems) * blck_bytes
        c_dims = [int(d) for d in reversed(t["dims"][1:])]
        n_rows = 1
        for d in c_dims:
            n_rows *= d
        n_bytes = n_rows * row_bytes
        t["type_name"] = tname
        t["row_bytes"] = row_bytes
        t["n_bytes"] = n_bytes
        t["onnx_u8_shape"] = [*c_dims, row_bytes] if c_dims else [row_bytes]
        if validate_bounds and t["data_offset"] + n_bytes > file_size:
            raise ValueError(
                f"Truncated GGUF tensor {t['name']} in {path}: declared range "
                f"[{t['data_offset']}, {t['data_offset'] + n_bytes}) exceeds file size {file_size}"
            )
    return kvs, tensors, data_offset


def read_gguf_tensor_infos(path: Path, *, validate_bounds: bool = False) -> tuple[list[dict], int]:
    """Return the tensor infos (`name`, `dims`, `ggml_type`, absolute `data_offset`) of the GGUF file `path` and its data offset."""
    _kvs, tensors, data_offset = parse_gguf_header_stream(path, validate_bounds=validate_bounds)
    return tensors, data_offset


def read_ple_table_spec(ple_table_path: Path | None = None) -> dict[str, object]:
    """Read PLE table geometry (`head_dim`, `vocab`, `offset`, `total_rows`, etc.) or return the Qwen3.8 default."""
    spec: dict[str, object] = {
        "n_heads": 16,
        "head_dim": 160,
        "heads_per_ngram": 8,
        "eos": 248044,
        "mult": list(DEFAULT_PLE_MULT),
        "vocab": list(DEFAULT_PLE_VOCAB),
        "offset": list(DEFAULT_PLE_OFFSETS),
    }
    if ple_table_path is not None and Path(ple_table_path).is_file():
        lines = Path(ple_table_path).read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != "qwen38_ple_table 1":
            raise ValueError(f"Unsupported PLE table header in {ple_table_path}")
        for raw_line in lines[1:]:
            parts = raw_line.split()
            if not parts:
                continue
            key = parts[0]
            if key in ("n_heads", "head_dim", "heads_per_ngram", "eos"):
                spec[key] = int(parts[1])
            elif key in ("mult", "vocab", "offset"):
                spec[key] = [int(x) for x in parts[1:]]
    head_dim = int(spec["head_dim"])
    if head_dim <= 0 or head_dim % 32 != 0:
        raise ValueError(f"GGUF PLE table requires head_dim to be a positive multiple of 32, got {head_dim}")
    vocab = [int(x) for x in spec["vocab"]]  # type: ignore[union-attr]
    offset = [int(x) for x in spec["offset"]]  # type: ignore[union-attr]
    if not vocab or len(vocab) != len(offset):
        raise ValueError("Invalid PLE vocab/offset metadata")
    spec["total_rows"] = max(o + v for o, v in zip(offset, vocab, strict=True))
    return spec


def validate_engram_gguf(
    engram_gguf: Path,
    ple_spec: dict[str, object],
    *,
    expected_kind: str | None = None,
) -> tuple[str, str, int, int]:
    """Validate an engram GGUF's tensor name, quantization type, shape, and byte bounds."""
    engram_path = Path(engram_gguf).expanduser()
    if not engram_path.is_file():
        raise FileNotFoundError(f"Missing engram GGUF: {engram_path}")
    _kvs, engram_ts, _data_offset = parse_gguf_header_stream(engram_path, validate_bounds=True)
    if len(engram_ts) != 1 or engram_ts[0]["name"] != "per_layer_token_embd.weight":
        raise ValueError(
            f"{engram_path} must hold only per_layer_token_embd.weight, got {[t['name'] for t in engram_ts]}"
        )
    t0 = engram_ts[0]
    head_dim = int(ple_spec["head_dim"])
    total_rows = int(ple_spec["total_rows"])
    expected_dims = [head_dim, total_rows]
    if [int(d) for d in t0["dims"]] != expected_dims:
        raise ValueError(f"Invalid engram tensor shape in {engram_path}: expected {expected_dims}, got {t0['dims']}")
    ggml_type = int(t0["ggml_type"])
    if ggml_type == 20:
        engram_dst_name = "Qwen3.8-Flash-Next-engram-IQ4_NL.gguf"
        ple_format = "iq4_nl"
        ple_row_bytes = (head_dim // 32) * 18
    elif ggml_type == 8:
        engram_dst_name = "Qwen3.8-Flash-Next-engram-Q8_0.gguf"
        ple_format = "q8_0"
        ple_row_bytes = (head_dim // 32) * 34
    else:
        raise ValueError(f"Unsupported engram GGUF tensor type {ggml_type} in {engram_path}")
    if expected_kind in ("iq4_nl", "q8_0") and ple_format != expected_kind:
        raise ValueError(f"Engram GGUF {engram_path} has format {ple_format}, but {expected_kind} was requested")
    # The tensor may start past the data section start (rel_offset > 0), so use its own absolute offset.
    engram_tensor_offset = int(t0["data_offset"])
    required_end = engram_tensor_offset + total_rows * ple_row_bytes
    actual_size = engram_path.stat().st_size
    if required_end > actual_size:
        raise ValueError(
            f"Truncated engram GGUF {engram_path}: required {required_end} bytes for {total_rows} rows, got {actual_size}"
        )
    return engram_dst_name, ple_format, ple_row_bytes, engram_tensor_offset


def format_gguf_ple_table_lines(
    engram_dst_name: str,
    ple_format: str,
    engram_tensor_offset: int,
    ple_row_bytes: int,
    ple_spec: dict[str, object] | None = None,
) -> list[str]:
    """Format `qwen38_ple_table 1` lines pointing at `engram_dst_name`.

    `engram_tensor_offset` is the absolute file offset of the engram tensor's first row.
    """
    spec = read_ple_table_spec() if ple_spec is None else ple_spec
    mult_str = " ".join(str(x) for x in spec["mult"])  # type: ignore[union-attr]
    vocab_str = " ".join(str(x) for x in spec["vocab"])  # type: ignore[union-attr]
    offset_str = " ".join(str(x) for x in spec["offset"])  # type: ignore[union-attr]
    ple_lines = [
        "qwen38_ple_table 1",
        f"n_heads {spec['n_heads']}",
        f"head_dim {spec['head_dim']}",
        f"heads_per_ngram {spec['heads_per_ngram']}",
        f"eos {spec['eos']}",
        f"format {ple_format}",
        f"mult {mult_str}",
        f"vocab {vocab_str}",
        f"offset {offset_str}",
    ]
    total_ple_rows = int(spec["total_rows"])
    rows_per_shard = max(1, min(DEFAULT_PLE_ROWS_PER_SHARD, (total_ple_rows + 127) // 128))
    if total_ple_rows == DEFAULT_PLE_OFFSETS[-1] + DEFAULT_PLE_VOCAB[-1]:
        rows_per_shard = DEFAULT_PLE_ROWS_PER_SHARD
    cur_row = 0
    for _s_idx in range(128):
        r_cnt = min(rows_per_shard, total_ple_rows - cur_row)
        if r_cnt <= 0:
            break
        byte_off = engram_tensor_offset + cur_row * ple_row_bytes
        ple_lines.append(f"shard {engram_dst_name} {byte_off} {r_cnt}")
        cur_row += r_cnt
    if cur_row < total_ple_rows:
        byte_off = engram_tensor_offset + cur_row * ple_row_bytes
        ple_lines.append(f"shard {engram_dst_name} {byte_off} {total_ple_rows - cur_row}")
    return ple_lines


# Linux copies file ranges in the kernel. Windows has no os.sendfile, and the macOS one only writes to sockets.
USE_SENDFILE = sys.platform.startswith("linux") and hasattr(os, "sendfile")
COPY_CHUNK_BYTES = 64 << 20


def write_all(out_fd: int, data: bytes) -> None:
    """Writes all of `data` to the file descriptor `out_fd`; os.write may write only part of it."""
    view = memoryview(data)
    while view:
        written = os.write(out_fd, view)
        if written <= 0:
            raise RuntimeError(f"os.write returned {written} with {len(view)} bytes left to write")
        view = view[written:]


def append_file_range(out_fd: int, src_path: Path, offset: int, length: int) -> None:
    """Appends `length` bytes of `src_path` starting at `offset` to the file descriptor `out_fd`.

    Uses os.sendfile where it writes to regular files (no user-space copy) and a bounded buffer elsewhere.
    """
    with open(src_path, "rb") as src:
        pos, remaining = offset, length
        if not USE_SENDFILE:
            src.seek(offset)
        while remaining > 0:
            if USE_SENDFILE:
                copied = os.sendfile(out_fd, src.fileno(), pos, min(remaining, 1 << 30))
            else:
                chunk = src.read(min(remaining, COPY_CHUNK_BYTES))
                write_all(out_fd, chunk)
                copied = len(chunk)
            if copied <= 0:
                raise RuntimeError(
                    f"{src_path} ends at byte {pos}, {remaining} bytes short of the range [{offset}, {offset + length})"
                )
            pos += copied
            remaining -= copied


def replace_file(path: Path, write: Callable[[Path], None]) -> None:
    """Write `path` through a temporary file that then replaces it.

    Package files may be hard links into the `--base-pkg` package. Writing them in place would modify that package as well.
    """
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_replacing(path: Path, text: str) -> None:
    replace_file(path, lambda tmp: tmp.write_text(text))


def save_model_replacing(model: onnx.ModelProto, path: Path) -> None:
    replace_file(path, lambda tmp: onnx.save(model, str(tmp)))


def ensure_unified_dynamic_cache_model(
    model: onnx.ModelProto,
    max_total_sequence_length: int | None = None,
    indexer_state_capacity_attr: int | None = None,
) -> onnx.ModelProto:
    """Convert `model` in place into a unified dynamic-capacity graph with symbolic `kv_capacity` / `state_capacity`."""
    explicit_capacity = max_total_sequence_length is not None
    if max_total_sequence_length is None:
        existing = next((t for t in model.graph.initializer if t.name == "contrib_const_dsa_total_seq"), None)
        max_total_sequence_length = int(numpy_helper.to_array(existing)[0]) if existing else 262152
    if explicit_capacity and indexer_state_capacity_attr is None:
        indexer_state_capacity_attr = max_total_sequence_length // 4 + 1
    for idx, t in enumerate(model.graph.initializer):
        if t.name == "contrib_const_dsa_total_seq":
            model.graph.initializer[idx].CopyFrom(
                numpy_helper.from_array(
                    np.array([max_total_sequence_length], dtype=np.int32), name="contrib_const_dsa_total_seq"
                )
            )
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = vi.type.tensor_type.shape.dim
            if vi.name in ("past_position_ids", "present_position_ids") and len(dims) == 2:
                dims[1].ClearField("dim_value")
                dims[1].dim_param = "kv_capacity"
            elif vi.name.endswith((".key", ".value")) and len(dims) == 4:
                dims[2].ClearField("dim_value")
                dims[2].dim_param = "kv_capacity"
            elif vi.name.endswith(".index_key") and len(dims) == 3:
                dims[1].ClearField("dim_value")
                dims[1].dim_param = "state_capacity"
    for n in model.graph.node:
        if n.op_type == "PackedSparseAttentionIndexer":
            for a in n.attribute:
                if a.type == onnx.AttributeProto.INT and a.name == "state_capacity":
                    if indexer_state_capacity_attr is not None:
                        a.i = indexer_state_capacity_attr
    set_capacity_metadata(model, max_total_sequence_length, "unified")
    return model


def build_context_tier_model(src_model: onnx.ModelProto, kv_capacity: int, state_capacity: int) -> onnx.ModelProto:
    """Create a fixed-capacity context-tier copy (`model_2k.onnx`, `model_8k.onnx`, `model_64k.onnx`, `model_128k.onnx`)."""
    m_tier = onnx.ModelProto()
    m_tier.CopyFrom(src_model)
    for idx, t in enumerate(m_tier.graph.initializer):
        if t.name == "contrib_const_dsa_total_seq":
            m_tier.graph.initializer[idx].CopyFrom(
                numpy_helper.from_array(np.array([kv_capacity], dtype=np.int32), name="contrib_const_dsa_total_seq")
            )
    for vi in list(m_tier.graph.input) + list(m_tier.graph.output) + list(m_tier.graph.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = vi.type.tensor_type.shape.dim
            if vi.name in ("past_position_ids", "present_position_ids") and len(dims) == 2:
                dims[1].ClearField("dim_param")
                dims[1].dim_value = kv_capacity
            elif vi.name.endswith((".key", ".value")) and len(dims) == 4:
                dims[2].ClearField("dim_param")
                dims[2].dim_value = kv_capacity
            elif vi.name.endswith(".index_key") and len(dims) == 3:
                dims[1].ClearField("dim_param")
                dims[1].dim_value = state_capacity
            else:
                for d in dims:
                    if d.HasField("dim_value"):
                        if d.dim_value in SRC_KV_CAPS:
                            d.dim_value = kv_capacity
                        elif d.dim_value in SRC_STATE_CAPS:
                            d.dim_value = state_capacity
    for n in m_tier.graph.node:
        for a in n.attribute:
            if (
                n.op_type == "PackedSparseAttentionIndexer"
                and a.type == onnx.AttributeProto.INT
                and a.name == "state_capacity"
            ):
                a.i = state_capacity
    set_capacity_metadata(m_tier, kv_capacity, "fixed")
    return m_tier


def set_capacity_metadata(model: onnx.ModelProto, kv_capacity: int, layout: str) -> None:
    props = {p.key: p.value for p in model.metadata_props}
    props["experiment.fixed_kv_capacity"] = str(kv_capacity)
    if "qwen38.position_config" in props:
        positions = json.loads(props["qwen38.position_config"])
        if positions["capacity"] is not None:
            capacity = ContextCapacity(kv_capacity - 8, layout)
            if min(positions["target_rows"], positions["draft_rows"]) < capacity.positional_rows:
                raise ValueError("Requested graph capacity exceeds positional coverage")
            positions["capacity"] = capacity.to_dict()
            props["qwen38.position_config"] = json.dumps(positions, sort_keys=True)
    helper.set_model_props(model, props)


GGUF_DFLASH2_RESERVED_BLOCKS = 13
GGUF_MIN_NUM_BLOCKS_32K = 128 + GGUF_DFLASH2_RESERVED_BLOCKS  # 141
GGUF_NUM_BLOCKS_32K = 129 + GGUF_DFLASH2_RESERVED_BLOCKS  # 142
GGUF_MIN_NUM_BLOCKS_64K = 256 + GGUF_DFLASH2_RESERVED_BLOCKS  # 269
GGUF_NUM_BLOCKS_64K = 257 + GGUF_DFLASH2_RESERVED_BLOCKS  # 270
GGUF_MIN_NUM_BLOCKS_128K = 512 + GGUF_DFLASH2_RESERVED_BLOCKS  # 525
GGUF_NUM_BLOCKS_128K = GGUF_MIN_NUM_BLOCKS_128K  # 525
GGUF_MIN_NUM_BLOCKS_256K = 1024 + GGUF_DFLASH2_RESERVED_BLOCKS  # 1037
GGUF_NUM_BLOCKS_256K = GGUF_MIN_NUM_BLOCKS_256K  # 1037


def compute_gguf_model_weight_stats(model: onnx.ModelProto) -> dict[str, int]:
    """Compute referenced external weight byte totals (`gguf_bytes`, `non_gguf_bytes`, `moe_bytes`) for `model`."""
    used_inputs = {inp for n in model.graph.node for inp in n.input if inp}
    inits_by_name = {t.name: t for t in model.graph.initializer if t.name in used_inputs}

    moe_slices: set[tuple[str, int, int]] = set()
    for n in model.graph.node:
        if n.op_type == "MoEPacked":
            for wi in n.input[2:5]:
                t = inits_by_name.get(wi)
                if t is not None and t.data_location == TensorProto.EXTERNAL:
                    ext = {e.key: e.value for e in t.external_data}
                    moe_slices.add((ext["location"], int(ext.get("offset", 0)), int(ext["length"])))

    gguf_slices: set[tuple[str, int, int]] = set()
    non_gguf_slices: set[tuple[str, int, int]] = set()
    for t in inits_by_name.values():
        if t.data_location != TensorProto.EXTERNAL:
            continue
        ext = {e.key: e.value for e in t.external_data}
        loc = ext.get("location", "")
        if not loc or loc in ("dflash2.onnx.data", "embedding.onnx.data"):
            continue
        key = (loc, int(ext.get("offset", 0)), int(ext["length"]))
        if loc.endswith(".gguf"):
            gguf_slices.add(key)
        else:
            non_gguf_slices.add(key)

    return {
        "gguf_bytes": sum(ln for _loc, _off, ln in gguf_slices),
        "non_gguf_bytes": sum(ln for _loc, _off, ln in non_gguf_slices),
        "moe_bytes": sum(ln for _loc, _off, ln in moe_slices),
    }


def estimate_gguf_peak_vram_and_thresholds(
    total_weight_bytes: int,
    moe_bytes: int = 0,
    *,
    kv_cache_dtype: str = "int8",
    prefill_chunk_32k: int = 4096,
    prefill_chunk_long: int = 8192,
) -> dict:
    """Estimate peak GPU VRAM (`cudaMemGetInfo` used bytes) and `runtime_profiles` VRAM thresholds for GGUF packages.

    Calibrated against empirical H200 measurements across all three GGUF variants (INT8 KV cache), and adjusted
    for final KV cache element width (`fp16`/`bf16` adds 12,288 bytes/token across 12 QSA layers, ~3.00 GiB at 256K)
    and configured prefill chunk sizes.
    """
    kv_extra_per_token = 12_288 if kv_cache_dtype in ("fp16", "bf16") else 0
    chunk_32k_extra = max(0, ((prefill_chunk_32k - 4096) * 2_190_000_000) // 4096)
    chunk_long_extra = max(0, ((prefill_chunk_long - 8192) * 2_190_000_000) // 4096)

    peak_32k_bytes = total_weight_bytes + 6_160_000_000 + 33_032 * kv_extra_per_token + chunk_32k_extra
    peak_32k_c8192_bytes = total_weight_bytes + 8_350_000_000 + 33_032 * kv_extra_per_token
    peak_64k_bytes = total_weight_bytes + 6_535_000_000 + 65_800 * kv_extra_per_token + chunk_long_extra
    peak_128k_bytes = total_weight_bytes + 7_720_000_000 + 131_080 * kv_extra_per_token + chunk_long_extra
    peak_256k_bytes = total_weight_bytes + 10_610_000_000 + 262_152 * kv_extra_per_token + chunk_long_extra

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
        "kv_cache_dtype": kv_cache_dtype,
        "prefill_chunk_32k": prefill_chunk_32k,
        "prefill_chunk_long": prefill_chunk_long,
        "min_num_blocks_32k": GGUF_MIN_NUM_BLOCKS_32K,
        "min_num_blocks_64k": GGUF_MIN_NUM_BLOCKS_64K,
        "min_num_blocks_128k": GGUF_MIN_NUM_BLOCKS_128K,
        "min_num_blocks_256k": GGUF_MIN_NUM_BLOCKS_256K,
        "num_blocks_32k": GGUF_NUM_BLOCKS_32K,
        "num_blocks_64k": GGUF_NUM_BLOCKS_64K,
        "num_blocks_128k": GGUF_NUM_BLOCKS_128K,
        "num_blocks_256k": GGUF_NUM_BLOCKS_256K,
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


def default_gguf_runtime_profiles(
    total_weight_bytes: int,
    decoder_filename: str | None = "model.onnx",
    *,
    kv_cache_dtype: str = "int8",
    prefill_chunk_32k: int = 4096,
    prefill_chunk_long: int = 8192,
) -> list[dict]:
    """Hardware-matched `runtime_profiles` sizing `num_blocks` (`32K`/`64K`/`128K`/`256K`) to the GGUF model's VRAM footprint."""
    vram_info = estimate_gguf_peak_vram_and_thresholds(
        total_weight_bytes,
        kv_cache_dtype=kv_cache_dtype,
        prefill_chunk_32k=prefill_chunk_32k,
        prefill_chunk_long=prefill_chunk_long,
    )
    min_32k = vram_info["min_vram_32k_bytes"]
    min_64k = vram_info["min_vram_64k_bytes"]
    min_128k = vram_info["min_vram_128k_bytes"]
    min_256k = vram_info["min_vram_256k_bytes"]

    def make_overlay(num_blocks: int, max_scheduled: int, chunk_size: int | None = None) -> dict:
        ov: dict[str, object] = {}
        if decoder_filename:
            ov["model"] = {"decoder": {"filename": decoder_filename}}
        ov["engine"] = {
            "dynamic_batching": {
                "num_blocks": num_blocks,
                "max_batch_size": 1,
                "max_scheduled_tokens": max_scheduled,
            }
        }
        if chunk_size is not None:
            ov["search"] = {"chunk_size": chunk_size}
        return ov

    return [
        {
            "id": "gpu-32k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_32k,
                "maximum_total_device_memory_bytes": min_64k - 1,
                "is_integrated": False,
            },
            "overlay": make_overlay(GGUF_NUM_BLOCKS_32K, prefill_chunk_32k, chunk_size=prefill_chunk_32k),
        },
        {
            "id": "gpu-64k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_64k,
                "maximum_total_device_memory_bytes": min_128k - 1,
                "is_integrated": False,
            },
            "overlay": make_overlay(GGUF_NUM_BLOCKS_64K, prefill_chunk_long),
        },
        {
            "id": "gpu-128k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_128k,
                "maximum_total_device_memory_bytes": min_256k - 1,
                "is_integrated": False,
            },
            "overlay": make_overlay(GGUF_NUM_BLOCKS_128K, prefill_chunk_long),
        },
        {
            "id": "gpu-256k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_256k,
                "is_integrated": False,
            },
            "overlay": make_overlay(GGUF_NUM_BLOCKS_256K, prefill_chunk_long),
        },
    ]
