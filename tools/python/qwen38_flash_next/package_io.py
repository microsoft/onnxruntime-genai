# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Checkpoint, ONNX initializer, context-tier and file helpers shared by `build_package.py` and `precision_options.py`.

This module imports neither script, so both can use it without an import cycle.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from collections.abc import Callable
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


class HFCheckpointReader:
    def __init__(self, hf_dir: Path) -> None:
        self.hf_dir = hf_dir
        idx_path = hf_dir / "model.safetensors.index.json"
        if idx_path.exists():
            self.weight_map: dict[str, str] = json.loads(idx_path.read_text())["weight_map"]
        else:
            self.weight_map = {}
        self._header_cache: dict[str, tuple[int, dict]] = {}

    def _get_header(self, filename: str) -> tuple[int, dict]:
        if filename not in self._header_cache:
            path = self.hf_dir / filename
            with open(path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len).decode("utf-8"))
            self._header_cache[filename] = (8 + hdr_len, hdr)
        return self._header_cache[filename]

    def read_raw_tensor(self, name: str) -> tuple[str, list[int], bytes]:
        shard = self.weight_map[name]
        base_off, hdr = self._get_header(shard)
        info = hdr[name]
        start, end = info["data_offsets"]
        with open(self.hf_dir / shard, "rb") as f:
            f.seek(base_off + start)
            raw = f.read(end - start)
        return info["dtype"], info["shape"], raw

    def read_bf16_tensor(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        if name in self.weight_map:
            dtype, shape, raw = self.read_raw_tensor(name)
            if dtype != "BF16":
                raise ValueError(f"Expected BF16 for {name}, got {dtype}")
            arr = np.frombuffer(raw, dtype=np.uint16).copy()
            return torch.from_numpy(arr).view(torch.bfloat16).reshape(shape).to(device)
        if name + "_scale_inv" in self.weight_map:
            dtype_w, shape_w, raw_w = self.read_raw_tensor(name)
            dtype_s, shape_s, raw_s = self.read_raw_tensor(name + "_scale_inv")
            if dtype_w != "F8_E4M3":
                raise ValueError(f"Expected F8_E4M3 for {name}, got {dtype_w}")
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
        raise KeyError(f"Tensor {name} not found in {self.hf_dir}")


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


def read_gguf_tensor_infos(path: Path) -> tuple[list[dict], int]:
    """Return the tensor infos (`name`, `dims`, `ggml_type`, absolute `data_offset`) of the GGUF file `path` and its data offset."""
    with open(path, "rb") as f:

        def read(fmt: str):
            size = struct.calcsize(fmt)
            buf = f.read(size)
            if len(buf) != size:
                raise ValueError(f"Truncated GGUF header in {path}")
            return struct.unpack(fmt, buf)

        def read_str() -> str:
            (n,) = read("<Q")
            return f.read(n).decode("utf-8", errors="replace")

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

        def read_value(vtype: int) -> object:
            if vtype in scalar_formats:
                return read(scalar_formats[vtype])[0]
            if vtype == 8:
                return read_str()
            if vtype == 9:
                atype, alen = read("<IQ")
                if atype in scalar_formats:
                    f.seek(alen * struct.calcsize(scalar_formats[atype]), os.SEEK_CUR)
                else:
                    for _ in range(alen):
                        read_value(atype)
                return None
            raise ValueError(f"Unknown GGUF value type {vtype} in {path}")

        magic, _version, n_tensors, n_kv = read("<4sIQQ")
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        kvs: dict[str, object] = {}
        for _ in range(n_kv):
            key = read_str()
            (vtype,) = read("<I")
            kvs[key] = read_value(vtype)
        tensors = []
        for _ in range(n_tensors):
            name = read_str()
            (n_dims,) = read("<I")
            dims = [read("<Q")[0] for _ in range(n_dims)]
            ggml_type, rel_offset = read("<IQ")
            tensors.append({"name": name, "dims": dims, "ggml_type": ggml_type, "rel_offset": rel_offset})
        alignment = int(kvs.get("general.alignment", 32))
        data_offset = align_up(f.tell(), alignment)
    for t in tensors:
        t["data_offset"] = data_offset + t["rel_offset"]
    return tensors, data_offset


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
