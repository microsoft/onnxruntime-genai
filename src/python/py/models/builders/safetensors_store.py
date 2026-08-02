# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Streaming weight I/O for very large checkpoints.

Building a 150 GiB ONNX model the usual way needs the whole checkpoint resident
plus a second full copy inside the ``ModelProto`` before anything is written.
The two classes here remove both copies so a model can be emitted layer by
layer:

* :class:`SafeTensorStore` reads tensors out of the original safetensors shards
  one at a time and never caches them, so the resident set is whatever the
  caller is currently holding.
* :class:`ExternalDataWriter` appends every transformed weight to the ONNX
  external-data blob the moment it is produced, so the builder only ever holds
  the tensor it is working on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import onnx_ir as ir
import torch

_F8_E8M0 = getattr(torch, "float8_e8m0fnu", None)

TORCH_TO_IR: dict[torch.dtype, ir.DataType] = {
    torch.bool: ir.DataType.BOOL,
    torch.uint8: ir.DataType.UINT8,
    torch.int8: ir.DataType.INT8,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.float16: ir.DataType.FLOAT16,
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
}
if _F8_E8M0 is not None:
    TORCH_TO_IR[_F8_E8M0] = ir.DataType.FLOAT8E8M0


def torch_bytes(tensor: torch.Tensor) -> memoryview:
    """Raw little-endian bytes of ``tensor``, without copying."""
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().data


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode an E8M0 (bare exponent) scale tensor to float32."""
    return torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)


class ExternalDataWriter:
    """Append-only writer for an ONNX external-data blob.

    ``add`` returns an ``ir.ExternalTensor`` referencing the bytes just written,
    so the caller can drop its copy immediately.
    """

    ALIGN = 64

    def __init__(self, path: str | os.PathLike):
        self.path = os.path.abspath(str(path))
        self.location = os.path.basename(self.path)
        self.base_dir = os.path.dirname(self.path)
        os.makedirs(self.base_dir, exist_ok=True)
        self._file = open(self.path, "wb")
        self._pos = 0

    def add(self, tensor: torch.Tensor, name: str, dtype: ir.DataType | None = None) -> ir.ExternalTensor:
        shape = tuple(tensor.shape)
        buf = torch_bytes(tensor)
        pad = -self._pos % self.ALIGN
        if pad:
            self._file.write(b"\0" * pad)
            self._pos += pad
        self._file.write(buf)
        tensor_ref = ir.ExternalTensor(
            self.location,
            self._pos,
            buf.nbytes,
            dtype if dtype is not None else TORCH_TO_IR[tensor.dtype],
            shape=ir.Shape(shape),
            name=name,
            base_dir=self.base_dir,
        )
        self._pos += buf.nbytes
        return tensor_ref

    def flush(self):
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def move_to(self, out_dir: str | os.PathLike):
        """Move the blob next to the model file. ``location`` is a bare
        basename, so every reference stays valid."""
        self.close()
        dst = os.path.join(os.path.abspath(str(out_dir)), self.location)
        if dst != self.path:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(self.path, dst)
            self.path = dst
            self.base_dir = os.path.dirname(dst)


class SafeTensorStore(Mapping):
    """Read-only ``Mapping`` view of a sharded safetensors checkpoint."""

    def __init__(self, model_dir: str | os.PathLike):
        self.dir = os.path.abspath(str(model_dir))
        self._handles: dict[str, object] = {}
        index = os.path.join(self.dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self._shard_of = json.load(f)["weight_map"]
        else:
            single = "model.safetensors"
            self._shard_of = dict.fromkeys(self._open(single).keys(), single)

    def _open(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(os.path.join(self.dir, shard), framework="pt")
            self._handles[shard] = handle
        return handle

    # -- Mapping ---------------------------------------------------------- #

    def __iter__(self):
        return iter(self._shard_of)

    def __len__(self):
        return len(self._shard_of)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._open(self._shard_of[name]).get_tensor(name)

    # -- block-scaled fp8 ------------------------------------------------- #

    def dequant(self, name: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16,
                block: int = 128) -> torch.Tensor:
        """Load ``name``, undoing 2-D block-scaled fp8 quantization if a sibling
        ``.scale`` tensor exists."""
        scale_key = name[: -len("weight")] + "scale" if name.endswith("weight") else None
        if scale_key is None or scale_key not in self._shard_of:
            return self[name].to(device=device, dtype=dtype)
        w = self[name].to(device=device, dtype=torch.float32)
        s = e8m0_to_float(self[scale_key].to(device))
        for axis in range(w.ndim):
            s = s.repeat_interleave(block, axis).narrow(axis, 0, w.shape[axis])
        return (w * s).to(dtype)
