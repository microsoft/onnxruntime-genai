# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Parse existing component files into IR without constructing model graphs.

The explicit ONNX dependency is a source-format validation boundary, not a graph
construction API. Public ONNX IR loaders discard sparse initializers, training
graphs, and malformed attribute payloads. Validate their original external-data
metadata before IR conversion. ONNX IR also has no public schema checker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import onnx
import onnx_ir as ir


@dataclass(frozen=True)
class ComponentModelLoader:
    """Validate a pre-existing component source and expose its IR model."""

    model_path: Path
    component_name: str

    def load(self) -> ir.Model:
        """Parse source metadata without materializing external tensor data."""
        try:
            proto = onnx.load(self.model_path, load_external_data=False)
        except Exception as error:
            raise ValueError(
                f"component {self.component_name!r} source is not a readable ONNX model: "
                f"{self.model_path}"
            ) from error
        external_tensors = list(self.iter_external_tensors(proto))
        self.external_data_files(external_tensors)
        model = ir.serde.deserialize_model(proto)
        model.meta["builder_config.external_tensors"] = external_tensors
        return model

    def validate(self) -> None:
        """Check source structure/schema using the format's public checker."""
        try:
            onnx.checker.check_model(self.model_path)
        except Exception as error:
            raise ValueError(
                f"component {self.component_name!r} source is not a valid ONNX model: "
                f"{self.model_path}"
            ) from error

    def external_data_files(
        self, tensors: list[ir.ExternalTensor]
    ) -> list[tuple[str, Path]]:
        """Check source-relative locations and complete tensor byte ranges."""
        source_root = self.model_path.parent.resolve()
        files = {}
        for tensor in tensors:
            location = str(tensor.location)
            if not location:
                raise ValueError(
                    f"component {self.component_name!r} has external tensor data without a location"
                )
            relative = Path(location)
            if (
                relative.is_absolute()
                or "\\" in location
                or ":" in location
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise ValueError(
                    f"component {self.component_name!r} has unsafe external data location {location!r}"
                )
            source = (source_root / relative).resolve()
            try:
                source.relative_to(source_root)
            except ValueError as error:
                raise ValueError(
                    f"component {self.component_name!r} has unsafe external data location {location!r}"
                ) from error
            if not source.is_file():
                raise ValueError(
                    f"component {self.component_name!r} external data file does not exist: {location}"
                )
            offset = tensor.offset if tensor.offset is not None else 0
            required_bytes = math.ceil(
                math.prod(tensor.shape.dims) * tensor.dtype.itemsize
            )
            length = tensor.length if tensor.length is not None else required_bytes
            if offset < 0 or length < required_bytes:
                raise ValueError(
                    f"component {self.component_name!r} external data range is too small"
                )
            if offset + length > source.stat().st_size:
                raise ValueError(
                    f"component {self.component_name!r} external data range exceeds "
                    f"{location!r}"
                )
            files[location] = source
        return list(files.items())

    def iter_external_tensors(self, message):
        """Visit all source tensors, including unsupported/malformed payloads."""
        if isinstance(message, onnx.TensorProto):
            entries = {entry.key: entry.value for entry in message.external_data}
            if message.data_location != onnx.TensorProto.EXTERNAL and not entries:
                return
            location = entries.get("location")
            if not location:
                raise ValueError(
                    f"component {self.component_name!r} has external tensor data without a location"
                )
            ranges = {}
            for key in ("offset", "length"):
                try:
                    ranges[key] = int(entries[key]) if key in entries else None
                except ValueError as error:
                    raise ValueError(
                        f"component {self.component_name!r} external data {key} is invalid"
                    ) from error
            yield ir.ExternalTensor(
                location,
                offset=ranges["offset"],
                length=ranges["length"],
                dtype=ir.DataType(message.data_type),
                shape=ir.Shape(message.dims),
                name=message.name,
            )
            return
        for field, value in message.ListFields():
            if field.type != field.TYPE_MESSAGE:
                continue
            children = value if field.is_repeated else (value,)
            for child in children:
                yield from self.iter_external_tensors(child)
