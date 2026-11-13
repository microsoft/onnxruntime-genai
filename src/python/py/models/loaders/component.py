# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Parse existing component files into IR without constructing model graphs.

The explicit ONNX dependency is a source-format validation boundary, not a graph
construction API. Public ONNX IR loaders discard sparse initializers, training
graphs, and malformed attribute payloads. Validate their original external-data
metadata before IR conversion. ONNX IR also has no public schema checker.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import ClassVar

import onnx
import onnx_ir as ir


@dataclass(frozen=True)
class ComponentModelLoader:
    """Validate a pre-existing component source and expose its IR model."""

    schema_validation_lock: ClassVar[RLock] = RLock()
    model_path: Path
    component_name: str

    def load(self, *, reject_sparse_initializers: bool = False) -> ir.Model:
        """Parse source metadata without materializing external tensor data."""
        try:
            proto = onnx.load(self.model_path, load_external_data=False)
        except Exception as error:
            raise ValueError(
                f"component {self.component_name!r} source is not a readable ONNX model: "
                f"{self.model_path}"
            ) from error
        if reject_sparse_initializers and proto.graph.sparse_initializer:
            raise ValueError(f"component {self.component_name!r} source uses unsupported sparse initializers.")
        external_tensors = list(self.iter_external_tensors(proto))
        self.external_data_files(external_tensors)
        model = ir.serde.deserialize_model(proto)
        model.meta["builder_config.external_tensors"] = external_tensors
        return model

    def validate(self, *, allow_ort_extensions: bool = False) -> None:
        """Check source structure/schema using the format's public checker."""
        with self.schema_validation_lock:
            registered = []
            try:
                if allow_ort_extensions:
                    ort_schema = importlib.import_module("onnxruntime.capi._pybind_state")

                    source = onnx.load(self.model_path, load_external_data=False)
                    opset = next((entry.version for entry in source.opset_import if entry.domain == ""), 1)
                    missing = {
                        node.op_type for node in source.graph.node
                        if node.domain == "" and not onnx.defs.has(node.op_type, opset, "")
                    }
                    schemas = ort_schema.get_all_operator_schema() if missing else []
                    for name in sorted(missing):
                        candidates = [schema for schema in schemas if schema.name == name and schema.domain == "" and schema.since_version <= opset]
                        if candidates:
                            schema = self.make_onnx_operator_schema(max(candidates, key=lambda item: item.since_version))
                            onnx.defs.register_schema(schema)
                            registered.append(schema)
                onnx.checker.check_model(self.model_path)
            except Exception as error:
                raise ValueError(
                    f"component {self.component_name!r} source is not a valid ONNX model: "
                    f"{self.model_path}"
                ) from error
            finally:
                for schema in reversed(registered):
                    onnx.defs.deregister_schema(schema.name, schema.since_version, schema.domain)

    def make_onnx_operator_schema(self, source):
        schema_type = onnx.defs.OpSchema

        def formal_parameter(parameter):
            return schema_type.FormalParameter(
                parameter.name, parameter.typeStr, parameter.description,
                param_option=schema_type.FormalParameterOption(parameter.option.value),
            )

        return schema_type(
            source.name, source.domain, source.since_version,
            inputs=[formal_parameter(parameter) for parameter in source.inputs],
            outputs=[formal_parameter(parameter) for parameter in source.outputs],
            type_constraints=[
                (constraint.type_param_str, constraint.allowed_type_strs, constraint.description)
                for constraint in source.type_constraints
            ],
            attributes=[
                schema_type.Attribute(name, schema_type.AttrType(attribute.type.value), attribute.description, required=attribute.required)
                for name, attribute in source.attributes.items()
            ],
        )

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
