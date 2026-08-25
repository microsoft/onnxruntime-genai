# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------

import copy
import os

import onnx_ir as ir
from quantization import QuantConfig


class MTPModel:
    """Base class for composite models with an optional MTP graph."""

    def make_mtp_init(self, config, extra_options):
        self.mtp_attrs = {
            "build": False,  # Whether the source checkpoint declares an MTP graph.
            "shared_initializers": [],  # Initializers shared by the decoder and MTP graph after saving.
            "shared_initializer_names": set(),  # Exact initializer names eligible for sharing.
            "shared_initializer_prefixes": (),  # Initializer name prefixes eligible for sharing.
            "io_dtype": None,  # MTP graph I/O dtype, independently configurable.
            "onnx_dtype": None,  # MTP graph weight dtype, independently configurable.
            "extra_options": None,  # Builder options inherited or overridden for MTP.
        }
        return copy.deepcopy(extra_options)

    def resolve_mtp_model_config(self, extra_options):
        mtp_quant_config_value = extra_options.get("mtp_quant_config")
        if mtp_quant_config_value is None:
            return

        inherited_options = {
            key: copy.deepcopy(extra_options[key]) for key in ("hf_token", "hf_remote") if key in extra_options
        }
        quant_config = (
            copy.deepcopy(mtp_quant_config_value)
            if isinstance(mtp_quant_config_value, QuantConfig)
            else QuantConfig.from_json(mtp_quant_config_value)
        )
        self.mtp_attrs["io_dtype"], self.mtp_attrs["onnx_dtype"] = quant_config.to_onnx_dtypes()
        inherited_options["_quant_config"] = quant_config
        self.mtp_attrs["extra_options"] = inherited_options

    def is_shared_initializer(self, name):
        return name in self.mtp_attrs["shared_initializer_names"] or name.startswith(
            self.mtp_attrs["shared_initializer_prefixes"]
        )

    def make_external_tensor(self, tensor, location, offset, length):
        return ir.ExternalTensor(
            location,
            offset,
            length,
            tensor.dtype,
            shape=tensor.shape,
            name=tensor.name,
            doc_string=tensor.doc_string,
            metadata_props=tensor.metadata_props,
            base_dir=tensor.base_dir,
        )

    def external_data_equal(self, path_a, offset_a, path_b, offset_b, length, chunk_size=1 << 22):
        with open(path_a, "rb") as file_a, open(path_b, "rb") as file_b:
            file_a.seek(offset_a)
            file_b.seek(offset_b)
            remaining = length
            while remaining:
                read_size = min(chunk_size, remaining)
                data_a = file_a.read(read_size)
                data_b = file_b.read(read_size)
                if len(data_a) != read_size or data_a != data_b:
                    return False
                remaining -= read_size
        return True

    def find_shared_initializers(self, source_model, target_model, source_data, target_data):
        source_data_name = os.path.basename(source_data)
        target_data_name = os.path.basename(target_data)
        source_info = {}
        for name, initializer in source_model.graph.initializers.items():
            tensor = initializer.const_value
            if (
                isinstance(tensor, ir.ExternalTensor)
                and os.fspath(tensor.location) == source_data_name
                and tensor.offset is not None
                and tensor.length is not None
                and self.is_shared_initializer(name)
            ):
                source_info[name] = (tensor.dtype.value, tuple(tensor.shape), tensor.offset, tensor.length)

        shared = {}
        for name, (data_type, dims, source_offset, source_length) in source_info.items():
            target_initializer = target_model.graph.initializers.get(name)
            if target_initializer is None:
                continue
            target_tensor = target_initializer.const_value
            if (
                not isinstance(target_tensor, ir.ExternalTensor)
                or os.fspath(target_tensor.location) != target_data_name
                or target_tensor.offset is None
                or target_tensor.length is None
                or target_tensor.dtype.value != data_type
                or tuple(target_tensor.shape) != dims
                or target_tensor.length != source_length
            ):
                continue
            if self.external_data_equal(
                source_data,
                source_offset,
                target_data,
                target_tensor.offset,
                source_length,
            ):
                shared[name] = (source_offset, source_length, target_tensor.offset)
        return source_info, shared

    def stage_shared_initializers(self, target_model, source_data_name, target_data, shared):
        target_data_name = os.path.basename(target_data)
        staged_data = target_data + ".tmp"
        kept = []
        for name, initializer in target_model.graph.initializers.items():
            tensor = initializer.const_value
            if (
                isinstance(tensor, ir.ExternalTensor)
                and os.fspath(tensor.location) == target_data_name
                and tensor.offset is not None
                and tensor.length is not None
                and name not in shared
            ):
                kept.append((tensor.offset, tensor.length, initializer))
        kept.sort(key=lambda value: value[0])

        with open(target_data, "rb") as source_file, open(staged_data, "wb") as target_file:
            new_offset = 0
            for old_offset, length, initializer in kept:
                source_file.seek(old_offset)
                remaining = length
                while remaining:
                    data = source_file.read(min(1 << 22, remaining))
                    if not data:
                        raise EOFError(
                            f"Unexpected end of {target_data_name} while copying initializer '{initializer.name}'."
                        )
                    target_file.write(data)
                    remaining -= len(data)
                initializer.const_value = self.make_external_tensor(
                    initializer.const_value,
                    target_data_name,
                    new_offset,
                    length,
                )
                new_offset += length

        for name, (source_offset, length, _) in shared.items():
            initializer = target_model.graph.initializers[name]
            initializer.const_value = self.make_external_tensor(
                initializer.const_value,
                source_data_name,
                source_offset,
                length,
            )
        return staged_data

    def replace_shared_initializer_files(self, target_model_path, target_data, staged_model, staged_data):
        backup_model = target_model_path + ".bak"
        backup_data = target_data + ".bak"
        try:
            os.replace(target_data, backup_data)
            os.replace(target_model_path, backup_model)
            os.replace(staged_data, target_data)
            os.replace(staged_model, target_model_path)
        except Exception as exc:
            rollback_errors = []
            for backup_path, original_path in ((backup_data, target_data), (backup_model, target_model_path)):
                if os.path.exists(backup_path):
                    try:
                        os.replace(backup_path, original_path)
                    except Exception as rollback_exc:
                        rollback_errors.append(rollback_exc)
            for staged_path in (staged_data, staged_model):
                if os.path.exists(staged_path):
                    os.remove(staged_path)
            if rollback_errors:
                raise RuntimeError("Failed to restore MTP files after replacement failure.") from exc
            return False

        os.remove(backup_data)
        os.remove(backup_model)
        return True

    def make_shared_initializer_config(self, source_info, shared, source_data_name):
        return [
            {
                "name": name,
                "data_file": source_data_name,
                "offset": str(source_offset),
                "length": str(length),
                "data_type": source_info[name][0],
                "shape": list(source_info[name][1]),
            }
            for name, (source_offset, length, _) in shared.items()
        ]

    def share_initializers(self, output_dir, source_file, target_file):
        source_model_path = os.path.join(output_dir, source_file)
        target_model_path = os.path.join(output_dir, target_file)
        source_data = source_model_path + ".data"
        target_data = target_model_path + ".data"
        required_paths = (source_model_path, target_model_path, source_data, target_data)
        if not all(os.path.exists(path) for path in required_paths):
            return []

        staged_model = target_model_path + ".tmp"
        staged_data = target_data + ".tmp"
        try:
            source_model = ir.load(source_model_path)
            target_model = ir.load(target_model_path)
            source_info, shared = self.find_shared_initializers(
                source_model, target_model, source_data, target_data
            )
            if not shared:
                return []
            self.stage_shared_initializers(target_model, os.path.basename(source_data), target_data, shared)
            ir.save(target_model, staged_model)
        except Exception as exc:
            for staged_path in (staged_data, staged_model):
                if os.path.exists(staged_path):
                    os.remove(staged_path)
            print(f"Warning: could not share MTP initializers ({exc}); duplicated copies remain in {target_data}.")
            return []

        if not self.replace_shared_initializer_files(target_model_path, target_data, staged_model, staged_data):
            print(f"Warning: could not commit shared MTP initializers; duplicated copies remain in {target_data}.")
            return []

        shared_size_mb = sum(length for _, length, _ in shared.values()) / 1e6
        print(f"Shared MTP initializers with the main model (saved {shared_size_mb:.0f} MB from {target_data}).")
        return self.make_shared_initializer_config(source_info, shared, os.path.basename(source_data))

    def add_shared_initializers_to_genai_config(self, genai_config):
        if not self.mtp_attrs["shared_initializers"]:
            return
        genai_config["model"]["decoder"]["shared_initializers"] = self.mtp_attrs["shared_initializers"]
        genai_config["model"]["mtp"]["shared_initializers"] = self.mtp_attrs["shared_initializers"]
