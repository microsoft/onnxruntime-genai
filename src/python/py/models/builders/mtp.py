# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------

import copy
import os
import shutil

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

    def export_indexshare_graphs(self, output_dir, source_file, max_draft_tokens=7):
        if not 1 <= max_draft_tokens <= 7:
            raise ValueError("IndexShare export requires between one and seven draft tokens.")
        source = ir.load(os.path.join(output_dir, source_file))
        indexers = [node for node in source.graph if node.op_type == "PackedSparseAttentionIndexer"]
        if len(indexers) != 1:
            raise ValueError("IndexShare export currently requires exactly one packed QSA indexer.")
        indexer = indexers[0]
        if any(value is not None for value in indexer.inputs[18:]):
            raise ValueError("IndexShare export requires a graph that has not already been converted.")
        attributes = indexer.attributes
        if "policy_mode" not in attributes or attributes["policy_mode"].as_string() != "qsa":
            raise ValueError("IndexShare MTP export requires QSA raw-token selections.")
        capacity = attributes["token_budget"].as_int() + attributes["compress_ratio"].as_int() - 1
        attention_nodes = [
            node
            for node in source.graph
            if node.op_type == "SparsePagedAttention" and tuple(node.inputs[9:11]) == tuple(indexer.outputs[:2])
        ]
        if len(attention_nodes) != 1:
            raise ValueError("The packed indexer must feed exactly one SparsePagedAttention node.")
        past_lengths = indexer.inputs[7]
        if past_lengths not in source.graph.inputs or past_lengths.dtype != ir.DataType.INT32:
            raise ValueError("IndexShare requires packed int32 past sequence lengths.")
        return self.export_packed_indexshare_graph(
            source, indexer, capacity, output_dir, source_file, max_draft_tokens
        )

    def export_packed_indexshare_graph(self, source, indexer, capacity, output_dir, source_file, max_draft_tokens):
        model = source
        graph = model.graph
        existing_names = {value.name for value in graph.inputs}
        existing_names.update(graph.initializers)
        existing_names.update(value.name for node in graph for value in node.outputs)
        existing_names.update(value.name for value in graph.outputs)
        prefix = (indexer.name or "").rsplit("/", 1)[0] + "/indexshare"
        gated_hidden = prefix + "/GatherProjectionRows/output_0"
        merged = [
            prefix + "/IndexerMerge/output_0",
            prefix + "/IndexerMerge/output_1",
            prefix + "/IndexerMerge/status",
        ]
        if existing_names.intersection({
            "past_indices", "past_counts", "present_indices", "present_counts", "indexer_status",
            "mode", "projection_rows", "base_row_indices", "range_starts", "range_ends",
            gated_hidden, *merged,
        }):
            raise ValueError("IndexShare graph input/output names must be unused.")
        query_producer = indexer.inputs[0].producer()
        split = None
        if len(indexer.inputs) > 1 and indexer.inputs[1] is not None:
            if (
                query_producer is None
                or query_producer.op_type != "Split"
                or tuple(query_producer.outputs) != tuple(indexer.inputs[:2])
            ):
                raise ValueError("Packed IndexShare requires a shared Q/K projection with an optional Split.")
            split = query_producer
            packed_qk = split.inputs[0]
        else:
            packed_qk = indexer.inputs[0]
        projection = packed_qk.producer()
        if projection is None or projection.op_type not in {"MatMul", "MatMulNBits"}:
            raise ValueError("Packed IndexShare requires an indexer MatMul or MatMulNBits projection.")
        projection_consumers = {use.node for use in packed_qk.uses()}
        if projection_consumers != {split if split is not None else indexer} or packed_qk.is_graph_output():
            raise ValueError("Packed IndexShare cannot gate a projection with other consumers.")
        if split is not None:
            consumers = {use.node for value in split.outputs for use in value.uses()}
            if consumers != {indexer} or any(value.is_graph_output() for value in split.outputs):
                raise ValueError("Packed IndexShare cannot remove a Split with other consumers.")

        inputs = {}
        for name, dtype, shape in (
            ("mode", ir.DataType.INT32, [1]),
            ("projection_rows", ir.DataType.INT64, ["indexer_projection_rows"]),
            ("past_indices", ir.DataType.INT32, ["indexshare_rows", capacity]),
            ("past_counts", ir.DataType.INT32, ["indexshare_rows"]),
            ("base_row_indices", ir.DataType.INT32, ["indexshare_merge_queries"]),
            ("range_starts", ir.DataType.INT32, ["indexshare_merge_queries"]),
            ("range_ends", ir.DataType.INT32, ["indexshare_merge_queries"]),
        ):
            inputs[name] = ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))
        output_capacity = capacity + max_draft_tokens - 1
        hidden = projection.inputs[0]
        gather_shape = hidden.shape.copy() if hidden.shape is not None else None
        if gather_shape is not None:
            gather_shape[0] = "indexer_projection_rows"
        gather = ir.Node(
            "",
            "Gather",
            [hidden, inputs["projection_rows"]],
            {"axis": ir.AttrInt64("axis", 0)},
            outputs=[ir.Value(name=gated_hidden, type=hidden.type, shape=gather_shape)],
            name=prefix + "/GatherProjectionRows",
        )
        merge_shapes = [
            ["indexshare_merge_queries", output_capacity],
            ["indexshare_merge_queries"],
            ["indexshare_merge_queries"],
        ]
        merge = ir.Node(
            "com.microsoft",
            "PackedSparseAttentionIndexerMerge",
            [inputs[name] for name in ("past_indices", "past_counts", "base_row_indices", "range_starts", "range_ends")],
            {
                "policy_mode": ir.AttrString("policy_mode", "append_range"),
                "max_output_entries": ir.AttrInt64("max_output_entries", output_capacity),
            },
            outputs=[
                ir.Value(name=name, type=ir.TensorType(ir.DataType.INT32), shape=ir.Shape(shape))
                for name, shape in zip(merged, merge_shapes, strict=True)
            ],
            name=prefix + "/IndexerMerge",
        )
        projection.replace_input_with(0, gather.outputs[0])
        if packed_qk.shape is not None:
            packed_qk.shape = packed_qk.shape.copy()
            packed_qk.shape[0] = "indexer_projection_rows"
        # Rebuild the node: resizing inputs/outputs is unavailable in onnx_ir 0.1.3.
        indexer_inputs = list(indexer.inputs[:18])
        indexer_inputs.extend([None] * (18 - len(indexer_inputs)))
        indexer_inputs[:2] = [packed_qk, None]
        indexer_inputs.extend([inputs["mode"], *merge.outputs])
        indexer_outputs = []
        for value in indexer.outputs:
            replacement = ir.Value(
                name=value.name,
                type=copy.deepcopy(value.type),
                shape=value.shape.copy() if value.shape is not None else None,
                doc_string=value.doc_string,
            )
            replacement.metadata_props.update(value.metadata_props)
            indexer_outputs.append(replacement)
        indexer_outputs.extend(ir.Value(name="") for _ in range(7 - len(indexer_outputs)))
        indexer_outputs.append(ir.Value(name="indexer_status"))
        replacement_indexer = ir.Node(
            indexer.domain,
            indexer.op_type,
            indexer_inputs,
            {
                **indexer.attributes,
                "max_output_entries": ir.AttrInt64("max_output_entries", output_capacity),
            },
            outputs=indexer_outputs,
            name=indexer.name,
            overload=indexer.overload,
            version=indexer.version,
            doc_string=indexer.doc_string,
            metadata_props=dict(indexer.metadata_props),
            **(
                {"device_configurations": indexer.device_configurations}
                if hasattr(indexer, "device_configurations") else {}
            ),
        )
        replacements = dict(zip(indexer.outputs, replacement_indexer.outputs, strict=False))
        graph.outputs[:] = [
            replacements.get(value, value) for value in graph.outputs if value not in indexer.outputs[:2]
        ]
        ir.convenience.replace_all_uses_with(
            indexer.outputs, replacement_indexer.outputs[:len(indexer.outputs)]
        )
        for value, name, shape in (
            (replacement_indexer.outputs[0], "present_indices", ["num_tokens", output_capacity]),
            (replacement_indexer.outputs[1], "present_counts", ["num_tokens"]),
            (replacement_indexer.outputs[7], "indexer_status", ["num_tokens"]),
        ):
            value.name = name
            value.dtype = ir.DataType.INT32
            if value.shape is not None and len(value.shape) == len(shape):
                value.shape = value.shape.copy()
                for axis, dim in enumerate(shape):
                    value.shape[axis] = dim
            else:
                value.shape = ir.Shape(shape)
            graph.outputs.append(value)
        graph.inputs.extend(inputs.values())
        graph.insert_before(projection, gather)
        graph.insert_before(indexer, [merge, replacement_indexer])
        graph.remove(indexer, safe=True)
        if split is not None:
            graph.remove(split, safe=True)
        self.remove_unused_constants(graph)
        # This in-place graph rewrite keeps its already private data file unchanged.
        ir.save(model, os.path.join(output_dir, source_file))
        return {
            "enabled": True,
            "base_capacity": capacity,
            "max_draft_tokens": max_draft_tokens,
            "indices_output": "present_indices",
            "counts_output": "present_counts",
            "status_output": "indexer_status",
        }

    def remove_unused_constants(self, graph):
        for node in tuple(graph):
            if node.op_type == "Constant" and not any(value.uses() or value.is_graph_output() for value in node.outputs):
                graph.remove(node, safe=True)
        for name, value in tuple(graph.initializers.items()):
            if not value.uses() and not value.is_graph_output() and not value.is_graph_input():
                del graph.initializers[name]

    def resolve_mtp_model_config(self, extra_options):
        mtp_quant_config_value = extra_options.get("mtp_quant_config")
        if mtp_quant_config_value is None:
            return

        inherited_options = {
            key: copy.deepcopy(extra_options[key])
            for key in ("hf_token", "hf_remote", "shared_embeddings")
            if key in extra_options
        }
        quant_config = (
            copy.deepcopy(mtp_quant_config_value)
            if isinstance(mtp_quant_config_value, QuantConfig)
            else QuantConfig.from_json(mtp_quant_config_value)
        )
        if (
            quant_config.weights.type == "int2"
            or quant_config.moe.type == "int2"
            or any(override.type == "int2" for override in quant_config.weights.overrides)
        ):
            raise ValueError("MTP quant_config does not support int2; use DFlash2")
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

    def find_shared_initializers(
        self,
        source_model,
        target_model,
        source_data,
        target_data,
        adopt_source_initializers=frozenset(),
        excluded_source_initializers=frozenset(),
    ):
        source_data_name = os.path.basename(source_data)
        target_data_name = os.path.basename(target_data)
        source_data_size = os.path.getsize(source_data)
        target_data_size = os.path.getsize(target_data)
        source_info = {}
        for name, initializer in source_model.graph.initializers.items():
            tensor = initializer.const_value
            if (
                isinstance(tensor, ir.ExternalTensor)
                and os.fspath(tensor.location) == source_data_name
                and tensor.offset is not None
                and tensor.length is not None
                and tensor.offset >= 0
                and tensor.length >= 0
                and tensor.offset <= source_data_size
                and tensor.length <= source_data_size - tensor.offset
                and name not in excluded_source_initializers
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
                or target_tensor.offset < 0
                or target_tensor.length < 0
                or target_tensor.offset > target_data_size
                or target_tensor.length > target_data_size - target_tensor.offset
                or target_tensor.dtype.value != data_type
                or tuple(target_tensor.shape) != dims
                or target_tensor.length != source_length
            ):
                continue
            if name in adopt_source_initializers or self.external_data_equal(
                source_data, source_offset, target_data, target_tensor.offset, source_length
            ):
                shared[name] = (source_offset, source_length, target_tensor.offset)
        return source_info, shared

    def stage_shared_initializers(self, target_model, source_data, target_data, shared):
        # Preserve offsets for every tensor, including attributes and subgraphs.
        # Runtime sharing is configured separately; graph references remain private.
        staged_data = target_data + ".tmp"
        target_data_size = os.path.getsize(target_data)
        tensors = [
            initializer.const_value
            for graph in target_model.graphs()
            for initializer in graph.initializers.values()
        ]
        for graph in target_model.graphs():
            for node in graph:
                for attribute in node.attributes.values():
                    if not attribute.is_ref() and attribute.type == ir.AttributeType.TENSOR:
                        tensors.append(attribute.as_tensor())
                    elif not attribute.is_ref() and attribute.type == ir.AttributeType.TENSORS:
                        tensors.extend(attribute.as_tensors())
        for tensor in tensors:
            if isinstance(tensor, ir.ExternalTensor) and os.fspath(tensor.location) == os.path.basename(target_data):
                if (
                    tensor.offset is None or tensor.length is None
                    or tensor.offset < 0 or tensor.length < 0
                    or tensor.offset > target_data_size
                    or tensor.length > target_data_size - tensor.offset
                ):
                    raise EOFError(f"Invalid external tensor range in {target_data}: {tensor.name}")
        shutil.copyfile(target_data, staged_data)
        with open(source_data, "rb") as source_file, open(staged_data, "r+b") as target_file:
            for name, (source_offset, length, target_offset) in shared.items():
                source_file.seek(source_offset)
                target_file.seek(target_offset)
                remaining = length
                while remaining:
                    data = source_file.read(min(1 << 22, remaining))
                    if not data:
                        raise EOFError(f"Unexpected end of {source_data} while copying initializer '{name}'.")
                    target_file.write(data)
                    remaining -= len(data)
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
                raise RuntimeError("Failed to restore auxiliary model files after replacement failure.") from exc
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

    def share_initializers(
        self,
        output_dir,
        source_file,
        target_file,
        adopt_source_initializers=frozenset(),
        required_source_initializers=frozenset(),
        excluded_source_initializers=frozenset(),
    ):
        """Configure runtime sharing while keeping each serialized graph self-contained.

        Initializers normally share only when their bytes match. Names in
        ``adopt_source_initializers`` instead use the source tensor whenever dtype, shape, and
        byte length match; callers must establish that those tensors are semantically identical.
        Names in ``excluded_source_initializers`` always keep the auxiliary model's private copy.
        """
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
                source_model,
                target_model,
                source_data,
                target_data,
                adopt_source_initializers,
                excluded_source_initializers,
            )
            missing = required_source_initializers - shared.keys()
            if missing:
                raise ValueError("Required shared initializers are unavailable: " + ", ".join(sorted(missing)))
            if not shared:
                return []
            self.stage_shared_initializers(target_model, source_data, target_data, shared)
            ir.save(target_model, staged_model)
        except Exception as exc:
            for staged_path in (staged_data, staged_model):
                if os.path.exists(staged_path):
                    os.remove(staged_path)
            print(
                f"Warning: could not share auxiliary initializers ({exc}); duplicated copies remain in {target_data}."
            )
            return []

        if not self.replace_shared_initializer_files(target_model_path, target_data, staged_model, staged_data):
            print(
                f"Warning: could not commit shared auxiliary initializers; duplicated copies remain in {target_data}."
            )
            return []

        shared_size_mb = sum(length for _, length, _ in shared.values()) / 1e6
        print(f"Configured {shared_size_mb:.0f} MB of runtime initializer sharing; {target_data} remains self-contained.")
        return self.make_shared_initializer_config(source_info, shared, os.path.basename(source_data))

    def add_shared_initializers_to_genai_config(self, genai_config):
        if not self.mtp_attrs["shared_initializers"]:
            return
        genai_config["model"]["decoder"]["shared_initializers"] = self.mtp_attrs["shared_initializers"]
        genai_config["model"]["mtp"]["shared_initializers"] = self.mtp_attrs["shared_initializers"]
