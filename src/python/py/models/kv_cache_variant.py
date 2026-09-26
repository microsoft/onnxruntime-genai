# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create a KV-cache quantization variant that reuses an ONNX graph's external weights."""

import json
import re
from pathlib import Path

import numpy as np
import onnx
from quantization import KV_CACHE_CALIBRATION_QMAX


class KVCacheVariant:
    """Create a static per-channel KV-cache variant without duplicating external model weights.

    KV scales are recalculated from the scale file and stored inside the variant ONNX file. Every
    other initializer keeps the source graph's external-data location, offset, and length.
    """

    def __init__(self, scheme: str):
        supported_schemes = ("int4_per_channel", "int8_per_channel")
        if scheme not in supported_schemes:
            raise ValueError(f"scheme must be one of {list(supported_schemes)}, got {scheme!r}.")
        self.target_bits = 4 if scheme == "int4_per_channel" else 8
        self.layer_pattern = re.compile(r"(?:^|/)model/layers\.(\d+)/attn/PagedAttention$")

    def create(self, source_model_path: str | Path, output_model_path: str | Path, scale_file: str | Path) -> None:
        source_model_path = Path(source_model_path)
        output_model_path = Path(output_model_path)
        if source_model_path.resolve().parent != output_model_path.resolve().parent:
            raise ValueError("Source and output graphs must share a directory to reuse external weights.")
        if source_model_path.resolve() == output_model_path.resolve():
            raise ValueError("Output graph must not overwrite the source graph.")
        scale_data = self.load_scale_data(Path(scale_file), source_model_path.stem)
        model = onnx.load(source_model_path, load_external_data=False)

        layer_ids = self.update_paged_attention_nodes(model)
        scale_indices = self.validate_scale_layers(scale_data, layer_ids)
        head_size = self.update_scale_initializers(model, scale_data, scale_indices)
        self.update_cache_values(model, layer_ids, head_size)

        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_model_path)

    def load_scale_data(self, scale_path: Path, scale_section: str) -> dict:
        with scale_path.open(encoding="utf-8") as stream:
            scale_data = json.load(stream)

        if scale_section in scale_data:
            scale_data = scale_data[scale_section]

        try:
            scale_data["scales"]["k_scales"]
            scale_data["scales"]["v_scales"]
        except (KeyError, TypeError) as error:
            raise ValueError("Scales file must contain scales.k_scales and scales.v_scales.") from error
        return scale_data

    def update_paged_attention_nodes(self, model: onnx.ModelProto) -> list[int]:
        layer_ids = []
        for node in model.graph.node:
            if node.op_type != "PagedAttention" or node.domain != "com.microsoft":
                continue
            match = self.layer_pattern.search(node.name)
            if not match:
                raise ValueError(f"Cannot determine the layer ID from PagedAttention node {node.name!r}.")
            attributes = {attribute.name: attribute for attribute in node.attribute}
            if attributes.get("k_quant_type") is None or attributes.get("v_quant_type") is None:
                raise ValueError(f"PagedAttention node {node.name!r} is not statically KV-cache quantized.")
            if attributes["k_quant_type"].s != b"PER_CHANNEL" or attributes["v_quant_type"].s != b"PER_CHANNEL":
                raise ValueError(f"PagedAttention node {node.name!r} does not use per-channel KV quantization.")

            cache_dtype_names = ("k_cache_dtype", "v_cache_dtype")
            del node.attribute[:]
            node.attribute.extend(attribute for name, attribute in attributes.items() if name not in cache_dtype_names)
            if self.target_bits == 4:
                node.attribute.extend(onnx.helper.make_attribute(name, "int4") for name in cache_dtype_names)
            layer_ids.append(int(match.group(1)))

        if not layer_ids:
            raise ValueError("Source graph does not contain a quantized com.microsoft::PagedAttention node.")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Source graph contains multiple PagedAttention nodes for one layer.")
        return layer_ids

    def validate_scale_layers(self, scale_data: dict, layer_ids: list[int]) -> dict[int, int]:
        k_scales = scale_data["scales"]["k_scales"]
        v_scales = scale_data["scales"]["v_scales"]
        scale_layer_ids = scale_data.get("layer_ids", list(range(len(k_scales))))
        if (
            not isinstance(scale_layer_ids, list)
            or any(type(layer_id) is not int for layer_id in scale_layer_ids)
            or len(scale_layer_ids) != len(layer_ids)
            or set(scale_layer_ids) != set(layer_ids)
        ):
            raise ValueError(
                f"Scale layer_ids must match PagedAttention layers; got {scale_layer_ids}, expected {layer_ids}."
            )
        if len(k_scales) != len(layer_ids) or len(v_scales) != len(layer_ids):
            raise ValueError(
                f"Scales file must provide {len(layer_ids)} layers, got k={len(k_scales)} v={len(v_scales)}."
            )
        return {layer_id: scale_index for scale_index, layer_id in enumerate(scale_layer_ids)}

    def get_calibration_factor(self, file_qmax) -> float:
        if file_qmax is None:
            return 1.0
        if not isinstance(file_qmax, (int, float)) or isinstance(file_qmax, bool) or file_qmax <= 0:
            raise ValueError("KV-cache scale qmax must be a positive number.")
        return float(file_qmax) / KV_CACHE_CALIBRATION_QMAX[f"int{self.target_bits}"]

    def update_scale_initializers(self, model: onnx.ModelProto, scale_data: dict, scale_indices: dict[int, int]) -> int:
        scale_factor = self.get_calibration_factor(scale_data.get("qmax"))
        initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        scale_shape = None
        for layer_id in scale_indices:
            for kind in ("k", "v"):
                name = f"model.layers.{layer_id}.attn.{kind}_scale"
                if name not in initializers:
                    raise ValueError(f"Source graph is missing KV-cache scale initializer {name!r}.")
                initializer = initializers[name]
                shape = tuple(initializer.dims)
                if len(shape) != 3 or shape[1] != 1 or shape[0] <= 0 or shape[2] <= 0:
                    raise ValueError(f"KV-cache scale {name!r} must have shape (num_kv_heads, 1, head_size).")
                if scale_shape is not None and shape != scale_shape:
                    raise ValueError(f"KV-cache scale {name!r} has shape {shape}, expected {scale_shape}.")
                scale_shape = shape

        for layer_id, scale_index in scale_indices.items():
            for kind in ("k", "v"):
                initializer = initializers[f"model.layers.{layer_id}.attn.{kind}_scale"]
                values = scale_data["scales"][f"{kind}_scales"][scale_index]
                self.replace_scale_initializer(initializer, values, scale_factor)
        return scale_shape[-1]

    def replace_scale_initializer(
        self, initializer: onnx.TensorProto, values: list[float], scale_factor: float
    ) -> None:
        shape = tuple(initializer.dims)
        scale = np.asarray(values, dtype=np.float32).reshape(-1)
        if scale.size != np.prod(shape, dtype=np.int64):
            raise ValueError(f"KV-cache scale {initializer.name} has {scale.size} values, expected {np.prod(shape)}.")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"KV-cache scale {initializer.name} must contain finite positive values.")

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale = (scale.astype(np.float64) * scale_factor).astype(np.float32)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"Rescaled KV-cache scale {initializer.name} must contain finite positive values.")

        initializer.CopyFrom(onnx.numpy_helper.from_array(scale.reshape(shape), initializer.name))

    def update_cache_values(self, model: onnx.ModelProto, layer_ids: list[int], head_size: int) -> None:
        # INT4 packs two values per byte, rounding up like the builder's packed_head_size.
        cache_dtype = onnx.TensorProto.UINT8 if self.target_bits == 4 else onnx.TensorProto.INT8
        cache_width = (head_size + 1) // 2 if self.target_bits == 4 else head_size
        cache_names = {
            f"{prefix}.{layer_id}.{kind}"
            for prefix in ("past_key_values", "present")
            for layer_id in layer_ids
            for kind in ("key", "value")
        }
        updated_cache_values = set()
        for value_info in (*model.graph.input, *model.graph.output):
            if value_info.name not in cache_names:
                continue
            tensor_type = value_info.type.tensor_type
            tensor_type.elem_type = cache_dtype
            dimensions = tensor_type.shape.dim
            if not dimensions or not dimensions[-1].HasField("dim_value"):
                raise ValueError(f"KV-cache value {value_info.name} must have a concrete final dimension.")
            dimensions[-1].dim_value = cache_width
            updated_cache_values.add(value_info.name)
        if updated_cache_values != cache_names:
            raise ValueError(
                f"Source graph is missing KV-cache graph values: {sorted(cache_names - updated_cache_values)}."
            )
