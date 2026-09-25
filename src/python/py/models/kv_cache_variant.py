# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create a KV-cache quantization variant that reuses an ONNX graph's external weights."""

import json
import re
from pathlib import Path

import numpy as np
import onnx
from quantization import KV_CACHE_CALIBRATION_QMAX

_SUPPORTED_SCHEMES = {"int4_per_channel", "int8_per_channel"}
_LAYER_PATTERN = re.compile(r"(?:^|/)model/layers\.(\d+)/attn/PagedAttention$")


def _attribute_map(node: onnx.NodeProto) -> dict[str, onnx.AttributeProto]:
    return {attribute.name: attribute for attribute in node.attribute}


def _load_scale_data(scale_path: Path, scale_section: str) -> dict:
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


def _replace_scale_initializer(
    initializer: onnx.TensorProto,
    values: list[float],
    scale_factor: float,
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


def _update_cache_value_info(value_info: onnx.ValueInfoProto, cache_dtype: int, cache_width: int) -> None:
    tensor_type = value_info.type.tensor_type
    tensor_type.elem_type = cache_dtype
    dimensions = tensor_type.shape.dim
    if not dimensions or not dimensions[-1].HasField("dim_value"):
        raise ValueError(f"KV-cache value {value_info.name} must have a concrete final dimension.")
    dimensions[-1].dim_value = cache_width


def create_kv_cache_variant(
    source_model_path: str | Path,
    output_model_path: str | Path,
    scheme: str,
    scale_file: str | Path,
) -> None:
    """Create a static per-channel KV-cache variant without duplicating external model weights.

    KV scales are recalculated from ``scale_file`` and stored inside the variant ONNX file. Every
    other initializer keeps the source graph's external-data location, offset, and length.
    """
    if scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(f"scheme must be one of {sorted(_SUPPORTED_SCHEMES)}, got {scheme!r}.")

    source_model_path = Path(source_model_path)
    output_model_path = Path(output_model_path)
    scale_data = _load_scale_data(Path(scale_file), source_model_path.stem)
    model = onnx.load(source_model_path, load_external_data=False)

    paged_attention_layers: list[int] = []
    head_size = None
    target_bits = int(scheme.removeprefix("int").split("_", 1)[0])
    for node in model.graph.node:
        if node.op_type != "PagedAttention" or node.domain != "com.microsoft":
            continue
        match = _LAYER_PATTERN.search(node.name)
        if not match:
            raise ValueError(f"Cannot determine the layer ID from PagedAttention node {node.name!r}.")
        attributes = _attribute_map(node)
        if attributes.get("k_quant_type", None) is None or attributes.get("v_quant_type", None) is None:
            raise ValueError(f"PagedAttention node {node.name!r} is not statically KV-cache quantized.")
        if attributes["k_quant_type"].s != b"PER_CHANNEL" or attributes["v_quant_type"].s != b"PER_CHANNEL":
            raise ValueError(f"PagedAttention node {node.name!r} does not use per-channel KV quantization.")

        if target_bits == 4:
            for name in ("k_cache_dtype", "v_cache_dtype"):
                if name in attributes:
                    attributes[name].s = b"int4"
                else:
                    node.attribute.append(onnx.helper.make_attribute(name, "int4"))
        else:
            del node.attribute[:]
            node.attribute.extend(
                attribute for name, attribute in attributes.items() if name not in {"k_cache_dtype", "v_cache_dtype"}
            )
        paged_attention_layers.append(int(match.group(1)))

    if not paged_attention_layers:
        raise ValueError("Source graph does not contain a quantized com.microsoft::PagedAttention node.")
    if len(set(paged_attention_layers)) != len(paged_attention_layers):
        raise ValueError("Source graph contains multiple PagedAttention nodes for one layer.")

    layer_ids = scale_data.get("layer_ids", list(range(len(scale_data["scales"]["k_scales"]))))
    if layer_ids != paged_attention_layers:
        raise ValueError(
            f"Scale layer_ids must match PagedAttention layers; got {layer_ids}, expected {paged_attention_layers}."
        )
    k_scales = scale_data["scales"]["k_scales"]
    v_scales = scale_data["scales"]["v_scales"]
    if len(k_scales) != len(layer_ids) or len(v_scales) != len(layer_ids):
        raise ValueError(f"Scales file must provide {len(layer_ids)} layers, got k={len(k_scales)} v={len(v_scales)}.")

    file_qmax = scale_data.get("qmax")
    if file_qmax is None:
        scale_factor = 1.0
    elif not isinstance(file_qmax, (int, float)) or isinstance(file_qmax, bool) or file_qmax <= 0:
        raise ValueError("KV-cache scale qmax must be a positive number.")
    else:
        scale_factor = float(file_qmax) / KV_CACHE_CALIBRATION_QMAX[f"int{target_bits}"]

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    for scale_index, layer_id in enumerate(layer_ids):
        for kind, values in (("k", k_scales), ("v", v_scales)):
            name = f"model.layers.{layer_id}.attn.{kind}_scale"
            if name not in initializers:
                raise ValueError(f"Source graph is missing KV-cache scale initializer {name!r}.")
            initializer = initializers[name]
            if head_size is None:
                if len(initializer.dims) != 3 or initializer.dims[1] != 1:
                    raise ValueError(f"KV-cache scale {name!r} must have shape (num_kv_heads, 1, head_size).")
                head_size = initializer.dims[-1]
            _replace_scale_initializer(initializer, values[scale_index], scale_factor)

    cache_dtype = onnx.TensorProto.UINT8 if target_bits == 4 else onnx.TensorProto.INT8
    cache_width = head_size // 2 if target_bits == 4 else head_size
    cache_names = {
        f"{prefix}.{layer_id}.{kind}"
        for prefix in ("past_key_values", "present")
        for layer_id in layer_ids
        for kind in ("key", "value")
    }
    updated_cache_values = set()
    for value_info in (*model.graph.input, *model.graph.output):
        if value_info.name in cache_names:
            _update_cache_value_info(value_info, cache_dtype, cache_width)
            updated_cache_values.add(value_info.name)
    if updated_cache_values != cache_names:
        raise ValueError(
            f"Source graph is missing KV-cache graph values: {sorted(cache_names - updated_cache_values)}."
        )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_model_path)
