#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build an isolated ported package with fused weights, hyper-connections, and INT8 attention."""

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def untile_weights(values, kind):
    if values.size != 3276800 or kind not in (0, 1):
        raise ValueError("Unsupported hyper-connection weight shape or kind")
    vectors = values.reshape(-1, 16)
    index = np.arange(len(vectors))
    if kind == 0:
        rp = index % 128
        rest = index // 128
        destination = ((rest // 200 * 10 + rest % 10) * 128 + rp) * 20 + rest // 10 % 20
        shape = (10240, 320)
    else:
        c = index % 4
        rest = index // 4
        rp = rest % 32
        rest //= 32
        destination = ((rest % 10 * 32 + rp) * 4 + c) * 160 + rest // 10
        shape = (320, 10240)
    output = np.empty_like(vectors)
    output[destination] = vectors
    return output.reshape(shape)


def ensure_opset_import(model, domain="com.microsoft", version=1):
    opset = next((item for item in model.opset_import if item.domain == domain), None)
    if opset is None:
        model.opset_import.append(helper.make_opsetid(domain, version))
    elif opset.version < version:
        opset.version = version


def optimize_model(path, transformed_provenance=None):
    model = onnx.load(path, load_external_data=False)
    initializers = {t.name: t for t in model.graph.initializer}
    producers = {out: n for n in model.graph.node for out in n.output}
    consumers = Counter(x for n in model.graph.node for x in n.input)
    outputs = externally_observed_values(model.graph)
    removed = set()
    replacements = {}
    new_initializers = {}
    names = graph_value_names(model.graph)
    transformed_weights = {}
    counts = Counter()
    data_path = path.with_name("performance_weights.bin")
    if data_path.exists():
        raise FileExistsError(data_path)

    def array(name):
        tensor = copy.deepcopy(initializers[name])
        if tensor.data_location == TensorProto.EXTERNAL:
            external = {item.key: item.value for item in tensor.external_data}
            data_file = (path.parent / external["location"]).resolve(strict=True)
            if not data_file.is_relative_to(path.parent.parent.resolve()):
                raise ValueError(f"External data escapes package: {data_file}")
            with data_file.open("rb") as stream:
                stream.seek(int(external.get("offset", 0)))
                tensor.raw_data = stream.read(int(external["length"]) if "length" in external else -1)
            tensor.ClearField("external_data")
            tensor.data_location = TensorProto.DEFAULT
        return numpy_helper.to_array(tensor)

    def add_weight(name, values, stream):
        tensor = numpy_helper.from_array(np.ascontiguousarray(values), name)
        offset = stream.tell()
        stream.write(tensor.raw_data)
        onnx.external_data_helper.set_external_data(
            tensor, location=data_path.name, offset=offset, length=len(tensor.raw_data)
        )
        tensor.ClearField("raw_data")
        new_initializers[name] = tensor
        return name

    with data_path.open("xb") as data:
        for node in model.graph.node:
            if node.op_type == "DynamicSparseAttention" and node.name.endswith("_generic_DynamicSparseAttention"):
                prefix = node.name.removesuffix("DynamicSparseAttention")
                wrappers = {n.name: n for n in model.graph.node if n.name.startswith(prefix)}
                fused = copy.deepcopy(node)
                fused.name += "_Int8Cache"
                fused.input[:3] = [wrappers[prefix + "Cast" + kind].input[0] for kind in ("Query", "Key", "Value")]
                fused.input[3:5] = [wrappers[prefix + kind + "_CacheToBSNH"].input[0] for kind in ("key", "value")]
                fused.input.extend([prefix + "k_scale", prefix + "v_scale"])
                fused.output[0] = wrappers[prefix + "CastAttentionToBf16"].output[0]
                fused.output[1:3] = [
                    wrappers[prefix + "present_" + kind + "_ToBNSH"].output[0] for kind in ("key", "value")
                ]
                retained = {node.name, prefix + "FlattenSelectedIndices", prefix + "FlattenSelectedCounts"}
                discarded = [n for name, n in wrappers.items() if name not in retained]
                preserved_outputs = set(fused.output)
                for discarded_node in discarded:
                    for output in discarded_node.output:
                        external_users = [
                            n.name for n in model.graph.node if output in n.input and not n.name.startswith(prefix)
                        ]
                        if output not in preserved_outputs and (output in outputs or external_users):
                            raise ValueError(f"Cannot remove externally consumed cache conversion: {output}")
                    removed.add(id(discarded_node))
                replacements[id(node)] = fused
                counts["direct_int8_attention"] += 1
            if node.op_type == "UntileHyperConnectionWeights":
                if node.input[0] not in initializers:
                    continue
                kind = next(a.i for a in node.attribute if a.name == "kind")
                add_weight(node.output[0], untile_weights(array(node.input[0]), kind), data)
                if transformed_provenance is not None:
                    transformed_provenance[node.output[0]] = (
                        node.input[0],
                        (
                            "untile",
                            kind,
                            tuple(new_initializers[node.output[0]].dims),
                            new_initializers[node.output[0]].data_type,
                        ),
                    )
                removed.add(id(node))
                counts["constant_untiles"] += 1
            if node.op_type != "MatMul":
                continue
            cast = producers.get(node.input[1])
            if cast is None or cast.op_type != "Cast":
                continue
            if helper.get_attribute_value(cast.attribute[0]) != TensorProto.BFLOAT16:
                continue
            dq = producers.get(cast.input[0])
            if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) != 2:
                continue
            weight_name, scale_name = dq.input
            weight = initializers.get(weight_name)
            if weight is None or weight.data_type != TensorProto.FLOAT8E4M3FN or len(weight.dims) != 2:
                continue
            axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)
            if axis not in (1, -1):
                continue
            k, n = weight.dims
            if scale_name not in initializers or initializers[scale_name].data_type != TensorProto.FLOAT:
                continue
            scales = array(scale_name)
            if scales.shape != (n,) or k % 16:
                continue
            key = (weight_name, scale_name)
            if key not in transformed_weights:
                transposed = add_weight(
                    unique_value_name(weight_name + ".performance_nk", names), array(weight_name).T, data
                )
                scale = add_weight(unique_value_name(scale_name + ".performance_n1", names), scales.reshape(n, 1), data)
                transformed_weights[key] = (transposed, scale)
                if transformed_provenance is not None:
                    transformed_provenance[transposed] = (
                        weight_name,
                        (
                            "transpose",
                            1,
                            tuple(new_initializers[transposed].dims),
                            new_initializers[transposed].data_type,
                        ),
                    )
                    transformed_provenance[scale] = (
                        scale_name,
                        ("reshape", (n, 1), new_initializers[scale].data_type),
                    )
            transposed, scale = transformed_weights[key]
            replacements[id(node)] = helper.make_node(
                "MatMulBlockQuantizedFp8Weight",
                [node.input[0], transposed, scale],
                list(node.output),
                name=node.name + "_FusedFp8",
                domain="com.microsoft",
                block_size=k,
            )
            counts["fused_fp8_matmuls"] += 1
            if consumers[cast.output[0]] == 1 and cast.output[0] not in outputs:
                removed.add(id(cast))
                if consumers[dq.output[0]] == 1 and dq.output[0] not in outputs:
                    removed.add(id(dq))
    nodes = [replacements.get(id(n), n) for n in model.graph.node if id(n) not in removed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input} | outputs
    initializers.update(new_initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(t for name, t in initializers.items() if name in used)
    del model.graph.value_info[:]
    if counts["fused_fp8_matmuls"]:
        ensure_opset_import(model)
    temporary = path.with_suffix(".optimized.onnx")
    onnx.save(model, temporary)
    os.replace(temporary, path)
    return dict(counts)


def quantize_prefill_outputs(path, scales, head=False):
    model = onnx.load(path, load_external_data=False)
    if head:
        captured = captured_values(model.graph)
        if any(output.name in scales and output.name in captured for output in model.graph.output):
            raise ValueError("Cannot quantize captured head output")
    changed = 0
    for output in list(model.graph.output):
        name = output.name
        if name not in scales:
            continue
        shape = output.type.tensor_type.shape
        rank = len(shape.dim)
        if head and rank == 0:
            producer = next(n for n in model.graph.node if name in n.output)
            expected_ops = {"key": {"RotaryEmbedding"}, "value": {"MatMul", "MatMulBlockQuantizedFp8Weight"}}[name]
            if producer.op_type not in expected_ops:
                raise ValueError(f"Unexpected head prefill producer: {producer.op_type}")
            rank = 3
            dimensions = [1, "prompt_length", 512]
            output.CopyFrom(helper.make_tensor_value_info(name, TensorProto.BFLOAT16, dimensions))
        raw = name + ".performance_unquantized" if head else name
        result = name if head else name + ".quantized"
        if head:
            for node in model.graph.node:
                node.input[:] = [raw if x == name else x for x in node.input]
                node.output[:] = [raw if x == name else x for x in node.output]
        prefix = name + ".performance_quant"
        source = raw
        nodes = [helper.make_node("Cast", [source], [prefix + ".float"], to=TensorProto.FLOAT)]
        source = prefix + ".float"
        if rank == 4:
            nodes.append(helper.make_node("Transpose", [source], [prefix + ".bsnh"], perm=[0, 2, 1, 3]))
            source = prefix + ".bsnh"
        elif not head or rank != 3:
            raise ValueError(f"Unexpected prefill cache shape: {name}")
        flat_shape = prefix + ".flat_shape"
        full_shape = prefix + ".full_shape"
        scale_name = prefix + ".scale"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray([0, 0, -1], np.int64), flat_shape),
                numpy_helper.from_array(np.asarray([0, 0, 2, 256], np.int64), full_shape),
                numpy_helper.from_array(scales[name], scale_name),
                numpy_helper.from_array(np.zeros(512, np.int8), prefix + ".zero"),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Reshape", [source, flat_shape], [prefix + ".flat"]),
                helper.make_node(
                    "QuantizeLinear", [prefix + ".flat", scale_name, prefix + ".zero"], [prefix + ".int8"], axis=2
                ),
            ]
        )
        if rank == 4:
            nodes.extend(
                [
                    helper.make_node("Reshape", [prefix + ".int8", full_shape], [prefix + ".bsnh_int8"]),
                    helper.make_node("Transpose", [prefix + ".bsnh_int8"], [result], perm=[0, 2, 1, 3]),
                ]
            )
        else:
            nodes.append(helper.make_node("Identity", [prefix + ".int8"], [result]))
        model.graph.node.extend(nodes)
        if head:
            output.type.tensor_type.elem_type = TensorProto.INT8
        else:
            quantized_output = copy.deepcopy(output)
            quantized_output.name = result
            quantized_output.type.tensor_type.elem_type = TensorProto.INT8
            model.graph.output.append(quantized_output)
        changed += 1
    temporary = path.with_suffix(".quantized.onnx")
    onnx.save(model, temporary)
    os.replace(temporary, path)
    return changed


def refresh_weight_pool(source, destination, transformed_provenance=None):
    model_initializers = {
        label: {
            t.name: t for t in onnx.load(destination / label / "model.onnx", load_external_data=False).graph.initializer
        }
        for label in ("target", "dense", "head_r2", "head_kv")
    }
    lines = (source / "weights_pool.txt").read_text().splitlines()
    if lines[0] != "qwen38_weights_pool 1":
        raise ValueError("Unsupported weight pool manifest")
    pools = {}
    sessions = defaultdict(list)
    source_pool_indices = {}
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "pool":
            pools[int(fields[1])] = fields
        elif fields[0] == "session":
            _, label, name, index = fields
            index = int(index)
            source_pool_indices[label, name] = index
            if name in model_initializers[label]:
                sessions[index].append((label, name))
        else:
            raise ValueError(f"Unsupported weight pool entry: {fields[0]}")
    output = [lines[0]]
    references = []
    total = 0
    count = 0
    for old_index, users in sessions.items():
        if len({label for label, _ in users}) < 2:
            continue
        fields = pools[old_index][:]
        fields[1] = str(count)
        output.append(" ".join(fields))
        references.extend(f"session {label} {name} {count}" for label, name in users)
        total += int(fields[4])
        count += 1

    transformed_sessions = defaultdict(list)
    for label, provenance in (transformed_provenance or {}).items():
        for name, (source_name, transformation) in provenance.items():
            if name not in model_initializers[label]:
                continue
            source_index = source_pool_indices.get((label, source_name))
            if source_index is not None:
                transformed_sessions[source_index, transformation].append((label, name))

    for users in transformed_sessions.values():
        if len({label for label, _ in users}) < 2:
            continue
        label, name = users[0]
        tensor = model_initializers[label][name]
        external = {item.key: item.value for item in tensor.external_data}
        if tensor.data_location != TensorProto.EXTERNAL or "location" not in external:
            raise ValueError(f"Transformed pooled initializer is not external: {label}:{name}")
        offset = int(external.get("offset", 0))
        length = int(external.get("length", 0))
        if not length:
            length = (destination / label / external["location"]).stat().st_size - offset
        dims = list(tensor.dims)
        fields = [
            "pool",
            str(count),
            str(tensor.data_type),
            str(offset),
            str(length),
            str(len(dims)),
            *(str(dimension) for dimension in dims),
            str(Path(label) / external["location"]),
        ]
        output.append(" ".join(fields))
        references.extend(f"session {session_label} {session_name} {count}" for session_label, session_name in users)
        total += length
        count += 1
    temporary = destination / "weights_pool.optimized.txt"
    temporary.write_text("\n".join(output + references) + "\n")
    os.replace(temporary, destination / "weights_pool.txt")
    return {"initializers": count, "gib": total / 2**30}


def fuse_hyper_connections(path):
    model = onnx.load(path, load_external_data=False)
    initializers = {t.name: t for t in model.graph.initializer}
    users = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            users[name].append(node)
    outputs = externally_observed_values(model.graph)
    removed = set()
    replacements = {}
    types = infer_value_types(model)
    shapes = {value.name: value.type.tensor_type.shape for value in (*model.graph.input, *model.graph.value_info)}
    available = {value.name for value in model.graph.input} | set(initializers) | {""}

    def successor(node, op):
        candidates = users[node.output[0]]
        if len(candidates) != 1 or candidates[0].op_type != op:
            raise ValueError(f"Unexpected hyper-connection consumer after {node.name}")
        return candidates[0]

    def scalar(name):
        values = numpy_helper.to_array(initializers[name])
        if values.size != 1:
            raise ValueError(f"Expected scalar hyper-connection parameter: {name}")
        return float(values.reshape(-1)[0])

    for norm in model.graph.node:
        if norm.op_type != "BranchwiseRMSNorm":
            available.update(norm.output)
            continue
        inputs_available = available.copy()
        available.update(norm.output)
        candidates = users[norm.output[0]]
        premix = next((n for n in candidates if n.op_type == "HyperConnectionPreMix"), None)
        if premix is None:
            continue
        down = next((n for n in candidates if n.op_type == "MatMul" and "input_mix_weight_down" in n.input[1]), None)
        inject = next((n for n in candidates if n.op_type == "MatMul" and "block_inject_weight" in n.input[1]), None)
        if down is None or inject is None:
            continue
        silu = successor(down, "ScaledSiLU")
        up = successor(silu, "MatMul")
        sigmoid = successor(up, "Sigmoid")
        if successor(sigmoid, "HyperConnectionPreMix") is not premix:
            raise ValueError("Inconsistent hyper-connection premix")
        divide = successor(inject, "Div")
        gate_sigmoid = successor(divide, "Sigmoid")
        multiply = successor(gate_sigmoid, "Mul")
        region = [norm, down, silu, up, sigmoid, premix, inject, divide, gate_sigmoid, multiply]
        region_ids = {id(n) for n in region}
        preserved = {premix.output[0], multiply.output[0]}
        for node in region:
            for name in node.output:
                if name not in preserved and (
                    name in outputs or any(id(user) not in region_ids for user in users[name])
                ):
                    raise ValueError(f"Cannot fuse externally consumed hyper-connection value: {name}")
        attrs = {a.name: helper.get_attribute_value(a) for a in norm.attribute}
        mix_attrs = {a.name: helper.get_attribute_value(a) for a in premix.attribute}
        silu_attrs = {a.name: helper.get_attribute_value(a) for a in silu.attribute}
        if attrs["num_branches"] != mix_attrs["num_branches"]:
            raise ValueError("Inconsistent hyper-connection branch counts")
        branches = attrs["num_branches"]
        scale = initializers.get(norm.input[1])
        shape = shapes.get(norm.input[0])
        if (
            branches <= 0
            or shape is None
            or not shape.dim
            or not shape.dim[-1].HasField("dim_value")
            or shape.dim[-1].dim_value <= 0
            or shape.dim[-1].dim_value % branches
            or scale is None
            or scale.data_type != types.get(norm.input[0])
            or list(scale.dims) != [shape.dim[-1].dim_value]
        ):
            continue
        fused_inputs = [norm.input[0], norm.input[1], down.input[1], up.input[1], inject.input[1]]
        if not all(name in inputs_available for name in fused_inputs):
            continue
        fused = helper.make_node(
            "HyperConnectionLowRankMix",
            fused_inputs,
            [premix.output[0], multiply.output[0]],
            name=norm.name + "_FusedLowRank",
            domain="com.microsoft",
            num_branches=attrs["num_branches"],
            epsilon=attrs.get("epsilon", 1e-5),
            silu_alpha=silu_attrs["alpha"],
            reduction_scale=mix_attrs["reduction_scale"],
            gate_div=scalar(divide.input[1]),
            gate_mul=scalar(multiply.input[1]),
        )
        replacements[id(norm)] = fused
        removed.update(region_ids - {id(norm)})
    nodes = [replacements.get(id(n), n) for n in model.graph.node if id(n) not in removed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input} | outputs
    del model.graph.initializer[:]
    model.graph.initializer.extend(t for name, t in initializers.items() if name in used)
    del model.graph.value_info[:]
    temporary = path.with_suffix(".fused.onnx")
    onnx.save(model, temporary)
    os.replace(temporary, path)
    return len(replacements)


def fuse_hyper_connection_postmix(path):
    model = onnx.load(path, load_external_data=False)
    top_level_outputs = {value.name for value in model.graph.output}
    nested_observations = externally_observed_values(model.graph) - top_level_outputs
    producers = {name: node for node in model.graph.node for name in node.output}
    users = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            users[name].append(node)
    indices = {id(node): index for index, node in enumerate(model.graph.node)}
    removed = set()
    for node in model.graph.node:
        if (
            node.domain != "com.microsoft"
            or node.op_type != "HyperConnectionLowRankMix"
            or len(node.input) != 5
            or len(node.output) != 2
        ):
            continue
        postmix = producers.get(node.input[0])
        if (
            postmix is None
            or postmix.domain != "com.microsoft"
            or postmix.op_type != "HyperConnectionPostMix"
            or len(postmix.input) != 3
        ):
            continue
        if sum(user.op_type == "HyperConnectionLowRankMix" for user in users[postmix.output[0]]) != 1:
            continue
        streams = postmix.output[0]
        if streams in nested_observations:
            continue
        if any(
            (user is not node and indices[id(user)] < indices[id(node)]) or (user is node and streams in node.input[1:])
            for user in users[streams]
        ):
            continue
        branches = next(a.i for a in node.attribute if a.name == "num_branches")
        if next(a.i for a in postmix.attribute if a.name == "num_branches") != branches:
            raise ValueError("Inconsistent post-mix branch counts")
        node.input[0] = postmix.input[0]
        node.input.extend(postmix.input[1:3])
        node.output.append(streams)
        removed.add(id(postmix))
    if removed:
        nodes = [node for node in model.graph.node if id(node) not in removed]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".postmix.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(removed)


def fuse_fp8_projection_split(path):
    """Write eligible FP8 projection partitions directly without changing matmul arithmetic."""
    model = onnx.load(path, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    users = Counter(name for node in model.graph.node for name in node.input)
    outputs = externally_observed_values(model.graph)
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in (*model.graph.input, *model.graph.value_info)
        if value.type.tensor_type.HasField("shape")
    }
    removed, replacements = set(), {}
    for split in model.graph.node:
        if split.domain or split.op_type != "Split" or len(split.input) != 2:
            continue
        projection = producers.get(split.input[0])
        if (
            projection is None
            or projection.domain != "com.microsoft"
            or projection.op_type != "MatMulBlockQuantizedFp8Weight"
            or len(projection.output) != 1
            or users[split.input[0]] != 1
            or split.input[0] in outputs
            or len(projection.input) < 3
            or any(attr.name == "split_sizes" for attr in projection.attribute)
        ):
            continue
        weight = initializers.get(projection.input[1])
        sizes = initializers.get(split.input[1])
        if (
            weight is None
            or len(weight.dims) != 2
            or sizes is None
            or sizes.data_type != TensorProto.INT64
            or len(sizes.dims) != 1
            or sizes.data_location == TensorProto.EXTERNAL
        ):
            continue
        widths = numpy_helper.to_array(sizes).tolist()
        axis = next((attr.i for attr in split.attribute if attr.name == "axis"), 0)
        rank = ranks.get(projection.input[0])
        if (
            (axis != -1 and (rank is None or axis != rank - 1))
            or not 1 <= len(widths) <= 8
            or len(widths) != len(split.output)
            or not all(width > 0 for width in widths)
            or sum(widths) != weight.dims[0]
        ):
            continue
        fused = copy.deepcopy(projection)
        fused.output[:] = split.output
        fused.attribute.append(helper.make_attribute("split_sizes", widths))
        replacements[projection.output[0]] = fused
        removed.add(split.output[0])
    if removed:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not node.output or node.output[0] not in removed
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".fp8_split.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(removed)


def fuse_fp8_projection_slices(path):
    """Partition complete projection ranges and rebase eligible indexed consumers."""
    model = onnx.load(path, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in (*inferred.graph.input, *inferred.graph.value_info)
        if value.type.tensor_type.HasField("shape")
    }
    ranks.update((tensor.name, len(tensor.dims)) for tensor in model.graph.initializer)
    shape_lengths = {
        tensor.name: tensor.dims[0]
        for tensor in model.graph.initializer
        if tensor.data_type == TensorProto.INT64 and len(tensor.dims) == 1
    }
    for value in (*inferred.graph.input, *inferred.graph.value_info):
        shape = value.type.tensor_type.shape
        if len(shape.dim) == 1 and shape.dim[0].HasField("dim_value"):
            shape_lengths[value.name] = shape.dim[0].dim_value
    for node in model.graph.node:
        if not node.domain and node.input:
            rank = None
            if node.op_type == "Reshape" and node.input[1] in shape_lengths:
                rank = shape_lengths[node.input[1]]
            elif node.op_type == "Expand" and node.input[1] in shape_lengths and node.input[0] in ranks:
                rank = max(ranks[node.input[0]], shape_lengths[node.input[1]])
            elif node.op_type in {"Cast", "Identity", "Sigmoid", "Slice"}:
                rank = ranks.get(node.input[0])
            elif node.op_type in {"Add", "Mul", "Sub", "Div"} and all(name in ranks for name in node.input):
                rank = max(ranks[name] for name in node.input)
            elif node.op_type == "Concat" and all(name in ranks for name in node.input):
                if len({ranks[name] for name in node.input}) == 1:
                    rank = ranks[node.input[0]]
            if rank is not None:
                for name in node.output:
                    ranks[name] = rank
        if node.domain != "com.microsoft" or not node.input or node.input[0] not in ranks:
            continue
        rank = ranks[node.input[0]]
        if node.op_type == "HyperConnectionLowRankMix":
            flattened = any(attr.name == "num_branches" and attr.i > 0 for attr in node.attribute)
            for index, name in enumerate(node.output):
                ranks[name] = rank if flattened or index == 2 else rank - 1
        elif node.op_type == "CausalConvWithState":
            ranks[node.output[0]] = rank
        elif node.op_type in {"HyperConnectionPostMix", "BranchwiseRMSNorm", "MatMulBlockQuantizedFp8Weight"}:
            for name in node.output:
                ranks[name] = rank
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    graph_outputs = externally_observed_values(model.graph)
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    removed, replacements = set(), {}
    count = 0

    def scalar(name):
        tensor = initializers.get(name)
        if (
            tensor is None
            or tensor.data_type != TensorProto.INT64
            or list(tensor.dims) != [1]
            or tensor.data_location == TensorProto.EXTERNAL
        ):
            return None
        return int(numpy_helper.to_array(tensor)[0])

    for projection in model.graph.node:
        if (
            projection.domain != "com.microsoft"
            or projection.op_type != "MatMulBlockQuantizedFp8Weight"
            or len(projection.output) != 1
            or projection.output[0] in graph_outputs
            or len(projection.input) < 3
            or any(attr.name == "split_sizes" for attr in projection.attribute)
        ):
            continue
        weight = initializers.get(projection.input[1])
        if weight is None or len(weight.dims) != 2:
            continue
        users = consumers[projection.output[0]]
        indexed = [
            node
            for node in users
            if node.domain == "com.microsoft" and node.op_type in {"CausalConvWithState", "SigmoidGateMul"}
        ]
        if len(indexed) != 1:
            continue
        indexed_node = indexed[0]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in indexed_node.attribute}
        is_conv = indexed_node.op_type == "CausalConvWithState"
        if is_conv:
            if (
                indexed_node.input[0] != projection.output[0]
                or attrs.get("ndim", 1) != 1
                or attrs.get("channels_last", 0) != 1
                or attrs.get("input_channels", 0) <= 0
            ):
                continue
            # A channels-last 1D convolution proves rank three even without value_info.
            start = attrs.get("input_channel_offset", 0)
            end = start + attrs["input_channels"]
        else:
            if (
                len(indexed_node.input) < 2
                or indexed_node.input[1] != projection.output[0]
                or attrs.get("broadcast_gate", 0)
                or attrs.get("silu_gate", 0)
                or ranks.get(projection.output[0]) != 3
                or attrs.get("head_dim", 0) <= 0
                or attrs.get("num_heads", 0) <= 0
                or attrs.get("q_stride", 0) < 2 * attrs["head_dim"]
                or attrs.get("tok_stride", 0) != weight.dims[0]
            ):
                continue
            start = attrs.get("q_offset", 0)
            end = start + attrs["num_heads"] * attrs["q_stride"]
        partition_name = projection.output[0] + ".indexed_partition"
        if partition_name in names:
            continue
        indexed_start, indexed_end = start, end
        intervals = [(start, end, partition_name)] if is_conv else []
        slices = []
        for node in users:
            if node is indexed_node:
                continue
            if (
                node.domain
                or node.op_type != "Slice"
                or node.input[0] != projection.output[0]
                or len(node.input) not in (4, 5)
                or len(node.output) != 1
            ):
                break
            begin, end, axis = (scalar(name) for name in node.input[1:4])
            step = scalar(node.input[4]) if len(node.input) == 5 else 1
            if begin is None or end is None or axis not in (2, -1) or step != 1:
                break
            intervals.append((begin, end, node.output[0]))
            slices.append(node)
        else:
            if not is_conv:
                partition = next(
                    (name for begin, stop, name in intervals if begin == indexed_start and stop == indexed_end), None
                )
                if partition is None:
                    continue
                partition_name = partition
            intervals.sort()
            if (
                not slices
                or len(intervals) > 8
                or intervals[0][0] != 0
                or intervals[-1][1] != weight.dims[0]
                or any(begin >= end for begin, end, _ in intervals)
                or any(left[1] != right[0] for left, right in pairwise(intervals))
            ):
                continue
            fused = copy.deepcopy(projection)
            fused.output[:] = [name for _, _, name in intervals]
            fused.attribute.append(helper.make_attribute("split_sizes", [end - begin for begin, end, _ in intervals]))
            rebased = copy.deepcopy(indexed_node)
            rebased.input[0 if is_conv else 1] = partition_name
            del rebased.attribute[:]
            if is_conv:
                attrs["input_channel_offset"] = 0
            else:
                attrs.update(q_offset=0, tok_stride=indexed_end - indexed_start)
            rebased.attribute.extend(helper.make_attribute(name, value) for name, value in attrs.items())
            replacements[projection.output[0]] = fused
            replacements[indexed_node.output[0]] = rebased
            removed.update(node.output[0] for node in slices)
            names.add(partition_name)
            count += 1
    if count:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not node.output or node.output[0] not in removed
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".fp8_slices.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return count


def graph_value_names(graph):
    names = {value.name for value in (*graph.input, *graph.output, *graph.value_info)}
    names.update(tensor.name for tensor in graph.initializer)
    for node in graph.node:
        names.update((*node.input, *node.output))
        names.add(node.name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                names.update(graph_value_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    names.update(graph_value_names(nested))
    return names


def unique_value_name(base, names):
    name = base
    suffix = 1
    while name in names:
        name = f"{base}_{suffix}"
        suffix += 1
    names.add(name)
    return name


def captured_values(graph):
    protected = set()

    def protect_captures(nested):
        protected.update(name for node in nested.node for name in node.input)
        protected.update(value.name for value in nested.output)
        visit(nested)

    def visit(current):
        for node in current.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    protect_captures(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        protect_captures(nested)

    visit(graph)
    return protected


def externally_observed_values(graph):
    return {value.name for value in graph.output} | captured_values(graph)


def fuse_standard_rms_norm(path):
    """Preserve standard FP32 reduction and rounding while fusing RMS chain launches."""
    model = onnx.load(path, load_external_data=False)
    types = infer_value_types(model)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    constants = {
        name: numpy_helper.to_array(tensor)
        for name, tensor in tensors.items()
        if tensor.data_location != TensorProto.EXTERNAL
    }
    producers = {name: node for node in model.graph.node for name in node.output}
    users = Counter(name for node in model.graph.node for name in node.input)
    outputs = externally_observed_values(model.graph)
    replacements, removed = {}, set()

    def producer(name, kind):
        node = producers.get(name)
        return node if node is not None and not node.domain and node.op_type == kind else None

    def attribute(node, name, default=None):
        return next((helper.get_attribute_value(attr) for attr in node.attribute if attr.name == name), default)

    def layout(name):
        node = producer(name, "Reshape")
        if node is not None and not attribute(node, "allowzero", 0):
            shape = constants.get(node.input[1])
            if shape is not None and shape.ndim == 1 and len(shape) >= 2 and shape[-1] > 0:
                return len(shape), int(shape[-1])
        node = producer(name, "Slice")
        if node is not None and len(node.input) in (4, 5):
            source_layout = layout(node.input[0])
            values = [constants.get(input_name) for input_name in node.input[1:]]
            if source_layout is None or any(value is None or value.shape != (1,) for value in values):
                return None
            start, end, axis = (int(value[0]) for value in values[:3])
            step = int(values[3][0]) if len(values) == 4 else 1
            rank, width = source_layout
            if axis in (-1, rank - 1) and step == 1 and 0 <= start < end <= width:
                return rank, end - start
        return None

    for cast_out in model.graph.node:
        if cast_out.domain or cast_out.op_type != "Cast":
            continue
        dtype = attribute(cast_out, "to")
        if dtype not in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
            continue
        multiply = producer(cast_out.input[0], "Mul")
        if multiply is None:
            continue
        divide = producer(multiply.input[0], "Div")
        scale = tensors.get(multiply.input[1])
        if divide is None or scale is None or scale.data_type != TensorProto.FLOAT or len(scale.dims) != 1:
            continue
        root = producer(divide.input[1], "Sqrt")
        add = producer(root.input[0], "Add") if root is not None else None
        mean = producer(add.input[0], "ReduceMean") if add is not None else None
        square = producer(mean.input[0], "Mul") if mean is not None else None
        cast_in = producer(divide.input[0], "Cast")
        if square is None or cast_in is None or list(square.input) != [divide.input[0], divide.input[0]]:
            continue
        shape = layout(cast_in.input[0])
        axes = constants.get(mean.input[1]) if len(mean.input) == 2 else attribute(mean, "axes")
        epsilon = constants.get(add.input[1])
        if (
            attribute(cast_in, "to") != TensorProto.FLOAT
            or types.get(cast_in.input[0]) != dtype
            or shape is None
            or shape[1] != scale.dims[0]
            or axes is None
            or not np.array_equal(axes, [-1])
            or attribute(mean, "keepdims", 1) != 1
            or epsilon is None
            or epsilon.dtype != np.float32
            or epsilon.size != 1
            or epsilon.ndim > shape[0]
            or not np.isfinite(epsilon).all()
            or epsilon.item() < 0
        ):
            continue
        chain = [cast_in, square, mean, add, root, divide, multiply]
        expected_users = [3, 1, 1, 1, 1, 1, 1]
        if any(
            node.output[0] in outputs or node.output[0] in removed or users[node.output[0]] != count
            for node, count in zip(chain, expected_users, strict=False)
        ):
            continue
        replacements[cast_out.output[0]] = helper.make_node(
            "BranchwiseRMSNorm",
            [cast_in.input[0], multiply.input[1]],
            list(cast_out.output),
            name=cast_out.name + "_StandardRMS",
            domain="com.microsoft",
            epsilon=float(epsilon.item()),
            use_onnx_arithmetic=1,
        )
        removed.update(node.output[0] for node in chain)
    if replacements:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not node.output or node.output[0] not in removed
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        ensure_opset_import(model, version=2)
        temporary = path.with_suffix(".standard_rms.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def fuse_rms_input_slices(path):
    """Read a private last-axis slice directly while preserving exact RMS reduction order."""
    model = onnx.load(path, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {
        value.name: value.type.tensor_type.shape
        for value in (*inferred.graph.input, *inferred.graph.value_info)
        if value.type.tensor_type.HasField("shape")
    }
    constants = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
        if tensor.data_location != TensorProto.EXTERNAL
    }
    producers = {name: node for node in model.graph.node for name in node.output}
    users = Counter(name for node in model.graph.node for name in node.input)
    protected = externally_observed_values(model.graph)
    replacements, removed = {}, set()

    def attr(node, name, default=0):
        return next((helper.get_attribute_value(value) for value in node.attribute if value.name == name), default)

    for norm in model.graph.node:
        if (
            norm.domain != "com.microsoft"
            or norm.op_type != "BranchwiseRMSNorm"
            or attr(norm, "use_onnx_arithmetic") != 1
            or attr(norm, "num_branches") != 0
            or attr(norm, "input_start") != 0
            or attr(norm, "input_width") != 0
        ):
            continue
        sliced = producers.get(norm.input[0])
        if (
            sliced is None
            or sliced.domain
            or sliced.op_type != "Slice"
            or len(sliced.input) not in (4, 5)
            or users[sliced.output[0]] != 1
            or sliced.output[0] in protected
        ):
            continue
        values = [constants.get(name) for name in sliced.input[1:]]
        if any(value is None or value.shape != (1,) or value.dtype not in (np.int32, np.int64) for value in values):
            continue
        start, end, axis = (int(value[0]) for value in values[:3])
        step = int(values[3][0]) if len(values) == 4 else 1
        shape = shapes.get(sliced.input[0])
        rank, width = (len(shape.dim), shape.dim[-1].dim_value) if shape is not None and shape.dim else (0, 0)
        if not width:
            reshape = producers.get(sliced.input[0])
            if (
                reshape is not None
                and not reshape.domain
                and reshape.op_type == "Reshape"
                and attr(reshape, "allowzero") == 0
            ):
                dimensions = constants.get(reshape.input[1])
                if dimensions is not None and dimensions.ndim == 1 and len(dimensions) >= 2:
                    rank, width = len(dimensions), int(dimensions[-1])
        if rank < 2 or axis not in (-1, rank - 1) or step != 1 or not 0 <= start < end <= width:
            continue
        fused = copy.deepcopy(norm)
        fused.input[0] = sliced.input[0]
        kept = [value for value in fused.attribute if value.name not in ("input_start", "input_width")]
        del fused.attribute[:]
        fused.attribute.extend(kept)
        fused.attribute.extend(
            [helper.make_attribute("input_start", start), helper.make_attribute("input_width", end - start)]
        )
        replacements[norm.output[0]] = fused
        removed.add(sliced.output[0])
    if replacements:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not removed.intersection(node.output)
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        for opset in model.opset_import:
            if opset.domain == "com.microsoft":
                opset.version = max(opset.version, 2)
        temporary = path.with_suffix(".rms_input_slices.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def fuse_rms_output_casts(path):
    """Store losslessly widened, already-rounded normalization outputs directly."""
    model = onnx.load(path, load_external_data=False)
    users = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            users[name].append(node)
    replacements, removed = {}, set()
    for norm in model.graph.node:
        if (
            norm.domain != "com.microsoft"
            or norm.op_type != "BranchwiseRMSNorm"
            or len(norm.output) != 1
            or not any(attr.name == "use_onnx_arithmetic" and attr.i == 1 for attr in norm.attribute)
        ):
            continue
        casts = [
            node
            for node in users[norm.output[0]]
            if not node.domain
            and node.op_type == "Cast"
            and any(attr.name == "to" and attr.i == TensorProto.FLOAT for attr in node.attribute)
        ]
        if len(casts) != 1:
            continue
        fused = copy.deepcopy(norm)
        fused.output.append(casts[0].output[0])
        replacements[norm.output[0]] = fused
        removed.add(casts[0].output[0])
    if replacements:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not node.output or node.output[0] not in removed
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".rms_output_casts.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(removed)


def fuse_gdn_qkv_split(path):
    """Reuse packed GDN input for complete Q/K/V splits with private layout views."""
    model = onnx.load(path, load_external_data=False)
    producers = {name: node for node in model.graph.node for name in node.output}
    users = Counter(name for node in model.graph.node for name in node.input)
    outputs = externally_observed_values(model.graph)
    constants = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
        if tensor.data_location != TensorProto.EXTERNAL
    }
    shapes_by_name = {
        value.name: value.type.tensor_type.shape for value in (*model.graph.input, *model.graph.value_info)
    }
    removed, replacements = set(), {}
    for gdn in model.graph.node:
        if (
            gdn.domain != "com.microsoft"
            or gdn.op_type != "GatedDeltaNet"
            or len(gdn.input) < 7
            or not all(gdn.input[index] for index in (0, 1, 2, 6))
        ):
            continue
        views = [producers.get(name) for name in gdn.input[:3]]
        if any(
            node is None
            or node.domain
            or node.op_type != "Reshape"
            or any(attr.name == "allowzero" and attr.i != 0 for attr in node.attribute)
            or users[node.output[0]] != 1
            or node.output[0] in outputs
            for node in views
        ):
            continue
        split = producers.get(views[0].input[0])
        if (
            split is None
            or split.domain
            or split.op_type != "Split"
            or len(split.output) != 3
            or len(split.input) != 2
            or [node.input[0] for node in views] != list(split.output)
            or any(users[name] != 1 or name in outputs for name in split.output)
        ):
            continue
        axis = next((attr.i for attr in split.attribute if attr.name == "axis"), 0)
        shapes = [constants.get(node.input[1]) for node in views]
        sizes = constants.get(split.input[1])
        if (
            axis not in (2, -1)
            or sizes is None
            or sizes.shape != (3,)
            or any(
                shape is None or shape.shape != (4,) or list(shape[:2]) != [0, 0] or min(shape[2:]) <= 0
                for shape in shapes
            )
            or shapes[0][2] != shapes[1][2]
            or shapes[0][3] != shapes[1][3]
            or any(int(shape[2]) * int(shape[3]) != int(size) for shape, size in zip(shapes, sizes, strict=False))
        ):
            continue
        state_shape = shapes_by_name.get(gdn.input[6])
        if (
            state_shape is None
            or len(state_shape.dim) != 4
            or [dim.dim_value for dim in state_shape.dim[1:]]
            != [int(shapes[2][2]), int(shapes[2][3]), int(shapes[0][3])]
        ):
            continue
        # A channels-last 1D convolution proves the packed input rank and row layout.
        conv = producers.get(split.input[0])
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in conv.attribute} if conv else {}
        if (
            conv is None
            or conv.domain != "com.microsoft"
            or conv.op_type != "CausalConvWithState"
            or conv.output[0] != split.input[0]
            or attrs.get("channels_last", 0) != 1
            or attrs.get("ndim", 1) != 1
        ):
            continue
        fused = copy.deepcopy(gdn)
        fused.input[:3] = [split.input[0], "", ""]
        replacements[gdn.output[0]] = fused
        removed.update(node.output[0] for node in [split, *views])
    if replacements:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not node.output or node.output[0] not in removed
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".gdn_packed.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def fuse_rotary_casts(path):
    """Use versioned FP32 rotary arithmetic with activation-type storage at both boundaries."""
    model = onnx.load(path, load_external_data=False)
    types = infer_value_types(model)
    producers = {name: node for node in model.graph.node for name in node.output}
    users = Counter(name for node in model.graph.node for name in node.input)
    protected = externally_observed_values(model.graph)
    replacements, retired, removed, clones = {}, set(), set(), {}
    node_names = {node.name for node in model.graph.node if node.name}

    def attr(node, name, default=None):
        return next((helper.get_attribute_value(value) for value in node.attribute if value.name == name), default)

    def unwiden(name, dtype, additions, pending):
        if (name, dtype) in clones:
            return clones[name, dtype]
        if (name, dtype) in pending:
            return pending[name, dtype]
        node = producers.get(name)
        if node is None:
            return None
        if (
            not node.domain
            and node.op_type == "Cast"
            and attr(node, "to") == TensorProto.FLOAT
            and types.get(node.input[0]) == dtype
        ):
            result = node.input[0]
        elif (
            node.domain == "com.microsoft"
            and node.op_type == "BranchwiseRMSNorm"
            and len(node.output) == 2
            and name == node.output[1]
            and attr(node, "use_onnx_arithmetic", 0) == 1
            and types.get(node.output[0]) == dtype
        ):
            return node.output[0]
        elif not node.domain and node.op_type in {"Reshape", "Expand"}:
            source = unwiden(node.input[0], dtype, additions, pending)
            if source is None:
                return None
            copied = copy.deepcopy(node)
            copied.input[0] = source
            result = name + "_ActivationStorage"
            if result in producers or result in types:
                return None
            copied.output[:] = [result]
            copied.name = (node.name or name) + "_ActivationStorage"
            while copied.name in node_names:
                copied.name += "_"
            node_names.add(copied.name)
            additions.append(copied)
        else:
            return None
        pending[name, dtype] = result
        return result

    for cast in model.graph.node:
        if cast.domain or cast.op_type != "Cast":
            continue
        dtype = attr(cast, "to")
        if dtype not in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
            continue
        view = producers.get(cast.input[0])
        rotary = producers.get(view.input[0]) if view is not None and view.input else None
        if (
            view is None
            or view.domain
            or view.op_type != "Reshape"
            or rotary is None
            or rotary.domain != "com.microsoft"
            or rotary.op_type != "RotaryEmbedding"
            or attr(rotary, "use_fp32_arithmetic", 0) != 0
            or len(rotary.input) != 4
            or cast.output[0] + "_RotaryActivationStorage" in producers
            or cast.output[0] + "_RotaryActivationStorage" in types
            or any(users[node.output[0]] != 1 or node.output[0] in protected for node in (rotary, view))
        ):
            continue
        additions, pending = [], {}
        inputs = [unwiden(rotary.input[index], dtype, additions, pending) for index in (0, 2, 3)]
        if any(name is None for name in inputs):
            continue
        copied = copy.deepcopy(rotary)
        copied.input[0], copied.input[2], copied.input[3] = inputs
        copied.output[:] = [cast.output[0] + "_RotaryActivationStorage"]
        mode = next((value for value in copied.attribute if value.name == "use_fp32_arithmetic"), None)
        if mode is None:
            copied.attribute.append(helper.make_attribute("use_fp32_arithmetic", 1))
        else:
            mode.i = 1
        rounded_view = copy.deepcopy(view)
        rounded_view.input[0] = copied.output[0]
        rounded_view.output[:] = list(cast.output)
        replacements[cast.output[0]] = [*additions, copied, rounded_view]
        clones.update(pending)
        retired.update(name for name, _ in pending)
        removed.update((rotary.output[0], view.output[0]))
    if not replacements:
        return 0
    nodes = [
        new
        for node in model.graph.node
        for new in replacements.get(node.output[0], [node])
        if node.output[0] not in removed
    ]
    while True:
        live = {name for node in nodes for name in node.input} | protected
        unused = {
            name
            for node in nodes
            if all(name not in live for name in node.output)
            for name in node.output
            if name in retired
        }
        if not unused:
            break
        nodes = [node for node in nodes if not unused.intersection(node.output)]
    live = {name for node in nodes for name in node.input} | protected
    for node in nodes:
        if (
            node.domain == "com.microsoft"
            and node.op_type == "BranchwiseRMSNorm"
            and len(node.output) == 2
            and node.output[1] not in live
        ):
            del node.output[1:]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            opset.version = max(opset.version, 2)
    temporary = path.with_suffix(".rotary_casts.onnx")
    onnx.save(model, temporary)
    os.replace(temporary, path)
    return len(replacements)


def share_rotary_cache_expands(path):
    """Share cache expands only when their batch/sequence dimensions are proven equal."""
    model = onnx.load(path, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    prefixes = {}
    ranks = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info):
        if not value.type.tensor_type.HasField("shape"):
            continue
        shape = value.type.tensor_type.shape
        ranks[value.name] = len(shape.dim)
        if len(shape.dim) >= 2:
            dims = tuple(dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in shape.dim[:2])
            if all(dim != "" and not (isinstance(dim, str) and dim.startswith("unk__")) for dim in dims):
                prefixes[value.name] = dims
    constants = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
        if tensor.data_location != TensorProto.EXTERNAL
    }
    ranks.update((tensor.name, len(tensor.dims)) for tensor in model.graph.initializer)
    producers = {name: node for node in model.graph.node for name in node.output}
    protected = externally_observed_values(model.graph)
    aliases, seen = {}, {}

    def attr(node, name, default=None):
        return next((helper.get_attribute_value(value) for value in node.attribute if value.name == name), default)

    for node in model.graph.node:
        node.input[:] = [aliases.get(name, name) for name in node.input]
        first = node.input[0] if node.input else ""
        prefix, rank = prefixes.get(first), ranks.get(first)
        output_rank = rank
        preserve = False
        if not node.domain and node.op_type == "CastLike" and rank is not None:
            ranks[node.output[0]] = rank
            if prefix is not None:
                prefixes[node.output[0]] = prefix
        if prefix is not None and node.domain == "com.microsoft":
            preserve = (
                (node.op_type in {"HyperConnectionLowRankMix", "MatMulBlockQuantizedFp8Weight"} and rank == 3)
                or node.op_type in {"HyperConnectionPostMix", "BranchwiseRMSNorm"}
                or (
                    node.op_type == "CausalConvWithState"
                    and rank == 3
                    and attr(node, "channels_last", 0) == 1
                    and attr(node, "ndim", 1) == 1
                )
            )
        elif prefix is not None and not node.domain:
            if node.op_type in {"Cast", "Identity", "Sigmoid", "Sign", "Abs", "Sqrt"}:
                preserve = True
            elif node.op_type == "Reshape" and not attr(node, "allowzero", 0):
                shape = constants.get(node.input[1])
                preserve = shape is not None and shape.ndim == 1 and len(shape) >= 2 and list(shape[:2]) == [0, 0]
                if preserve:
                    output_rank = len(shape)
            elif node.op_type == "Slice" and len(node.input) >= 4:
                axes = constants.get(node.input[3])
                preserve = (
                    axes is not None
                    and rank is not None
                    and all(-rank <= int(axis) < rank and int(axis) % rank >= 2 for axis in axes)
                )
            elif node.op_type in {"ReduceSum", "ReduceMean"} and len(node.input) == 2:
                axes = constants.get(node.input[1])
                preserve = (
                    axes is not None
                    and axes.size > 0
                    and rank is not None
                    and attr(node, "keepdims", 1) == 1
                    and all(-rank <= int(axis) < rank and int(axis) % rank >= 2 for axis in axes)
                )
            elif node.op_type == "Unsqueeze" and len(node.input) == 2:
                axes = constants.get(node.input[1])
                if axes is not None and axes.ndim == 1 and rank is not None:
                    output_rank = rank + len(axes)
                    preserve = all(
                        -output_rank <= int(axis) < output_rank and int(axis) % output_rank >= 2 for axis in axes
                    )
            elif node.op_type in {"Add", "Mul", "Div", "Max"} and len(node.input) == 2:
                other = node.input[1]
                preserve = (
                    ranks.get(other) == 0
                    or (ranks.get(other) == 1 and rank >= 3)
                    or (ranks.get(other) == rank and prefixes.get(other) == prefix)
                )
            elif node.op_type == "Concat" and rank is not None:
                preserve = attr(node, "axis", 0) % rank >= 2 and all(
                    ranks.get(name) == rank and prefixes.get(name) == prefix for name in node.input
                )
        if preserve:
            names = node.output[:1] if node.op_type == "CausalConvWithState" else node.output
            for name in names:
                prefixes[name], ranks[name] = prefix, output_rank
        if node.domain or node.op_type != "Concat" or attr(node, "axis", 0) != 0 or len(node.input) != 2:
            continue
        gather = producers.get(node.input[0])
        shape = producers.get(gather.input[0]) if gather is not None and gather.input else None
        width = constants.get(node.input[1])
        if (
            gather is None
            or gather.domain
            or gather.op_type != "Gather"
            or attr(gather, "axis", 0) != 0
            or len(gather.input) != 2
            or not np.array_equal(constants.get(gather.input[1]), [0, 1])
            or shape is None
            or shape.domain
            or shape.op_type != "Shape"
            or shape.attribute
            or shape.input[0] not in prefixes
            or width is None
            or width.shape != (1,)
            or width.dtype != np.int64
            or width[0] <= 0
        ):
            continue
        key = (prefixes[shape.input[0]], int(width[0]))
        if key in seen and not protected.intersection(node.output):
            aliases[node.output[0]] = seen[key]
        else:
            seen.setdefault(key, node.output[0])
    if aliases:
        nodes = [node for node in model.graph.node if not any(name in aliases for name in node.output)]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".shared_cache_shapes.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(aliases)


def deduplicate_standard_expressions(path):
    """Reuse exact deterministic cast/layout expressions without changing arithmetic."""
    model = onnx.load(path, load_external_data=False)
    reusable = {"Cast", "Shape", "Gather", "Concat", "Reshape", "Expand", "Unsqueeze", "Squeeze", "Slice", "Constant"}
    protected = externally_observed_values(model.graph)

    aliases, seen, counts = {}, {}, Counter()
    nodes = []
    for node in model.graph.node:
        node.input[:] = [aliases.get(name, name) for name in node.input]
        if node.domain or node.op_type not in reusable or not node.output or any(not name for name in node.output):
            nodes.append(node)
            continue
        key = (
            node.op_type,
            tuple(node.input),
            len(node.output),
            tuple(
                sorted(
                    (attribute.name, attribute.SerializeToString(deterministic=True)) for attribute in node.attribute
                )
            ),
        )
        previous = seen.get(key)
        if previous is not None and not protected.intersection(node.output):
            aliases.update(zip(node.output, previous, strict=False))
            counts[node.op_type] += 1
        else:
            seen.setdefault(key, tuple(node.output))
            nodes.append(node)
    if counts:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        values = [value for value in model.graph.value_info if value.name not in aliases]
        del model.graph.value_info[:]
        model.graph.value_info.extend(values)
        temporary = path.with_suffix(".deduplicated.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return dict(counts)


def infer_value_types(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type for v in (*inferred.graph.input, *inferred.graph.value_info)}
    types.update((t.name, t.data_type) for t in model.graph.initializer)
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "FoldMid2" and len(node.output) == 7:
            for name in node.output[4:6]:
                if types.get(name, TensorProto.BFLOAT16) not in (0, TensorProto.BFLOAT16):
                    raise ValueError(f"FoldMid2 rotary outputs require BF16: {name}")
                types[name] = TensorProto.BFLOAT16
        if node.domain == "com.microsoft" and node.op_type == "SigmoidGateMul":
            for name in (*node.input, *node.output):
                if not name:
                    continue
                if types.get(name, TensorProto.BFLOAT16) not in (0, TensorProto.BFLOAT16):
                    raise ValueError(f"SigmoidGateMul requires BF16: {name}")
                types[name] = TensorProto.BFLOAT16
    preserving = {
        "": {"Gather", "Concat", "Split", "Reshape", "Slice", "Mul", "Sigmoid"},
        "com.microsoft": {
            "MatMulBlockQuantizedFp8Weight",
            "HyperConnectionLowRankMix",
            "HyperConnectionPostMix",
            "BranchwiseRMSNorm",
            "SigmoidGateMul",
            "CausalConvWithState",
        },
    }
    for node in model.graph.node:
        if (
            node.domain == "com.microsoft"
            and node.op_type == "HyperConnectionLowRankMix"
            and len(node.input) > 2
            and types.get(node.input[2])
        ):
            for output in node.output:
                types[output] = types[node.input[2]]
        if node.op_type in preserving.get(node.domain, set()) and types.get(node.input[0]):
            for output in node.output:
                types[output] = types[node.input[0]]
        if node.domain == "com.microsoft" and node.op_type == "BranchwiseRMSNorm" and len(node.output) > 1:
            types[node.output[1]] = TensorProto.FLOAT
    return types


def fuse_gdn_gate_casts(path):
    """Load losslessly widened activation-type gates directly in the CUDA recurrence."""
    model = onnx.load(path, load_external_data=False)
    types = infer_value_types(model)
    producers = {name: node for node in model.graph.node for name in node.output}
    users = Counter(name for node in model.graph.node for name in node.input)
    outputs = externally_observed_values(model.graph)
    removed, replacements = set(), {}
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "GatedDeltaNet":
            continue
        dtype = types.get(node.input[0])
        if dtype not in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
            continue
        casts = []
        for index in (4, 5):
            if len(node.input) <= index or not node.input[index]:
                continue
            cast = producers.get(node.input[index])
            if (
                cast is None
                or cast.domain
                or cast.op_type != "Cast"
                or next((attr.i for attr in cast.attribute if attr.name == "to"), None) != TensorProto.FLOAT
                or types.get(cast.input[0]) != dtype
            ):
                break
            casts.append((index, cast))
        else:
            if not casts:
                continue
            fused = copy.deepcopy(node)
            for index, cast in casts:
                fused.input[index] = cast.input[0]
                if users[cast.output[0]] == 1 and cast.output[0] not in outputs:
                    removed.add(cast.output[0])
            replacements[node.output[0]] = fused
    if replacements:
        nodes = [
            replacements.get(node.output[0], node) if node.output else node
            for node in model.graph.node
            if not node.output or node.output[0] not in removed
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        temporary = path.with_suffix(".gdn_gate_casts.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(removed)


def fuse_silu_gate(path):
    """Fuse BF16 Sigmoid/Mul/Mul while preserving each intermediate rounding boundary."""
    model = onnx.load(path, load_external_data=False)
    types = infer_value_types(model)
    producers = {out: n for n in model.graph.node for out in n.output}
    users = Counter(name for n in model.graph.node for name in n.input)
    outputs = externally_observed_values(model.graph)
    removed, replacements = set(), {}
    for node in model.graph.node:
        if node.domain or node.op_type != "Mul":
            continue
        for side in (0, 1):
            silu = producers.get(node.input[side])
            if silu is None or silu.domain or silu.op_type != "Mul":
                continue
            sigmoid = producers.get(silu.input[1])
            if (
                sigmoid is None
                or sigmoid.domain
                or sigmoid.op_type != "Sigmoid"
                or sigmoid.input[0] != silu.input[0]
                or types.get(silu.input[0]) != TensorProto.BFLOAT16
                or types.get(node.input[1 - side]) != TensorProto.BFLOAT16
            ):
                continue
            gate_source = producers.get(silu.input[0])
            up_source = producers.get(node.input[1 - side])
            # A shared Split proves identical token axes. Equal constant split sizes prove equal widths.
            if (
                gate_source is None
                or gate_source is not up_source
                or gate_source.op_type != "Split"
                or gate_source.domain
            ):
                continue
            sizes = next(
                (t for t in model.graph.initializer if len(gate_source.input) == 2 and t.name == gate_source.input[1]),
                None,
            )
            if sizes is None or sizes.data_location == TensorProto.EXTERNAL:
                continue
            values = numpy_helper.to_array(sizes)
            gate_index = list(gate_source.output).index(silu.input[0])
            up_index = list(gate_source.output).index(node.input[1 - side])
            if len(values) != len(gate_source.output) or values[gate_index] != values[up_index]:
                continue
            if any(id(n) in removed or users[n.output[0]] != 1 or n.output[0] in outputs for n in (silu, sigmoid)):
                continue
            replacements[id(node)] = helper.make_node(
                "SigmoidGateMul",
                [node.input[1 - side], silu.input[0]],
                list(node.output),
                name=node.name + "_SiluGate",
                domain="com.microsoft",
                silu_gate=1,
                sigmoid_in_fp32=0,
            )
            removed.update(id(n) for n in (sigmoid, silu, node))
            break
    if replacements:
        nodes = [replacements.get(id(n), n) for n in model.graph.node if id(n) not in removed or id(n) in replacements]
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        ensure_opset_import(model)
        temporary = path.with_suffix(".optimized.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def fuse_shared_expert_residual(path):
    """Fuse scalar-per-row sigmoid gating and residual addition, retaining BF16 rounding."""
    model = onnx.load(path, load_external_data=False)
    tensors = {t.name: t for t in model.graph.initializer}
    producers = {out: n for n in model.graph.node for out in n.output}
    users = Counter(name for n in model.graph.node for name in n.input)
    outputs = externally_observed_values(model.graph)
    removed, replacements = set(), {}

    def prefix_root(name):
        node = producers.get(name)
        if node is None:
            return name
        if (
            (node.op_type == "MatMul" and node.domain == "")
            or (node.op_type == "MatMulBlockQuantizedFp8Weight" and node.domain == "com.microsoft")
        ) and len(node.input) >= 2:
            weight = tensors.get(node.input[1])
            if weight is not None and len(weight.dims) == 2:
                return prefix_root(node.input[0])
        if node.domain == "" and node.op_type == "Sigmoid":
            return prefix_root(node.input[0])
        if node.domain == "" and node.op_type == "Split":
            axis = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "axis"), 0)
            if axis == -1:
                return prefix_root(node.input[0])
        if node.domain == "" and node.op_type == "Mul":
            left, right = (prefix_root(x) for x in node.input)
            if left == right:
                return left
        return name

    for node in model.graph.node:
        if node.op_type != "Add" or node.domain or len(node.input) != 2:
            continue
        routed, contribution = (producers.get(x) for x in node.input)
        if (
            routed is None
            or routed.op_type != "QMoE"
            or routed.domain != "com.microsoft"
            or contribution is None
            or contribution.op_type != "Mul"
            or contribution.domain
        ):
            continue
        down, sigmoid = (producers.get(x) for x in contribution.input)
        if (
            down is None
            or down.op_type != "MatMulBlockQuantizedFp8Weight"
            or down.domain != "com.microsoft"
            or sigmoid is None
            or sigmoid.op_type != "Sigmoid"
            or sigmoid.domain
        ):
            continue
        gate = producers.get(sigmoid.input[0])
        if gate is None or gate.op_type != "MatMul" or gate.domain or gate.input[0] != routed.input[0]:
            continue
        gate_weight, down_weight = tensors.get(gate.input[1]), tensors.get(down.input[1])
        if (
            gate_weight is None
            or down_weight is None
            or len(gate_weight.dims) != 2
            or len(down_weight.dims) != 2
            or gate_weight.data_type != TensorProto.BFLOAT16
            or gate_weight.dims[1] != 1
            or gate_weight.dims[0] != down_weight.dims[0]
            or prefix_root(down.input[0]) != prefix_root(gate.input[0])
        ):
            continue
        internals = (sigmoid, contribution)
        if any(users[n.output[0]] != 1 or n.output[0] in outputs or id(n) in removed for n in internals):
            continue
        fused = helper.make_node(
            "SigmoidGateMul",
            [down.output[0], gate.output[0], routed.output[0]],
            list(node.output),
            name=node.name + "_SigmoidGateMul",
            domain="com.microsoft",
            broadcast_gate=1,
            sigmoid_in_fp32=0,
        )
        replacements[id(node)] = fused
        removed.update(id(n) for n in (*internals, node))
    if replacements:
        nodes = [replacements.get(id(n), n) for n in model.graph.node if id(n) not in removed or id(n) in replacements]
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        temporary = path.with_suffix(".optimized.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def fuse_attention_rotary(path):
    """Reuse FP32 rotary kernels while preserving normalization and BF16 rounding boundaries."""
    model = onnx.load(path, load_external_data=False)
    graph_names = graph_value_names(model.graph)
    constants = {
        t.name: numpy_helper.to_array(t) for t in model.graph.initializer if t.data_location != TensorProto.EXTERNAL
    }
    inference_model = copy.deepcopy(model)
    for node in inference_model.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "FoldMid2" and len(node.output) == 7:
            # FoldMid2's CUDA output contract is fixed, independently of its mutable control state.
            for name in node.output[4:6]:
                for info in (
                    *inference_model.graph.input,
                    *inference_model.graph.value_info,
                    *inference_model.graph.output,
                ):
                    if info.name != name:
                        continue
                    tensor = info.type.tensor_type
                    if tensor.elem_type not in (0, TensorProto.BFLOAT16):
                        raise ValueError(f"FoldMid2 rotary outputs require BF16: {name}")
                    if tensor.HasField("shape") and (
                        len(tensor.shape.dim) != 3
                        or any(
                            dim.HasField("dim_value") and dim.dim_value != expected
                            for dim, expected in zip(tensor.shape.dim, (1, 2, 32), strict=False)
                        )
                    ):
                        raise ValueError(f"FoldMid2 rotary outputs require shape [1, 2, 32]: {name}")
                existing = [info for info in inference_model.graph.value_info if info.name != name]
                inference_model.graph.ClearField("value_info")
                inference_model.graph.value_info.extend(existing)
                inference_model.graph.value_info.append(
                    helper.make_tensor_value_info(name, TensorProto.BFLOAT16, [1, 2, 32])
                )
    inferred = onnx.shape_inference.infer_shapes(inference_model)
    shapes = {v.name: v.type.tensor_type.shape for v in (*inferred.graph.input, *inferred.graph.value_info)}
    tensors = {t.name: t for t in model.graph.initializer}
    producers = {out: n for n in model.graph.node for out in n.output}
    users = defaultdict(set)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            users[name].add(index)
    graph_outputs = externally_observed_values(model.graph)
    indices = {id(n): i for i, n in enumerate(model.graph.node)}
    removed, replacements, cache_views = set(), {}, set()
    fused_count = 0
    norm_layouts = {}

    def attr(node, name, default=None):
        return next((helper.get_attribute_value(a) for a in node.attribute if a.name == name), default)

    def producer(name, kind):
        node = producers.get(name)
        return node if node is not None and node.op_type == kind and node.domain == "" else None

    def constant(name, values):
        return name in constants and np.array_equal(constants[name], values)

    def internal(nodes, boundary):
        selected = {indices[id(n)] for n in nodes}
        return not selected.intersection(removed) and all(
            out == boundary or (out not in graph_outputs and users[out].issubset(selected))
            for n in nodes
            for out in n.output
        )

    def input_layout(name):
        cast = producer(name, "Cast")
        if cast is None or attr(cast, "to") != TensorProto.FLOAT:
            return None
        selected = producer(cast.input[0], "Slice")
        reshape = producer(selected.input[0] if selected is not None else cast.input[0], "Reshape")
        shape = constants.get(reshape.input[1]) if reshape is not None else None
        if shape is None or shape.shape != (4,) or list(shape[:2]) != [0, 0] or min(shape[2:]) <= 0:
            return None
        width = int(shape[3])
        if selected is not None:
            ends = constants.get(selected.input[2])
            if ends is None or ends.shape != (1,) or not 0 < ends[0] <= width:
                return None
            width = int(ends[0])
            if not slice_spec(selected, reshape.output[0], 0, width):
                return None
        return int(shape[2]), width

    def slice_spec(node, source, start, end):
        return (
            node is not None
            and len(node.input) in (4, 5)
            and node.input[0] == source
            and constant(node.input[1], [start])
            and constant(node.input[2], [end])
            and constant(node.input[3], [3])
            and (len(node.input) == 4 or constant(node.input[4], [1]))
        )

    for _index, node in enumerate(model.graph.node):
        if node.op_type != "Mul" or node.domain or len(node.input) != 2:
            continue
        divide = producer(node.input[0], "Div")
        gamma = tensors.get(node.input[1])
        if divide is None or gamma is None or gamma.data_type != TensorProto.FLOAT or len(gamma.dims) != 1:
            continue
        root = producer(divide.input[1], "Sqrt")
        add = producer(root.input[0], "Add") if root is not None else None
        mean = producer(add.input[0], "ReduceMean") if add is not None else None
        square = producer(mean.input[0], "Mul") if mean is not None else None
        if square is None or list(square.input) != [divide.input[0], divide.input[0]]:
            continue
        if len(mean.input) != 2 or not constant(mean.input[1], [-1]) or attr(mean, "keepdims", 1) != 1:
            continue
        epsilon = constants.get(add.input[1])
        cast = producer(divide.input[0], "Cast")
        layout = input_layout(divide.input[0])
        if (
            epsilon is None
            or epsilon.size != 1
            or not np.isfinite(epsilon).all()
            or float(epsilon.item()) < 0
            or cast is None
            or attr(cast, "to") != TensorProto.FLOAT
            or layout is None
            or gamma.dims[0] != layout[1]
        ):
            continue
        norm_layouts[node.output[0]] = layout

    def head_count(source, width):
        cast_float = producer(source, "Cast")
        cast_bf16 = producer(cast_float.input[0], "Cast") if cast_float is not None else None
        layout = norm_layouts.get(cast_bf16.input[0]) if cast_bf16 is not None else None
        if (
            layout is None
            or attr(cast_float, "to") != TensorProto.FLOAT
            or attr(cast_bf16, "to") != TensorProto.BFLOAT16
        ):
            return None
        return layout[0] if layout[1] == width else None

    for index, node in enumerate(model.graph.node):
        if node.op_type != "Concat" or node.domain or len(node.input) != 3 or attr(node, "axis") != 3:
            continue
        first = producer(node.input[0], "Sub")
        second = producer(node.input[1], "Add")
        tail = producer(node.input[2], "Slice")
        if first is None or second is None or tail is None:
            continue
        ac, bs = (producer(x, "Mul") for x in first.input)
        bc, ass = (producer(x, "Mul") for x in second.input)
        if any(n is None for n in (ac, bs, bc, ass)):
            continue
        a, b = producer(ac.input[0], "Slice"), producer(bs.input[0], "Slice")
        cosine, sine = producer(ac.input[1], "Unsqueeze"), producer(bs.input[1], "Unsqueeze")
        if a is None or b is None or cosine is None or sine is None:
            continue
        ends, tail_ends = constants.get(a.input[2]), constants.get(tail.input[2])
        if ends is None or ends.shape != (1,) or tail_ends is None or tail_ends.shape != (1,):
            continue
        half, width = int(ends[0]), int(tail_ends[0])
        source = a.input[0]
        if (
            half <= 0
            or width < 2 * half
            or not slice_spec(a, source, 0, half)
            or not slice_spec(b, source, half, 2 * half)
            or not slice_spec(tail, source, 2 * half, width)
            or list(bc.input) != [b.output[0], cosine.output[0]]
            or list(ass.input) != [a.output[0], sine.output[0]]
            or not constant(cosine.input[1], [2])
            or not constant(sine.input[1], [2])
        ):
            continue
        heads = head_count(source, width)
        caches = [shapes.get(n.input[0]) for n in (cosine, sine)]
        nodes = [a, b, tail, ac, bs, bc, ass, first, second, node]
        if (
            heads is None
            or any(s is None or len(s.dim) != 3 or s.dim[2].dim_value != half for s in caches)
            or not internal(nodes, node.output[0])
        ):
            continue
        prefix = f"{node.name or 'Concat'}_{index}_Rotary"
        generated_suffixes = (
            "_flat_shape",
            "_cache_shape",
            "_position",
            "_token_axes",
            "_cache_width",
            "_shape",
            "_token_shape",
            "_expanded_shape",
            "_input",
            "_expanded_cos",
            "_expanded_sin",
            "_cos",
            "_sin",
            "_output",
        )
        while prefix in graph_names or any(prefix + suffix in graph_names for suffix in generated_suffixes):
            prefix += "_"
        graph_names.add(prefix)
        graph_names.update(prefix + suffix for suffix in generated_suffixes)
        names = [
            prefix + suffix for suffix in ("_flat_shape", "_cache_shape", "_position", "_token_axes", "_cache_width")
        ]
        values = ([1, -1, heads * width], [-1, half], [0], [0, 1], [half])
        for name, value in zip(names, values, strict=False):
            model.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name))
        replacements[index] = [
            helper.make_node("Shape", [source], [prefix + "_shape"]),
            helper.make_node("Gather", [prefix + "_shape", names[3]], [prefix + "_token_shape"], axis=0),
            helper.make_node("Concat", [prefix + "_token_shape", names[4]], [prefix + "_expanded_shape"], axis=0),
            helper.make_node("Reshape", [source, names[0]], [prefix + "_input"]),
            helper.make_node("Expand", [cosine.input[0], prefix + "_expanded_shape"], [prefix + "_expanded_cos"]),
            helper.make_node("Expand", [sine.input[0], prefix + "_expanded_shape"], [prefix + "_expanded_sin"]),
            helper.make_node("Reshape", [prefix + "_expanded_cos", names[1]], [prefix + "_cos"]),
            helper.make_node("Reshape", [prefix + "_expanded_sin", names[1]], [prefix + "_sin"]),
            helper.make_node(
                "RotaryEmbedding",
                [prefix + "_input", names[2], prefix + "_cos", prefix + "_sin"],
                [prefix + "_output"],
                name=prefix,
                domain="com.microsoft",
                num_heads=heads,
                rotary_embedding_dim=2 * half,
                interleaved=0,
            ),
            helper.make_node("Reshape", [prefix + "_output", prefix + "_shape"], list(node.output)),
        ]
        removed.update(indices[id(n)] for n in nodes)
        cache_views.update(indices[id(n)] for n in (cosine, sine))
        fused_count += 1
    if fused_count:
        nodes = [
            replacement
            for i, node in enumerate(model.graph.node)
            for replacement in replacements.get(i, [] if i in removed else [node])
        ]
        live = {name for node in nodes for name in node.input} | graph_outputs
        dead_views = {model.graph.node[i].output[0] for i in cache_views if model.graph.node[i].output[0] not in live}
        nodes = [node for node in nodes if not node.output or node.output[0] not in dead_views]
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        ensure_opset_import(model)
        temporary = path.with_suffix(".optimized.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return fused_count


def fuse_causal_conv_state_updates(path):
    """Absorb channels-first carry layouts and first-token snapshots into CUDA convolution."""
    model = onnx.load(path, load_external_data=False)
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer if not t.external_data}
    tensors = {t.name: t for t in model.graph.initializer}
    shapes = {v.name: v.type.tensor_type.shape for v in list(model.graph.input) + list(model.graph.value_info)}
    protected = externally_observed_values(model.graph)
    replacements, removed = {}, set()

    def attr(node, name, default=None):
        return next((helper.get_attribute_value(a) for a in node.attribute if a.name == name), default)

    def standard(node, kind):
        return node is not None and node.domain in ("", "ai.onnx") and node.op_type == kind

    def rank(name, visiting=None):
        visiting = set() if visiting is None else visiting
        if name in visiting:
            return None
        visiting.add(name)
        if name in shapes:
            return len(shapes[name].dim) or None
        node = producers.get(name)
        if node is None or not node.input:
            return None
        preserves_rank = (
            (
                node.domain == "com.microsoft"
                and node.op_type in {"BranchwiseRMSNorm", "DotProductGate", "MatMulBlockQuantizedFp8Weight"}
            )
            or standard(node, "Identity")
            or standard(node, "Cast")
        )
        return rank(node.input[0], visiting) if preserves_rank else None

    def sole_user(name, kind):
        users = consumers[name]
        return users[0] if len(users) == 1 and standard(users[0], kind) else None

    def slice_spec(node):
        if not standard(node, "Slice") or len(node.input) not in (4, 5):
            return None
        arrays = [constants.get(name) for name in node.input[1:]]
        if any(a is None or a.shape != (1,) or a.dtype not in (np.int32, np.int64) for a in arrays):
            return None
        values = [int(a.item()) for a in arrays]
        return tuple(values) if len(values) == 4 else (*values, 1)

    for conv in model.graph.node:
        if (
            conv.domain != "com.microsoft"
            or conv.op_type != "CausalConvWithState"
            or len(conv.input) != 4
            or len(conv.output) != 2
            or attr(conv, "channels_last", 0) != 1
            or attr(conv, "ndim", 1) != 1
            or any(
                attr(conv, name, default) != default
                for name, default in (
                    ("state_window", 0),
                    ("input_channels", 0),
                    ("input_channel_offset", 0),
                    ("state_length", 0),
                    ("state_channels_last", -1),
                )
            )
            or rank(conv.input[0]) != 3
        ):
            continue
        state_in = producers.get(conv.input[3])
        state_out = sole_user(conv.output[1], "Transpose")
        if any(not standard(node, "Transpose") or attr(node, "perm") != [0, 2, 1] for node in (state_in, state_out)):
            continue
        past = state_in.input[0]
        shape = shapes.get(past)
        weight = tensors.get(conv.input[1])
        dilation = attr(conv, "dilation", 1)
        if (
            shape is None
            or len(shape.dim) != 3
            or not all(shape.dim[i].HasField("dim_value") for i in (1, 2))
            or weight is None
            or len(weight.dims) != 3
            or weight.dims[1] != 1
            or dilation <= 0
        ):
            continue
        channels, history = (shape.dim[i].dim_value for i in (1, 2))
        if channels <= 0 or history <= 0 or weight.dims[0] != channels or (weight.dims[2] - 1) * dilation != history:
            continue
        tails = [n for n in consumers[past] if slice_spec(n) == (1, 2**31 - 1, 2, 1)]
        firsts = [n for n in consumers[conv.input[0]] if slice_spec(n) == (0, 1, 1, 1)]
        if len(tails) != 1 or len(firsts) != 1:
            continue
        tail, first = tails[0], firsts[0]
        first_transpose = sole_user(first.output[0], "Transpose")
        if first_transpose is None or attr(first_transpose, "perm") != [0, 2, 1]:
            continue
        mid = sole_user(first_transpose.output[0], "Concat")
        if mid is None or attr(mid, "axis") != 2 or list(mid.input) != [tail.output[0], first_transpose.output[0]]:
            continue
        chain = [conv, state_in, state_out, tail, first, first_transpose, mid]
        ids = {id(node) for node in chain}
        retained = {conv.output[0], state_out.output[0], mid.output[0]}
        if ids & removed or any(
            name not in retained and (name in protected or any(id(user) not in ids for user in consumers[name]))
            for node in chain
            for name in node.output
        ):
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
        attributes.update(input_channels=channels, state_length=history, state_channels_last=0)
        replacements[id(conv)] = helper.make_node(
            "CausalConvWithState",
            [conv.input[0], conv.input[1], conv.input[2], past],
            [conv.output[0], state_out.output[0], mid.output[0]],
            domain="com.microsoft",
            name=conv.name + "_FusedStateUpdates",
            **attributes,
        )
        removed.update(ids)
    if replacements:
        nodes = [replacements.get(id(n), n) for n in model.graph.node if id(n) not in removed or id(n) in replacements]
        live = {name for node in nodes for name in (*node.input, *node.output)} | protected
        values = [v for v in model.graph.value_info if v.name in live]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        model.graph.value_info.extend(values)
        temporary = path.with_suffix(".conv_state_updates.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def fuse_causal_conv_projection(path):
    """Fold channel selection, layout changes and full/first carry updates into CUDA convolution."""
    model = onnx.load(path, load_external_data=False)
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer if not t.external_data}
    shapes = {v.name: v.type.tensor_type.shape for v in list(model.graph.input) + list(model.graph.value_info)}
    graph_outputs = externally_observed_values(model.graph)
    removed, replacements = set(), {}

    def attr(node, name, default=None):
        return next((helper.get_attribute_value(a) for a in node.attribute if a.name == name), default)

    def standard(node, op):
        return node is not None and node.op_type == op and node.domain in ("", "ai.onnx")

    def slice_spec(node):
        if not standard(node, "Slice") or len(node.input) not in (4, 5):
            return None
        arrays = [constants.get(name) for name in node.input[1:]]
        if any(a is None or a.size != 1 or a.dtype.kind not in "iu" for a in arrays):
            return None
        values = [int(a.item()) for a in arrays]
        return tuple(values) if len(values) == 4 else (*values, 1)

    def sole_user(name, op):
        users = consumers[name]
        return users[0] if len(users) == 1 and standard(users[0], op) else None

    for conv in model.graph.node:
        if conv.domain != "com.microsoft" or conv.op_type != "CausalConvWithState" or len(conv.input) != 4:
            continue
        if (
            attr(conv, "channels_last", 0) != 0
            or attr(conv, "ndim", 1) != 1
            or attr(conv, "state_window", 0) != 0
            or any(
                attr(conv, name, default) != default
                for name, default in (
                    ("input_channels", 0),
                    ("input_channel_offset", 0),
                    ("state_length", 0),
                    ("state_channels_last", -1),
                )
            )
        ):
            continue
        transpose = producers.get(conv.input[0])
        tail = producers.get(conv.input[3])
        if not standard(transpose, "Transpose") or attr(transpose, "perm") != [0, 2, 1]:
            continue
        selection = producers.get(transpose.input[0])
        spec, tail_spec = slice_spec(selection), slice_spec(tail)
        if spec is None or spec[2:] != (2, 1) or spec[0] < 0 or spec[1] <= spec[0]:
            continue
        if tail_spec is None or tail_spec[2:] != (2, 1) or tail_spec[1] < 2**30 or tail_spec[0] >= 0:
            continue
        past = tail.input[0]
        state_shape = shapes.get(past)
        if state_shape is None or len(state_shape.dim) != 3 or not state_shape.dim[2].HasField("dim_value"):
            continue
        history = state_shape.dim[2].dim_value
        channels = spec[1] - spec[0]
        if history <= 0 or not state_shape.dim[1].HasField("dim_value") or state_shape.dim[1].dim_value != channels:
            continue
        weight = next((t for t in model.graph.initializer if t.name == conv.input[1]), None)
        if weight is None or list(weight.dims[:2]) != [channels, 1] or len(weight.dims) != 3:
            continue
        pad = (weight.dims[2] - 1) * attr(conv, "dilation", 1)
        if pad <= 0 or tail_spec[0] != -pad or history < pad:
            continue
        output_transpose = sole_user(conv.output[0], "Transpose")
        if output_transpose is None or attr(output_transpose, "perm") != [0, 2, 1]:
            continue
        histories = [
            n
            for n in consumers[transpose.output[0]]
            if standard(n, "Concat") and list(n.input) == [past, transpose.output[0]] and attr(n, "axis") == 2
        ]
        first_slices = [n for n in consumers[transpose.output[0]] if slice_spec(n) == (0, 1, 2, 1)]
        if len(histories) != 1 or len(first_slices) != 1:
            continue
        full, first = histories[0], first_slices[0]
        mid = sole_user(first.output[0], "Concat")
        if mid is None or list(mid.input) != [past, first.output[0]] or attr(mid, "axis") != 2:
            continue
        final_slice, mid_slice = sole_user(full.output[0], "Slice"), sole_user(mid.output[0], "Slice")
        if any(slice_spec(n) != (-history, 2**30, 2, 1) for n in (final_slice, mid_slice)):
            continue
        nodes = [selection, transpose, tail, conv, output_transpose, full, first, mid, final_slice, mid_slice]
        ids = {id(n) for n in nodes}
        retained_outputs = {output_transpose.output[0], final_slice.output[0], mid_slice.output[0]}
        if any(
            name not in retained_outputs
            and (name in graph_outputs or any(id(user) not in ids for user in consumers[name]))
            for node in nodes
            for name in node.output
        ):
            continue
        earlier = []
        for node in model.graph.node:
            if id(node) == id(conv):
                break
            if id(node) not in ids:
                earlier.append(node)
        early_graph = helper.make_graph(earlier, "earlier_consumers", [], [])
        early_inputs = {name for node in earlier for name in node.input} | captured_values(early_graph)
        if retained_outputs & early_inputs:
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
        attributes.update(
            channels_last=1,
            input_channels=channels,
            input_channel_offset=spec[0],
            state_length=history,
            state_channels_last=0,
        )
        fused = helper.make_node(
            "CausalConvWithState",
            [selection.input[0], conv.input[1], conv.input[2], past],
            [output_transpose.output[0], final_slice.output[0], mid_slice.output[0]],
            domain="com.microsoft",
            name=conv.name + "_projection",
            **attributes,
        )
        replacements[id(conv)] = fused
        removed.update(ids)
    if replacements:
        nodes = [replacements.get(id(n), n) for n in model.graph.node if id(n) not in removed or id(n) in replacements]
        retained_names = {name for n in nodes for name in (*n.input, *n.output)} | graph_outputs
        value_info = [v for v in model.graph.value_info if v.name in retained_names]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        model.graph.value_info.extend(value_info)
        temporary = path.with_suffix(".conv_projection.onnx")
        onnx.save(model, temporary)
        os.replace(temporary, path)
    return len(replacements)


def write_performance_metadata(destination, source, results, pool, provider, runtime, genai_library, benchmark=None):
    artifacts = {}
    for binary, relative in (
        (provider, "provider/libonnxruntime_providers_cuda.so"),
        (runtime, "runtime/libonnxruntime.so"),
        (genai_library, "runtime/libonnxruntime-genai.so"),
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".optimized.so")
        shutil.copyfile(binary, temporary)
        os.replace(temporary, target)
        with target.open("rb") as stream:
            artifacts[relative] = hashlib.file_digest(stream, "sha256").hexdigest()

    config_path = destination / "genai_config.json"
    config = json.loads(config_path.read_text())
    graph_config = config["model"]["qwen38_graph"]
    graph_config["warmup"] = True
    graph_config["prefill"]["enabled"] = True
    graph_config["prefill"]["chunk"] = 512
    graph_config["prefill"]["warmup_lengths"] = [512, 128, 37]
    for provider_options in config["model"]["decoder"]["session_options"]["provider_options"]:
        if "cuda" in provider_options:
            provider_options["cuda"]["enable_cuda_graph"] = "1"
    temporary = config_path.with_suffix(".optimized.json")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(temporary, config_path)

    manifest_path = destination / "package_manifest.json"
    manifest = copy.deepcopy(json.loads(manifest_path.read_text()))
    manifest["performance_rewrites"] = results
    manifest["performance_source_package"] = str(source)
    manifest["weights_pool"] = pool
    manifest.pop("pool", None)
    manifest["performance_runtime_sha256"] = artifacts
    manifest["provider_sha256"] = {
        "libonnxruntime_providers_cuda.so": artifacts["provider/libonnxruntime_providers_cuda.so"]
    }
    manifest["performance_environment"] = {"ORT_QMOE_NVFP4_PACKED_PREFILL": "1"}
    rewrites = manifest["experimental_graph_rewrites"]
    rewrites["cuda_graph_capture"] = "Enabled with warmup; folded state updates are muted during capture setup."
    rewrites["SkinnyGemm"] = "com.microsoft::MatMulBlockQuantizedFp8Weight for eligible target FP8 projections."
    rewrites["HcBlock"] = "com.microsoft::HyperConnectionLowRankMix with offline constant untiling."
    rewrites["HcPostBlock"] = (
        "HyperConnectionLowRankMix identity post-mix prologue where eligible; "
        "otherwise HyperConnectionPostMix followed by HyperConnectionLowRankMix."
    )
    if any(result.get("fused_attention_rotary", 0) for result in results.values()):
        rewrites["QsaProjPrep"] = (
            "Standard slicing and RMS normalization with com.microsoft::RotaryEmbedding for eligible "
            "FP32 rotary chains; BF16 normalization/output rounding and cache broadcasting are preserved."
        )
    if any(result.get("fused_shared_expert_residual", 0) for result in results.values()):
        rewrites["SharedExpertDown"] = (
            "MatMulBlockQuantizedFp8Weight plus standard SiLU and shared-gate MatMul; eligible sigmoid, "
            "multiply and residual-add chains use SigmoidGateMul with broadcast_gate=1 and sigmoid_in_fp32=0."
        )
    if any(result.get("fused_silu_gate", 0) for result in results.values()):
        rewrites["SharedExpertSiLU"] = (
            "SigmoidGateMul with silu_gate=1 and sigmoid_in_fp32=0 replaces eligible Sigmoid/Mul/Mul "
            "chains, preserving BF16 sigmoid, SiLU and gated-up rounding boundaries."
        )
    if any(result.get("fused_fp8_projection_split", 0) for result in results.values()):
        rewrites["Fp8ProjectionSplit"] = (
            "MatMulBlockQuantizedFp8Weight split_sizes replaces eligible last-axis Split nodes; "
            "tensor-core decode stores directly into partition outputs with unchanged arithmetic."
        )
    if any(result.get("fused_fp8_projection_slices", 0) for result in results.values()):
        rewrites["Fp8ProjectionSlices"] = (
            "MatMulBlockQuantizedFp8Weight split_sizes supplies complete convolution/attention Slice "
            "partitions directly; indexed convolution or packed gating is rebased to its partition."
        )
    if any(result.get("fused_causal_conv_state_updates", 0) for result in results.values()):
        rewrites["CausalConvStateUpdates"] = (
            "CausalConvWithState writes channels-first final and first-token carry directly, "
            "replacing private state transposes and first-token Slice/Concat chains."
        )
    if any(result.get("fused_gdn_gate_casts", 0) for result in results.values()):
        rewrites["GdnGateCasts"] = (
            "GatedDeltaNet directly loads FP16/BF16 decay and beta, losslessly widening to FP32 "
            "before existing gate arithmetic instead of launching standalone Cast kernels."
        )
    if any(result.get("deduplicated_standard_expressions", {}) for result in results.values()):
        rewrites["SharedStandardExpressions"] = (
            "Exact deterministic standard Cast/layout expressions share their first result; "
            "operator attributes, rounding boundaries, graph outputs and subgraph captures are preserved."
        )
    if any(result.get("fused_standard_rms_norm", 0) for result in results.values()):
        rewrites["StandardRMSNorm"] = (
            "BranchwiseRMSNorm use_onnx_arithmetic=1 reuses the standard FP32 reduction "
            "and explicit mean/sqrt/divide/multiply rounding boundaries."
        )
    if any(result.get("fused_gdn_qkv_split", 0) for result in results.values()):
        rewrites["GdnQkvSplit"] = (
            "GatedDeltaNet consumes complete convolution Q/K/V partitions directly for short "
            "warp-specialized decode; longer/prefill inputs retain the existing unpack path."
        )
    if any(result.get("fused_rms_output_casts", 0) for result in results.values()):
        rewrites["RmsOutputCasts"] = (
            "BranchwiseRMSNorm optionally stores a float32 widening of its already-rounded "
            "activation-type output, preserving the rounding boundary without a Cast launch."
        )
    if any(result.get("fused_rms_input_slices", 0) for result in results.values()):
        rewrites["RmsInputSlices"] = (
            "BranchwiseRMSNorm version 2 reads a private last-axis slice directly from packed rows "
            "using input_start/input_width, preserving the standard reduction order and rounding."
        )
    if any(result.get("shared_rotary_cache_shapes", 0) for result in results.values()):
        rewrites["SharedRotaryCaches"] = (
            "Rotary cache expansions share results only after proving equal batch/sequence dimensions "
            "through standard layouts and generic contrib operator contracts."
        )
    if any(result.get("fused_rotary_casts", 0) for result in results.values()):
        rewrites["RotaryCasts"] = (
            "RotaryEmbedding version 2 with use_fp32_arithmetic=1 widens activation-type inputs/caches "
            "inside the kernel, performs separately rounded FP32 arithmetic, and rounds the output once. "
            "Redundant standalone casts and private RMS widened outputs are removed."
        )
    manifest["runtime_validation_blocker"] = None
    findings = manifest["unported_operator_findings"]
    findings["GuardedDSA"] = "Generic DynamicSparseAttention directly consumes INT8 BNSH caches and per-channel scales."
    findings["end_to_end_model_run"] = (
        "See performance benchmark status; full-model numerical parity remains unverified."
    )
    findings["PackedQSAIndexer"] = (
        "Generic PackedSparseAttentionIndexer is retained; warmup safely captures folded graphs with state updates muted."
    )
    counts = Counter()
    for label in ("target", "dense", "head_r2", "head_kv"):
        model = onnx.load(destination / label / "model.onnx", load_external_data=False)
        counts.update(n.op_type for n in model.graph.node if n.domain.startswith("qwen."))
    manifest["remaining_qwen_domain_nodes"] = dict(counts, total=sum(counts.values()))
    manifest["hard_linked_files"] = sum(p.is_file() and p.stat().st_nlink > 1 for p in destination.rglob("*"))
    manifest["benchmark_status"] = "Experimental: benchmark pending; full-model numerical parity is not established."
    if benchmark is not None:
        measured = json.loads(benchmark.read_text())
        if (
            Path(measured["model"]).resolve() != destination.resolve()
            or measured["provider_sha256"] != artifacts["provider/libonnxruntime_providers_cuda.so"]
            or measured["ort_sha256"] != artifacts["runtime/libonnxruntime.so"]
            or not measured["cuda_graph"]
            or not measured["fast_prefill"]
        ):
            raise ValueError("Benchmark does not describe this package and runtime")
        if not measured["rows"] or any(
            not row["runs"]
            or any(
                run["output_tokens"] != measured["output_tokens"]
                or len(run["generated_token_ids"]) != measured["output_tokens"]
                or run["speculative_stats"]["mtp_failures"] != 0
                or run["speculative_stats"]["standard_fallback_steps"] != 0
                for run in row["runs"]
            )
            for row in measured["rows"]
        ):
            raise ValueError("Benchmark contains incomplete generation or MTP failures")
        benchmark_path = destination / "performance_benchmark.json"
        temporary = benchmark_path.with_suffix(".optimized.json")
        temporary.write_text(json.dumps(measured, indent=2) + "\n")
        os.replace(temporary, benchmark_path)
        manifest["benchmark_status"] = (
            "Partial throughput recovery measured with Generator, warmed CUDA graphs and fast prefill; "
            "see performance_benchmark.json. Pre-port throughput and full-model numerical parity are not established."
        )
    temporary = manifest_path.with_suffix(".optimized.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--provider", type=Path, required=True, help="CUDA plugin built with INT8 DynamicSparseAttention"
    )
    parser.add_argument("--runtime", type=Path, required=True, help="Matching libonnxruntime.so with updated schemas")
    parser.add_argument(
        "--genai-library", type=Path, required=True, help="GenAI library with INT8 fast-prefill handoff"
    )
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    provider = args.provider.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    genai_library = args.genai_library.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Destination must be a new package outside the source package")
    shutil.copytree(source, destination, copy_function=os.link)
    scale_sets = {}
    for label, relative in (("target", "target/model.onnx"), ("head", "head_r2/model.onnx")):
        graph = onnx.load(destination / relative, load_external_data=False).graph
        initializers = {t.name: t for t in graph.initializer}
        scales = {}
        for node in graph.node:
            if node.op_type != "DynamicSparseAttention":
                continue
            prefix = node.name.removesuffix("DynamicSparseAttention")
            for kind, short in (("key", "k"), ("value", "v")):
                # The cache conversion wrapper records one FP32 scale for every KV channel.
                cache_name = next(n.input[0] for n in graph.node if n.name == prefix + kind + "_CacheToBSNH")
                scale_name = prefix + short + "_scale"
                output_name = kind if label == "head" else cache_name.replace("past_key_values.", "present.")
                scales[output_name] = numpy_helper.to_array(initializers[scale_name])
        scale_sets[label] = scales
    results = {}
    transformed_provenance = {}
    for relative in ("target/model.onnx", "dense/model.onnx", "head_r2/model.onnx", "head_kv/model.onnx"):
        label = relative.split("/", 1)[0]
        transformed_provenance[label] = {}
        results[relative] = optimize_model(destination / relative, transformed_provenance[label])
        results[relative]["fused_hyper_connections"] = fuse_hyper_connections(destination / relative)
        results[relative]["fused_hyper_connection_postmix"] = fuse_hyper_connection_postmix(destination / relative)
        results[relative]["fused_causal_conv_projection"] = fuse_causal_conv_projection(destination / relative)
        results[relative]["fused_causal_conv_state_updates"] = fuse_causal_conv_state_updates(destination / relative)
        results[relative]["fused_attention_rotary"] = fuse_attention_rotary(destination / relative)
        results[relative]["fused_shared_expert_residual"] = fuse_shared_expert_residual(destination / relative)
        results[relative]["fused_silu_gate"] = fuse_silu_gate(destination / relative)
        results[relative]["fused_fp8_projection_split"] = fuse_fp8_projection_split(destination / relative)
        results[relative]["fused_fp8_projection_slices"] = fuse_fp8_projection_slices(destination / relative)
        results[relative]["fused_gdn_gate_casts"] = fuse_gdn_gate_casts(destination / relative)
        results[relative]["fused_standard_rms_norm"] = fuse_standard_rms_norm(destination / relative)
        results[relative]["fused_rms_input_slices"] = fuse_rms_input_slices(destination / relative)
        results[relative]["fused_gdn_qkv_split"] = fuse_gdn_qkv_split(destination / relative)
        results[relative]["fused_rms_output_casts"] = fuse_rms_output_casts(destination / relative)
    results["dense/model.onnx"]["quantized_prefill_outputs"] = quantize_prefill_outputs(
        destination / "dense/model.onnx", scale_sets["target"]
    )
    results["head_kv/model.onnx"]["quantized_prefill_outputs"] = quantize_prefill_outputs(
        destination / "head_kv/model.onnx", scale_sets["head"], head=True
    )
    for relative, result in results.items():
        result["shared_rotary_cache_shapes"] = share_rotary_cache_expands(destination / relative)
        result["deduplicated_standard_expressions"] = deduplicate_standard_expressions(destination / relative)
        result["fused_rotary_casts"] = fuse_rotary_casts(destination / relative)
    pool = refresh_weight_pool(source, destination, transformed_provenance)
    write_performance_metadata(destination, source, results, pool, provider, runtime, genai_library)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
