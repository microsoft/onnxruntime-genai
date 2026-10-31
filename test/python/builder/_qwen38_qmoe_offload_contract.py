# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A model of the ONNX Runtime QMoE offload contract, for the tests of the Qwen3.8 offload package builder.

It is transcribed from microsoft/onnxruntime#33622 at commit 1d5ae2f3ad7bc9e5ce5579808ee74cc9f93284b9: the ``QMoE`` kernel
(``contrib_ops/cuda/moe/moe_quantization.cc``), the session option parser (``core/framework/moe_offload_plan.h``) and the
planner (``core/providers/cuda/qmoe_session_controller.h``). Released ONNX Runtime has no QMoE offload and ignores the
``session.moe.offload.*`` entries, and the unit tests have no CUDA build of that pull request. ``simulate_session_creation``
therefore reproduces what decides whether session creation fails there: the operators and attributes of the graph, the
session entries, the per-node eligibility, the layer numbers and the node overrides. The byte budgets, which depend on the free
GPU memory, are not modeled.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from pathlib import Path

import onnx
from onnx import helper

OFFLOAD_PREFIX = "session.moe.offload."
INT32_MAX = 2**31 - 1
UINT64_MAX = 2**64 - 1
STRP_MAGIC = 0x50525453

# The attributes of the com.microsoft::QMoE schema; the runtime rejects any other attribute at graph load.
QMOE_ATTRIBUTES = frozenset(
    {
        "activation_type",
        "k",
        "normalize_routing_weights",
        "use_sparse_mixer",
        "expert_weight_bits",
        "fc1_expert_weight_bits",
        "fc2_expert_weight_bits",
        "fc3_expert_weight_bits",
        "swiglu_fusion",
        "swiglu_limit",
        "activation_alpha",
        "activation_beta",
        "accuracy_level",
        "block_size",
        "weights_prepacked",
        "quant_type",
    }
)
UNSUPPORTED_SLOTS = (8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20)
C_WHITESPACE = " \t\n\v\f\r"


class ContractError(Exception):
    """Session creation fails in the runtime."""


def extract_layer_index(name: str) -> int | None:
    """``MoeOffloadSessionPolicy::TryExtractLayerIndexFromName``: the number after "layers.", "layer." or "blocks."."""
    for prefix in ("layers.", "layer.", "blocks."):
        for match in re.finditer(re.escape(prefix) + "([0-9]+)", name):
            if int(match.group(1)) > INT32_MAX:
                return None
            following = name[match.end() : match.end() + 1]
            if not following or not (following.isascii() and following.isalnum()):
                return int(match.group(1))
    return None


def _int32(text: str) -> int | None:
    if re.fullmatch("-?[0-9]+", text) and -(2**31) <= int(text) <= INT32_MAX:
        return int(text)
    return None


def _uint64(text: str) -> int | None:
    if re.fullmatch("[0-9]+", text) and int(text) <= UINT64_MAX:
        return int(text)
    return None


def parse_node_overrides(raw: str) -> list[dict]:
    """``TryParseNodeOverrides``: "name:max_resident_slots=N,device_expert_bytes=B;..." -> one dict per entry."""
    trimmed = raw.strip(C_WHITESPACE)
    if not trimmed or trimmed == "{}":
        return []
    invalid = ContractError(
        f"Invalid {OFFLOAD_PREFIX}node_overrides value '{raw}'; expected semicolon-separated "
        "'node_name:max_resident_slots=<N>,device_expert_bytes=<B>'."
    )
    overrides = []
    for chunk in trimmed.split(";"):
        entry = chunk.strip(C_WHITESPACE)
        if not entry:
            continue
        colon = entry.find(":")
        if colon <= 0 or colon + 1 >= len(entry):
            raise invalid
        override = {"name": entry[:colon].strip(C_WHITESPACE), "max_resident_slots": None, "device_expert_bytes": None}
        if not override["name"]:
            raise invalid
        fields = entry[colon + 1 :].strip(C_WHITESPACE)
        position = 0
        while position < len(fields):
            comma = fields.find(",", position)
            end = len(fields) if comma < 0 else comma
            field = fields[position:end].strip(C_WHITESPACE)
            position = len(fields) if comma < 0 else comma + 1
            equals = field.find("=")
            if not field or equals <= 0 or equals + 1 >= len(field):
                raise invalid
            key, value = field[:equals].strip(C_WHITESPACE), field[equals + 1 :].strip(C_WHITESPACE)
            if key == "max_resident_slots":
                slots = _int32(value)
                if slots is None or slots < 0:
                    raise invalid
                override[key] = slots
            elif key == "device_expert_bytes":
                override[key] = _uint64(value)
                if override[key] is None:
                    raise invalid
            else:
                raise invalid
        if override["max_resident_slots"] is None and override["device_expert_bytes"] is None:
            raise invalid
        overrides.append(override)
    return overrides


def _choice(key: str, value: str, choices: Mapping[str, str], expected: str) -> str:
    if value.lower() not in choices:
        raise ContractError(f"Invalid {key} value '{value}'; expected {expected}.")
    return choices[value.lower()]


def _auto_or_uint64(key: str, value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = _uint64(value)
    if parsed is None:
        raise ContractError(f"Invalid {key} value '{value}'; expected 'auto' or non-negative integer bytes.")
    return parsed


def parse_session_entries(entries: Mapping[str, str], model_dir: Path | None = None) -> dict:
    """``MoeOffloadSessionPolicy::ParseFromConfigOptions``: the policy, or ContractError for an entry it rejects."""
    policy = {
        "mode": "disabled",
        "placement": "adaptive",
        "cpu_execution": "auto",
        "popularity_file": None,
        "node_overrides": [],
    }
    for key, value in entries.items():
        if key.startswith("moe.cpu_offload."):
            raise ContractError(
                f"Prototype configuration key '{key}' is discarded and not supported in QMoE Offload v4; "
                "use 'session.moe.offload.*'."
            )
        if not key.startswith(OFFLOAD_PREFIX):
            continue
        name = key[len(OFFLOAD_PREFIX) :]
        if name == "mode":
            choices = {"disabled": "disabled", "0": "disabled", "off": "disabled", "auto": "auto"}
            choices.update({"required": "required", "1": "required", "on": "required"})
            policy["mode"] = _choice(key, value, choices, "'disabled', 'auto', or 'required'")
        elif name == "cpu_execution":
            choices = {"auto": "auto", "enabled": "enabled", "1": "enabled", "disabled": "disabled", "0": "disabled"}
            policy["cpu_execution"] = _choice(key, value, choices, "'auto', 'enabled', or 'disabled'")
        elif name == "placement":
            policy["placement"] = _choice(
                key, value, {"static": "static", "adaptive": "adaptive"}, "'static' or 'adaptive'"
            )
        elif name in ("device_expert_bytes", "host_expert_bytes", "pinned_bytes", "reserve_device_bytes"):
            policy[name] = _auto_or_uint64(key, value)
        elif name in ("device_working_bytes", "host_working_bytes"):
            if not _uint64(value):
                raise ContractError(f"Invalid {key} value '{value}'; expected positive integer bytes.")
        elif name == "allow_file_backing":
            _choice(key, value, {"1": "", "true": "", "0": "", "false": ""}, "'0' or '1'")
        elif name in ("cpu_threads", "async_max_tokens"):
            if (_int32(value) or -1) < 0 and value != "0":
                raise ContractError(f"Invalid {key} value '{value}'; expected non-negative integer.")
        elif name == "prefill_wave_experts":
            if (_int32(value) or 0) <= 0:
                raise ContractError(f"Invalid {key} value '{value}'; expected positive integer.")
        elif name == "skip_zero_cold_doorbell":
            if value not in ("0", "1"):
                raise ContractError(f"Invalid {key} value '{value}'; expected '0' or '1'.")
        elif name in ("popularity_file", "source_root", "derived_cache_dir"):
            path = Path(value) if value else None
            if path is not None and not path.is_absolute() and model_dir is not None:
                path = model_dir / path
            policy[name] = path
        elif name == "node_overrides":
            policy[name] = parse_node_overrides(value)
        elif name != "budget_group":
            raise ContractError(f"Unknown session.moe.offload.* configuration key: {key}")
    return policy


def _attribute(node: onnx.NodeProto, name: str, default):
    for attribute in node.attribute:
        if attribute.name == name:
            value = helper.get_attribute_value(attribute)
            return value.decode() if isinstance(value, bytes) else value
    return default


def node_is_eligible(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    """The ``node_eligible`` expression of the QMoE kernel constructor."""
    quant_type = _attribute(node, "quant_type", "int")
    if quant_type not in ("int", "fp4", "nvfp4", "fp8", "wfp4afp8"):
        raise ContractError(f"quant_type must be 'int', 'fp4', 'nvfp4', 'fp8', or 'wfp4afp8', but got '{quant_type}'")
    bits = _attribute(node, "expert_weight_bits", 4)
    fc1_bits = _attribute(node, "fc1_expert_weight_bits", bits)
    fc2_bits = _attribute(node, "fc2_expert_weight_bits", bits)
    block_size = _attribute(node, "block_size", -1)
    prepacked_mode = _attribute(node, "weights_prepacked", -1)
    if prepacked_mode not in (-1, 0, 1):
        raise ContractError(f"weights_prepacked must be -1 (default), 0, or 1, but got {prepacked_mode}")
    if quant_type == "nvfp4":
        if block_size not in (-1, 16):
            raise ContractError(f"QMoE quant_type='nvfp4' requires block_size=16, but got {block_size}")
        block_size = 16
    prepacked = prepacked_mode != 0

    nvfp4_row_major = quant_type == "nvfp4" and prepacked_mode == 1
    int4_prepacked = quant_type == "int" and bits == fc1_bits == fc2_bits == 4 and prepacked and block_size == 32
    int3_raw = (
        quant_type == "int" and bits == fc1_bits == fc2_bits == 3 and not prepacked and block_size in (32, 64, 128)
    )

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}

    def constant(slot: int):
        name = node.input[slot] if slot < len(node.input) else ""
        overridable = model.ir_version >= 4 and name in graph_inputs
        return initializers.get(name) if name and not overridable else None

    has_unsupported_slot = any(slot < len(node.input) and node.input[slot] for slot in UNSUPPORTED_SLOTS)
    weights = [constant(slot) for slot in (2, 3, 5, 6)]
    has_const_weights = all(tensor is not None for tensor in weights) and len(weights[0].dims) == 3 == len(
        weights[2].dims
    )
    nvfp4_globals_ok = quant_type != "nvfp4" or (constant(15) is not None and constant(16) is not None)
    interleaved_swiglu = _attribute(node, "activation_type", "relu") == "swiglu" and _attribute(
        node, "swiglu_fusion", 0
    ) in (0, 1)
    return bool(
        (nvfp4_row_major or int4_prepacked or int3_raw)
        and (bits == 4 or int3_raw)
        and interleaved_swiglu
        and _attribute(node, "use_sparse_mixer", 0) != 1
        and not has_unsupported_slot
        and has_const_weights
        and nvfp4_globals_ok
    )


def node_layer_index(node: onnx.NodeProto) -> int | None:
    """The layer number from the node name, else from the name of the fc1 weights input; None when neither has one."""
    layer = extract_layer_index(node.name)
    if layer is None and len(node.input) > 2 and node.input[2]:
        layer = extract_layer_index(node.input[2])
    return layer


def layer_indices(nodes: list[onnx.NodeProto]) -> list[int]:
    """The layer index of each registered node, in registration order, as the session controller assigns them."""
    assigned: list[int] = []
    for node in nodes:
        layer = node_layer_index(node)
        if layer is None:
            layer = len(assigned)
            while layer in assigned:
                layer += 1
        assigned.append(layer)
    return assigned


def validate_popularity_file(path: Path) -> None:
    """The header and size checks the planner applies to an explicitly requested popularity file."""
    invalid = ContractError(f"MoE Offload failed to load or validate explicitly requested popularity_file: '{path}'.")
    if not path.is_file():
        raise invalid
    data = path.read_bytes()
    if len(data) < 24:
        raise invalid
    magic, version, layers, experts, _slots, ranked = struct.unpack("<6I", data[:24])
    if magic != STRP_MAGIC or version != 1 or layers == 0 or experts == 0 or ranked > layers * experts:
        raise invalid
    if len(data) < 24 + 4 * ranked:
        raise invalid


def simulate_session_creation(
    model: onnx.ModelProto, entries: Mapping[str, str], model_dir: Path | None = None
) -> list[str]:
    """The names of the QMoE nodes the runtime offloads, or ContractError where its session creation fails."""
    qmoe_nodes = []
    for node in model.graph.node:
        if node.op_type == "QMoEOffload":
            raise ContractError(
                f"Type Error: Type of operator ({node.op_type}) is invalid: no such operator is registered"
            )
        if node.domain == "com.microsoft" and node.op_type == "QMoE":
            unknown = [attribute.name for attribute in node.attribute if attribute.name not in QMOE_ATTRIBUTES]
            if unknown:
                raise ContractError(f"Unrecognized attribute: {unknown[0]} for operator QMoE")
            qmoe_nodes.append(node)

    policy = parse_session_entries(entries, model_dir)
    eligible = [node_is_eligible(model, node) for node in qmoe_nodes]
    registered = []
    if policy["mode"] != "disabled":
        for node, ok in zip(qmoe_nodes, eligible, strict=True):
            if ok:
                registered.append(node)
            elif policy["mode"] == "required":
                raise ContractError(f"QMoE offload mode='required' rejected node '{node.name}'")

    layers = layer_indices(registered)
    for index, layer in enumerate(layers):
        if layer in layers[:index]:
            raise ContractError(f"MoE Offload session admission failed: duplicate layer_index={layer}")
    for override in policy["node_overrides"]:
        matches = [
            override["name"] in (node.name, str(layer), f"layer.{layer}")
            for node, layer in zip(registered, layers, strict=True)
        ]
        if not any(matches):
            raise ContractError(f"MoE Offload node_overrides references unknown node '{override['name']}'.")
    if registered and entries.get(OFFLOAD_PREFIX + "popularity_file"):
        validate_popularity_file(policy["popularity_file"])
    return [node.name for node in registered]
