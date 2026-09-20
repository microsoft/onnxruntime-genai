# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Normalize builder inputs without constructing an ONNX graph.

This is a partial implementation of docs/ModelBuilderConfiguration.md. The
result bridges structured policy to existing exporters; it is not proof that
the resulting graph, shared tensors, or runtime profile meet the full contract.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any

from quantization import QuantConfig

STRUCTURED_FIELDS = (
    "target_options",
    "drafter_options",
    "speculative_options",
    "runtime_config",
)


@dataclass
class EffectiveBuilderConfig:
    """Carry normalized policy and the options consumed by existing exporters.

    ``extra_options`` contains live QuantConfig instances and is not a JSON
    schema. ``to_dict`` omits those internal options and is a policy summary,
    not a complete export manifest: later checkpoint and graph decisions are
    not reflected in it.
    """

    version: int
    execution_provider: str
    precision: str
    extra_options: dict[str, Any]
    target_options: dict[str, Any]
    drafter_options: dict[str, Any] | None
    speculative_options: dict[str, Any]
    runtime_config: dict[str, Any]
    target_moe_explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "builder_config_version": self.version,
            "execution_provider": self.execution_provider,
            "precision": self.precision,
            "target_options": copy.deepcopy(self.target_options),
            "drafter_options": copy.deepcopy(self.drafter_options),
            "speculative_options": copy.deepcopy(self.speculative_options),
            "runtime_config": copy.deepcopy(self.runtime_config),
        }


def reject_duplicate_key(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def reject_null(value: Any, path: str):
    if value is None:
        raise ValueError(f"{path} must not contain null")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_null(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_null(child, f"{path}[{index}]")


def load_json_object(value: Any, field_name: str) -> dict[str, Any]:
    """Load a detached object; file paths are relative to the process directory.

    Resource staging belongs to the caller. This function does not resolve
    nested checkpoint/calibration paths relative to the containing JSON file.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        result = copy.deepcopy(value)
    elif isinstance(value, (str, os.PathLike)):
        text_or_path = os.fspath(value).strip()
        if text_or_path.startswith("{"):
            result = json.loads(text_or_path, object_pairs_hook=reject_duplicate_key)
        else:
            with open(text_or_path, encoding="utf-8") as handle:
                result = json.load(handle, object_pairs_hook=reject_duplicate_key)
    else:
        raise ValueError(f"{field_name} must be an object, inline JSON object, or JSON file path")
    if not isinstance(result, dict):
        raise ValueError(f"{field_name} must resolve to a JSON object")
    reject_null(result, field_name)
    return result


def merge_objects(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Copy and recursively merge objects, replacing arrays and scalars whole."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_objects(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def check_fields(data: dict[str, Any], allowed: set[str], path: str):
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be an object")
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {path} field(s): {sorted(unknown)}")


def normalize_provider(execution_provider: str) -> str:
    return "trt-rtx" if execution_provider == "NvTensorRtRtx" else execution_provider


def precision_from_quant_data(quant_data: dict[str, Any], precision: str | None) -> str:
    weights_type = quant_data.get("weights", {}).get("type")
    if weights_type in ("int4", "uint4"):
        return "int4"
    if weights_type in ("int8", "uint8"):
        return "int8"
    if weights_type == "none":
        return quant_data.get("io_dtype", precision or "fp16")
    if precision is not None:
        return precision
    raise ValueError("precision is required unless target_options.quant_config.weights.type is explicit")


def canonical_quant_data(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    format_data = result.get("format")
    runtime_data = result.get("runtime")
    if format_data is not None and runtime_data is not None and format_data != runtime_data:
        raise ValueError("quantization format and compatibility alias runtime conflict")
    if runtime_data is not None:
        result["format"] = runtime_data
        del result["runtime"]
    return result


def warn_structured_override(legacy_options: dict[str, Any], legacy_key: str, structured_path: str):
    if legacy_key in legacy_options:
        warnings.warn(
            f"{structured_path} overrides legacy extra_options.{legacy_key}",
            UserWarning,
            stacklevel=3,
        )


def normalize_target_quant_config(
    data: dict[str, Any],
    legacy_options: dict[str, Any],
    precision: str | None,
    execution_provider: str,
) -> tuple[QuantConfig, str]:
    """Seed target policy from legacy options, then overlay structured leaves."""
    canonical = canonical_quant_data(data)
    seed_precision = precision_from_quant_data(canonical, precision)
    normalized_legacy = copy.deepcopy(legacy_options)
    op_types = normalized_legacy.get("op_types_to_quantize")
    if isinstance(op_types, str):
        normalized_legacy["op_types_to_quantize"] = tuple(op_types.split("/"))
    exclusions = normalized_legacy.get("nodes_to_exclude")
    if isinstance(exclusions, str):
        normalized_legacy["nodes_to_exclude"] = exclusions.split(",")
    legacy_config = QuantConfig.from_extra_options(normalized_legacy, seed_precision, execution_provider)
    merged = merge_objects(legacy_config.to_dict(), canonical)

    if "moe" not in canonical or "type" not in canonical.get("moe", {}):
        weights_type = merged["weights"]["type"]
        merged["moe"]["type"] = {
            "int4": "int4",
            "uint4": "int4",
            "int8": "int8",
            "uint8": "int8",
            "none": "none",
        }.get(weights_type, merged["moe"]["type"])

    weights_type = merged["weights"]["type"]
    if merged["moe"]["type"] in ("mxfp4", "nvfp4") and execution_provider != "cuda":
        raise ValueError(f"moe.type={merged['moe']['type']} is supported only on CUDA")
    if merged["format"]["matmulnbits_weights_prepacked"] and execution_provider != "cuda":
        raise ValueError("format.matmulnbits_weights_prepacked is supported only on CUDA")
    if execution_provider == "trt-rtx" and weights_type in ("int4", "uint4", "int8", "uint8"):
        if "format" in canonical and canonical["format"].get("use_qdq") is False:
            raise ValueError("TRT-RTX integer dense weights require quant_config.format.use_qdq=true")
        merged["format"]["use_qdq"] = True

    aliases = {
        "block_size": "weights.block_size",
        "is_symmetric": "weights.symmetric",
        "accuracy_level": "weights.accuracy_level",
        "op_types_to_quantize": "weights.op_types",
        "algo_config": "weights.method",
        "moe_quant_type": "moe.type",
        "qmoe_block_size": "moe.block_size",
        "qmoe_weights_prepacked": "moe.weights_prepacked",
        "use_qdq": "format.use_qdq",
        "matmulnbits_weights_prepacked": "format.matmulnbits_weights_prepacked",
    }
    for legacy_key, path in aliases.items():
        section, leaf = path.split(".")
        if section in canonical and leaf in canonical[section]:
            warn_structured_override(legacy_options, legacy_key, f"target_options.quant_config.{path}")

    quant_config = QuantConfig.from_dict(merged)
    return quant_config, precision_from_quant_data(quant_config.to_dict(), precision)


def flatten_target_options(
    options: dict[str, Any],
    legacy_options: dict[str, Any],
    precision: str | None,
    execution_provider: str,
) -> tuple[dict[str, Any], QuantConfig, str]:
    check_fields(options, {"quant_config", "attention", "optimizations", "vision"}, "target_options")
    if "vision" in options:
        raise ValueError("target_options.vision is reserved for a future schema capability")

    quant_config, effective_precision = normalize_target_quant_config(
        options.get("quant_config", {}), legacy_options, precision, execution_provider
    )
    flattened = copy.deepcopy(legacy_options)
    flattened["_quant_config"] = quant_config
    flattened["_target_io_dtype"] = quant_config.io_dtype
    flattened["is_symmetric"] = quant_config.weights.symmetric
    flattened["op_types_to_quantize"] = quant_config.weights.op_types
    flattened["use_qdq"] = quant_config.format.use_qdq
    flattened["matmulnbits_weights_prepacked"] = quant_config.format.matmulnbits_weights_prepacked

    attention = options.get("attention", {})
    check_fields(attention, {"implementation", "paged", "kv_cache"}, "target_options.attention")
    implementation = attention.get("implementation")
    if implementation is not None:
        if implementation not in ("auto", "paged"):
            raise ValueError("target_options.attention.implementation must be 'auto' or 'paged'")
        warn_structured_override(legacy_options, "use_paged_attention", "target_options.attention.implementation")
        flattened["use_paged_attention"] = implementation == "paged"
    paged = attention.get("paged", {})
    check_fields(paged, {"block_size"}, "target_options.attention.paged")
    if paged and implementation != "paged":
        raise ValueError("target_options.attention.paged is valid only when implementation='paged'")
    if "block_size" in paged:
        warn_structured_override(legacy_options, "paged_block_size", "target_options.attention.paged.block_size")
        flattened["paged_block_size"] = paged["block_size"]

    kv_cache = attention.get("kv_cache", {})
    check_fields(kv_cache, {"scheme", "scale_file", "windowed"}, "target_options.attention.kv_cache")
    for field_name, legacy_key in (
        ("scheme", "kv_cache_quant_scheme"),
        ("scale_file", "kv_cache_scale_file"),
        ("windowed", "windowed_kv_cache"),
    ):
        if field_name in kv_cache:
            warn_structured_override(legacy_options, legacy_key, f"target_options.attention.kv_cache.{field_name}")
            flattened[legacy_key] = kv_cache[field_name]

    optimizations = options.get("optimizations", {})
    check_fields(optimizations, {"fuse_mlp_gate_up"}, "target_options.optimizations")
    if "fuse_mlp_gate_up" in optimizations:
        warn_structured_override(legacy_options, "fuse_mlp_gate_up", "target_options.optimizations.fuse_mlp_gate_up")
        flattened["fuse_mlp_gate_up"] = optimizations["fuse_mlp_gate_up"]

    return flattened, quant_config, effective_precision


def normalize_drafter_quant_config(data: dict[str, Any], drafter_type: str, execution_provider: str) -> QuantConfig:
    canonical = canonical_quant_data(data)
    # Drafter body defaults must not copy the target's overrides, KV policy, or
    # quantization layout. Borrowed embedding/head tensors need separate checks.
    defaults = {
        "io_dtype": "bf16" if drafter_type in ("dflash2", "dspark") else "fp16",
        "checkpoint_policy": "preserve",
        "weights": {"type": "none", "block_size": 32},
        "moe": {"type": "none", "block_size": 32, "weights_prepacked": 0},
        "format": {"use_qdq": False, "matmulnbits_weights_prepacked": 0},
    }
    quant_config = QuantConfig.from_dict(merge_objects(defaults, canonical))
    if drafter_type in ("dflash2", "dspark") and quant_config.io_dtype != "bf16":
        raise ValueError(
            f"{drafter_type} body io_dtype must be bf16 because its activations can exceed the fp16 range"
        )
    if drafter_type == "dspark" and quant_config.weights.type != "none":
        raise ValueError("DSpark integer weight quantization is not supported")
    if drafter_type == "dflash2":
        weights = canonical.get("weights", {})
        for field_name in ("accuracy_level", "op_types", "overrides"):
            if field_name in weights:
                raise ValueError(f"DFlash2 weights.{field_name} is not supported")
        if quant_config.weights.type not in ("none", "int4", "int8"):
            raise ValueError("DFlash2 weights.type must be none, int4, or int8")
        if quant_config.weights.method != "default" or not quant_config.weights.symmetric:
            raise ValueError("DFlash2 supports only symmetric DEFAULT integer weight quantization")
        if quant_config.format.use_qdq or quant_config.format.matmulnbits_weights_prepacked != 0:
            raise ValueError("DFlash2 body weights require raw QOperator format")
    if quant_config.moe.type != "none":
        raise ValueError(f"{drafter_type} does not support MoE expert quantization")
    if execution_provider != "cuda" and quant_config.format.matmulnbits_weights_prepacked:
        raise ValueError("prepacked MatMulNBits weights are supported only on CUDA")
    return quant_config


def flatten_drafter_options(
    options: dict[str, Any] | None,
    flattened: dict[str, Any],
    execution_provider: str,
) -> dict[str, Any] | None:
    if options is None:
        return None
    check_fields(
        options,
        {"drafter_type", "path", "num_draft_tokens", "shared_weights", "quant_config", "attention", "optimizations", "dspark"},
        "drafter_options",
    )
    drafter_type = options.get("drafter_type")
    if drafter_type not in ("none", "mtp", "dflash2", "dspark"):
        raise ValueError("drafter_options.drafter_type must be mtp, dflash2, dspark, or none")

    legacy_drafters = {
        name.removesuffix("_path")
        for name in ("dflash2_path", "dspark_path")
        if flattened.get(name)
    }
    if legacy_drafters and legacy_drafters != {drafter_type}:
        raise ValueError(
            f"drafter_options.drafter_type={drafter_type} conflicts with legacy drafter selection "
            + ", ".join(sorted(legacy_drafters))
        )

    incompatible = {
        "none": set(options) - {"drafter_type"},
        "mtp": set(options) & {"path", "num_draft_tokens", "attention", "dspark"},
        "dflash2": set(options) & {"dspark"},
        "dspark": set(),
    }[drafter_type]
    if incompatible:
        raise ValueError(f"drafter_options fields {sorted(incompatible)} are not valid for drafter_type={drafter_type}")

    if drafter_type == "none":
        flattened["exclude_mtp"] = True
        return {"drafter_type": "none"}

    if drafter_type in ("dflash2", "dspark") and not options.get("path"):
        raise ValueError(f"drafter_options.path is required for drafter_type={drafter_type}")
    if drafter_type == "mtp" and "path" in options:
        raise ValueError("a separate MTP path is not supported")
    if drafter_type in ("dflash2", "dspark"):
        if not flattened.get("use_paged_attention", False):
            raise ValueError(f"drafter_type={drafter_type} requires target paged attention")
        checkpoint_path = os.fspath(options["path"])
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                f"drafter_options.path must be a local checkpoint directory containing config.json: {checkpoint_path}"
            )
        if "aux_hidden_state_layers" not in flattened:
            # SpecForge names layer outputs, while the target exporter names
            # incoming residuals. Resolve this +1 translation before building it.
            with open(config_path, encoding="utf-8") as handle:
                checkpoint_config = json.load(handle, object_pairs_hook=reject_duplicate_key)
            dflash_config = checkpoint_config.get("dflash_config", {})
            target_layer_ids = dflash_config.get("target_layer_ids")
            if not isinstance(target_layer_ids, list) or not target_layer_ids:
                raise ValueError(f"the {drafter_type} checkpoint must define target_layer_ids")
            flattened["aux_hidden_state_layers"] = ",".join(str(int(layer_id) + 1) for layer_id in target_layer_ids)

    quant_config = normalize_drafter_quant_config(options.get("quant_config", {}), drafter_type, execution_provider)
    if drafter_type == "mtp":
        flattened["mtp_quant_config"] = quant_config
    else:
        flattened[f"{drafter_type}_path"] = options["path"]
        if "num_draft_tokens" in options:
            flattened[f"{drafter_type}_num_draft_tokens"] = options["num_draft_tokens"]
        flattened["_drafter_quant_config"] = quant_config
        flattened[f"{drafter_type}_precision"] = (
            quant_config.weights.type if quant_config.weights.type != "none" else "bf16"
        )

    attention = options.get("attention", {})
    check_fields(attention, {"implementation", "paged", "kv_cache"}, "drafter_options.attention")
    if attention:
        if drafter_type not in ("dflash2", "dspark"):
            raise ValueError(f"attention settings are not supported for drafter_type={drafter_type}")
        if attention.get("implementation", "paged") != "paged":
            raise ValueError(f"{drafter_type} requires paged attention")
        paged = attention.get("paged", {})
        check_fields(paged, {"block_size"}, "drafter_options.attention.paged")
        if "block_size" in paged and paged["block_size"] != flattened.get("paged_block_size", 256):
            raise ValueError("target and block drafter paged block sizes must match")
        kv_cache = attention.get("kv_cache", {})
        check_fields(kv_cache, {"scheme", "scale_file", "windowed"}, "drafter_options.attention.kv_cache")
        if kv_cache.get("scheme", "none") != "none" or "scale_file" in kv_cache:
            raise ValueError(f"{drafter_type} KV cache quantization is not supported")
        if kv_cache.get("windowed", False):
            raise ValueError(f"{drafter_type} windowed KV cache is not supported")

    optimizations = options.get("optimizations", {})
    check_fields(optimizations, {"fuse_mlp_gate_up"}, "drafter_options.optimizations")
    fuse_gate_up = optimizations.get("fuse_mlp_gate_up", False)
    if fuse_gate_up and drafter_type != "dflash2":
        raise ValueError(f"fuse_mlp_gate_up is not supported for drafter_type={drafter_type}")
    if drafter_type == "dflash2":
        flattened["dflash2_fuse_gate_up"] = fuse_gate_up

    shared_weights = options.get("shared_weights", {})
    check_fields(shared_weights, {"embedding", "lm_head"}, "drafter_options.shared_weights")
    policies = {name: shared_weights.get(name, "auto") for name in ("embedding", "lm_head")}
    if any(policy not in ("auto", "required", "off") for policy in policies.values()):
        raise ValueError("shared weight policies must be auto, required, or off")
    if drafter_type != "dflash2" and any(policy != "auto" for policy in policies.values()):
        raise ValueError(
            f"explicit shared weight policies are not yet supported for drafter_type={drafter_type}"
        )
    flattened["_shared_weight_policies"] = policies

    dspark = options.get("dspark", {})
    check_fields(dspark, {"top_k"}, "drafter_options.dspark")
    if dspark and drafter_type != "dspark":
        raise ValueError("drafter_options.dspark is valid only for drafter_type=dspark")
    if "top_k" in dspark:
        flattened["dspark_top_k"] = dspark["top_k"]

    effective = copy.deepcopy(options)
    effective["quant_config"] = quant_config.to_dict()
    effective["shared_weights"] = policies
    return effective


def flatten_speculative_options(options: dict[str, Any], flattened: dict[str, Any]):
    check_fields(options, {"aux_hidden_state_layers", "state_update_capacity"}, "speculative_options")
    if "aux_hidden_state_layers" in options:
        layers = options["aux_hidden_state_layers"]
        if not isinstance(layers, list) or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
            raise ValueError("speculative_options.aux_hidden_state_layers must be a list of integers")
        flattened["aux_hidden_state_layers"] = ",".join(str(layer) for layer in layers)
    if "state_update_capacity" in options:
        flattened["state_update_capacity"] = options["state_update_capacity"]


def normalize_builder_config(
    precision: str | None,
    execution_provider: str,
    extra_options: dict[str, Any] | None = None,
    builder_config_version: int | str | None = None,
    target_options: Any = None,
    drafter_options: Any = None,
    speculative_options: Any = None,
    runtime_config: Any = None,
    search: Any = None,
) -> EffectiveBuilderConfig:
    """Select the legacy adapter or the structured configuration path.

    Omission is significant: absent drafter_options retains automatic discovery,
    whereas an explicit group requests a separate policy. No weights are loaded
    here, although block-drafter config.json is read to resolve auxiliary taps.
    """
    legacy_options = copy.deepcopy(extra_options or {})
    structured_present = any(value is not None for value in (target_options, drafter_options, speculative_options, runtime_config))
    explicit_version = builder_config_version is not None
    version = int(builder_config_version) if explicit_version else 2 if structured_present else 1
    if version == 1 and structured_present:
        raise ValueError("builder_config_version=1 cannot be combined with structured configuration fields")
    if version not in (1, 2):
        raise ValueError(f"unsupported builder_config_version={version}; supported versions are 1 and 2")
    if precision is None and version == 1:
        raise ValueError("precision is required for legacy model-builder configuration")

    provider = normalize_provider(execution_provider)
    if version == 1:
        runtime = load_json_object(runtime_config, "runtime_config") if runtime_config is not None else {}
        if search is not None:
            runtime = merge_objects({"search": load_json_object(search, "search")}, runtime)
        return EffectiveBuilderConfig(
            version=1,
            execution_provider=provider,
            precision=precision,
            extra_options=legacy_options,
            target_options={},
            drafter_options=None,
            speculative_options={},
            runtime_config=runtime,
        )

    target = load_json_object(target_options, "target_options")
    drafter = None if drafter_options is None else load_json_object(drafter_options, "drafter_options")
    speculative = load_json_object(speculative_options, "speculative_options")
    runtime = load_json_object(runtime_config, "runtime_config")
    if search is not None:
        legacy_search = load_json_object(search, "search")
        runtime = merge_objects({"search": legacy_search}, runtime)

    flattened, quant_config, effective_precision = flatten_target_options(
        target, legacy_options, precision, provider
    )
    flatten_speculative_options(speculative, flattened)
    effective_drafter = flatten_drafter_options(drafter, flattened, provider)
    flattened["_runtime_config"] = runtime

    effective_target = copy.deepcopy(target)
    effective_target["quant_config"] = quant_config.to_dict()
    target_quant_data = target.get("quant_config", {})
    target_moe_explicit = "moe" in target_quant_data and "type" in target_quant_data["moe"]
    return EffectiveBuilderConfig(
        version=2,
        execution_provider=provider,
        precision=effective_precision,
        extra_options=flattened,
        target_options=effective_target,
        drafter_options=effective_drafter,
        speculative_options=speculative,
        runtime_config=runtime,
        target_moe_explicit=target_moe_explicit,
    )


def validate_model_dependent_config(effective_config: EffectiveBuilderConfig, model_config: Any):
    if effective_config.version != 2 or effective_config.target_moe_explicit:
        return
    quant_config = effective_config.extra_options.get("_quant_config")
    if quant_config is None or quant_config.weights.type != "none":
        return
    text_config = getattr(model_config, "text_config", model_config)
    num_experts = getattr(text_config, "num_local_experts", getattr(text_config, "num_experts", 0))
    if num_experts:
        raise ValueError(
            "target_options.quant_config.moe.type is required for an MoE checkpoint when weights.type=none"
        )


def validate_runtime_config(runtime_config: dict[str, Any], generated_config: dict[str, Any]):
    """Check a runtime overlay against the completed exported configuration."""
    check_fields(runtime_config, {"search", "speculative", "engine", "model"}, "runtime_config")
    if "search" in runtime_config and not isinstance(runtime_config["search"], dict):
        raise ValueError("runtime_config.search must be an object")

    speculative = runtime_config.get("speculative", {})
    if not isinstance(speculative, dict):
        raise ValueError("runtime_config.speculative must be an object")
    check_fields(speculative, {"max_draft_tokens"}, "runtime_config.speculative")
    if "speculative" in runtime_config and "speculative" not in generated_config:
        raise ValueError("runtime_config references absent speculative configuration")
    if "max_draft_tokens" in speculative:
        max_draft_tokens = speculative["max_draft_tokens"]
        if isinstance(max_draft_tokens, bool) or not isinstance(max_draft_tokens, int) or not 1 <= max_draft_tokens <= 16:
            raise ValueError("runtime_config.speculative.max_draft_tokens must be an integer between 1 and 16")
        capacities = [
            component["num_draft_tokens"]
            for component in generated_config.get("model", {}).values()
            if isinstance(component, dict) and isinstance(component.get("num_draft_tokens"), int)
        ]
        decoder_capacity = generated_config.get("model", {}).get("decoder", {}).get("state_update_capacity")
        if isinstance(decoder_capacity, int) and decoder_capacity > 0:
            capacities.append(decoder_capacity)
        if capacities and max_draft_tokens > min(capacities):
            raise ValueError(
                "runtime_config.speculative.max_draft_tokens exceeds the exported drafter/state capacity"
            )

    engine = runtime_config.get("engine", {})
    if not isinstance(engine, dict):
        raise ValueError("runtime_config.engine must be an object")
    check_fields(engine, {"dynamic_batching"}, "runtime_config.engine")
    dynamic_batching = engine.get("dynamic_batching", {})
    if not isinstance(dynamic_batching, dict):
        raise ValueError("runtime_config.engine.dynamic_batching must be an object")
    check_fields(
        dynamic_batching,
        {"max_batch_size", "max_scheduled_tokens", "num_blocks", "gpu_utilization_factor"},
        "runtime_config.engine.dynamic_batching",
    )
    if "engine" in runtime_config and "engine" not in generated_config:
        raise ValueError("runtime_config references absent engine configuration")
    if "num_blocks" in dynamic_batching and "gpu_utilization_factor" in dynamic_batching:
        raise ValueError("runtime_config cannot specify both num_blocks and gpu_utilization_factor")
    for field_name in ("max_batch_size", "max_scheduled_tokens", "num_blocks"):
        if field_name not in dynamic_batching:
            continue
        value = dynamic_batching[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"runtime_config.engine.dynamic_batching.{field_name} must be a positive integer")
    if dynamic_batching.get("max_batch_size", 1) > 256:
        raise ValueError("runtime_config.engine.dynamic_batching.max_batch_size must be at most 256")
    if "gpu_utilization_factor" in dynamic_batching:
        value = dynamic_batching["gpu_utilization_factor"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(
                "runtime_config.engine.dynamic_batching.gpu_utilization_factor must be greater than 0 and at most 1"
            )

    model = runtime_config.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("runtime_config.model must be an object")
    generated_components = generated_config.get("model", {})
    for component_name, component_options in model.items():
        if component_name not in generated_components:
            raise ValueError(f"runtime_config references absent model component '{component_name}'")
        if not isinstance(component_options, dict):
            raise ValueError(f"runtime_config.model.{component_name} must be an object")
        check_fields(
            component_options,
            {"session_options", "run_options"},
            f"runtime_config.model.{component_name}",
        )
        generated_component = generated_components[component_name]
        if not isinstance(generated_component, dict) or "session_options" not in generated_component:
            raise ValueError(f"runtime_config.model.{component_name} is not a session-bearing component")
        generated_session = generated_component["session_options"]
        runtime_session = component_options.get("session_options", {})
        if not isinstance(runtime_session, dict):
            raise ValueError(f"runtime_config.model.{component_name}.session_options must be an object")
        for key, value in runtime_session.items():
            if key in generated_session and key not in ("log_id", "provider_options") and value != generated_session[key]:
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options cannot overwrite required session option '{key}'"
                )
        if "provider_options" in runtime_session:
            generated_providers = generated_session.get("provider_options", [])
            runtime_providers = runtime_session["provider_options"]
            if not isinstance(runtime_providers, list) or any(
                not isinstance(entry, dict) for entry in runtime_providers
            ):
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options.provider_options must be an array of objects"
                )
            generated_names = {name for entry in generated_providers for name in entry}
            runtime_names = {name for entry in runtime_providers for name in entry}
            if generated_names != runtime_names:
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options cannot change execution providers"
                )


def apply_runtime_config(generated_config: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Apply a validated profile after all component sections exist."""
    if not runtime_config:
        return generated_config
    validate_runtime_config(runtime_config, generated_config)
    baseline = copy.deepcopy(generated_config)
    overlay = copy.deepcopy(runtime_config)
    dynamic_batching = overlay.get("engine", {}).get("dynamic_batching", {})
    generated_dynamic_batching = baseline.get("engine", {}).get("dynamic_batching", {})
    if "num_blocks" in dynamic_batching:
        generated_dynamic_batching.pop("gpu_utilization_factor", None)
    elif "gpu_utilization_factor" in dynamic_batching:
        generated_dynamic_batching.pop("num_blocks", None)
    return merge_objects(baseline, overlay)
