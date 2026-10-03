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
import math
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from quantization import QuantConfig

STRUCTURED_FIELDS = (
    "target_options",
    "drafter_options",
    "speculative_options",
    "runtime_config",
    "component_options",
)

GRAPH_DERIVED_PROVIDER_OPTIONS = {
    "webgpu": {"enableGraphCapture", "multiRotaryCacheConcatOffset"},
    "nvtensorrtrtx": {"multi_rotary_cache_concat_offset"},
}

RUNTIME_TUNABLE_PROVIDER_OPTIONS = {
    "cuda": {
        "arena_extend_strategy",
        "cudnn_conv1d_pad_to_nc1d",
        "cudnn_conv_algo_search",
        "cudnn_conv_use_max_workspace",
        "device_id",
        "do_copy_in_default_stream",
        "enable_cuda_graph",
        "enable_skip_layer_norm_strict_mode",
        "gpu_mem_limit",
        "prefer_nhwc",
        "tunable_op_enable",
        "tunable_op_max_tuning_duration_ms",
        "tunable_op_tuning_enable",
        "use_ep_level_unified_stream",
        "use_tf32",
    },
    "webgpu": {"validationMode"},
    "nvtensorrtrtx": {"enable_cuda_graph"},
}

# Value types accepted by SessionOptions_Element in src/config.cpp. Names outside these
# groups become AddConfigEntry entries, which the parser reads as strings.
SESSION_OPTION_INTEGER_FIELDS = (
    "intra_op_num_threads",
    "inter_op_num_threads",
    "log_severity_level",
    "log_verbosity_level",
)
SESSION_OPTION_BOOLEAN_FIELDS = ("enable_cpu_mem_arena", "enable_mem_pattern")
GRAPH_OPTIMIZATION_LEVELS = (
    "ORT_DISABLE_ALL",
    "ORT_ENABLE_BASIC",
    "ORT_ENABLE_EXTENDED",
    "ORT_ENABLE_ALL",
)


@dataclass(frozen=True)
class ComponentBinding:
    """Map a manifest-facing input or output name to an ONNX graph value."""

    name: str
    graph_name: str


@dataclass(frozen=True)
class HeadComponent:
    """Describe one validated, pre-built head graph staged beside a backbone."""

    name: str
    source: str
    filename: str
    inputs: tuple[ComponentBinding, ...]
    outputs: tuple[ComponentBinding, ...]

@dataclass(frozen=True)
class ComponentOptions:
    """Describe either generic ONNX heads or one pinned model-specific artifact.

    Generic packages populate ``heads``. Model-specific packages instead carry
    ``model_source`` and immutable artifact/base revisions for later dispatch.
    """

    backbone_filename: str
    heads: tuple[HeadComponent, ...]
    model_source: str | None = None
    artifact_revision: str | None = None
    base_revision: str | None = None

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
    component_options: ComponentOptions | None = None
    target_moe_explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = {
            "builder_config_version": self.version,
            "execution_provider": self.execution_provider,
            "precision": self.precision,
            "target_options": copy.deepcopy(self.target_options),
            "drafter_options": copy.deepcopy(self.drafter_options),
            "speculative_options": copy.deepcopy(self.speculative_options),
            "runtime_config": copy.deepcopy(self.runtime_config),
        }
        if self.component_options is not None:
            result["component_options"] = component_options_dict(
                self.component_options
            )
        return result


def head_component_dict(head: HeadComponent) -> dict[str, Any]:
    """Serialize one immutable head declaration."""
    return {
        "name": head.name,
        "source": head.source,
        "filename": head.filename,
        "inputs": {binding.name: binding.graph_name for binding in head.inputs},
        "outputs": {binding.name: binding.graph_name for binding in head.outputs},
    }


def component_options_dict(options: ComponentOptions) -> dict[str, Any]:
    """Serialize normalized component options."""
    result = {
        "backbone": {"filename": options.backbone_filename},
        "heads": [head_component_dict(head) for head in options.heads],
    }
    if options.model_source is not None:
        result["model_source"] = options.model_source
        result["artifact_revision"] = options.artifact_revision
        result["base_revision"] = options.base_revision
    return result


def component_manifest_dict(options: ComponentOptions) -> dict[str, Any]:
    """Build the runtime-neutral manifest for generic pre-built heads."""
    components = {
        "backbone": {
            "role": "backbone",
            "filename": options.backbone_filename,
            "outputs": {"hidden_states": "hidden_states"},
        }
    }
    components.update(
        {
            head.name: {
                "role": "head",
                "filename": head.filename,
                "inputs": {
                    binding.name: binding.graph_name for binding in head.inputs
                },
                "outputs": {
                    binding.name: binding.graph_name for binding in head.outputs
                },
            }
            for head in options.heads
        }
    )
    return {
        "schema_version": 1,
        "model_type": "generic-non-generative",
        "components": components,
    }


def normalize_component_options(value: Any) -> ComponentOptions | None:
    """Normalize generic or model-specific component configuration.

    ``value`` may be a JSON object, inline JSON, or JSON-file path. Generic
    declarations require at least one head; model-specific declarations require
    a source and both revision pins. Invalid names, filenames, duplicate
    bindings, mixed declaration forms, and missing pins raise ``ValueError``.
    """
    if value is None:
        return None
    data = load_json_object(value, "component_options")
    check_fields(
        data,
        {"backbone", "heads", "model_source", "artifact_revision", "base_revision"},
        "component_options",
    )

    backbone = data.get("backbone", {})
    check_fields(backbone, {"filename"}, "component_options.backbone")
    backbone_filename = backbone.get("filename", "model.onnx")
    validate_component_filename(backbone_filename, "component_options.backbone.filename")

    model_source = data.get("model_source")
    if model_source is not None and (
        not isinstance(model_source, (str, os.PathLike)) or not os.fspath(model_source)
    ):
        raise ValueError("component_options.model_source must be a non-empty path")
    heads_data = data.get("heads", [])
    if model_source is not None:
        return normalize_model_specific_component_options(
            data, backbone_filename, model_source, heads_data
        )
    return normalize_generic_component_options(data, backbone_filename, heads_data)


def normalize_model_specific_component_options(
    data: dict[str, Any],
    backbone_filename: str,
    model_source: str | os.PathLike,
    heads_data: Any,
) -> ComponentOptions:
    """Normalize one pinned model-specific artifact declaration."""
    if not isinstance(heads_data, list):
        raise ValueError("component_options.heads must be an array")
    if heads_data:
        raise ValueError("component_options.model_source cannot be combined with pre-built heads")
    for revision_name in ("artifact_revision", "base_revision"):
        revision = data.get(revision_name)
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"component_options.{revision_name} is required for model-specific exports")
    return ComponentOptions(
        backbone_filename=backbone_filename,
        heads=(),
        model_source=os.fspath(model_source),
        artifact_revision=data["artifact_revision"],
        base_revision=data["base_revision"],
    )


def normalize_generic_component_options(
    data: dict[str, Any], backbone_filename: str, heads_data: Any
) -> ComponentOptions:
    """Normalize generic pre-built ONNX head declarations."""
    if not isinstance(heads_data, list) or not heads_data:
        raise ValueError("component_options.heads must be a non-empty array")
    for revision_name in ("artifact_revision", "base_revision"):
        if data.get(revision_name) is not None:
            raise ValueError(
                f"component_options.{revision_name} requires model_source"
            )

    heads = []
    names = {"backbone"}
    filenames = {backbone_filename.casefold()}
    for index, head_data in enumerate(heads_data):
        path = f"component_options.heads[{index}]"
        check_fields(head_data, {"name", "source", "filename", "inputs", "outputs"}, path)
        name = head_data.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name):
            raise ValueError(f"{path}.name must start with a letter and contain only letters, digits, '.', '_', or '-'")
        if name.casefold() in names:
            raise ValueError(f"duplicate component name {name!r}")
        source = head_data.get("source")
        if not isinstance(source, (str, os.PathLike)) or not os.fspath(source):
            raise ValueError(f"{path}.source must be a non-empty path")
        source = os.fspath(source)
        filename = head_data.get("filename", os.path.basename(source))
        validate_component_filename(filename, f"{path}.filename")
        if filename.casefold() in filenames:
            raise ValueError(f"duplicate component filename {filename!r}")
        inputs = normalize_component_bindings(
            head_data.get("inputs", {"hidden_states": "hidden_states"}), f"{path}.inputs"
        )
        outputs = normalize_component_bindings(head_data.get("outputs", {}), f"{path}.outputs")
        names.add(name.casefold())
        filenames.add(filename.casefold())
        heads.append(
            HeadComponent(name=name, source=source, filename=filename, inputs=inputs, outputs=outputs)
        )
    return ComponentOptions(
        backbone_filename=backbone_filename,
        heads=tuple(heads),
    )


def normalize_component_bindings(value: Any, path: str) -> tuple[ComponentBinding, ...]:
    """Validate one logical-to-graph binding map and return immutable bindings.

    ``path`` identifies the configuration field in errors. Empty names/values
    and duplicate concrete graph names raise ``ValueError``.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    bindings = []
    graph_names = set()
    for name, graph_name in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path} binding names must be non-empty strings")
        if not isinstance(graph_name, str) or not graph_name.strip():
            raise ValueError(f"{path}.{name} must be a non-empty graph value name")
        if graph_name in graph_names:
            raise ValueError(f"{path} graph value names must be unique; duplicate {graph_name!r}")
        graph_names.add(graph_name)
        bindings.append(ComponentBinding(name=name, graph_name=graph_name))
    return tuple(bindings)


def validate_component_filename(value: Any, path: str):
    """Require a simple relative ``.onnx`` filename safe for package staging."""
    if (
        not isinstance(value, str)
        or not value
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.onnx", value)
    ):
        raise ValueError(f"{path} must be a relative ONNX filename without directories")


@dataclass(frozen=True)
class ComponentPackagePlan:
    """A fully validated generic component package operation."""

    copies: tuple[tuple[Path, Path], ...]
    manifest: dict[str, Any]


def export_component_package(options: ComponentOptions, output_dir: str):
    """Export component heads and their manifest into ``output_dir``.

    Model-specific sources are delegated to their graph builders. Generic heads
    and referenced external tensor files are copied only after every source and
    destination has been validated; malformed ONNX, unsafe paths, missing data,
    or destination conflicts raise ``ValueError``.
    """
    output_root = Path(output_dir).resolve()
    validate_component_backbone_destination(
        options, output_root, require_exists=True
    )
    if options.model_source is not None:
        from builders.non_generative import export_model_specific_components  # noqa: PLC0415

        export_model_specific_components(options, output_root)
        return
    execute_component_package(plan_component_package(options, output_root), output_root)


def plan_component_package(
    options: ComponentOptions, output_root: Path
) -> ComponentPackagePlan:
    """Validate a generic package and return an immutable copy/write plan."""
    copy_plan: dict[Path, Path] = {}
    reserved = {
        output_root / options.backbone_filename,
        output_root / f"{options.backbone_filename}.data",
        output_root / "component_manifest.json",
        *(output_root / head.filename for head in options.heads),
    }
    reserved.update(path for path in output_root.rglob("*"))

    for head in options.heads:
        source = Path(head.source)
        if not source.is_file():
            raise ValueError(f"component {head.name!r} source graph does not exist: {head.source}")
        destination = output_root / head.filename
        validate_destination_path(output_root, destination, head.name)
        model = load_component_model(source, head.name)
        validate_component_graph_bindings(head, model)
        add_component_copy(copy_plan, destination, source.resolve(), head.name)
        # Inspect external references before copying anything so an unsafe or
        # incomplete head cannot leave a partially staged component package.
        external_files = find_external_data_files(model, source, head.name)
        validate_component_model(source, head.name)
        for relative_path, external_source in external_files:
            external_destination = output_root / relative_path
            validate_destination_path(
                output_root, external_destination, head.name
            )
            add_component_copy(
                copy_plan, external_destination, external_source, head.name, reserved=reserved
            )
    return ComponentPackagePlan(
        copies=tuple(copy_plan.items()), manifest=component_manifest_dict(options)
    )


def execute_component_package(
    plan: ComponentPackagePlan, output_root: Path
) -> None:
    """Execute a previously validated package plan."""
    for destination, source in plan.copies:
        if source == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    with open(output_root / "component_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(plan.manifest, handle, indent=4)


def add_component_copy(
    copy_plan: dict[Path, Path],
    destination: Path,
    source: Path,
    component_name: str,
    reserved: set[Path] | None = None,
):
    """Add one validated source/destination pair to an atomic staging plan.

    Conflicting sources or collisions with reserved package files raise
    ``ValueError`` before filesystem writes begin.
    """
    existing = copy_plan.get(destination)
    if existing is not None and existing != source:
        raise ValueError(
            f"component {component_name!r} file {destination.name!r} conflicts with another packaged file"
        )
    if reserved is not None and source != destination:
        conflict = next(
            (
                path
                for path in reserved
                if paths_collide(destination, path)
            ),
            None,
        )
        if conflict is not None:
            raise ValueError(
                f"component {component_name!r} external data conflicts with packaged file "
                f"{conflict.name!r}"
            )
    for planned_destination, planned_source in copy_plan.items():
        if paths_collide(destination, planned_destination) and (
            destination != planned_destination or source != planned_source
        ):
            raise ValueError(
                f"component {component_name!r} file {destination.name!r} "
                "conflicts with another packaged path"
            )
    copy_plan[destination] = source


def paths_collide(left: Path, right: Path) -> bool:
    """Return whether two package paths are equal or ancestor-related."""
    return left == right or left in right.parents or right in left.parents


def validate_destination_path(
    output_root: Path, destination: Path, component_name: str
) -> None:
    """Reject lexical escapes and destination symlinks before staging."""
    try:
        relative = destination.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            f"component {component_name!r} destination escapes the package"
        ) from error
    current = output_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"component {component_name!r} destination contains symlink "
                f"{current}"
            )
    if destination.exists() and destination.is_dir():
        raise ValueError(
            f"component {component_name!r} destination is an existing directory "
            f"{destination}"
        )


def validate_component_backbone_destination(
    options: ComponentOptions,
    output_root: str | Path,
    *,
    require_exists: bool,
) -> Path:
    """Validate the backbone path before writing or packaging it."""
    output_root = Path(output_root).resolve()
    destination = output_root / options.backbone_filename
    validate_destination_path(output_root, destination, "backbone")
    if require_exists and not destination.is_file():
        raise ValueError(
            f"component backbone does not exist: {destination}"
        )
    return destination


def load_component_model(
    model_path: Path, component_name: str
) -> onnx.ModelProto:
    """Load one component graph without materializing external tensor data."""
    try:
        return onnx.load(model_path, load_external_data=False)
    except Exception as error:
        raise ValueError(
            f"component {component_name!r} source is not a readable ONNX model: "
            f"{model_path}"
        ) from error


def validate_component_model(model_path: Path, component_name: str) -> None:
    """Run the ONNX structural checker after external paths are validated."""
    try:
        onnx.checker.check_model(model_path)
    except Exception as error:
        raise ValueError(
            f"component {component_name!r} source is not a valid ONNX model: "
            f"{model_path}"
        ) from error


def validate_component_graph_bindings(
    head: HeadComponent, model: onnx.ModelProto
) -> None:
    """Require configured graph bindings to name actual graph values."""
    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    for binding in head.inputs:
        if binding.graph_name not in graph_inputs:
            raise ValueError(
                f"component {head.name!r} input binding {binding.name!r} "
                f"references unknown graph input {binding.graph_name!r}"
            )
    for binding in head.outputs:
        if binding.graph_name not in graph_outputs:
            raise ValueError(
                f"component {head.name!r} output binding {binding.name!r} "
                f"references unknown graph output {binding.graph_name!r}"
            )


def find_external_data_files(
    model: onnx.ModelProto, model_path: Path, component_name: str
) -> list[tuple[str, Path]]:
    """Resolve a head graph's external tensor files without loading their data.

    Returned locations remain relative to the graph directory. Absolute,
    traversing, malformed, or missing locations raise ``ValueError`` to prevent
    external-data references from escaping either source or destination roots.
    """
    source_root = model_path.resolve().parent
    files = {}
    for tensor in iter_tensor_protos(model):
        entries = {entry.key: entry.value for entry in tensor.external_data}
        if tensor.data_location != onnx.TensorProto.EXTERNAL and not entries:
            continue
        location = entries.get("location")
        if not location:
            raise ValueError(f"component {component_name!r} has external tensor data without a location")
        relative = Path(location)
        if (
            relative.is_absolute()
            or "\\" in location
            or ":" in location
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValueError(f"component {component_name!r} has unsafe external data location {location!r}")
        source = (source_root / relative).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"component {component_name!r} has unsafe external data location {location!r}"
            ) from error
        if not source.is_file():
            raise ValueError(
                f"component {component_name!r} external data file does not exist: {location}"
            )
        files[location] = source
    return list(files.items())


def iter_tensor_protos(message):
    """Yield every nested TensorProto that may carry external-data metadata."""
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        children = value if field.is_repeated else (value,)
        for child in children:
            yield from iter_tensor_protos(child)


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


def require_integer(value: Any, path: str) -> int:
    """Reject JSON values that int() would silently truncate or reinterpret."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def validate_session_option(key: str, value: Any, path: str):
    """Reject values the C++ session-options parser would fail to read."""
    if key in SESSION_OPTION_BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            raise ValueError(f"{path}.{key} must be a boolean")
        return
    if key in SESSION_OPTION_INTEGER_FIELDS:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value != math.trunc(value)
            or not -2_147_483_648 <= value <= 2_147_483_647
        ):
            raise ValueError(f"{path}.{key} must be an integer within the int32 range")
        return
    if key == "graph_optimization_level":
        if value not in GRAPH_OPTIMIZATION_LEVELS:
            raise ValueError(f"{path}.{key} must be one of {list(GRAPH_OPTIMIZATION_LEVELS)}")
        return
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key} must be a string")


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


def quant_config_schema_dict(quant_config: QuantConfig) -> dict[str, Any]:
    result = quant_config.to_dict()
    result["format"] = result.pop("runtime")
    result["checkpoint_policy"] = quant_config.checkpoint_policy
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
    if "checkpoint_policy" in canonical:
        raise ValueError("target_options.quant_config.checkpoint_policy is not supported by target loaders")
    target_weights = canonical.get("weights", {})
    if (
        target_weights.get("type") == "int2"
        or canonical.get("moe", {}).get("type") == "int2"
        or any(override.get("type") == "int2" for override in target_weights.get("overrides", []))
    ):
        raise ValueError("target_options.quant_config does not support int2; use drafter_options.quant_config")
    seed_precision = precision_from_quant_data(canonical, precision)
    normalized_legacy = copy.deepcopy(legacy_options)
    op_types = normalized_legacy.get("op_types_to_quantize")
    if isinstance(op_types, str):
        normalized_legacy["op_types_to_quantize"] = tuple(op_types.split("/"))
    exclusions = normalized_legacy.get("nodes_to_exclude")
    if isinstance(exclusions, str):
        normalized_legacy["nodes_to_exclude"] = exclusions.split(",")
    legacy_config = QuantConfig.from_extra_options(normalized_legacy, seed_precision, execution_provider)
    merged = merge_objects(quant_config_schema_dict(legacy_config), canonical)
    structured_weights = canonical.get("weights", {})
    if "type" in structured_weights and "symmetric" not in structured_weights and "is_symmetric" not in legacy_options:
        merged["weights"].pop("symmetric", None)

    legacy_moe_explicit = "moe_quant_type" in legacy_options or "use_8bits_moe" in legacy_options
    if ("moe" not in canonical or "type" not in canonical.get("moe", {})) and not legacy_moe_explicit:
        weights_type = merged["weights"]["type"]
        merged["moe"]["type"] = {
            "int4": "int4",
            "uint4": "int4",
            "int8": "int8",
            "uint8": "int8",
            "none": "none",
        }.get(weights_type, merged["moe"]["type"])

    weights_type = merged["weights"]["type"]
    if weights_type == "none" and merged["weights"]["overrides"]:
        raise ValueError("target weight overrides are not supported when weights.type=none")
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
    if "overrides" not in structured_weights:
        quant_config.legacy_nodes_to_exclude = legacy_config.legacy_nodes_to_exclude
    return quant_config, precision_from_quant_data(quant_config_schema_dict(quant_config), precision)


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
    if "type" in options.get("quant_config", {}).get("moe", {}):
        if quant_config.moe.type == "none":
            flattened.pop("moe_quant_type", None)
            flattened.pop("use_8bits_moe", None)
        else:
            flattened["moe_quant_type"] = quant_config.moe.type

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
        flattened["paged_block_size"] = require_integer(
            paged["block_size"], "target_options.attention.paged.block_size"
        )

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
    check_fields(optimizations, {"fuse_mlp_gate_up", "fuse_qkv"}, "target_options.optimizations")
    if "fuse_mlp_gate_up" in optimizations:
        warn_structured_override(legacy_options, "fuse_mlp_gate_up", "target_options.optimizations.fuse_mlp_gate_up")
        flattened["fuse_mlp_gate_up"] = optimizations["fuse_mlp_gate_up"]
    if "fuse_qkv" in optimizations:
        warn_structured_override(legacy_options, "fuse_qkv", "target_options.optimizations.fuse_qkv")
        warn_structured_override(legacy_options, "disable_qkv_fusion", "target_options.optimizations.fuse_qkv")
        flattened["fuse_qkv"] = optimizations["fuse_qkv"]

    return flattened, quant_config, effective_precision


def normalize_drafter_quant_config(
    data: dict[str, Any],
    drafter_type: str,
    execution_provider: str,
    target_io_dtype: str,
) -> QuantConfig:
    canonical = canonical_quant_data(data)
    # Drafter body defaults must not copy the target's overrides, KV policy, or
    # quantization layout. Borrowed embedding/head tensors need separate checks.
    defaults = {
        "io_dtype": "bf16" if drafter_type in ("dflash2", "dspark") else target_io_dtype,
        "checkpoint_policy": "preserve",
        "weights": {"type": "none", "block_size": 32},
        "moe": {"type": "none", "block_size": 32, "weights_prepacked": 0},
        "format": {
            "use_qdq": False,
            "matmulnbits_weights_prepacked": 0,
        },
    }
    quant_config = QuantConfig.from_dict(merge_objects(defaults, canonical))
    if drafter_type == "dspark" and quant_config.io_dtype != "bf16":
        raise ValueError("dspark body io_dtype must be bf16 because its activations can exceed the fp16 range")
    if drafter_type == "dflash2" and quant_config.io_dtype not in ("fp16", "bf16"):
        raise ValueError("DFlash2 body io_dtype must be fp16 or bf16")
    if drafter_type == "mtp" and quant_config.io_dtype != target_io_dtype:
        # The MTP graph consumes the decoder hidden state directly; no exporter converts it.
        raise ValueError(f"MTP io_dtype must match the target io_dtype '{target_io_dtype}'")
    if drafter_type == "mtp" and (
        quant_config.weights.type == "int2"
        or quant_config.moe.type == "int2"
        or any(override.type == "int2" for override in quant_config.weights.overrides)
    ):
        raise ValueError("MTP quant_config does not support int2; use DFlash2")
    if drafter_type == "dspark" and quant_config.weights.type != "none":
        raise ValueError("DSpark integer weight quantization is not supported")
    if drafter_type == "dspark":
        supported = QuantConfig.from_dict(defaults)
        if (
            quant_config.checkpoint_policy != supported.checkpoint_policy
            or quant_config.weights != supported.weights
            or quant_config.moe != supported.moe
            or quant_config.format != supported.format
        ):
            raise ValueError("DSpark quantization settings other than bf16 I/O are not supported")
    if drafter_type == "dflash2":
        if quant_config.checkpoint_policy != "preserve":
            raise ValueError("DFlash2 checkpoint_policy other than 'preserve' is not supported")
        weights = canonical.get("weights", {})
        for field_name in ("accuracy_level", "op_types", "overrides"):
            if field_name in weights:
                raise ValueError(f"DFlash2 weights.{field_name} is not supported")
        if quant_config.weights.type not in ("none", "int2", "int4", "int8"):
            raise ValueError("DFlash2 weights.type must be none, int2, int4, or int8")
        if quant_config.weights.type != "none" and quant_config.weights.block_size not in (16, 32, 64, 128, 256):
            raise ValueError("DFlash2 integer weights.block_size must be one of 16, 32, 64, 128, or 256")
        if quant_config.weights.method != "default" or not quant_config.weights.symmetric:
            raise ValueError("DFlash2 supports only symmetric DEFAULT integer weight quantization")
        if quant_config.format.use_qdq:
            raise ValueError("DFlash2 body weights require QOperator format")
        prepack = quant_config.format.matmulnbits_weights_prepacked
        if prepack and quant_config.weights.type == "none":
            raise ValueError("DFlash2 offline prepacking requires integer weights")
        if prepack == 2 and quant_config.weights.type == "int2":
            raise ValueError("DFlash2 INT2 weights support only the SM80 prepacked layout")
        if prepack and quant_config.weights.type == "int2" and quant_config.weights.block_size not in (64, 128):
            raise ValueError("DFlash2 INT2 offline prepacking requires weights.block_size=64 or 128")
        supported_blocks = (32, 64, 128) if prepack == 1 else (64, 128)
        if (
            prepack
            and quant_config.weights.type in ("int4", "int8")
            and quant_config.weights.block_size not in supported_blocks
        ):
            raise ValueError(f"DFlash2 INT4/INT8 offline prepacking requires weights.block_size in {supported_blocks}")
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
        {
            "drafter_type",
            "path",
            "num_draft_tokens",
            "shared_weights",
            "quant_config",
            "attention",
            "optimizations",
            "dspark",
        },
        "drafter_options",
    )
    drafter_type = options.get("drafter_type")
    if drafter_type not in ("none", "mtp", "dflash2", "dspark"):
        raise ValueError("drafter_options.drafter_type must be mtp, dflash2, dspark, or none")
    if drafter_type == "mtp" and flattened.get("exclude_mtp", False):
        raise ValueError("drafter_options.drafter_type=mtp conflicts with legacy extra_options.exclude_mtp")

    legacy_drafters = {name.removesuffix("_path") for name in ("dflash2_path", "dspark_path") if flattened.get(name)}
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

    quant_config = normalize_drafter_quant_config(
        options.get("quant_config", {}), drafter_type, execution_provider, flattened["_target_io_dtype"]
    )
    if drafter_type == "mtp":
        flattened["mtp_quant_config"] = quant_config
    else:
        flattened[f"{drafter_type}_path"] = options["path"]
        if "num_draft_tokens" in options:
            flattened[f"{drafter_type}_num_draft_tokens"] = require_integer(
                options["num_draft_tokens"], "drafter_options.num_draft_tokens"
            )
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
    check_fields(optimizations, {"fuse_mlp_gate_up", "fuse_qkv"}, "drafter_options.optimizations")
    fuse_gate_up = optimizations.get("fuse_mlp_gate_up", False)
    if fuse_gate_up and drafter_type != "dflash2":
        raise ValueError(f"fuse_mlp_gate_up is not supported for drafter_type={drafter_type}")
    fuse_qkv = optimizations.get("fuse_qkv", False)
    if "fuse_qkv" in optimizations and drafter_type != "dflash2":
        raise ValueError(f"fuse_qkv is not supported for drafter_type={drafter_type}")
    if drafter_type == "dflash2":
        flattened["dflash2_fuse_gate_up"] = fuse_gate_up
        flattened["dflash2_fuse_qkv"] = fuse_qkv

    shared_weights = options.get("shared_weights", {})
    check_fields(shared_weights, {"embedding", "lm_head"}, "drafter_options.shared_weights")
    policies = {name: shared_weights.get(name, "auto") for name in ("embedding", "lm_head")}
    if any(policy not in ("auto", "required", "off") for policy in policies.values()):
        raise ValueError("shared weight policies must be auto, required, or off")
    if drafter_type != "dflash2" and any(policy != "auto" for policy in policies.values()):
        raise ValueError(f"explicit shared weight policies are not yet supported for drafter_type={drafter_type}")
    flattened["_shared_weight_policies"] = policies

    dspark = options.get("dspark", {})
    check_fields(dspark, {"top_k"}, "drafter_options.dspark")
    if dspark and drafter_type != "dspark":
        raise ValueError("drafter_options.dspark is valid only for drafter_type=dspark")
    if "top_k" in dspark:
        flattened["dspark_top_k"] = require_integer(dspark["top_k"], "drafter_options.dspark.top_k")

    effective = copy.deepcopy(options)
    effective["quant_config"] = quant_config_schema_dict(quant_config)
    effective["shared_weights"] = policies
    return effective


def flatten_speculative_options(options: dict[str, Any], flattened: dict[str, Any]):
    check_fields(options, {"aux_hidden_state_layers", "state_update_capacity"}, "speculative_options")
    if "aux_hidden_state_layers" in options:
        layers = options["aux_hidden_state_layers"]
        if not isinstance(layers, list) or any(
            isinstance(layer, bool) or not isinstance(layer, int) for layer in layers
        ):
            raise ValueError("speculative_options.aux_hidden_state_layers must be a list of integers")
        flattened["aux_hidden_state_layers"] = ",".join(str(layer) for layer in layers)
    if "state_update_capacity" in options:
        flattened["state_update_capacity"] = require_integer(
            options["state_update_capacity"], "speculative_options.state_update_capacity"
        )


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
    component_options: Any = None,
) -> EffectiveBuilderConfig:
    """Select the legacy adapter or the structured configuration path.

    Omission is significant: absent drafter_options retains automatic discovery,
    whereas an explicit group requests a separate policy. No weights are loaded
    here, although block-drafter config.json is read to resolve auxiliary taps.
    """
    legacy_options = copy.deepcopy(extra_options or {})
    structured_present = any(
        value is not None
        for value in (target_options, drafter_options, speculative_options, runtime_config, component_options)
    )
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
    components = normalize_component_options(component_options)
    drafter = None if drafter_options is None else load_json_object(drafter_options, "drafter_options")
    speculative = load_json_object(speculative_options, "speculative_options")
    runtime = load_json_object(runtime_config, "runtime_config")
    if search is not None:
        legacy_search = load_json_object(search, "search")
        runtime = merge_objects({"search": legacy_search}, runtime)

    flattened, quant_config, effective_precision = flatten_target_options(target, legacy_options, precision, provider)
    if components is not None:
        if drafter is not None:
            raise ValueError("component_options cannot be combined with drafter_options")
        flattened["exclude_lm_head"] = True
        flattened["exclude_mtp"] = True
        flattened["filename"] = components.backbone_filename
    flatten_speculative_options(speculative, flattened)
    effective_drafter = flatten_drafter_options(drafter, flattened, provider)
    validate_runtime_quantization_policy(runtime, effective_drafter, flattened, provider)
    flattened["_runtime_config"] = runtime

    effective_target = copy.deepcopy(target)
    effective_target["quant_config"] = quant_config_schema_dict(quant_config)
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
        component_options=components,
        target_moe_explicit=target_moe_explicit,
    )


def validate_model_dependent_config(effective_config: EffectiveBuilderConfig, model_config: Any):
    if effective_config.version != 2:
        return
    if effective_config.drafter_options is not None and effective_config.drafter_options["drafter_type"] == "mtp":
        architectures = getattr(model_config, "architectures", ())
        if not architectures or architectures[0] not in (
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ):
            raise ValueError("drafter_options.drafter_type=mtp requires a supported Qwen3.5 architecture")
        text_config = getattr(model_config, "text_config", model_config)
        num_mtp_layers = getattr(text_config, "mtp_num_hidden_layers", None)
        if num_mtp_layers is None:
            num_mtp_layers = getattr(model_config, "mtp_num_hidden_layers", 0)
        if not isinstance(num_mtp_layers, int) or isinstance(num_mtp_layers, bool) or num_mtp_layers <= 0:
            raise ValueError("drafter_options.drafter_type=mtp requires a checkpoint with an MTP head")
    quant_config = effective_config.extra_options.get("_quant_config")
    checkpoint_moe_type = effective_config.extra_options.get("moe_quant_type")
    if quant_config is not None and checkpoint_moe_type is not None and quant_config.moe.type != checkpoint_moe_type:
        if effective_config.target_moe_explicit:
            raise ValueError(
                f"target_options.quant_config.moe.type={quant_config.moe.type!r} conflicts with "
                f"checkpoint moe_quant_type={checkpoint_moe_type!r}"
            )
        quant_data = quant_config_schema_dict(quant_config)
        quant_data["moe"]["type"] = checkpoint_moe_type
        quant_config = QuantConfig.from_dict(quant_data)
        effective_config.extra_options["_quant_config"] = quant_config
        effective_config.target_options["quant_config"] = quant_config_schema_dict(quant_config)
    if effective_config.target_moe_explicit or checkpoint_moe_type is not None:
        return
    if quant_config is None or quant_config.weights.type != "none":
        return
    text_config = getattr(model_config, "text_config", model_config)
    num_experts = getattr(text_config, "num_local_experts", getattr(text_config, "num_experts", 0))
    if num_experts:
        raise ValueError(
            "target_options.quant_config.moe.type is required for an MoE checkpoint when weights.type=none"
        )


RUNTIME_TUNABLE_SESSION_OPTIONS = frozenset({"ep.cuda.fpa_intb_gemm"})


def validate_runtime_quantization_policy(
    runtime_config: dict[str, Any],
    drafter_options: dict[str, Any] | None,
    flattened: dict[str, Any],
    execution_provider: str,
):
    """Validate runtime kernel selection against the exported quantization policy."""
    model = runtime_config.get("model", {})
    if not isinstance(model, dict):
        return
    enabled_components = []
    for component_name, component_options in model.items():
        if not isinstance(component_options, dict):
            continue
        session_options = component_options.get("session_options", {})
        if not isinstance(session_options, dict) or "ep.cuda.fpa_intb_gemm" not in session_options:
            continue
        value = session_options["ep.cuda.fpa_intb_gemm"]
        if value not in ("0", "1"):
            raise ValueError(
                f"runtime_config.model.{component_name}.session_options.ep.cuda.fpa_intb_gemm must be '0' or '1'"
            )
        if value == "1":
            enabled_components.append(component_name)
    if enabled_components and execution_provider != "cuda":
        raise ValueError("runtime_config ep.cuda.fpa_intb_gemm=1 is supported only on CUDA")
    if "dflash2" not in enabled_components:
        return
    if not drafter_options or drafter_options.get("drafter_type") != "dflash2":
        return
    quant_config = flattened.get("_drafter_quant_config")
    if quant_config is None or quant_config.weights.type == "none":
        raise ValueError("DFlash2 ep.cuda.fpa_intb_gemm=1 requires integer weights")
    if quant_config.weights.type == "int2" and quant_config.weights.block_size not in (64, 128):
        raise ValueError("DFlash2 INT2 fpA_intB requires weights.block_size=64 or 128")


def validate_runtime_config(runtime_config: dict[str, Any], generated_config: dict[str, Any]):
    """Check a runtime overlay against the completed exported configuration."""
    check_fields(runtime_config, {"search", "speculative", "engine", "model", "runtime_profiles"}, "runtime_config")
    validate_runtime_profiles(runtime_config.get("runtime_profiles", []), generated_config)
    search = runtime_config.get("search", {})
    if not isinstance(search, dict):
        raise ValueError("runtime_config.search must be an object")
    check_fields(
        search,
        {
            "batch_size",
            "blank_penalty",
            "chunk_size",
            "diversity_penalty",
            "do_sample",
            "early_stopping",
            "length_penalty",
            "max_length",
            "min_length",
            "no_repeat_ngram_size",
            "num_beams",
            "num_return_sequences",
            "past_present_share_buffer",
            "random_seed",
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
        },
        "runtime_config.search",
    )
    boolean_search_fields = {"do_sample", "early_stopping", "past_present_share_buffer"}
    integer_search_bounds = {
        "min_length": (0, 2_147_483_647),
        "max_length": (1, 2_147_483_647),
        "batch_size": (1, 32),
        "num_beams": (1, 32),
        "num_return_sequences": (1, 2_147_483_647),
        "top_k": (0, 2_147_483_647),
        "no_repeat_ngram_size": (0, 2_147_483_647),
        "random_seed": (-1, 2_147_483_647),
        "chunk_size": (1, None),
    }
    numeric_search_bounds = {
        "top_p": (0, 1, False),
        "temperature": (0, None, False),
        "repetition_penalty": (0, None, True),
        "blank_penalty": (None, None, False),
        "diversity_penalty": (None, None, False),
        "length_penalty": (None, None, False),
    }
    for field_name in boolean_search_fields & search.keys():
        if not isinstance(search[field_name], bool):
            raise ValueError(f"runtime_config.search.{field_name} must be a boolean")
    for field_name, (minimum, maximum) in integer_search_bounds.items():
        if field_name not in search:
            continue
        value = search[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value != math.trunc(value)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
            raise ValueError(f"runtime_config.search.{field_name} must be an integer {bounds}")
    for field_name, (minimum, maximum, exclusive_minimum) in numeric_search_bounds.items():
        if field_name not in search:
            continue
        value = search[field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"runtime_config.search.{field_name} must be a finite number")
        if minimum is not None and (value < minimum or (exclusive_minimum and value == minimum)):
            comparison = "greater than" if exclusive_minimum else "at least"
            raise ValueError(f"runtime_config.search.{field_name} must be {comparison} {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"runtime_config.search.{field_name} must be at most {maximum}")
    if "max_length" in search:
        max_length = search["max_length"]
        context_length = generated_config.get("model", {}).get("context_length")
        if context_length is None:
            context_length = generated_config.get("search", {}).get("max_length")
        if isinstance(context_length, int) and max_length > context_length:
            raise ValueError("runtime_config.search.max_length exceeds the exported model context_length")
    effective_search = merge_objects(generated_config.get("search", {}), search)
    if effective_search.get("min_length", 0) > effective_search.get("max_length", float("inf")):
        raise ValueError("runtime_config.search.min_length must not exceed max_length")
    if effective_search.get("num_return_sequences", 1) > effective_search.get("num_beams", 1):
        raise ValueError("runtime_config.search.num_return_sequences must not exceed num_beams")

    speculative = runtime_config.get("speculative", {})
    if not isinstance(speculative, dict):
        raise ValueError("runtime_config.speculative must be an object")
    check_fields(speculative, {"max_draft_tokens"}, "runtime_config.speculative")
    if "max_draft_tokens" in speculative:
        max_draft_tokens = speculative["max_draft_tokens"]
        if (
            isinstance(max_draft_tokens, bool)
            or not isinstance(max_draft_tokens, int)
            or not 1 <= max_draft_tokens <= 16
        ):
            raise ValueError("runtime_config.speculative.max_draft_tokens must be an integer between 1 and 16")
        generated_model = generated_config.get("model", {})
        capacities = [
            component["num_draft_tokens"]
            for component in generated_model.values()
            if isinstance(component, dict) and isinstance(component.get("num_draft_tokens"), int)
        ]
        decoder_capacity = generated_model.get("decoder", {}).get("state_update_capacity")
        if isinstance(decoder_capacity, int) and decoder_capacity > 0:
            capacities.append(decoder_capacity)
        has_dynamic_mtp = isinstance(generated_model.get("mtp"), dict)
        if not capacities and not has_dynamic_mtp:
            raise ValueError("runtime_config references absent speculative configuration")
        if not has_dynamic_mtp and capacities and max_draft_tokens > min(capacities):
            raise ValueError("runtime_config.speculative.max_draft_tokens exceeds the exported drafter/state capacity")

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
    if dynamic_batching and not isinstance(generated_config.get("engine", {}).get("dynamic_batching"), dict):
        raise ValueError("runtime_config references absent dynamic_batching configuration")
    if "num_blocks" in dynamic_batching and "gpu_utilization_factor" in dynamic_batching:
        raise ValueError("runtime_config cannot specify both num_blocks and gpu_utilization_factor")
    for field_name in ("max_batch_size", "max_scheduled_tokens", "num_blocks"):
        if field_name not in dynamic_batching:
            continue
        value = dynamic_batching[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"runtime_config.engine.dynamic_batching.{field_name} must be a positive integer")
        if value > 2_147_483_647:
            raise ValueError(f"runtime_config.engine.dynamic_batching.{field_name} must be at most 2147483647")
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
        if not isinstance(generated_component, dict) or (
            "session_options" not in generated_component and component_name != "mtp"
        ):
            raise ValueError(f"runtime_config.model.{component_name} is not a session-bearing component")
        generated_session = generated_component.get("session_options", {})
        runtime_session = component_options.get("session_options", {})
        if not isinstance(runtime_session, dict):
            raise ValueError(f"runtime_config.model.{component_name}.session_options must be an object")
        for key, value in runtime_session.items():
            if key == "provider_options":
                continue
            if key == "ep.cuda.fpa_intb_gemm" and value not in ("0", "1"):
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options.ep.cuda.fpa_intb_gemm must be '0' or '1'"
                )
            validate_session_option(key, value, f"runtime_config.model.{component_name}.session_options")
            if (
                key in generated_session
                and key != "log_id"
                and key not in RUNTIME_TUNABLE_SESSION_OPTIONS
                and value != generated_session[key]
            ):
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options cannot overwrite required session option '{key}'"
                )
        run_options = component_options.get("run_options", {})
        if not isinstance(run_options, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in run_options.items()
        ):
            raise ValueError(f"runtime_config.model.{component_name}.run_options must be an object of strings")
        if (
            component_name in ("dflash2", "dspark")
            and run_options.get("disable_synchronize_execution_providers") == "1"
        ):
            raise ValueError(
                f"runtime_config.model.{component_name}.run_options cannot disable execution-provider synchronization"
            )
        if "provider_options" in runtime_session:
            generated_providers = generated_session.get("provider_options", [])
            runtime_providers = runtime_session["provider_options"]
            if not isinstance(runtime_providers, list) or any(
                not isinstance(entry, dict)
                or len(entry) != 1
                or any(not isinstance(name, str) or not isinstance(options, dict) for name, options in entry.items())
                for entry in runtime_providers
            ):
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options.provider_options "
                    "must be an array of single-provider objects"
                )
            generated_names = {name.casefold() for entry in generated_providers for name in entry}
            runtime_names = {name.casefold() for entry in runtime_providers for name in entry}
            if component_name in ("mtp", "dflash2", "dspark") and not generated_names:
                decoder_providers = (
                    generated_components.get("decoder", {}).get("session_options", {}).get("provider_options", [])
                )
                generated_providers = decoder_providers
                generated_names = {name.casefold() for entry in decoder_providers for name in entry}
            if generated_names != runtime_names:
                raise ValueError(
                    f"runtime_config.model.{component_name}.session_options cannot change execution providers"
                )
            generated_by_name = {
                name.casefold(): options for entry in generated_providers for name, options in entry.items()
            }
            for entry in runtime_providers:
                for provider_name, runtime_options in entry.items():
                    normalized_name = provider_name.casefold()
                    protected_options = GRAPH_DERIVED_PROVIDER_OPTIONS.get(normalized_name, set())
                    tunable_options = RUNTIME_TUNABLE_PROVIDER_OPTIONS.get(normalized_name, set())
                    generated_options = generated_by_name[normalized_name]
                    changed_protected = [
                        option_name
                        for option_name in protected_options & runtime_options.keys()
                        if runtime_options[option_name] != generated_options.get(option_name)
                    ]
                    if changed_protected:
                        raise ValueError(
                            f"runtime_config.model.{component_name}.session_options cannot overwrite "
                            f"graph-derived provider option(s) {sorted(changed_protected)}"
                        )
                    unsupported_options = runtime_options.keys() - protected_options - tunable_options
                    if unsupported_options:
                        raise ValueError(
                            f"runtime_config.model.{component_name}.session_options contains unsupported "
                            f"runtime provider option(s) {sorted(unsupported_options)} for {provider_name}"
                        )
                    if any(not isinstance(value, str) for value in runtime_options.values()):
                        raise ValueError(
                            f"runtime_config.model.{component_name}.session_options provider option values "
                            "must be strings"
                        )


def validate_runtime_profiles(runtime_profiles: Any, generated_config: dict[str, Any]):
    if not isinstance(runtime_profiles, list):
        raise ValueError("runtime_config.runtime_profiles must be an array")

    ids = set()
    ranges = []
    for index, profile in enumerate(runtime_profiles):
        path = f"runtime_config.runtime_profiles[{index}]"
        if not isinstance(profile, dict):
            raise ValueError(f"{path} must be an object")
        check_fields(profile, {"id", "eligibility", "overlay"}, path)
        profile_id = profile.get("id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError(f"{path}.id must be a non-empty string")
        if profile_id in ids:
            raise ValueError(f"duplicate runtime profile id: {profile_id}")
        ids.add(profile_id)

        eligibility = profile.get("eligibility")
        if not isinstance(eligibility, dict):
            raise ValueError(f"{path}.eligibility must be an object")
        check_fields(
            eligibility,
            {"minimum_total_device_memory_bytes", "maximum_total_device_memory_bytes"},
            f"{path}.eligibility",
        )
        minimum = eligibility.get("minimum_total_device_memory_bytes")
        maximum = eligibility.get("maximum_total_device_memory_bytes", 2**53 - 1)
        for field_name, value in (
            ("minimum_total_device_memory_bytes", minimum),
            ("maximum_total_device_memory_bytes", maximum),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**53 - 1:
                raise ValueError(f"{path}.eligibility.{field_name} must be a non-negative integer")
        if maximum < minimum:
            raise ValueError(f"{path}.eligibility maximum must not be below minimum")
        ranges.append((minimum, maximum, profile_id))

        overlay = profile.get("overlay")
        if not isinstance(overlay, dict):
            raise ValueError(f"{path}.overlay must be an object")
        check_fields(overlay, {"model", "engine", "search", "speculative"}, f"{path}.overlay")

        model = overlay.get("model", {})
        if not isinstance(model, dict):
            raise ValueError(f"{path}.overlay.model must be an object")
        check_fields(model, {"decoder"}, f"{path}.overlay.model")
        decoder = model.get("decoder", {})
        if not isinstance(decoder, dict):
            raise ValueError(f"{path}.overlay.model.decoder must be an object")
        check_fields(decoder, {"filename"}, f"{path}.overlay.model.decoder")
        if "filename" in decoder and (not isinstance(decoder["filename"], str) or not decoder["filename"]):
            raise ValueError(f"{path}.overlay.model.decoder.filename must be a non-empty string")

        engine = overlay.get("engine", {})
        if not isinstance(engine, dict):
            raise ValueError(f"{path}.overlay.engine must be an object")
        check_fields(engine, {"dynamic_batching"}, f"{path}.overlay.engine")
        dynamic_batching = engine.get("dynamic_batching", {})
        if not isinstance(dynamic_batching, dict):
            raise ValueError(f"{path}.overlay.engine.dynamic_batching must be an object")
        check_fields(
            dynamic_batching,
            {"num_blocks", "max_batch_size", "max_scheduled_tokens"},
            f"{path}.overlay.engine.dynamic_batching",
        )
        search = overlay.get("search", {})
        if not isinstance(search, dict):
            raise ValueError(f"{path}.overlay.search must be an object")
        check_fields(search, {"chunk_size"}, f"{path}.overlay.search")
        for field_name, value in search.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{path}.overlay.search.{field_name} must be a positive integer")

        speculative = overlay.get("speculative", {})
        if not isinstance(speculative, dict):
            raise ValueError(f"{path}.overlay.speculative must be an object")
        check_fields(speculative, {"max_draft_tokens"}, f"{path}.overlay.speculative")
        runtime_overlay = {key: overlay[key] for key in ("engine", "speculative") if key in overlay}
        try:
            validate_runtime_config(runtime_overlay, generated_config)
        except ValueError as error:
            raise ValueError(f"{path}.overlay: {error}") from error

        if not (decoder or dynamic_batching or search or speculative):
            raise ValueError(f"{path}.overlay must contain at least one overlay field")

    for index, (minimum, maximum, profile_id) in enumerate(ranges):
        for other_minimum, other_maximum, other_id in ranges[index + 1 :]:
            if minimum <= other_maximum and other_minimum <= maximum:
                raise ValueError(f"runtime profile eligibility ranges overlap: {profile_id!r} and {other_id!r}")


def apply_runtime_config(generated_config: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Apply a validated profile after all component sections exist."""
    if not runtime_config:
        return generated_config
    validate_runtime_config(runtime_config, generated_config)
    baseline = copy.deepcopy(generated_config)
    overlay = copy.deepcopy(runtime_config)
    for component_name, component_options in overlay.get("model", {}).items():
        runtime_session = component_options.get("session_options", {})
        if "provider_options" not in runtime_session:
            continue
        generated_components = baseline["model"]
        generated_providers = (
            generated_components[component_name].get("session_options", {}).get("provider_options", [])
        )
        if component_name in ("mtp", "dflash2", "dspark") and not generated_providers:
            generated_providers = (
                generated_components.get("decoder", {}).get("session_options", {}).get("provider_options", [])
            )
        generated_by_name = {
            name.casefold(): (name, options) for entry in generated_providers for name, options in entry.items()
        }
        merged_providers = []
        for entry in runtime_session["provider_options"]:
            for runtime_name, runtime_options in entry.items():
                generated_name, generated_options = generated_by_name[runtime_name.casefold()]
                merged_providers.append({generated_name: merge_objects(generated_options, runtime_options)})
        runtime_session["provider_options"] = merged_providers
    dynamic_batching = overlay.get("engine", {}).get("dynamic_batching", {})
    generated_dynamic_batching = baseline.get("engine", {}).get("dynamic_batching", {})
    if "num_blocks" in dynamic_batching:
        generated_dynamic_batching.pop("gpu_utilization_factor", None)
    elif "gpu_utilization_factor" in dynamic_batching:
        generated_dynamic_batching.pop("num_blocks", None)
    return merge_objects(baseline, overlay)
