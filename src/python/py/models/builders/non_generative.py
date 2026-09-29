# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Dispatch non-generative head exports by their verified artifact layout."""

from __future__ import annotations

import json
from pathlib import Path

from .clm import CLM_ARTIFACT_REVISION, export_clm_components, is_clm_artifact
from .kev import (
    KEV_ADAPTER_REVISION,
    KEV_BASE_REVISION,
    export_kev_components,
    is_kev_artifact,
    load_kev_checkpoint,
)


def detect_model_specific_artifact(source: str | Path) -> str:
    """Identify one supported artifact from its local file/checkpoint layout.

    Returns ``"clm"`` or ``"kev"``. Repository IDs are intentionally not used,
    so copied pinned snapshots behave identically. No match or multiple matches
    raise ``ValueError`` rather than guessing a builder.
    """
    source = Path(source)
    # Evaluate every signature so a directory containing two model layouts is
    # rejected as ambiguous instead of depending on dispatch order.
    matches = [
        name
        for name, predicate in (("clm", is_clm_artifact), ("kev", is_kev_artifact))
        if predicate(source)
    ]
    if len(matches) != 1:
        detail = "ambiguous" if matches else "unrecognized"
        raise ValueError(f"{detail} model-specific artifact layout at {source}")
    return matches[0]


def validate_model_specific_backbone(options, config) -> str:
    """Validate a detected artifact against its required backbone contract.

    ``options`` supplies the component source and revision pins; ``config`` is
    the loaded base Hugging Face configuration. Returns the detected model type.
    Wrong revisions, architecture, or hidden size raise ``ValueError``. CLM's
    caller-selected base revision is deliberately not compared to an invented
    upstream pin.
    """
    model_type = detect_model_specific_artifact(options.model_source)
    architecture = config.architectures[0]
    if model_type == "clm":
        if options.artifact_revision != CLM_ARTIFACT_REVISION:
            raise ValueError(f"CLM artifact_revision must be {CLM_ARTIFACT_REVISION}")
        if architecture != "Qwen3ForCausalLM":
            raise ValueError("CLM requires a Qwen3ForCausalLM backbone")
        if int(config.hidden_size) != 4096:
            raise ValueError("CLM requires the Qwen/Qwen3-8B hidden size (4096)")
    else:
        if options.artifact_revision != KEV_ADAPTER_REVISION:
            raise ValueError(f"KEV artifact_revision must be {KEV_ADAPTER_REVISION}")
        if options.base_revision != KEV_BASE_REVISION:
            raise ValueError(f"KEV base_revision must be {KEV_BASE_REVISION}")
        if architecture != "Qwen3_5ForConditionalGeneration":
            raise ValueError("KEV requires a Qwen3.5 dense backbone")
        if int(config.hidden_size) != 2560:
            raise ValueError("KEV requires the Qwen/Qwen3.5-4B-Base hidden size (2560)")
    return model_type


def configure_model_specific_export(options, config, extra_options) -> str:
    """Apply graph-export options after validating the base configuration.

    The returned model type identifies the selected builder. KEV wires its PEFT
    adapter from the separate component source; both models remove the LM head.
    A conflicting explicit adapter path raises ``ValueError``.
    """
    model_type = validate_model_specific_backbone(options, config)
    if model_type == "kev":
        existing = extra_options.get("adapter_path")
        if existing is not None and Path(existing).resolve() != Path(options.model_source).resolve():
            raise ValueError("KEV component model_source must match adapter_path")
        extra_options["adapter_path"] = options.model_source
    extra_options["exclude_lm_head"] = True
    return model_type


def prepare_model_specific_hf(options, extra_options, base_source: str | None = None) -> str:
    """Prepare pinned Hugging Face sources before config/tokenizer loading.

    ``base_source`` remains the source for model configuration, weights, and the
    tokenizer, while ``options.model_source`` remains the head/adapter artifact.
    The function returns the detected model type and records this split in
    ``extra_options``. Missing base sources, invalid pins, or malformed KEV
    checkpoints raise ``ValueError`` before expensive backbone export.
    """
    if not isinstance(base_source, str) or not base_source:
        raise ValueError("model-specific exports require a separate non-empty base model source")
    model_type = detect_model_specific_artifact(options.model_source)
    extra_options["base_revision"] = options.base_revision
    extra_options["_model_specific_hf"] = {
        "base_model": base_source,
        "base_revision": options.base_revision,
        "tokenizer": base_source,
        "adapter": options.model_source if model_type == "kev" else None,
        "component": options.model_source,
    }
    if model_type == "kev":
        if options.artifact_revision != KEV_ADAPTER_REVISION:
            raise ValueError(f"KEV artifact_revision must be {KEV_ADAPTER_REVISION}")
        if options.base_revision != KEV_BASE_REVISION:
            raise ValueError(f"KEV base_revision must be {KEV_BASE_REVISION}")
        load_kev_checkpoint(options.model_source)
        extra_options["adapter_path"] = options.model_source
    elif options.artifact_revision != CLM_ARTIFACT_REVISION:
        raise ValueError(f"CLM artifact_revision must be {CLM_ARTIFACT_REVISION}")
    return model_type


def export_model_specific_components(options, output_root: Path):
    """Build the detected model-specific head and write its package manifest.

    ``options`` contains the pinned local artifact and output naming policy.
    Detection or checkpoint/schema failures propagate as ``ValueError``; a
    successful call writes the ONNX head and ``component_manifest.json``.
    """
    model_type = detect_model_specific_artifact(options.model_source)
    if model_type == "clm":
        manifest = export_clm_components(options, output_root)
    else:
        manifest = export_kev_components(options, output_root)
    with open(output_root / "component_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=4)
