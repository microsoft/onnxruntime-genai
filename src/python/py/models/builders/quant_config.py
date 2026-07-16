# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unified model-builder quantization config.

This module implements the *option surface* proposed in the model-builder
quantization design: a single, structured ``QuantConfig`` (loadable from JSON /
``dict``) plus a back-compat adapter that desugars today's flat ``extra_options``
into the same structure. It is intentionally **pure data + validation**: it does
not touch the ONNX graph or import any quantizer. The builder can consume a
resolved ``QuantConfig`` later without changing this file.

Scope note: this covers the targets the model builder implements today — dense
``weights`` (MatMul), ``moe`` (QMoE) experts, and layout-only ``runtime`` knobs —
together with per-node/per-layer ``overrides`` (used for mixed precision). The
KV-cache and auxiliary-model (MTP) targets from the broader design are out of
scope here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 1. Unified quant-dtype vocabulary
# ---------------------------------------------------------------------------
#
# A dtype string names a storage/quantization scheme. It resolves to a descriptor
# with a ``kind``, bit-width, and (for integers) a default signedness. Adding a
# dtype is a single row here — no new option name is introduced.


@dataclass(frozen=True)
class DtypeDescriptor:
    name: str
    kind: str  # "float" | "int" | "mx"
    bits: int
    signed: Optional[bool] = None  # int only: True == symmetric int*, False == uint*
    block_size: Optional[int] = None  # fixed block size for mx dtypes (else None)

    @property
    def is_quantized(self) -> bool:
        """A float pass-through dtype (fp32/fp16/bf16) does not quantize weights."""
        return self.kind != "float"


# Only the dtypes the builder supports today. Float dtypes are I/O pass-through
# (no weight quantization). ``mxfp4`` fixes block size 32 and is MoE-only.
_DTYPES: dict[str, DtypeDescriptor] = {
    "fp32": DtypeDescriptor("fp32", "float", 32),
    "fp16": DtypeDescriptor("fp16", "float", 16),
    "bf16": DtypeDescriptor("bf16", "float", 16),
    "int8": DtypeDescriptor("int8", "int", 8, signed=True),
    "uint8": DtypeDescriptor("uint8", "int", 8, signed=False),
    "int4": DtypeDescriptor("int4", "int", 4, signed=True),
    "uint4": DtypeDescriptor("uint4", "int", 4, signed=False),
    "mxfp4": DtypeDescriptor("mxfp4", "mx", 4, block_size=32),
    "none": DtypeDescriptor("none", "float", 0),  # explicit "do not quantize this target"
}

IO_DTYPES = ("fp16", "bf16", "fp32")


def resolve_dtype(name: str) -> DtypeDescriptor:
    """Resolve a dtype string to its descriptor. Raises ``ValueError`` if unknown."""
    if not isinstance(name, str):
        raise ValueError(f"quant dtype must be a string, got {type(name).__name__}")
    key = name.strip().lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown quant dtype '{name}'. Supported: {sorted(_DTYPES)}")
    return _DTYPES[key]


# ---------------------------------------------------------------------------
# 2. Per-node / per-layer overrides
# ---------------------------------------------------------------------------
#
# Each weight-bearing target accepts an ordered ``overrides`` list. Each entry has
# a ``match`` and the fields to apply (``type`` or ``exclude``). First match wins.

# Named node groups preserved from the current ``matmul_mixed_precision`` surface.
MATCH_PRESETS = ("last_matmul", "mixed_layers", "linear_attn")
# Supported match keys (a subset of the full design; enough for today's presets +
# explicit exclusions). Multiple keys in one match are ANDed.
MATCH_KEYS = ("name", "name_regex", "layers", "role", "op_type", "preset")


@dataclass
class Override:
    match: dict[str, Any]
    type: Optional[str] = None
    exclude: bool = False

    def __post_init__(self):
        if not isinstance(self.match, dict) or not self.match:
            raise ValueError("override 'match' must be a non-empty object")
        for key in self.match:
            if key not in MATCH_KEYS:
                raise ValueError(f"override match key must be one of {list(MATCH_KEYS)}, got '{key}'")
        preset = self.match.get("preset")
        if preset is not None and preset not in MATCH_PRESETS:
            raise ValueError(f"override match preset must be one of {list(MATCH_PRESETS)}, got '{preset}'")
        if not self.exclude and self.type is None:
            raise ValueError("override must set either 'type' or 'exclude: true'")
        if self.exclude and self.type is not None:
            raise ValueError("override cannot set both 'type' and 'exclude'")
        if self.type is not None:
            resolve_dtype(self.type)  # validate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Override":
        unknown = set(data) - {"match", "type", "exclude"}
        if unknown:
            raise ValueError(f"unknown override field(s): {sorted(unknown)}")
        return cls(match=data["match"], type=data.get("type"), exclude=bool(data.get("exclude", False)))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"match": dict(self.match)}
        if self.exclude:
            out["exclude"] = True
        if self.type is not None:
            out["type"] = self.type
        return out


# ---------------------------------------------------------------------------
# 3. Targets
# ---------------------------------------------------------------------------


def _normalize_block_size(value: Any) -> int:
    """Normalize a block size. ``0`` / ``"per_channel"`` both mean per-channel."""
    if isinstance(value, str):
        if value.strip().lower() == "per_channel":
            return 0
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"block_size must be an int or 'per_channel', got {value!r}")
    if value < 0:
        raise ValueError(f"block_size must be >= 0 (0 == per_channel), got {value}")
    return value


@dataclass
class WeightsConfig:
    """Dense (non-MoE) weight quantization."""

    type: str = "none"
    block_size: int = 32
    symmetric: bool = True
    method: str = "default"  # default | rtn | k_quant
    accuracy_level: int = 0
    op_types: tuple[str, ...] = ("MatMul",)
    overrides: list[Override] = field(default_factory=list)

    METHODS = ("default", "rtn", "k_quant")

    def __post_init__(self):
        descriptor = resolve_dtype(self.type)
        if self.method not in self.METHODS:
            raise ValueError(f"weights.method must be one of {list(self.METHODS)}, got '{self.method}'")
        self.block_size = _normalize_block_size(self.block_size)
        if descriptor.kind == "mx" and self.block_size not in (0, descriptor.block_size):
            raise ValueError(
                f"weights.type={self.type} fixes block_size={descriptor.block_size}; got {self.block_size}"
            )
        self.op_types = tuple(self.op_types)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeightsConfig":
        unknown = set(data) - {"type", "block_size", "symmetric", "method", "accuracy_level", "op_types", "overrides"}
        if unknown:
            raise ValueError(f"unknown weights field(s): {sorted(unknown)}")
        overrides = [Override.from_dict(o) for o in data.get("overrides", [])]
        return cls(
            type=data.get("type", "none"),
            block_size=data.get("block_size", 32),
            symmetric=bool(data.get("symmetric", True)),
            method=data.get("method", "default"),
            accuracy_level=int(data.get("accuracy_level", 0)),
            op_types=tuple(data.get("op_types", ("MatMul",))),
            overrides=overrides,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "block_size": self.block_size,
            "symmetric": self.symmetric,
            "method": self.method,
            "accuracy_level": self.accuracy_level,
            "op_types": list(self.op_types),
            "overrides": [o.to_dict() for o in self.overrides],
        }


@dataclass
class MoEConfig:
    """MoE expert (QMoE) weight quantization."""

    type: str = "int4"
    block_size: int = 32
    weights_prepacked: int = -1  # CUDA QMoE layout: -1 auto | 0 raw | 1 prepacked

    def __post_init__(self):
        descriptor = resolve_dtype(self.type)
        self.block_size = _normalize_block_size(self.block_size)
        if descriptor.kind == "mx":
            # mxfp4 mandates block size 32.
            self.block_size = descriptor.block_size
        if self.weights_prepacked not in (-1, 0, 1):
            raise ValueError(f"moe.weights_prepacked must be -1, 0, or 1, got {self.weights_prepacked}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoEConfig":
        unknown = set(data) - {"type", "block_size", "weights_prepacked"}
        if unknown:
            raise ValueError(f"unknown moe field(s): {sorted(unknown)}")
        return cls(
            type=data.get("type", "int4"),
            block_size=data.get("block_size", 32),
            weights_prepacked=int(data.get("weights_prepacked", -1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "block_size": self.block_size, "weights_prepacked": self.weights_prepacked}


@dataclass
class RuntimeConfig:
    """Layout / emission knobs (not numeric)."""

    use_qdq: bool = False
    matmulnbits_weights_prepacked: int = 0  # CUDA fpA_intB layout: 0 off | 1 SM80 | 2 SM90

    def __post_init__(self):
        if self.matmulnbits_weights_prepacked not in (0, 1, 2):
            raise ValueError(
                f"runtime.matmulnbits_weights_prepacked must be 0, 1, or 2, got {self.matmulnbits_weights_prepacked}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        unknown = set(data) - {"use_qdq", "matmulnbits_weights_prepacked"}
        if unknown:
            raise ValueError(f"unknown runtime field(s): {sorted(unknown)}")
        return cls(
            use_qdq=bool(data.get("use_qdq", False)),
            matmulnbits_weights_prepacked=int(data.get("matmulnbits_weights_prepacked", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_qdq": self.use_qdq,
            "matmulnbits_weights_prepacked": self.matmulnbits_weights_prepacked,
        }


# ---------------------------------------------------------------------------
# 4. Top-level config
# ---------------------------------------------------------------------------

# Legacy compound ``int4_algo_config`` names -> (base method, mixed-precision presets).
# This is the single source of truth for the alias desugaring shared by the builder
# (via ``desugar_int4_algo_config``) and ``QuantConfig``.
_LEGACY_INT4_ALGO_ALIASES: dict[str, tuple[str, dict[str, str]]] = {
    "rtn_last": ("rtn", {"last_matmul": "int8"}),
    "k_quant_last": ("k_quant", {"last_matmul": "int8"}),
    "k_quant_mixed": ("k_quant", {"last_matmul": "int8", "mixed_layers": "int8"}),
    "k_quant_linear": ("k_quant", {"last_matmul": "int8", "linear_attn": "int8"}),
}

# ``--precision`` -> weights.type. Float precisions do not quantize weights.
_PRECISION_TO_WEIGHTS_TYPE = {
    "int4": "int4",
    "int8": "int8",
    "fp16": "none",
    "bf16": "none",
    "fp32": "none",
}


def desugar_int4_algo_config(extra_options: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Desugar the flat int4 options into ``(base_method, {preset: quant_type})``.

    Expands ``int4_algo_config`` (including the legacy compound aliases) and merges the
    explicit ``matmul_mixed_precision`` entries on top. The base method is **not** validated
    here — the builder defers rejecting an unknown method to algo-config creation, and
    ``WeightsConfig`` validates it for the structured surface. This is the single source of
    truth for the desugaring shared by the builder and ``QuantConfig``.
    """
    algo = extra_options.get("int4_algo_config", "default")
    base_method = algo
    placement: dict[str, str] = {}
    if algo in _LEGACY_INT4_ALGO_ALIASES:
        base_method, implied = _LEGACY_INT4_ALGO_ALIASES[algo]
        placement.update(implied)
    placement.update(_normalize_mixed_precision(extra_options.get("matmul_mixed_precision", {})))
    return base_method, placement


@dataclass
class QuantConfig:
    io_dtype: str = "fp16"
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self):
        if self.io_dtype not in IO_DTYPES:
            raise ValueError(f"io_dtype must be one of {list(IO_DTYPES)}, got '{self.io_dtype}'")

    # -- Loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantConfig":
        """Load a structured ``quantization`` object (the value of the ``quantization`` key)."""
        if not isinstance(data, dict):
            raise ValueError("quantization config must be an object")
        # Allow either the bare object or a wrapper with a top-level "quantization" key.
        if "quantization" in data and isinstance(data["quantization"], dict):
            data = data["quantization"]
        unknown = set(data) - {"io_dtype", "weights", "moe", "runtime"}
        if unknown:
            raise ValueError(f"unknown quantization field(s): {sorted(unknown)}")
        return cls(
            io_dtype=data.get("io_dtype", "fp16"),
            weights=WeightsConfig.from_dict(data.get("weights", {})),
            moe=MoEConfig.from_dict(data.get("moe", {})),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {})),
        )

    @classmethod
    def from_json(cls, text_or_path: str) -> "QuantConfig":
        """Load from an inline JSON string or a path to a JSON file."""
        stripped = text_or_path.strip()
        if stripped.startswith("{"):
            data = json.loads(stripped)
        else:
            with open(text_or_path, encoding="utf-8") as handle:
                data = json.load(handle)
        return cls.from_dict(data)

    # -- Back-compat adapter ----------------------------------------------

    @classmethod
    def from_extra_options(
        cls,
        extra_options: dict[str, Any],
        precision: str = "int4",
        execution_provider: str = "cuda",
    ) -> "QuantConfig":
        """Desugar today's flat ``extra_options`` (+ ``precision``) into a ``QuantConfig``.

        This is the §9 back-compat mapping, scoped to the weights / moe / runtime
        targets the builder implements. It intentionally mirrors the desugaring in
        ``Model`` (``resolve_int4_quant_config`` / ``moe_quant_type`` handling) so the
        two surfaces stay in lock-step.
        """
        extra_options = dict(extra_options or {})

        weights_type = _PRECISION_TO_WEIGHTS_TYPE.get(precision, "none")

        # --- weights: method + mixed-precision placement -----------------
        base_method, placement = desugar_int4_algo_config(extra_options)

        overrides: list[Override] = [
            Override(match={"preset": selector}, type=quant_type) for selector, quant_type in placement.items()
        ]
        for node in extra_options.get("int4_nodes_to_exclude", []) or []:
            overrides.append(Override(match={"name": node}, exclude=True))

        is_symmetric = extra_options.get("int4_is_symmetric", True)
        weights = WeightsConfig(
            type=weights_type,
            block_size=int(extra_options.get("int4_block_size", 32)),
            symmetric=bool(is_symmetric),
            method=base_method,
            accuracy_level=int(
                extra_options.get("int4_accuracy_level", 4 if execution_provider in ("cpu", "webgpu") else 0)
            ),
            op_types=tuple(extra_options.get("int4_op_types_to_quantize", ("MatMul",))),
            overrides=overrides,
        )

        # --- moe ---------------------------------------------------------
        moe_quant_type = extra_options.get("moe_quant_type")
        if moe_quant_type is None:
            moe_quant_type = "int8" if extra_options.get("use_8bits_moe", False) else "int4"
        # QMoE default block size: 128 on TRT-RTX, 32 elsewhere (mxfp4 is pinned to 32 in MoEConfig).
        default_moe_block = 128 if execution_provider == "trt-rtx" else 32
        moe = MoEConfig(
            type=moe_quant_type,
            block_size=int(extra_options.get("qmoe_block_size", default_moe_block)),
            weights_prepacked=int(extra_options.get("qmoe_weights_prepacked", -1)),
        )

        # --- runtime -----------------------------------------------------
        runtime = RuntimeConfig(
            use_qdq=bool(extra_options.get("use_qdq", False)),
            matmulnbits_weights_prepacked=int(extra_options.get("matmulnbits_weights_prepacked", 0)),
        )

        io_dtype = precision if precision in IO_DTYPES else "fp16"
        return cls(io_dtype=io_dtype, weights=weights, moe=moe, runtime=runtime)

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "io_dtype": self.io_dtype,
            "weights": self.weights.to_dict(),
            "moe": self.moe.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


def _normalize_mixed_precision(value: Any) -> dict[str, str]:
    """Normalize a ``matmul_mixed_precision`` value into a ``{preset: quant_type}`` dict.

    Accepts an already-parsed dict or a ``"selector:quant_type[,...]"`` string, matching
    ``Model.normalize_matmul_mixed_precision``.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        items = list(value.items())
    else:
        items = []
        for entry in str(value).split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(f"matmul_mixed_precision entries must be 'selector:quant_type', got '{entry}'.")
            selector, quant_type = entry.split(":", 1)
            items.append((selector.strip(), quant_type.strip()))

    normalized: dict[str, str] = {}
    for selector, quant_type in items:
        if selector not in MATCH_PRESETS:
            raise ValueError(
                f"matmul_mixed_precision selector must be one of {list(MATCH_PRESETS)}, got '{selector}'."
            )
        resolve_dtype(quant_type)  # validate the quant type name
        normalized[selector] = quant_type
    return normalized
