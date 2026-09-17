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
KV cache is out of scope. Auxiliary models such as MTP consume an independent
``QuantConfig`` instance rather than adding an auxiliary-model target here.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import onnx_ir as ir

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
# (no weight quantization). The microscaling FP4 dtypes are MoE-only and fix their
# block size: ``mxfp4`` -> 32 (ue8m0 block scales), ``nvfp4`` -> 16 (FP8-E4M3 block
# scales + per-expert FP32 global scale).
_DTYPES: dict[str, DtypeDescriptor] = {
    "fp32": DtypeDescriptor("fp32", "float", 32),
    "fp16": DtypeDescriptor("fp16", "float", 16),
    "bf16": DtypeDescriptor("bf16", "float", 16),
    "int8": DtypeDescriptor("int8", "int", 8, signed=True),
    "uint8": DtypeDescriptor("uint8", "int", 8, signed=False),
    "int4": DtypeDescriptor("int4", "int", 4, signed=True),
    "uint4": DtypeDescriptor("uint4", "int", 4, signed=False),
    "mxfp4": DtypeDescriptor("mxfp4", "mx", 4, block_size=32),
    "nvfp4": DtypeDescriptor("nvfp4", "mx", 4, block_size=16),
    "none": DtypeDescriptor("none", "float", 0),  # explicit "do not quantize this target"
}

IO_DTYPES = ("fp16", "bf16", "fp32")

# Accepted values for the `kv_cache_quant_scheme` extra option: a bit width plus a scale
# granularity. The KV cache is not part of `QuantConfig` yet (see the scope note above), so
# this stays a standalone vocabulary that both `check_extra_options()` and the builder read.
KV_CACHE_QUANT_SCHEMES = frozenset(
    {"none", "int4_per_token", "int8_per_token"}
    | {
        f"{bit_width}_{granularity}"
        for bit_width in ("int8", "int4", "fp8")
        for granularity in ("per_tensor", "per_channel")
    }
)

# Divisor used to turn a calibrated threshold into a stored scale: scale = threshold / qmax.
# Signed integers use the full 2^(bits-1) range; fp8 e4m3 uses its largest finite magnitude.
# A calibration file that declares its own `qmax` can therefore be retargeted to another bit
# width by the ratio of the two values.
KV_CACHE_CALIBRATION_QMAX = {"int8": 128.0, "int4": 8.0, "fp8": 448.0}


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
# Supported selectors are exact ONNX node names and named groups.
MATCH_KEYS = ("name", "preset")


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
        if len(self.match) != 1:
            raise ValueError("override match must contain exactly one of 'name' or 'preset'")
        if any(not isinstance(value, str) or not value.strip() for value in self.match.values()):
            raise ValueError("override match values must be non-empty strings")
        if not isinstance(self.exclude, bool):
            raise ValueError("override exclude must be a boolean")
        preset = self.match.get("preset")
        if preset is not None and preset not in MATCH_PRESETS:
            raise ValueError(f"override match preset must be one of {list(MATCH_PRESETS)}, got '{preset}'")
        if not self.exclude and self.type is None:
            raise ValueError("override must set either 'type' or 'exclude: true'")
        if self.exclude and self.type is not None:
            raise ValueError("override cannot set both 'type' and 'exclude'")
        if self.type is not None:
            self.type = resolve_dtype(self.type).name
            if self.type not in ("int4", "int8"):
                raise ValueError("override type must be int4 or int8; use exclude: true to keep graph precision")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Override":
        if not isinstance(data, dict):
            raise ValueError("override must be an object")
        unknown = set(data) - {"match", "type", "exclude"}
        if unknown:
            raise ValueError(f"unknown override field(s): {sorted(unknown)}")
        return cls(match=data.get("match"), type=data.get("type"), exclude=data.get("exclude", False))

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
        self.type = descriptor.name
        if descriptor.signed is False:
            self.symmetric = False
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
        if not isinstance(data.get("overrides", []), list):
            raise ValueError("weights.overrides must be a list")
        if not isinstance(data.get("symmetric", True), bool):
            raise ValueError("weights.symmetric must be a boolean")
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
        self.type = descriptor.name
        self.block_size = _normalize_block_size(self.block_size)
        if descriptor.kind == "mx":
            # Microscaling FP4 mandates a fixed block size (mxfp4 -> 32, nvfp4 -> 16).
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
        if not isinstance(data.get("use_qdq", False), bool):
            raise ValueError("runtime.use_qdq must be a boolean")
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

# Legacy compound ``algo_config`` values -> (base method, mixed-precision presets).
# This is the single source of truth for the alias desugaring shared by the builder
# (via ``desugar_algo_config``) and ``QuantConfig``.
_LEGACY_ALGO_ALIASES: dict[str, tuple[str, dict[str, str]]] = {
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


def desugar_algo_config(extra_options: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Desugar the flat weight-only quant options into ``(base_method, {preset: quant_type})``.

    Expands ``algo_config`` (including the legacy compound aliases) and merges the
    explicit ``matmul_mixed_precision`` entries on top. The base method is **not** validated
    here — the builder defers rejecting an unknown method to algo-config creation, and
    ``WeightsConfig`` validates it for the structured surface. This is the single source of
    truth for the desugaring shared by the builder and ``QuantConfig``.
    """
    algo = extra_options.get("algo_config", "default")
    base_method = algo
    placement: dict[str, str] = {}
    if algo in _LEGACY_ALGO_ALIASES:
        base_method, implied = _LEGACY_ALGO_ALIASES[algo]
        placement.update(implied)
    placement.update(_normalize_mixed_precision(extra_options.get("matmul_mixed_precision", {})))
    return base_method, placement


@dataclass
class QuantConfig:
    io_dtype: str = "fp16"
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint_policy: str = "preserve"
    specified_fields: frozenset[str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.io_dtype not in IO_DTYPES:
            raise ValueError(f"io_dtype must be one of {list(IO_DTYPES)}, got '{self.io_dtype}'")
        if self.checkpoint_policy not in ("preserve", "requantize"):
            raise ValueError("checkpoint_policy must be preserve or requantize")
        if self.specified_fields is None:
            self.specified_fields = frozenset(
                f"{section}.{name}" if isinstance(value, dict) else section
                for section, value in self.to_dict().items()
                for name in (value if isinstance(value, dict) else (section,))
            )

    def to_onnx_dtypes(self) -> tuple[ir.DataType, ir.DataType]:
        io_dtype = {
            "fp16": ir.DataType.FLOAT16,
            "bf16": ir.DataType.BFLOAT16,
            "fp32": ir.DataType.FLOAT,
        }[self.io_dtype]
        weights = resolve_dtype(self.weights.type)
        if weights.kind == "mx":
            raise ValueError(
                f"Dense weights.type={weights.name} is not supported; use int4/int8/none for dense weights "
                "and select mxfp4/nvfp4 independently through moe.type"
            )
        if weights.kind != "int":
            return io_dtype, io_dtype

        signed = weights.signed is not False and self.weights.symmetric
        if weights.bits == 8:
            return io_dtype, ir.DataType.INT8 if signed else ir.DataType.UINT8
        return io_dtype, ir.DataType.INT4 if signed else ir.DataType.UINT4

    # -- Loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantConfig":
        """Load a structured ``quantization`` object (the value of the ``quantization`` key)."""
        if not isinstance(data, dict):
            raise ValueError("quantization config must be an object")
        # Allow either the bare object or a wrapper with a top-level "quantization" key.
        if set(data) == {"quantization"} and isinstance(data["quantization"], dict):
            data = data["quantization"]
        unknown = set(data) - {"io_dtype", "weights", "moe", "runtime", "checkpoint_policy"}
        if unknown:
            raise ValueError(f"unknown quantization field(s): {sorted(unknown)}")
        for section in ("weights", "moe", "runtime"):
            if not isinstance(data.get(section, {}), dict):
                raise ValueError(f"{section} must be an object")
        specified_fields = frozenset(
            f"{key}.{field_name}" if isinstance(value, dict) else key
            for key, value in data.items()
            for field_name in (value if isinstance(value, dict) else (key,))
        )
        return cls(
            io_dtype=data.get("io_dtype", "fp16"),
            weights=WeightsConfig.from_dict(data.get("weights", {})),
            moe=MoEConfig.from_dict(data.get("moe", {})),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {})),
            checkpoint_policy=data.get("checkpoint_policy", "preserve"),
            specified_fields=specified_fields,
        )

    @classmethod
    def from_json(cls, text_or_path: str) -> "QuantConfig":
        """Load from an inline JSON string or a path to a JSON file."""
        return cls.load(text_or_path)

    @classmethod
    def load(cls, value: Any, defaults: QuantConfig | None = None) -> QuantConfig:
        """Load a full config, applying only supplied fields over optional defaults."""
        if isinstance(value, cls):
            return copy.deepcopy(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                value = json.loads(stripped)
            else:
                with open(value, encoding="utf-8") as handle:
                    value = json.load(handle)
        if isinstance(value, dict) and set(value) == {"quantization"}:
            value = value["quantization"]
        parsed = cls.from_dict(value)
        if defaults is None:
            return parsed
        merged = defaults.to_dict()
        for key, setting in value.items():
            if isinstance(setting, dict):
                for name, field_value in setting.items():
                    if key == "weights" and name == "overrides":
                        merged[key][name] = copy.deepcopy(field_value) + merged[key][name]
                    else:
                        merged[key][name] = field_value
            else:
                merged[key] = setting
        result = cls.from_dict(merged)
        result.specified_fields = parsed.specified_fields
        return result

    def validate(self, execution_provider: str):
        """Validate supported export combinations before loading weights."""
        self.to_onnx_dtypes()
        weights = resolve_dtype(self.weights.type)
        if weights.kind == "int" and (
            self.weights.block_size < 16 or self.weights.block_size & (self.weights.block_size - 1)
        ):
            raise ValueError("Dense integer block_size must be a power of two of at least 16")
        if self.weights.type in IO_DTYPES and self.weights.type != self.io_dtype:
            raise ValueError("Unquantized weights.type must match io_dtype; use none for graph precision")
        if any(not override.exclude for override in self.weights.overrides) and weights.kind != "int":
            raise ValueError("Integer overrides require integer weights.type")
        if self.runtime.use_qdq and (
            weights.bits == 8 or any(override.type == "int8" for override in self.weights.overrides)
        ):
            raise ValueError("INT8 weights and overrides require QOperator, not use_qdq")
        if self.moe.type in ("mxfp4", "nvfp4") and execution_provider != "cuda":
            raise ValueError(f"moe.type={self.moe.type} requires the CUDA EP")
        if self.moe.type in ("mxfp4", "nvfp4") and self.io_dtype == "fp32":
            raise ValueError("FP4 MoE requires fp16 or bf16 I/O")
        if self.moe.type in ("uint4", "uint8"):
            raise ValueError("MoE supports symmetric int4/int8, not unsigned quantization")
        if self.runtime.matmulnbits_weights_prepacked and execution_provider != "cuda":
            raise ValueError("matmulnbits_weights_prepacked requires the CUDA EP")
        if self.runtime.matmulnbits_weights_prepacked and (self.runtime.use_qdq or not self.weights.symmetric):
            raise ValueError("matmulnbits_weights_prepacked requires symmetric QOperator weights")
        if self.weights.method != "default" and self.runtime.use_qdq:
            raise ValueError("use_qdq requires weights.method=default")
        if set(self.weights.op_types) - {"MatMul", "Gather"}:
            raise ValueError("weights.op_types supports only MatMul and Gather")
        if (
            execution_provider == "cuda"
            and resolve_dtype(self.moe.type).kind == "int"
            and self.moe.block_size not in (0, 32, 64, 128)
        ):
            raise ValueError("CUDA integer MoE block_size must be 0, 32, 64 or 128")

    # -- Back-compat adapter ----------------------------------------------

    @classmethod
    def from_extra_options(
        cls,
        extra_options: dict[str, Any],
        precision: ir.DataType | str = ir.DataType.INT4,
        execution_provider: str = "cuda",
    ) -> "QuantConfig":
        """Desugar today's flat ``extra_options`` (+ ``precision``) into a ``QuantConfig``.

        This is the §9 back-compat mapping, scoped to the weights / moe / runtime
        targets the builder implements. It intentionally mirrors the desugaring in
        ``Model`` (``resolve_quant_config`` / ``moe_quant_type`` handling) so the
        two surfaces stay in lock-step.
        """
        precision = onnx_dtype_to_precision(precision)
        extra_options = dict(extra_options or {})
        if isinstance(extra_options.get("quant_config"), cls):
            return copy.deepcopy(extra_options["quant_config"])

        weights_type = _PRECISION_TO_WEIGHTS_TYPE.get(precision, "none")

        # --- weights: method + mixed-precision placement -----------------
        base_method, placement = desugar_algo_config(extra_options)

        overrides: list[Override] = []
        for node in extra_options.get("nodes_to_exclude", []) or []:
            overrides.append(Override(match={"name": node}, exclude=True))
        overrides.extend(
            Override(match={"preset": selector}, type=placement[selector])
            for selector in ("linear_attn", "mixed_layers", "last_matmul")
            if selector in placement
        )

        is_symmetric = extra_options.get("is_symmetric", True)
        weights = WeightsConfig(
            type=weights_type,
            block_size=int(extra_options.get("block_size", 32)),
            symmetric=bool(is_symmetric),
            method=base_method,
            accuracy_level=int(
                extra_options.get("accuracy_level", 4 if execution_provider in ("cpu", "webgpu") else 0)
            ),
            op_types=tuple(extra_options.get("op_types_to_quantize", ("MatMul",))),
            overrides=overrides,
        )

        # --- moe ---------------------------------------------------------
        moe_quant_type = extra_options.get("moe_quant_type")
        if moe_quant_type is None:
            # Match the model precision unless the MoE target is configured independently.
            if precision == "int8" or extra_options.get("use_8bits_moe", False):
                moe_quant_type = "int8"
            elif precision in IO_DTYPES:
                moe_quant_type = "none"
            else:
                moe_quant_type = "int4"
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
        if precision in ("int4", "int8"):
            if execution_provider == "cpu" or (
                execution_provider == "webgpu" and extra_options.get("use_webgpu_fp32", False)
            ):
                io_dtype = "fp32"
            elif (
                precision == "int4"
                and execution_provider in ("cuda", "trt-rtx")
                and extra_options.get("use_cuda_bf16", False)
            ):
                io_dtype = "bf16"
        defaults = cls(io_dtype=io_dtype, weights=weights, moe=moe, runtime=runtime)
        if "quant_config" not in extra_options:
            return defaults
        result = cls.load(extra_options["quant_config"], defaults)
        flat_fields = {
            "block_size": "weights.block_size",
            "is_symmetric": "weights.symmetric",
            "algo_config": "weights.method",
            "accuracy_level": "weights.accuracy_level",
            "op_types_to_quantize": "weights.op_types",
            "moe_quant_type": "moe.type",
            "qmoe_block_size": "moe.block_size",
            "qmoe_weights_prepacked": "moe.weights_prepacked",
            "use_qdq": "runtime.use_qdq",
            "matmulnbits_weights_prepacked": "runtime.matmulnbits_weights_prepacked",
        }
        replaced = [
            key
            for key, field_name in flat_fields.items()
            if key in extra_options and field_name in result.specified_fields
        ]
        if replaced:
            warnings.warn(f"quant_config overrides flat options: {', '.join(replaced)}", stacklevel=2)
        return result

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "io_dtype": self.io_dtype,
            "checkpoint_policy": self.checkpoint_policy,
            "weights": self.weights.to_dict(),
            "moe": self.moe.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


def _normalize_mixed_precision(value: Any) -> dict[str, str]:
    """Normalize a ``matmul_mixed_precision`` value into a ``{preset: quant_type}`` dict.

    Accepts an already-parsed dict or a ``"selector:quant_type[,...]"`` string. This is the
    single source of truth for ``matmul_mixed_precision`` parsing shared by the builder
    (via ``desugar_algo_config``) and ``QuantConfig``.
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
            raise ValueError(f"matmul_mixed_precision selector must be one of {list(MATCH_PRESETS)}, got '{selector}'.")
        resolve_dtype(quant_type)  # validate the quant type name
        normalized[selector] = quant_type
    return normalized


def onnx_dtype_to_precision(onnx_dtype: ir.DataType | str) -> str:
    """Map the resolved ONNX weight dtype to a `QuantConfig` precision string.

    Only used to seed the QuantConfig's `weights.type` / `io_dtype`; the builder's numeric
    quantization knobs are read from the resolved config, not from this precision.
    """
    if isinstance(onnx_dtype, str):
        return onnx_dtype.lower()
    return {
        ir.DataType.INT4: "int4",
        ir.DataType.UINT4: "int4",
        ir.DataType.INT8: "int8",
        ir.DataType.UINT8: "int8",
        ir.DataType.BFLOAT16: "bf16",
        ir.DataType.FLOAT16: "fp16",
        ir.DataType.FLOAT: "fp32",
    }.get(onnx_dtype, "fp16")
