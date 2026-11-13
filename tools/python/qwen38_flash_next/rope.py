# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Export-time positional encoding and independent cache-capacity contracts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HF_YARN_REVISION = "598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1"
HF_ROPE_SHA256 = "56fe0e57136198248913418b759a40986870fe1048a564dc32c675690a57d9df"
TARGET_NAMES = (
    "model.rotary_emb.cos_cache",
    "model.rotary_emb.sin_cache",
    "capture.indexer_cos_cache.folded",
    "capture.indexer_sin_cache.folded",
)
DRAFT_NAMES = ("dflash2.cos_cache", "dflash2.sin_cache")


@dataclass(frozen=True)
class RopeConfig:
    rope_type: str = "default"
    factor: float = 1.0
    original_max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None
    mrope_sections: tuple[int, ...] = (11, 11, 10)

    def __post_init__(self):
        if self.rope_type not in ("default", "yarn"):
            raise ValueError("rope_type must be 'default' or 'yarn'")
        if self.head_dim != 256 or self.partial_rotary_factor != 0.25:
            raise ValueError("The RC5 graph requires head_dim=256 and partial_rotary_factor=0.25 (rotary_dim=64)")
        if tuple(self.mrope_sections) != (11, 11, 10):
            raise ValueError("The RC5 rotary layout requires mrope_sections=[11,11,10]")
        object.__setattr__(self, "mrope_sections", tuple(self.mrope_sections))
        if type(self.original_max_position_embeddings) is not int or self.original_max_position_embeddings <= 0:
            raise ValueError("original_max_position_embeddings must be a positive integer")
        for name in ("factor", "rope_theta", "beta_fast", "beta_slow"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.factor < 1 or self.rope_theta <= 1 or self.beta_fast < self.beta_slow:
            raise ValueError("Require factor>=1, rope_theta>1 and beta_fast>=beta_slow")
        if self.attention_factor is not None and (
            not isinstance(self.attention_factor, (int, float))
            or isinstance(self.attention_factor, bool)
            or not math.isfinite(self.attention_factor)
            or self.attention_factor <= 0
        ):
            raise ValueError("attention_factor must be finite and positive")
        if self.rope_type == "default" and (self.factor != 1 or self.attention_factor not in (None, 1)):
            raise ValueError("Scaling requires rope_type='yarn'")

    @classmethod
    def from_dict(cls, value: dict) -> RopeConfig:
        if not isinstance(value, dict):
            raise ValueError("RoPE configuration must be a JSON object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"Invalid RoPE configuration: {exc}") from exc

    def to_dict(self) -> dict:
        result = asdict(self)
        result["mrope_sections"] = list(self.mrope_sections)
        return result


@dataclass(frozen=True)
class ContextCapacity:
    context_length: int
    cache_layout: str = "unified"

    def __post_init__(self):
        if type(self.context_length) is not int or not 0 < self.context_length <= 2**31 - 17:
            raise ValueError("max-context-length must be a positive integer with room for 16 lookahead positions")
        if self.cache_layout not in ("unified", "fixed"):
            raise ValueError("cache-layout must be 'unified' or 'fixed'")
        if self.cache_layout == "unified" and self.context_length % 256:
            raise ValueError("Unified context length must be a multiple of the Engine's 256-token block size")

    @property
    def kv_capacity(self) -> int:
        return self.context_length + 8

    @property
    def state_capacity(self) -> int:
        return self.kv_capacity // 4 + 1

    @property
    def positional_rows(self) -> int:
        # Eight verification rows, plus another eight absolute-position lookahead rows.
        return self.kv_capacity + 8

    @property
    def graph_filename(self) -> str:
        if self.cache_layout == "unified":
            return "model.onnx"
        count = self.context_length
        suffix = (
            f"{count // 1048576}m"
            if count % 1048576 == 0
            else (f"{count // 1024}k" if count % 1024 == 0 else str(count))
        )
        return f"model_{suffix}.onnx"

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "kv_capacity": self.kv_capacity,
            "state_capacity": self.state_capacity,
            "positional_rows": self.positional_rows,
        }


def yarn_parameters(config: RopeConfig) -> tuple[torch.Tensor, float]:
    """Use the exact upstream implementation qualified by the 1M prototype."""
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise ImportError("YaRN export requires tools/python/qwen38_flash_next/requirements-yarn.txt")
    source = Path(spec.origin).parent / "modeling_rope_utils.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != HF_ROPE_SHA256:
        raise ValueError(f"Unqualified Transformers YaRN implementation; install transformers@{HF_YARN_REVISION}")
    reference = importlib.import_module("transformers.modeling_rope_utils").ROPE_INIT_FUNCTIONS["yarn"]
    hf_config = SimpleNamespace(
        head_dim=config.head_dim,
        hidden_size=6144,
        num_attention_heads=24,
        max_position_embeddings=config.original_max_position_embeddings,
        standardize_rope_params=lambda: None,
        rope_parameters=config.to_dict(),
    )
    return reference(hf_config, torch.device("cpu"))


DEFAULT_ROPE = RopeConfig()


def target_rope(positions, config: RopeConfig = DEFAULT_ROPE) -> tuple[torch.Tensor, torch.Tensor]:
    if config.rope_type == "yarn":
        inv, magnitude = yarn_parameters(config)
    else:
        inv = 1.0 / (config.rope_theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        magnitude = 1.0
    phase = torch.as_tensor(positions, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = torch.cos(phase), torch.sin(phase)
    if magnitude != 1.0:
        cos, sin = cos * magnitude, sin * magnitude
    return cos, sin


def dflash_rope(positions, head_dim: int = 128, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    phase = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    return tuple(torch.from_numpy(fn(phase).astype(np.float32)) for fn in (np.cos, np.sin))


def position_config(rope: RopeConfig, target_rows: int, draft_rows: int, capacity: ContextCapacity | None) -> dict:
    return {
        "rope": rope.to_dict(),
        "target_rows": target_rows,
        "draft_rows": draft_rows,
        "capacity": capacity.to_dict() if capacity else None,
        "yarn_reference_revision": HF_YARN_REVISION if rope.rope_type == "yarn" else None,
    }


def validate_reused_positions(root: Path, locations: dict[str, dict]) -> dict:
    """Validate actual table bytes, including legacy banks without positional provenance."""
    meta_path = root / "bank_metadata.json"
    declared = json.loads(meta_path.read_text()).get("position_config") if meta_path.is_file() else None
    rope = RopeConfig.from_dict(declared["rope"]) if declared else RopeConfig()
    if declared and rope.rope_type == "yarn" and declared.get("yarn_reference_revision") != HF_YARN_REVISION:
        raise ValueError("Unqualified reused YaRN reference revision")
    rows_by_group = {}
    for names, width0, key in ((TARGET_NAMES, 32, "target_rows"), (DRAFT_NAMES, 64, "draft_rows")):
        group_rows = set()
        for index, name in enumerate(names):
            if name not in locations:
                raise ValueError(f"Missing positional asset {name}")
            meta = locations[name]
            dims = meta["dims"]
            width = width0 * (2 if names == TARGET_NAMES and index >= 2 else 1)
            if len(dims) != 2 or type(dims[0]) is not int or dims[0] <= 0 or dims[1] != width:
                raise ValueError(f"Invalid positional shape for {name}: {dims}")
            path = root / meta["location"]
            offset, length = int(meta["offset"]), int(meta["length"])
            dtype_id = int(meta["dtype"])
            allowed_dtypes = (16, 1) if names == TARGET_NAMES else (16,)
            elem_bytes = 4 if dtype_id == 1 else 2
            if (
                not path.resolve().is_relative_to(root.resolve())
                or not path.is_file()
                or dtype_id not in allowed_dtypes
                or offset < 0
                or offset % elem_bytes
                or length != dims[0] * width * elem_bytes
                or offset + length > path.stat().st_size
            ):
                raise ValueError(f"Invalid positional external data for {name}")
            group_rows.add(dims[0])
            np_dtype = "<f4" if dtype_id == 1 else "<u2"
            actual = np.memmap(path, mode="r", dtype=np_dtype, offset=offset, shape=tuple(dims))
            for start in range(0, dims[0], 4096):
                positions = np.arange(start, min(start + 4096, dims[0]))
                values = target_rope(positions, rope) if names == TARGET_NAMES else dflash_rope(positions)
                expected = values[index % 2]
                if dtype_id == 16:
                    expected = expected.to(torch.bfloat16)
                else:
                    expected = expected.to(torch.float32)
                if width == 64 and names == TARGET_NAMES:
                    expected = torch.cat((expected, expected), dim=-1)
                expected_np = expected.numpy() if dtype_id == 1 else expected.view(torch.uint16).numpy()
                if not np.array_equal(actual[start : start + len(positions)], expected_np):
                    raise ValueError(f"Stale or unqualified positional asset {name} at row {start}")
            del actual
        if len(group_rows) != 1:
            raise ValueError(f"Inconsistent {key} in reused banks")
        rows_by_group[key] = group_rows.pop()
        if declared and declared[key] != rows_by_group[key]:
            raise ValueError(f"Stale positional coverage metadata: {key}")
    if declared and declared.get("capacity"):
        cap = declared["capacity"]
        capacity = ContextCapacity(cap["context_length"], cap["cache_layout"])
        if capacity.to_dict() != cap or min(rows_by_group.values()) < capacity.positional_rows:
            raise ValueError("Stale positional capacity metadata")
    return declared or position_config(rope, **rows_by_group, capacity=None)
