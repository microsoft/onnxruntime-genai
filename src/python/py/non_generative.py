# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
"""Python dictionary adapters for the native structured C++ sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from onnxruntime_genai.onnxruntime_genai import (
    ComponentSession,
    _DecisionSession,
    _RankingSession,
)


class RankingSession:
    """CLM session with a bounded, session-local projected-action cache."""

    def __init__(self, package_path: str | Path, providers: list[str] | None = None,
                 cache_capacity: int = 256,
                 cache_capacity_bytes: int = 64 * 1024 * 1024):
        self._native = _RankingSession(
            str(package_path), providers or [], cache_capacity, cache_capacity_bytes
        )

    def rank(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._native.run(request)

    @property
    def cache_stats(self) -> dict[str, int]:
        return self._native.cache_stats()

    def clear_cache(self) -> None:
        self._native.clear_cache()

    def invalidate_cache(self) -> None:
        self._native.invalidate_cache()

    def set_cache_capacity(self, capacity: int, capacity_bytes: int) -> None:
        self._native.set_cache_capacity(capacity, capacity_bytes)

    __call__ = rank


class DecisionSession:
    """KEV session with bounded token/row and model-state prefix caches."""

    def __init__(self, package_path: str | Path, providers: list[str] | None = None,
                 cache_capacity: int = 512,
                 cache_capacity_bytes: int = 16 * 1024 * 1024,
                 prefix_reuse: bool = True,
                 prefix_cache_capacity: int = 32,
                 prefix_cache_capacity_bytes: int = 512 * 1024 * 1024):
        self._native = _DecisionSession(
            str(package_path), providers or [], cache_capacity, cache_capacity_bytes,
            prefix_reuse, prefix_cache_capacity, prefix_cache_capacity_bytes
        )

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._native.run(request)

    @property
    def cache_stats(self) -> dict[str, int]:
        return self._native.cache_stats()

    @property
    def prefix_cache_stats(self) -> dict[str, int]:
        return self._native.prefix_cache_stats()

    @property
    def prefix_reuse_enabled(self) -> bool:
        return self._native.prefix_reuse_enabled

    @prefix_reuse_enabled.setter
    def prefix_reuse_enabled(self, enabled: bool) -> None:
        self._native.prefix_reuse_enabled = enabled

    @property
    def prefix_reuse_status(self) -> str:
        return self._native.prefix_reuse_status

    def clear_cache(self) -> None:
        self._native.clear_cache()

    def invalidate_cache(self) -> None:
        self._native.invalidate_cache()

    def set_cache_capacity(self, capacity: int, capacity_bytes: int) -> None:
        self._native.set_cache_capacity(capacity, capacity_bytes)

    def set_prefix_cache_capacity(self, capacity: int, capacity_bytes: int) -> None:
        self._native.set_prefix_cache_capacity(capacity, capacity_bytes)

    __call__ = decide


__all__ = ["ComponentSession", "RankingSession", "DecisionSession"]
