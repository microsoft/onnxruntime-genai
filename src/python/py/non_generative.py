# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
"""Python dictionary adapters for the native structured C++ sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from onnxruntime_genai.onnxruntime_genai import (
    _DecisionSession,
    _RankingSession,
)


class RankingSession:
    """CLM session with a bounded, session-local projected-action cache."""

    def __init__(
        self,
        package_path: str | Path,
        providers: list[str] | None = None,
    ):
        self._native = _RankingSession(str(package_path), providers or [])

    def rank(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._native.rank(request)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run structured CLM questions."""
        return self._native.run(request)

    __call__ = rank


class DecisionSession:
    """KEV session with bounded token/row and model-state prefix caches."""

    def __init__(
        self,
        package_path: str | Path,
        providers: list[str] | None = None,
    ):
        self._native = _DecisionSession(str(package_path), providers or [])

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._native.run(request)

    __call__ = decide


__all__ = ["DecisionSession", "RankingSession"]
