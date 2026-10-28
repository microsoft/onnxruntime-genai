# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
"""Python dictionary adapters for the native structured C++ sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnxruntime_genai.onnxruntime_genai import (
    _NonGenerativeSession,
)


class RankingSession:
    """CLM session with a bounded, session-local projected-action cache."""

    def __init__(
        self,
        package_path: str | Path,
        providers: list[str] | None = None,
    ):
        self._native = _NonGenerativeSession(str(package_path), providers or [])

    def rank(self, request: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self._native.run(json.dumps(request)))

    __call__ = rank


class DecisionSession:
    """KEV session with bounded token/row and model-state prefix caches."""

    def __init__(
        self,
        package_path: str | Path,
        providers: list[str] | None = None,
    ):
        self._native = _NonGenerativeSession(str(package_path), providers or [])

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self._native.run(json.dumps(request)))

    __call__ = decide


__all__ = ["DecisionSession", "RankingSession"]
