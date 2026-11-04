# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
"""Python dictionary adapters for the native structured C++ sessions."""

from __future__ import annotations

import json
from typing import Any

from onnxruntime_genai.onnxruntime_genai import (
    Config,
    _NonGenerativeSession,
)


class NonGenerativeSession:
    """Package-selected non-generative session using JSON-compatible dictionaries."""

    def __init__(self, config: Config):
        self._native = _NonGenerativeSession(config)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self._native.run(json.dumps(request)))

    __call__ = run


__all__ = ["NonGenerativeSession"]
