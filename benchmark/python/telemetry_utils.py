# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Source/wheel compatibility import for benchmark telemetry privacy helpers."""

import os
import sys
from contextlib import contextmanager

_SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))


@contextmanager
def _source_path():
    path_added = _SOURCE_ROOT not in sys.path
    if path_added:
        sys.path.insert(0, _SOURCE_ROOT)
    try:
        yield
    finally:
        if path_added and _SOURCE_ROOT in sys.path:
            sys.path.remove(_SOURCE_ROOT)


try:
    from onnxruntime_genai.telemetry.path_utils import normalize_execution_provider, sanitize_model_identifier
except ModuleNotFoundError:
    with _source_path():
        from telemetry.path_utils import normalize_execution_provider, sanitize_model_identifier


def get_telemetry():
    """Create telemetry from either the installed wheel or repository source."""
    try:
        from onnxruntime_genai.telemetry import GenAITelemetry  # noqa: PLC0415
    except ModuleNotFoundError:
        with _source_path():
            from telemetry import GenAITelemetry  # noqa: PLC0415

    return GenAITelemetry()


__all__ = ["get_telemetry", "normalize_execution_provider", "sanitize_model_identifier"]
