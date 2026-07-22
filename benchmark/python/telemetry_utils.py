# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Source/wheel compatibility import for benchmark telemetry privacy helpers."""

import os
import sys

try:
    from onnxruntime_genai.telemetry.path_utils import normalize_execution_provider, sanitize_model_identifier
except ImportError:
    telemetry_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))
    path_added = telemetry_root not in sys.path
    if path_added:
        sys.path.insert(0, telemetry_root)
    try:
        from telemetry.path_utils import normalize_execution_provider, sanitize_model_identifier
    finally:
        if path_added and telemetry_root in sys.path:
            sys.path.remove(telemetry_root)


def get_telemetry():
    """Create telemetry from either the installed wheel or repository source."""
    try:
        from onnxruntime_genai.telemetry import GenAITelemetry  # noqa: PLC0415
    except ImportError:
        telemetry_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))
        path_added = telemetry_root not in sys.path
        if path_added:
            sys.path.insert(0, telemetry_root)
        try:
            from telemetry import GenAITelemetry  # noqa: PLC0415
        finally:
            if path_added and telemetry_root in sys.path:
                sys.path.remove(telemetry_root)

    return GenAITelemetry()


__all__ = ["get_telemetry", "normalize_execution_provider", "sanitize_model_identifier"]
