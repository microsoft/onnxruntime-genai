# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Source/wheel compatibility import for benchmark telemetry privacy helpers."""

import os
import sys

try:
    from onnxruntime_genai.telemetry_path_utils import sanitize_model_identifier
except ImportError:
    telemetry_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))
    path_added = telemetry_root not in sys.path
    if path_added:
        sys.path.insert(0, telemetry_root)
    try:
        from telemetry_path_utils import sanitize_model_identifier
    finally:
        if path_added:
            sys.path.remove(telemetry_root)

__all__ = ["sanitize_model_identifier"]
