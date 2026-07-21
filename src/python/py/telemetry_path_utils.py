# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Platform-neutral privacy helpers for telemetry identifiers."""

import ntpath
import os
import posixpath


def normalize_execution_provider(value):
    """Return the canonical telemetry name for an execution provider."""
    return "trt-rtx" if value == "NvTensorRtRtx" else value


def sanitize_model_identifier(value):
    """Preserve model IDs while reducing local paths to a safe basename."""
    if not isinstance(value, str) or not value:
        return value

    drive, _ = ntpath.splitdrive(value)
    is_windows_path = bool(drive) or ntpath.isabs(value)
    is_posix_path = posixpath.isabs(value)
    is_explicit_relative_path = value.startswith(("./", "../", ".\\", "..\\", "~/", "~\\"))
    is_unprefixed_path = "\\" in value or value.count("/") > 1
    if not (is_windows_path or is_posix_path or is_explicit_relative_path or is_unprefixed_path):
        if value.count("/") == 1:
            return value
        try:
            if not os.path.exists(value):
                return value
        except (OSError, ValueError):
            return value

    path_module = ntpath if is_windows_path or "\\" in value else posixpath
    basename = path_module.basename(path_module.normpath(value))
    return basename if basename not in {"", ".", ".."} else "<path>"
