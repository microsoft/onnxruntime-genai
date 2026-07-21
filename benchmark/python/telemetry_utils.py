# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Privacy helpers shared by benchmark telemetry call sites."""

import ntpath
import os
import posixpath


def sanitize_model_identifier(value: str) -> str:
    """Preserve model IDs while reducing local paths to a safe basename."""
    if not value:
        return value

    drive, _ = ntpath.splitdrive(value)
    is_windows_path = bool(drive) or ntpath.isabs(value)
    is_posix_path = posixpath.isabs(value)
    is_explicit_relative_path = value.startswith(("./", "../", ".\\", "..\\", "~/", "~\\"))
    if not (is_windows_path or is_posix_path or is_explicit_relative_path or os.path.exists(value)):
        return value

    path_module = ntpath if is_windows_path or "\\" in value else posixpath
    basename = path_module.basename(path_module.normpath(value))
    return basename if basename not in {"", ".", ".."} else "<path>"
