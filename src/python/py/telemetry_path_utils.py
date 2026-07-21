# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Platform-neutral privacy helpers for telemetry identifiers."""

import ntpath
import os
import posixpath

MAX_TELEMETRY_STRING_LENGTH = 256


def _token_start(value: str, index: int) -> int:
    while index > 0 and not value[index - 1].isspace() and value[index - 1] not in "\"'":
        index -= 1
    return index


def _find_path_anchor(value: str):
    for index, char in enumerate(value):
        if char == "\\" and index + 1 < len(value) and value[index + 1] == "\\":
            return index
        if char == "~" and index + 1 < len(value) and value[index + 1] in "/\\":
            return index
        if (
            char.isascii()
            and char.isalpha()
            and index + 2 < len(value)
            and value[index + 1] == ":"
            and value[index + 2] in "/\\"
        ):
            return index
        if char == "\\":
            separators = 0
            for candidate in value[index:]:
                if candidate in "\r\n":
                    break
                if candidate == "\\":
                    separators += 1
                    if separators >= 2:
                        return _token_start(value, index)
        if char == "/":
            segments = 0
            cursor = index
            while cursor < len(value) and value[cursor] == "/":
                separator_end = cursor + 1
                while separator_end < len(value) and value[separator_end] == "/":
                    separator_end += 1
                cursor = separator_end
                segment_start = cursor
                while cursor < len(value) and value[cursor] not in "/\r\n \t":
                    cursor += 1
                if cursor == segment_start:
                    break
                segments += 1
            if segments >= 2:
                return _token_start(value, index)
    return None


def _truncate_utf8(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_TELEMETRY_STRING_LENGTH:
        return value
    return encoded[:MAX_TELEMETRY_STRING_LENGTH].decode("utf-8", errors="ignore")


def scrub_string_for_telemetry(value: str) -> str:
    """Apply ONNX Runtime's free-text telemetry redaction contract."""
    anchor = _find_path_anchor(value)
    scrubbed = value if anchor is None else value[:anchor] + "[path]"
    return _truncate_utf8(scrubbed)


def normalize_execution_provider(value):
    """Return the canonical telemetry name for an execution provider."""
    return "trt-rtx" if value == "NvTensorRtRtx" else value


def sanitize_model_identifier(value):
    """Preserve model IDs while replacing local paths with ``[path]``."""
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

    return "[path]"
