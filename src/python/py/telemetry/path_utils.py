# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Privacy helpers for telemetry identifiers."""

import ntpath
import os
import posixpath
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from datetime import date, datetime, time, timedelta
from uuid import UUID

MAX_TELEMETRY_STRING_LENGTH = 40_960
MAX_ERROR_MESSAGE_LENGTH = 40_960


def _token_start(value: str, index: int) -> int:
    while index > 0 and not value[index - 1].isspace() and value[index - 1] not in "\"'":
        index -= 1
    return index


def _find_path_anchor(value: str):
    index = 0
    slash_token_end = 0
    slash_token_start = 0
    slash_token_analyzed = False
    while index < len(value):
        char = value[index]
        if index == 0 and char == "/":
            return 0
        if (
            char == "."
            and (
                index == 0
                or value[index - 1].isspace()
                or value[index - 1] in "\"'=([{,;:"
            )
            and (
                value.startswith("./", index)
                or value.startswith("../", index)
                or value.startswith(".\\", index)
                or value.startswith("..\\", index)
            )
        ):
            return index
        if char == ":" and index + 2 < len(value) and value[index + 1 : index + 3] == "//":
            scheme_start = index
            while scheme_start > 0 and (
                value[scheme_start - 1].isascii()
                and (value[scheme_start - 1].isalnum() or value[scheme_start - 1] in "+-.")
            ):
                scheme_start -= 1
            return scheme_start
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
            and not (value[index + 2] == "/" and index + 3 < len(value) and value[index + 3] == "/")
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
            if index >= slash_token_end:
                slash_token_start = _token_start(value, index)
                slash_token_end = index
                while (
                    slash_token_end < len(value)
                    and not value[slash_token_end].isspace()
                    and value[slash_token_end] not in "\"'"
                ):
                    slash_token_end += 1
                slash_token_analyzed = False
            if (
                index + 1 < len(value)
                and value[index + 1] not in "/\r\n \t"
                and value[index - 1] in "\"' \t=([{,;"
            ):
                return index
            if not slash_token_analyzed:
                slash_token_analyzed = True
                segments = 0
                cursor = index
                while cursor < slash_token_end and value[cursor] == "/":
                    separator_end = cursor + 1
                    while separator_end < slash_token_end and value[separator_end] == "/":
                        separator_end += 1
                    cursor = separator_end
                    segment_start = cursor
                    while cursor < slash_token_end and value[cursor] not in "/\r\n \t":
                        cursor += 1
                    if cursor == segment_start:
                        break
                    segments += 1
                if segments >= 2:
                    return slash_token_start
        index += 1
    return None


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _scrub_string_for_telemetry(value: str, max_bytes: int) -> str:
    anchor = _find_path_anchor(value)
    scrubbed = value if anchor is None else value[:anchor] + "[path]"
    return _truncate_utf8(scrubbed, max_bytes)


def scrub_string_for_telemetry(value: str) -> str:
    """Redact and cap a general telemetry string."""
    return _scrub_string_for_telemetry(value, MAX_TELEMETRY_STRING_LENGTH)


def scrub_error_message_for_telemetry(value: str) -> str:
    """Redact and cap an error message at 40,960 UTF-8 bytes."""
    return _scrub_string_for_telemetry(value, MAX_ERROR_MESSAGE_LENGTH)


def scrub_value_for_telemetry(value):
    """Recursively scrub strings and path-like values before serialization."""
    if isinstance(value, os.PathLike):
        return "[path]"
    if isinstance(value, str):
        return scrub_string_for_telemetry(value)
    if value is None or isinstance(
        value,
        (bool, int, float, bytes, bytearray, datetime, date, time, timedelta, UUID),
    ):
        return value
    if isinstance(value, Mapping):
        entries = []
        for key, child in value.items():
            if isinstance(key, os.PathLike):
                safe_key = "[path]"
            elif isinstance(key, str):
                safe_key = scrub_string_for_telemetry(key)
            else:
                try:
                    safe_key = scrub_string_for_telemetry(str(key))
                except Exception:
                    safe_key = f"[unsupported:{type(key).__name__}]"
            if safe_key:
                entries.append((safe_key, scrub_value_for_telemetry(child)))
        return dict(sorted(entries, key=lambda entry: entry[0]))
    if isinstance(value, list):
        return [scrub_value_for_telemetry(child) for child in value]
    if isinstance(value, tuple):
        return tuple(scrub_value_for_telemetry(child) for child in value)
    if isinstance(value, AbstractSet):
        scrubbed = [scrub_value_for_telemetry(child) for child in value]
        return sorted(scrubbed, key=repr)
    try:
        return scrub_string_for_telemetry(str(value))
    except Exception:
        return f"[unsupported:{type(value).__name__}]"


def normalize_execution_provider(value):
    """Return the canonical telemetry name for an execution provider."""
    return "trt-rtx" if value == "NvTensorRtRtx" else value


def sanitize_model_identifier(value):
    """Preserve model IDs while replacing local paths with ``[path]``."""
    if isinstance(value, os.PathLike):
        return "[path]"
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
