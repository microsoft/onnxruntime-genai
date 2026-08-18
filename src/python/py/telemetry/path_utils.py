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
_SECRET_KEYS = frozenset(
    {
        "access-key",
        "access_key",
        "accesskey",
        "access-token",
        "access_token",
        "account-key",
        "account_key",
        "accountkey",
        "api-key",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "client-secret",
        "client_secret",
        "connection-string",
        "connection_string",
        "connectionstring",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private-key",
        "private_key",
        "privatekey",
        "pwd",
        "secret",
        "sig",
        "signature",
        "token",
    }
)
_SECRET_SUFFIXES = (
    "access-key",
    "access_key",
    "accesskey",
    "account-key",
    "account_key",
    "accountkey",
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "connection-string",
    "connection_string",
    "connectionstring",
    "credential",
    "credentials",
    "passwd",
    "password",
    "private-key",
    "private_key",
    "privatekey",
    "pwd",
    "secret",
    "signature",
    "token",
)
_ASCII_WHITESPACE = " \t\r\n\v\f"


def _token_start(value: str, index: int) -> int:
    while index > 0 and not value[index - 1].isspace() and value[index - 1] not in "\"'":
        index -= 1
    return index


def _ascii_lower(value: str) -> str:
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value)


def _is_ascii_alpha(char: str) -> bool:
    return "A" <= char <= "Z" or "a" <= char <= "z"


def _is_ascii_alnum(char: str) -> bool:
    return _is_ascii_alpha(char) or "0" <= char <= "9"


def _is_secret_key(key: str) -> bool:
    folded = _ascii_lower(key)
    return folded in _SECRET_KEYS or any(
        len(folded) > len(suffix) and folded.endswith(suffix)
        for suffix in _SECRET_SUFFIXES
    )


def _find_url_anchor(value: str) -> int | None:
    for index, char in enumerate(value):
        if not _is_ascii_alpha(char) or (
            index > 0
            and (_is_ascii_alnum(value[index - 1]) or value[index - 1] in "+-.")
        ):
            continue
        end = index + 1
        while end < len(value) and (
            _is_ascii_alnum(value[end]) or value[end] in "+-."
        ):
            end += 1
        if value.startswith("://", end):
            return index
    return None


def _secret_key_end(value: str, key_start: int) -> int:
    key_end = key_start + 1
    while key_end < len(value) and (
        _is_ascii_alnum(value[key_end]) or value[key_end] in "_-."
    ):
        key_end += 1
    return key_end


def _secret_value_start(
    value: str, key_start: int, cli_option: bool, key_end: int | None = None
) -> int | None:
    key_end = _secret_key_end(value, key_start) if key_end is None else key_end
    if not _is_secret_key(value[key_start:key_end]):
        return None

    separator = key_end
    if separator < len(value) and value[separator] in "\"'":
        separator += 1
    before_whitespace = separator
    while separator < len(value) and value[separator] in _ASCII_WHITESPACE:
        separator += 1
    assignment = separator < len(value) and value[separator] in "=:"
    separated_cli_value = (
        cli_option
        and separator > before_whitespace
        and separator < len(value)
        and value[separator] != "-"
    )
    if not assignment and not separated_cli_value:
        return None

    value_start = separator + 1 if assignment else separator
    while value_start < len(value) and value[value_start] in _ASCII_WHITESPACE:
        value_start += 1
    if (
        value_start < len(value)
        and value[value_start] not in "&;\r\n"
    ):
        return value_start
    return None


def _find_secret_value_anchor(value: str) -> int | None:
    boundaries = _ASCII_WHITESPACE + "?&#;,\"'([{/-"
    index = 0
    while index < len(value):
        char = value[index]
        if not _is_ascii_alpha(char) or (
            index > 0 and value[index - 1] not in boundaries
        ):
            index += 1
            continue
        key_end = _secret_key_end(value, index)
        if not _is_secret_key(value[index:key_end]):
            index = key_end
            continue
        cli_option = index > 0 and value[index - 1] in "-/"
        value_start = _secret_value_start(value, index, cli_option, key_end)
        if value_start is not None:
            return value_start
        index = key_end
    return None


def _find_credential_anchor(value: str) -> int | None:
    userinfo_terminators = _ASCII_WHITESPACE + "\"\\/?#[]{}"
    authority_terminators = _ASCII_WHITESPACE + "\"')},;/?#"
    token_start = 0
    colon = None
    index = 0
    while index < len(value):
        char = value[index]
        if colon is None:
            if char in userinfo_terminators:
                token_start = index + 1
            elif char == ":" and index > token_start:
                colon = index
            index += 1
            continue

        if char in userinfo_terminators:
            token_start = index + 1
            colon = None
            index += 1
            continue
        if char != "@" or colon + 1 == index:
            index += 1
            continue

        host_start = index + 1
        if host_start == len(value):
            index += 1
            continue
        if value[host_start] == "[":
            host_end = value.find("]", host_start + 1)
            if host_end > host_start + 1:
                return token_start
            return None

        host_end = host_start
        while host_end < len(value) and value[host_end] not in authority_terminators:
            host_end += 1
        if host_end > host_start:
            return token_start
        if host_end == len(value):
            return None
        token_start = host_end + 1
        colon = None
        index = host_end + 1
    return None


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
                and (
                    not _is_ascii_alpha(value[index + 1])
                    or _secret_value_start(value, index + 1, True) is None
                )
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


def _find_redaction_anchor(value: str) -> int | None:
    positions = (
        _find_url_anchor(value),
        _find_path_anchor(value),
        _find_secret_value_anchor(value),
        _find_credential_anchor(value),
    )
    return min((position for position in positions if position is not None), default=None)


def _find_truncated_sensitive_token_anchor(value: str, next_char: str) -> int | None:
    token_start = len(value)
    while token_start > 0 and value[token_start - 1] not in _ASCII_WHITESPACE + "\"":
        token_start -= 1
    token = value[token_start:]
    continues_path = bool(token) and next_char in "/\\"
    contains_path_separator = "/" in token or "\\" in token
    if continues_path and not contains_path_separator:
        previous_end = token_start
        while previous_end > 0:
            while (
                previous_end > 0
                and value[previous_end - 1] in _ASCII_WHITESPACE + "\""
            ):
                previous_end -= 1
            previous_start = previous_end
            while (
                previous_start > 0
                and value[previous_start - 1] not in _ASCII_WHITESPACE + "\""
            ):
                previous_start -= 1
            previous = value[previous_start:previous_end]
            if "/" in previous or "\\" in previous:
                return previous_start
            previous_end = previous_start
    if (
        continues_path
        or (bool(token) and next_char == ":")
        or ":" in token
        or contains_path_separator
    ):
        return token_start
    return None


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _scrub_string_for_telemetry(value: str, max_bytes: int) -> str:
    scan = _truncate_utf8(value, max_bytes)
    anchor = _find_redaction_anchor(scan)
    if anchor is None and len(scan) < len(value):
        anchor = _find_truncated_sensitive_token_anchor(scan, value[len(scan)])
    if anchor is None:
        return scan

    prefix = _truncate_utf8(value[:anchor], max_bytes - len("[path]"))
    return prefix + "[path]"


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
                safe_child = (
                    "[path]"
                    if isinstance(key, str) and _is_secret_key(key)
                    else scrub_value_for_telemetry(child)
                )
                entries.append((safe_key, safe_child))
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
