# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""JSON serialization helper for Common Schema format."""

import base64
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID


class CommonSchemaJsonSerializationHelper:
    """Helper for serializing values to Common Schema JSON v4.0 format."""

    ONE_COLLECTOR_TENANCY_SYMBOL = "o"
    SCHEMA_VERSION = "4.0"

    @staticmethod
    def serialize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            utc_value = value.astimezone(timezone.utc)
            return utc_value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, time):
            return value.isoformat()
        if isinstance(value, timedelta):
            total_seconds = int(value.total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            sign = "-" if total_seconds < 0 else ""
            return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")
        if isinstance(value, (list, tuple)):
            return [CommonSchemaJsonSerializationHelper.serialize_value(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k:
                    result[str(k)] = CommonSchemaJsonSerializationHelper.serialize_value(v)
            return result
        try:
            return str(value)
        except Exception:
            return f"ERROR: type {type(value).__name__} is not supported"

    @staticmethod
    def create_event_envelope(
        event_name: str, timestamp: datetime, ikey: str, data: dict[str, Any], extensions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        envelope = {
            "ver": CommonSchemaJsonSerializationHelper.SCHEMA_VERSION,
            "name": event_name,
            "time": CommonSchemaJsonSerializationHelper.serialize_value(timestamp),
            "iKey": ikey,
            "data": CommonSchemaJsonSerializationHelper.serialize_value(data),
        }
        if extensions:
            envelope["ext"] = CommonSchemaJsonSerializationHelper.serialize_value(extensions)
        return envelope

    @staticmethod
    def serialize_to_json_bytes(envelope: dict[str, Any]) -> bytes:
        return json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
