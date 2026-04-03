# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Connection string parser for OneCollector exporter."""


class ConnectionStringParser:
    """Parses OneCollector connection strings to extract configuration."""

    def __init__(self, connection_string: str):
        if not connection_string:
            raise ValueError("Connection string cannot be empty")

        self.instrumentation_key: str | None = None
        self._parse(connection_string)

        if not self.instrumentation_key:
            raise ValueError("InstrumentationKey not found in connection string")

    def _parse(self, connection_string: str) -> None:
        parts = connection_string.split(";")
        for raw_part in parts:
            part = raw_part.strip()
            if not part or "=" not in part:
                continue

            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "instrumentationkey":
                self.instrumentation_key = value
