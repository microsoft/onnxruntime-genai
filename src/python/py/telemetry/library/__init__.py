# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector building blocks (standard library only).

Helpers for serializing telemetry to Common Schema JSON and posting it to the
Microsoft OneCollector endpoint. These modules have no third-party dependency
and are driven directly by the SQLite-backed uploader.
"""

from .connection_string_parser import ConnectionStringParser
from .options import (
    CompressionType,
    OneCollectorExporterOptions,
    OneCollectorExporterValidationError,
    OneCollectorTransportOptions,
)
from .payload_builder import PayloadBuilder
from .serialization import CommonSchemaJsonSerializationHelper
from .transport import HttpJsonPostTransport

__all__ = [
    "CommonSchemaJsonSerializationHelper",
    "CompressionType",
    "ConnectionStringParser",
    "HttpJsonPostTransport",
    "OneCollectorExporterOptions",
    "OneCollectorExporterValidationError",
    "OneCollectorTransportOptions",
    "PayloadBuilder",
]
