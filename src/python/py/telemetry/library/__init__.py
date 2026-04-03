# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector Exporter for OpenTelemetry Python.

Vendored from Olive (olive/telemetry/library/) and adapted for
ONNX Runtime GenAI telemetry integration.
"""

from .callback_manager import CallbackManager, PayloadTransmittedCallbackArgs
from .connection_string_parser import ConnectionStringParser
from .event_source import OneCollectorEventId, OneCollectorEventSource, event_source
from .exporter import OneCollectorLogExporter
from .options import (
    CompressionType,
    OneCollectorExporterOptions,
    OneCollectorExporterValidationError,
    OneCollectorTransportOptions,
)
from .payload_builder import PayloadBuilder
from .retry import RetryHandler
from .serialization import CommonSchemaJsonSerializationHelper
from .telemetry_logger import (
    TelemetryLogger,
    get_telemetry_logger,
    log_event,
    shutdown_telemetry,
)
from .transport import HttpJsonPostTransport, ITransport

__version__ = "0.0.1"

__all__ = [
    "CallbackManager",
    "CommonSchemaJsonSerializationHelper",
    "CompressionType",
    "ConnectionStringParser",
    "HttpJsonPostTransport",
    "ITransport",
    "OneCollectorEventId",
    "OneCollectorEventSource",
    "OneCollectorExporterOptions",
    "OneCollectorExporterValidationError",
    "OneCollectorLogExporter",
    "OneCollectorTransportOptions",
    "PayloadBuilder",
    "PayloadTransmittedCallbackArgs",
    "RetryHandler",
    "TelemetryLogger",
    "event_source",
    "get_telemetry_logger",
    "log_event",
    "shutdown_telemetry",
]
