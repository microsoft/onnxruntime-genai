# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector Exporter for OpenTelemetry Python.

This package provides an OpenTelemetry exporter that sends telemetry data
to Microsoft OneCollector using the Common Schema JSON format.

The OpenTelemetry-based exporter (OneCollectorLogExporter / TelemetryLogger)
is optional: it is imported only when the ``opentelemetry`` packages are
installed. The transport, serialization, retry, and payload-building helpers
have no OpenTelemetry dependency and are always available, so an application
that ships its own uploader (e.g. a SQLite-backed offline store) can use this
package without installing OpenTelemetry.

Example usage:

    from onecollector_exporter import (
        OneCollectorLogExporter,
        OneCollectorExporterOptions,
        get_telemetry_logger,
    )

    # Option 1: Use with OpenTelemetry SDK directly
    options = OneCollectorExporterOptions(
        connection_string="InstrumentationKey=your-key-here"
    )
    exporter = OneCollectorLogExporter(options=options)

    # Add to logger provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    provider = LoggerProvider()
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    # Option 2: Use the simplified telemetry logger
    logger = get_telemetry_logger(
        connection_string="InstrumentationKey=your-key-here"
    )
    logger.log("MyEvent", {"key": "value"})
    logger.shutdown()
"""

from .callback_manager import CallbackManager, PayloadTransmittedCallbackArgs
from .connection_string_parser import ConnectionStringParser
from .event_source import OneCollectorEventId, OneCollectorEventSource, event_source
from .options import (
    CompressionType,
    OneCollectorExporterOptions,
    OneCollectorExporterValidationError,
    OneCollectorTransportOptions,
)
from .payload_builder import PayloadBuilder
from .retry import RetryHandler
from .serialization import CommonSchemaJsonSerializationHelper
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
    "OneCollectorTransportOptions",
    "PayloadBuilder",
    "PayloadTransmittedCallbackArgs",
    "RetryHandler",
    "event_source",
]

# The OpenTelemetry exporter is optional. Import it only if opentelemetry is
# installed; applications using their own uploader do not need it.
try:
    from .exporter import OneCollectorLogExporter
    from .telemetry_logger import (
        TelemetryLogger,
        get_telemetry_logger,
        log_event,
        shutdown_telemetry,
    )

    __all__ += [
        "OneCollectorLogExporter",
        "TelemetryLogger",
        "get_telemetry_logger",
        "log_event",
        "shutdown_telemetry",
    ]
except ImportError:
    pass
