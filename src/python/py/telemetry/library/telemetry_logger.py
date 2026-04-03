# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""High-level telemetry logger facade."""

import logging
import uuid
from typing import Any, Callable, Optional

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from .exporter import OneCollectorLogExporter
from .options import OneCollectorExporterOptions

# Version will be set from the main telemetry module
_APP_VERSION = "unknown"


def set_app_version(version: str) -> None:
    global _APP_VERSION
    _APP_VERSION = version


class TelemetryLogger:
    """Singleton telemetry logger for simplified OneCollector integration."""

    _instance: Optional["TelemetryLogger"] = None
    _default_logger: Optional["TelemetryLogger"] = None
    _logger: Optional[logging.Logger] = None
    _logger_exporter: Optional[OneCollectorLogExporter] = None
    _logger_provider: Optional[LoggerProvider] = None

    def __new__(cls, options: Optional[OneCollectorExporterOptions] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(options)
        return cls._instance

    def _initialize(self, options: Optional[OneCollectorExporterOptions]) -> None:
        try:
            self._logger_exporter = OneCollectorLogExporter(options=options)

            self._logger_provider = LoggerProvider(
                resource=Resource.create(
                    {
                        "service.name": "onnxruntime-genai",
                        "service.version": _APP_VERSION,
                        "service.instance.id": str(uuid.uuid4()),
                    }
                )
            )

            set_logger_provider(self._logger_provider)

            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    self._logger_exporter,
                    schedule_delay_millis=1000,
                )
            )

            handler = LoggingHandler(level=logging.INFO, logger_provider=self._logger_provider)

            logger = logging.getLogger("onnxruntime_genai.telemetry")
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            self._logger = logger

        except Exception:
            self._logger = None
            self._logger_provider = None
            self._logger_exporter = None

    def add_global_metadata(self, metadata: dict[str, Any]) -> None:
        if self._logger_exporter:
            self._logger_exporter.add_metadata(metadata)

    def register_payload_transmitted_callback(
        self, callback, include_failures: bool = False
    ) -> Optional[Callable[[], None]]:
        if self._logger_exporter:
            return self._logger_exporter.register_payload_transmitted_callback(callback, include_failures)
        return None

    def log(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        if self._logger:
            extra = attributes if attributes else {}
            self._logger.info(event_name, extra=extra)

    def disable_telemetry(self) -> None:
        if self._logger:
            self._logger.disabled = True

    def enable_telemetry(self) -> None:
        if self._logger:
            self._logger.disabled = False

    def shutdown(self) -> None:
        if self._logger_provider:
            self._logger_provider.shutdown()

    @classmethod
    def get_default_logger(cls, connection_string: Optional[str] = None) -> "TelemetryLogger":
        if cls._default_logger is None:
            options = None
            if connection_string:
                options = OneCollectorExporterOptions(connection_string=connection_string)
            cls._default_logger = cls(options=options)
        return cls._default_logger

    @classmethod
    def shutdown_default_logger(cls) -> None:
        if cls._default_logger:
            cls._default_logger.shutdown()
            cls._default_logger = None


def get_telemetry_logger(connection_string: Optional[str] = None) -> TelemetryLogger:
    return TelemetryLogger.get_default_logger(connection_string=connection_string)


def log_event(event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    logger = get_telemetry_logger()
    logger.log(event_name, attributes)


def shutdown_telemetry() -> None:
    TelemetryLogger.shutdown_default_logger()
