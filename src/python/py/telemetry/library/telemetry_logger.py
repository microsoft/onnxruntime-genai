# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""High-level telemetry logger facade for easy usage."""

import logging
import threading
import uuid
from typing import Any, Callable, Optional

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from .exporter import OneCollectorLogExporter
from .options import OneCollectorExporterOptions
# The native onnxruntime_genai version is not importable here without
# loading the extension module, so the GenAI telemetry layer injects it via
# set_app_version() before the first event is logged.
_APP_VERSION = "unknown"


def set_app_version(version: str) -> None:
    global _APP_VERSION
    _APP_VERSION = version


class TelemetryLogger:
    """Singleton telemetry logger for simplified OneCollector integration.

    Provides a simple interface for logging telemetry events without
    needing to configure OpenTelemetry directly.
    """

    _instance: Optional["TelemetryLogger"] = None
    _default_logger: Optional["TelemetryLogger"] = None
    _instance_lock = threading.RLock()
    _default_logger_lock = threading.RLock()
    _logger: Optional[logging.Logger] = None
    _logger_exporter: Optional[OneCollectorLogExporter] = None
    _logger_provider: Optional[LoggerProvider] = None

    def __new__(cls, options: Optional[OneCollectorExporterOptions] = None, shutdown_on_exit: bool = True):
        """Create or return the singleton instance.

        Args:
            options: Exporter options (only used on first instantiation)
            shutdown_on_exit: Whether OpenTelemetry should register an atexit
                flush (only used on first instantiation). Set False when the
                caller manages flushing itself to avoid an unbounded blocking
                flush against a slow/unreachable collector at process exit.

        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    # Publish only after initialization completes so a concurrent
                    # lock-free reader never observes a half-initialized instance.
                    instance = super().__new__(cls)
                    instance._initialize(options, shutdown_on_exit)
                    cls._instance = instance

        return cls._instance

    def _initialize(
        self, options: Optional[OneCollectorExporterOptions], shutdown_on_exit: bool = True
    ) -> None:
        """Initialize the logger (called only once).

        Args:
            options: Exporter configuration options
            shutdown_on_exit: Whether OpenTelemetry registers an atexit flush

        """
        try:
            # Create exporter
            self._logger_exporter = OneCollectorLogExporter(options=options)

            # Create logger provider
            service_name = (
                options.service_name if options and options.service_name else __name__.split(".", maxsplit=1)[0]
            )
            self._logger_provider = LoggerProvider(
                shutdown_on_exit=shutdown_on_exit,
                resource=Resource.create(
                    {
                        "service.name": service_name,
                        "service.version": _APP_VERSION,
                        "service.instance.id": str(uuid.uuid4()),  # Unique instance ID; can double as session ID
                    }
                )
            )

            # Set as global logger provider
            set_logger_provider(self._logger_provider)

            # Add batch processor
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    self._logger_exporter,
                    schedule_delay_millis=1000,
                )
            )

            # Create logging handler
            handler = LoggingHandler(level=logging.INFO, logger_provider=self._logger_provider)

            # Set up Python logger
            logger = logging.getLogger(__name__)
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            self._logger = logger

        except Exception:
            # Silently fail initialization - logger will be None
            self._logger = None
            self._logger_provider = None
            self._logger_exporter = None

    def add_global_metadata(self, metadata: dict[str, Any]) -> None:
        """Add metadata fields to all telemetry events.

        Args:
            metadata: Dictionary of metadata to add

        """
        if self._logger_exporter:
            self._logger_exporter.add_metadata(metadata)

    def register_payload_transmitted_callback(
        self, callback, include_failures: bool = False
    ) -> Optional[Callable[[], None]]:
        """Register a callback for payload transmission events."""
        if self._logger_exporter:
            return self._logger_exporter.register_payload_transmitted_callback(callback, include_failures)
        return None

    def log(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Log a telemetry event.

        Args:
            event_name: Name of the event
            attributes: Optional event attributes

        """
        if self._logger:
            extra = attributes if attributes else {}
            self._logger.info(event_name, extra=extra)

    def disable_telemetry(self) -> None:
        """Disable telemetry logging."""
        if self._logger:
            self._logger.disabled = True

    def enable_telemetry(self) -> None:
        """Enable telemetry logging."""
        if self._logger:
            self._logger.disabled = False

    def shutdown(self) -> None:
        """Shutdown the telemetry logger and flush pending data."""
        if self._logger_provider:
            self._logger_provider.shutdown()

    @classmethod
    def get_default_logger(
        cls,
        connection_string: Optional[str] = None,
        service_name: Optional[str] = None,
        shutdown_on_exit: bool = True,
    ) -> "TelemetryLogger":
        """Get or create the default telemetry logger.

        Args:
            connection_string: OneCollector connection string (only used on first call)
            service_name: Logical application/service name for emitted telemetry (only used on first call)
            shutdown_on_exit: Whether OpenTelemetry registers an atexit flush (only used on first call)

        Returns:
            TelemetryLogger instance

        """
        if cls._default_logger is None:
            with cls._default_logger_lock:
                if cls._default_logger is None:
                    options = None
                    if connection_string:
                        options = OneCollectorExporterOptions(
                            connection_string=connection_string, service_name=service_name
                        )
                    cls._default_logger = cls(options=options, shutdown_on_exit=shutdown_on_exit)

        return cls._default_logger

    @classmethod
    def shutdown_default_logger(cls) -> None:
        """Shutdown the default telemetry logger."""
        with cls._default_logger_lock:
            if cls._default_logger:
                cls._default_logger.shutdown()
                cls._default_logger = None


def get_telemetry_logger(
    connection_string: Optional[str] = None,
    service_name: Optional[str] = None,
    shutdown_on_exit: bool = True,
) -> TelemetryLogger:
    """Get or create the default telemetry logger.

    Args:
        connection_string: OneCollector connection string (only used on first call)
        service_name: Logical application/service name for emitted telemetry (only used on first call)
        shutdown_on_exit: Whether OpenTelemetry registers an atexit flush (only used on first call)

    Returns:
        TelemetryLogger instance

    """
    return TelemetryLogger.get_default_logger(
        connection_string=connection_string, service_name=service_name, shutdown_on_exit=shutdown_on_exit
    )


def log_event(event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Log a telemetry event using the default logger.

    Args:
        event_name: Name of the event
        attributes: Optional event attributes

    """
    logger = get_telemetry_logger()
    logger.log(event_name, attributes)


def shutdown_telemetry() -> None:
    """Shutdown the default telemetry logger."""
    TelemetryLogger.shutdown_default_logger()
