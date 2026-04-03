# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""EventSource-style logging for OneCollector exporter diagnostics."""

import logging
from enum import IntEnum


class OneCollectorEventId(IntEnum):
    """Event IDs matching .NET EventSource implementation."""

    EXPORT_EXCEPTION = 1
    TRANSPORT_DATA_SENT = 2
    SINK_DATA_WRITTEN = 3
    DATA_DROPPED = 4
    TRANSPORT_EXCEPTION = 5
    HTTP_ERROR_RESPONSE = 6
    EVENT_FULL_NAME_DISCARDED = 7
    EVENT_NAMESPACE_INVALID = 8
    EVENT_NAME_INVALID = 9
    USER_CODE_EXCEPTION = 10
    ATTRIBUTE_DROPPED = 11


class OneCollectorEventSource:
    """EventSource for OneCollector exporter diagnostics."""

    def __init__(self):
        self.logger = logging.getLogger("OpenTelemetry.Exporter.OneCollector")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @property
    def is_informational_logging_enabled(self) -> bool:
        return self.logger.isEnabledFor(logging.INFO)

    @property
    def is_warning_logging_enabled(self) -> bool:
        return self.logger.isEnabledFor(logging.WARNING)

    @property
    def is_error_logging_enabled(self) -> bool:
        return self.logger.isEnabledFor(logging.ERROR)

    def export_exception_thrown(self, item_type: str, exception: Exception) -> None:
        if self.is_error_logging_enabled:
            self.logger.error(
                "Exception thrown exporting '%s' batch: %s",
                item_type,
                exception,
                exc_info=exception,
                extra={"event_id": OneCollectorEventId.EXPORT_EXCEPTION},
            )

    def transport_data_sent(self, item_type: str, num_records: int, transport_description: str) -> None:
        if self.is_informational_logging_enabled:
            self.logger.info(
                "Sent '%s' batch of %s item(s) to '%s' transport",
                item_type,
                num_records,
                transport_description,
                extra={"event_id": OneCollectorEventId.TRANSPORT_DATA_SENT},
            )

    def sink_data_written(self, item_type: str, num_records: int, sink_description: str) -> None:
        if self.is_informational_logging_enabled:
            self.logger.info(
                "Wrote '%s' batch of %s item(s) to '%s' sink",
                item_type,
                num_records,
                sink_description,
                extra={"event_id": OneCollectorEventId.SINK_DATA_WRITTEN},
            )

    def data_dropped(
        self, item_type: str, num_records: int, during_serialization: int, during_transmission: int
    ) -> None:
        if self.is_warning_logging_enabled:
            self.logger.warning(
                "Dropped %s '%s' item(s). %s during serialization. %s due to transmission failure",
                num_records,
                item_type,
                during_serialization,
                during_transmission,
                extra={"event_id": OneCollectorEventId.DATA_DROPPED},
            )

    def transport_exception_thrown(self, transport_type: str, exception: Exception) -> None:
        if self.is_error_logging_enabled:
            self.logger.error(
                "Exception thrown by '%s' transport: %s",
                transport_type,
                exception,
                exc_info=exception,
                extra={"event_id": OneCollectorEventId.TRANSPORT_EXCEPTION},
            )

    def http_transport_error_response(
        self, transport_type: str, status_code: int, error_message: str, error_details: str
    ) -> None:
        if self.is_error_logging_enabled:
            self.logger.error(
                "Error response received by '%s' transport. StatusCode: %s, ErrorMessage: '%s', ErrorDetails: '%s'",
                transport_type,
                status_code,
                error_message,
                error_details,
                extra={"event_id": OneCollectorEventId.HTTP_ERROR_RESPONSE},
            )

    def event_full_name_discarded(self, event_namespace: str, event_name: str) -> None:
        if self.is_warning_logging_enabled:
            self.logger.warning(
                "Event full name discarded. EventNamespace: '%s', EventName: '%s'",
                event_namespace,
                event_name,
                extra={"event_id": OneCollectorEventId.EVENT_FULL_NAME_DISCARDED},
            )

    def event_namespace_invalid(self, event_namespace: str) -> None:
        if self.is_warning_logging_enabled:
            self.logger.warning(
                "Event namespace invalid. EventNamespace: '%s'",
                event_namespace,
                extra={"event_id": OneCollectorEventId.EVENT_NAMESPACE_INVALID},
            )

    def event_name_invalid(self, event_name: str) -> None:
        if self.is_warning_logging_enabled:
            self.logger.warning(
                "Event name invalid. EventName: '%s'",
                event_name,
                extra={"event_id": OneCollectorEventId.EVENT_NAME_INVALID},
            )

    def exception_thrown_from_user_code(self, user_code_type: str, exception: Exception) -> None:
        if self.is_error_logging_enabled:
            self.logger.error(
                "Exception thrown by '%s' user code: %s",
                user_code_type,
                exception,
                exc_info=exception,
                extra={"event_id": OneCollectorEventId.USER_CODE_EXCEPTION},
            )

    def attribute_dropped(self, item_type: str, attribute_name: str, reason: str) -> None:
        if self.is_warning_logging_enabled:
            self.logger.warning(
                "Dropped %s attribute '%s': %s",
                item_type,
                attribute_name,
                reason,
                extra={"event_id": OneCollectorEventId.ATTRIBUTE_DROPPED},
            )

    def disable(self) -> None:
        self.logger.disabled = True


# Global event source instance
event_source = OneCollectorEventSource()
