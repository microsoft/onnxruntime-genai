# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Main OneCollector log exporter implementation."""

import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from time import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import requests
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogExportResult, LogRecordExporter
from opentelemetry.sdk.resources import Resource

from .callback_manager import CallbackManager
from .event_source import event_source
from .options import OneCollectorExporterOptions
from .payload_builder import PayloadBuilder
from .retry import RetryHandler
from .serialization import CommonSchemaJsonSerializationHelper
from .transport import HttpJsonPostTransport

if TYPE_CHECKING:
    from .callback_manager import PayloadTransmittedCallbackArgs


class OneCollectorLogExporter(LogRecordExporter):
    """OpenTelemetry log exporter for Microsoft OneCollector."""

    def __init__(
        self,
        options: Optional[OneCollectorExporterOptions] = None,
        excluded_attributes: Optional[set[str]] = None,
    ):
        if options is None:
            raise ValueError("OneCollectorExporterOptions is required")
        options.validate()

        self._options = options
        self._shutdown_lock = threading.Lock()
        self._shutdown = False
        self._shutdown_event = threading.Event()
        if excluded_attributes is None:
            self._excluded_attributes = {
                "code.filepath",
                "code.function",
                "code.lineno",
                "code.file.path",
                "code.function.name",
                "code.line.number",
            }
        else:
            self._excluded_attributes = set(excluded_attributes)

        transport_opts = options.transport_options

        if transport_opts.http_client_factory:
            self._session = transport_opts.http_client_factory()
            self._owns_session = False
        else:
            self._session = requests.Session()
            self._owns_session = True

        try:
            self._ikey = f"{CommonSchemaJsonSerializationHelper.ONE_COLLECTOR_TENANCY_SYMBOL}:{options.tenant_token}"

            self._callback_manager = CallbackManager()

            self._transport = HttpJsonPostTransport(
                endpoint=transport_opts.endpoint,
                ikey=options.instrumentation_key,
                compression=transport_opts.compression,
                session=self._session,
                callback_manager=self._callback_manager,
            )

            self._payload_builder = PayloadBuilder(
                max_size_bytes=transport_opts.max_payload_size_bytes, max_items=transport_opts.max_items_per_payload
            )

            self._retry_handler = RetryHandler(max_retries=3)

            self._metadata: dict[str, Any] = {}
            self._resource: Optional[Resource] = None
        except Exception:
            if self._owns_session:
                self._session.close()
            raise

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        self._metadata.update(metadata)

    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        return self._transport.register_payload_transmitted_callback(callback, include_failures)

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogExportResult:
        if self._shutdown:
            return LogExportResult.FAILURE

        try:
            if self._resource is None:
                first_item = batch[0] if batch else None
                resource = getattr(first_item, "resource", None)
                if resource is None and first_item is not None:
                    resource = getattr(first_item.log_record, "resource", None)
                self._resource = resource or Resource.create()

            serialized_items = []
            for log_data in batch:
                try:
                    item_bytes = self._serialize_log_data(log_data)
                    serialized_items.append(item_bytes)
                except Exception as ex:
                    event_source.export_exception_thrown("ReadableLogRecord", ex)

            if not serialized_items:
                return LogExportResult.FAILURE

            payloads = self._build_payloads(serialized_items)

            deadline_sec = time() + self._options.transport_options.timeout_seconds

            for payload in payloads:
                item_count = payload.count(b"\n") + 1 if payload else 0
                success = self._retry_handler.execute_with_retry(
                    operation=lambda payload=payload, item_count=item_count: self._transport.send(
                            payload, max(0.1, deadline_sec - time()), item_count=item_count
                    ),
                    deadline_sec=deadline_sec,
                    shutdown_event=self._shutdown_event,
                )

                if not success:
                    return LogExportResult.FAILURE

                if self._shutdown:
                    return LogExportResult.FAILURE

            event_source.sink_data_written("ReadableLogRecord", len(batch), "OneCollector")

            return LogExportResult.SUCCESS

        except Exception as ex:
            event_source.export_exception_thrown("ReadableLogRecord", ex)
            return LogExportResult.FAILURE

    def _serialize_log_data(self, log_data: ReadableLogRecord) -> bytes:
        log_record = log_data.log_record

        data = {}

        if self._resource and self._resource.attributes:
            for key, value in self._resource.attributes.items():
                if key == "service.name" and "app_name" not in data:
                    data["app_name"] = value
                elif key == "service.version" and "app_version" not in data:
                    data["app_version"] = value
                elif key == "service.instance.id" and "app_instance_id" not in data:
                    data["app_instance_id"] = value
                else:
                    data[key] = value

        if log_record.attributes:
            data.update(
                {key: value for key, value in log_record.attributes.items() if key not in self._excluded_attributes}
            )

        data.update(self._metadata)

        if log_record.timestamp:
            timestamp = datetime.fromtimestamp(log_record.timestamp / 1e9, tz=timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        event_name = str(log_record.body) if log_record.body else "UnnamedEvent"

        envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
            event_name=event_name, timestamp=timestamp, ikey=self._ikey, data=data
        )

        return CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)

    def _build_payloads(self, serialized_items: list[bytes]) -> list[bytes]:
        payloads = []
        self._payload_builder.reset()

        for item_bytes in serialized_items:
            if not self._payload_builder.can_add(item_bytes) and not self._payload_builder.is_empty:
                payloads.append(self._payload_builder.build())
                self._payload_builder.reset()

            self._payload_builder.add(item_bytes)

        if not self._payload_builder.is_empty:
            payloads.append(self._payload_builder.build())

        return payloads

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_event.set()

        # Close HTTP session (only if we own it)
        if hasattr(self, "_session") and getattr(self, "_owns_session", True):
            self._session.close()
        if hasattr(self, "_callback_manager"):
            self._callback_manager.close()
