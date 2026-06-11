# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""HTTP transport implementation for OneCollector exporter."""

import gzip
import zlib
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Optional

import requests

from .event_source import event_source
from .options import CompressionType

if TYPE_CHECKING:
    from .callback_manager import CallbackManager, PayloadTransmittedCallbackArgs


class ITransport(ABC):
    @abstractmethod
    def send(self, payload: bytes, timeout_sec: float, item_count: int = 1) -> tuple[bool, Optional[int]]:
        ...

    @abstractmethod
    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        ...


class HttpJsonPostTransport(ITransport):
    """HTTP JSON POST transport for OneCollector."""

    def __init__(
        self,
        endpoint: str,
        ikey: str,
        compression: CompressionType,
        session: requests.Session,
        callback_manager: Optional["CallbackManager"] = None,
        sdk_version: str = "OTel-python-1.0.0",
    ):
        self.endpoint = endpoint
        self.ikey = ikey
        self.compression = compression
        self.session = session
        self.sdk_version = sdk_version
        self.callback_manager = callback_manager

        self.headers = {
            "x-apikey": ikey,
            "User-Agent": "Python/3 HttpClient",
            "Host": "mobile.events.data.microsoft.com",
            "Content-Type": "application/x-json-stream; charset=utf-8",
            "sdk-version": sdk_version,
            "NoResponseBody": "true",
        }

        if compression != CompressionType.NO_COMPRESSION:
            self.headers["Content-Encoding"] = compression.value

    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        if self.callback_manager is None:
            from .callback_manager import CallbackManager

            self.callback_manager = CallbackManager()

        return self.callback_manager.register(callback, include_failures)

    def send(self, payload: bytes, timeout_sec: float, item_count: int = 1) -> tuple[bool, Optional[int]]:
        payload_size_bytes = len(payload)

        try:
            compressed_payload = self._compress(payload)
            headers = {**self.headers, "Content-Length": str(len(compressed_payload))}

            try:
                response = self.session.post(
                    url=self.endpoint, data=compressed_payload, headers=headers, timeout=timeout_sec
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Retry once on transient transport errors
                response = self.session.post(
                    url=self.endpoint, data=compressed_payload, headers=headers, timeout=timeout_sec
                )

            success = response.ok
            status_code = response.status_code

            if self.callback_manager:
                from .callback_manager import PayloadTransmittedCallbackArgs

                self.callback_manager.notify(
                    PayloadTransmittedCallbackArgs(
                        succeeded=success,
                        status_code=status_code,
                        payload_size_bytes=payload_size_bytes,
                        item_count=item_count,
                        payload_bytes=payload,
                    )
                )

            if success:
                return True, status_code
            else:
                if event_source.is_error_logging_enabled:
                    collector_error = response.headers.get("Collector-Error", "")
                    error_details = response.text[:100] if response.text else ""
                    event_source.http_transport_error_response(
                        "HttpJsonPost", status_code, collector_error, error_details
                    )
                return False, status_code

        except requests.exceptions.Timeout:
            if self.callback_manager:
                from .callback_manager import PayloadTransmittedCallbackArgs

                self.callback_manager.notify(
                    PayloadTransmittedCallbackArgs(
                        succeeded=False,
                        status_code=None,
                        payload_size_bytes=payload_size_bytes,
                        item_count=item_count,
                        payload_bytes=payload,
                    )
                )
            event_source.transport_exception_thrown("HttpJsonPost", Exception("Request timeout"))
            return False, None
        except Exception as ex:
            if self.callback_manager:
                from .callback_manager import PayloadTransmittedCallbackArgs

                self.callback_manager.notify(
                    PayloadTransmittedCallbackArgs(
                        succeeded=False,
                        status_code=None,
                        payload_size_bytes=payload_size_bytes,
                        item_count=item_count,
                        payload_bytes=payload,
                    )
                )
            event_source.transport_exception_thrown("HttpJsonPost", ex)
            return False, None

    def _compress(self, data: bytes) -> bytes:
        if self.compression == CompressionType.DEFLATE:
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            compressed = compressor.compress(data)
            compressed += compressor.flush()
            return compressed
        elif self.compression == CompressionType.GZIP:
            gzip_buffer = BytesIO()
            with gzip.GzipFile(fileobj=gzip_buffer, mode="w") as gzip_file:
                gzip_file.write(data)
            return gzip_buffer.getvalue()
        else:
            return data

    @staticmethod
    def is_retryable(status_code: Optional[int]) -> bool:
        if status_code is None:
            return True
        return status_code in {408, 429, 500, 502, 503, 504}
