# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Callback manager for payload transmission events."""

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .event_source import event_source


@dataclass
class PayloadTransmittedCallbackArgs:
    """Arguments passed to payload transmitted callbacks."""

    succeeded: bool
    status_code: Optional[int]
    payload_size_bytes: int
    item_count: int
    payload_bytes: Optional[bytes] = None


class CallbackManager:
    """Manages callbacks for payload transmission events."""

    def __init__(self):
        self._callbacks: list[tuple[Callable[[PayloadTransmittedCallbackArgs], None], bool]] = []
        self._lock = threading.Lock()
        self._closed = False

    def register(
        self, callback: Callable[[PayloadTransmittedCallbackArgs], None], include_failures: bool = False
    ) -> Callable[[], None]:
        with self._lock:
            if self._closed:
                return lambda: None
            entry = (callback, include_failures)
            self._callbacks.append(entry)

        def unregister():
            with self._lock:
                try:
                    self._callbacks.remove(entry)
                except ValueError:
                    pass

        return unregister

    def notify(self, args: PayloadTransmittedCallbackArgs) -> None:
        with self._lock:
            if self._closed:
                return
            callbacks_snapshot = self._callbacks.copy()

        for callback, include_failures in callbacks_snapshot:
            if not args.succeeded and not include_failures:
                continue
            try:
                callback(args)
            except Exception as ex:
                event_source.exception_thrown_from_user_code("PayloadTransmittedCallback", ex)

    def close(self) -> None:
        with self._lock:
            self._callbacks.clear()
            self._closed = True
