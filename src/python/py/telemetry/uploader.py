# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Background uploader that drains the SQLite offline store to OneCollector.

Mirrors the Microsoft 1DS C++ SDK upload loop: reserve a batch of records
(leased so other workers/processes don't double-send), POST them, then delete
on success or release (with retry increment) on failure. Because durability is
provided by the on-disk store, the process can exit at any time without losing
events -- there is no exit-time flush requirement.
"""

import threading
from typing import Optional

import requests

from .library.options import CompressionType, OneCollectorTransportOptions
from .library.payload_builder import PayloadBuilder
from .library.transport import HttpJsonPostTransport
from .offline_store import OfflineStorageSqlite


class EventUploader:
    """Drains the offline store for a single tenant and ships events over HTTP."""

    def __init__(
        self,
        store: OfflineStorageSqlite,
        instrumentation_key: str,
        tenant_token: str,
        endpoint: str = OneCollectorTransportOptions.DEFAULT_ENDPOINT,
        compression: CompressionType = CompressionType.DEFLATE,
        drain_interval_seconds: float = 2.0,
        max_items_per_drain: int = 256,
        lease_time_ms: int = 30_000,
        send_timeout_seconds: float = 10.0,
        idle_backoff_seconds: float = 30.0,
    ):
        self._store = store
        self._tenant_token = tenant_token
        self._drain_interval = drain_interval_seconds
        self._max_items = max_items_per_drain
        self._lease_time_ms = lease_time_ms
        self._send_timeout = send_timeout_seconds
        self._idle_backoff = idle_backoff_seconds

        self._session = requests.Session()
        self._transport = HttpJsonPostTransport(
            endpoint=endpoint,
            ikey=instrumentation_key,
            compression=compression,
            session=self._session,
        )

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----- control -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="genai-telemetry-uploader", daemon=True)
        self._thread.start()

    def request_drain(self) -> None:
        """Nudge the uploader to drain promptly (e.g. after logging an event)."""
        self._wake.set()

    def stop_loop(self, join_timeout_seconds: float = 12.0) -> None:
        """Stop the background loop and wait for any in-flight send to finish.

        Leaves the HTTP session open so a final synchronous flush can still run.
        """
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(join_timeout_seconds)
            self._thread = None

    def close(self) -> None:
        """Release the HTTP session. Call after stop_loop()/flush()."""
        try:
            self._session.close()
        except Exception:
            pass

    def stop(self, timeout_seconds: float = 12.0) -> None:
        """Stop the loop and close the session (convenience)."""
        self.stop_loop(timeout_seconds)
        self.close()

    # ----- draining ------------------------------------------------------

    def drain_once(self) -> tuple[int, int]:
        """Attempt to upload one batch. Returns (sent_count, failed_count)."""
        records = self._store.get_and_reserve_records(
            lease_time_ms=self._lease_time_ms,
            tenant_token=self._tenant_token,
            max_count=self._max_items,
        )
        if not records:
            return (0, 0)

        builder = PayloadBuilder(
            max_size_bytes=OneCollectorTransportOptions.DEFAULT_MAX_PAYLOAD_SIZE_BYTES,
            max_items=OneCollectorTransportOptions.DEFAULT_MAX_ITEMS_PER_PAYLOAD,
        )
        ids = [r.record_id for r in records]
        for r in records:
            if not builder.can_add(r.payload) and not builder.is_empty:
                break
            builder.add(r.payload)
        payload = builder.build()

        try:
            success, _status = self._transport.send(payload, self._send_timeout, item_count=len(ids))
        except Exception:
            success = False

        if success:
            self._store.delete_records(ids)
            return (len(ids), 0)
        self._store.release_records(ids, increment_retry=True)
        return (0, len(ids))

    def flush(self, max_seconds: float = 5.0) -> None:
        """Best-effort drain of all pending events, bounded by max_seconds."""
        import time

        deadline = time.time() + max_seconds
        while time.time() < deadline:
            sent, failed = self.drain_once()
            if sent == 0 and failed == 0:
                return  # nothing left
            if failed:
                return  # network down; leave the rest on disk for next run

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sent, failed = self.drain_once()
                while sent > 0 and not self._stop.is_set():
                    sent, failed = self.drain_once()
            except Exception:
                failed = 1

            # Wait until nudged, the poll interval elapses, or (when offline)
            # a longer backoff elapses.
            wait = self._idle_backoff if failed else self._drain_interval
            self._wake.wait(wait)
            self._wake.clear()
