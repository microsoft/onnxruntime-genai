# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Background uploader that drains the SQLite offline store to OneCollector."""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import NamedTuple

from .library.options import CompressionType, OneCollectorTransportOptions
from .library.payload_builder import PayloadBuilder
from .library.transport import HttpJsonPostTransport
from .offline_store import OfflineEventStore
from .process_lock import ProcessDrainLock


class DrainOutcome(Enum):
    EMPTY = "empty"
    PROGRESS = "progress"
    SPLIT = "split"
    ACKNOWLEDGE_RETRY = "acknowledge_retry"
    DELETE_RETRY = "delete_retry"
    STORAGE_RETRY = "storage_retry"
    TRANSPORT_RETRY = "transport_retry"


class DrainResult(NamedTuple):
    delivered: int
    left: int
    outcome: DrainOutcome


class EventUploader:
    """Drains the shared offline store from one process at a time."""

    def __init__(
        self,
        store: OfflineEventStore,
        instrumentation_key: str,
        endpoint: str = OneCollectorTransportOptions.DEFAULT_ENDPOINT,
        compression: CompressionType = CompressionType.DEFLATE,
        drain_interval_seconds: float = 2.0,
        max_items_per_drain: int = 256,
        send_timeout_seconds: float = 10.0,
        idle_backoff_seconds: float = 30.0,
    ):
        self._store = store
        self._drain_interval = drain_interval_seconds
        self._max_items = max_items_per_drain
        self._send_timeout = send_timeout_seconds
        self._idle_backoff = idle_backoff_seconds
        self._drain_lock = ProcessDrainLock(store.db_path + ".lock")
        self._transport = HttpJsonPostTransport(
            endpoint=endpoint,
            ikey=instrumentation_key,
            compression=compression,
        )
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._retain_rows = threading.Event()
        self._mutation_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending_ack_ids: list[int] = []
        self._pending_delete_ids: list[int] = []
        self._split_batch_size: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="genai-telemetry-uploader", daemon=True)
        self._thread.start()

    def request_drain(self) -> None:
        if self._drain_lock.held:
            self._wake.set()

    def stop_loop(self, join_timeout_seconds: float = 5.0) -> bool:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(join_timeout_seconds)
        stopped = not thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def signal_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def retain_queued_rows(self) -> None:
        """Stop new draining and prevent any later queue mutation in this process."""
        with self._mutation_lock:
            self._retain_rows.set()
        self.signal_stop()

    def close(self) -> None:
        self._drain_lock.release()

    def stop(self, timeout_seconds: float = 12.0) -> None:
        if self.stop_loop(timeout_seconds):
            self.close()

    def _delete_unless_retained(self, ids: list[int], deadline: float | None = None) -> bool | None:
        with self._mutation_lock:
            if self._retain_rows.is_set():
                return None
            return self._store.delete(ids, deadline)

    def _finish_handled_rows(self, ids: list[int], deadline: float | None = None) -> DrainResult:
        self._pending_ack_ids = ids
        with self._mutation_lock:
            if self._retain_rows.is_set():
                return DrainResult(0, len(ids), DrainOutcome.TRANSPORT_RETRY)
            if not self._store.acknowledge(ids, deadline):
                return DrainResult(0, len(ids), DrainOutcome.ACKNOWLEDGE_RETRY)
            self._pending_ack_ids = []
            self._pending_delete_ids = ids
            deleted = self._store.delete(ids, deadline)
        if not deleted:
            return DrainResult(0, len(ids), DrainOutcome.DELETE_RETRY)
        self._pending_delete_ids = []
        return DrainResult(len(ids), 0, DrainOutcome.PROGRESS)

    def drain_once(self, deadline: float | None = None) -> DrainResult:
        if self._retain_rows.is_set():
            return DrainResult(
                0,
                len(self._pending_ack_ids) + len(self._pending_delete_ids),
                DrainOutcome.TRANSPORT_RETRY,
            )
        if self._pending_ack_ids:
            return self._finish_handled_rows(self._pending_ack_ids, deadline)
        if self._pending_delete_ids:
            pending_ids = self._pending_delete_ids
            deleted = self._delete_unless_retained(pending_ids, deadline)
            if deleted is None:
                return DrainResult(0, len(pending_ids), DrainOutcome.TRANSPORT_RETRY)
            if not deleted:
                return DrainResult(0, len(pending_ids), DrainOutcome.DELETE_RETRY)
            self._pending_delete_ids = []
            return DrainResult(len(pending_ids), 0, DrainOutcome.PROGRESS)

        batch_limit = min(self._max_items, self._split_batch_size or self._max_items)
        acknowledged_ids = self._store.get_acknowledged_ids(batch_limit, deadline)
        if acknowledged_ids is None:
            return DrainResult(0, 0, DrainOutcome.STORAGE_RETRY)
        if acknowledged_ids:
            self._pending_delete_ids = acknowledged_ids
            deleted = self._delete_unless_retained(acknowledged_ids, deadline)
            if deleted is None:
                return DrainResult(0, len(acknowledged_ids), DrainOutcome.TRANSPORT_RETRY)
            if not deleted:
                return DrainResult(0, len(acknowledged_ids), DrainOutcome.DELETE_RETRY)
            self._pending_delete_ids = []
            return DrainResult(len(acknowledged_ids), 0, DrainOutcome.PROGRESS)

        batch = self._store.get_batch_for_upload(batch_limit, deadline)
        if batch is None:
            return DrainResult(0, 0, DrainOutcome.STORAGE_RETRY)
        if not batch:
            return DrainResult(0, 0, DrainOutcome.EMPTY)

        builder = PayloadBuilder(
            max_size_bytes=OneCollectorTransportOptions.DEFAULT_MAX_PAYLOAD_SIZE_BYTES,
            max_items=OneCollectorTransportOptions.DEFAULT_MAX_ITEMS_PER_PAYLOAD,
        )
        included: list[int] = []
        for row_id, payload in batch:
            if not builder.can_add(payload):
                if builder.is_empty:
                    return self._finish_handled_rows([row_id], deadline)
                break
            builder.add(payload)
            included.append(row_id)
        payload_bytes = builder.build()

        timeout = self._send_timeout
        if deadline is not None:
            timeout = min(timeout, max(0.0, deadline - time.monotonic()))
            if timeout <= 0.0:
                return DrainResult(0, len(included), DrainOutcome.TRANSPORT_RETRY)

        admission_released = False

        def release_admission() -> None:
            nonlocal admission_released
            if not admission_released:
                admission_released = True
                self._mutation_lock.release()

        self._mutation_lock.acquire()
        try:
            if self._retain_rows.is_set():
                return DrainResult(0, len(included), DrainOutcome.TRANSPORT_RETRY)
            success, status = self._transport.send(
                payload_bytes,
                timeout,
                item_count=len(included),
                on_send_admitted=release_admission,
            )
        except Exception:
            success, status = (False, None)
        finally:
            release_admission()

        if success:
            self._split_batch_size = None
            return self._finish_handled_rows(included, deadline)
        if not HttpJsonPostTransport.is_retryable(status):
            if status in {400, 413, 422} and len(included) > 1:
                self._split_batch_size = max(1, len(included) // 2)
                return DrainResult(0, len(included), DrainOutcome.SPLIT)
            self._split_batch_size = None
            return self._finish_handled_rows(included, deadline)
        return DrainResult(0, len(included), DrainOutcome.TRANSPORT_RETRY)

    def flush(self, max_seconds: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._drain_lock.acquire():
            return
        try:
            deadline = time.monotonic() + max(0.0, max_seconds)
            delete_retries = 0
            while time.monotonic() < deadline:
                result = self.drain_once(deadline)
                if result.outcome is DrainOutcome.EMPTY:
                    return
                if result.outcome in {DrainOutcome.STORAGE_RETRY, DrainOutcome.TRANSPORT_RETRY}:
                    return
                if result.outcome in {DrainOutcome.ACKNOWLEDGE_RETRY, DrainOutcome.DELETE_RETRY}:
                    delete_retries += 1
                    if delete_retries >= 2:
                        return
                else:
                    delete_retries = 0
        finally:
            self._drain_lock.release()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._wake.clear()
                transient_failure = False
                if self._drain_lock.acquire():
                    try:
                        delete_retries = 0
                        while not self._stop.is_set():
                            result = self.drain_once()
                            if result.outcome is DrainOutcome.EMPTY:
                                break
                            if result.outcome in {DrainOutcome.STORAGE_RETRY, DrainOutcome.TRANSPORT_RETRY}:
                                transient_failure = True
                                break
                            if result.outcome in {DrainOutcome.ACKNOWLEDGE_RETRY, DrainOutcome.DELETE_RETRY}:
                                delete_retries += 1
                                if delete_retries >= 2:
                                    transient_failure = True
                                    break
                            else:
                                delete_retries = 0
                    except Exception:
                        transient_failure = True
                else:
                    transient_failure = True
                wait = self._idle_backoff if transient_failure else self._drain_interval
                self._wake.wait(wait)
        finally:
            self._drain_lock.release()
