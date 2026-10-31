# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SQLite-backed durable queue for telemetry events.

A deliberately small subset of the Microsoft 1DS C++ SDK offline store
(cpp_client_telemetry/lib/offline/OfflineStorage_SQLite.cpp): a single FIFO
table of serialized event payloads. An uploader drains it, deleting rows on
success, dropping them on a permanent (non-retryable) send result, and leaving
them for the next attempt on a transient failure. Because every event is
written to disk before any network call, the process can exit at any time
without losing data and without an exit-time flush.

Uses only the Python standard library (``sqlite3``), so it adds no dependency.

Intentionally omitted from the full 1DS store (not needed for low-volume CLI
telemetry): per-event priority (``latency``), persistence classes, general
reservation/leasing, per-row retry counters, tenant multiplexing, and the
``settings`` table.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager, suppress

from .storage_paths import StorageDirectory

SCHEMA_VERSION = 3


class OfflineEventStore:
    """Durable FIFO queue of serialized telemetry event payloads.

    All methods are best-effort and swallow storage errors: telemetry must
    never crash the host application. Thread-safe via a per-instance lock;
    tolerant of concurrent processes via WAL mode + ``busy_timeout``.
    """

    def __init__(self, db_path: str, max_records: int = 2048, busy_timeout_ms: int = 3000):
        self._db_path = db_path
        self._max_records = max_records
        # When full, trim back to this watermark so we don't trim on every insert.
        self._trim_target = max(1, (max_records * 3) // 4)
        self._busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._directory: StorageDirectory | None = None
        self._db_fd: int | None = None
        self._initialize()

    def _initialize(self) -> None:
        conn = None
        try:
            self._directory = StorageDirectory(os.path.dirname(self._db_path) or ".", create=True)
            self._directory.__enter__()
            name = os.path.basename(self._db_path)
            suffixes = ("", "-wal", "-shm", "-journal")
            for suffix in suffixes:
                self._directory.validate_file(name + suffix)
            self._db_fd, identity = self._directory.prepare_sqlite_file(name)
            conn = sqlite3.connect(
                (self._directory.path / name).as_uri() + "?mode=rw",
                timeout=self._busy_timeout_ms / 1000.0,
                check_same_thread=False,
                uri=True,
            )
            self._directory.verify_file_identity(name, identity)
            for suffix in suffixes:
                self._directory.validate_file(name + suffix)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL, "
                "available_at REAL NOT NULL DEFAULT 0, acknowledged INTEGER NOT NULL DEFAULT 0)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            if "available_at" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN available_at REAL NOT NULL DEFAULT 0")
            if "acknowledged" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN acknowledged INTEGER NOT NULL DEFAULT 0")
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.commit()
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            self._conn = conn
            self._harden_permissions()
        except Exception:
            logging.getLogger(__name__).debug("Telemetry storage initialization failed", exc_info=True)
            if conn is not None:
                with suppress(Exception):
                    conn.close()
            self._conn = None
            self._close_storage_handles()

    def _close_storage_handles(self) -> None:
        if self._db_fd is not None:
            os.close(self._db_fd)
            self._db_fd = None
        if self._directory is not None:
            self._directory.__exit__(None, None, None)
            self._directory = None

    @contextmanager
    def _bounded_lock(self, deadline: float | None):
        acquired = (
            self._lock.acquire()
            if deadline is None
            else self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        )
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    @contextmanager
    def _bounded_busy_timeout(self, deadline: float | None):
        if deadline is None or self._conn is None:
            yield
            return
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        self._conn.execute(f"PRAGMA busy_timeout={min(self._busy_timeout_ms, remaining_ms)}")
        try:
            yield
        finally:
            with suppress(Exception):
                self._conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")

    def _harden_permissions(self) -> None:
        directory = self._directory
        if directory is None:
            raise RuntimeError("Telemetry storage directory is not open")
        name = os.path.basename(self._db_path)
        for suffix in ("", "-wal", "-shm"):
            try:
                directory.harden_file(name + suffix)
            except FileNotFoundError:
                continue

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._conn is not None

    @property
    def db_path(self) -> str:
        return self._db_path

    def store(self, payload: bytes) -> bool:
        """Append one serialized event; trims the oldest rows if over capacity."""
        return self._store(payload, 0.0) is not None

    def reserve(self, payload: bytes, available_after_seconds: float) -> int | None:
        """Persist an event but defer draining until it is released or the delay expires."""
        return self._store(payload, time.time() + max(0.0, available_after_seconds))

    def _store(self, payload: bytes, available_at: float) -> int | None:
        if not payload:
            return None
        with self._lock:
            if self._conn is None:
                return None
            try:
                cursor = self._conn.execute(
                    "INSERT INTO events (payload, available_at) VALUES (?, ?)",
                    (sqlite3.Binary(payload), available_at),
                )
                count = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                if count > self._max_records:
                    self._conn.execute(
                        "DELETE FROM events WHERE id IN "
                        "(SELECT id FROM events WHERE available_at <= ? ORDER BY id ASC LIMIT ?)",
                        (time.time(), count - self._trim_target),
                    )
                self._conn.commit()
                self._harden_permissions()
                return int(cursor.lastrowid)
            except Exception:
                with suppress(Exception):
                    self._conn.rollback()
                return None

    def get_batch(self, max_count: int) -> list[tuple[int, bytes]]:
        """Return up to ``max_count`` oldest events as (id, payload) pairs."""
        return self.get_batch_for_upload(max_count) or []

    def get_batch_for_upload(self, max_count: int, deadline: float | None = None) -> list[tuple[int, bytes]] | None:
        """Return an uploadable batch, or None when local storage failed."""
        with self._bounded_lock(deadline) as acquired:
            if not acquired or self._conn is None:
                return None
            try:
                with self._bounded_busy_timeout(deadline):
                    rows = self._conn.execute(
                        "SELECT id, payload FROM events "
                        "WHERE available_at <= ? AND acknowledged=0 ORDER BY id ASC LIMIT ?",
                        (time.time(), max_count if max_count > 0 else -1),
                    ).fetchall()
                return [(r[0], bytes(r[1])) for r in rows]
            except Exception:
                return None

    def get_acknowledged_ids(self, max_count: int, deadline: float | None = None) -> list[int] | None:
        """Return terminally handled rows that only need local deletion."""
        with self._bounded_lock(deadline) as acquired:
            if not acquired or self._conn is None:
                return None
            try:
                with self._bounded_busy_timeout(deadline):
                    rows = self._conn.execute(
                        "SELECT id FROM events WHERE acknowledged=1 ORDER BY id ASC LIMIT ?",
                        (max_count if max_count > 0 else -1,),
                    ).fetchall()
                return [int(row[0]) for row in rows]
            except Exception:
                return None

    def release(self, row_id: int, payload: bytes | None = None) -> bool:
        """Make a reserved event drainable, optionally replacing its payload."""
        with self._lock:
            if self._conn is None:
                return False
            try:
                if payload is None:
                    cursor = self._conn.execute("UPDATE events SET available_at=0 WHERE id=?", (row_id,))
                else:
                    cursor = self._conn.execute(
                        "UPDATE events SET payload=?, available_at=0 WHERE id=?",
                        (sqlite3.Binary(payload), row_id),
                    )
                self._conn.commit()
                return cursor.rowcount == 1
            except Exception:
                with suppress(Exception):
                    self._conn.rollback()
                return False

    def acknowledge(self, ids: list[int], deadline: float | None = None) -> bool:
        """Persist that rows were delivered or permanently rejected."""
        if not ids:
            return True
        with self._bounded_lock(deadline) as acquired:
            if not acquired or self._conn is None:
                return False
            try:
                with self._bounded_busy_timeout(deadline):
                    self._conn.executemany("UPDATE events SET acknowledged=1 WHERE id=?", [(i,) for i in ids])
                    self._conn.commit()
                return True
            except Exception:
                with suppress(Exception):
                    self._conn.rollback()
                return False

    def delete(self, ids: list[int], deadline: float | None = None) -> bool:
        """Remove rows by id (after a successful upload or a permanent drop)."""
        if not ids:
            return True
        with self._bounded_lock(deadline) as acquired:
            if not acquired or self._conn is None:
                return False
            try:
                with self._bounded_busy_timeout(deadline):
                    self._conn.executemany("DELETE FROM events WHERE id=?", [(i,) for i in ids])
                    self._conn.commit()
                return True
            except Exception:
                # Failed deletes leave acknowledged rows durable for deletion-only retry.
                with suppress(Exception):
                    self._conn.rollback()
                return False

    def count(self) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            try:
                return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            except Exception:
                return 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                with suppress(Exception):
                    self._conn.close()
                self._conn = None
            self._close_storage_handles()
