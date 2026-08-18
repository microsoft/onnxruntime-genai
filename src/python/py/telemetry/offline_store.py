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
``settings`` table. The schema version is tracked with SQLite's built-in
``PRAGMA user_version``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import suppress

SCHEMA_VERSION = 2
_RECONNECT_BUSY_TIMEOUT_MS = 50
_RECONNECT_INTERVAL_SECONDS = 5.0


def _chmod_best_effort(path: str, mode: int) -> None:
    if os.name == "nt" or not path:
        return
    # Permission tightening is best-effort on filesystems that do not support chmod.
    with suppress(OSError):
        os.chmod(path, mode)


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
        self._reconnect_enabled = False
        self._next_reconnect_attempt = 0.0
        self._initialize()

    def _initialize(self, busy_timeout_ms: int | None = None) -> None:
        parent = os.path.dirname(self._db_path)
        effective_busy_timeout_ms = self._busy_timeout_ms if busy_timeout_ms is None else busy_timeout_ms
        # sqlite3.connect below reports whether storage can actually be opened.
        with suppress(Exception):
            os.makedirs(parent, mode=0o700, exist_ok=True)
            _chmod_best_effort(parent, 0o700)
        conn = None
        try:
            conn = sqlite3.connect(
                self._db_path,
                timeout=effective_busy_timeout_ms / 1000.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={effective_busy_timeout_ms}")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL, available_at REAL NOT NULL DEFAULT 0)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
            if "available_at" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN available_at REAL NOT NULL DEFAULT 0")
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.commit()
            # A reconnect may use a short probe timeout, but normal operations
            # retain the configured contention tolerance.
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            self._conn = conn
            self._next_reconnect_attempt = 0.0
            self._harden_permissions()
        except Exception:
            if conn is not None:
                with suppress(Exception):
                    conn.close()
            self._conn = None

    def _ensure_open(self) -> bool:
        if self._conn is not None:
            return True
        if not self._reconnect_enabled:
            return False
        now = time.monotonic()
        if now < self._next_reconnect_attempt:
            return False
        self._next_reconnect_attempt = now + _RECONNECT_INTERVAL_SECONDS
        self._initialize(min(self._busy_timeout_ms, _RECONNECT_BUSY_TIMEOUT_MS))
        return self._conn is not None

    def _harden_permissions(self) -> None:
        _chmod_best_effort(os.path.dirname(self._db_path), 0o700)
        for path in (self._db_path, self._db_path + "-wal", self._db_path + "-shm"):
            if os.path.exists(path):
                _chmod_best_effort(path, 0o600)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._ensure_open()

    @property
    def db_path(self) -> str:
        return self._db_path

    def store(self, payload: bytes) -> bool:
        """Append one serialized event; trims the oldest rows if over capacity."""
        return self.store_with_id(payload) is not None

    def store_with_id(self, payload: bytes, available_after_seconds: float = 0.0) -> int | None:
        """Append one serialized event and return its row id."""
        if not payload:
            return None
        with self._lock:
            if not self._ensure_open():
                return None
            try:
                available_at = time.time() + max(0.0, available_after_seconds)
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
        with self._lock:
            if not self._ensure_open():
                return []
            try:
                rows = self._conn.execute(
                    "SELECT id, payload FROM events WHERE available_at <= ? ORDER BY id ASC LIMIT ?",
                    (time.time(), max_count if max_count > 0 else -1),
                ).fetchall()
                return [(r[0], bytes(r[1])) for r in rows]
            except Exception:
                return []

    def replace(self, row_id: int, payload: bytes) -> bool:
        """Replace a queued payload if it has not already been drained."""
        if not payload:
            return False
        with self._lock:
            if not self._ensure_open():
                return False
            try:
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

    def make_available(self, row_id: int) -> bool:
        """Release a deferred row without changing its payload."""
        with self._lock:
            if not self._ensure_open():
                return False
            try:
                cursor = self._conn.execute(
                    "UPDATE events SET available_at=0 WHERE id=?",
                    (row_id,),
                )
                self._conn.commit()
                return cursor.rowcount == 1
            except Exception:
                with suppress(Exception):
                    self._conn.rollback()
                return False

    def delete(self, ids: list[int]) -> bool:
        """Remove rows by id (after a successful upload or a permanent drop)."""
        if not ids:
            return True
        with self._lock:
            if not self._ensure_open():
                return False
            # Failed deletes leave rows durable for a later drain attempt.
            try:
                self._conn.executemany("DELETE FROM events WHERE id=?", [(i,) for i in ids])
                self._conn.commit()
                return True
            except Exception:
                with suppress(Exception):
                    self._conn.rollback()
                return False

    def count(self) -> int:
        with self._lock:
            if not self._ensure_open():
                return 0
            try:
                return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            except Exception:
                return 0

    def close(self, reconnect: bool = False) -> None:
        with self._lock:
            self._reconnect_enabled = reconnect
            if self._conn is not None:
                with suppress(Exception):
                    self._conn.close()
                self._conn = None

    def prepare_for_fork(self) -> None:
        """Close the live connection in the parent before it can be inherited."""
        self.close(reconnect=True)

    def discard_after_fork(self) -> None:
        """Reset child-only state after the parent closed the connection pre-fork."""
        self._lock = threading.Lock()
        self._reconnect_enabled = False
        self._next_reconnect_attempt = 0.0
        self._conn = None
