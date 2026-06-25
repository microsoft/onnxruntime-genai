# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SQLite-backed offline storage for telemetry events.

Emulates the Microsoft 1DS C++ SDK offline store
(cpp_client_telemetry/lib/offline/OfflineStorage_SQLite.cpp): a durable
``events`` table multiplexed by ``tenant_token``, with reserve-on-read
leasing (``reserved_until``) so multiple uploaders/processes never double-send,
retry accounting, size trimming, and a ``settings`` key/value table.

Uses only the Python standard library (``sqlite3``) so it adds no dependency.
The database is shared-capable: multiple applications (e.g. ONNX Runtime GenAI
and Olive) may point at the same file and are kept apart by ``tenant_token``.

Schema (identical to the 1DS store, plus a stamped ``schema_version``)::

    CREATE TABLE events (
        record_id      TEXT,
        tenant_token   TEXT NOT NULL,
        latency        INTEGER,
        persistence    INTEGER,
        timestamp      INTEGER,
        retry_count    INTEGER DEFAULT 0,
        reserved_until INTEGER DEFAULT 0,
        payload        BLOB);
    CREATE INDEX k_latency_timestamp ON events (latency, timestamp);
    CREATE TABLE settings (name TEXT PRIMARY KEY, value TEXT);
"""

import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

SCHEMA_VERSION = "1"

# EventLatency, mirroring 1DS Enums.hpp (higher = uploaded sooner).
LATENCY_OFF = 0
LATENCY_NORMAL = 1
LATENCY_COST_DEFERRED = 2
LATENCY_REAL_TIME = 3

# EventPersistence, mirroring 1DS Enums.hpp (higher = trimmed last).
PERSISTENCE_NORMAL = 1
PERSISTENCE_CRITICAL = 2


@dataclass
class StorageRecord:
    """A single stored telemetry event (mirrors 1DS StorageRecord)."""

    tenant_token: str
    payload: bytes
    latency: int = LATENCY_NORMAL
    persistence: int = PERSISTENCE_NORMAL
    timestamp: int = 0
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    retry_count: int = 0
    reserved_until: int = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


class OfflineStorageSqlite:
    """Durable SQLite event store with 1DS-style reserve/release/trim semantics.

    All public methods are best-effort and swallow storage errors: telemetry
    must never crash the host application. Thread-safe via a per-instance lock;
    cross-process safe via WAL mode + ``busy_timeout`` + ``reserved_until``
    leasing.
    """

    def __init__(
        self,
        db_path: str,
        max_records: int = 2048,
        max_retry_count: int = 6,
        trim_percent: int = 25,
        busy_timeout_ms: int = 3000,
    ):
        self._db_path = db_path
        self._max_records = max_records
        self._max_retry_count = max_retry_count
        self._trim_percent = max(1, min(100, trim_percent))
        self._busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    # ----- lifecycle -----------------------------------------------------

    def _initialize(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        except Exception:
            pass
        try:
            conn = sqlite3.connect(
                self._db_path, timeout=self._busy_timeout_ms / 1000.0, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "record_id TEXT,"
                "tenant_token TEXT NOT NULL,"
                "latency INTEGER,"
                "persistence INTEGER,"
                "timestamp INTEGER,"
                "retry_count INTEGER DEFAULT 0,"
                "reserved_until INTEGER DEFAULT 0,"
                "payload BLOB)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS k_latency_timestamp ON events (latency, timestamp)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "INSERT OR IGNORE INTO settings (name, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
            self._conn = conn
        except Exception:
            self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # ----- writes --------------------------------------------------------

    def store_record(self, record: StorageRecord) -> bool:
        """Persist one event. Trims oldest records if over the size limit."""
        if not record.tenant_token or not record.payload:
            return False
        if record.timestamp <= 0:
            record.timestamp = _now_ms()
        with self._lock:
            if self._conn is None:
                return False
            try:
                self._conn.execute(
                    "INSERT INTO events "
                    "(record_id, tenant_token, latency, persistence, timestamp, retry_count, reserved_until, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.record_id,
                        record.tenant_token,
                        int(record.latency),
                        int(record.persistence),
                        int(record.timestamp),
                        int(record.retry_count),
                        int(record.reserved_until),
                        sqlite3.Binary(record.payload),
                    ),
                )
                self._conn.commit()
                self._trim_if_needed_locked()
                return True
            except Exception:
                return False

    # ----- reserve / release / delete ------------------------------------

    def get_and_reserve_records(
        self, lease_time_ms: int, tenant_token: Optional[str] = None, min_latency: int = LATENCY_OFF, max_count: int = 0
    ) -> list[StorageRecord]:
        """Atomically lease the best records for upload.

        Expired reservations are first reclaimed (and their retry_count bumped),
        then unreserved rows with ``latency >= min_latency`` are selected in 1DS
        order (latency desc, persistence desc, timestamp asc) and reserved for
        ``lease_time_ms``. Reserved rows are not returned again until released,
        deleted, or their lease expires.
        """
        now = _now_ms()
        with self._lock:
            if self._conn is None:
                return []
            try:
                # Reclaim expired reservations (count the timed-out attempt as a retry).
                self._conn.execute(
                    "UPDATE events SET reserved_until=0, retry_count=retry_count+1 "
                    "WHERE reserved_until<>0 AND reserved_until<=?",
                    (now,),
                )
                self._drop_over_retry_locked()

                params: list = [int(min_latency)]
                sql = (
                    "SELECT record_id, tenant_token, latency, persistence, timestamp, retry_count, reserved_until, payload "
                    "FROM events WHERE latency>=? AND reserved_until=0"
                )
                if tenant_token is not None:
                    sql += " AND tenant_token=?"
                    params.append(tenant_token)
                sql += " ORDER BY latency DESC, persistence DESC, timestamp ASC LIMIT ?"
                params.append(max_count if max_count > 0 else -1)

                rows = self._conn.execute(sql, params).fetchall()
                if not rows:
                    self._conn.commit()
                    return []

                reserved_until = now + int(lease_time_ms)
                ids = [r[0] for r in rows]
                self._conn.executemany(
                    "UPDATE events SET reserved_until=? WHERE record_id=?",
                    [(reserved_until, rid) for rid in ids],
                )
                self._conn.commit()

                return [
                    StorageRecord(
                        record_id=r[0],
                        tenant_token=r[1],
                        latency=r[2],
                        persistence=r[3],
                        timestamp=r[4],
                        retry_count=r[5],
                        reserved_until=reserved_until,
                        payload=bytes(r[7]),
                    )
                    for r in rows
                ]
            except Exception:
                return []

    def delete_records(self, ids: list[str]) -> None:
        """Permanently remove records (after a successful upload)."""
        if not ids:
            return
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.executemany("DELETE FROM events WHERE record_id=?", [(i,) for i in ids])
                self._conn.commit()
            except Exception:
                pass

    def release_records(self, ids: list[str], increment_retry: bool = True) -> None:
        """Un-reserve records (after a failed upload), optionally bumping retry.

        Records whose retry_count reaches the configured maximum are dropped.
        """
        if not ids:
            return
        delta = 1 if increment_retry else 0
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.executemany(
                    "UPDATE events SET reserved_until=0, retry_count=retry_count+? "
                    "WHERE record_id=? AND reserved_until>0",
                    [(delta, i) for i in ids],
                )
                if increment_retry:
                    self._drop_over_retry_locked()
                self._conn.commit()
            except Exception:
                pass

    # ----- maintenance ---------------------------------------------------

    def _drop_over_retry_locked(self) -> None:
        try:
            self._conn.execute("DELETE FROM events WHERE retry_count>=?", (self._max_retry_count,))
        except Exception:
            pass

    def _trim_if_needed_locked(self) -> None:
        try:
            count = self._conn.execute("SELECT COUNT(record_id) FROM events").fetchone()[0]
            if count <= self._max_records:
                return
            # Drop the oldest, least-persistent slice (mirrors 1DS percent trim).
            to_drop = max(1, (count * self._trim_percent) // 100)
            self._conn.execute(
                "DELETE FROM events WHERE record_id IN ("
                "SELECT record_id FROM events ORDER BY persistence ASC, timestamp ASC LIMIT ?)",
                (to_drop,),
            )
            self._conn.commit()
        except Exception:
            pass

    # ----- introspection / settings --------------------------------------

    def record_count(self, tenant_token: Optional[str] = None) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            try:
                if tenant_token is None:
                    row = self._conn.execute("SELECT COUNT(record_id) FROM events").fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT COUNT(record_id) FROM events WHERE tenant_token=?", (tenant_token,)
                    ).fetchone()
                return int(row[0]) if row else 0
            except Exception:
                return 0

    def get_setting(self, name: str) -> str:
        with self._lock:
            if self._conn is None:
                return ""
            try:
                row = self._conn.execute("SELECT value FROM settings WHERE name=?", (name,)).fetchone()
                return row[0] if row else ""
            except Exception:
                return ""

    def store_setting(self, name: str, value: str) -> bool:
        with self._lock:
            if self._conn is None:
                return False
            try:
                if value == "":
                    self._conn.execute("DELETE FROM settings WHERE name=?", (name,))
                else:
                    self._conn.execute(
                        "INSERT INTO settings (name, value) VALUES (?, ?) "
                        "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                        (name, value),
                    )
                self._conn.commit()
                return True
            except Exception:
                return False

    def release_reserved(self, tenant_token: Optional[str] = None) -> None:
        """Clear leases (reserved_until) without bumping retry.

        Used at shutdown after the background drainer has been stopped, so a
        final synchronous flush can pick up rows the loop had reserved.
        """
        with self._lock:
            if self._conn is None:
                return
            try:
                if tenant_token is None:
                    self._conn.execute("UPDATE events SET reserved_until=0 WHERE reserved_until>0")
                else:
                    self._conn.execute(
                        "UPDATE events SET reserved_until=0 WHERE tenant_token=? AND reserved_until>0",
                        (tenant_token,),
                    )
                self._conn.commit()
            except Exception:
                pass

    def delete_all_records(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute("DELETE FROM events")
                self._conn.commit()
            except Exception:
                pass
