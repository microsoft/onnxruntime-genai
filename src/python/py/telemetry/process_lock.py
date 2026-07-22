# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Cross-platform single-holder advisory lock (standard library only).

Used so that, when several processes on one device share a telemetry database,
only one of them runs the uploader's drain loop at a time. Other processes keep
writing events durably to the store; the lock holder drains everyone's rows.
This avoids the same event being uploaded twice by concurrent drainers without
needing per-row reservation bookkeeping.

The lock is an OS advisory lock on a sidecar file (``msvcrt`` on Windows,
``fcntl`` on POSIX). It is released explicitly and also by the OS when the
process exits, so a crashed holder never blocks other processes permanently.
"""

import os
from contextlib import suppress


class ProcessDrainLock:
    """Non-blocking exclusive advisory lock backed by a sidecar file."""

    def __init__(self, lock_path: str):
        self._lock_path = lock_path
        self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> bool:
        """Try to acquire the lock without blocking. Returns True if held."""
        if self._fh is not None:
            return True
        fh = None
        try:
            with suppress(Exception):
                os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
            # The handle owns the advisory lock and must remain open until release().
            fh = open(self._lock_path, "a+b")  # noqa: SIM115
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fh = fh
            return True
        except Exception:
            if fh is not None:
                with suppress(Exception):
                    fh.close()
            return False

    def release(self) -> None:
        if self._fh is None:
            return
        fh = self._fh
        self._fh = None
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                with suppress(Exception):
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                with suppress(Exception):
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            with suppress(Exception):
                fh.close()
