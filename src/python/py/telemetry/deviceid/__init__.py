# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Cross-platform persistent device ID shared with native GenAI telemetry."""

import functools
import hashlib
import logging
import os
import platform
import re
import threading
import time
import uuid
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import ClassVar

from ..process_lock import ProcessDrainLock
from ..storage_paths import StorageDirectory

ORT_SUPPORT_DIR = r"Microsoft/DeveloperTools/.onnxruntime"
_MAX_DEVICE_ID_FILE_SIZE = 256
_DEVICE_ID_PATTERN = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_logger = logging.getLogger(__name__)


class DeviceIdStatus(Enum):
    NEW = "New"
    EXISTING = "Existing"
    CORRUPTED = "Corrupted"
    FAILED = "Failed"


_device_id_state = {"device_id": None, "status": DeviceIdStatus.NEW}
_device_id_lock = threading.Lock()


def _fnv1a_hex_bytes(value: bytes) -> str:
    hash_value = 14695981039346656037
    for byte in value:
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


def _is_valid_device_id(value: str) -> bool:
    return isinstance(value, str) and _DEVICE_ID_PATTERN.fullmatch(value) is not None


def _resolve_home_dir() -> Path:
    """Resolve the user home directory with fallbacks for container environments."""
    home = os.getenv("HOME")
    if home and Path(home).is_absolute():
        return Path(home)
    if platform.system() != "Windows":
        try:
            import pwd  # noqa: PLC0415

            passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
            if passwd_home.is_absolute():
                return passwd_home
        except (AttributeError, ImportError, KeyError, OSError):
            # Fall through to pathlib's cross-platform home-directory resolution.
            pass
    try:
        fallback_home = Path.home()
        if fallback_home.is_absolute():
            return fallback_home
    except (RuntimeError, KeyError):
        # The caller raises a single explicit error after all supported fallbacks fail.
        pass
    raise RuntimeError("No absolute per-user telemetry storage directory is available")


@functools.lru_cache(maxsize=1)
def get_telemetry_base_dir() -> Path:
    os_name = platform.system()
    if os_name == "Windows":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if not base_dir:
            base_dir = str(Path.home() / "AppData" / "Local")
        return Path(base_dir) / ORT_SUPPORT_DIR

    if os_name == "Darwin":
        home = _resolve_home_dir()
        return home / "Library" / "Application Support" / ORT_SUPPORT_DIR

    cache_dir = os.getenv("XDG_CACHE_HOME")
    if cache_dir and Path(cache_dir).is_absolute():
        return Path(cache_dir) / ORT_SUPPORT_DIR

    return _resolve_home_dir() / ".cache" / ORT_SUPPORT_DIR


class _FileStore:
    """File-based device ID persistence (Linux/macOS)."""

    def __init__(self) -> None:
        self._file_path: Path = get_telemetry_base_dir() / "deviceid"

    @property
    def retrieve_id(self) -> str:
        try:
            with StorageDirectory(self._file_path.parent) as directory:
                # Atomic publication briefly links two names; reads must not chmod that inode.
                fd = directory.open_file(self._file_path.name, os.O_RDONLY, allow_shared=True)
                with os.fdopen(fd, "rb") as device_id_file:
                    raw_value = device_id_file.read(_MAX_DEVICE_ID_FILE_SIZE + 1)
            if len(raw_value) > _MAX_DEVICE_ID_FILE_SIZE:
                raise ValueError(f"File {self._file_path.stem} exceeds {_MAX_DEVICE_ID_FILE_SIZE} bytes")
            return raw_value.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise ValueError(f"File {self._file_path.stem} is not valid UTF-8") from None

    def store_id(self, device_id: str, replace_existing: bool = False) -> bool:
        with StorageDirectory(self._file_path.parent, create=True) as directory:
            directory.validate_file(self._file_path.name)
            temporary_name = f"deviceid.tmp.{uuid.uuid4().hex}"
            fd = directory.open_file(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                    temp_file.write(device_id)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                try:
                    directory.publish(temporary_name, self._file_path.name, replace_existing)
                except FileExistsError:
                    return False
                return True
            finally:
                with suppress(FileNotFoundError):
                    directory.unlink(temporary_name)


class _WindowsStore:
    """Windows registry-based device ID persistence."""

    REGISTRY_PATH = r"SOFTWARE\Microsoft\DeveloperTools\.onnxruntime"
    REGISTRY_KEY = "deviceid"

    @property
    def retrieve_id(self) -> str:
        import winreg  # noqa: PLC0415

        with winreg.OpenKeyEx(
            winreg.HKEY_CURRENT_USER, self.REGISTRY_PATH, reserved=0, access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        ) as key_handle:
            device_id, value_type = winreg.QueryValueEx(key_handle, self.REGISTRY_KEY)
        if value_type != winreg.REG_SZ or not isinstance(device_id, str):
            raise ValueError(f"Registry value {self.REGISTRY_KEY} is not a string")
        return device_id.strip()

    def store_id(self, device_id: str, replace_existing: bool = False) -> bool:
        import winreg  # noqa: PLC0415

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            self.REGISTRY_PATH,
            reserved=0,
            access=winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY | winreg.KEY_WOW64_64KEY,
        ) as key_handle:
            winreg.SetValueEx(key_handle, self.REGISTRY_KEY, 0, winreg.REG_SZ, device_id)
        return True


class _WindowsDeviceIdMutex:
    """Named mutex compatible with the native GenAI device-id protocol."""

    def __init__(self) -> None:
        self._handle = None
        self._acquired = False
        self._kernel32 = None

    def acquire(self) -> bool:
        try:
            import ctypes  # noqa: PLC0415
            import ctypes.wintypes  # noqa: PLC0415

            wintypes = ctypes.wintypes

            class SidAndAttributes(ctypes.Structure):
                _fields_: ClassVar = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

            class TokenUser(ctypes.Structure):
                _fields_: ClassVar = [("user", SidAndAttributes)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            advapi32.OpenProcessToken.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.HANDLE),
            ]
            advapi32.GetTokenInformation.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            advapi32.IsValidSid.argtypes = [ctypes.c_void_p]
            advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
            advapi32.GetLengthSid.restype = wintypes.DWORD
            token = wintypes.HANDLE()
            if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
                return False
            try:
                size = wintypes.DWORD()
                advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
                if not size.value:
                    return False
                token_info = ctypes.create_string_buffer(size.value)
                if not advapi32.GetTokenInformation(token, 1, token_info, size.value, ctypes.byref(size)):
                    return False
                sid = ctypes.cast(token_info, ctypes.POINTER(TokenUser)).contents.user.sid
                if not sid or not advapi32.IsValidSid(sid):
                    return False
                sid_size = advapi32.GetLengthSid(sid)
                sid_hash = _fnv1a_hex_bytes(ctypes.string_at(sid, sid_size))
            finally:
                kernel32.CloseHandle(token)

            name = f"Global\\Microsoft.DeveloperTools.OnnxRuntime.DeviceId.{sid_hash}"
            handle = kernel32.CreateMutexW(None, False, name)
            if not handle:
                return False
            self._handle = handle
            self._kernel32 = kernel32
            wait_result = kernel32.WaitForSingleObject(handle, 1000)
            self._acquired = wait_result in (0x00000000, 0x00000080)
            return self._acquired
        except Exception:
            self.release()
            return False

    def release(self) -> None:
        if self._handle is None or self._kernel32 is None:
            return
        if self._acquired:
            with suppress(Exception):
                self._kernel32.ReleaseMutex(self._handle)
        with suppress(Exception):
            self._kernel32.CloseHandle(self._handle)
        self._handle = None
        self._acquired = False
        self._kernel32 = None


def _initialize_device_id() -> tuple[str, DeviceIdStatus]:
    r"""Read or publish an ID; the caller owns the process cache.

    Storage locations:
        POSIX: $XDG_CACHE_HOME/Microsoft/DeveloperTools/.onnxruntime/deviceid
        macOS: ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/deviceid
        Windows: HKEY_CURRENT_USER\SOFTWARE\Microsoft\DeveloperTools\.onnxruntime\deviceid
    """

    try:
        system = platform.system()
        if system == "Windows":
            store = _WindowsStore()
        elif system in ("Linux", "Darwin") or os.name == "posix":
            store = _FileStore()
        else:
            return str(uuid.uuid4()), DeviceIdStatus.FAILED
    except Exception:
        _logger.debug("Telemetry device-ID storage initialization failed", exc_info=True)
        return str(uuid.uuid4()), DeviceIdStatus.FAILED

    def read_existing() -> tuple[str, str]:
        deadline = time.monotonic() + 1.0
        while True:
            try:
                existing = store.retrieve_id
            except (FileExistsError, FileNotFoundError):
                return ("missing", "")
            except ValueError:
                return ("invalid", "")
            except Exception as error:
                if getattr(error, "winerror", None) in {32, 33} and time.monotonic() < deadline:
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                    continue
                _logger.debug("Telemetry device-ID retrieval failed", exc_info=True)
                return ("failed", "")
            return ("valid", existing) if _is_valid_device_id(existing) else ("invalid", "")

    initial_state, existing = read_existing()
    if initial_state == "valid":
        return existing, DeviceIdStatus.EXISTING
    if initial_state == "failed":
        return str(uuid.uuid4()), DeviceIdStatus.FAILED

    lock = None
    acquired = True
    if system == "Windows":
        lock = _WindowsDeviceIdMutex()
        acquired = lock.acquire()
    elif initial_state == "invalid":
        lock = ProcessDrainLock(str(get_telemetry_base_dir() / "deviceid.lock"))
        acquired = lock.acquire(1.0)

    try:
        if not acquired:
            _logger.debug("Telemetry device-ID publication lock unavailable")
            winner_state, winner = read_existing()
            if winner_state == "valid":
                return winner, DeviceIdStatus.EXISTING
            return str(uuid.uuid4()), DeviceIdStatus.FAILED

        current_state, current = read_existing()
        if current_state == "valid":
            return current, DeviceIdStatus.EXISTING
        if current_state == "failed":
            return str(uuid.uuid4()), DeviceIdStatus.FAILED

        corrupted = initial_state == "invalid" or current_state == "invalid"
        generated = str(uuid.uuid4())
        try:
            stored = store.store_id(generated, replace_existing=corrupted)
        except Exception:
            _logger.debug("Telemetry device-ID publication failed", exc_info=True)
            stored = False
        if stored:
            return generated, DeviceIdStatus.CORRUPTED if corrupted else DeviceIdStatus.NEW

        winner_state, winner = read_existing()
        if winner_state == "valid":
            return winner, DeviceIdStatus.EXISTING
        return generated, DeviceIdStatus.FAILED
    finally:
        if lock is not None:
            lock.release()


def _get_device_id_and_status() -> tuple[str, DeviceIdStatus]:
    with _device_id_lock:
        if _device_id_state["device_id"] is None:
            device_id, status = _initialize_device_id()
            _device_id_state.update({"device_id": device_id, "status": status})
        return _device_id_state["device_id"], _device_id_state["status"]


def get_device_id() -> str:
    """Get the device ID cached for this process, including persistence failures."""
    return _get_device_id_and_status()[0]


def get_hashed_device_id_and_status() -> tuple[str, DeviceIdStatus]:
    """Get the shared hashed device ID and its status."""
    device_id, status = _get_device_id_and_status()
    return f"c:{hashlib.sha256(device_id.encode('utf-8')).hexdigest()}", status
