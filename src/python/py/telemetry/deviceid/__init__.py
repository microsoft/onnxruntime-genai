# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Cross-platform persistent device ID shared with native GenAI telemetry."""

import functools
import hashlib
import os
import platform
import stat
import tempfile
import uuid
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import ClassVar

from ..process_lock import ProcessDrainLock

ORT_SUPPORT_DIR = r"Microsoft/DeveloperTools/.onnxruntime"
_MAX_DEVICE_ID_FILE_SIZE = 256


class DeviceIdStatus(Enum):
    NEW = "New"
    EXISTING = "Existing"
    CORRUPTED = "Corrupted"
    FAILED = "Failed"


_device_id_state = {"device_id": None, "status": DeviceIdStatus.NEW}


def _fnv1a_hex_bytes(value: bytes) -> str:
    hash_value = 14695981039346656037
    for byte in value:
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


def _is_valid_device_id(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 36:
        return False
    hyphens = {8, 13, 18, 23}
    return all(
        char == "-" if index in hyphens else char.lower() in "0123456789abcdef"
        for index, char in enumerate(value)
    )


def _chmod_best_effort(path: Path, mode: int) -> None:
    # Permission tightening is best-effort on filesystems that do not support chmod.
    with suppress(OSError):
        path.chmod(mode)


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
            pass
    try:
        fallback_home = Path.home()
        if fallback_home.is_absolute():
            return fallback_home
    except (RuntimeError, KeyError):
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

    def _validate_parent(self) -> None:
        parent_info = self._file_path.parent.lstat()
        if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
            raise PermissionError("Device ID storage directory is not a regular directory")
        if hasattr(os, "geteuid") and parent_info.st_uid != os.geteuid():
            raise PermissionError("Device ID storage directory is not owned by the current user")

    @property
    def retrieve_id(self) -> str:
        self._validate_parent()
        try:
            file_info = self._file_path.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self._file_path.stem} does not exist") from None
        if stat.S_ISLNK(file_info.st_mode) or not stat.S_ISREG(file_info.st_mode):
            raise PermissionError(f"File {self._file_path.stem} is not a regular file")
        try:
            with self._file_path.open("rb") as device_id_file:
                raw_value = device_id_file.read(_MAX_DEVICE_ID_FILE_SIZE + 1)
            if len(raw_value) > _MAX_DEVICE_ID_FILE_SIZE:
                raise ValueError(f"File {self._file_path.stem} exceeds {_MAX_DEVICE_ID_FILE_SIZE} bytes")
            return raw_value.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise ValueError(f"File {self._file_path.stem} is not valid UTF-8") from None

    def store_id(self, device_id: str, replace_existing: bool = False) -> bool:
        # create the folder location if it does not exist, owner-only (0700) so other users on the
        # machine cannot traverse into it to reach the device id.
        self._file_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._validate_parent()
        _chmod_best_effort(self._file_path.parent, 0o700)

        fd, temp_path = tempfile.mkstemp(prefix="deviceid.tmp.", dir=self._file_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(device_id)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            _chmod_best_effort(Path(temp_path), 0o600)
            if replace_existing:
                os.replace(temp_path, self._file_path)
                temp_path = ""
                return True
            try:
                os.link(temp_path, self._file_path)
            except FileExistsError:
                return False
            return True
        finally:
            if temp_path:
                with suppress(OSError):
                    os.unlink(temp_path)
            _chmod_best_effort(self._file_path, 0o600)


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
            from ctypes import wintypes  # noqa: PLC0415

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


def get_device_id() -> str:
    r"""Get or create a persistent device ID.

    Storage locations:
        Linux: $XDG_CACHE_HOME/Microsoft/DeveloperTools/.onnxruntime/deviceid
        macOS: ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/deviceid
        Windows: HKEY_CURRENT_USER\SOFTWARE\Microsoft\DeveloperTools\.onnxruntime\deviceid
    """
    def failed_fallback() -> str:
        generated = str(uuid.uuid4()).lower()
        _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
        return generated

    try:
        system = platform.system()
        if system == "Windows":
            store = _WindowsStore()
        elif system in ("Linux", "Darwin"):
            store = _FileStore()
        else:
            return failed_fallback()
    except Exception:
        return failed_fallback()

    def read_existing() -> tuple[str, str]:
        try:
            existing = store.retrieve_id
        except (FileExistsError, FileNotFoundError):
            return ("missing", "")
        except ValueError:
            return ("invalid", "")
        except Exception:
            return ("failed", "")
        return ("valid", existing) if _is_valid_device_id(existing) else ("invalid", "")

    initial_state, existing = read_existing()
    if initial_state == "valid":
        _device_id_state.update({"status": DeviceIdStatus.EXISTING, "device_id": existing})
        return existing
    if initial_state == "failed":
        generated = str(uuid.uuid4()).lower()
        _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
        return generated

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
            winner_state, winner = read_existing()
            generated = winner if winner_state == "valid" else str(uuid.uuid4()).lower()
            status = DeviceIdStatus.EXISTING if winner_state == "valid" else DeviceIdStatus.FAILED
            _device_id_state.update({"status": status, "device_id": generated})
            return generated

        current_state, current = read_existing()
        if current_state == "valid":
            _device_id_state.update({"status": DeviceIdStatus.EXISTING, "device_id": current})
            return current
        if current_state == "failed":
            generated = str(uuid.uuid4()).lower()
            _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
            return generated

        corrupted = initial_state == "invalid" or current_state == "invalid"
        generated = str(uuid.uuid4()).lower()
        try:
            stored = store.store_id(generated, replace_existing=corrupted)
        except Exception:
            stored = False
        if stored:
            status = DeviceIdStatus.CORRUPTED if corrupted else DeviceIdStatus.NEW
            _device_id_state.update({"status": status, "device_id": generated})
            return generated

        winner_state, winner = read_existing()
        if winner_state == "valid":
            _device_id_state.update({"status": DeviceIdStatus.EXISTING, "device_id": winner})
            return winner
        _device_id_state.update({"status": DeviceIdStatus.FAILED, "device_id": generated})
        return generated
    finally:
        if lock is not None:
            lock.release()


def get_hashed_device_id_and_status() -> tuple[str, DeviceIdStatus]:
    """Get the shared hashed device ID and its status."""
    device_id = _device_id_state["device_id"] if _device_id_state["device_id"] is not None else get_device_id()
    hashed = hashlib.sha256(device_id.encode("utf-8")).hexdigest() if device_id else ""
    return f"c:{hashed}" if hashed else "", _device_id_state["status"]
