# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Cross-platform persistent device ID with SHA-256 encryption."""

import functools
import hashlib
import os
import platform
import tempfile
import uuid
from enum import Enum
from pathlib import Path

ORT_SUPPORT_DIR = r"Microsoft/DeveloperTools/.onnxruntime"


class DeviceIdStatus(Enum):
    NEW = "new"
    EXISTING = "existing"
    CORRUPTED = "corrupted"
    FAILED = "failed"


_device_id_state = {"device_id": None, "status": DeviceIdStatus.NEW}


def _resolve_home_dir() -> Path:
    """Resolve the user home directory with fallbacks for container environments."""
    home = os.getenv("HOME")
    if home:
        return Path(home)
    try:
        return Path.home()
    except (RuntimeError, KeyError):
        if platform.system() != "Windows":
            return Path("/var/tmp")
        return Path(tempfile.gettempdir())


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
    if not cache_dir:
        cache_dir = str(_resolve_home_dir() / ".cache")

    return Path(cache_dir) / ORT_SUPPORT_DIR


class _FileStore:
    """File-based device ID persistence (Linux/macOS)."""

    def __init__(self) -> None:
        self._file_path: Path = get_telemetry_base_dir() / "deviceid"

    @property
    def retrieve_id(self) -> str:
        if not self._file_path.is_file():
            raise FileNotFoundError(f"File {self._file_path.stem} does not exist")
        return self._file_path.read_text(encoding="utf-8").strip()

    def store_id(self, device_id: str) -> None:
        # create the folder location if it does not exist, owner-only (0700) so other users on the
        # machine cannot traverse into it to reach the device id.
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.parent.chmod(0o700)

        # Owner-only (0600): the device id must not be world-readable by other users on the machine.
        # touch(mode=...) creates it already restricted; chmod also tightens a pre-existing file before
        # writing, so the id is never left at the umask default (commonly world-readable 0644).
        self._file_path.touch(mode=0o600)
        self._file_path.chmod(0o600)
        self._file_path.write_text(device_id, encoding="utf-8")


class _WindowsStore:
    """Windows registry-based device ID persistence."""

    REGISTRY_PATH = r"SOFTWARE\Microsoft\DeveloperTools\.onnxruntime"
    REGISTRY_KEY = "deviceid"

    @property
    def retrieve_id(self) -> str:
        import winreg

        with winreg.OpenKeyEx(
            winreg.HKEY_CURRENT_USER, self.REGISTRY_PATH, reserved=0, access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        ) as key_handle:
            device_id = winreg.QueryValueEx(key_handle, self.REGISTRY_KEY)
        return device_id[0].strip()

    def store_id(self, device_id: str) -> None:
        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            self.REGISTRY_PATH,
            reserved=0,
            access=winreg.KEY_ALL_ACCESS | winreg.KEY_WOW64_64KEY,
        ) as key_handle:
            winreg.SetValueEx(key_handle, self.REGISTRY_KEY, 0, winreg.REG_SZ, device_id)


def get_device_id() -> str:
    r"""Get or create a persistent device ID.

    Storage locations:
        Linux: $XDG_CACHE_HOME/Microsoft/DeveloperTools/.onnxruntime/deviceid
        macOS: ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/deviceid
        Windows: HKEY_CURRENT_USER\SOFTWARE\Microsoft\DeveloperTools\.onnxruntime\deviceid
    """
    device_id: str = ""
    create_new_id = False

    try:
        if platform.system() == "Windows":
            store = _WindowsStore()
        elif platform.system() in ("Linux", "Darwin"):
            store = _FileStore()
        else:
            _device_id_state["status"] = DeviceIdStatus.FAILED
            _device_id_state["device_id"] = device_id
            return device_id

        device_id = store.retrieve_id
        if len(device_id) > 256:
            _device_id_state["status"] = DeviceIdStatus.CORRUPTED
            _device_id_state["device_id"] = ""
            create_new_id = True
        else:
            try:
                uuid.UUID(device_id)
            except ValueError:
                _device_id_state["status"] = DeviceIdStatus.CORRUPTED
                _device_id_state["device_id"] = ""
                create_new_id = True
            else:
                _device_id_state["status"] = DeviceIdStatus.EXISTING
                _device_id_state["device_id"] = device_id
                return device_id
    except (FileExistsError, FileNotFoundError):
        _device_id_state["status"] = DeviceIdStatus.NEW
        _device_id_state["device_id"] = ""
        create_new_id = True
    except (PermissionError, ValueError, NotImplementedError):
        _device_id_state["status"] = DeviceIdStatus.FAILED
        _device_id_state["device_id"] = device_id
        return device_id
    except Exception:
        _device_id_state["status"] = DeviceIdStatus.FAILED
        _device_id_state["device_id"] = device_id
        return device_id

    if create_new_id:
        device_id = str(uuid.uuid4()).lower()
        try:
            store.store_id(device_id)
        except Exception:
            _device_id_state["status"] = DeviceIdStatus.FAILED
            device_id = ""
        _device_id_state["device_id"] = device_id

    return device_id


def get_encrypted_device_id_and_status() -> tuple[str, DeviceIdStatus]:
    """Get SHA-256 hashed device ID and its status."""
    device_id = _device_id_state["device_id"] if _device_id_state["device_id"] is not None else get_device_id()
    encrypted = hashlib.sha256(device_id.encode("utf-8")).digest().hex().upper() if device_id else ""
    return encrypted, _device_id_state["status"]
